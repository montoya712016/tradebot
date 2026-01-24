#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coleta trades da Binance e grava um único preço de fechamento por segundo
em MySQL (DB: crypto_second).

Salvo por símbolo, a cada janela de 1 s:
    • dates         (BIGINT ms, chave primária)
    • closing_prices (FLOAT)

Se em algum segundo não houver trade, não será gravado nada (buracos permitidos).
"""
import json, logging, sys, threading, time, websocket
from collections import defaultdict, deque
from itertools import islice
import mysql.connector

# ───────────────────────── CONFIGURAÇÃO ──────────────────────────────
DB_CFG = dict(
    host="localhost",
    user="root",
    password="2017",
    database="crypto_second",
)

CANDLE_SEC    = 1      # resolução de 1 segundo
WS_CHUNK      = 80     # número de pares por conexão websocket
PING_INTERVAL = 20
PING_TIMEOUT  = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("collector1s")

# ────────────────────────── Lista de símbolos ─────────────────────────
def obter_lista_criptomoedas() -> list[str]:
    return deque([
        'ACAUSDT','AGLDUSDT','ALGOUSDT','ALICEUSDT','ALPACAUSDT','ALPHAUSDT',
        'ANKRUSDT','APEUSDT','ARDRUSDT','ARUSDT','ASRUSDT','ASTRUSDT','ATAUSDT',
        'ATOMUSDT','AUCTIONUSDT','AUROUSDT','AVAXUSDT','AXSUSDT','BALUSDT',
        'BANDUSDT','BARUSDT','BATUSDT','BELUSDT','BETAUSDT','BNTUSDT','BTCUSDT',
        'BTSUSDT','BURGERUSDT','C98USDT','CELRUSDT','CHRUSDT','CHZUSDT','CITYUSDT',
        'CLVUSDT','COMPUSDT','COTIUSDT','CRVUSDT','CTKUSDT','CTSIUSDT','CVCUSDT',
        'CVXUSDT','DENTUSDT','DFUSDT','DGBUSDT','DIAUSDT','DODOUSDT','DOGEUSDT',
        'DOTUSDT','DYDXUSDT','DUSKUSDT','EGLDUSDT','ENSUSDT','EOSUSDT','ERNUSDT',
        'ETHUSDT','FETUSDT','FILUSDT','FIDAUSDT','FIROUSDT','FISUSDT','FLMUSDT',
        'FLOWUSDT','FORTHUSDT','FTMUSDT','GALAUSDT','GLMRUSDT','GNOUSDT','GRTUSDT',
        'GTCUSDT','HARDUSDT','HIGHUSDT','HNTUSDT','ICXUSDT','ILVUSDT','IMXUSDT',
        'INJUSDT','IOSTUSDT','IOTXUSDT','IRISUSDT','JSTUSDT','KAVAUSDT','KEEPUSDT',
        'KLAYUSDT','KMDUSDT','KNCUSDT','KSMUSDT','LAZIOUSDT','LINKUSDT','LITUSDT',
        'LRCUSDT','LTCUSDT','LUNAUSDT','MANAUSDT','MATICUSDT','MBLUSDT','MKRUSDT',
        'MLNUSDT','MOVRUSDT','MTLUSDT','NEARUSDT','NKNUSDT','OCEANUSDT','OGNUSDT',
        'OMGUSDT','OMUSDT','ONEUSDT','OOKIUSDT','PAXGUSDT','PEOPLEUSDT','PERPUSDT',
        'PHAUSDT','POLYUSDT','PORTOUSDT','PSGUSDT','PYRUSDT','QKCUSDT','QNTUSDT',
        'QUICKUSDT','RADUSDT','RAREUSDT','REEFUSDT','RENUSDT','RIFUSDT','RLCUSDT',
        'RNDRUSDT','ROSEUSDT','RSRUSDT','RUNEUSDT','SANDUSDT','SCRTUSDT','SFPUSDT',
        'SHIBUSDT','SLPUSDT','SNXUSDT','SOLUSDT','SOSUSDT','SRMUSDT','STMXUSDT',
        'STORJUSDT','STXUSDT','SUNUSDT','SUSHIUSDT','TROYUSDT','TRUUSDT','TRXUSDT',
        'TWTUSDT','UMAUSDT','UNIUSDT','UNFIUSDT','VETUSDT','VIDTUSDT','VITEUSDT',
        'VTHOUSDT','WAXPUSDT','WAVESUSDT','WINGUSDT','WOOUSDT','XEMUSDT','XLMUSDT',
        'XMRUSDT','XNOUSDT','XRPUSDT','XTZUSDT','YFIUSDT','ZILUSDT','ZRXUSDT'
    ])

# ─────────────────── MySQL Helpers ───────────────────
def conectar_mysql():
    conn = mysql.connector.connect(**DB_CFG)
    conn.autocommit = True
    return conn

def ensure_database():
    cfg = DB_CFG.copy(); cfg.pop("database", None)
    with mysql.connector.connect(**cfg) as c:
        cur = c.cursor()
        cur.execute(
            "CREATE DATABASE IF NOT EXISTS `crypto_second` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        cur.close()

COL_DEFS = (
    "dates BIGINT PRIMARY KEY",
    "closing_prices FLOAT"
)
COL_NAMES = [c.split()[0] for c in COL_DEFS]

def preparar_tabela(sym: str):
    tbl = sym.lower()
    with conectar_mysql() as conn, conn.cursor() as cur:
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS `{tbl}` "
            f"({', '.join(COL_DEFS)}) ENGINE=InnoDB;"
        )
        conn.commit()

# ───────────────── Estado em memória ─────────────────
lock        = threading.Lock()
agg_data    = {}                # sym -> {'bucket_ts', 'close'}
queue_to_db = defaultdict(list) # sym -> [ (dates, closing_prices) ]

# ───────────────── Core: processamento de trades ─────────────────
def process_trade(d: dict):
    """
    Recebe evento 'trade' e guarda o último preço visto em cada segundo.
    """
    sym   = d["s"]
    price = float(d["p"])
    ts_s  = d["E"] // 1000              # timestamp em segundos
    bucket = ts_s - (ts_s % CANDLE_SEC) # bucket de 1 s

    a = agg_data.get(sym)
    if (not a) or (a["bucket_ts"] != bucket):
        agg_data[sym] = {"bucket_ts": bucket, "close": price}
    else:
        a["close"] = price

# ───────────── Snapshot worker (flush a cada 1 s) ─────────────
def snapshot_worker():
    cols         = ", ".join(COL_NAMES)
    placeholders = ", ".join(["%s"] * len(COL_NAMES))
    update_str   = f"{COL_NAMES[1]}=VALUES({COL_NAMES[1]})"

    while True:
        now = time.time()
        time.sleep(CANDLE_SEC - (now % CANDLE_SEC))

        with lock:
            to_flush = dict(agg_data)
            agg_data.clear()

        for sym, agg in to_flush.items():
            # bucket_ts em segundos → ms
            queue_to_db[sym].append((agg["bucket_ts"] * 1000, agg["close"]))

        if not queue_to_db:
            continue

        try:
            conn = conectar_mysql(); cur = conn.cursor()
            for sym, rows in queue_to_db.items():
                tbl = sym.lower()
                cur.executemany(
                    f"INSERT INTO `{tbl}` ({cols}) VALUES ({placeholders}) "
                    f"ON DUPLICATE KEY UPDATE {update_str}",
                    rows
                )
            conn.commit()
            queue_to_db.clear()
            log.info("Flushed %d rows", sum(len(r) for r in to_flush.values()))
        except Exception as e:
            log.error("DB insert error: %s", e)

# ───────────── WebSocket callbacks ─────────────
def on_message(_, msg):
    try:
        d = json.loads(msg)
        data = d.get("data", d)
        if data.get("e") == "trade":
            with lock:
                process_trade(data)
    except Exception as e:
        log.warning("Erro ao processar mensagem: %s", e)

def on_error(_, err):
    log.error("WS error: %s", err)

def on_close(_, code, msg):
    log.warning("WebSocket closed – reconnecting… code=%s msg=%s", code, msg)

def launch_ws(chunk):
    streams = [f"{s.lower()}@trade" for s in chunk]
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
    while True:
        ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        try:
            ws.run_forever(ping_interval=PING_INTERVAL, ping_timeout=PING_TIMEOUT)
        except Exception as e:
            log.warning("WS reconnect loop: %s", e)
            time.sleep(5)

def chunked(it, n):
    it = iter(it)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk

# ───────────────────────── main ─────────────────────────
def main():
    ensure_database()
    symbols = list(obter_lista_criptomoedas())
    log.info("Criando/verificando tabelas em MySQL…")
    for s in symbols:
        preparar_tabela(s)

    threading.Thread(target=snapshot_worker, daemon=True).start()

    for chunk in chunked(symbols, WS_CHUNK):
        threading.Thread(target=launch_ws, args=(chunk,),
                         name=f"ws-{chunk[0]}", daemon=True).start()

    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
