# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import subprocess
import sys
import time


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "tradebot":
            return p
    return here.parent


REPO_ROOT = _repo_root()
WORKSPACE_ROOT = REPO_ROOT.parent
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"train_until_success_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def _fmt_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_fmt_ts()}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _is_valid_parquet(path: Path) -> bool:
    try:
        if (not path.exists()) or path.stat().st_size < 16:
            return False
        with path.open("rb") as f:
            head = f.read(4)
            if head != b"PAR1":
                return False
            f.seek(-4, os.SEEK_END)
            tail = f.read(4)
            return tail == b"PAR1"
    except Exception:
        return False


def _purge_invalid_parquet(cache_dir: Path, *, label: str) -> int:
    removed = 0
    if not cache_dir.exists():
        return 0
    for p in cache_dir.rglob("*.parquet"):
        if not _is_valid_parquet(p):
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    if removed > 0:
        _log(f"[auto-heal] removidos {removed} parquet(s) corrompidos em {label}: {cache_dir}")
    return removed


def _auto_heal() -> None:
    raw_candle = (os.getenv("SNIPER_CANDLE_SEC") or os.getenv("PF_CRYPTO_CANDLE_SEC") or "60").strip()
    try:
        candle_sec = max(1, int(raw_candle))
    except Exception:
        candle_sec = 60
    feat_tag = f"{int(candle_sec // 60)}m" if candle_sec % 60 == 0 else f"{int(candle_sec)}s"
    feat_cache = os.getenv("SNIPER_FEATURE_CACHE_DIR", "").strip()
    if feat_cache:
        feat_dir = Path(feat_cache)
    else:
        feat_dir = WORKSPACE_ROOT / "cache_sniper" / f"features_pf_{feat_tag}"
    ohlc_cache = os.getenv("PF_OHLC_CACHE_DIR", "").strip()
    if ohlc_cache:
        ohlc_dir = Path(ohlc_cache)
    else:
        ohlc_dir = WORKSPACE_ROOT / "cache_sniper" / "ohlc_1m"
    _purge_invalid_parquet(feat_dir, label="features")
    _purge_invalid_parquet(ohlc_dir, label="ohlc")


def _run_once() -> int:
    cmd = [sys.executable, str(REPO_ROOT / "crypto" / "train_sniper_wf.py")]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    _log(f"[runner] iniciando comando: {' '.join(cmd)}")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        proc.wait()
        return int(proc.returncode or 0)


def main() -> None:
    _log(f"[runner] log em: {LOG_PATH}")
    attempt = 0
    while True:
        attempt += 1
        _log(f"[runner] tentativa #{attempt}")
        rc = _run_once()
        if rc == 0:
            _log("[runner] treino concluido com sucesso (exit_code=0)")
            return
        _log(f"[runner] falha detectada (exit_code={rc}); iniciando auto-heal e retry")
        _auto_heal()
        sleep_s = 20
        _log(f"[runner] aguardando {sleep_s}s antes de tentar novamente")
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
