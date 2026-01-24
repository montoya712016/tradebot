import mysql.connector
import requests
import numpy as np
import time as t 
from binance.client import Client
from datetime import datetime 
from binance.exceptions import BinanceAPIException, BinanceOrderException
from decimal import Decimal, getcontext, ROUND_DOWN
import csv
import os

def ajustar_quantidade(quantidade, precisao):
    final_value = precisao.normalize()
    quantidade_ajustada = quantidade.quantize(final_value, rounding=ROUND_DOWN)
    return quantidade_ajustada

def calcular_valor_total_em_dolares():
    #Abre a instância do cliente
    while True:
        try:
            client = Client(api_key, api_secret)
            break
        except Exception:
            enviar_mensagem_Pushover('erro ao abrir o cliente', pushover_token_error)
            t.sleep(10)

    account_spot = client.get_account()
    account_margin = client.get_margin_account()
    
    balances = account_spot['balances'] + account_margin['userAssets']

    total_usd = 0.0
    
    for balance in balances:
        ticker = balance['asset']
        free_balance = float(balance['free'])
        
        if free_balance > 0:
            if ticker != 'USD' and ticker != 'USDT':
                try:
                    symbol = ticker + 'USDT'
                    ticker_price = client.get_symbol_ticker(symbol=symbol)
                    price_usd = float(ticker_price['price'])
                    total_usd += free_balance * price_usd
                except Exception as e:
                    print(f"Não foi possível obter o preço para {ticker}: {e}")
            else:
                total_usd += free_balance

    for balance in account_margin['userAssets']:
        ticker = balance['asset']
        borrowed = float(balance['borrowed'])

        if borrowed > 0:
            if ticker != 'USD' and ticker != 'USDT':
                try:
                    symbol = ticker + 'USDT'
                    ticker_price = client.get_symbol_ticker(symbol=symbol)
                    price_usd = float(ticker_price['price'])
                    total_usd -= borrowed * price_usd
                except Exception as e:
                    print(f"Não foi possível obter o preço para {ticker} emprestado: {e}")
            else:
                total_usd -= borrowed

    return total_usd

def enviar_mensagem_Pushover(mensagem, token):
    print(mensagem)
    ordem = mensagem
    data = {
        'token': token,
        'user': pushover_user_key,
        'message': ordem,
    }
    while True:
        try:
            response = requests.post(pushover_url, data=data)
            break
        except Exception as e:
            print('Erro ao enviar mensagem', str(e))
            t.sleep(10)

def enviar_mensagem_erro_lot_size(symbol, quantidade_tentada, lot_size_info):
    step_size = lot_size_info['stepSize']
    min_qty = lot_size_info['minQty']
    max_qty = lot_size_info['maxQty']
    erro_msg = (
        f"Erro ao realizar a ordem para {symbol}: LOT_SIZE\n"
        f"Quantidade tentada: {quantidade_tentada}\n"
        f"Exigências da corretora - Step Size: {step_size}, Min Qty: {min_qty}, Max Qty: {max_qty}"
    )
    enviar_mensagem_Pushover(erro_msg, pushover_token_error)

def verificar_posicao(symbol):
    symbol_to_number = {}
    arquivo_path = 'C:\\Users\\NovoLucas\\programacao\\investimento\\crypto\\programa_tempo_real\\posicoes.txt'
    
    with open(arquivo_path, 'r') as arquivo:
        for linha in arquivo:
            partes = linha.split()
            symbol_arquivo = partes[0]
            numero = int(partes[1])
            symbol_to_number[symbol_arquivo] = numero

    posicao = symbol_to_number.get(symbol, None)
    return posicao

def atualizar_posicao(symbol, novo_valor):
    arquivo_path = 'C:\\Users\\NovoLucas\\programacao\\investimento\\crypto\\programa_tempo_real\\posicoes.txt'
    symbol_to_number = {}
    
    with open(arquivo_path, 'r') as arquivo:
        for linha in arquivo:
            partes = linha.split()
            symbol_arquivo = partes[0]
            numero = int(partes[1])
            symbol_to_number[symbol_arquivo] = numero

    symbol_to_number[symbol] = novo_valor

    with open(arquivo_path, 'w') as arquivo:
        for symbol_arquivo, numero in symbol_to_number.items():
            arquivo.write(f'{symbol_arquivo} {numero}\n')

def calculate_ema(data, janela, partida):
    multiplier = 2 / (janela + 1)
    ema = (data - partida) * multiplier + partida
    return ema

def calculate_diff(ema_small, ema_large, diff_window, partida):
    indicator = 100 * abs(ema_large - ema_small) / ema_large
    ema_indicator = calculate_ema(indicator, diff_window, partida)
    return ema_indicator

def calcular_factor(shitcoin_index, cte_factor, coef_angular, coef_linear, relacao_ordem, maximo, minimo):
    if shitcoin_index != 0:
        relacao = cte_factor * (coef_angular * shitcoin_index + coef_linear) ** relacao_ordem
    else:
        if relacao_ordem > 0:
            relacao = 0
        elif relacao_ordem < 0:
            relacao = 10 * 1e10
        else:
            relacao = cte_factor
    if relacao > maximo:
        relacao = maximo
    elif relacao < minimo:
        relacao = minimo
    fator = relacao
    
    return fator

def comprar(symbol, alavancagem, posicao=150):
    print("COMPRAR!")
    client = None
    tentativa = 0
    while True:
        try:
            client = Client(api_key, api_secret)
            break
        except Exception as e:
            enviar_mensagem_Pushover('erro ao abrir o cliente: ' + str(e), pushover_token_error)
            t.sleep(10 * (2 ** tentativa))
            tentativa += 1

    symbol_info = client.get_symbol_info(symbol)
    lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', symbol_info['filters']))
    step_size = Decimal(lot_size['stepSize'])
    min_qty = Decimal(lot_size['minQty'])
    max_qty = Decimal(lot_size['maxQty'])

    quantidade_em_USDT = Decimal(f'{posicao}')
    ticker_price = Decimal(client.get_symbol_ticker(symbol=symbol)['price'])
    quantidade_em_crypto = ajustar_quantidade((alavancagem * quantidade_em_USDT / ticker_price), step_size)

    if alavancagem > 1:
        tentativa = 0
        while True:
            try:
                loan = client.create_margin_loan(asset='USDT', amount=str(quantidade_em_USDT * (alavancagem - 1)))
                break
            except Exception as e:
                enviar_mensagem_Pushover(f'erro ao pegar empréstimo para usar alavancagem: {str(e)}', pushover_token_error)
                t.sleep(10 * (2 ** tentativa))
                tentativa += 1

    tentativa = 0
    while True:
        try:
            order = client.create_margin_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(quantidade_em_crypto)
            )
            print("Ordem realizada com sucesso:", order)
            break
        except (BinanceAPIException, BinanceOrderException) as e:
            if 'LOT_SIZE' in e.message:
                enviar_mensagem_erro_lot_size(symbol, quantidade_em_crypto, {'stepSize': step_size, 'minQty': min_qty, 'maxQty': max_qty})
            else:
                enviar_mensagem_Pushover(f'erro ao realizar a compra para {symbol}: {str(e)}', pushover_token_error)
            t.sleep(10 * (2 ** tentativa))
            tentativa += 1

def vender(symbol, alavancagem):
    print("VENDER!")
    client = None
    tentativa = 0
    while True:
        try:
            client = Client(api_key, api_secret)
            break
        except Exception as e:
            enviar_mensagem_Pushover('erro ao abrir o cliente: ' + str(e), pushover_token_error)
            t.sleep(10 * (2 ** tentativa))  # Aplica backoff exponencial
            tentativa += 1

    symbol_info = client.get_symbol_info(symbol)
    lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', symbol_info['filters']))
    step_size = Decimal(lot_size['stepSize'])

    moeda = symbol.replace('USDT', '')
    margin_account = client.get_margin_account()

    for asset in margin_account['userAssets']:
        if asset['asset'] == moeda:
            quantidade_em_crypto = ajustar_quantidade(Decimal(asset['netAsset']), step_size)
            print(f"Quantidade de {moeda} em margem: {quantidade_em_crypto}")
            break
    
    tentativa = 0
    while True:
        try:
            order = client.create_margin_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(quantidade_em_crypto)
            )
            print("Ordem realizada com sucesso:", order)
            break
        except (BinanceAPIException, BinanceOrderException) as e:
            erro_msg = f'erro ao vender a quantidade disponivel: {quantidade_em_crypto}\n{str(e)}'
            enviar_mensagem_Pushover(erro_msg, pushover_token_error)
            quantidade_em_crypto = (quantidade_em_crypto * Decimal('0.99')).quantize(Decimal(step_size), rounding=ROUND_DOWN)
            t.sleep(10 * (2 ** tentativa))  # Aplica backoff exponencial
            tentativa += 1

    if alavancagem > 1:
        try:
            repayment = client.repay_margin_loan(asset='USDT', amount=str(100))
            print(f"Reembolso de empréstimo realizado com sucesso para {moeda}")
        except Exception as e:
            erro_msg = f'erro ao pagar o emprestimo usado na alavancagem para {moeda}\n{str(e)}'
            enviar_mensagem_Pushover(erro_msg, pushover_token_error)

def shortar(symbol, alavancagem, posicao=150):
    print("SHORTAR!")
    client = None
    tentativa = 0
    while True:
        try:
            client = Client(api_key, api_secret)
            break
        except Exception as e:
            enviar_mensagem_Pushover('erro ao abrir o cliente: ' + str(e), pushover_token_error)
            t.sleep(10 * (2 ** tentativa))  # Aplica backoff exponencial
            tentativa += 1

    moeda = symbol.replace('USDT', '')
    symbol_info = client.get_symbol_info(symbol)
    lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', symbol_info['filters']))
    step_size = Decimal(lot_size['stepSize'])

    ticker_price = Decimal(client.get_symbol_ticker(symbol=symbol)['price'])
    quantidade_usdt = Decimal(f'{posicao}')
    quantidade_em_crypto = ajustar_quantidade(Decimal(str(alavancagem * (quantidade_usdt / ticker_price))), step_size)

    # Empréstimo do ativo
    attempts = 0
    while True:
        try:
            loan = client.create_margin_loan(asset=moeda, amount=str(quantidade_em_crypto))
            print("Empréstimo realizado com sucesso:", loan)
            break
        except Exception as e:
            enviar_mensagem_Pushover(f'erro ao pegar empréstimo\nquantidade para emprestar: {quantidade_em_crypto}\nErro: {str(e)}', pushover_token_error)
            t.sleep(10 * (2 ** attempts))  # Aplica backoff exponencial
            attempts += 1

    # Cria a ordem de venda a descoberto
    attempts = 0
    while True:
        try:
            order = client.create_margin_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(quantidade_em_crypto)  # A mesma quantidade que foi emprestada
            )
            print("Ordem de short realizada com sucesso:", order)
            break
        except Exception as e:
            enviar_mensagem_Pushover(f'erro ao realizar a ordem de venda a descoberto\nquantidade para vender: {quantidade_em_crypto}\nErro: {str(e)}', pushover_token_error)
            t.sleep(10 * (2 ** attempts))
            attempts += 1

def encerrar_short(symbol):
    print("ENCERRAR SHORT!")
    client = None
    tentativa = 0
    while True:
        try:
            client = Client(api_key, api_secret)
            break
        except Exception as e:
            enviar_mensagem_Pushover('erro ao abrir o cliente: ' + str(e), pushover_token_error)
            t.sleep(10 * (2 ** tentativa))  # Aplica backoff exponencial
            tentativa += 1

    moeda = symbol.replace('USDT', '')
    symbol_info = client.get_symbol_info(symbol)
    lot_size = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', symbol_info['filters']))
    step_size = Decimal(lot_size['stepSize'])

    # Obtém informações da conta de margem
    margin_account = None
    while True:
        try:
            margin_account = client.get_margin_account()
            break
        except Exception as e:
            enviar_mensagem_Pushover(f'erro ao buscar os dados da conta: {str(e)}', pushover_token_error)
            t.sleep(10)

    saldo_devedor = Decimal('0')
    for asset in margin_account['userAssets']:
        if asset['asset'] == moeda:
            saldo_devedor = ajustar_quantidade((Decimal(asset['borrowed']) + Decimal(asset['interest'])), step_size)

    # Compra para cobrir a posição a descoberto
    attempts = 0
    while True:
        try:
            order = client.create_margin_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(saldo_devedor)
            )
            print("Posição a descoberto encerrada com sucesso:", order)
            break
        except Exception as e:
            enviar_mensagem_Pushover(f'erro ao comprar para cobrir a posição: {str(e)}', pushover_token_error)
            t.sleep(10 * (2 ** attempts))  # Aplica backoff exponencial
            attempts += 1

    # Reembolso do empréstimo
    attempts = 0
    while True:
        try:
            repayment = client.repay_margin_loan(asset=moeda, amount=str(saldo_devedor))
            print("Empréstimo reembolsado com sucesso.")
            break
        except Exception as e:
            enviar_mensagem_Pushover(f'erro ao pagar o empréstimo: {str(e)}', pushover_token_error)
            saldo_devedor = (saldo_devedor * Decimal('0.99')).quantize(Decimal(step_size), rounding=ROUND_DOWN)
            t.sleep(10 * (2 ** attempts))
            attempts += 1

def carregar_parametros(arquivo_csv):
    parametros = {}
    caminho_completo = os.path.join('parametros', arquivo_csv)
    with open(caminho_completo, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Pular cabeçalho
        for row in reader:
            symbol = row[0]
            valores = []
            for valor in row[1:]:
                valor_float = float(valor)
                # Se o valor não tiver casas decimais, convertê-lo para inteiro
                if valor_float.is_integer():
                    valores.append(int(valor_float))
                else:
                    valores.append(valor_float)
            parametros[symbol] = valores
    return parametros

#Senhas

pushover_token_trade = 'a3b7tr2aoosgfgors3w4365ihsknkk'
pushover_token_hourly_notification = 'a9s3vzfh2qqsw4s32mcf6q8m3zd1uf'
pushover_token_error = 'aw1aqdwnq6zdc9f3n23two8xvsc1bz'

pushover_user_key = 'uhizfw5san2qdnwmi9d82i3th3g97t'
pushover_url = 'https://api.pushover.net/1/messages.json'

api_key = "dql1UlltOzAD5Xk5Y6NDNdE5KoCgszw2oiCLm2BrBXX5PNuSDTcmquZk9ycdgmg8"
api_secret = "WxSrIpRpvOqHFNQdRkS93GE0PkhyU8UMVVSckJfMnkkuaJCGPxu5geMde5M0fMYp"

api_key_sub_1= "xnTQYmkbV268ehYKIeNVpkwRExRkauFa1AYd7efUEgjOMbsgDMDmD0DOFdtQi3Vc"
api_secret_sub_1 = "3gn93UFEE9PuC18z7o5mv1Y3wZg2LN5F9kvOLzEasJ7oKo6aXOjdkhek9plKh7LU"

base_url = "https://api.binance.com/api/v3"
interval = '1m'

#Criptomoedas a serem analisadas

moedas = ['AAVEUSDT', 'ADAUSDT', 'ARBUSDT', 'ATOMUSDT', 'AVAXUSDT', 'BEAMXUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 'DOTUSDT', \
        'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'HBARUSDT', 'LINKUSDT', 'LTCUSDT', 'MKRUSDT', 'NTRNUSDT', 'PEPEUSDT', \
        'ROSEUSDT', 'SHIBUSDT', 'SOLUSDT', 'STXUSDT', 'SUSHIUSDT', 'TIAUSDT', 'UNIUSDT', 'XLMUSDT', 'XRPUSDT', 'YGGUSDT']

#moedas_so_notificacao = ['YGGUSDT', 'TIAUSDT', 'STXUSDT', 'ATOMUSDT']
moedas_so_notificacao = []

#Parâmetros utilizados na análise

alavancagem = 1
valor_posicao = 100

operar = False
notificar = False

if operar == False:
    moedas_so_notificacao = moedas

cte_small_factor = {}
cte_factor = {}
cte_limiar = {}
relacao_limiar = {}
relacao_fator = {}
janela_shitcoin = {}
janela_diff = {}
janela_ema_ffd = {}
relacao_small_factor = {}
coef_angular_factor = {}
coef_angular_small_factor = {}
coef_linear_factor = {}
coef_linear_small_factor = {}

parametros = carregar_parametros('C:\\Users\\NovoLucas\\programacao\\investimento\\crypto\\otimizacao_em_c\\parametros\\tendencias.csv')

for symbol in moedas:
    if symbol not in parametros:
        parametros[symbol] = parametros['coringa']

for moeda in moedas:
    array = parametros[moeda]

    cte_small_factor[moeda] = array[0]
    cte_factor[moeda] = array[1]
    cte_limiar[moeda] = array[2]
    relacao_limiar[moeda] = array[3]
    relacao_fator[moeda] = array[4]
    janela_shitcoin[moeda] = array[5]
    janela_diff[moeda] = array[6]
    janela_ema_ffd[moeda] = array[7]
    relacao_small_factor[moeda] = array[8]
    coef_angular_factor[moeda] = array[9]
    coef_angular_small_factor[moeda] = coef_angular_factor[moeda] / 2
    coef_linear_factor[moeda] = array[10]
    coef_linear_small_factor[moeda] = coef_linear_factor[moeda]

while True:
    try:
        # Coloque o código principal do programa aqui, começando logo após o try
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='2017',
            database='crypto'
        )

        cursor = conn.cursor()

        ultimos_tempos = []
        salvar_dados = np.zeros(len(moedas))

        for symbol in moedas:
            consulta_sql = f"SELECT dates FROM {symbol} ORDER BY dates DESC LIMIT 1"
            cursor.execute(consulta_sql)
            response = cursor.fetchone()
            last_dates = response[0]
            ultimos_tempos.append(last_dates)

        last_notified_hour = None
        last_check_time = datetime.now()

        while True:

            current_time = datetime.now()
            current_hour = current_time.hour

            if (current_hour != last_check_time.hour and current_time.minute >= 1) or (last_check_time.minute < 1 and current_time.minute >= 1):
                valor_total_usd = calcular_valor_total_em_dolares()
                enviar_mensagem_Pushover(f"Programa está funcionando corretamente\nValor atual da conta: {valor_total_usd:.2f}", pushover_token_hourly_notification)
                last_notified_hour = current_hour

            last_check_time = current_time

            momento_unix = int(t.time())

            for i in range (len(ultimos_tempos)):
                if momento_unix >= (ultimos_tempos[i] / 1000) + 60:
                    salvar_dados[i] = 1

            for i in range (len(moedas)):

                if salvar_dados[i]:

                    # Deve ser menos de 2 segundos para ocorrer em tempo real sem delay
                    t.sleep(1)

                    print(moedas[i])

                    symbol = moedas[i]
                    ultima = ultimos_tempos[i]

                    consulta_sql = f"SELECT ema_prices, ema_ffd, shitcoin, indicator, ema_diff FROM {symbol} ORDER BY dates DESC LIMIT 1"
                    cursor.execute(consulta_sql)
                    response = cursor.fetchone()

                    last_ema_prices = response[0]
                    last_ema_ffd = response[1]
                    last_shitcoin = response[2]
                    last_indicator = response[3]
                    last_ema_diff = response[4]

                    momento_unix = int(t.time())

                    end_time = momento_unix * 1000
                    start_time = ultima + 60 * 1000
                    duracao = end_time - start_time
                    limit = int(duracao / (60 * 1000) + 1)

                    print(limit)

                    while limit > 0:
                        while True:
                            try:
                                response = requests.get(
                                    f"{base_url}/klines",
                                    params={
                                        "symbol": symbol,
                                        "interval": interval,
                                        "startTime": start_time,
                                        "limit": limit
                                    }
                                )
                                data = response.json()
                                if isinstance(data[0][0], (int, float)):
                                    break
                            except Exception:
                                print('erro ao buscar dados da binance')
                                t.sleep(10)
                        limit -= len(data)

                        print(limit)

                        for entry in data:

                            dates = int(entry[0])
                            open_prices = float(entry[1])
                            high_prices = float(entry[2])
                            low_prices = float(entry[3])
                            closing_prices = float(entry[4])
                            volume = float(entry[5])

                            # Cálculos de indicadores permanecem os mesmos
                            ema_ffd = calculate_ema(closing_prices, janela_ema_ffd[symbol], last_ema_ffd)
                            last_ema_ffd = ema_ffd

                            shitcoin_index = calculate_diff(closing_prices, ema_ffd, janela_shitcoin[symbol], last_shitcoin)
                            last_shitcoin = shitcoin_index

                            limiar_diff = calcular_factor(shitcoin_index, cte_limiar[symbol], 1, 0, relacao_limiar[symbol], 100, 1)
                            fator = calcular_factor(shitcoin_index, cte_factor[symbol], coef_angular_factor[symbol], coef_linear_factor[symbol], relacao_fator[symbol], 1e10, 0)
                            small_factor = calcular_factor(shitcoin_index, cte_small_factor[symbol], coef_angular_small_factor[symbol], coef_linear_small_factor[symbol], relacao_small_factor[symbol], 1e10, 0)

                            ema_prices = calculate_ema(closing_prices, small_factor, last_ema_prices)
                            last_ema_prices = ema_prices

                            ema_adjusted_by_diff = calculate_ema(closing_prices, fator, last_indicator)
                            last_indicator = ema_adjusted_by_diff

                            ema_diff = calculate_diff(closing_prices, ema_adjusted_by_diff, janela_diff[symbol], last_ema_diff)
                            last_ema_diff = ema_diff

                            # Atualizar a consulta de inserção para incluir os novos preços
                            insert_query = f"""
                            INSERT INTO {symbol} 
                            (dates, open_prices, high_prices, low_prices, closing_prices, ema_prices, ema_ffd, shitcoin, indicator, ema_diff, volume) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                            cursor.execute(insert_query, (dates, open_prices, high_prices, low_prices, closing_prices, ema_prices, ema_ffd, shitcoin_index, ema_adjusted_by_diff, ema_diff, volume))

                            conn.commit()

                        start_time = dates + 60 * 1000

                    if ema_adjusted_by_diff >= ema_prices:
                        decisao = 'vender'

                    if ema_adjusted_by_diff <= ema_prices:
                        decisao = 'comprar'

                    posicao = verificar_posicao(symbol)

                    if decisao == 'comprar':
                        if posicao == 2:

                            if symbol not in moedas_so_notificacao:

                                encerrar_short(symbol)
                                if notificar:
                                    enviar_mensagem_Pushover(f'{symbol}: SHORT ENCERRADO!', pushover_token_trade)
                            
                            else:
                                if notificar:
                                    enviar_mensagem_Pushover(f'{symbol}: ENCERRAR SHORT!', pushover_token_trade)

                            posicao = 0
                            atualizar_posicao(symbol, posicao)   

                    if decisao == 'vender':
                        if posicao == 1:

                            if symbol not in moedas_so_notificacao:

                                vender(symbol, alavancagem)
                                if notificar:
                                    enviar_mensagem_Pushover(f'{symbol}: COMPRA ENCERRADA!!', pushover_token_trade)
                            
                            else:
                                if notificar:
                                    enviar_mensagem_Pushover(f'{symbol}: ENCERRAR COMPRA!', pushover_token_trade)

                            posicao = 0
                            atualizar_posicao(symbol, posicao)

                    if decisao == 'comprar':
                        if posicao == 0 and ema_diff >= limiar_diff:

                            if symbol not in moedas_so_notificacao:
                                comprar(symbol, alavancagem, posicao=valor_posicao)
                                if notificar:
                                    enviar_mensagem_Pushover(f'{symbol}: COMPRA REALIZADA!\n${valor_posicao*alavancagem:.2f}', pushover_token_trade)

                            else:
                                if notificar:
                                    enviar_mensagem_Pushover(f'{symbol}: REALIZAR COMPRA!', pushover_token_trade)

                            posicao = 1
                            atualizar_posicao(symbol, posicao)

                    if decisao == 'vender':
                        if posicao == 0 and ema_diff >= limiar_diff:

                            if symbol not in moedas_so_notificacao:
                                shortar(symbol, alavancagem, posicao=valor_posicao)
                                if notificar:
                                    enviar_mensagem_Pushover(f'{symbol}: SHORT REALIZADO!\n${valor_posicao*alavancagem:.2f}', pushover_token_trade)

                            else:
                                if notificar:
                                    enviar_mensagem_Pushover(f'{symbol}: REALIZAR SHORT!', pushover_token_trade)

                            posicao = 2
                            atualizar_posicao(symbol, posicao)    

                    ultimos_tempos[i] = dates
                    salvar_dados[i] = 0
            t.sleep(1)
    except Exception as e:
        print(e)
        enviar_mensagem_Pushover('Programa parou', pushover_token_error)
        t.sleep(5)  # Pausa de 5 segundos antes de tentar novamente
