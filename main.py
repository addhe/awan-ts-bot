import ccxt
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Initialize logging
log_handler = RotatingFileHandler('spot_trade_log.log', maxBytes=5*1024*1024, backupCount=2)
logging.basicConfig(handlers=[log_handler], level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# API Configuration
API_KEY = os.environ.get('API_KEY_BINANCE')
API_SECRET = os.environ.get('API_SECRET_BINANCE')

if API_KEY is None or API_SECRET is None:
    logging.error('API credentials not found in environment variables')
    exit(1)

# Initialize Binance exchange
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True
})

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
EMA_SHORT_PERIOD = 12
EMA_LONG_PERIOD = 26
RISK_PERCENTAGE = 0.01  # 1% of balance risked per trade

def fetch_ohlcv(symbol, timeframe, limit=100):
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

def calculate_indicators(df):
    df['ema_short'] = df['close'].ewm(span=EMA_SHORT_PERIOD, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=EMA_LONG_PERIOD, adjust=False).mean()

def place_order(side, amount, price):
    try:
        order = exchange.createOrder(SYMBOL, 'limit', side, amount, price)
        logging.info(f"Order placed: {side} {amount} {SYMBOL} at {price}")
        return order
    except Exception as e:
        logging.error(f"Order placement failed: {e}")

def main():
    try:
        # Fetch market data
        ohlcv = fetch_ohlcv(SYMBOL, TIMEFRAME)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calculate indicators
        calculate_indicators(df)
        
        # Determine current market conditions
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        if last_row['ema_short'] > last_row['ema_long'] and prev_row['ema_short'] <= prev_row['ema_long']:
            side = 'buy'
        elif last_row['ema_short'] < last_row['ema_long'] and prev_row['ema_short'] >= prev_row['ema_long']:
            side = 'sell'
        else:
            logging.info("No trade signal")
            return

        # Fetch balance and compute order size
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        risk_amount = usdt_balance * RISK_PERCENTAGE

        order_amount = risk_amount / last_row['close']
        order_amount = round(order_amount, 5)  # Adjust the precision according to market rules

        # Place order
        place_order(side, order_amount, last_row['close'])
        
    except Exception as e:
        logging.error(f"Error in main loop: {e}")

if __name__ == '__main__':
    while True:
        main()
        time.sleep(60 * 60)  # Run every hour, adjust based on TIMEFRAME