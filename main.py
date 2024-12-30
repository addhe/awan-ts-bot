# main_spot.py
import ccxt
import os
import logging
import time
import pandas as pd
import numpy as np
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime

from config.config import CONFIG
from src.modules.send_telegram_notification import send_telegram_notification

# Initialize logging with a rotating file handler
log_handler = RotatingFileHandler('trade_log_spot.log', maxBytes=5*1024*1024, backupCount=2)
logging.basicConfig(handlers=[log_handler], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# API Configuration
API_KEY = os.environ.get('API_KEY_SPOT_BINANCE')
API_SECRET = os.environ.get('API_SECRET_SPOT_BINANCE')

if API_KEY is None or API_SECRET is None:
    error_message = 'API credentials not found in environment variables'
    logging.error(error_message)
    send_telegram_notification(error_message)
    exit(1)

class PerformanceMetrics:
    def __init__(self):
        self.metrics_file = 'performance_metrics_spot.json'
        self.load_metrics()

    def load_metrics(self):
        try:
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        except FileNotFoundError:
            self.metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_profit': 0,
                'max_drawdown': 0,
                'daily_trades': 0,
                'daily_loss': 0,
                'trade_history': [],
                'last_reset_date': datetime.now().strftime('%Y-%m-%d')
            }
            self.save_metrics()

    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f)

    def update_trade(self, profit, won=False):
        today = datetime.now().strftime('%Y-%m-%d')

        if today != self.metrics['last_reset_date']:
            self.metrics['daily_trades'] = 0
            self.metrics['daily_loss'] = 0
            self.metrics['last_reset_date'] = today

        self.metrics['total_trades'] += 1
        self.metrics['daily_trades'] += 1

        if won:
            self.metrics['winning_trades'] += 1

        self.metrics['total_profit'] += profit
        if profit < 0:
            self.metrics['daily_loss'] += abs(profit)

        self.metrics['trade_history'].append({
            'timestamp': datetime.now().isoformat(),
            'profit': profit,
            'won': won
        })

        self.calculate_metrics()
        self.save_metrics()

    def calculate_metrics(self):
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = (self.metrics['winning_trades'] / self.metrics['total_trades']) * 100

            if len(self.metrics['trade_history']) > 0:
                profits = [trade['profit'] for trade in self.metrics['trade_history']]
                self.metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(profits)
                self.metrics['max_drawdown'] = self.calculate_max_drawdown(profits)

    @staticmethod
    def calculate_sharpe_ratio(profits, risk_free_rate=0.02):
        if len(profits) < 2:
            return 0
        returns = pd.Series(profits)
        excess_returns = returns - (risk_free_rate / 252)
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    @staticmethod
    def calculate_max_drawdown(profits):
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return np.max(drawdown) if len(drawdown) > 0 else 0

    def can_trade(self):
        if self.metrics['daily_trades'] >= CONFIG['max_daily_trades']:
            logging.warning('Maximum daily trades reached')
            return False

        if self.metrics['daily_loss'] >= (CONFIG['max_daily_loss_percent'] / 100):
            logging.warning('Maximum daily loss reached')
            return False

        if self.metrics['max_drawdown'] >= CONFIG['max_drawdown_percent']:
            logging.warning('Maximum drawdown reached')
            return False

        return True

def get_min_trade_amount_and_notional(exchange, symbol):
    """Fetch minimum trade amount and notional value for a specific symbol."""
    try:
        markets = safe_api_call(exchange.load_markets)
        if not markets:
            logging.error("Markets not loaded.")
            return None, None

        market = markets.get(symbol)
        if not market:
            logging.error(f"Market data not available for symbol: {symbol}")
            return None, None

        min_amount = market['limits']['amount'].get('min')
        min_notional_value = None

        if 'info' in market and 'filters' in market['info']:
            for f in market['info']['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    min_notional_value = float(f['minNotional'])
                    break

        if min_amount is None:
            logging.error(f"Minimum amount not found for symbol: {symbol}")
        if min_notional_value is None:
            logging.error(f"MIN_NOTIONAL filter not found for symbol: {symbol}")

        return min_amount, min_notional_value
    except Exception as e:
        logging.error(f"Failed to fetch market info: {str(e)}")
        return None, None

def initialize_exchange():
    try:
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        if CONFIG.get('use_testnet', False):
            exchange.set_sandbox_mode(True)  # Enable testnet mode
        return exchange
    except ccxt.BaseError as e:
        error_message = f"Failed to initialize exchange: {str(e)}"
        logging.error(error_message)
        send_telegram_notification(error_message)
        return None

def safe_api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except ccxt.NetworkError as e:
        logging.warning(f"Network error: {str(e)}. Retrying...")
        time.sleep(10)
        return safe_api_call(func, *args, **kwargs)
    except ccxt.RateLimitExceeded as e:
        logging.error(f"Rate limit exceeded: {str(e)}. Waiting before retrying...")
        time.sleep(60)
        return safe_api_call(func, *args, **kwargs)
    except ccxt.ExchangeError as e:
        logging.error(f"Exchange error: {str(e)}")
        return None

def fetch_market_data(exchange, symbol, timeframe):
    try:
        candles = safe_api_call(exchange.fetch_ohlcv, symbol, timeframe)
        if candles is None:
            return None
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        logging.error(f"Failed to fetch market data: {str(e)}")
        return None

def validate_config():
    try:
        required_fields = [
            'symbol', 'risk_percentage', 'min_balance',
            'max_daily_trades', 'max_daily_loss_percent'
        ]
        
        for field in required_fields:
            if field not in CONFIG:
                raise ValueError(f"Missing required config field: {field}")
            
        if CONFIG['min_balance'] < 0:
            raise ValueError("Min balance cannot be negative")
            
        if CONFIG['timeframe'] not in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h']:
            raise ValueError("Invalid timeframe")

        # Additional Validations
        if not (0 < CONFIG['risk_percentage'] <= 100):
            raise ValueError("Risk percentage must be between 0 and 100.")

        if not (0 < CONFIG['fee_rate'] < 1):
            raise ValueError("Fee rate must be a percentage less than 1.")

        if CONFIG['max_daily_loss_percent'] <= 0 or CONFIG['max_drawdown_percent'] <= 0:
            raise ValueError("Max daily loss and max drawdown must be positive numbers.")

        if CONFIG['ema_short_period'] <= 0 or CONFIG['ema_long_period'] <= 0:
            raise ValueError("EMA periods must be positive integers.")

        logging.info("Config validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Config validation failed: {e}")
        return False

def calculate_ema(df, period, column='close'):
    return df[column].ewm(span=period, adjust=False).mean()

def execute_trade(exchange, side, amount, symbol, performance):

    try:
        if side == "buy":
            order = exchange.create_market_buy_order(symbol, amount)
        elif side == "sell":
            order = exchange.create_market_sell_order(symbol, amount)
        logging.info(f"Executed {side} order: {order}")
        send_telegram_notification(f"Executed {side} order: {order}")

        # Determine profit/loss and trade success
        profit = calculate_profit(order)
        won = profit > 0
        performance.update_trade(profit, won)

    except Exception as e:
        logging.error(f"Failed to execute {side} order: {str(e)}")
        send_telegram_notification(f"Failed to execute {side} order: {str(e)}")

def get_min_trade_amount(exchange, symbol):
    """ Fetch the minimum trade amount for a specific symbol from the exchange. """
    try:
        # Fetch the market info for the specific symbol
        markets = safe_api_call(exchange.load_markets)
        market = markets.get(symbol)
        if market:
            return market['limits']['amount']['min']
        else:
            logging.error(f"Market data not available for symbol: {symbol}")
            return None
    except Exception as e:
        logging.error(f"Failed to fetch market info: {str(e)}")
        return None

def main(performance):
    exchange = initialize_exchange()
    symbol_base = CONFIG['symbol'].split('/')[0]  # Extract base currency (like ETH)

    if exchange is None:
        logging.error("Exchange initialization failed")
        return

    try:
        if not validate_config():
            logging.error("Configuration validation failed")
            return

        if not performance.can_trade():
            message = "Trading limits reached"
            logging.info(message)
            send_telegram_notification(message)
            return

        balance = safe_api_call(exchange.fetch_balance)
        if balance is not None and 'USDT' in balance and 'free' in balance['USDT']:
            usdt_balance = balance['USDT']['free']
            logging.info(f"Available USDT balance: {usdt_balance}")
        else:
            error_message = "Failed to retrieve balance"
            logging.error(error_message)
            send_telegram_notification(error_message)
            return

        if usdt_balance < CONFIG['min_balance']:
            error_message = f"Insufficient balance: {usdt_balance} USDT"
            logging.error(error_message)
            send_telegram_notification(error_message)
            return

        market_data = fetch_market_data(exchange, CONFIG['symbol'], CONFIG['timeframe'])
        if market_data is None:
            logging.error("Failed to retrieve market data")
            return

        market_data['ema_short'] = calculate_ema(market_data, CONFIG['ema_short_period'])
        market_data['ema_long'] = calculate_ema(market_data, CONFIG['ema_long_period'])

        ema_short_last, ema_short_prev = market_data['ema_short'].iloc[-1], market_data['ema_short'].iloc[-2]
        ema_long_last, ema_long_prev = market_data['ema_long'].iloc[-1], market_data['ema_long'].iloc[-2]

        latest_close_price = market_data['close'].iloc[-1]
        logging.info(f"Latest close price: {latest_close_price}")

        # Calculate the amount to trade while considering fees
        gross_amount_to_trade = (CONFIG['risk_percentage'] / 100) * usdt_balance / latest_close_price
        estimated_fee = gross_amount_to_trade * CONFIG['fee_rate']
        amount_to_trade = gross_amount_to_trade - estimated_fee  # Account for the fee in the trade size

        logging.info(f"Calculated gross trade amount: {gross_amount_to_trade} {symbol_base}")
        logging.info(f"Estimated fee: {estimated_fee} {symbol_base}")
        logging.info(f"Amount to trade after fee: {amount_to_trade} {symbol_base}")

        # Get dynamic minimum trade amount and notional
        min_trade_amount, min_notional = get_min_trade_amount_and_notional(exchange, CONFIG['symbol'])

        if min_trade_amount is None or min_notional is None:
            logging.warning("Unable to fetch dynamic minimum requirements; skipping trade execution.")
            return

        # Add precision handling before executing the trade
        decimals_allowed = 4  # Assume the allowed place for ETH/USDT; verify with Binance API
        amount_to_trade_formatted = round(amount_to_trade, decimals_allowed)
        logging.info(f"Formatted amount to trade: {amount_to_trade_formatted} {symbol_base}")

        # Calculate the notional value to verify against the minimum
        notional_value = amount_to_trade_formatted * latest_close_price

        if amount_to_trade_formatted < min_trade_amount:
            logging.warning(f"Formatted trade amount {amount_to_trade_formatted} {symbol_base} is below minimum threshold {min_trade_amount} {symbol_base}")
            return

        if notional_value < min_notional:
            logging.warning(f"Trade notional value {notional_value} USDT is below minimum notional threshold {min_notional} USDT")
            return

        if ema_short_prev < ema_long_prev and ema_short_last > ema_long_last:
            logging.info(f"Buy signal confirmed: EMA short {ema_short_last:.4f} over EMA long {ema_long_last:.4f}")
            execute_trade(exchange, "buy", amount_to_trade_formatted, CONFIG['symbol'], performance)

        elif ema_short_prev > ema_long_prev and ema_short_last < ema_long_last:
            logging.info(f"Sell signal confirmed: EMA short {ema_short_last:.4f} under EMA long {ema_long_last:.4f}")
            execute_trade(exchange, "sell", amount_to_trade_formatted, CONFIG['symbol'], performance)

    except Exception as e:
        error_message = f'Critical error in main loop: {str(e)}'
        logging.error(error_message)
        send_telegram_notification(error_message)


if __name__ == '__main__':
    performance = PerformanceMetrics()
    while True:
        main(performance)
        time.sleep(60)  # Execute the loop every minute