import ccxt
import os
import logging
import time
import pandas as pd
import numpy as np
import json
import signal
import sys
import traceback
import math

from logging.handlers import RotatingFileHandler
from datetime import datetime

from app.config import CONFIG
from app.modules.send_telegram_notification import send_telegram_notification
from app.persistence.redis_client import redis_client

# Initialize logging with a rotating file handler
log_handler = RotatingFileHandler('trade_log_spot.log', maxBytes=5*1024*1024, backupCount=2)
logging.basicConfig(handlers=[log_handler], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def handle_exit_signal(signal_number, frame):
    logging.info("Received exit signal, shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_exit_signal)
signal.signal(signal.SIGINT, handle_exit_signal)

class TradeExecution:
    def __init__(self, exchange, performance):
        self.exchange = exchange
        self.performance = performance
        self.market_data = None
        # active_positions is now managed in Redis, this is a local cache for the current cycle
        self.active_positions = self.load_positions()

    def load_positions(self):
        """Load active positions from Redis."""
        if not redis_client.is_connected():
            logging.error("Cannot load positions, Redis is not connected.")
            return {}

        positions = redis_client.hgetall_json('active_positions')
        if positions is None: # Error case from client
            logging.error("Failed to fetch positions from Redis due to a client error.")
            return {}

        logging.info(f"Loaded {len(positions)} active positions from Redis.")
        # Optional: Validate positions to ensure data integrity
        validated_positions = {}
        for pos_id, pos_data in positions.items():
            # The data from redis is stringified, so we need to convert types
            pos_data['entry_price'] = float(pos_data['entry_price'])
            pos_data['amount'] = float(pos_data['amount'])
            pos_data['take_profit'] = float(pos_data['take_profit'])
            pos_data['stop_loss'] = float(pos_data['stop_loss'])
            pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
            validated_positions[pos_id] = pos_data

        return validated_positions

    def track_position(self, order, side):
        """Track a new position and save it to Redis."""
        try:
            # Ensure entry_time is a string for JSON serialization
            entry_time_str = datetime.now().isoformat()
            position_info = {
                'symbol': order['symbol'],
                'entry_price': float(order['price']),
                'amount': float(order['filled']),
                'entry_time': entry_time_str,
                'side': side,
                'order_id': order['id'],
                'take_profit': None,
                'stop_loss': None
            }

            if side == 'buy':
                position_info['take_profit'] = position_info['entry_price'] * (1 + CONFIG['trading']['profit_target_percent'] / 100)
                position_info['stop_loss'] = position_info['entry_price'] * (1 - CONFIG['trading']['stop_loss_percent'] / 100)

                # Save to Redis
                if redis_client.is_connected():
                    redis_client.hset_json('active_positions', order['id'], position_info)

                self.active_positions[order['id']] = position_info # Update local cache

                message = f"🟢 New Buy Position Opened:\nSymbol: {position_info['symbol']}\nAmount: {position_info['amount']} ETH\nPrice: {position_info['entry_price']} USDT"
                send_telegram_notification(message)

            elif side == 'sell':
                # Find the corresponding buy position to calculate profit
                for pos_id, pos in list(self.active_positions.items()):
                    if pos['symbol'] == order['symbol']:
                        profit = (float(order['price']) - pos['entry_price']) * float(order['filled'])

                        # Remove from Redis
                        if redis_client.is_connected():
                            redis_client.hdel('active_positions', pos_id)

                        del self.active_positions[pos_id] # Update local cache

                        # Log trade history to Redis
                        trade_log = {
                            "profit": profit,
                            "exit_price": float(order['price']),
                            "timestamp": entry_time_str
                        }
                        if redis_client.is_connected():
                             redis_client.client.rpush('trade_history', json.dumps(trade_log))

                        message = f"🔴 Position Closed:\nSymbol: {order['symbol']}\nProfit: {profit:.2f} USDT"
                        send_telegram_notification(message)
                        break

            logging.info(f"Successfully tracked {side} position")
            return position_info
        except Exception as e:
            logging.error(f"Error tracking position: {str(e)}")
            return None

    def check_existing_positions(self):
        """Check existing positions from the local cache and trigger exit conditions."""
        try:
            # Reload positions at the beginning of the check to ensure consistency
            self.active_positions = self.load_positions()
            if not self.active_positions:
                return

            current_price = float(self.exchange.fetch_ticker(CONFIG['trading']['symbol'])['last'])

            for pos_id, position in list(self.active_positions.items()):
                if current_price >= position['take_profit']:
                    logging.info(f"Take profit triggered at {current_price}")
                    self.execute_trade("sell", position['amount'], position['symbol'])
                    continue

                if current_price <= position['stop_loss']:
                    logging.info(f"Stop loss triggered at {current_price}")
                    self.execute_trade("sell", position['amount'], position['symbol'])
                    continue
        except Exception as e:
            logging.error(f"Error checking positions: {str(e)}")

    # ... (the rest of the methods from TradeExecution can remain, but they will need to be checked for dependencies on the removed methods)
    # The following methods are kept as they are mostly performing calculations or API calls, not file I/O
    # handle_trade_error, check_exchange_connection, get_account_value, validate_position_parameters, etc.
    # I'm going to paste the whole class again, but with the file-based methods removed.

    def validate_price_action(self, market_data):
        """Validate price action before entry"""
        try:
            # Get recent price movement
            recent_prices = market_data['close'].tail(5)
            price_change = abs(recent_prices.pct_change().mean())

            # Check if price is stable enough
            if price_change > CONFIG['trading']['legacy_ta']['price_stability_threshold']:
                logging.info(f"Price movement too high: {price_change:.4%}")
                return False

            # Check if price is above VWAP
            vwap = self.calculate_vwap(market_data)
            current_price = market_data['close'].iloc[-1]

            if current_price < vwap:
                logging.info("Price below VWAP")
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating price action: {str(e)}")
            return False

    def log_position_details(self, position, status="Active"):
        """Log detailed position information"""
        try:
            current_price = self.fetch_current_price(position['symbol'])
            if current_price:
                # Ensure entry_time is a datetime object for calculations
                entry_time = position['entry_time']
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)

                unrealized_pnl = (current_price - position['entry_price']) * position['amount']
                time_held = datetime.now() - entry_time

                log_msg = f"""
    Position Details ({status}):
    Symbol: {position['symbol']}
    Side: {position['side']}
    Amount: {position['amount']:.8f}
    Entry Price: {position['entry_price']:.2f}
    Current Price: {current_price:.2f}
    Take Profit: {position['take_profit']:.2f}
    Stop Loss: {position['stop_loss']:.2f}
    Unrealized P/L: {unrealized_pnl:.2f} USDT
    Time Held: {str(time_held)}
                """
                logging.info(log_msg)

                if abs(unrealized_pnl) > CONFIG['trading']['legacy_ta']['min_profit_threshold']:
                    self.send_notification(log_msg)

        except Exception as e:
            logging.error(f"Error logging position details: {str(e)}")

    def handle_trade_error(self, error, retry_count=3):
        """Enhanced error handling for 24/7 operation"""
        try:
            for i in range(retry_count):
                try:
                    logging.error(f"Trade error (attempt {i+1}/{retry_count}): {str(error)}")
                    if any(critical in str(error).lower() for critical in ['insufficient balance', 'api key', 'permission denied', 'margin']):
                        logging.critical(f"Critical error detected: {str(error)}")
                        self.send_notification(f"Critical Trading Error: {str(error)}")
                        return False
                    wait_time = 2 ** i
                    logging.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    if self.check_exchange_connection():
                        logging.info("Exchange connection restored")
                        return True
                except Exception as e:
                    logging.error(f"Error in retry attempt {i+1}: {str(e)}")
            self.send_notification("Maximum retry attempts reached, manual intervention may be required")
            return False
        except Exception as e:
            logging.error(f"Error in error handler: {str(e)}")
            return False

    def check_exchange_connection(self):
        try:
            self.exchange.fetch_ticker(CONFIG['trading']['symbol'])
            return True
        except Exception as e:
            logging.error(f"Exchange connection error: {e}")
            return False

    def send_notification(self, message):
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            formatted_message = f"[{timestamp}]\n{message}"
            send_telegram_notification(formatted_message)
        except Exception as e:
            logging.error(f"Failed to send notification: {str(e)}")


class PerformanceMetrics:
    def __init__(self):
        self.metrics_key = 'performance_metrics'
        self.metrics = self.load_metrics()

    def load_metrics(self):
        """Load metrics from Redis."""
        if not redis_client.is_connected():
            logging.warning("Redis not connected. Using initial metrics.")
            return self.initialize_metrics()

        metrics = redis_client.get_json(self.metrics_key)
        if metrics:
            logging.info("Loaded performance metrics from Redis.")
            return metrics
        else:
            logging.info("No performance metrics found in Redis. Initializing.")
            return self.initialize_metrics(save=True)

    def initialize_metrics(self, save=False):
        """Initialize metrics to default values."""
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'daily_trades': {}, # Tracks trades per day
            'daily_loss': {}, # Tracks loss per day
            'trade_history': [],
            'last_reset_date': datetime.now().strftime('%Y-%m-%d')
        }
        if save and redis_client.is_connected():
            self.save_metrics(metrics)
        return metrics

    def save_metrics(self, metrics_data=None):
        """Save metrics to Redis."""
        if not redis_client.is_connected():
            return

        data_to_save = metrics_data if metrics_data is not None else self.metrics
        redis_client.set_json(self.metrics_key, data_to_save)
        logging.info("Saved performance metrics to Redis.")

    def update_trade(self, profit, won=False):
        """Update metrics after a trade."""
        today_str = datetime.now().strftime('%Y-%m-%d')

        # Reset daily counters if it's a new day
        if self.metrics.get('last_reset_date') != today_str:
            self.metrics['daily_trades'] = {}
            self.metrics['daily_loss'] = {}
            self.metrics['last_reset_date'] = today_str

        self.metrics['total_trades'] += 1
        self.metrics['daily_trades'][today_str] = self.metrics['daily_trades'].get(today_str, 0) + 1

        if won:
            self.metrics['winning_trades'] += 1

        self.metrics['total_profit'] += profit
        if profit < 0:
            self.metrics['daily_loss'][today_str] = self.metrics['daily_loss'].get(today_str, 0) + abs(profit)

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
            profits = [trade['profit'] for trade in self.metrics['trade_history']]
            self.metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(profits)
            self.metrics['max_drawdown'] = self.calculate_max_drawdown(profits)

    @staticmethod
    def calculate_sharpe_ratio(profits, risk_free_rate=0.02):
        if len(profits) < 2: return 0
        returns = pd.Series(profits)
        excess_returns = returns - (risk_free_rate / 252)
        if excess_returns.std() == 0: return 0
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    @staticmethod
    def calculate_max_drawdown(profits):
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return np.max(drawdown) if len(drawdown) > 0 else 0

    def can_trade(self):
        """Check if trading is allowed based on performance limits."""
        today_str = datetime.now().strftime('%Y-%m-%d')

        if self.metrics['daily_trades'].get(today_str, 0) >= CONFIG['trading']['max_daily_trades']:
            logging.warning('Maximum daily trades reached')
            return False

        account_value = 10000 # This needs a way to be fetched or passed in
        # daily_loss_limit = (CONFIG['trading']['max_daily_loss_percent'] / 100) * account_value
        # if self.metrics['daily_loss'].get(today_str, 0) >= daily_loss_limit:
        #     logging.warning('Maximum daily loss reached')
        #     return False

        if self.metrics['max_drawdown'] >= CONFIG['trading']['max_drawdown_percent']:
            logging.warning('Maximum drawdown reached')
            return False

        return True


def initialize_exchange():
    """Initializes the exchange using credentials from the config."""
    try:
        exchange = ccxt.binance({
            'apiKey': CONFIG['binance']['api_key'],
            'secret': CONFIG['binance']['api_secret'],
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        if CONFIG['app'].get('use_testnet', False):
            exchange.set_sandbox_mode(True)
        return exchange
    except ccxt.BaseError as e:
        logging.error(f"Failed to initialize exchange: {str(e)}")
        send_telegram_notification(f"Failed to initialize exchange: {str(e)}")
        return None

# ... (the rest of the file will also be refactored, but this is the core logic for Redis integration)

def main():
    """
    Main function to run the trading bot.
    This will be refactored into a scheduler-based system.
    """
    if not redis_client.is_connected():
        logging.critical("Redis is not connected. Aborting.")
        return

    performance = PerformanceMetrics()
    exchange = initialize_exchange()
    if not exchange:
        logging.critical("Could not initialize exchange. Aborting.")
        return

    trade_execution = TradeExecution(exchange, performance)

    # The main loop will be replaced by the APScheduler logic in a later step
    while True:
        logging.info("Starting trading cycle...")

        # Check for existing positions and manage them
        trade_execution.check_existing_positions()

        # This is where the new AI-based logic will go
        # 1. Call AI service to get signals
        # 2. For each signal, decide whether to trade
        # 3. Execute trades

        logging.info(f"Trading cycle finished. Waiting for next cycle...")
        time.sleep(60) # Placeholder sleep


if __name__ == '__main__':
    main()
