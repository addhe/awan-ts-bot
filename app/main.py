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
from app.ai.connector import get_ai_signals

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
        self.market_data = {} # Store market data per symbol
        self.active_positions = self.load_positions()

    def load_positions(self):
        """Load active positions from Redis."""
        if not redis_client.is_connected():
            logging.error("Cannot load positions, Redis is not connected.")
            return {}

        positions = redis_client.hgetall_json('active_positions')
        if positions is None:
            logging.error("Failed to fetch positions from Redis due to a client error.")
            return {}

        logging.info(f"Loaded {len(positions)} active positions from Redis.")
        validated_positions = {}
        for pos_id, pos_data in positions.items():
            try:
                # Make the conversion robust: only convert if it's a string
                if isinstance(pos_data.get('entry_time'), str):
                    pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])

                pos_data['entry_price'] = float(pos_data['entry_price'])
                pos_data['amount'] = float(pos_data['amount'])
                pos_data['take_profit'] = float(pos_data['take_profit']) if pos_data.get('take_profit') else None
                pos_data['stop_loss'] = float(pos_data['stop_loss']) if pos_data.get('stop_loss') else None
                validated_positions[pos_id] = pos_data
            except (ValueError, TypeError) as e:
                logging.error(f"Could not validate position {pos_id} from Redis. Error: {e}. Data: {pos_data}")

        return validated_positions

    def track_position(self, order, side):
        """Track a new position and save it to Redis."""
        try:
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

                if redis_client.is_connected():
                    redis_client.hset_json('active_positions', order['id'], position_info)

                self.active_positions[order['id']] = position_info

                message = f"🟢 New Buy Position Opened:\nSymbol: {position_info['symbol']}\nAmount: {position_info['amount']}\nPrice: {position_info['entry_price']}"
                send_telegram_notification(message)

            elif side == 'sell':
                for pos_id, pos in list(self.active_positions.items()):
                    if pos['symbol'] == order['symbol']:
                        profit = (float(order['price']) - pos['entry_price']) * float(order['filled'])

                        if redis_client.is_connected():
                            redis_client.hdel('active_positions', pos_id)

                        del self.active_positions[pos_id]

                        trade_log = {"profit": profit, "exit_price": float(order['price']), "timestamp": entry_time_str}
                        if redis_client.is_connected():
                             redis_client.client.rpush('trade_history', json.dumps(trade_log))

                        message = f"🔴 Position Closed:\nSymbol: {order['symbol']}\nProfit: {profit:.2f} USDT"
                        send_telegram_notification(message)
                        self.performance.update_trade(profit, won=profit > 0)
                        break

            logging.info(f"Successfully tracked {side} position")
            return position_info
        except Exception as e:
            logging.error(f"Error tracking position: {str(e)}")
            return None

    def check_existing_positions(self):
        """Check all active positions for exit conditions."""
        logging.info(f"Checking {len(self.active_positions)} active positions...")
        self.active_positions = self.load_positions()
        for pos_id, position in list(self.active_positions.items()):
            try:
                symbol = position['symbol']
                current_price = self.fetch_current_price(symbol)
                if not current_price:
                    logging.warning(f"Could not fetch price for {symbol}, skipping check.")
                    continue

                if position.get('take_profit') and current_price >= position['take_profit']:
                    logging.info(f"Take profit for {symbol} triggered at {current_price}")
                    self.execute_trade("sell", position['amount'], symbol)
                    continue

                if position.get('stop_loss') and current_price <= position['stop_loss']:
                    logging.info(f"Stop loss for {symbol} triggered at {current_price}")
                    self.execute_trade("sell", position['amount'], symbol)
                    continue
            except Exception as e:
                logging.error(f"Error checking position {pos_id} ({position.get('symbol')}): {e}")

    def execute_trade(self, side, amount, symbol):
        """Executes a trade for a given symbol."""
        logging.info(f"Attempting to execute {side} order for {amount} of {symbol}")
        try:
            if side == "buy":
                order = self.exchange.create_market_buy_order(symbol, amount)
            elif side == "sell":
                order = self.exchange.create_market_sell_order(symbol, amount)
            else:
                logging.error(f"Invalid trade side: {side}")
                return None

            logging.info(f"Successfully executed {side} order for {symbol}: {order}")
            self.track_position(order, side)
            return order
        except Exception as e:
            logging.error(f"Failed to execute {side} order for {symbol}: {e}")
            self.send_notification(f"⚠️ Trade Execution Failed for {symbol}: {e}")
            return None

    def fetch_current_price(self, symbol):
        """Fetch current price for a given symbol."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logging.error(f"Error fetching current price for {symbol}: {e}")
            return None

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
            'daily_trades': {},
            'daily_loss': {},
            'trade_history': [],
            'last_reset_date': datetime.now().strftime('%Y-%m-%d')
        }
        if save and redis_client.is_connected():
            self.save_metrics(metrics)
        return metrics

    def save_metrics(self, metrics_data=None):
        """Save metrics to Redis."""
        if not redis_client.is_connected(): return
        data_to_save = metrics_data if metrics_data is not None else self.metrics
        redis_client.set_json(self.metrics_key, data_to_save)

    def update_trade(self, profit, won=False):
        """Update metrics after a trade."""
        today_str = datetime.now().strftime('%Y-%m-%d')
        if self.metrics.get('last_reset_date') != today_str:
            self.metrics['daily_trades'] = {}
            self.metrics['daily_loss'] = {}
            self.metrics['last_reset_date'] = today_str

        self.metrics['total_trades'] += 1
        self.metrics['daily_trades'][today_str] = self.metrics['daily_trades'].get(today_str, 0) + 1
        if won: self.metrics['winning_trades'] += 1
        self.metrics['total_profit'] += profit
        if profit < 0: self.metrics['daily_loss'][today_str] = self.metrics['daily_loss'].get(today_str, 0) + abs(profit)
        self.metrics['trade_history'].append({'timestamp': datetime.now().isoformat(), 'profit': profit, 'won': won})
        self.calculate_metrics()
        self.save_metrics()

    def calculate_metrics(self):
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = (self.metrics['winning_trades'] / self.metrics['total_trades']) * 100
            profits = [trade['profit'] for trade in self.metrics['trade_history']]
            self.metrics['max_drawdown'] = self.calculate_max_drawdown(profits)

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

def run_trading_cycle():
    """Main function to run one cycle of the trading bot."""
    logging.info("---------- Starting new trading cycle ----------")
    if not redis_client.is_connected():
        logging.critical("Redis is not connected. Aborting trading cycle.")
        return

    performance = PerformanceMetrics()
    if not performance.can_trade():
        logging.warning("Trading is currently suspended due to performance limits.")
        return

    exchange = initialize_exchange()
    if not exchange:
        logging.critical("Could not initialize exchange. Aborting trading cycle.")
        return

    trade_execution = TradeExecution(exchange, performance)

    # 1. Manage existing positions
    trade_execution.check_existing_positions()

    # 2. Get new trading signals from AI
    signals = get_ai_signals()
    if not signals:
        logging.info("No new trading signals from AI.")
        logging.info("---------- Trading cycle finished ----------")
        return

    # 3. Process signals
    for signal in signals:
        asset = signal.get('asset')
        action = signal.get('signal')

        if not asset or not action:
            logging.warning(f"Skipping invalid signal: {signal}")
            continue

        logging.info(f"Processing signal: {action} {asset}")

        if action.upper() == 'BUY':
            amount_to_trade = CONFIG['trading'].get('min_trade_amount', 0.001)
            trade_execution.execute_trade('buy', amount_to_trade, asset)

        elif action.upper() == 'SELL':
            position_to_sell = None
            for pos_id, pos in trade_execution.active_positions.items():
                if pos['symbol'] == asset:
                    position_to_sell = pos
                    break

            if position_to_sell:
                logging.info(f"Found active position for {asset}. Closing it.")
                trade_execution.execute_trade('sell', position_to_sell['amount'], asset)
            else:
                logging.info(f"Received SELL signal for {asset}, but no active position found.")

    logging.info("---------- Trading cycle finished ----------")
