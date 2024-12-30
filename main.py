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

from config import CONFIG
from src.modules.send_telegram_notification import send_telegram_notification

# Initialize logging with a rotating file handler
log_handler = RotatingFileHandler('trade_log_spot.log', maxBytes=5*1024*1024, backupCount=2)
logging.basicConfig(handlers=[log_handler], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# API Configuration
API_KEY = os.environ.get('API_KEY_BINANCE')
API_SECRET = os.environ.get('API_SECRET_BINANCE')

if API_KEY is None or API_SECRET is None:
    logging.error('API credentials not found in environment variables')
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
            
        logging.info("Config validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Config validation failed: {e}")
        return False

def main(performance):
    exchange = None
    try:
        if not validate_config():
            logging.error("Configuration validation failed")
            return

        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET
        })
        
        performance = PerformanceMetrics()

        if not performance.can_trade():
            logging.info("Trading limits reached")
            return

        balance = exchange.fetch_balance()
        if balance['USDT']['free'] < CONFIG['min_balance']:
            logging.error(f"Insufficient balance: {balance['USDT']['free']} USDT")
            return

        # Fetching and processing market data would go here
        
    except Exception as e:
        logging.error(f'Critical error in main loop: {str(e)}')

if __name__ == '__main__':
    performance = PerformanceMetrics()
    while True:
        main(performance)
        time.sleep(60)  # execute the loop every minute