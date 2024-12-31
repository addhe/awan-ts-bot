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

def check_for_config_updates(last_checked_time):
    logging.info("Configuration updates are managed through code changes.")
    return False, last_checked_time

def handle_exit_signal(signal_number, frame):
    logging.info("Received exit signal, shutting down gracefully...")
    # Perform any cleanup operations here
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_exit_signal)
signal.signal(signal.SIGINT, handle_exit_signal)

class TradeExecution:
    def __init__(self, exchange, performance, trade_history):
        self.exchange = exchange
        self.performance = performance
        self.trade_history = trade_history
        self.market_data = None

    def check_spread(self, market_data):
        """Check if spread is within acceptable range"""
        try:
            if 'ask' not in market_data or 'bid' not in market_data:
                logging.warning("Ask/Bid data not available, skipping spread check")
                return True

            spread = (market_data['ask'].iloc[-1] - market_data['bid'].iloc[-1]) / market_data['bid'].iloc[-1]
            logging.info(f"Current spread: {spread:.4%}")

            if spread > CONFIG['max_spread_percent'] / 100:
                logging.warning(f"Spread too high: {spread:.4%}")
                return False

            return True
        except Exception as e:
            logging.error(f"Error checking spread: {str(e)}")
            return True

    def calculate_optimal_position_size(self, balance, current_price):
        """Calculate optimal position size based on risk management"""
        try:
            # Calculate base position size from account risk
            max_risk_amount = balance * (CONFIG['risk_percentage'] / 100)
            position_size = max_risk_amount / current_price

            # Apply volatility adjustment
            adjusted_position = self.manage_position_size(position_size)

            # Apply maximum position limit
            max_position = balance * CONFIG['max_position_size'] / current_price
            final_position = min(adjusted_position, max_position)

            logging.info(f"Calculated position sizes:")
            logging.info(f"Base position: {position_size:.4f}")
            logging.info(f"Volatility adjusted: {adjusted_position:.4f}")
            logging.info(f"Final position: {final_position:.4f}")

            return final_position
        except Exception as e:
            logging.error(f"Error calculating optimal position size: {str(e)}")
            return None

    def fetch_market_data(self, symbol, timeframe):
        """Fetch market data method for class"""
        try:
            candles = safe_api_call(self.exchange.fetch_ohlcv, symbol, timeframe)
            if candles is None:
                return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Fetch current ticker for ask/bid
            ticker = safe_api_call(self.exchange.fetch_ticker, symbol)
            if ticker:
                df['ask'] = ticker['ask']
                df['bid'] = ticker['bid']

            return df
        except Exception as e:
            logging.error(f"Failed to fetch market data: {str(e)}")
            return None

    def check_profitability(self, historical_data, current_price):
        if historical_data is None or historical_data.empty:
            logging.error("Historical data is empty or None")
            return False

        # Calculate sell target taking into account the required profit margin
        latest_price = historical_data['close'].iloc[-1]
        target_price = latest_price * (1 + CONFIG['profit_target_percent'] / 100)

        # Ensure profitability after fees
        profit_margin = target_price - latest_price
        fee_estimate = current_price * CONFIG['fee_rate']

        logging.info(f"Current price: {current_price}, Target price for selling: {target_price}, Fee estimate: {fee_estimate}")

        # Subtract fees to see if profitability is meaningful
        net_profit = profit_margin - fee_estimate
        logging.info(f"Net profit after fee: {net_profit}")

        return net_profit > 0

    def analyze_historical_data(self, symbol, timeframe='1h', limit=100):
        try:
            candles = safe_api_call(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit)
            if candles is None:
                logging.error(f"Failed to fetch historical data for {symbol}")
                return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(f"Exception in fetching historical data: {str(e)}")
            return None

    def place_limit_sell_order(self, amount, price):
        """Place limit sell order"""
        try:
            order = safe_api_call(
                self.exchange.create_limit_sell_order,
                CONFIG['symbol'],
                amount,
                price
            )
            logging.info(f"Placed limit sell order: {order}")
            return order
        except Exception as e:
            logging.error(f"Failed to place limit sell order: {str(e)}")
            return None

    def get_original_buy_price(self, symbol, executed_qty):
        """Get weighted average buy price from trade history"""
        try:
            trades = self.trade_history.get(symbol, [])
            buy_trades = [t for t in trades if t['side'] == 'buy']

            if not buy_trades:
                return 0

            total_cost = 0
            total_qty = 0

            for trade in reversed(buy_trades):
                if total_qty >= executed_qty:
                    break

                qty = min(trade['amount'], executed_qty - total_qty)
                total_cost += qty * trade['price']
                total_qty += qty

            if total_qty == 0:
                return 0

            return total_cost / total_qty

        except Exception as e:
            logging.error(f"Error getting original buy price: {str(e)}")
            return 0

    def calculate_profit(self, order):
        """Calculate profit for the trade considering the fee."""
        try:
            if order is None:
                return 0

            executed_qty = float(order['filled'])
            avg_price = float(order['price'])
            symbol = order['symbol']

            # Fetch current price
            current_price = self.fetch_current_price(symbol)
            if current_price is None:
                return 0

            # Calculate profit based on side
            if order['side'] == 'buy':
                profit = (current_price - avg_price) * executed_qty
            elif order['side'] == 'sell':
                original_price = self.get_original_buy_price(symbol, executed_qty)
                profit = (avg_price - original_price) * executed_qty

            # Deduct fees
            fee_cost = executed_qty * avg_price * CONFIG['fee_rate']
            profit -= fee_cost

            return profit

        except Exception as e:
            logging.error(f"Error calculating profit: {str(e)}")
            return 0

    def fetch_current_price(self, symbol):
        """Fetch current price for symbol"""
        try:
            ticker = safe_api_call(self.exchange.fetch_ticker, symbol)
            return ticker['last']
        except Exception as e:
            logging.error(f"Error fetching current price for {symbol}: {str(e)}")
            return None

    def execute_trade(self, side, amount, symbol):
        """
        Execute trade with performance tracking and trade history recording

        Args:
            side (str): "buy" or "sell"
            amount (float): Amount to trade
            symbol (str): Trading pair symbol
        """
        try:
            # Validate market conditions first
            if not self.validate_market_conditions(self.market_data):
                logging.warning(f"Market conditions not met for {symbol}, skipping trade")
                return

            # Get current balance
            balance = safe_api_call(self.exchange.fetch_balance)
            base_currency = symbol.split('/')[0]
            base_balance = balance[base_currency]['free']

            # Validate balance for sell orders
            if side == "sell" and base_balance < amount:
                logging.warning(f"Insufficient balance for selling {base_currency}. Available: {base_balance}, Required: {amount}")
                return

            # Adjust position size based on volatility
            adjusted_amount = self.manage_position_size(amount)
            current_price = self.market_data['close'].iloc[-1]
            stop_loss = self.calculate_stop_loss(current_price)
            take_profit = current_price * (1 + CONFIG['profit_target_percent'] / 100)

            # Validate risk-reward
            if not self.validate_risk_reward(current_price, stop_loss, take_profit):
                logging.warning(f"Trade doesn't meet risk-reward criteria: {symbol}")
                return

            # Execute the order
            try:
                if side == "buy":
                    order = self.exchange.create_market_buy_order(symbol, adjusted_amount)
                else:
                    order = self.exchange.create_market_sell_order(symbol, adjusted_amount)

                order_info = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'side': side,
                    'amount': adjusted_amount,
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }

                # Update trade history
                if symbol not in self.trade_history:
                    self.trade_history[symbol] = []
                self.trade_history[symbol].append(order_info)

                # Calculate and update performance metrics
                profit = self.calculate_profit(order)
                won = profit > 0
                self.performance.update_trade(profit, won)

                # Log and notify
                logging.info(f"Executed {side} order: {order}")
                self.send_notification(
                    f"Executed {side} order:\n"
                    f"Symbol: {symbol}\n"
                    f"Amount: {adjusted_amount}\n"
                    f"Price: {current_price}\n"
                    f"Stop Loss: {stop_loss}\n"
                    f"Take Profit: {take_profit}"
                )

                # Implement trailing stop and take profits
                if side == "buy":
                    self.implement_trailing_stop(current_price, current_price, adjusted_amount)
                    self.implement_partial_take_profits(current_price, adjusted_amount)

            except Exception as e:
                error_msg = f"Failed to execute {side} order: {str(e)}"
                logging.error(error_msg)
                self.send_notification(error_msg)

        except Exception as e:
            error_msg = f"Error in execute_trade: {str(e)}"
            logging.error(error_msg)
            self.send_notification(error_msg)

    def send_notification(self, message):
        try:
            send_telegram_notification(message)
        except Exception as e:
            logging.error(f"Failed to send notification: {str(e)}")

    def check_for_sell_signal(self, symbol, current_price):
        eth_balance = safe_api_call(self.exchange.fetch_balance)[symbol.split('/')[0]]['free']
        if eth_balance <= 0:
            logging.warning(f"No {symbol.split('/')[0]} to sell.")
            return

        historical_data = self.analyze_historical_data(symbol)
        if self.check_profitability(historical_data, current_price):
            logging.info(f"Profit target met. Preparing to sell {eth_balance} {symbol.split('/')[0]}")
            self.execute_trade("sell", eth_balance, symbol)

    def process_trade_signals(self, market_data, symbol, amount_to_trade_formatted):
        ema_short_last, ema_short_prev = market_data['ema_short'].iloc[-1], market_data['ema_short'].iloc[-2]
        ema_long_last, ema_long_prev = market_data['ema_long'].iloc[-1], market_data['ema_long'].iloc[-2]

        if ema_short_prev < ema_long_prev and ema_short_last > ema_long_last:
            logging.info(f"Buy signal confirmed: EMA short {ema_short_last:.4f} over EMA long {ema_long_last:.4f}")
            self.execute_trade("buy", amount_to_trade_formatted, symbol)

    def validate_market_conditions(self, market_data):
        """Additional trade validation filters"""
        try:
            # Check spread first
            if not self.check_spread(market_data):
                logging.warning("Spread check failed")
                return False
            # Volume filter
            if market_data['volume'].iloc[-1] * market_data['close'].iloc[-1] < CONFIG['min_volume_usdt']:
                return False
            # ATR filter for volatility
            atr = self.calculate_atr(market_data, CONFIG['atr_period'])
            if atr > CONFIG['max_atr_threshold']:
                return False
            # VWAP filter
            vwap = self.calculate_vwap(market_data)
            if market_data['close'].iloc[-1] < vwap:
                return False
            # Funding rate check for market sentiment
            funding_rate = self.get_funding_rate()
            if abs(funding_rate) > CONFIG['funding_rate_threshold']:
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating trade conditions: {str(e)}")
            return False

    def calculate_vwap(self, market_data):
        """Calculate Volume Weighted Average Price"""
        try:
            v = market_data['volume'].values
            tp = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            return (tp * v).sum() / v.sum()
        except Exception as e:
            logging.error(f"Error calculating VWAP: {str(e)}")
            return None

    def calculate_atr(self, market_data, period):
        """Calculate Average True Range"""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(window=period).mean()

            return atr.iloc[-1]

        except Exception as e:
            logging.error(f"Error calculating ATR: {str(e)}")
            return float('inf')

    def get_funding_rate(self):
        """Get funding rate for the symbol"""
        try:
            funding_rate = safe_api_call(
                self.exchange.fetch_funding_rate,
                CONFIG['symbol']
            )
            return funding_rate['fundingRate'] if funding_rate else 0
        except Exception as e:
            logging.error(f"Error fetching funding rate: {str(e)}")
            return 0

    def manage_position_size(self, base_position_size):
        """Adjust position size based on market conditions"""
        try:
            volatility = self.calculate_volatility()

            if volatility > CONFIG['high_volatility_threshold']:
                return base_position_size * (1 - CONFIG['high_volatility_adjustment'])

            if volatility < CONFIG['low_volatility_threshold']:
                return base_position_size * (1 + CONFIG['low_volatility_adjustment'])

            return base_position_size

        except Exception as e:
            logging.error(f"Error managing position size: {str(e)}")
            return base_position_size

    def calculate_volatility(self, lookback_period=20):
        """Calculate current market volatility"""
        try:
            if self.market_data is None:
                return None

            returns = np.log(self.market_data['close'] / self.market_data['close'].shift(1))
            return returns.std() * np.sqrt(24) # Annualized volatility

        except Exception as e:
            logging.error(f"Error calculating volatility: {str(e)}")
            return None

    def validate_risk_reward(self, entry_price, stop_loss, take_profit):
        """Validate if trade meets minimum risk-reward ratio"""
        try:
            risk = entry_price - stop_loss
            reward = take_profit - entry_price

            if risk <= 0:
                return False

            rr_ratio = reward / risk
            if rr_ratio < CONFIG['min_risk_reward_ratio']:
                logging.warning(f"Risk-reward ratio {rr_ratio:.2f} below minimum {CONFIG['min_risk_reward_ratio']}")
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating risk-reward: {str(e)}")
            return False

    def implement_trailing_stop(self, entry_price, current_price, position_size):
        """Implement trailing stop logic"""
        try:
            initial_stop = entry_price * (1 - CONFIG['stop_loss_percent'] / 100)
            profit_threshold = entry_price * (1 + CONFIG['initial_profit_for_trailing_stop'])

            if current_price >= profit_threshold:
                trailing_stop = current_price * (1 - CONFIG['trailing_distance_pct'])
                if trailing_stop > initial_stop:
                    logging.info(f"Updating trailing stop to: {trailing_stop}")
                    return trailing_stop

            return initial_stop

        except Exception as e:
            logging.error(f"Error implementing trailing stop: {str(e)}")
            return initial_stop

    def implement_partial_take_profits(self, entry_price, position_size):
        """Implement scaled take profit orders"""
        try:
            # First take profit level
            tp1_size = position_size * CONFIG['partial_tp_1']
            tp1_price = entry_price * (1 + CONFIG['tp1_target'])

            # Second take profit level
            tp2_size = position_size * CONFIG['partial_tp_2']
            tp2_price = entry_price * (1 + CONFIG['tp2_target'])

            # Place take profit orders
            self.place_limit_sell_order(tp1_size, tp1_price)
            self.place_limit_sell_order(tp2_size, tp2_price)

            # Implement trailing stop for remaining position
            self.implement_trailing_stop(entry_price, entry_price, position_size)

        except Exception as e:
            logging.error(f"Error setting take profits: {str(e)}")

    def implement_stop_loss(self, symbol, amount):
        try:
            last_price = self.fetch_current_price(symbol)
            if last_price is None:
                return
            stop_loss_price = self.calculate_stop_loss(last_price)
            logging.info(f"Setting stop-loss order at {stop_loss_price}")
        except Exception as e:
            logging.error(f"Failed to set stop-loss: {str(e)}")

    def calculate_stop_loss(self, current_price):
        return current_price * (1 - CONFIG['stop_loss_percent'] / 100)

    @staticmethod
    def calculate_ema(df, period, column='close'):
        return df[column].ewm(span=period, adjust=False).mean

    @staticmethod
    def validate_ema_strategy(config):
        if config['ema_short_period'] >= config['ema_long_period']:
            raise ValueError("Short EMA period should be less than Long EMA period.")
        logging.info("EMA strategy validation passed")

    def validate_trade_conditions(self, market_data):
        """Additional trade validation filters"""
        try:
            # Volume filter
            if market_data['volume'].iloc[-1] * market_data['close'].iloc[-1] < CONFIG['min_volume_usdt']:
                return False

            # ATR filter for volatility
            atr = self.calculate_atr(market_data, CONFIG['atr_period'])
            if atr > CONFIG['max_atr_threshold']:
                return False

            # VWAP filter
            vwap = self.calculate_vwap(market_data)
            if market_data['close'].iloc[-1] < vwap:
                return False

            # Funding rate check for market sentiment
            funding_rate = self.get_funding_rate()
            if abs(funding_rate) > CONFIG['funding_rate_threshold']:
                return False

            return True

        except Exception as e:
            logging.error(f"Error validating trade conditions: {str(e)}")
            return False

class PerformanceMetrics:
    def __init__(self):
        self.metrics_file = 'performance_metrics_spot.json'
        self.load_metrics()

    def load_metrics(self):
        try:
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        except FileNotFoundError:
            self.initialize_metrics()

    def initialize_metrics(self):
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

def initialize_exchange():
    try:
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        if CONFIG.get('use_testnet', False):
            exchange.set_sandbox_mode(True)  # Enable testnet mode
        return exchange
    except ccxt.BaseError as e:
        error_message = f"Failed to initialize exchange: {str(e)}"
        logging.error(error_message)
        send_telegram_notification(error_message)
        return None

def adjust_trade_amount(amount_to_trade, latest_close_price, min_trade_amount, min_notional):
    decimals_allowed = 4
    amount_to_trade_formatted = round(amount_to_trade, decimals_allowed)
    notional_value = amount_to_trade_formatted * latest_close_price

    if amount_to_trade_formatted < min_trade_amount or notional_value < min_notional:
        logging.warning("Adjusted trade amount or notional value is below minimum thresholds, skipping trade.")
        return None

    return amount_to_trade_formatted

def safe_api_call(func, *args, **kwargs):
    retry_count = kwargs.pop('retry_count', 3)
    retry_delay = kwargs.pop('retry_delay', 5)
    exponential_backoff = kwargs.pop('exponential_backoff', True)

    last_error = None

    for attempt in range(retry_count):
        try:
            # Attempt the API call
            result = func(*args, **kwargs)

            # Validate the response
            if result is None:
                raise ValueError("API call returned None")

            # Log successful call after retries if it's not the first attempt
            if attempt > 0:
                logging.info(f"API call successful after {attempt + 1} attempts")

            return result

        except ccxt.NetworkError as e:
            last_error = e
            if attempt == retry_count - 1:
                logging.error(f"Network error persists after {retry_count} attempts: {str(e)}")
                break

            wait_time = retry_delay * (2 ** attempt if exponential_backoff else 1)
            logging.warning(f"Network error: {str(e)}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retry_count})")
            time.sleep(wait_time)

        except ccxt.RateLimitExceeded as e:
            last_error = e
            if attempt == retry_count - 1:
                logging.error(f"Rate limit exceeded after {retry_count} attempts: {str(e)}")
                break

            wait_time = 30 * (2 ** attempt if exponential_backoff else 1)
            logging.warning(f"Rate limit exceeded. Waiting {wait_time} seconds... (Attempt {attempt + 1}/{retry_count})")
            time.sleep(wait_time)

        except ccxt.ExchangeError as e:
            last_error = e
            if "insufficient balance" in str(e).lower():
                logging.error(f"Insufficient balance error: {str(e)}")
                raise  # Don't retry on balance errors

            if attempt == retry_count - 1:
                logging.error(f"Exchange error persists after {retry_count} attempts: {str(e)}")
                break

            wait_time = retry_delay * (2 ** attempt if exponential_backoff else 1)
            logging.warning(f"Exchange error: {str(e)}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retry_count})")
            time.sleep(wait_time)

        except ccxt.RequestTimeout as e:
            last_error = e
            if attempt == retry_count - 1:
                logging.error(f"Request timeout after {retry_count} attempts: {str(e)}")
                break

            wait_time = retry_delay * (2 ** attempt if exponential_backoff else 1)
            logging.warning(f"Request timeout: {str(e)}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retry_count})")
            time.sleep(wait_time)

        except Exception as e:
            last_error = e
            logging.critical(f"Unexpected error during API call [{func.__name__}]: {str(e)}")
            logging.critical(f"Stack trace: {traceback.format_exc()}")
            raise  # Don't retry on unexpected errors

    # If we've exhausted all retries, log the error and raise the last exception
    error_message = f"API call [{func.__name__}] failed after {retry_count} attempts. Last error: {str(last_error)}"
    logging.error(error_message)

    send_telegram_notification(f"Critical API Error: {error_message}")
    raise last_error

def validate_config():
    try:
        TradeExecution.validate_ema_strategy(CONFIG)

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

        if not (0 < CONFIG['risk_percentage'] <= 100):
            raise ValueError("Risk percentage must be between 0 and 100.")

        if not (0 < CONFIG['fee_rate'] < 1):
            raise ValueError("Fee rate must be a percentage less than 1.")

        if CONFIG['max_daily_loss_percent'] <= 0 or CONFIG['max_drawdown_percent'] <= 0:
            raise ValueError("Max daily loss and max drawdown must be positive numbers.")

        if CONFIG['ema_short_period'] <= 0 or CONFIG['ema_long_period'] <= 0:
            raise ValueError("EMA periods must be positive integers.")

        if CONFIG['max_spread_percent'] <= 0:
            raise ValueError("Max spread percent should be positive")

        if not (0 < CONFIG['max_position_size'] <= 1):
            raise ValueError("Max position size must be between 0 and 1")

        if CONFIG['max_daily_trades'] <= 0:
            raise ValueError("Max daily trades should be positive.")

        if CONFIG['stop_loss_percent'] < 0:
            raise ValueError("Stop loss percent should not be negative.")

        logging.info("Config validation passed")
        return True
    except Exception as e:
        logging.error(f"Config validation failed: {e}")
        return False


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

        logging.debug(f"Market data for {symbol}: {market}")

        min_amount = market['limits']['amount'].get('min')
        min_notional_value = None

        if 'info' in market and 'filters' in market['info']:
            for f in market['info']['filters']:
                if f['filterType'] == 'NOTIONAL':
                    min_notional_value = float(f['minNotional'])
                    break

        if min_amount is None:
            logging.error(f"Minimum amount not found for symbol: {symbol}")
        if min_notional_value is None:
            logging.error(f"NOTIONAL filter not found for symbol: {symbol}")

        return min_amount, min_notional_value
    except Exception as e:
        logging.error(f"Failed to fetch market info: {str(e)}")
        return None, None

def main(performance, trade_history):
    global CONFIG
    exchange = initialize_exchange()
    symbol_base = CONFIG['symbol'].split('/')[0]
    trade_execution = TradeExecution(exchange, performance, trade_history)

    if exchange is None:
        logging.error("Exchange initialization failed")
        return

    try:
        global last_checked_time
        config_update, last_checked_time = check_for_config_updates(last_checked_time)
        if config_update:
            check_for_config_updates()
            logging.info("Configuration updated, reloading...")

        if not validate_config():
            logging.error("Configuration validation failed")
            return

        if not performance.can_trade():
            message = "Trading limits reached"
            logging.info(message)
            send_telegram_notification(message)
            return

        balance = safe_api_call(exchange.fetch_balance)
        usdt_balance = balance['USDT']['free']

        if usdt_balance < CONFIG['min_balance']:
            error_message = f"Insufficient balance: {usdt_balance} USDT"
            logging.error(error_message)
            send_telegram_notification(error_message)
            return

        # Fetch and analyze market data
        market_data = trade_execution.fetch_market_data(CONFIG['symbol'], CONFIG['timeframe'])
        if market_data is None:
            error_message = f"Failed to retrieve market data for {CONFIG['symbol']}."
            logging.error(error_message)
            send_telegram_notification(error_message)
            return

        # Then update trade execution with market data
        trade_execution.market_data = market_data

        # Validate market conditions including spread
        if not trade_execution.validate_market_conditions(market_data):
            info_message = f"Market conditions not met for {CONFIG['symbol']}, skipping trading opportunity."
            logging.info(info_message)
            send_telegram_notification(info_message)
            return

        # Calculate amount to trade using optimal position sizing
        current_price = market_data['close'].iloc[-1]
        optimal_position = trade_execution.calculate_optimal_position_size(usdt_balance, current_price)

        if optimal_position is None:
            logging.error("Failed to calculate optimal position size")
            return

        # Apply fees
        estimated_fee = optimal_position * CONFIG['fee_rate']
        amount_to_trade = optimal_position - estimated_fee

        # Ensure minimal conditions align with final trading amount
        min_trade_amount, min_notional = get_min_trade_amount_and_notional(exchange, CONFIG['symbol'])
        amount_to_trade_formatted = adjust_trade_amount(amount_to_trade, current_price, min_trade_amount, min_notional)

        if amount_to_trade_formatted is None:
            return

        # Calculate EMAs
        market_data['ema_short'] = TradeExecution.calculate_ema(market_data, CONFIG['ema_short_period'])
        market_data['ema_long'] = TradeExecution.calculate_ema(market_data, CONFIG['ema_long_period'])

        latest_close_price = market_data['close'].iloc[-1]
        logging.info(f"Latest close price: {latest_close_price}")

        # Check for sell signals
        trade_execution.check_for_sell_signal(CONFIG['symbol'], latest_close_price)

        logging.info(f"Calculated optimal position: {optimal_position} {symbol_base}")
        logging.info(f"Estimated fee: {estimated_fee} {symbol_base}")
        logging.info(f"Amount to trade after fees: {amount_to_trade} {symbol_base}")
        logging.info(f"Formatted amount to trade: {amount_to_trade_formatted} {symbol_base}")

        # Process EMA based buy signal
        trade_execution.process_trade_signals(market_data, CONFIG['symbol'], amount_to_trade_formatted)

    except Exception as e:
        error_message = f'Critical error in main loop: {str(e)}'
        logging.error(error_message)
        send_telegram_notification(error_message)

if __name__ == '__main__':
    performance = PerformanceMetrics()
    trade_history = {}
    last_checked_time = 0
    while True:
        main(performance, trade_history)
        time.sleep(60)