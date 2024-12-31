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

    def can_trade_time_based(self):
        """Check time-based trading restrictions for 24/7 trading"""
        try:
            now = datetime.now()

            # Check minimum trade interval only
            if self.trade_history:
                last_trade = max(
                    (trade for trades in self.trade_history.values() for trade in trades),
                    key=lambda x: x['timestamp']
                )
                last_trade_time = datetime.fromisoformat(last_trade['timestamp'])
                time_since_last_trade = (now - last_trade_time).total_seconds()

                if time_since_last_trade < CONFIG['min_trade_interval']:
                    logging.debug(f"Minimum trade interval not met. Time since last trade: {time_since_last_trade}s")
                    return False

            # Check if we have reached daily trade limits
            today_trades = [
                trade for trades in self.trade_history.values()
                for trade in trades
                if trade['timestamp'].startswith(now.strftime('%Y-%m-%d'))
            ]

            if len(today_trades) >= CONFIG['max_daily_trades']:
                logging.info("Daily trade limit reached")
                return False

            # Check daily profit target
            if self.check_daily_profit_target():
                logging.info("Daily profit target reached")
                return False

            return True

        except Exception as e:
            logging.error(f"Error checking time-based restrictions: {str(e)}")
            return False

    def monitor_performance(self):
        """Monitor trading performance metrics"""
        try:
            # Calculate performance metrics
            win_rate = self.performance.metrics['winning_trades'] / self.performance.metrics['total_trades']
            profit_factor = abs(self.performance.metrics['total_profit']) / abs(self.performance.metrics['max_drawdown'])

            # Log performance metrics
            logging.info(f"""
            Performance Metrics:
            Win Rate: {win_rate:.2%}
            Profit Factor: {profit_factor:.2f}
            Total Trades: {self.performance.metrics['total_trades']}
            Max Drawdown: {self.performance.metrics['max_drawdown']:.2%}
            """)

            # Alert if metrics are below thresholds
            if win_rate < 0.4 or profit_factor < 1.5:
                send_telegram_notification("Warning: Performance metrics below threshold")

        except Exception as e:
            logging.error(f"Error monitoring performance: {str(e)}")

    def validate_entry_conditions(self, market_data, amount_to_trade):
        """Validate entry conditions before trade execution"""
        try:
            # Add current price validation
            if 'close' not in market_data.columns:
                logging.error("Missing close price data")
                return False

            current_price = market_data['close'].iloc[-1]
            if current_price <= 0:
                logging.error("Invalid current price")
                return False

            # Calculate average volume with error handling
            try:
                avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
            except Exception as e:
                logging.error(f"Error calculating average volume: {str(e)}")
                return False

            # Calculate market impact with safeguards
            try:
                market_impact = (amount_to_trade * current_price) / (avg_volume * current_price)
                logging.info(f"Market impact: {market_impact:.4%}")

                if market_impact > CONFIG['market_impact_threshold']:
                    logging.warning(f"Market impact too high: {market_impact:.4%}")
                    return False
            except ZeroDivisionError:
                logging.error("Zero average volume detected")
                return False

            # Enhanced liquidity check
            try:
                liquidity_ratio = amount_to_trade / avg_volume
                logging.info(f"Liquidity ratio: {liquidity_ratio:.4%}")

                if liquidity_ratio > CONFIG['min_liquidity_ratio']:
                    logging.warning(f"Order size too large compared to average volume. Ratio: {liquidity_ratio:.4%}")
                    return False
            except ZeroDivisionError:
                logging.error("Zero average volume in liquidity check")
                return False

            # Enhanced consecutive losses check with safeguards
            try:
                if CONFIG['symbol'] not in self.trade_history:
                    recent_trades = []
                else:
                    recent_trades = self.trade_history[CONFIG['symbol']][-CONFIG['max_consecutive_losses']:]

                consecutive_losses = sum(1 for trade in recent_trades if trade.get('profit', 0) < 0)

                if consecutive_losses >= CONFIG['max_consecutive_losses']:
                    logging.warning(f"Maximum consecutive losses reached: {consecutive_losses}")
                    return False

                logging.info(f"Current consecutive losses: {consecutive_losses}")
            except Exception as e:
                logging.error(f"Error checking consecutive losses: {str(e)}")
                return False

            # Additional validation checks
            if not self.validate_time_window():
                return False

            if not self.validate_market_volatility(market_data):
                return False

            logging.info("All entry conditions validated successfully")
            return True

        except Exception as e:
            logging.error(f"Error validating entry conditions: {str(e)}")
            return False

    def validate_time_window(self):
        """Validate if enough time has passed since last trade"""
        try:
            if not self.trade_history.get(CONFIG['symbol']):
                return True

            last_trade = self.trade_history[CONFIG['symbol']][-1]
            last_trade_time = datetime.fromisoformat(last_trade['timestamp'])
            time_since_last_trade = (datetime.now() - last_trade_time).total_seconds()

            if time_since_last_trade < CONFIG['min_trade_interval']:
                logging.info(f"Minimum trade interval not met. Time since last trade: {time_since_last_trade}s")
                return False

            return True
        except Exception as e:
            logging.error(f"Error validating time window: {str(e)}")
            return False

    def validate_market_volatility(self, market_data):
        """Validate if market volatility is within acceptable range"""
        try:
            # Calculate current volatility
            volatility = self.calculate_volatility(market_data)
            if volatility is None:
                return False

            logging.info(f"Current market volatility: {volatility:.4%}")

            # Check against thresholds
            if volatility > CONFIG['high_volatility_threshold']:
                logging.warning(f"Volatility too high: {volatility:.4%}")
                return False

            if volatility < CONFIG['low_volatility_threshold']:
                logging.warning(f"Volatility too low: {volatility:.4%}")
                return False

            return True
        except Exception as e:
            logging.error(f"Error validating market volatility: {str(e)}")
            return False

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            def handle_shutdown(signum, frame):
                logging.info(f"Received signal {signum}, initiating graceful shutdown")
                self.cleanup()
                sys.exit(0)

            # Remove global handlers
            signal.signal(signal.SIGTERM, handle_shutdown)
            signal.signal(signal.SIGINT, handle_shutdown)
            logging.info("Signal handlers setup completed")
        except Exception as e:
            logging.error(f"Error setting up signal handlers: {str(e)}")

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"Received signal {signum}, initiating graceful shutdown")
        self.cleanup()
        sys.exit(0)

    def has_open_positions(self, symbol):
        """Check if there are open positions for the symbol"""
        try:
            balance = safe_api_call(self.exchange.fetch_balance)
            if balance is None:
                return False

            base_currency = symbol.split('/')[0]
            position_size = balance[base_currency]['free']

            return position_size > 0
        except Exception as e:
            logging.error(f"Error checking open positions: {str(e)}")
            return False

    def check_daily_profit_target(self):
        """Check if daily profit target is reached"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            today_trades = [
                trade for trade in self.performance.metrics['trade_history']
                if trade['timestamp'].startswith(today)
            ]

            daily_profit = sum(trade['profit'] for trade in today_trades)
            daily_profit_percent = (daily_profit / self.performance.metrics['total_profit']) * 100

            if daily_profit_percent >= CONFIG['daily_profit_target']:
                logging.info(f"Daily profit target reached: {daily_profit_percent:.2f}%")
                return True

            return False

        except Exception as e:
            logging.error(f"Error checking daily profit target: {str(e)}")
            return False

    def check_technical_exit_signals(self, market_data):
        """Check technical indicators for exit signals"""
        try:
            # Get latest technical analysis
            analysis = self.perform_technical_analysis(market_data)
            if analysis is None:
                return False

            trend = analysis['trend_analysis']

            # Exit conditions
            exit_signals = (
                trend['rsi'] > CONFIG['rsi_overbought'] or
                trend['adx'] < CONFIG['adx_threshold'] or
                trend['momentum'] < 0 or
                trend['trend_strength'] < CONFIG['trend_strength_threshold']
            )

            return exit_signals

        except Exception as e:
            logging.error(f"Error checking technical exit signals: {str(e)}")
            return False

    def should_exit_position(self, position, current_price, market_data):
        """Determine if position should be exited"""
        try:
            # Stop loss hit
            if current_price <= position['current_stop']:
                logging.info("Stop loss triggered")
                return True

            # Take profit hit
            if current_price >= position['take_profit']:
                logging.info("Take profit target reached")
                return True

            # Technical exit signals
            if self.check_technical_exit_signals(market_data):
                logging.info("Technical exit signal triggered")
                return True

            return False

        except Exception as e:
            logging.error(f"Error checking exit conditions: {str(e)}")
            return False

    def manage_existing_positions(self, symbol, current_price, market_data):
        """Enhanced position management"""
        try:
            # Get position details
            position = self.get_position_entry(symbol)
            if position is None:
                return

            # Check position duration
            position_age = (datetime.now() - position['entry_time']).total_seconds()
            if position_age > CONFIG['position_max_duration']:
                logging.info("Position exceeded maximum duration, closing")
                self.execute_trade("sell", position['position_size'], symbol)
                return

            # Check update interval
            time_since_update = (datetime.now() - position['last_update']).total_seconds()
            if time_since_update < CONFIG['position_update_interval']:
                return

            # Update trailing stop
            new_stop = self.implement_trailing_stop(
                position['entry_price'],
                current_price,
                position['position_size']
            )

            # Update position info
            position['last_update'] = datetime.now()
            position['current_stop'] = new_stop

            # Check exit conditions
            if self.should_exit_position(position, current_price, market_data):
                self.execute_trade("sell", position['position_size'], symbol)

        except Exception as e:
            logging.error(f"Error managing positions: {str(e)}")

    def cleanup(self):
        """Cleanup resources before shutdown"""
        try:
            logging.info("Initiating cleanup process")

            # Cancel any pending orders
            open_orders = safe_api_call(self.exchange.fetch_open_orders, CONFIG['symbol'])
            if open_orders:
                for order in open_orders:
                    safe_api_call(self.exchange.cancel_order, order['id'], CONFIG['symbol'])
                    logging.info(f"Cancelled order {order['id']}")

            # Save performance metrics
            self.performance.save_metrics()

            # Close exchange connection if available
            if hasattr(self.exchange, 'close'):
                self.exchange.close()

            # Clear market data
            if hasattr(self, 'market_data'):
                del self.market_data

            # Clear large objects
            import gc
            gc.collect()

            logging.info("Cleanup completed successfully")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    def validate_trading_conditions(self, market_data):
        """Comprehensive trading conditions validation"""
        try:
            # Market health check
            if not self.check_market_health():
                logging.debug("Failed market health check")
                return False

            # Market conditions validation
            if not self.validate_market_conditions(market_data):
                logging.debug("Failed market conditions validation")
                return False

            # Spread check
            if not self.check_spread(market_data):
                logging.debug("Failed spread check")
                return False

            logging.debug("All trading conditions met")

            return True

        except Exception as e:
            logging.error(f"Error validating trading conditions: {str(e)}")
            return False

    def implement_risk_management(self, symbol, entry_price, position_size, order):
        """Enhanced risk management implementation"""
        try:
            # Basic setup
            stop_loss = self.calculate_stop_loss(entry_price)
            take_profit = entry_price * (1 + CONFIG['profit_target_percent'] / 100)

            # Order tracking
            order_id = order['id']
            entry_time = datetime.now()

            # Save position info
            position_info = {
                'order_id': order_id,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'last_update': entry_time
            }

            # Implement trailing stop
            trailing_stop = self.implement_trailing_stop(
                entry_price=entry_price,
                current_price=entry_price,
                position_size=position_size
            )

            # Check slippage
            actual_entry = float(order['price'])
            slippage = abs(actual_entry - entry_price) / entry_price
            if slippage > CONFIG['max_slippage']:
                logging.warning(f"High slippage detected: {slippage:.2%}")

            # Set take profits
            self.implement_partial_take_profits(entry_price, position_size)

            return position_info

        except Exception as e:
            logging.error(f"Error implementing risk management: {str(e)}")
            return None

    def get_position_entry(self, symbol):
        """Get entry details for current position"""
        try:
            trades = self.trade_history.get(symbol, [])
            if not trades:
                return None

            # Get most recent buy trade
            buy_trades = [t for t in trades if t['side'] == 'buy']
            if not buy_trades:
                return None

            return buy_trades[-1]

        except Exception as e:
            logging.error(f"Error getting position entry: {str(e)}")
            return None

    def validate_market_data(self, market_data):
        """Validate market data structure and content"""
        try:
            if market_data is None or market_data.empty:
                return False

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in market_data.columns for col in required_columns):
                logging.error("Missing required columns in market data")
                return False

            if len(market_data) < CONFIG['min_candles_required']:
                logging.error("Insufficient historical data")
                return False

            return True
        except Exception as e:
            logging.error(f"Error validating market data: {str(e)}")
            return False

    def log_trading_metrics(self, symbol_base, optimal_position, amount_to_trade, current_price):
        """Log comprehensive trading metrics"""
        try:
            logging.info("=== Trading Metrics ===")
            logging.info(f"Symbol: {symbol_base}")
            logging.info(f"Current Price: {current_price:.2f}")
            logging.info(f"Optimal Position: {optimal_position:.4f}")
            logging.info(f"Trade Amount: {amount_to_trade:.4f}")
            logging.info(f"Notional Value: {amount_to_trade * current_price:.2f} USDT")
            logging.info("===================")

        except Exception as e:
            logging.error(f"Error logging trading metrics: {str(e)}")

    def execute_trade_with_safety(self, side, amount, symbol, current_price):
        """Execute trade with additional safety checks"""
        try:
            # Check open orders count
            open_orders = safe_api_call(self.exchange.fetch_open_orders, symbol)
            if len(open_orders) >= CONFIG['max_open_orders']:
                logging.warning("Maximum open orders reached")
                return False

            # Check daily profit target
            if self.check_daily_profit_target():
                logging.info("Daily profit target reached, skipping trade")
                return False

            # Pre-trade validation
            if not self.validate_trading_conditions(self.market_data):
                return False

            # Execute trade
            order = self.execute_trade(side, amount, symbol)
            if order is None:
                return False

            # Post-trade actions
            self.implement_risk_management(
                symbol=symbol,
                entry_price=current_price,
                position_size=amount,
                order=order
            )

            # Monitor performance after trade
            self.monitor_performance()

            return True

        except Exception as e:
            logging.error(f"Error executing trade with safety: {str(e)}")
            return False

    def should_execute_trade(self, analysis_result, market_data):
        """Determine if trade should be executed based on analysis"""
        try:
            if analysis_result is None:
                return False

            # Check for buy conditions
            ema_short = analysis_result['ema_data']['ema_short']
            ema_long = analysis_result['ema_data']['ema_long']
            trend = analysis_result['trend_analysis']

            # Enhanced buy conditions
            buy_conditions = (
                ema_short.iloc[-2] < ema_long.iloc[-2] and  # Previous crossover
                ema_short.iloc[-1] > ema_long.iloc[-1] and  # Current crossover
                trend['rsi'] < CONFIG['rsi_overbought'] and
                trend['adx'] > CONFIG['adx_threshold'] and
                trend['momentum'] > CONFIG['momentum_threshold'] and
                trend['trend_strength'] > CONFIG['trend_strength_threshold']
            )

            return buy_conditions

        except Exception as e:
            logging.error(f"Error checking trade conditions: {str(e)}")
            return False

    def perform_technical_analysis(self, market_data):
        """Perform comprehensive technical analysis"""
        try:
            # Calculate EMAs
            market_data['ema_short'] = self.calculate_ema(market_data, CONFIG['ema_short_period'])
            market_data['ema_long'] = self.calculate_ema(market_data, CONFIG['ema_long_period'])

            # Get trend analysis
            trend_analysis = self.analyze_price_trend(market_data)

            if None in [market_data['ema_short'], market_data['ema_long'], trend_analysis]:
                return None

            return {
                'ema_data': market_data[['ema_short', 'ema_long']],
                'trend_analysis': trend_analysis
            }

        except Exception as e:
            logging.error(f"Error performing technical analysis: {str(e)}")
            return None

    def calculate_position_size(self, balance, current_price, market_data):
        """Calculate and validate position size"""
        try:
            # Calculate optimal position
            optimal_position = self.calculate_optimal_position_size(balance, current_price)
            if optimal_position is None:
                return None

            # Apply fees
            estimated_fee = optimal_position * CONFIG['fee_rate']
            amount_to_trade = optimal_position - estimated_fee

            # Get minimum trade requirements
            min_trade_amount, min_notional = get_min_trade_amount_and_notional(
                self.exchange,
                CONFIG['symbol']
            )

            # Format and validate final amount
            amount_to_trade_formatted = adjust_trade_amount(
                amount_to_trade,
                current_price,
                min_trade_amount,
                min_notional
            )

            if amount_to_trade_formatted is None:
                return None

            # Validate position size
            if not self.validate_position_size(amount_to_trade_formatted, current_price):
                return None

            return optimal_position, amount_to_trade_formatted

        except Exception as e:
            logging.error(f"Error calculating position size: {str(e)}")
            return None

    def validate_position_size(self, amount, price):
        try:
            notional_value = amount * price
            min_notional = CONFIG.get('min_notional_value', 10)  # Add to config
            max_notional = CONFIG.get('max_notional_value', 1000)  # Add to config

            if notional_value < min_notional:
                logging.warning(f"Position size too small: {notional_value} USDT")
                return False

            if notional_value > max_notional:
                logging.warning(f"Position size too large: {notional_value} USDT")
                return False

            return True
        except Exception as e:
            logging.error(f"Error validating position size: {str(e)}")
            return False

    def check_market_health(self):
        try:
            # Check 24h volume
            ticker = safe_api_call(self.exchange.fetch_ticker, CONFIG['symbol'])
            if ticker['quoteVolume'] < CONFIG['min_volume_usdt'] * 24:
                logging.warning("24h volume too low")
                return False

            # Check if market is trending or ranging
            atr = self.calculate_atr(self.market_data, CONFIG['atr_period'])
            if atr is None or atr == float('inf'):
                return False

            return True
        except Exception as e:
            logging.error(f"Error checking market health: {str(e)}")
            return False

    def check_spread(self, market_data):
        try:
            if 'ask' not in market_data or 'bid' not in market_data:
                logging.warning("Ask/Bid data not available, skipping spread check")
                return True

            spread = (market_data['ask'].iloc[-1] - market_data['bid'].iloc[-1]) / market_data['bid'].iloc[-1]
            logging.info(f"Current spread: {spread:.4%}")

            # Add monitoring for unusual spreads
            if spread < 0:
                logging.warning(f"Negative spread detected: {spread:.4%}")
                return False

            if spread > CONFIG['max_spread_percent'] / 100:
                logging.warning(f"Spread too high: {spread:.4%}")
                return False

            # Log when spread is approaching maximum
            if spread > (CONFIG['max_spread_percent'] / 100) * 0.8:
                logging.warning(f"Spread approaching maximum threshold: {spread:.4%}")

            return True
        except Exception as e:
            logging.error(f"Error checking spread: {str(e)}")
            return False  # Changed to return False on error

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
        Execute trade with enhanced error handling and order tracking

        Args:
            side (str): "buy" or "sell"
            amount (float): Amount to trade
            symbol (str): Trading pair symbol
        """
        try:
            start_time = time.time()

            # Validate market conditions first
            if not self.validate_market_conditions(self.market_data):
                logging.warning(f"Market conditions not met for {symbol}, skipping trade")
                return None

            # Get current balance
            balance = safe_api_call(self.exchange.fetch_balance)
            base_currency = symbol.split('/')[0]
            base_balance = balance[base_currency]['free']

            # Validate balance for sell orders
            if side == "sell" and base_balance < amount:
                logging.warning(f"Insufficient balance for selling {base_currency}. Available: {base_balance}, Required: {amount}")
                return None

            # Execute order with timeout check
            while time.time() - start_time < CONFIG['order_timeout']:
                try:
                    # Place the order
                    if side == "buy":
                        order = self.exchange.create_market_buy_order(symbol, amount)
                    else:
                        order = self.exchange.create_market_sell_order(symbol, amount)

                    # Check if order is filled
                    if order['status'] == 'closed':
                        # Update trade history
                        order_info = {
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'side': side,
                            'amount': amount,
                            'price': float(order['price']),
                            'order_id': order['id']
                        }

                        if symbol not in self.trade_history:
                            self.trade_history[symbol] = []
                        self.trade_history[symbol].append(order_info)

                        # Log and notify
                        logging.info(f"Executed {side} order: {order}")
                        self.send_notification(
                            f"Executed {side} order:\n"
                            f"Symbol: {symbol}\n"
                            f"Amount: {amount}\n"
                            f"Price: {order['price']}"
                        )

                        return order

                    time.sleep(1)  # Wait before checking again

                except Exception as e:
                    logging.error(f"Order execution error: {str(e)}")
                    return None

            logging.error(f"Order timeout after {CONFIG['order_timeout']} seconds")
            return None

        except Exception as e:
            logging.error(f"Error in execute_trade: {str(e)}")
            return None

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
        """Process trading signals with enhanced trend analysis"""
        try:
            # Validate inputs
            if market_data is None or market_data.empty:
                logging.warning("Empty market data in process_trade_signals")
                return

            if 'ema_short' not in market_data.columns or 'ema_long' not in market_data.columns:
                logging.warning("EMA columns not found in market data")
                return

            if amount_to_trade_formatted is None or amount_to_trade_formatted <= 0:
                logging.warning("Invalid trade amount")
                return

            # Get EMA signals
            ema_short_last, ema_short_prev = market_data['ema_short'].iloc[-1], market_data['ema_short'].iloc[-2]
            ema_long_last, ema_long_prev = market_data['ema_long'].iloc[-1], market_data['ema_long'].iloc[-2]

            # Get trend analysis
            trend_analysis = self.analyze_price_trend(market_data)

            if trend_analysis is None:
                logging.warning("Could not analyze price trend, skipping trade signal")
                return

            # Log trend analysis results
            logging.info(f"Trend Analysis - RSI: {trend_analysis['rsi']:.2f}, "
                        f"Momentum: {trend_analysis['momentum']:.4f}, "
                        f"ADX: {trend_analysis['adx']:.2f}, "
                        f"Trend Strength: {trend_analysis['trend_strength']:.2f}")

            # Enhanced buy conditions
            buy_conditions = (
                # EMA crossover
                ema_short_prev < ema_long_prev and ema_short_last > ema_long_last
                # RSI not overbought
                and trend_analysis['rsi'] < 70
                # Strong trend
                and trend_analysis['adx'] > 25
                # Positive momentum
                and trend_analysis['momentum'] > 0
                # Strong upward trend
                and trend_analysis['trend_strength'] > 0
            )

            if buy_conditions:
                logging.info(f"Buy signal confirmed:")
                logging.info(f"- EMA cross: Short {ema_short_last:.4f} over Long {ema_long_last:.4f}")
                logging.info(f"- RSI: {trend_analysis['rsi']:.2f}")
                logging.info(f"- ADX: {trend_analysis['adx']:.2f}")
                logging.info(f"- Momentum: {trend_analysis['momentum']:.4f}")

                # Execute the trade
                self.execute_trade("buy", amount_to_trade_formatted, symbol)
            else:
                logging.debug("Buy conditions not met:")
                if ema_short_prev >= ema_long_prev or ema_short_last <= ema_long_last:
                    logging.debug("- EMA crossover condition not met")
                if trend_analysis['rsi'] >= 70:
                    logging.debug("- RSI overbought")
                if trend_analysis['adx'] <= 25:
                    logging.debug("- Weak trend (ADX)")
                if trend_analysis['momentum'] <= 0:
                    logging.debug("- Negative momentum")
                if trend_analysis['trend_strength'] <= 0:
                    logging.debug("- Weak upward trend")

        except Exception as e:
            logging.error(f"Error in process_trade_signals: {str(e)}")
            logging.error(traceback.format_exc())

    def validate_market_conditions(self, market_data):
        """Enhanced market conditions validation"""
        try:
            # Add detailed market data logging
            logging.debug(f"""
            Market Conditions:
            Price: {market_data['close'].iloc[-1]}
            Volume: {market_data['volume'].iloc[-1]}
            EMA Short: {market_data['ema_short'].iloc[-1] if 'ema_short' in market_data else 'N/A'}
            EMA Long: {market_data['ema_long'].iloc[-1] if 'ema_long' in market_data else 'N/A'}
            RSI: {self.analyze_price_trend(market_data)['rsi'] if market_data is not None else 'N/A'}
            """)

            # Market health check
            if not self.check_market_health():
                logging.debug("Failed market health check")
                return False

            # 1. Check spread
            if not self.check_spread(market_data):
                logging.debug("Failed spread check")
                return False

            # 2. Enhanced volume analysis
            current_volume = market_data['volume'].iloc[-1] * market_data['close'].iloc[-1]
            volume_ma = market_data['volume'].rolling(window=CONFIG['volume_ma_period']).mean().iloc[-1]

            if current_volume < volume_ma * CONFIG['min_volume_multiplier']:
                logging.debug(f"Volume too low: {current_volume:.2f} < {volume_ma * CONFIG['min_volume_multiplier']:.2f}")
                return False

            # 3. Price movement check
            price_change = abs(market_data['close'].pct_change().iloc[-1])
            if price_change > CONFIG['price_change_threshold']:
                logging.debug(f"Price change too high: {price_change:.4%}")
                return False

            # 4. Trend strength analysis
            ema_short = self.calculate_ema(market_data, CONFIG['ema_short_period'])
            ema_long = self.calculate_ema(market_data, CONFIG['ema_long_period'])
            trend_strength = abs(ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1]

            if trend_strength < CONFIG['trend_strength_threshold']:
                logging.debug(f"Trend strength too weak: {trend_strength:.4f}")
                return False

            # 5. Volatility check
            atr = self.calculate_atr(market_data, CONFIG['atr_period'])
            current_price = market_data['close'].iloc[-1]
            atr_percentage = atr / current_price

            if atr_percentage > CONFIG['max_atr_threshold']:
                logging.debug(f"ATR too high: {atr_percentage:.4f}")
                return False

            # 6. VWAP analysis
            vwap = self.calculate_vwap(market_data)
            if abs(current_price - vwap) / vwap > CONFIG['max_spread_percent'] / 100:
                logging.debug(f"Price too far from VWAP: {abs(current_price - vwap) / vwap:.4%}")
                return False

            # Add detailed logging for successful conditions
            logging.info("Market conditions summary:")
            logging.info(f"Volume: {current_volume:.2f} USDT")
            logging.info(f"Price change: {price_change:.4%}")
            logging.info(f"Trend strength: {trend_strength:.4f}")
            logging.info(f"ATR percentage: {atr_percentage:.4f}")
            logging.info(f"VWAP distance: {abs(current_price - vwap) / vwap:.4%}")

            return True

        except Exception as e:
            logging.error(f"Error validating market conditions: {str(e)}")
            return False

    def analyze_price_trend(self, market_data, lookback_period=20):
        """Analyze price trend using multiple indicators"""
        try:
            close_prices = market_data['close']

            # Calculate RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Calculate price momentum
            momentum = close_prices.pct_change(periods=lookback_period)

            # Calculate average directional index (ADX)
            high_prices = market_data['high']
            low_prices = market_data['low']

            plus_dm = high_prices.diff()
            minus_dm = low_prices.diff()
            plus_dm = plus_dm.where(plus_dm > 0, 0)
            minus_dm = minus_dm.where(minus_dm > 0, 0)

            tr = pd.DataFrame({
                'hl': high_prices - low_prices,
                'hc': abs(high_prices - close_prices.shift(1)),
                'lc': abs(low_prices - close_prices.shift(1))
            }).max(axis=1)

            smoothing = 14
            plus_di = 100 * (plus_dm.rolling(smoothing).mean() / tr.rolling(smoothing).mean())
            minus_di = 100 * (minus_dm.rolling(smoothing).mean() / tr.rolling(smoothing).mean())
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(smoothing).mean()

            return {
                'rsi': rsi.iloc[-1],
                'momentum': momentum.iloc[-1],
                'adx': adx.iloc[-1],
                'trend_strength': (plus_di.iloc[-1] - minus_di.iloc[-1])
            }

        except Exception as e:
            logging.error(f"Error in trend analysis: {str(e)}")
            return None

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
        try:
            return df[column].ewm(span=period, adjust=False).mean()
        except Exception as e:
            logging.error(f"Error calculating EMA: {str(e)}")
            return None

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

        if CONFIG['max_consecutive_losses'] <= 0:
            raise ValueError("max_consecutive_losses must be positive")

        if CONFIG['daily_profit_target'] <= 0:
            raise ValueError("daily_profit_target must be positive")

        if CONFIG['market_impact_threshold'] <= 0:
            raise ValueError("market_impact_threshold must be positive")

        if CONFIG['position_sizing_atr_multiplier'] <= 0:
            raise ValueError("position_sizing_atr_multiplier must be positive")

        if CONFIG['max_open_orders'] <= 0:
            raise ValueError("max_open_orders must be positive")

        if not (0 < CONFIG['min_liquidity_ratio'] <= 1):
            raise ValueError("min_liquidity_ratio must be between 0 and 1")

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
    recovery_delay = 60  # seconds
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Initialize exchange and trade execution
            exchange = initialize_exchange()
            if exchange is None:
                raise ValueError("Exchange initialization failed")

            symbol_base = CONFIG['symbol'].split('/')[0]
            trade_execution = TradeExecution(exchange, performance, trade_history)

            # Add time-based check
            if not trade_execution.can_trade_time_based():
                logging.info("Time-based trading restrictions in effect")
                return

            # Setup proper signal handling
            trade_execution.setup_signal_handlers()

            # Configuration validation
            global last_checked_time
            config_update, last_checked_time = check_for_config_updates(last_checked_time)
            if config_update or not validate_config():
                logging.error("Configuration validation failed")
                return

            # Performance checks
            if not performance.can_trade():
                logging.info("Trading limits reached, skipping trading cycle")
                return

            # Balance validation
            balance = safe_api_call(exchange.fetch_balance)
            if balance is None:
                raise ValueError("Failed to fetch balance")

            usdt_balance = balance['USDT']['free']
            if usdt_balance < CONFIG['min_balance']:
                logging.warning(f"Insufficient balance: {usdt_balance} USDT")
                return

            # Market data fetching and validation
            market_data = trade_execution.fetch_market_data(CONFIG['symbol'], CONFIG['timeframe'])
            if not trade_execution.validate_market_data(market_data):
                logging.error("Invalid market data structure")
                return

            trade_execution.market_data = market_data

            # Comprehensive trading conditions validation
            if not trade_execution.validate_trading_conditions(market_data):
                logging.info(f"Trading conditions not met for {CONFIG['symbol']}")
                return

            # Position sizing and validation
            current_price = market_data['close'].iloc[-1]
            position_sizing_result = trade_execution.calculate_position_size(
                balance=usdt_balance,
                current_price=current_price,
                market_data=market_data
            )

            if position_sizing_result is None:
                logging.warning("Failed to calculate valid position size")
                return

            optimal_position, amount_to_trade_formatted = position_sizing_result
            if trade_execution.validate_entry_conditions(market_data, amount_to_trade_formatted):

                # Technical analysis
                analysis_result = trade_execution.perform_technical_analysis(market_data)
                if analysis_result is None:
                    logging.error("Failed to perform technical analysis")
                    return

                # Check existing positions and manage them
                if trade_execution.has_open_positions(CONFIG['symbol']):
                    trade_execution.manage_existing_positions(
                        symbol=CONFIG['symbol'],
                        current_price=current_price,
                        market_data=market_data
                    )

                # Process trading signals
                if trade_execution.should_execute_trade(analysis_result, market_data):
                    trade_execution.execute_trade_with_safety(
                        side="buy",
                        amount=amount_to_trade_formatted,
                        symbol=CONFIG['symbol'],
                        current_price=current_price
                    )

                # Log trading metrics
                trade_execution.log_trading_metrics(
                    symbol_base=symbol_base,
                    optimal_position=optimal_position,
                    amount_to_trade=amount_to_trade_formatted,
                    current_price=current_price
                )

        except Exception as e:
            retry_count += 1
            error_message = f'Error in main loop (attempt {retry_count}/{max_retries}): {str(e)}'
            logging.error(error_message)
            send_telegram_notification(error_message)

            if retry_count < max_retries:
                logging.info(f"Waiting {recovery_delay} seconds before retry...")
                time.sleep(recovery_delay)
                recovery_delay *= 2  # Exponential backoff
        finally:
            # Enhanced cleanup
            try:
                trade_execution.cleanup()
            except Exception as cleanup_error:
                logging.error(f"Error during cleanup: {str(cleanup_error)}")

if __name__ == '__main__':
    performance = PerformanceMetrics()
    trade_history = {}
    last_checked_time = 0
    while True:
        main(performance, trade_history)
        time.sleep(60)