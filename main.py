# Standard library imports
import json
import logging
import math
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import ccxt
import numpy as np
import pandas as pd

# Local imports
from config.config import CONFIG
from src.modules.send_telegram_notification import send_telegram_notification
from src.modules.utils.trade_history import TradeHistory
from src.modules.trade_execution.trade_executor import (
    TradeExecutor, TradeExecutionError, InsufficientBalanceError,
    MarketConditionError, OrderTimeoutError, ExchangeConnectionError,
    RateLimitError, StaleDataError, OrderValidationError, ExchangeAPIError
)
from src.modules.market_analysis.market_analyzer import MarketAnalyzer
from src.modules.market_analysis.market_validation import MarketValidation
from src.modules.market_analysis.trend_analysis import TrendAnalysis
from src.modules.market_analysis.indicators import Indicators

# Configure logging
def setup_logging() -> None:
    """Initialize logging configuration with rotating file handler."""
    log_handler = RotatingFileHandler(
        'trade_log_spot.log',
        maxBytes=5*1024*1024,
        backupCount=2
    )
    logging.basicConfig(
        handlers=[log_handler],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def initialize_api_credentials() -> Tuple[str, str]:
    """
    Initialize API credentials from environment variables.

    Returns:
        Tuple[str, str]: API key and secret

    Raises:
        SystemExit: If credentials are not found
    """
    api_key = os.environ.get('API_KEY_SPOT_BINANCE')
    api_secret = os.environ.get('API_SECRET_SPOT_BINANCE')

    if api_key is None or api_secret is None:
        error_message = 'API credentials not found in environment variables'
        logging.error(error_message)
        send_telegram_notification(error_message)
        sys.exit(1)

    return api_key, api_secret

# Setup logging and API credentials
setup_logging()
API_KEY, API_SECRET = initialize_api_credentials()

def check_for_config_updates(last_checked_time: float) -> Tuple[bool, float]:
    """Check for configuration updates."""
    logging.info("Configuration updates are managed through code changes.")
    return False, last_checked_time

def handle_exit_signal(signal_number: int, frame: Any) -> None:
    """Handle exit signals for graceful shutdown."""
    logging.info("Received exit signal, shutting down gracefully...")
    sys.exit(0)

# Setup signal handlers
signal.signal(signal.SIGTERM, handle_exit_signal)
signal.signal(signal.SIGINT, handle_exit_signal)

class SpotTradeManager:
    """Manages spot trading operations with enhanced error handling and monitoring."""

    def __init__(self, exchange: Any, performance: Any, trade_history: TradeHistory) -> None:
        """Initialize trading components with proper validation."""
        # Detailed validation of each parameter
        if exchange is None:
            raise ValueError("Exchange instance must be provided")
        if performance is None:
            raise ValueError("Performance metrics instance must be provided")
        if trade_history is None:
            raise ValueError("Trade history instance must be provided")

        # Log parameter states for debugging
        logging.debug(f"Initializing SpotTradeManager with:")
        logging.debug(f"Exchange type: {type(exchange)}")
        logging.debug(f"Performance type: {type(performance)}")
        logging.debug(f"Trade history type: {type(trade_history)}")
        logging.debug(f"Trade history content: {trade_history.history}")

        self.exchange = exchange
        self.performance = performance
        self.trade_history = trade_history.history
        self.market_data = None

        # Initialize trading components
        self.executor = TradeExecutor(exchange, performance, trade_history)
        self.market_analyzer = MarketAnalyzer(exchange)
        self.market_validator = MarketValidation(exchange)
        self.trend_analyzer = TrendAnalysis()
        self.indicators = Indicators()

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def handle_signal(signum: int, frame: Any) -> None:
            logging.info(f"Received signal {signum}, initiating graceful shutdown")
            self.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        logging.info("Signal handlers setup complete")

    def cleanup(self) -> None:
        """Cleanup resources before shutdown."""
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

    def report_balance_to_telegram(self) -> None:
        """Report current balances via Telegram."""
        try:
            balance = safe_api_call(self.exchange.fetch_balance)
            if balance is None:
                logging.error("Failed to fetch balance")
                return

            message = "Current Balances:\n"
            for currency in ['ETH', 'USDT']:
                if currency in balance and balance[currency]['total'] > 0:
                    total = balance[currency]['total']
                    free = balance[currency]['free']
                    used = balance[currency]['used']
                    message += f"{currency}:\n"
                    message += f"  Total: {total:.8f}\n"
                    message += f"  Free: {free:.8f}\n"
                    message += f"  In Use: {used:.8f}\n"

            send_telegram_notification(message)

        except Exception as e:
            logging.error(f"Error reporting balance to Telegram: {str(e)}")
            send_telegram_notification(f"Failed to report balance: {str(e)}")

class PerformanceMetrics:
    """Tracks and manages trading performance metrics."""

    def __init__(self) -> None:
        self.metrics_file = 'performance_metrics_spot.json'
        self.load_metrics()

    def load_metrics(self) -> None:
        """Load metrics from file or initialize if file doesn't exist."""
        try:
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        except FileNotFoundError:
            self.initialize_metrics()

    def initialize_metrics(self) -> None:
        """Initialize empty metrics structure."""
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

    def save_metrics(self) -> None:
        """Save current metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f)

    def update_trade(self, profit: float, won: bool = False) -> None:
        """Update metrics with new trade result."""
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

    def calculate_metrics(self) -> None:
        """Calculate derived metrics."""
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = (self.metrics['winning_trades'] / self.metrics['total_trades']) * 100
            profits = [trade['profit'] for trade in self.metrics['trade_history']]
            self.metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(profits)
            self.metrics['max_drawdown'] = self.calculate_max_drawdown(profits)

    @staticmethod
    def calculate_sharpe_ratio(profits: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from profit history."""
        if len(profits) < 2:
            return 0
        returns = pd.Series(profits)
        excess_returns = returns - (risk_free_rate / 252)
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    @staticmethod
    def calculate_max_drawdown(profits: List[float]) -> float:
        """Calculate maximum drawdown from profit history."""
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return np.max(drawdown) if len(drawdown) > 0 else 0

    def can_trade(self) -> bool:
        """Check if trading is allowed based on current metrics."""
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

def initialize_exchange() -> Optional[ccxt.Exchange]:
    """Initialize exchange connection with API credentials."""
    try:
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        if CONFIG.get('use_testnet', False):
            exchange.set_sandbox_mode(True)
        return exchange
    except ccxt.BaseError as e:
        error_message = f"Failed to initialize exchange: {str(e)}"
        logging.error(error_message)
        send_telegram_notification(error_message)
        return None

def adjust_trade_amount(amount_to_trade: float, latest_close_price: float,
                       min_trade_amount: float, min_notional: float) -> Optional[float]:
    """Adjust trade amount to meet minimum requirements."""
    try:
        decimals_allowed = 4
        amount_to_trade_formatted = round(amount_to_trade, decimals_allowed)
        notional_value = amount_to_trade_formatted * latest_close_price

        # If amount is below minimum, increase it to minimum
        if amount_to_trade_formatted < min_trade_amount:
            amount_to_trade_formatted = min_trade_amount
            notional_value = amount_to_trade_formatted * latest_close_price
            logging.info(f"Adjusted amount up to minimum: {amount_to_trade_formatted}")

        # If notional value is below minimum, adjust amount accordingly
        if notional_value < min_notional:
            amount_to_trade_formatted = math.ceil((min_notional / latest_close_price) * 10000) / 10000
            logging.info(f"Adjusted amount for minimum notional value: {amount_to_trade_formatted}")

        # Final validation
        final_notional = amount_to_trade_formatted * latest_close_price
        if amount_to_trade_formatted >= min_trade_amount and final_notional >= min_notional:
            logging.info(f"Final trade amount: {amount_to_trade_formatted} ({final_notional} USDT)")
            return amount_to_trade_formatted

        logging.warning(f"Could not meet minimum requirements: Amount={amount_to_trade_formatted}, Notional={final_notional}")
        return None

    except Exception as e:
        logging.error(f"Error in adjust_trade_amount: {str(e)}")
        return None

def safe_api_call(func: Any, *args: Any, **kwargs: Any) -> Optional[Any]:
    """
    Execute API calls with enhanced error handling and retry logic.

    Args:
        func: The API function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function. Special kwargs:
            - retry_count: Number of retry attempts (default: 3)
            - retry_delay: Initial delay between retries in seconds (default: 5)
            - exponential_backoff: Whether to use exponential backoff (default: True)

    Returns:
        The result of the API call if successful, None otherwise

    Raises:
        ccxt.ExchangeError: For critical exchange errors
        Exception: For unexpected errors
    """
    retry_count = kwargs.pop('retry_count', 3)
    retry_delay = kwargs.pop('retry_delay', 5)
    exponential_backoff = kwargs.pop('exponential_backoff', True)

    last_error = None
    func_name = getattr(func, '__name__', str(func))

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

def validate_config() -> bool:
    """Validate configuration settings."""
    try:
        # Validate EMA strategy
        if CONFIG['ema_short_period'] >= CONFIG['ema_long_period']:
            raise ValueError("Short EMA period should be less than Long EMA period.")
        logging.info("EMA strategy validation passed")

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

def get_min_trade_amount_and_notional(exchange: ccxt.Exchange, symbol: str) -> Tuple[Optional[float], Optional[float]]:
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

def perform_daily_operations(trade_manager: SpotTradeManager, current_day: datetime.date) -> None:
    """Perform daily operations like balance reporting."""
    global last_reported_day
    if last_reported_day is None or last_reported_day != current_day:
        trade_manager.report_balance_to_telegram()
        last_reported_day = current_day

def validate_trading_session(
    trade_manager: SpotTradeManager,
    performance: PerformanceMetrics
) -> bool:
    """Validate if trading session can proceed."""
    if not trade_manager.executor.can_trade_time_based():
        logging.info("Time-based trading restrictions in effect")
        return False

    global last_checked_time
    config_update, last_checked_time = check_for_config_updates(last_checked_time)
    if config_update or not validate_config():
        logging.error("Configuration validation failed")
        return False

    if not performance.can_trade():
        logging.info("Trading limits reached, skipping trading cycle")
        return False

    return True

def validate_balance(exchange: ccxt.Exchange) -> Optional[float]:
    """Validate balance and return USDT balance if sufficient."""
    balance = safe_api_call(exchange.fetch_balance)
    if balance is None:
        raise ExchangeAPIError("Failed to fetch balance")

    usdt_balance = balance['USDT']['free']
    if usdt_balance < CONFIG['min_balance']:
        logging.warning(f"Insufficient balance: {usdt_balance} USDT")
        return None

    return usdt_balance

def analyze_market_conditions(
    trade_manager: SpotTradeManager,
    market_data: pd.DataFrame,
    amount_to_trade: float
) -> Tuple[bool, Optional[dict], float]:
    """Analyze market conditions and return analysis results."""
    if not trade_manager.market_validator.validate_market_conditions(market_data):
        logging.info(f"Market conditions not met for {CONFIG['symbol']}")
        return False, None, 0

    current_price = market_data['close'].iloc[-1]

    if not trade_manager.market_validator.validate_entry_conditions(
        market_data, amount_to_trade
    ):
        return False, None, current_price

    analysis_result = trade_manager.market_analyzer.perform_technical_analysis(market_data)
    if analysis_result is None:
        raise MarketConditionError("Failed to perform technical analysis")

    return True, analysis_result, current_price

def execute_trading_cycle(
    trade_manager: SpotTradeManager,
    analysis_result: dict,
    amount_to_trade: float,
    current_price: float,
    optimal_position: float
) -> None:
    """Execute the trading cycle if conditions are met."""
    if trade_manager.executor.has_open_positions(CONFIG['symbol']):
        trade_manager.executor.manage_existing_positions(
            symbol=CONFIG['symbol'],
            current_price=current_price,
            market_data=trade_manager.market_data
        )

    if trade_manager.market_analyzer.should_execute_trade(
        analysis_result, trade_manager.market_data
    ):
        trade_manager.executor.execute_trade_with_safety(
            side="buy",
            amount=amount_to_trade,
            symbol=CONFIG['symbol'],
            current_price=current_price
        )

    trade_manager.executor.log_trading_metrics(
        symbol_base=CONFIG['symbol'].split('/')[0],
        optimal_position=optimal_position,
        amount_to_trade=amount_to_trade,
        current_price=current_price
    )

def main(performance: PerformanceMetrics, trade_history: TradeHistory) -> None:
    """Enhanced main trading loop with proper error handling and monitoring."""
    recovery_delay = 60  # seconds
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        trade_manager = None
        try:
            # Initialize exchange
            exchange = initialize_exchange()
            if exchange is None:
                logging.error("Failed to initialize exchange")
                return

            # Verify exchange connection
            try:
                exchange.fetch_ticker(CONFIG['symbol'])
            except Exception as e:
                logging.error(f"Failed to verify exchange connection: {str(e)}")
                return

            # Initialize trade manager with validated components
            try:
                logging.info("Attempting to initialize SpotTradeManager...")
                logging.info(f"Exchange initialized: {exchange is not None}")
                logging.info(f"Performance metrics initialized: {performance is not None}")
                logging.info(f"Trade history initialized: {trade_history is not None}")
                trade_manager = SpotTradeManager(exchange, performance.metrics, trade_history)
                logging.info("SpotTradeManager initialized successfully")
            except ValueError as e:
                logging.error(f"Failed to initialize trade manager: {str(e)}")
                logging.error(f"Exchange: {type(exchange) if exchange else 'None'}")
                logging.error(f"Performance: {type(performance) if performance else 'None'}")
                logging.error(f"Trade history: {type(trade_history) if trade_history else 'None'}")
                return

            # Verify trade manager connection
            if not trade_manager.executor.check_exchange_connection():
                raise ExchangeConnectionError("Exchange connection is not stable")

            # Setup and validation
            trade_manager.setup_signal_handlers()
            perform_daily_operations(trade_manager, datetime.now().date())

            if not validate_trading_session(trade_manager, performance):
                return

            usdt_balance = validate_balance(exchange)
            if usdt_balance is None:
                return

            # Market analysis
            market_data = trade_manager.executor.fetch_market_data(
                CONFIG['symbol'], CONFIG['timeframe']
            )
            if not trade_manager.market_validator.validate_market_data(market_data):
                raise MarketConditionError("Invalid market data structure")

            trade_manager.market_data = market_data
            trade_manager.market_analyzer.log_market_summary(market_data)

            # Position sizing
            position_sizing_result = trade_manager.executor.calculate_position_size(
                balance=usdt_balance,
                current_price=market_data['close'].iloc[-1],
                market_data=market_data
            )

            if position_sizing_result[0] is None or position_sizing_result[1] is None:
                raise OrderValidationError("Failed to calculate valid position size")

            optimal_position, amount_to_trade = position_sizing_result

            # Market analysis and execution
            conditions_met, analysis_result, current_price = analyze_market_conditions(
                trade_manager, market_data, amount_to_trade
            )

            if conditions_met and analysis_result:
                execute_trading_cycle(
                    trade_manager,
                    analysis_result,
                    amount_to_trade,
                    current_price,
                    optimal_position
                )

            return  # Successful execution

        except (ExchangeConnectionError, MarketConditionError,
                OrderValidationError, ExchangeAPIError) as e:
            retry_count += 1
            error_message = f'Error in main loop (attempt {retry_count}/{max_retries}): {str(e)}'
            logging.error(error_message)
            send_telegram_notification(error_message)

            if retry_count < max_retries:
                logging.info(f"Waiting {recovery_delay} seconds before retry...")
                time.sleep(recovery_delay)
                recovery_delay *= 2  # Exponential backoff
            else:
                logging.critical("Maximum retries reached, manual intervention required")
                send_telegram_notification("Trading bot stopped: Maximum retries reached")
                break

        except Exception as e:
            logging.critical(f"Unexpected error in main loop: {str(e)}")
            logging.critical(f"Stack trace: {traceback.format_exc()}")
            send_telegram_notification(f"Critical error: {str(e)}")
            break

        finally:
            if trade_manager is not None:
                try:
                    trade_manager.cleanup()
                except Exception as cleanup_error:
                    logging.error(f"Error during cleanup: {str(cleanup_error)}")
                    if isinstance(cleanup_error, ExchangeConnectionError):
                        send_telegram_notification(
                            "Warning: Cleanup failed due to exchange connection issues"
                        )

# Ensure this is declared outside the function to maintain its state across iterations
last_reported_day = None

if __name__ == '__main__':
    performance = PerformanceMetrics()
    trade_history = TradeHistory()
    last_checked_time = 0
    while True:
        main(performance, trade_history)
        time.sleep(60)
