import logging
import time
from datetime import datetime
from typing import Optional, Dict, List, Any, Union, NoReturn
import pandas as pd
import numpy as np

from ..market_analysis.market_analyzer import MarketAnalyzer
from ..market_analysis.market_validation import MarketValidation
from ..risk_management.risk_manager import RiskManager
from ..utils.api_utils import safe_api_call
from ..utils.trade_utils import adjust_trade_amount
from ..send_telegram_notification import send_telegram_notification
from config.config import CONFIG

class TradeExecutionError(Exception):
    """Base exception for trade execution errors."""
    pass

class InsufficientBalanceError(TradeExecutionError):
    """Raised when account balance is insufficient for trade."""
    pass

class MarketConditionError(TradeExecutionError):
    """Raised when market conditions are not suitable for trading."""
    pass

class OrderTimeoutError(TradeExecutionError):
    """Raised when an order placement times out."""
    pass

class ExchangeConnectionError(TradeExecutionError):
    """Raised when there are issues connecting to the exchange."""
    pass

class RateLimitError(TradeExecutionError):
    """Raised when exchange rate limit is hit."""
    pass

class StaleDataError(TradeExecutionError):
    """Raised when market data is too old for reliable trading."""
    pass

class OrderValidationError(TradeExecutionError):
    """Raised when order parameters fail validation."""
    pass

class ExchangeAPIError(TradeExecutionError):
    """Raised when exchange API returns an error response."""
    pass

class TradeExecutor:
    """Handles trade execution with market validation and risk management.
    
    This class is responsible for executing trades while ensuring market conditions
    are favorable and risk management rules are followed.
    """
    
    def __init__(self, 
                 exchange: Any,
                 performance: Dict[str, Any],
                 trade_history: Dict[str, List[Dict[str, Any]]]) -> None:
        """Initialize TradeExecutor with exchange and tracking components.
        
        Args:
            exchange: The exchange instance for executing trades
            performance: Dictionary tracking trading performance metrics
            trade_history: Dictionary storing historical trade data
            
        Raises:
            ValueError: If any required parameters are None
        """
        if not all([exchange, performance, trade_history]):
            raise ValueError("All parameters must be provided")
        self.exchange = exchange
        self.performance = performance
        self.trade_history = trade_history
        self.market_data: Optional[pd.DataFrame] = None
        self.market_analyzer = MarketAnalyzer()
        self.market_validator = MarketValidation()
        self.risk_manager = RiskManager(exchange, trade_history)

    def _validate_trade_prerequisites(self, side: str, amount: float, symbol: str) -> bool:
        """Validate all prerequisites before executing a trade.
        
        Args:
            side: Trade direction ('buy' or 'sell')
            amount: Amount of base currency to trade
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            bool: True if all prerequisites are met
            
        Raises:
            ValueError: If side is not 'buy' or 'sell'
            ValueError: If amount is not positive
            ValueError: If symbol format is invalid
            ExchangeConnectionError: If exchange connection fails
            MarketConditionError: If market conditions are not suitable
            InsufficientBalanceError: If balance is insufficient for trade
        """
        # Validate input parameters
        if side not in ["buy", "sell"]:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
        if amount <= 0:
            raise ValueError(f"Invalid amount: {amount}. Must be positive")
        if '/' not in symbol:
            raise ValueError(f"Invalid symbol format: {symbol}. Must be in format 'BTC/USDT'")
        if not self.check_exchange_connection():
            logging.error("Exchange connection check failed before trade execution")
            return False

        if not self.validate_market_conditions(self.market_data):
            logging.warning(f"Market conditions not met for {symbol}, skipping trade")
            return False

        # Validate balance for sell orders
        if side == "sell":
            balance = safe_api_call(self.exchange.fetch_balance)
            base_currency = symbol.split('/')[0]
            base_balance = balance[base_currency]['free']
            
            if base_balance < amount:
                logging.warning(f"Insufficient balance for selling {base_currency}. Available: {base_balance}, Required: {amount}")
                return False

        return True

    def _check_rate_limit(self) -> None:
        """Check if we're within rate limits and wait if necessary.
        
        Raises:
            RateLimitError: If rate limit is consistently exceeded
        """
        if hasattr(self, '_last_api_call'):
            elapsed = time.time() - self._last_api_call
            min_interval = CONFIG.get('min_api_interval', 0.1)  # 100ms default
            
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                logging.debug(f"Rate limit: waiting {wait_time:.3f}s")
                time.sleep(wait_time)
        
        self._last_api_call = time.time()

    def _validate_market_data_freshness(self) -> None:
        """Validate that market data is recent enough for trading.
        
        Raises:
            StaleDataError: If market data is too old
        """
        if self.market_data is None:
            raise StaleDataError("No market data available")
            
        last_timestamp = self.market_data.index[-1]
        if isinstance(last_timestamp, str):
            last_timestamp = pd.to_datetime(last_timestamp)
            
        staleness = (datetime.now() - last_timestamp).total_seconds()
        max_staleness = CONFIG.get('max_data_staleness', 60)  # 1 minute default
        
        if staleness > max_staleness:
            raise StaleDataError(
                f"Market data is {staleness:.1f}s old (max {max_staleness}s)"
            )

    def _place_market_order(self, side: str, amount: float, symbol: str, start_time: float) -> Optional[Dict[str, Any]]:
        """Place a market order with enhanced error handling and rate limiting.
        
        Args:
            side: Trade direction ('buy' or 'sell')
            amount: Amount of base currency to trade
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            start_time: Timestamp when order placement started
            
        Returns:
            Optional[Dict[str, Any]]: Order details if successful, None if failed.
                The order details dictionary contains:
                - id: Unique order identifier
                - status: Order status ('closed' if successful)
                - price: Executed price
                - amount: Executed amount
                - timestamp: Order execution time
                - fees: Trading fees charged
                - cost: Total cost including fees
            
        Raises:
            OrderTimeoutError: If order placement exceeds timeout
            ExchangeConnectionError: If exchange request fails
            RateLimitError: If exchange rate limit is hit
            InsufficientBalanceError: If balance is insufficient
            OrderValidationError: If order parameters are invalid
            ExchangeAPIError: If exchange returns an error
            ValueError: If side is invalid, amount is not positive, or symbol format is invalid
            
        Examples:
            >>> executor = TradeExecutor(exchange, performance, trade_history)
            >>> start_time = time.time()
            >>> order = executor._place_market_order("buy", 0.1, "BTC/USDT", start_time)
            >>> if order:
            ...     print(f"Order executed at {order['price']}")
            ...     print(f"Total cost with fees: {order['cost']}")
        """
        # Pre-order validations
        if side not in ["buy", "sell"]:
            raise OrderValidationError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
        if amount <= 0:
            raise OrderValidationError(f"Invalid amount: {amount}. Must be positive")
        if '/' not in symbol:
            raise OrderValidationError(f"Invalid symbol format: {symbol}. Must be in format 'BTC/USDT'")
            
        # Validate market data freshness
        try:
            self._validate_market_data_freshness()
        except StaleDataError as e:
            logging.error(f"Stale market data: {str(e)}")
            raise
            
        # Log order attempt with detailed parameters
        logging.info(
            "Initiating order placement",
            extra={
                "side": side,
                "amount": amount,
                "symbol": symbol,
                "current_price": self.market_data['close'].iloc[-1] if self.market_data is not None else None
            }
        )
        
        timeout = CONFIG['order_timeout']
        retry_count = 0
        max_retries = CONFIG.get('max_order_retries', 3)
        
        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            logging.debug(f"Order attempt {retry_count + 1}/{max_retries} ({elapsed:.1f}s / {timeout}s)")
            
            try:
                # Check rate limits before API call
                self._check_rate_limit()
                
                # Place the order with comprehensive error handling
                try:
                    order = (self.exchange.create_market_buy_order(symbol, amount) 
                            if side == "buy" 
                            else self.exchange.create_market_sell_order(symbol, amount))

                    # Enhanced order validation
                    if not isinstance(order, dict) or 'id' not in order:
                        raise ExchangeAPIError("Invalid order response format")
                        
                    # Check if order is filled
                    if order['status'] == 'closed':
                        # Calculate and log execution metrics
                        slippage = ((float(order['price']) - self.market_data['close'].iloc[-1]) 
                                  / self.market_data['close'].iloc[-1] * 100)
                        
                        logging.info(
                            "Order executed successfully",
                            extra={
                                "order_id": order['id'],
                                "execution_price": order['price'],
                                "slippage_percent": f"{slippage:.2f}%",
                                "execution_time": f"{elapsed:.3f}s"
                            }
                        )
                        return order

                    logging.debug(f"Order not yet filled, status: {order['status']}")
                    time.sleep(1)  # Wait before checking again

                except Exception as e:
                    error_msg = str(e).lower()
                    if 'insufficient balance' in error_msg:
                        raise InsufficientBalanceError(f"Insufficient balance for {side} order: {str(e)}")
                    elif 'rate limit' in error_msg:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise RateLimitError(f"Rate limit exceeded after {retry_count} retries")
                        wait_time = 2 ** retry_count  # Exponential backoff
                        logging.warning(f"Rate limit hit, waiting {wait_time}s before retry {retry_count + 1}")
                        time.sleep(wait_time)
                        continue
                    elif any(msg in error_msg for msg in ['invalid', 'minimum', 'maximum']):
                        raise OrderValidationError(f"Order validation failed: {str(e)}")
                    else:
                        logging.error(f"Order execution error: {str(e)}")
                        raise ExchangeAPIError(f"Exchange API error: {str(e)}")

            except (InsufficientBalanceError, OrderValidationError, RateLimitError, 
                   ExchangeAPIError) as e:
                logging.error(f"Order placement failed: {str(e)}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error during order placement: {str(e)}")
                return None

        timeout_msg = f"Order timeout after {timeout} seconds"
        logging.error(timeout_msg)
        raise OrderTimeoutError(timeout_msg)

    def _update_trade_history(self, order: Dict[str, Any], side: str, amount: float, symbol: str) -> None:
        """Update trade history and send notifications for successful trades.
        
        Args:
            order: The executed order details
            side: Trade direction ('buy' or 'sell')
            amount: Amount of base currency traded
            symbol: Trading pair symbol
        """
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

    def execute_trade(self, side: str, amount: float, symbol: str) -> Optional[Dict[str, Any]]:
        """Execute a market trade with validation and error handling.
        
        This method coordinates the entire trade execution process including:
        1. Validating prerequisites (connection, market conditions, balance)
        2. Placing the market order with timeout handling
        3. Updating trade history and sending notifications
        
        Args:
            side: Trade direction ('buy' or 'sell')
            amount: Amount of base currency to trade
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Optional[Dict[str, Any]]: Order details if successful, None if failed.
                The order details dictionary contains:
                - id: Unique order identifier
                - status: Order status ('closed' if successful)
                - price: Executed price
                - amount: Executed amount
                - timestamp: Order execution time
                - side: Trade direction
                - symbol: Trading pair
            
        Raises:
            ValueError: If side is invalid, amount is not positive, or symbol format is invalid
            InsufficientBalanceError: If account balance is insufficient for trade
            MarketConditionError: If market conditions are not suitable
            OrderTimeoutError: If order placement exceeds timeout
            ExchangeConnectionError: If exchange request fails
            
        Examples:
            >>> executor = TradeExecutor(exchange, performance, trade_history)
            >>> # Execute a buy order
            >>> order = executor.execute_trade("buy", 0.1, "BTC/USDT")
            >>> if order:
            ...     print(f"Trade executed at {order['price']}")
            >>> # Execute a sell order
            >>> order = executor.execute_trade("sell", 0.05, "ETH/USDT")
            >>> if order:
            ...     print(f"Sold at {order['price']}")
        """
        logging.info(f"Starting trade execution - {side} {amount} {symbol}")
        
        try:
            # Validate prerequisites
            if not self._validate_trade_prerequisites(side, amount, symbol):
                logging.warning("Trade prerequisites validation failed")
                return None

            # Execute order
            start_time = time.time()
            try:
                order = self._place_market_order(side, amount, symbol, start_time)
            except OrderTimeoutError as e:
                logging.error(f"Order timeout: {str(e)}")
                self.send_notification(f"Order timeout: {symbol} {side}", "high")
                return None
            except InsufficientBalanceError as e:
                logging.error(f"Insufficient balance: {str(e)}")
                self.send_notification(f"Insufficient balance for {symbol} {side}", "high")
                return None
            except ExchangeConnectionError as e:
                logging.error(f"Exchange connection error: {str(e)}")
                if not self.handle_trade_error(e):
                    self.send_notification(f"Exchange error: {str(e)}", "high")
                return None
            
            # Update history if order successful
            if order:
                self._update_trade_history(order, side, amount, symbol)
                logging.info(f"Trade execution completed successfully: {order['id']}")
                return order

            logging.warning("Trade execution failed - no order returned")
            return None

        except ValueError as e:
            logging.error(f"Invalid parameters in execute_trade: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in execute_trade: {str(e)}")
            self.handle_trade_error(e)
            return None

    def check_exchange_connection(self) -> bool:
        """Check if exchange connection is active and working.
        
        Returns:
            bool: True if connection is working
            
        Raises:
            ExchangeConnectionError: If connection check fails
        """
        try:
            self.exchange.fetch_ticker(CONFIG['symbol'])
            logging.debug("Exchange connection check successful")
            return True
        except Exception as e:
            logging.error(f"Exchange connection error: {e}")
            raise ExchangeConnectionError(f"Failed to connect to exchange: {str(e)}")

    def validate_market_conditions(self, market_data: Optional[pd.DataFrame]) -> bool:
        """Validate market conditions specifically for spot trading.
        
        Performs comprehensive validation of market conditions including:
        - Data structure and quality checks
        - Market health assessment
        - Spread analysis
        - Volume validation
        - Entry signal evaluation
        
        Args:
            market_data: DataFrame containing OHLCV market data with columns:
                        [open, high, low, close, volume]
            
        Returns:
            bool: True if market conditions are valid for trading, False otherwise
            
        Raises:
            MarketConditionError: If market data is invalid or missing
            ValueError: If DataFrame structure is incorrect
            TypeError: If market_data is not a pandas DataFrame
            
        Examples:
            >>> executor = TradeExecutor(exchange, performance, trade_history)
            >>> df = pd.DataFrame({
            ...     'open': [100, 101, 102],
            ...     'high': [103, 104, 105],
            ...     'low': [98, 99, 100],
            ...     'close': [101, 102, 103],
            ...     'volume': [1000, 1100, 1200]
            ... })
            >>> is_valid = executor.validate_market_conditions(df)
        """
        logging.debug("Starting market conditions validation")
        
        try:
            # Type checking
            if market_data is not None and not isinstance(market_data, pd.DataFrame):
                raise TypeError("market_data must be a pandas DataFrame or None")
            # Data validation
            if market_data is None or market_data.empty:
                raise MarketConditionError("No market data available for validation")
                
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in market_data.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in market_data.columns]
                raise MarketConditionError(f"Market data missing required columns: {', '.join(missing_cols)}")
                
            # Comprehensive data quality checks
            for col in required_columns:
                if market_data[col].isna().any():
                    raise MarketConditionError(f"NaN values detected in {col} column")
                if (market_data[col] <= 0).any():
                    raise MarketConditionError(f"Non-positive values detected in {col} column")
                
            # Market analysis
            current_price = market_data['close'].iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            price_change = (current_price - market_data['close'].iloc[-2]) / market_data['close'].iloc[-2] * 100
            
            logging.debug(f"Validating market conditions at price: {current_price}")
            trend_analysis = self.market_analyzer.analyze_price_trend(market_data)
            rsi = trend_analysis['rsi']

            logging.info(f"""
            === Detailed Market Analysis ===
            Price: {current_price:.2f} USDT
            24h Change: {price_change:.2f}%
            RSI: {rsi:.2f}
            Volume: {current_volume * current_price:.2f} USDT
            Timestamp: {datetime.now().isoformat()}
            """)

            # Use MarketValidation class for all validations
            if not self.market_validator.check_market_health(market_data, self.exchange):
                logging.info("❌ Market health check failed")
                return False

            if not self.market_validator.check_spread(market_data):
                logging.info("❌ High spread detected")
                return False

            # Check entry conditions
            if self.market_validator.check_strong_buy_signal(market_data):
                logging.info(f"✅ Strong buy signal - RSI oversold: {rsi:.2f}")
                return True
            
            if self.market_validator.check_moderate_buy_signal(market_data):
                logging.info(f"✅ Moderate buy signal - RSI in accumulation zone: {rsi:.2f}")
                return True

            if not self.market_validator.validate_volume(market_data):
                logging.info("❌ Volume conditions not met")
                return False

            logging.info(f"❌ No clear entry signal - RSI: {rsi:.2f}")
            return False

        except Exception as e:
            logging.error(f"Error validating market conditions: {str(e)}")
            return False

    def handle_trade_error(self, error: Exception, retry_count: int = 3) -> bool:
        """Handle trade errors with retries and notifications.
        
        Implements exponential backoff for retries and sends notifications
        for critical errors that require manual intervention.
        
        Args:
            error: The exception that occurred during trade execution
            retry_count: Maximum number of retry attempts (default: 3)
            
        Returns:
            bool: True if error was handled successfully, False otherwise
            
        Raises:
            ExchangeConnectionError: If unable to reconnect to exchange after retries
            ValueError: If retry_count is less than 1
            
        Examples:
            >>> executor = TradeExecutor(exchange, performance, trade_history)
            >>> try:
            ...     # Some trading operation
            ...     pass
            ... except Exception as e:
            ...     success = executor.handle_trade_error(e)
            ...     if not success:
            ...         # Handle failure
            ...         pass
        """
        if retry_count < 1:
            raise ValueError("retry_count must be at least 1")
            
        logging.debug(f"Starting error handling for: {str(error)}")
        try:
            for i in range(retry_count):
                try:
                    logging.error(f"Trade error (attempt {i+1}/{retry_count}): {str(error)}")

                    if any(critical in str(error).lower() for critical in [
                        'insufficient balance',
                        'api key',
                        'permission denied',
                        'margin'
                    ]):
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

    def send_notification(self, message: str, priority: str = "normal") -> bool:
        """Send a notification via Telegram with priority handling and rate limiting.
        
        Sends trade-related notifications through Telegram with enhanced error handling,
        rate limiting, and priority-based message formatting. Supports different priority
        levels for message formatting and delivery urgency.
        
        Args:
            message: The message to send. Must be a non-empty string with length <= 4096
                    (Telegram's message length limit).
            priority: Priority level of the message. Must be one of:
                     "low": Regular trading updates
                     "normal": Trade executions and important events (default)
                     "high": Critical errors and urgent alerts
        
        Returns:
            bool: True if notification was sent successfully, False otherwise
        
        Raises:
            ValueError: If message is empty, not a string, or exceeds length limit
            ValueError: If priority is not one of ["low", "normal", "high"]
            ConnectionError: If network connection fails
            TimeoutError: If notification request times out
            RuntimeError: If notification service is unavailable
        
        Examples:
            >>> executor = TradeExecutor(exchange, performance, trade_history)
            >>> # Send normal trade notification
            >>> executor.send_notification("Trade executed: BTC/USDT at 50000")
            True
            >>> # Send high priority error notification
            >>> executor.send_notification("Critical: Insufficient balance", "high")
            True
            >>> # Send low priority update
            >>> executor.send_notification("Market analysis complete", "low")
            True
        """
        # Validate priority
        valid_priorities = ["low", "normal", "high"]
        if priority not in valid_priorities:
            raise ValueError(f"Invalid priority. Must be one of: {valid_priorities}")
        
        # Enhanced message validation
        if not isinstance(message, str):
            raise ValueError("Message must be a string")
        if not message.strip():
            raise ValueError("Message cannot be empty")
        if len(message) > 4096:
            raise ValueError("Message exceeds Telegram's 4096 character limit")
            
        # Format message based on priority
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        priority_prefixes = {
            "low": "ℹ️",
            "normal": "📊",
            "high": "🚨"
        }
        formatted_message = (
            f"{priority_prefixes[priority]} {timestamp}\n"
            f"{message}"
        )
        
        logging.debug(f"Sending {priority} priority notification: {message[:100]}...")
        
        try:
            # Implement rate limiting
            if hasattr(self, '_last_notification_time'):
                time_since_last = time.time() - self._last_notification_time
                if time_since_last < CONFIG.get('notification_rate_limit', 1):
                    wait_time = CONFIG.get('notification_rate_limit', 1) - time_since_last
                    logging.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
            
            send_telegram_notification(formatted_message)
            self._last_notification_time = time.time()
            
            logging.info(
                "Notification sent",
                extra={
                    "priority": priority,
                    "message_length": len(message),
                    "timestamp": timestamp
                }
            )
            return True
            
        except ConnectionError as e:
            logging.error(f"Network error sending notification: {str(e)}")
            return False
        except TimeoutError as e:
            logging.error(f"Notification timeout: {str(e)}")
            return False
        except Exception as e:
            error_msg = f"Failed to send notification: {str(e)}"
            logging.error(error_msg)
            if priority == "high":
                # For high priority, re-raise to ensure caller handles it
                raise RuntimeError(error_msg) from e
            return False
