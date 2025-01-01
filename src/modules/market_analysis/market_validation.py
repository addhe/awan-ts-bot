import logging
from datetime import datetime
from typing import Dict, Optional, Union, Any
from decimal import Decimal

import pandas as pd
from pandas import DataFrame
from config.config import CONFIG

from .indicators import Indicators

# Custom types
MarketData = DataFrame
ExchangeType = Any  # Type for exchange instance

class MarketValidation:
    """Class responsible for market condition validation and health checks.
    
    This class provides comprehensive market validation functionality including:
    - Market health monitoring through volume and liquidity checks
    - Spread validation and monitoring
    - Entry condition validation using technical indicators
    - Volume analysis and validation
    """
    
    def __init__(self, exchange: ExchangeType) -> None:
        """Initialize MarketValidation with exchange instance.
        
        Args:
            exchange: Exchange instance for market operations. Must implement
                     fetch_ticker method for market data retrieval.
        
        Attributes:
            exchange: The exchange instance used for market operations
            market_data: Optional DataFrame containing market OHLCV data
        """
        self.exchange = exchange
        self.market_data: Optional[MarketData] = None

    def validate_market_conditions(self, market_data: MarketData) -> bool:
        """Validate market conditions specifically for spot trading.
        
        Performs comprehensive market validation including:
        - Market health check (volume, liquidity)
        - Spread validation
        - RSI-based entry conditions
        - Volume validation
        
        Args:
            market_data: DataFrame containing OHLCV data
            
        Returns:
            bool: True if market conditions are valid for trading
            
        Raises:
            KeyError: If required columns are missing from market_data
            ValueError: If market data validation fails
            pd.errors.EmptyDataError: If market_data is empty
        """
        if not self.validate_market_data(market_data):
            raise ValueError("Invalid market data structure")
            
        try:
            # Get current market state
            current_price = market_data['close'].iloc[-1]
            indicators = Indicators()
            rsi = indicators.calculate_rsi(market_data['close']).iloc[-1]
            
            # Log initial market analysis
            self._log_initial_analysis(current_price, rsi, market_data['volume'].iloc[-1])
            
            # Perform validation checks in sequence
            if not self._validate_health_and_spread(market_data):
                return False
                
            # Check entry conditions
            if self._check_strong_buy_signal(rsi):
                return True
                
            if self._check_moderate_buy_signal(rsi, market_data):
                return True
                
            # Final volume validation
            if not self.validate_volume(market_data):
                logging.info("❌ Volume conditions not met")
                return False
            
            logging.info(f"❌ No clear entry signal - RSI: {rsi:.2f}")
            return False

        except KeyError as e:
            logging.error(f"Missing required market data column: {str(e)}")
            return False
        except pd.errors.EmptyDataError:
            logging.error("Market data is empty")
            return False
        except Exception as e:
            logging.error(f"Unexpected error validating market conditions: {str(e)}")
            return False

    def validate_volume(self, market_data: MarketData) -> bool:
        """Validate if current volume meets minimum requirements.
        
        Calculates the ratio between current volume and moving average volume
        to determine if trading activity is sufficient.
        
        Args:
            market_data: DataFrame containing OHLCV data with 'volume' and 'close' columns
            
        Returns:
            bool: True if volume conditions are met (current volume >= minimum multiplier * average volume)
            
        Raises:
            KeyError: If volume or close price data is missing
            ValueError: If volume calculations fail due to insufficient data
            pd.errors.EmptyDataError: If market_data is empty
        """
        try:
            if market_data.empty:
                raise pd.errors.EmptyDataError("Empty market data provided")
                
            current_volume = market_data['volume'].iloc[-1] * market_data['close'].iloc[-1]
            avg_volume = (market_data['volume'].rolling(window=CONFIG['volume_ma_period'])
                        .mean().iloc[-1] * market_data['close'].iloc[-1])

            if pd.isna(avg_volume) or avg_volume == 0:
                raise ValueError("Invalid average volume calculation")

            volume_ratio = current_volume / avg_volume
            logging.info(f"Volume ratio: {volume_ratio:.2f}")

            return volume_ratio >= CONFIG['min_volume_multiplier']
            
        except (KeyError, IndexError) as e:
            logging.error(f"Missing required market data: {str(e)}")
            return False
        except ValueError as e:
            logging.error(f"Volume calculation error: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error in volume validation: {str(e)}")
            return False

    def check_market_health(self) -> bool:
        """Check overall market health using 24h volume.
        
        Compares 24-hour trading volume against historical average to determine
        if the market is sufficiently active and healthy for trading.
        
        Returns:
            bool: True if market is healthy (24h volume >= 80% of average daily volume)
            
        Raises:
            KeyError: If required ticker data is missing
            ValueError: If volume calculations fail
            RuntimeError: If exchange API request fails
        """
        try:
            if self.market_data is None:
                raise ValueError("No historical market data available")

            ticker = self.exchange.fetch_ticker(CONFIG['symbol'])
            if 'quoteVolume' not in ticker:
                raise KeyError("Quote volume data not available in ticker")
                
            volume_24h = ticker['quoteVolume']
            if volume_24h <= 0:
                raise ValueError("Invalid 24h volume value")

            # Enhanced volume check
            avg_volume = self.market_data['volume'].mean() * self.market_data['close'].mean()
            if pd.isna(avg_volume) or avg_volume == 0:
                raise ValueError("Invalid average volume calculation")
                
            volume_ratio = volume_24h / (avg_volume * 24)
            logging.info(f"Volume ratio: {volume_ratio:.2f}")

            if volume_ratio < 0.8:  # Volume should be at least 80% of average
                logging.warning("Volume below average")
                return False

            return True
            
        except (KeyError, ValueError) as e:
            logging.error(f"Market health check failed: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error in market health check: {str(e)}")
            return False

    def check_spread(self, market_data: MarketData) -> float:
        """Check if current spread is within acceptable range.
        
        Calculates the bid-ask spread as a percentage of the bid price to determine
        if the market is sufficiently liquid for trading.
        
        Args:
            market_data: DataFrame containing OHLCV data with bid/ask prices
            
        Returns:
            float: Current spread as a decimal (0.0 if spread cannot be calculated)
            
        Raises:
            KeyError: If bid/ask price data is missing
            ValueError: If spread calculation fails
            pd.errors.EmptyDataError: If market_data is empty
        """
        try:
            if market_data.empty:
                raise pd.errors.EmptyDataError("Empty market data provided")
                
            if 'ask' not in market_data.columns or 'bid' not in market_data.columns:
                raise KeyError("Bid/Ask price data not available")

            bid_price = market_data['bid'].iloc[-1]
            ask_price = market_data['ask'].iloc[-1]
            
            if bid_price <= 0 or ask_price <= 0:
                raise ValueError("Invalid bid/ask prices")

            spread = (ask_price - bid_price) / bid_price
            logging.info(f"Current spread: {spread:.4%}")

            return spread
            
        except (KeyError, IndexError) as e:
            logging.error(f"Missing required price data: {str(e)}")
            return 0.0
        except ValueError as e:
            logging.error(f"Spread calculation error: {str(e)}")
            return 0.0
        except Exception as e:
            logging.error(f"Unexpected error checking spread: {str(e)}")
            return 0.0

    def validate_market_data(self, market_data: Optional[MarketData]) -> bool:
        """Validate market data structure and content.
        
        Performs comprehensive validation of market data including:
        - Checks for None or empty DataFrame
        - Validates presence of required OHLCV columns
        - Ensures sufficient historical data points
        - Verifies data integrity (no NaN values in critical columns)
        
        Args:
            market_data: DataFrame containing OHLCV data
            
        Returns:
            bool: True if market data is valid and meets all requirements
            
        Raises:
            ValueError: If market data is None or empty
            KeyError: If required columns are missing
            pd.errors.EmptyDataError: If insufficient historical data
        """
        try:
            if market_data is None:
                raise ValueError("Market data is None")
                
            if market_data.empty:
                raise pd.errors.EmptyDataError("Empty market data provided")

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            if missing_columns:
                raise KeyError(f"Missing required columns: {', '.join(missing_columns)}")

            if len(market_data) < CONFIG['min_candles_required']:
                raise pd.errors.EmptyDataError(
                    f"Insufficient historical data. Required: {CONFIG['min_candles_required']}, "
                    f"Got: {len(market_data)}"
                )
                
            # Check for NaN values in critical columns
            if market_data[required_columns].isna().any().any():
                raise ValueError("NaN values detected in critical columns")

            return True
            
        except (ValueError, KeyError, pd.errors.EmptyDataError) as e:
            logging.error(f"Market data validation failed: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error in market data validation: {str(e)}")
            return False

    def _log_initial_analysis(self, current_price: float, rsi: float, 
                            current_volume: float) -> None:
        """Log initial market analysis details.
        
        Provides a formatted log output of key market metrics for analysis tracking
        and debugging purposes.
        
        Args:
            current_price: Current market price in quote currency
            rsi: Current RSI value (0-100 range)
            current_volume: Current trading volume in base currency
            
        Note:
            Volume is converted to quote currency (USDT) in the log output
            for better readability and comparison
        """
        logging.info(f"""
        === Detailed Market Analysis ===
        Price: {current_price:.2f} USDT
        RSI: {rsi:.2f}
        Volume: {current_volume * current_price:.2f} USDT
        """)
        
    def _validate_health_and_spread(self, market_data: MarketData) -> bool:
        """Validate market health and spread conditions.
        
        Performs two key validations:
        1. Market health check using 24h volume comparison
        2. Spread validation against maximum allowed threshold
        
        Args:
            market_data: DataFrame containing OHLCV data with bid/ask prices
            
        Returns:
            bool: True if both health and spread conditions are met
            
        Note:
            Logs specific failure reasons for debugging and monitoring
        """
        if not self.check_market_health():
            logging.info("❌ Market health check failed")
            return False
            
        current_spread = self.check_spread(market_data)
        if current_spread > CONFIG['max_spread_percent'] / 100:
            logging.info("❌ High spread detected")
            return False
            
        return True
        
    def _check_strong_buy_signal(self, rsi: float) -> bool:
        """Check if RSI indicates a strong buy signal (oversold condition).
        
        Args:
            rsi: Current RSI value
            
        Returns:
            bool: True if strong buy signal is detected
        """
        if rsi <= CONFIG['rsi_oversold']:
            logging.info(f"✅ Strong buy signal - RSI oversold: {rsi:.2f}")
            return True
        return False
        
    def _check_moderate_buy_signal(self, rsi: float, market_data: MarketData) -> bool:
        """Check if market conditions indicate a moderate buy signal.
        
        A moderate buy signal is determined by:
        - RSI in accumulation zone (< 45)
        - Price change below threshold
        
        Args:
            rsi: Current RSI value
            market_data: DataFrame containing OHLCV data
            
        Returns:
            bool: True if moderate buy signal is detected
            
        Raises:
            KeyError: If close price data is missing
            ValueError: If price change calculation fails
        """
        try:
            if rsi < 45:
                price_change = abs(market_data['close'].pct_change().iloc[-1])
                if price_change < CONFIG['price_change_threshold']:
                    logging.info(f"✅ Moderate buy signal - RSI in accumulation zone: {rsi:.2f}")
                    return True
            return False
        except (KeyError, ValueError) as e:
            logging.error(f"Error checking moderate buy signal: {str(e)}")
            return False
