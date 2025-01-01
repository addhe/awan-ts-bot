import logging
from typing import Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from config.config import CONFIG

# Configure logging
logger = logging.getLogger(__name__)

# Custom types
MarketData = DataFrame
AnalysisResult = Dict[str, Union[float, Dict[str, float]]]

class InvalidDataError(Exception):
    """Raised when input data is invalid or missing required fields."""
    pass

class TrendAnalysis:
    """Class responsible for analyzing market trends and trade signals.
    
    This class provides methods to analyze market trends, detect trading signals,
    and validate trading conditions based on technical indicators like EMA,
    RSI, ADX, and momentum.
    
    Methods:
        check_ema_crossover: Detect bullish EMA crossover signals
        check_trend_conditions: Validate if trend conditions meet trading criteria
        should_execute_trade: Determine if trade execution conditions are met
        analyze_trend_direction: Analyze overall trend direction
        calculate_trend_momentum: Calculate trend momentum score
    """

    @staticmethod
    def check_ema_crossover(ema_data: DataFrame) -> bool:
        """Check for EMA crossover signal.
        
        Detects bullish EMA crossover by comparing short and long EMAs
        for the current and previous periods.
        
        Args:
            ema_data: DataFrame containing 'ema_short' and 'ema_long' columns
                     with at least 2 periods of data
            
        Returns:
            bool: True if bullish EMA crossover occurred (short EMA crosses above long EMA)
            
        Raises:
            InvalidDataError: If ema_data is missing required columns or has insufficient data
            
        Example:
            >>> ema_df = pd.DataFrame({
            ...     'ema_short': [10, 11],
            ...     'ema_long': [12, 10]
            ... })
            >>> TrendAnalysis.check_ema_crossover(ema_df)
            True
        """
        try:
            # Validate input data
            if not isinstance(ema_data, DataFrame):
                raise InvalidDataError("ema_data must be a pandas DataFrame")
                
            required_columns = ['ema_short', 'ema_long']
            if not all(col in ema_data.columns for col in required_columns):
                raise InvalidDataError(f"ema_data missing required columns: {required_columns}")
                
            if len(ema_data) < 2:
                raise InvalidDataError("ema_data must contain at least 2 periods")
                
            # Check for NaN values
            if ema_data[required_columns].isna().any().any():
                raise InvalidDataError("ema_data contains NaN values")
            
            crossover = (
                ema_data['ema_short'].iloc[-2] < ema_data['ema_long'].iloc[-2] and  # Previous
                ema_data['ema_short'].iloc[-1] > ema_data['ema_long'].iloc[-1]      # Current
            )
            
            logger.debug(f"EMA Crossover check result: {crossover}")
            return crossover
            
        except InvalidDataError as e:
            logger.error(f"Invalid data in EMA crossover check: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error checking EMA crossover: {str(e)}")
            return False

    @staticmethod
    def check_trend_conditions(trend_analysis: Dict[str, float]) -> bool:
        """Check if trend conditions meet trading criteria.
        
        Validates multiple technical indicators against configured thresholds
        to determine if market conditions are favorable for trading.
        
        Args:
            trend_analysis: Dictionary containing trend indicators with keys:
                          'rsi', 'adx', 'momentum', 'trend_strength'
            
        Returns:
            bool: True if all trend conditions are favorable
            
        Raises:
            InvalidDataError: If trend_analysis is missing required indicators
            
        Example:
            >>> analysis = {
            ...     'rsi': 45,
            ...     'adx': 30,
            ...     'momentum': 0.5,
            ...     'trend_strength': 0.8
            ... }
            >>> TrendAnalysis.check_trend_conditions(analysis)
            True
        """
        try:
            # Validate input data
            required_indicators = ['rsi', 'adx', 'momentum', 'trend_strength']
            if not all(indicator in trend_analysis for indicator in required_indicators):
                raise InvalidDataError(f"trend_analysis missing required indicators: {required_indicators}")
            
            # Check for valid numeric values
            for indicator in required_indicators:
                if not isinstance(trend_analysis[indicator], (int, float)):
                    raise InvalidDataError(f"Invalid value for {indicator}: {trend_analysis[indicator]}")
                if np.isnan(trend_analysis[indicator]):
                    raise InvalidDataError(f"NaN value detected for {indicator}")
            
            conditions_met = (
                trend_analysis['rsi'] < CONFIG['rsi_overbought'] and
                trend_analysis['adx'] > CONFIG['adx_threshold'] and
                trend_analysis['momentum'] > CONFIG['momentum_threshold'] and
                trend_analysis['trend_strength'] > CONFIG['trend_strength_threshold']
            )
            
            logger.debug(
                f"Trend conditions check: RSI={trend_analysis['rsi']:.2f}, "
                f"ADX={trend_analysis['adx']:.2f}, "
                f"Momentum={trend_analysis['momentum']:.2f}, "
                f"Strength={trend_analysis['trend_strength']:.2f}, "
                f"Result={conditions_met}"
            )
            
            return conditions_met
            
        except InvalidDataError as e:
            logger.error(f"Invalid trend analysis data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error checking trend conditions: {str(e)}")
            return False

    @staticmethod
    def should_execute_trade(analysis_result: Optional[AnalysisResult]) -> bool:
        """Determine if trade should be executed based on technical analysis.
        
        Combines EMA crossover signals and trend conditions to make a final
        trading decision.
        
        Args:
            analysis_result: Dictionary containing:
                           - 'ema_data': DataFrame with EMA values
                           - 'trend_analysis': Dict with trend indicators
            
        Returns:
            bool: True if all trading conditions are met
            
        Raises:
            InvalidDataError: If analysis_result is missing required data
            
        Example:
            >>> result = {
            ...     'ema_data': pd.DataFrame({
            ...         'ema_short': [10, 11],
            ...         'ema_long': [12, 10]
            ...     }),
            ...     'trend_analysis': {
            ...         'rsi': 45,
            ...         'adx': 30,
            ...         'momentum': 0.5,
            ...         'trend_strength': 0.8
            ...     }
            ... }
            >>> TrendAnalysis.should_execute_trade(result)
            True
        """
        try:
            if analysis_result is None:
                logger.warning("Analysis result is None")
                return False

            # Validate required data
            required_keys = ['ema_data', 'trend_analysis']
            if not all(key in analysis_result for key in required_keys):
                raise InvalidDataError(f"analysis_result missing required keys: {required_keys}")
            
            # Extract required data
            ema_data = analysis_result['ema_data']
            trend_analysis = analysis_result['trend_analysis']
            
            # Check both EMA crossover and trend conditions
            should_trade = (TrendAnalysis.check_ema_crossover(ema_data) and 
                          TrendAnalysis.check_trend_conditions(trend_analysis))
            
            logger.info(f"Trade execution decision: {should_trade}")
            return should_trade

        except InvalidDataError as e:
            logger.error(f"Invalid analysis result data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error checking trade conditions: {str(e)}")
            return False

    @staticmethod
    def analyze_trend_direction(prices: Series, window: int = 20) -> Tuple[str, float]:
        """Analyze the overall trend direction using linear regression.
        
        Args:
            prices: Series of price data
            window: Number of periods to analyze (default: 20)
            
        Returns:
            Tuple[str, float]: (trend direction, slope coefficient)
            
        Raises:
            InvalidDataError: If prices data is invalid
            
        Example:
            >>> prices = pd.Series([1, 2, 3, 4, 5])
            >>> TrendAnalysis.analyze_trend_direction(prices, window=5)
            ('uptrend', 1.0)
        """
        try:
            if not isinstance(prices, Series):
                raise InvalidDataError("prices must be a pandas Series")
            
            if len(prices) < window:
                raise InvalidDataError(f"Insufficient data: need at least {window} periods")
            
            if prices.isna().any():
                raise InvalidDataError("prices contains NaN values")
            
            # Calculate linear regression
            x = np.arange(window)
            y = prices.iloc[-window:].values
            slope = np.polyfit(x, y, 1)[0]
            
            # Determine trend direction
            if slope > 0.001:
                direction = 'uptrend'
            elif slope < -0.001:
                direction = 'downtrend'
            else:
                direction = 'sideways'
                
            logger.debug(f"Trend direction analysis: {direction} (slope: {slope:.4f})")
            return direction, slope
            
        except InvalidDataError as e:
            logger.error(f"Invalid price data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing trend direction: {str(e)}")
            return 'unknown', 0.0

    @staticmethod
    def calculate_trend_momentum(prices: Series, window: int = 14) -> float:
        """Calculate trend momentum using Rate of Change (ROC).
        
        Args:
            prices: Series of price data
            window: Period for momentum calculation (default: 14)
            
        Returns:
            float: Momentum score (-100 to 100)
            
        Raises:
            InvalidDataError: If prices data is invalid
            
        Example:
            >>> prices = pd.Series([10, 11, 12, 13, 14])
            >>> TrendAnalysis.calculate_trend_momentum(prices, window=5)
            40.0
        """
        try:
            if not isinstance(prices, Series):
                raise InvalidDataError("prices must be a pandas Series")
                
            if len(prices) < window:
                raise InvalidDataError(f"Insufficient data: need at least {window} periods")
                
            if prices.isna().any():
                raise InvalidDataError("prices contains NaN values")
            
            # Calculate Rate of Change
            momentum = ((prices.iloc[-1] - prices.iloc[-window]) / 
                       prices.iloc[-window] * 100)
            
            logger.debug(f"Trend momentum: {momentum:.2f}")
            return momentum
            
        except InvalidDataError as e:
            logger.error(f"Invalid price data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error calculating trend momentum: {str(e)}")
            return 0.0
