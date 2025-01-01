import logging
from datetime import datetime
from typing import Dict, Optional, Union, List, Tuple

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from config.config import CONFIG

from .indicators import Indicators
from .market_validation import MarketValidation

# Custom types
MarketData = DataFrame
AnalysisResult = Dict[str, Union[float, Dict[str, float]]]
TrendAnalysis = Dict[str, float]

class MarketAnalyzer:
    def __init__(self, exchange) -> None:
        """Initialize MarketAnalyzer with exchange instance.
        
        Args:
            exchange: Exchange instance for market operations
        """
        self.exchange = exchange
        self.market_data: Optional[MarketData] = None
        self.market_validator = MarketValidation(exchange)

    def set_market_data(self, market_data: MarketData) -> None:
        """Set market data for analysis
        
        Args:
            market_data: DataFrame containing OHLCV data
        """
        self.market_data = market_data

    def analyze_price_trend(self, market_data: MarketData, lookback_period: int = 20) -> Optional[TrendAnalysis]:
        """Analyze price trend using multiple indicators
        
        Args:
            market_data: DataFrame containing OHLCV data
            lookback_period: Period for momentum calculation
            
        Returns:
            Dictionary containing RSI, momentum, ADX and trend strength values
        """
        try:
            close_prices = market_data['close']
            high_prices = market_data['high']
            low_prices = market_data['low']

            # Calculate indicators using the Indicators class
            rsi = Indicators.calculate_rsi(close_prices)
            momentum = Indicators.calculate_momentum(close_prices, lookback_period)
            adx_data = Indicators.calculate_adx(high_prices, low_prices, close_prices)

            return {
                'rsi': rsi.iloc[-1],
                'momentum': momentum.iloc[-1],
                'adx': adx_data['adx'].iloc[-1],
                'trend_strength': (adx_data['plus_di'].iloc[-1] - adx_data['minus_di'].iloc[-1])
            }

        except Exception as e:
            logging.error(f"Error in trend analysis: {str(e)}")
            return None

    def calculate_trend_strength(self, market_data: MarketData) -> float:
        """Calculate weighted trend strength using multiple indicators
        
        This method combines multiple technical indicators to produce a comprehensive
        trend strength value. Components include:
        - EMA divergence (35% weight)
        - Price momentum (25% weight)
        - Volume trend (20% weight)
        - Price direction (20% weight)
        The final value is adjusted for volatility.
        
        Args:
            market_data: DataFrame containing OHLCV data
            
        Returns:
            float: Weighted trend strength value between 0 and 1
        """
        try:
            close_prices = market_data['close']
            
            # Calculate EMAs using Indicators class
            ema_short = Indicators.calculate_ema(close_prices, CONFIG['ema_short_period'])
            ema_long = Indicators.calculate_ema(close_prices, CONFIG['ema_long_period'])
            
            # Calculate directional movement
            price_change = Indicators.calculate_momentum(close_prices, CONFIG['trend_lookback'])
            direction = np.sign(price_change.mean())
            
            # Calculate momentum using Indicators class
            momentum = Indicators.calculate_momentum(close_prices, 5).mean()
            
            # Calculate volume trend
            volume = market_data['volume']
            volume_sma = volume.rolling(window=10).mean()
            volume_trend = (volume.iloc[-1] / volume_sma.iloc[-1]) - 1
            
            # Calculate base trend strength from EMA divergence
            basic_trend_strength = abs(ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1]
            
            # Weight the components
            weighted_strength = (
                basic_trend_strength * 0.35 +  # Base trend
                abs(momentum) * 0.25 +         # Momentum
                abs(volume_trend) * 0.20 +     # Volume trend
                direction * 0.20               # Direction
            )
            
            # Apply volatility adjustment using Indicators class
            volatility = Indicators.calculate_volatility(close_prices)
            if volatility > 0:
                volatility_factor = 1 - (volatility / CONFIG['max_volatility_threshold'])
                weighted_strength *= max(0.5, volatility_factor)  # Cap minimum at 0.5
            
            logging.debug(f"""
            Trend Components:
            Basic: {basic_trend_strength:.6f}
            Momentum: {momentum:.6f}
            Volume: {volume_trend:.6f}
            Direction: {direction}
            Volatility Adj: {volatility_factor if volatility > 0 else 'N/A'}
            Final: {weighted_strength:.6f}
            """)
            
            return float(weighted_strength)
            
        except Exception as e:
            logging.error(f"Error calculating trend strength: {str(e)}")
            return 0.0

    def calculate_volatility(self, market_data: MarketData, lookback_period: int = 20) -> Optional[float]:
        """Calculate current market volatility
        
        Args:
            market_data: DataFrame containing OHLCV data
            lookback_period: Period for volatility calculation (default: 20)
            
        Returns:
            float: Annualized volatility value, or None if calculation fails
        """
        try:
            return Indicators.calculate_volatility(market_data['close'], lookback_period)
        except Exception as e:
            logging.error(f"Error calculating volatility: {str(e)}")
            return None

    def calculate_vwap(self, market_data: MarketData) -> Optional[float]:
        """Calculate Volume Weighted Average Price
        
        Args:
            market_data: DataFrame containing OHLCV data
            
        Returns:
            float: VWAP value, or None if calculation fails
        """
        try:
            return Indicators.calculate_vwap(
                market_data['high'],
                market_data['low'],
                market_data['close'],
                market_data['volume']
            )
        except Exception as e:
            logging.error(f"Error calculating VWAP: {str(e)}")
            return None

    def calculate_atr(self, market_data: MarketData, period: int = 14) -> float:
        """Calculate Average True Range
        
        Args:
            market_data: DataFrame containing OHLCV data
            period: ATR period (default: 14)
            
        Returns:
            float: ATR value, returns infinity if calculation fails
        """
        try:
            atr_series = Indicators.calculate_atr(
                market_data['high'],
                market_data['low'],
                market_data['close'],
                period
            )
            return atr_series.iloc[-1]
        except Exception as e:
            logging.error(f"Error calculating ATR: {str(e)}")
            return float('inf')

    def _calculate_ema_data(self, market_data: MarketData) -> Tuple[Optional[Series], Optional[Series]]:
        """Calculate short and long EMAs for the given market data.
        
        Args:
            market_data: DataFrame containing OHLCV market data
            
        Returns:
            Tuple containing short and long EMA Series, or (None, None) if calculation fails
        """
        try:
            close_prices = market_data['close']
            ema_short = Indicators.calculate_ema(close_prices, CONFIG['ema_short_period'])
            ema_long = Indicators.calculate_ema(close_prices, CONFIG['ema_long_period'])
            return ema_short, ema_long
        except Exception as e:
            logging.error(f"Error calculating EMAs: {str(e)}")
            return None, None

    def perform_technical_analysis(self, market_data: MarketData) -> Optional[AnalysisResult]:
        """Perform comprehensive technical analysis on market data.
        
        Conducts a thorough technical analysis including:
        1. EMA calculations (short and long period)
        2. Trend analysis (RSI, Momentum, ADX)
        3. Market condition validation
        
        Args:
            market_data: DataFrame containing OHLCV market data
            
        Returns:
            Dictionary containing:
            - ema_data: DataFrame with short and long EMAs
            - trend_analysis: Dictionary with RSI, momentum, ADX, and trend strength
            Returns None if any calculations fail or market data is invalid
            
        Raises:
            ValueError: If market data validation fails
        """
        try:
            if not self.validate_market_data(market_data):
                raise ValueError("Invalid market data for technical analysis")

            # Calculate EMAs
            ema_short, ema_long = self._calculate_ema_data(market_data)
            if None in (ema_short, ema_long):
                logging.error("Failed to calculate EMA values")
                return None
                
            # Add EMAs to market data
            market_data_with_ema = market_data.copy()
            market_data_with_ema['ema_short'] = ema_short
            market_data_with_ema['ema_long'] = ema_long

            # Get trend analysis
            trend_analysis = self.analyze_price_trend(market_data)
            if trend_analysis is None:
                logging.error("Failed to perform trend analysis")
                return None

            logging.debug(f"""
            Technical Analysis Results:
            EMA Short (latest): {ema_short.iloc[-1]:.2f}
            EMA Long (latest): {ema_long.iloc[-1]:.2f}
            RSI: {trend_analysis['rsi']:.2f}
            ADX: {trend_analysis['adx']:.2f}
            """)

            return {
                'ema_data': market_data_with_ema[['ema_short', 'ema_long']],
                'trend_analysis': trend_analysis
            }

        except ValueError as ve:
            logging.error(f"Validation error in technical analysis: {str(ve)}")
            return None
        except Exception as e:
            logging.error(f"Error performing technical analysis: {str(e)}")
            return None

    def _check_ema_crossover(self, ema_data: DataFrame) -> bool:
        """Check for bullish EMA crossover condition.
        
        Args:
            ema_data: DataFrame containing short and long EMA values
            
        Returns:
            bool: True if bullish EMA crossover occurred
        """
        try:
            return (
                ema_data['ema_short'].iloc[-2] < ema_data['ema_long'].iloc[-2] and
                ema_data['ema_short'].iloc[-1] > ema_data['ema_long'].iloc[-1]
            )
        except Exception as e:
            logging.error(f"Error checking EMA crossover: {str(e)}")
            return False

    def _check_trend_conditions(self, trend_analysis: Dict[str, float]) -> bool:
        """Validate trend conditions against configured thresholds.
        
        Args:
            trend_analysis: Dictionary containing trend indicators (RSI, ADX, momentum, trend_strength)
            
        Returns:
            bool: True if all trend conditions are met
        """
        try:
            return (
                trend_analysis['rsi'] < CONFIG['rsi_overbought'] and
                trend_analysis['adx'] > CONFIG['adx_threshold'] and
                trend_analysis['momentum'] > CONFIG['momentum_threshold'] and
                trend_analysis['trend_strength'] > CONFIG['trend_strength_threshold']
            )
        except Exception as e:
            logging.error(f"Error checking trend conditions: {str(e)}")
            return False

    def should_execute_trade(self, analysis_result: Optional[AnalysisResult], market_data: MarketData) -> bool:
        """Determine if trade should be executed based on technical analysis.
        
        Evaluates multiple conditions for trade execution:
        1. Bullish EMA crossover (short EMA crosses above long EMA)
        2. RSI below overbought level
        3. Strong trend indicated by ADX
        4. Positive momentum above threshold
        5. Strong trend strength
        
        Args:
            analysis_result: Dictionary containing EMA data and trend analysis results
            market_data: DataFrame containing OHLCV market data
            
        Returns:
            bool: True if all trading conditions are met, False otherwise
        """
        try:
            if analysis_result is None:
                return False

            ema_data = analysis_result['ema_data']
            trend_analysis = analysis_result['trend_analysis']
            
            return (
                self._check_ema_crossover(ema_data) and
                self._check_trend_conditions(trend_analysis)
            )

        except Exception as e:
            logging.error(f"Error checking trade conditions: {str(e)}")
            return False

    def validate_market_conditions(self, market_data: MarketData) -> bool:
        """Validate market conditions for spot trading.
        
        Args:
            market_data: DataFrame containing OHLCV data
            
        Returns:
            bool: True if market conditions are valid for trading
        """
        return self.market_validator.validate_market_conditions(market_data)

    def validate_market_data(self, market_data: Optional[MarketData]) -> bool:
        """Validate market data structure and content.
        
        Args:
            market_data: DataFrame containing OHLCV data
            
        Returns:
            bool: True if market data is valid
        """
        return self.market_validator.validate_market_data(market_data)

    def log_market_summary(self, market_data: MarketData) -> None:
        """Log detailed market summary with key technical indicators.
        
        Calculates and logs essential market metrics including:
        - Current price and trading pair
        - Technical indicators (RSI, Volatility)
        - Trend analysis (Strength, Direction)
        - Timestamp of analysis
        
        Args:
            market_data: DataFrame containing OHLCV market data
            
        Raises:
            ValueError: If market data is invalid or calculations fail
        """
        try:
            if not self.validate_market_data(market_data):
                raise ValueError("Invalid market data for summary")

            current_price = market_data['close'].iloc[-1]
            trend_analysis = self.analyze_price_trend(market_data)
            
            if trend_analysis is None:
                raise ValueError("Failed to calculate trend analysis")
                
            volatility = self.calculate_volatility(market_data)
            if volatility is None:
                raise ValueError("Failed to calculate volatility")
                
            trend_strength = self.calculate_trend_strength(market_data)

            market_metrics = {
                'symbol': CONFIG['symbol'],
                'price': f"{current_price:.2f} USDT",
                'rsi': f"{trend_analysis['rsi']:.2f}",
                'volatility': f"{volatility:.4%}",
                'trend_strength': f"{trend_strength:.4f}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            summary = "\n=== Market Summary ===\n" + \
                     "\n".join(f"{k.title()}: {v}" for k, v in market_metrics.items()) + \
                     "\n==================="

            logging.info(summary)

        except Exception as e:
            logging.error(f"Error generating market summary: {str(e)}")
            raise  # Re-raise to allow caller to handle the error
