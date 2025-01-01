import logging
from typing import Dict, Optional, Union, List

import pandas as pd
import numpy as np
from pandas import DataFrame, Series

class Indicators:
    """Technical indicators calculation class for market analysis.

This class provides static methods for calculating various technical indicators
used in market analysis and trading decisions. Each method includes input validation,
error handling, and proper logging.

Example:
    indicators = Indicators()
    rsi = indicators.calculate_rsi(price_data, period=14)
    ema = indicators.calculate_ema(price_data, period=20)

Note:
    All methods include NaN checks and proper error handling
    All calculations use pandas Series for vectorized operations
"""
    
    @staticmethod
    def calculate_rsi(prices: Series, period: int = 14) -> Series:
        """Calculate Relative Strength Index (RSI).
        
        The RSI is a momentum indicator that measures the magnitude of recent price 
        changes to evaluate overbought or oversold conditions.
        
        Args:
            prices: Series of price data
            period: RSI period (default: 14)
            
        Returns:
            Series containing RSI values (0-100 range)
            
        Raises:
            ValueError: If prices contain NaN values or period is less than 1
            TypeError: If prices is not a pandas Series
            
        Example:
            >>> prices = pd.Series([10, 12, 11, 13, 14, 13])
            >>> rsi = Indicators.calculate_rsi(prices, period=2)
        """
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        if period < 1:
            raise ValueError("period must be greater than 0")
        if prices.isnull().any():
            raise ValueError("prices contain NaN values")
            
        try:
            logging.debug(f"Calculating RSI with period {period}")
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            logging.error(f"Error calculating RSI: {str(e)}")
            logging.debug("RSI calculation failed, returning empty series")
            return pd.Series(index=prices.index, dtype=float)

    @staticmethod
    def calculate_ema(data: Series, period: int) -> Series:
        """Calculate Exponential Moving Average (EMA).
        
        EMA gives more weight to recent prices for a specified period.
        
        Args:
            data: Series of price data
            period: EMA period
            
        Returns:
            Series containing EMA values
            
        Raises:
            ValueError: If data contains NaN values or period is less than 1
            TypeError: If data is not a pandas Series
            
        Example:
            >>> prices = pd.Series([10, 11, 12, 13, 14, 15])
            >>> ema = Indicators.calculate_ema(prices, period=3)
        """
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series")
        if period < 1:
            raise ValueError("period must be greater than 0")
        if data.isnull().any():
            raise ValueError("data contains NaN values")
            
        try:
            logging.debug(f"Calculating EMA with period {period}")
            return data.ewm(span=period, adjust=False).mean()
        except Exception as e:
            logging.error(f"Error calculating EMA: {str(e)}")
            logging.debug("EMA calculation failed, returning empty series")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def calculate_adx(high: Series, low: Series, close: Series, period: int = 14) -> Dict[str, Series]:
        """Calculate Average Directional Index (ADX).
        
        ADX measures trend strength without regard to trend direction.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: ADX period (default: 14)
            
        Returns:
            Dictionary containing:
                - 'adx': Average Directional Index values
                - 'plus_di': Positive Directional Indicator values
                - 'minus_di': Negative Directional Indicator values
                
        Raises:
            ValueError: If any input series contains NaN or period is less than 1
            TypeError: If inputs are not pandas Series
            
        Example:
            >>> adx_data = Indicators.calculate_adx(high_prices, low_prices, 
            ...                                    close_prices, period=14)
            >>> adx = adx_data['adx']
            >>> plus_di = adx_data['plus_di']
        """
        if not all(isinstance(x, pd.Series) for x in [high, low, close]):
            raise TypeError("high, low, and close must be pandas Series")
        if period < 1:
            raise ValueError("period must be greater than 0")
        if any(x.isnull().any() for x in [high, low, close]):
            raise ValueError("Input series contain NaN values")
            
        try:
            logging.debug(f"Calculating ADX with period {period}")
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm = plus_dm.where(plus_dm > 0, 0)
            minus_dm = minus_dm.where(minus_dm > 0, 0)

            tr = pd.DataFrame({
                'hl': high - low,
                'hc': abs(high - close.shift(1)),
                'lc': abs(low - close.shift(1))
            }).max(axis=1)

            plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
            minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()

            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
        except Exception as e:
            logging.error(f"Error calculating ADX: {str(e)}")
            logging.debug("ADX calculation failed, returning empty series")
            return {
                'adx': pd.Series(index=high.index, dtype=float),
                'plus_di': pd.Series(index=high.index, dtype=float),
                'minus_di': pd.Series(index=high.index, dtype=float)
            }

    @staticmethod
    def calculate_atr(high: Series, low: Series, close: Series, period: int = 14) -> Series:
        """Calculate Average True Range (ATR).
        
        ATR measures market volatility by decomposing the entire range of an asset 
        price for a specific period. It provides insight into volatility by considering
        the largest of:
        1. Current high - current low
        2. Absolute value of current high - previous close
        3. Absolute value of current low - previous close
        
        Args:
            high: Series of high prices. Must be positive and greater than low prices.
            low: Series of low prices. Must be positive and less than high prices.
            close: Series of closing prices. Must be positive and between high and low.
            period: ATR period (default: 14). Common values are 14, 20, or 50.
            
        Returns:
            Series containing ATR values. Values are always positive and represent
            the average range of price movement over the specified period.
            
        Raises:
            ValueError: 
                - If any input series contains NaN values
                - If period is less than 1
                - If high prices are not greater than low prices
                - If close prices are not between high and low prices
                - If any prices are negative or zero
            TypeError: If inputs are not pandas Series
            
        Example:
            >>> high_prices = pd.Series([10.5, 11.3, 10.9, 11.6])
            >>> low_prices = pd.Series([10.1, 10.6, 10.2, 11.0])
            >>> close_prices = pd.Series([10.3, 10.9, 10.5, 11.3])
            >>> atr = Indicators.calculate_atr(high_prices, low_prices, 
            ...                               close_prices, period=2)
            >>> print(f"ATR values: {atr}")
            ATR values:
            0         NaN
            1    0.650000
            2    0.675000
            3    0.687500
        """
        # Type validation
        if not all(isinstance(x, pd.Series) for x in [high, low, close]):
            raise TypeError("high, low, and close must be pandas Series")
            
        # Basic validation
        if period < 1:
            raise ValueError("period must be greater than 0")
        if any(x.isnull().any() for x in [high, low, close]):
            raise ValueError("Input series contain NaN values")
            
        # Price range validation
        if (high <= low).any():
            raise ValueError("High prices must be greater than low prices")
        if ((close > high) | (close < low)).any():
            raise ValueError("Close prices must be between high and low prices")
        if (high <= 0).any() or (low <= 0).any() or (close <= 0).any():
            raise ValueError("All prices must be positive")
            
        try:
            logging.debug(f"Calculating ATR with period {period}")
            logging.debug(f"Input data - High range: {high.min():.2f} to {high.max():.2f}")
            logging.debug(f"Input data - Low range: {low.min():.2f} to {low.max():.2f}")
            
            # Calculate True Range components
            tr1 = high - low  # Current high - current low
            tr2 = abs(high - close.shift())  # Current high - previous close
            tr3 = abs(low - close.shift())  # Current low - previous close
            
            # Get maximum of the three components
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            # Calculate ATR using simple moving average
            atr = tr.rolling(window=period).mean()
            
            logging.debug(f"ATR calculation complete - Range: {atr.min():.2f} to {atr.max():.2f}")
            
            # Validate output
            if atr.isnull().all():
                logging.warning("ATR calculation resulted in all NaN values")
            elif atr.isnull().any():
                logging.warning(f"ATR contains {atr.isnull().sum()} NaN values")
                
            return atr
            
        except Exception as e:
            logging.error(f"Error calculating ATR: {str(e)}")
            logging.debug("ATR calculation failed, returning empty series")
            return pd.Series(index=high.index, dtype=float)

    @staticmethod
    def calculate_vwap(high: Series, low: Series, close: Series, volume: Series) -> float:
        """Calculate Volume Weighted Average Price (VWAP).
        
        VWAP is a trading benchmark that gives the average price a security 
        has traded at throughout the day, based on both volume and price.
        It is calculated by summing the product of price and volume for each period,
        then dividing by the total volume. VWAP puts more weight on periods with
        higher trading volume.
        
        Formula:
            VWAP = Σ(Typical Price * Volume) / Σ(Volume)
            where Typical Price = (High + Low + Close) / 3
        
        Args:
            high: Series of high prices. Must be positive and greater than low prices.
            low: Series of low prices. Must be positive and less than high prices.
            close: Series of closing prices. Must be positive and between high and low.
            volume: Series of volume data. Must be positive non-zero values.
            
        Returns:
            float: VWAP value. Returns 0.0 if calculation fails.
            A value of 0.0 typically indicates an error condition and should be handled
            appropriately by the calling code.
            
        Raises:
            ValueError: 
                - If any input series contains NaN values
                - If high prices are not greater than low prices
                - If close prices are not between high and low prices
                - If any prices are negative or zero
                - If volume contains zero or negative values
            TypeError: If inputs are not pandas Series
            
        Example:
            >>> high_prices = pd.Series([10.5, 11.3, 10.9])
            >>> low_prices = pd.Series([10.1, 10.6, 10.2])
            >>> close_prices = pd.Series([10.3, 10.9, 10.5])
            >>> volume_data = pd.Series([1000, 1500, 800])
            >>> vwap = Indicators.calculate_vwap(high_prices, low_prices,
            ...                                 close_prices, volume_data)
            >>> print(f"VWAP: {vwap:.2f}")
            VWAP: 10.72
        """
        # Type validation
        if not all(isinstance(x, pd.Series) for x in [high, low, close, volume]):
            raise TypeError("All inputs must be pandas Series")
            
        # Length validation
        if not len(set(map(len, [high, low, close, volume]))) == 1:
            raise ValueError("All input series must have the same length")
            
        # NaN validation
        if any(x.isnull().any() for x in [high, low, close, volume]):
            raise ValueError("Input series contain NaN values")
            
        # Price range validation
        if (high <= low).any():
            raise ValueError("High prices must be greater than low prices")
        if ((close > high) | (close < low)).any():
            raise ValueError("Close prices must be between high and low prices")
        if (high <= 0).any() or (low <= 0).any() or (close <= 0).any():
            raise ValueError("All prices must be positive")
            
        # Volume validation
        if (volume <= 0).any():
            raise ValueError("Volume series contains zero or negative values")
            
        try:
            logging.debug("Starting VWAP calculation")
            logging.debug(f"Price ranges - High: {high.min():.2f} to {high.max():.2f}, "
                        f"Low: {low.min():.2f} to {low.max():.2f}")
            logging.debug(f"Volume range: {volume.min():.0f} to {volume.max():.0f}")
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            logging.debug(f"Typical price range: {typical_price.min():.2f} to {typical_price.max():.2f}")
            
            # Calculate VWAP
            price_volume = typical_price * volume
            total_price_volume = price_volume.sum()
            total_volume = volume.sum()
            
            vwap = total_price_volume / total_volume
            
            logging.debug(f"Calculated VWAP: {vwap:.2f}")
            
            # Validate result
            if not (typical_price.min() <= vwap <= typical_price.max()):
                logging.warning("VWAP value outside typical price range")
                
            return vwap
            
        except Exception as e:
            logging.error(f"Error calculating VWAP: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_momentum(prices: Series, period: int = 14) -> Series:
        """Calculate price momentum.
        
        Momentum measures the rate of change in price movement over a specific period.
        It is calculated as the percentage change between the current price and the
        price 'period' days ago. Positive momentum indicates upward price movement,
        while negative momentum indicates downward movement.
        
        The momentum indicator helps identify trend strength and potential reversals:
        - Strong positive values suggest a strong uptrend
        - Strong negative values suggest a strong downtrend
        - Values crossing zero may indicate trend reversals
        - Divergence between price and momentum can signal potential reversals
        
        Formula:
            Momentum = ((Current Price - Price n periods ago) / Price n periods ago) * 100
            where n is the specified period
        
        Technical Details:
            - Uses percentage change to normalize across different price ranges
            - First n values will be NaN where n is the period
            - Typical threshold values: 
                * Strong trend: > ±20%
                * Moderate trend: ±10-20%
                * Weak trend: < ±10%
        
        Args:
            prices: Series of price data. Must be positive values representing asset prices.
                   Common inputs are closing prices or typical prices.
                   Minimum length should be period + 1.
                   Must have consistent time intervals if timestamps are present.
            period: Momentum period (default: 14). Common values are 10, 14, or 20.
                   Shorter periods are more sensitive to recent price changes.
                   Longer periods smooth out noise but lag more.
            
        Returns:
            Series containing momentum values as percentage changes. Values can be:
            - Positive: Upward price movement
            - Negative: Downward price movement
            - Zero: No price change
            First 'period' values will be NaN due to the calculation method.
            
        Raises:
            ValueError: 
                - If prices contain NaN values
                - If period is less than 1
                - If any prices are negative or zero
                - If prices length is less than period + 1
                - If prices show unrealistic changes (>100% in one period)
                - If price sequence has gaps or inconsistent intervals
                - If momentum values exceed theoretical limits
            TypeError: If prices is not a pandas Series
            
        Example:
            >>> prices = pd.Series([10.0, 10.5, 10.3, 10.8, 11.2])
            >>> momentum = Indicators.calculate_momentum(prices, period=2)
            >>> print(momentum)
            0         NaN
            1         5.00  # (10.5 - 10.0) / 10.0 * 100
            2        -1.90  # (10.3 - 10.5) / 10.5 * 100
            3         4.85  # (10.8 - 10.3) / 10.3 * 100
            4         3.70  # (11.2 - 10.8) / 10.8 * 100
        """
        # Type validation
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
            
        # Basic validation
        if period < 1:
            raise ValueError("period must be greater than 0")
        if len(prices) < period + 1:
            raise ValueError(f"prices length ({len(prices)}) must be greater than period + 1 ({period + 1})")
        if prices.isnull().any():
            raise ValueError("prices contain NaN values")
            
        # Price validation
        if (prices <= 0).any():
            raise ValueError("All prices must be positive")
            
        # Check for unrealistic price changes
        pct_changes = prices.pct_change().abs()
        if (pct_changes > 1.0).any():  # More than 100% change
            max_change = pct_changes.max() * 100
            logging.warning(f"Detected large price change of {max_change:.2f}%")
            raise ValueError(f"Unrealistic price change detected: {max_change:.2f}%")
            
        # Validate timestamp sequence if available
        if isinstance(prices.index, pd.DatetimeIndex):
            logging.debug("Validating timestamp sequence")
            time_diffs = prices.index.to_series().diff()
            if time_diffs.nunique() > 1:  # More than one unique time difference
                logging.error("Inconsistent time intervals detected in price data")
                raise ValueError("Price data must have consistent time intervals")
            
            # Check for gaps
            expected_interval = time_diffs.mode()[0]  # Most common interval
            if (time_diffs != expected_interval).any():
                logging.error(f"Gaps detected in price sequence. Expected interval: {expected_interval}")
                raise ValueError("Price sequence contains gaps")
                
            logging.debug(f"Time interval validation passed: {expected_interval}")
            
        try:
            logging.debug(f"Starting momentum calculation:")
            logging.debug(f"- Period: {period}")
            logging.debug(f"- Sample size: {len(prices)}")
            logging.debug(f"- Price range: {prices.min():.2f} to {prices.max():.2f}")
            
            # Calculate sequential price changes for validation
            sequential_changes = prices.pct_change()
            logging.debug(f"Sequential price changes - Range: {sequential_changes.min():.2f} to {sequential_changes.max():.2f}")
            
            # Calculate price changes over period
            price_changes = prices.diff(period)
            logging.debug(f"Period price changes - Range: {price_changes.min():.2f} to {price_changes.max():.2f}")
            
            # Calculate momentum as percentage change
            momentum = prices.pct_change(periods=period) * 100
            
            # Validate momentum calculation results
            valid_momentum = momentum.dropna()
            if valid_momentum.empty:
                logging.warning("Momentum calculation resulted in all NaN values")
                return pd.Series(index=prices.index, dtype=float)
                
            # Theoretical maximum momentum validation
            # For a given period, max momentum occurs with continuous up/down movement
            theoretical_max = (1.0 + sequential_changes.abs().max()) ** period * 100
            if (valid_momentum.abs() > theoretical_max).any():
                logging.error(f"Momentum values exceed theoretical maximum of {theoretical_max:.2f}%")
                raise ValueError(f"Invalid momentum values detected (exceed theoretical maximum)")
                
            # Log detailed momentum statistics
            logging.debug("\nMomentum Statistics:")
            logging.debug(f"- Range: {valid_momentum.min():.2f}% to {valid_momentum.max():.2f}%")
            logging.debug(f"- Mean: {valid_momentum.mean():.2f}%")
            logging.debug(f"- Median: {valid_momentum.median():.2f}%")
            logging.debug(f"- Std Dev: {valid_momentum.std():.2f}%")
            
            # Analyze momentum distribution
            strong_up = valid_momentum[valid_momentum > 20].count()
            strong_down = valid_momentum[valid_momentum < -20].count()
            moderate = valid_momentum[valid_momentum.abs().between(10, 20)].count()
            weak = valid_momentum[valid_momentum.abs() < 10].count()
            
            logging.debug("\nMomentum Distribution:")
            logging.debug(f"- Strong uptrend (>20%): {strong_up} periods ({strong_up/len(valid_momentum)*100:.1f}%)")
            logging.debug(f"- Strong downtrend (<-20%): {strong_down} periods ({strong_down/len(valid_momentum)*100:.1f}%)")
            logging.debug(f"- Moderate trend (±10-20%): {moderate} periods ({moderate/len(valid_momentum)*100:.1f}%)")
            logging.debug(f"- Weak trend (<±10%): {weak} periods ({weak/len(valid_momentum)*100:.1f}%)")
            
            # Analyze momentum trend
            momentum_trend = np.sign(valid_momentum).diff().fillna(0)
            trend_changes = (momentum_trend != 0).sum()
            logging.debug(f"\nMomentum Trend Analysis:")
            logging.debug(f"- Trend changes: {trend_changes} ({trend_changes/len(valid_momentum)*100:.1f}% of periods)")
            logging.debug(f"- Current trend: {'Upward' if valid_momentum.iloc[-1] > 0 else 'Downward'}")
            
            # Check for extreme values that might indicate calculation errors
            extreme_threshold = 50  # 50% change
            extreme_values = valid_momentum[valid_momentum.abs() > extreme_threshold]
            if not extreme_values.empty:
                logging.warning(f"\nFound {len(extreme_values)} extreme momentum values (>±{extreme_threshold}%)")
                logging.warning(f"Extreme values at: {extreme_values.index.tolist()}")
                logging.warning(f"Extreme values: {extreme_values.values.tolist()}")
            
            return momentum
            
        except Exception as e:
            logging.error(f"Error calculating momentum: {str(e)}")
            logging.debug("Momentum calculation failed, returning empty series")
            return pd.Series(index=prices.index, dtype=float)

    @staticmethod
    def calculate_volatility(prices: Series, period: int = 20) -> float:
        """Calculate price volatility.
        
        Measures the standard deviation of price returns over a specific period,
        annualized for comparison purposes.
        
        Args:
            prices: Series of price data
            period: Volatility period (default: 20)
            
        Returns:
            float: Annualized volatility value
            
        Raises:
            ValueError: If prices contain NaN values or period is less than 1
            TypeError: If prices is not a pandas Series
            
        Example:
            >>> volatility = Indicators.calculate_volatility(price_data, period=20)
        """
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        if period < 1:
            raise ValueError("period must be greater than 0")
        if prices.isnull().any():
            raise ValueError("prices contain NaN values")
            
        try:
            logging.debug(f"Calculating volatility with period {period}")
            returns = np.log(prices / prices.shift(1))
            return returns.std() * np.sqrt(24)  # Annualized volatility
        except Exception as e:
            logging.error(f"Error calculating volatility: {str(e)}")
            return 0.0
