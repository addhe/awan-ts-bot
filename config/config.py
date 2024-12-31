CONFIG = {
    'use_testnet': False,
    'symbol': 'ETH/USDT',
    'risk_percentage': 1.0,  # Adjust risk for higher trade frequency
    'min_balance': 10.0,
    'max_daily_trades': 50,  # Increase for faster timeframes
    'max_daily_loss_percent': 5,
    'max_drawdown_percent': 10,
    'fee_rate': 0.0004,
    'timeframe': '1m',  # Set to 1-minute timeframe

    # Moving Average Settings
    'ema_short_period': 5,
    'ema_long_period': 15,

    # Risk Management
    'stop_loss_percent': 2.0,  # More conservative for quick trades
    'profit_target_percent': 0.8,
    'max_positions': 3,
    'max_position_size': 0.1,

    # Market Analysis
    'atr_period': 14,
    'max_atr_threshold': 1.0,

    # Volume Settings
    'volume_ma_period': 10,
    'min_volume_usdt': 2000,
    'min_volume_multiplier': 1.5,  # Changed from volume_multiplier

    # Spread and Market Impact
    'max_spread_percent': 0.5,
    'market_impact_threshold': 0.1,

    # Volatility Settings
    'max_volatility_threshold': 0.05,
    'high_volatility_threshold': 0.03,
    'low_volatility_threshold': 0.005,
    'high_volatility_adjustment': 0.2,
    'low_volatility_adjustment': 0.1,

    # Technical Indicators
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'adx_threshold': 20,
    'momentum_threshold': 0,
    'trend_strength_threshold': 0.5,

    # Position Management
    'min_candles_required': 100,
    'cleanup_timeout': 30,
    'position_timeout': 300,
    'position_max_duration': 3600,
    'position_update_interval': 30,
    'max_slippage': 0.002,

    # Take Profit and Stop Loss
    'min_risk_reward_ratio': 1.5,
    'initial_profit_for_trailing_stop': 0.005,
    'trailing_distance_pct': 0.002,
    'partial_tp_1': 0.3,
    'partial_tp_2': 0.3,
    'tp1_target': 0.005,
    'tp2_target': 0.01,

    # Trading Limits
    'daily_profit_target': 2.0,
    'min_trade_interval': 30,
    'max_position_loss_percent': 2.0,
    'order_timeout': 30,
    'max_open_orders': 5,
    'max_consecutive_losses': 3,
    'min_liquidity_ratio': 0.1,
    'min_profit_threshold': 0.2,
    'price_precision': 8,
    'amount_precision': 8,

    # Additional Settings
    'funding_rate_threshold': 0.01,
    'position_sizing_atr_multiplier': 1.5,
    'price_change_threshold': 0.02,
}