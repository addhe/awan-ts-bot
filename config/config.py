CONFIG = {
    'use_testnet': False,
    'symbol': 'ETH/USDT',
    'risk_percentage': 2.0,  # Good conservative value
    'min_balance': 5.0,
    'max_daily_trades': 20,
    'max_daily_loss_percent': 5,
    'max_drawdown_percent': 10,
    'fee_rate': 0.0004,
    'timeframe': '1m',
    'ema_short_period': 5,
    'ema_long_period': 15,

    # Risk Management
    'stop_loss_percent': 5.0,
    'profit_target_percent': 0.5,
    'max_positions': 3,
    'max_position_size': 0.1,
    'max_loss_per_position': 2,

    # Take Profit Settings
    'trailing_stop_percent': 0.005,
    'partial_tp_1': 0.5,
    'partial_tp_2': 0.5,
    'tp1_target': 0.008,
    'tp2_target': 0.015,
    'initial_profit_for_trailing_stop': 0.01,
    'trailing_distance_pct': 0.005,

    # Market Analysis
    'leverage': 2,
    'funding_rate_threshold': 0.0030,
    'atr_period': 10,
    'max_atr_threshold': 0.8,
    'vwap_period': 20,
    'volume_ma_period': 20,
    'min_volume_multiplier': 0.5,
    'min_volume_usdt': 1000,

    # Volatility Settings
    'max_volatility_threshold': 0.04,
    'high_volatility_threshold': 0.08,
    'low_volatility_threshold': 0.01,
    'high_volatility_adjustment': 0.3,
    'low_volatility_adjustment': 0.2,
    're_evaluate_volatility_threshold': 0.06,

    # Price Action
    'price_change_threshold': 0.002,
    're_evaluate_position_threshold': 0.02,
    'min_risk_reward_ratio': 1.5,
    'max_spread_percent': 0.1,  # Using the more conservative value

    # Technical Indicators
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'adx_threshold': 25,
    'momentum_threshold': 0,
    'trend_strength_threshold': 0.6,  # Using the more meaningful value
    'confirmation_candles': 3,
    'volume_increase_threshold': 1.5,

    'min_candles_required': 100,  # Minimum candles needed for analysis
    'cleanup_timeout': 30,        # Seconds to wait for cleanup
    'position_timeout': 300,      # Maximum time to wait for position execution
    'max_slippage': 0.001,       # Maximum allowed slippage (0.1%)

    'position_update_interval': 60,  # Seconds between position updates
    'trailing_activation_threshold': 0.005,  # 0.5% profit to activate trailing
    'position_max_duration': 24 * 60 * 60,  # Maximum position duration in seconds
    'min_notional_value': 10,  # Minimum trade value in USDT
    'max_notional_value': 1000,  # Maximum trade value in USDT

    'max_consecutive_losses': 3,  # Maximum consecutive losing trades
    'daily_profit_target': 5.0,   # Daily profit target in percentage
    'market_impact_threshold': 0.02,  # Maximum allowed market impact
    'position_sizing_atr_multiplier': 1.5,  # Position sizing based on ATR
    'max_open_orders': 5,  # Maximum number of open orders
    'min_liquidity_ratio': 0.1,  # Minimum ratio of order size to average volume

    'min_trade_interval': 60,      # Minimum time between trades (seconds)
    'max_position_loss_percent': 2.0,  # Maximum loss per position in percentage
    'min_profit_threshold': 0.2,   # Minimum profit threshold for trades
    'price_precision': 8,          # Price precision for orders
    'amount_precision': 8,         # Amount precision for orders
    'order_timeout': 60,           # Timeout for order execution in seconds
}