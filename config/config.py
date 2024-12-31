CONFIG = {
    'use_testnet': False,
    'symbol': 'ETH/USDT',
    'timeframe': '1m',
    'risk_percentage': 0.5,
    'min_balance': 10.0,
    'max_daily_trades': 48,
    'max_daily_loss_percent': 3.0,
    'max_drawdown_percent': 5.0,
    'fee_rate': 0.0004,
    'ema_short_period': 5,
    'ema_long_period': 13,

    'stop_loss_percent': 1.0,
    'profit_target_percent': 0.5,
    'max_positions': 3,
    'max_position_size': 0.05,

    'atr_period': 14,
    'max_atr_threshold': 1.0,

    'volume_ma_period': 10,
    'min_volume_usdt': 1000,
    'min_volume_multiplier': 1.2,
    'max_spread_percent': 0.1,
    'market_impact_threshold': 0.05,

    'max_volatility_threshold': 0.03,
    'high_volatility_threshold': 0.02,
    'low_volatility_threshold': 0.003,
    'high_volatility_adjustment': 0.2,
    'low_volatility_adjustment': 0.1,

    'rsi_period': 14,
    'rsi_overbought': 75,
    'rsi_oversold': 25,
    'adx_threshold': 15,
    'momentum_threshold': 0,
    'trend_strength_threshold': 0.0005,

    'min_candles_required': 100,
    'cleanup_timeout': 30,
    'position_timeout': 300,
    'position_max_duration': 3600,
    'position_update_interval': 30,
    'max_slippage': 0.002,

    'min_risk_reward_ratio': 1.5,
    'initial_profit_for_trailing_stop': 0.005,
    'trailing_distance_pct': 0.001,
    'partial_tp_1': 0.3,
    'partial_tp_2': 0.3,
    'tp1_target': 0.003,
    'tp2_target': 0.005,

    'daily_profit_target': 2.0,
    'min_trade_interval': 180,
    'max_position_loss_percent': 2.0,
    'order_timeout': 30,
    'max_open_orders': 5,
    'max_consecutive_losses': 3,
    'min_liquidity_ratio': 0.05,
    'min_profit_threshold': 0.2,
    'price_precision': 8,
    'amount_precision': 8,

    'funding_rate_threshold': 0.01,
    'position_sizing_atr_multiplier': 1.5,
    'price_change_threshold': 0.02,
}