CONFIG = {
    'use_testnet': False,
    'symbol': 'ETH/USDT',
    'risk_percentage': 2.0,
    'min_balance': 10.0,  # Memastikan likuiditas yang cukup
    'max_daily_trades': 10,  # Setel lebih rendah untuk lebih terukur
    'max_daily_loss_percent': 5,
    'max_drawdown_percent': 10,
    'fee_rate': 0.0004,
    'timeframe': '5m',  # Meningkatkan dari 1m ke 5m untuk stabilitas sinyal
    'ema_short_period': 10,
    'ema_long_period': 30,  # Memperbaiki periode EMA untuk pembacaan tren lebih baik

    # Risk Management
    'stop_loss_percent': 4.0,  # Sedikit lebih ketat
    'profit_target_percent': 1.0,
    'max_positions': 3,
    'max_position_size': 0.1,

    # Take Profit Settings
    'trailing_stop_percent': 0.01,  # Sedikit lepas untuk memanfaatkan tren
    'partial_tp_1': 0.5,
    'partial_tp_2': 0.5,
    'tp1_target': 0.01,
    'tp2_target': 0.02,
    'initial_profit_for_trailing_stop': 0.015,
    'trailing_distance_pct': 0.007,

    # Market Analysis
    'leverage': 1,  # Menghindari leverage sebagai perdagangan spot
    'funding_rate_threshold': 0.0030,
    'atr_period': 14,
    'max_atr_threshold': 1.0,  # Melonggarkan batasan untuk adaptasi lebih

    # Volatility Settings
    'max_volatility_threshold': 0.05,
    'volume_ma_period': 10,
    'min_volume_multiplier': 1.0,
    'min_volume_usdt': 2000,  # Meningkatkan untuk memastikan likuiditas

    # Price Action
    'price_change_threshold': 0.0025,
    're_evaluate_position_threshold': 0.015,
    'min_risk_reward_ratio': 1.8,
    'max_spread_percent': 0.1,

    # Technical Indicators
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'adx_threshold': 25,
    'momentum_threshold': 0,
    'trend_strength_threshold': 0.6,
    'confirmation_candles': 3,
    'volume_increase_threshold': 1.5,

    'min_candles_required': 100,
    'cleanup_timeout': 30,
    'position_timeout': 300,
    'max_slippage': 0.002,

    'position_update_interval': 60,
    'trailing_activation_threshold': 0.005,
    'position_max_duration': 24 * 60 * 60,
    'min_notional_value': 10,
    'max_notional_value': 1000,

    'max_consecutive_losses': 3,
    'daily_profit_target': 5.0,
    'market_impact_threshold': 0.02,
    'position_sizing_atr_multiplier': 1.5,
    'max_open_orders': 5,
    'min_liquidity_ratio': 0.1,

    'min_trade_interval': 60,
    'max_position_loss_percent': 2.0,
    'min_profit_threshold': 0.2,
    'price_precision': 8,
    'amount_precision': 8,
    'order_timeout': 60,
}