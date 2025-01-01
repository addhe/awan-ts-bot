import logging
from typing import Dict, Any, Union, Tuple

def adjust_trade_amount(
    amount: float,
    symbol: str,
    exchange: Any,
    side: str = "buy"
) -> float:
    """
    Adjust trade amount to comply with exchange limits and precision requirements.
    
    This function ensures the trade amount meets exchange requirements by:
    1. Checking against exchange minimum/maximum trade amounts
    2. Adjusting decimal precision to match exchange requirements
    3. Accounting for exchange fees and price impact
    
    Args:
        amount: Original trade amount in base currency
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        exchange: Exchange instance with market info
        side: Trade direction ('buy' or 'sell', default: 'buy')
        
    Returns:
        float: Adjusted trade amount that complies with exchange requirements
        
    Raises:
        ValueError: If amount is negative or zero
        ValueError: If side is not 'buy' or 'sell'
        ValueError: If symbol format is invalid
        RuntimeError: If exchange market info is unavailable
        
    Example:
        >>> exchange = ccxt.binance()
        >>> adjusted = adjust_trade_amount(0.12345, 'BTC/USDT', exchange)
        >>> print(f"Adjusted amount: {adjusted}")
    """
    if amount <= 0:
        raise ValueError(f"Invalid amount: {amount}. Must be positive")
        
    if side not in ["buy", "sell"]:
        raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
        
    if '/' not in symbol:
        raise ValueError(f"Invalid symbol format: {symbol}. Must be in format 'BTC/USDT'")
    
    try:
        # Get market info for symbol
        market = exchange.market(symbol)
        if not market:
            raise RuntimeError(f"No market info available for {symbol}")
            
        # Extract limits and precision
        limits = market.get('limits', {})
        precision = market.get('precision', {})
        
        # Get amount limits
        min_amount = float(limits.get('amount', {}).get('min', 0))
        max_amount = float(limits.get('amount', {}).get('max', float('inf')))
        
        # Get amount precision (decimal places)
        amount_precision = precision.get('amount', 8)
        
        # Adjust amount to precision
        adjusted = round(amount, amount_precision)
        
        # Enforce minimum amount
        if adjusted < min_amount:
            logging.warning(
                f"Amount {adjusted} below minimum {min_amount} for {symbol}, "
                f"adjusting to minimum"
            )
            adjusted = min_amount
            
        # Enforce maximum amount
        if adjusted > max_amount:
            logging.warning(
                f"Amount {adjusted} above maximum {max_amount} for {symbol}, "
                f"adjusting to maximum"
            )
            adjusted = max_amount
            
        # Account for fees and slippage
        fee_rate = market.get('taker', 0.001)  # Default 0.1% taker fee
        slippage = 0.001  # Estimated 0.1% slippage
        
        if side == "buy":
            # Reduce buy amount to account for fees and slippage
            total_cost = adjusted * (1 + fee_rate + slippage)
            adjusted = round(adjusted * (1 - fee_rate - slippage), amount_precision)
        else:
            # Reduce sell amount to ensure it covers fees
            adjusted = round(adjusted * (1 - fee_rate), amount_precision)
            
        logging.info(
            f"Adjusted {side} amount for {symbol}: "
            f"{amount} -> {adjusted} "
            f"(precision: {amount_precision}, "
            f"min: {min_amount}, max: {max_amount})"
        )
        
        return adjusted
        
    except Exception as e:
        logging.error(f"Error adjusting trade amount: {str(e)}")
        raise
