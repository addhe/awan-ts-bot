import pytest
import ccxt # Import ccxt for its exception classes
from unittest.mock import MagicMock, patch
from app.main import run_trading_cycle
from app.config import CONFIG
from datetime import datetime

@pytest.fixture
def mock_exchange(mocker):
    """Fixture to mock the ccxt exchange object."""
    mock_exc = MagicMock()
    mocker.patch('app.main.initialize_exchange', return_value=mock_exc)
    return mock_exc

@pytest.fixture
def mock_ai_connector(mocker):
    """Fixture to mock the AI signal connector."""
    return mocker.patch('app.main.get_ai_signals')

@pytest.fixture
def mock_redis(mocker):
    """Fixture to mock the Redis client."""
    mock_rc = mocker.patch('app.main.redis_client')
    mock_rc.is_connected.return_value = True
    return mock_rc

@pytest.fixture
def mock_performance_metrics(mocker):
    """Fixture to mock the performance metrics loading and checks."""
    # Mock the load_metrics to prevent Redis calls and return a default state
    mocker.patch('app.main.PerformanceMetrics.load_metrics', return_value={
        'total_trades': 0, 'winning_trades': 0, 'total_profit': 0,
        'max_drawdown': 0, 'daily_trades': {}, 'daily_loss': {},
        'trade_history': [], 'last_reset_date': '2023-01-01'
    })
    # Also mock save_metrics to prevent any write attempts
    mocker.patch('app.main.PerformanceMetrics.save_metrics')
    # Mock can_trade so we can control it in tests
    return mocker.patch('app.main.PerformanceMetrics.can_trade')

def test_run_trading_cycle_buy_signal(mock_exchange, mock_ai_connector, mock_redis, mock_performance_metrics):
    """
    Test the trading cycle with a simple BUY signal for a new asset.
    """
    # Arrange
    mock_performance_metrics.return_value = True
    mock_ai_connector.return_value = [{"asset": "BTC/USDT", "signal": "BUY", "confidence": 0.9}]
    mock_redis.hgetall_json.return_value = {}  # No existing positions
    mock_order = {'id': '123', 'symbol': 'BTC/USDT', 'price': 50000, 'filled': 0.004}
    mock_exchange.create_market_buy_order.return_value = mock_order

    # Act
    run_trading_cycle()

    # Assert
    mock_exchange.create_market_buy_order.assert_called_once_with('BTC/USDT', CONFIG['trading']['min_trade_amount'])

def test_run_trading_cycle_sell_signal_with_position(mock_exchange, mock_ai_connector, mock_redis, mock_performance_metrics):
    """
    Test the trading cycle with a SELL signal for an existing position.
    """
    # Arrange
    asset_to_sell = "ETH/USDT"
    mock_performance_metrics.return_value = True
    mock_ai_connector.return_value = [{"asset": asset_to_sell, "signal": "SELL", "confidence": 0.9}]
    mock_redis.hgetall_json.return_value = {
        'order1': {
            'symbol': asset_to_sell,
            'amount': 0.1,
            'entry_price': 3000.0,
            'take_profit': 3300.0,
            'stop_loss': 2700.0,
            'entry_time': datetime.now().isoformat()
        }
    }
    mock_order = {'id': '124', 'symbol': asset_to_sell, 'price': 3100, 'filled': 0.1}
    mock_exchange.create_market_sell_order.return_value = mock_order

    # Act
    run_trading_cycle()

    # Assert
    mock_exchange.create_market_sell_order.assert_called_once_with(asset_to_sell, 0.1)

def test_run_trading_cycle_sell_signal_no_position(mock_exchange, mock_ai_connector, mock_redis, mock_performance_metrics):
    """
    Test the trading cycle with a SELL signal for which no position exists.
    """
    # Arrange
    mock_performance_metrics.return_value = True
    mock_ai_connector.return_value = [{"asset": "ETH/USDT", "signal": "SELL", "confidence": 0.9}]
    mock_redis.hgetall_json.return_value = {}

    # Act
    run_trading_cycle()

    # Assert
    mock_exchange.create_market_sell_order.assert_not_called()

def test_run_trading_cycle_performance_limit_prevents_trading(mock_exchange, mock_ai_connector, mock_performance_metrics):
    """
    Test that if performance.can_trade() is False, no trading happens.
    """
    # Arrange
    mock_performance_metrics.return_value = False

    # Act
    run_trading_cycle()

    # Assert
    mock_ai_connector.assert_not_called()
    mock_exchange.create_market_buy_order.assert_not_called()
    mock_exchange.create_market_sell_order.assert_not_called()

def test_check_existing_positions_closes_on_take_profit(mock_exchange, mock_redis, mock_performance_metrics):
    """
    Test that check_existing_positions closes a trade if take profit is hit.
    """
    # Arrange
    asset = "BTC/USDT"
    mock_performance_metrics.return_value = True
    mock_redis.hgetall_json.return_value = {
        'order1': {
            'symbol': asset, 'amount': 0.1, 'entry_price': 50000,
            'take_profit': 51000, 'stop_loss': 49000, 'entry_time': datetime.now().isoformat()
        }
    }
    # Current price is above take_profit
    mock_exchange.fetch_ticker.return_value = {'last': 51500}

    # Act
    from app.main import TradeExecution, PerformanceMetrics
    trade_exec = TradeExecution(mock_exchange, PerformanceMetrics())
    trade_exec.check_existing_positions()

    # Assert
    mock_exchange.create_market_sell_order.assert_called_once_with(asset, 0.1)

def test_check_existing_positions_closes_on_stop_loss(mock_exchange, mock_redis, mock_performance_metrics):
    """
    Test that check_existing_positions closes a trade if stop loss is hit.
    """
    # Arrange
    asset = "BTC/USDT"
    mock_performance_metrics.return_value = True
    mock_redis.hgetall_json.return_value = {
        'order1': {
            'symbol': asset, 'amount': 0.1, 'entry_price': 50000,
            'take_profit': 51000, 'stop_loss': 49000, 'entry_time': datetime.now().isoformat()
        }
    }
    # Current price is below stop_loss
    mock_exchange.fetch_ticker.return_value = {'last': 48500}

    # Act
    from app.main import TradeExecution, PerformanceMetrics
    trade_exec = TradeExecution(mock_exchange, PerformanceMetrics())
    trade_exec.check_existing_positions()

    # Assert
    mock_exchange.create_market_sell_order.assert_called_once_with(asset, 0.1)

def test_execute_trade_handles_exception(mock_exchange, mock_redis, mock_performance_metrics):
    """
    Test that execute_trade handles exceptions from the exchange.
    """
    # Arrange
    asset = "BTC/USDT"
    amount = 0.01
    mock_performance_metrics.return_value = True
    mock_exchange.create_market_buy_order.side_effect = ccxt.NetworkError("Test exchange error")

    # Act
    from app.main import TradeExecution, PerformanceMetrics
    trade_exec = TradeExecution(mock_exchange, PerformanceMetrics())
    result = trade_exec.execute_trade('buy', amount, asset)

    # Assert
    assert result is None
    # Check if notification was sent
    # This requires mocking send_telegram_notification
    # For now, we just ensure it doesn't crash and returns None
