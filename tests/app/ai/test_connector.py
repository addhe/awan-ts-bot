import pytest
from app.ai.connector import get_ai_signals

def test_get_ai_signals_returns_list():
    """
    Tests that the mock AI connector returns a list.
    """
    signals = get_ai_signals()
    assert isinstance(signals, list)

def test_get_ai_signals_structure():
    """
    Tests that each signal in the list has the correct structure and keys.
    """
    signals = get_ai_signals()
    # The mock should return at least one signal
    assert len(signals) > 0

    for signal in signals:
        assert isinstance(signal, dict)
        assert 'asset' in signal
        assert 'signal' in signal
        assert 'confidence' in signal
        assert isinstance(signal['asset'], str)
        assert isinstance(signal['signal'], str)
        assert isinstance(signal['confidence'], float)
