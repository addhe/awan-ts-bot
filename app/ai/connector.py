import logging
import requests
from app.config import CONFIG

def get_ai_signals():
    """
    Fetches trading signals from the AI service.

    This is currently a MOCK implementation. It returns a hardcoded list of signals.
    This should be replaced with a real API call to the AI service.
    """
    logging.info("Fetching trading signals from AI service...")

    # --- MOCK IMPLEMENTATION START ---
    # In a real implementation, you would make an HTTP request here.
    # For example:
    # try:
    #     url = CONFIG['app']['ai_service_url']
    #     response = requests.get(url, timeout=30)
    #     response.raise_for_status()  # Raise an exception for bad status codes
    #     signals = response.json()
    #     logging.info(f"Received {len(signals.get('actions', []))} signals from AI.")
    #     return signals.get('actions', [])
    # except requests.exceptions.RequestException as e:
    #     logging.error(f"Could not get signals from AI service: {e}")
    #     return []
    # --- MOCK IMPLEMENTATION END ---

    # For now, return a hardcoded list for development purposes.
    mock_signals = [
        {"asset": "BTC/USDT", "signal": "BUY", "confidence": 0.85},
        {"asset": "ETH/USDT", "signal": "SELL", "confidence": 0.78},
        {"asset": "SOL/USDT", "signal": "NEUTRAL", "confidence": 0.60}
    ]

    logging.warning("Using MOCK AI signals for development.")
    return mock_signals
