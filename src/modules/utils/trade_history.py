import json
import logging
from datetime import datetime
from typing import Dict, List, Any

class TradeHistory:
    """Manages trading history with persistent storage."""
    
    def __init__(self) -> None:
        """Initialize trade history from file or create new if not exists."""
        self.history_file = 'trade_history_spot.json'
        self.history: Dict[str, List[Dict[str, Any]]] = self.load_history()

    def load_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load trade history from file or initialize if file doesn't exist."""
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
                logging.info("Trade history loaded successfully")
                return history
        except FileNotFoundError:
            logging.info("No existing trade history found, initializing new history")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding trade history file: {str(e)}")
            return {}
        except Exception as e:
            logging.error(f"Unexpected error loading trade history: {str(e)}")
            return {}

    def save_history(self) -> None:
        """Save current trade history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            logging.debug("Trade history saved successfully")
        except Exception as e:
            logging.error(f"Error saving trade history: {str(e)}")

    def add_trade(self, symbol: str, side: str, amount: float, 
                  price: float, order_id: str) -> None:
        """Add a new trade to history."""
        if symbol not in self.history:
            self.history[symbol] = []
            
        trade_info = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'order_id': order_id
        }
        
        self.history[symbol].append(trade_info)
        self.save_history()
        logging.info(f"Trade recorded: {trade_info}")

    def get_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all trades for a specific symbol."""
        return self.history.get(symbol, [])

    def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        """Get the most recent trade for a symbol."""
        trades = self.history.get(symbol, [])
        return trades[-1] if trades else {}

    def get_trade_count(self, symbol: str) -> int:
        """Get total number of trades for a symbol."""
        return len(self.history.get(symbol, []))

    def clear_history(self, symbol: str = None) -> None:
        """Clear trade history for a symbol or all symbols."""
        if symbol:
            if symbol in self.history:
                self.history[symbol] = []
                logging.info(f"Trade history cleared for {symbol}")
        else:
            self.history = {}
            logging.info("All trade history cleared")
        self.save_history()
