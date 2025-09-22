import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """
    Loads configuration from a YAML file and merges it with environment variables.
    """
    # Default config path
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yml')

    # Load base config from YAML
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        # Exit or raise a critical error because the bot cannot run without config
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

    # --- Load sensitive data from environment variables and merge ---

    # Binance API credentials
    config['binance'] = {
        'api_key': os.environ.get('API_KEY_SPOT_BINANCE'),
        'api_secret': os.environ.get('API_SECRET_SPOT_BINANCE')
    }

    # Telegram credentials
    config['telegram']['bot_token'] = os.environ.get('TELEGRAM_BOT_TOKEN')
    config['telegram']['chat_id'] = os.environ.get('TELEGRAM_CHAT_ID')

    # --- Validate essential configurations ---
    if not config.get('binance', {}).get('api_key') or not config.get('binance', {}).get('api_secret'):
        logging.warning("Binance API key/secret not found in environment variables. Trading will not be possible.")

    if config.get('telegram', {}).get('enabled') and (not config.get('telegram', {}).get('bot_token') or not config.get('telegram', {}).get('chat_id')):
        logging.warning("Telegram is enabled, but token or chat ID is missing. Notifications will be disabled.")
        config['telegram']['enabled'] = False

    return config

# Load the configuration once when the module is imported
CONFIG = load_config()

if __name__ == '__main__':
    # For debugging purposes, print the loaded config
    import json
    print(json.dumps(CONFIG, indent=2))
