import os
import logging
import requests

def send_telegram_notification(message):
    """
    Sends a message to a Telegram chat.
    The bot token and chat ID are read from environment variables.
    """
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        logging.warning('Telegram bot token or chat ID not set. Cannot send notification.')
        return

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}

        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            logging.error(f"Failed to send Telegram message: {response.text}")

    except Exception as e:
        logging.error(f'Error occurred in send_telegram_notification: {e}')
