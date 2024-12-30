import os
import logging
import requests

import src.modules.fetch_position_details as fetch_position_details

TELEGRAM_CONFIG = {
    'bot_token': os.environ.get('TELEGRAM_BOT_SPOT_TOKEN'),
    'chat_id': os.environ.get('TELEGRAM_CHAT_SPOT_ID')
}

def send_telegram_notification(message, exchange=None):
    bot_token = TELEGRAM_CONFIG['bot_token']
    chat_id = TELEGRAM_CONFIG['chat_id']  # This should be the group chat ID

    if not bot_token or not chat_id:
        logging.warning('Telegram bot token or chat ID not set')
        return

    try:
        extra_info = ""
        if exchange:
            try:
                balance = exchange.fetch_balance()
                usdt_balance = balance.get('USDT', {}).get('free', 0)
                extra_info += f"\nCurrent Balance: {usdt_balance:.2f} USDT"

                position_details = fetch_position_details(exchange)
                if position_details:
                    position_info = (f"Buy Positions: {position_details['buy']} "
                                     f"({position_details['total_buy']:.4f} contracts)\n"
                                     f"Sell Positions: {position_details['sell']} "
                                     f"({position_details['total_sell']:.4f} contracts)")
                    extra_info += f"\n{position_info}"

            except Exception as balance_error:
                logging.error(f"Error fetching exchange details: {balance_error}")

        final_message = f"{message}{extra_info}"
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": final_message}

        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"Failed to send Telegram message: {response.text}")

    except Exception as e:
        logging.error(f'Error occurred in send_telegram_notification: {e}')