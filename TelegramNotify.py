import os
import requests

class TelegramNotifier:
    def __init__(self, token: str = None, chat_id: str = None, timeout: float = 5.0):
        # Retrieve credentials from arguments or environment variables
        self.token = token or os.getenv("8555129415:AAF8FOdqbFxlxpLPYFZ0_gFzsxArx2QT_WQ")
        self.chat_id = chat_id or os.getenv("-5105924827")
        self.timeout = timeout

        # Ensure credentials are present
        if not self.token or not self.chat_id:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    def send(self, text: str) -> bool:
        # Construct the API URL and payload
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        try:
            # Send the POST request to Telegram
            r = requests.post(url, json=payload, timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            # Handle network errors silently
            return False
