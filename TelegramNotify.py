import os
import requests

class TelegramNotifier:
    def __init__(self, token: str = None, chat_id: str = None, timeout: float = 5.0):
        self.token = token or os.getenv("8555129415:AAF8FOdqbFxlxpLPYFZ0_gFzsxArx2QT_WQ")
        self.chat_id = chat_id or os.getenv("464019501")
        self.timeout = timeout

        if not self.token or not self.chat_id:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    def send(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False
