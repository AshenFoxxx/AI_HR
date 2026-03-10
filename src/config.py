import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class BotConfig:
    token: str
    model_name: str = "cointegrated/rubert-tiny2"
    top_k_vacancies: int = 3


def load_config() -> BotConfig:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment or .env file")
    return BotConfig(token=token)

