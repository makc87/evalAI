"""Общие настройки агента и модели-оценщика."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    base_url: str
    chat_model: str
    eval_model: str


def load_settings() -> Settings:
    """Прочитать настройки после загрузки .env вызывающим скриптом."""
    api_key = os.getenv('DOTPOIN_API_KEY')
    if not api_key:
        raise ValueError('DOTPOIN_API_KEY не найден в окружении или .env')

    return Settings(
        api_key=api_key,
        base_url=os.getenv('OPENAI_BASE_URL', 'https://llms.dotpoin.com/v1'),
        chat_model=os.getenv('CHAT_MODEL', 'deepseek-chat'),
        eval_model=os.getenv('EVAL_MODEL', 'openai:deepseek-chat'),
    )
