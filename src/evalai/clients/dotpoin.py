"""Отправка запросов агенту через OpenAI-совместимый API Dotpoin."""

from openai import OpenAI

from evalai.config import Settings


class DotpoinClient:
    def __init__(self, settings: Settings):
        self._sdk = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self._model = settings.chat_model

    def generate_answer(self, user_input: str, system_prompt: str) -> str:
        """Отправить запрос выбранной модели и вернуть текст ответа."""
        response = self._sdk.chat.completions.create(
            model=self._model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_input},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
