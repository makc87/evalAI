"""Отправка запросов агенту через OpenAI-совместимый API Dotpoin."""

import asyncio

from openai import OpenAI

from evalai.config import Settings


class DotpoinClient:
    def __init__(self, settings: Settings):
        self._sdk = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self._model = settings.chat_model

    def generate_answer(self, case: dict, system_prompt: str) -> str:
        """Отправить запрос выбранной модели с историей диалога и вернуть текст ответа."""
        messages = [{'role': 'system', 'content': system_prompt}]

        for message in case.get('conversation', []):
            messages.append({
                'role': message['role'],
                'content': message['content'],
            })

        messages.append({
            'role': 'user',
            'content': case['input'],
        })

        response = self._sdk.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    async def generate_answer_async(self, case: dict, system_prompt: str) -> str:
        """Асинхронно выполнить блокирующий запрос к агенту."""
        return await asyncio.to_thread(
            self.generate_answer,
            case,
            system_prompt,
        )
