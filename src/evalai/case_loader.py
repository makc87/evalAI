"""Загрузка и проверка JSON-набора сценариев до запуска оценки."""

import json
from pathlib import Path

REQUIRED_FIELDS = ('id', 'input', 'expected_behavior')
VALID_ROLES = ('user', 'assistant')


def _validate_conversation(case, index):
    conversation = case.get('conversation', [])
    if not isinstance(conversation, list):
        raise ValueError(f'Case {index}: conversation must be a list')

    for message_index, message in enumerate(conversation, start=1):
        if not isinstance(message, dict):
            raise ValueError(f'Case {index}: conversation message {message_index} must be an object')

        role = message.get('role')
        content = message.get('content')

        if role not in VALID_ROLES:
            raise ValueError(f'Case {index}: conversation message {message_index} role must be user or assistant')
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f'Case {index}: conversation message {message_index} content must be a non-empty string')


def load_cases(path):
    """Прочитать и проверить сценарии из указанного файла."""
    cases = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not cases:
        raise ValueError('Dataset must be a non-empty list')

    seen = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f'Case {index} must be an object')
        for field in REQUIRED_FIELDS:
            value = case.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Case {index}: {field} must be a non-empty string')

        case_id = case['id'].strip()
        if case_id in seen:
            raise ValueError(f'Duplicate case ID: {case_id}')
        seen.add(case_id)

        _validate_conversation(case, index)

    return cases
