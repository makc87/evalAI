"""Офлайн-проверки загрузчика кейсов. Модель и API здесь не используются."""

import json

import pytest

from case_loader import load_cases


def write_cases(tmp_path, data):
    """Пишем тестовые данные во временный файл, не меняя golden_cases.json."""
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def test_loads_repository_dataset():
    """Проверяем, что реальный набор читается и опечатка в ID исправлена."""
    cases = load_cases()
    ids = {case['id'] for case in cases}
    assert 'case_3' in ids
    assert 'cade_3' not in ids


def test_rejects_duplicate_ids(tmp_path):
    """Пробелы по краям ID не делают его уникальным."""
    case = {'id': 'case_1', 'input': 'question', 'expected_behavior': 'answer'}
    path = write_cases(tmp_path, [case, dict(case, id=' case_1 ')])
    with pytest.raises(ValueError, match='Duplicate case ID: case_1'):
        load_cases(path)


@pytest.mark.parametrize('field', ['id', 'input', 'expected_behavior'])
@pytest.mark.parametrize('invalid', [None, '', '   ', 42])
def test_rejects_invalid_fields(tmp_path, field, invalid):
    """Для каждого обязательного поля проверяем четыре вида ошибок."""
    case = {'id': 'case_1', 'input': 'question', 'expected_behavior': 'answer'}
    # None в параметрах означает отсутствие ключа, а не JSON null.
    if invalid is None:
        del case[field]
    else:
        case[field] = invalid
    path = write_cases(tmp_path, [case])
    with pytest.raises(ValueError, match=field):
        load_cases(path)


@pytest.mark.parametrize('data', [{}, [], ['not an object']])
def test_rejects_invalid_dataset_structure(tmp_path, data):
    """Отклоняем объект вместо списка, пустой список и строку вместо кейса."""
    with pytest.raises(ValueError):
        load_cases(write_cases(tmp_path, data))
