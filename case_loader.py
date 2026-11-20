"""Временный совместимый импорт для существующих скриптов и тестов."""

from pathlib import Path

from evalai.case_loader import load_cases as _load_cases

DEFAULT_CASES_PATH = Path(__file__).resolve().parent / 'cases' / 'current.json'


def load_cases(path=DEFAULT_CASES_PATH):
    return _load_cases(path)
