import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def test_invalid_last_case_prevents_all_model_requests(tmp_path, monkeypatch):
    """Даже первый корректный кейс не отправляется, если последний невалиден."""
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps([
        {'id': 'case_1', 'input': 'first', 'expected_behavior': 'answer'},
        {'id': 'case_2', 'input': 'second'},
    ]), encoding='utf-8')

    # Изолируем запуск от SDK, eval_lib и реального .env.
    for name, exports in {
        'dotenv': {'load_dotenv': Mock()},
        'openai': {'OpenAI': Mock(return_value=SimpleNamespace())},
        'eval_lib': {'EvalTestCase': dict, 'evaluate': Mock()},
        'eval_lib.metrics': {name: Mock() for name in (
            'AnswerRelevancyMetric', 'ToxicityMetric', 'GEval'
        )},
    }.items():
        module = ModuleType(name)
        vars(module).update(exports)
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv('DOTPOIN_API_KEY', 'offline-test-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://example.invalid/v1')
    spec = importlib.util.find_spec('main_dotpoin_eval')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'CASES_PATH', path)
    generate_answer = Mock(return_value='answer')
    monkeypatch.setattr(runner, 'generate_answer', generate_answer)

    with pytest.raises(ValueError, match='Case 2: expected_behavior'):
        runner.build_test_cases('system prompt')

    generate_answer.assert_not_called()
