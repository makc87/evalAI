import importlib.util
import json
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


def test_invalid_last_case_prevents_all_model_requests(tmp_path, monkeypatch):
    """При ошибке в последнем кейсе клиент даже не создаётся."""
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps([
        {'id': 'case_1', 'input': 'first', 'expected_behavior': 'answer'},
        {'id': 'case_2', 'input': 'second'},
    ]), encoding='utf-8')

    client_factory = Mock()
    for name, exports in {
        'dotenv': {'load_dotenv': Mock()},
        'evalai.clients.dotpoin': {'DotpoinClient': client_factory},
        'eval_lib': {'EvalTestCase': dict, 'evaluate': Mock()},
        'eval_lib.metrics': {name: Mock() for name in (
            'AnswerRelevancyMetric', 'ToxicityMetric', 'GEval'
        )},
    }.items():
        module = ModuleType(name)
        vars(module).update(exports)
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.delenv('DOTPOIN_API_KEY', raising=False)
    spec = importlib.util.find_spec('main_dotpoin_eval')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'CASES_PATH', path)

    with pytest.raises(ValueError, match='Case 2: expected_behavior'):
        runner.build_test_cases('system prompt', settings=object())

    client_factory.assert_not_called()
