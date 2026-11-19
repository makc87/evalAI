import asyncio
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from evalai.run_report import save_prompt


@pytest.fixture
def runner(monkeypatch, tmp_path):
    client = Mock()
    client.generate_answer.return_value = 'offline answer'
    for name, exports in {
        'dotenv': {'load_dotenv': Mock()},
        'evalai.clients.dotpoin': {'DotpoinClient': Mock(return_value=client)},
        'eval_lib': {'EvalTestCase': dict, 'evaluate': AsyncMock()},
        'eval_lib.metrics': {key: Mock() for key in (
            'AnswerRelevancyMetric', 'ToxicityMetric', 'GEval')},
    }.items():
        module = ModuleType(name)
        vars(module).update(exports)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.find_spec('main_dotpoin_eval')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'BASE_DIR', tmp_path)
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps([
        {'id': 'one', 'input': 'question', 'expected_behavior': 'answer'}
    ]), encoding='utf-8')
    monkeypatch.setattr(module, 'CASES_PATH', path)
    prompt = tmp_path / 'current.txt'
    prompt.write_bytes('  Точный промпт\n'.encode('utf-8'))
    monkeypatch.setattr(module, 'PROMPT_PATH', prompt)
    monkeypatch.setattr(module, 'load_settings', lambda: SimpleNamespace(
        api_key='offline-only', base_url='https://example.invalid',
        chat_model='agent', eval_model='judge'))
    monkeypatch.delenv('DOTPOIN_API_KEY', raising=False)
    module.evaluate.return_value = {'score': 0.8}
    return module


def test_snapshot_deduplicates_and_versions_changes(tmp_path):
    first = save_prompt('промпт\n', tmp_path)
    assert save_prompt('промпт\n', tmp_path) == first
    assert save_prompt('промпт', tmp_path)[0] != first[0]
    assert first[1].read_bytes() == 'промпт\n'.encode('utf-8')
    assert len(list((tmp_path / 'prompts/versions').glob('*.txt'))) == 2


def test_corrupted_snapshot_is_rejected(tmp_path):
    _, path = save_prompt('prompt', tmp_path)
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='corrupted'):
        save_prompt('prompt', tmp_path)


def test_main_binds_report_snapshot_client_and_session(runner, tmp_path):
    asyncio.run(runner.main())
    report = json.loads(next((tmp_path / 'reports').glob('*.json')).read_text('utf-8'))
    sent = runner.DotpoinClient.return_value.generate_answer.call_args.args[1]
    assert sent == 'Точный промпт'
    assert (tmp_path / report['prompt_snapshot']).read_bytes() == sent.encode('utf-8')
    assert report['prompt_sha256'] == hashlib.sha256(sent.encode('utf-8')).hexdigest()
    assert runner.evaluate.call_args.kwargs['session_name'] == report['run_id']
    assert report['prompt_version'] in report['run_id']
    assert report['results'] == {'score': 0.8}
    assert report['status'] == 'completed'
    assert report['chat_model'] == 'agent'
    assert report['eval_model'] == 'judge'
    assert 'offline-only' not in json.dumps(report)


def test_same_prompt_creates_distinct_reports(runner, tmp_path):
    asyncio.run(runner.main())
    asyncio.run(runner.main())
    assert len(list((tmp_path / 'reports').glob('*.json'))) == 2
    assert len(list((tmp_path / 'prompts/versions').glob('*.txt'))) == 1


def test_invalid_last_case_creates_no_artifacts_or_client(runner, tmp_path):
    cases = json.loads(runner.CASES_PATH.read_text('utf-8'))
    cases.append({'id': 'two', 'input': 'invalid'})
    runner.CASES_PATH.write_text(json.dumps(cases), encoding='utf-8')
    with pytest.raises(ValueError, match='Case 2: expected_behavior'):
        asyncio.run(runner.main())
    runner.DotpoinClient.assert_not_called()
    runner.evaluate.assert_not_called()
    assert not (tmp_path / 'prompts').exists()
    assert not (tmp_path / 'reports').exists()


def test_archive_failure_prevents_client(runner, tmp_path):
    (tmp_path / 'prompts').write_text('not a directory')
    with pytest.raises(OSError):
        asyncio.run(runner.main())
    runner.DotpoinClient.assert_not_called()
    runner.evaluate.assert_not_called()


@pytest.mark.parametrize('stage', ['generation', 'evaluation'])
def test_failures_leave_linked_failed_report(runner, tmp_path, stage):
    if stage == 'generation':
        runner.DotpoinClient.return_value.generate_answer.side_effect = RuntimeError('failure')
    else:
        runner.evaluate.side_effect = RuntimeError('failure')
    with pytest.raises(RuntimeError, match='failure'):
        asyncio.run(runner.main())
    report = json.loads(next((tmp_path / 'reports').glob('*.json')).read_text('utf-8'))
    assert report['status'] == 'failed'
    assert report['error_type'] == 'RuntimeError'
    assert (tmp_path / report['prompt_snapshot']).exists()


def test_dataclass_result_serializes_structurally(runner, tmp_path):
    @dataclass
    class Result:
        score: float
    runner.evaluate.return_value = Result(0.9)
    asyncio.run(runner.main())
    report = json.loads(next((tmp_path / 'reports').glob('*.json')).read_text('utf-8'))
    assert report['results'] == {'score': 0.9}
