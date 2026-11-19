"""Local prompt snapshots and evaluation reports; no API dependencies."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4


def _json_default(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return {'serialization': 'repr', 'type': type(value).__name__, 'text': repr(value)}


def _write_report(path, report):
    text = json.dumps(report, ensure_ascii=False, indent=2, default=_json_default)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def save_prompt(system_prompt, root):
    """Save exactly the UTF-8 bytes passed to the client; verify existing snapshots."""
    data = system_prompt.encode('utf-8')
    digest = sha256(data).hexdigest()
    path = Path(root) / 'prompts' / 'versions' / f'{digest}.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f'Prompt snapshot is corrupted: {path.name}')
    return digest, path


async def run_evaluation(system_prompt, settings, build_cases, evaluate_cases, root):
    """Validate before archiving; bind one prompt to a uniquely named run."""
    root = Path(root)
    report = {}
    report_path = None

    def start_run():
        nonlocal report_path
        digest, snapshot = save_prompt(system_prompt, root)
        now = datetime.now(timezone.utc)
        run_id = f'dotpoin-eval-{now:%Y%m%dT%H%M%S%fZ}-{digest[:12]}-{uuid4().hex[:8]}'
        report_path = root / 'reports' / f'{run_id}.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.update(
            schema_version=1, run_id=run_id, session_name=run_id,
            started_at=now.isoformat(), status='running',
            prompt_sha256=digest, prompt_version=digest[:12],
            prompt_snapshot=snapshot.relative_to(root).as_posix(),
            chat_model=settings.chat_model, eval_model=settings.eval_model,
        )
        _write_report(report_path, report)
        print(f'Session: {run_id}\nPrompt SHA-256: {digest}\nReport: {report_path}')

    try:
        test_cases = build_cases(system_prompt, settings, on_validated=start_run)
        if report_path is None:
            raise RuntimeError('build_cases did not call on_validated')
        results = await evaluate_cases(test_cases, report['session_name'])
        report.update(status='completed', results=results,
                      finished_at=datetime.now(timezone.utc).isoformat())
        _write_report(report_path, report)
        return results
    except Exception as error:
        if report_path is not None and report:
            report.update(status='failed', error_type=type(error).__name__,
                          finished_at=datetime.now(timezone.utc).isoformat())
            try:
                _write_report(report_path, report)
            except Exception:
                pass
        raise
