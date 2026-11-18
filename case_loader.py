import json
from pathlib import Path


DEFAULT_CASES_PATH = Path(__file__).with_name('golden_cases.json')
REQUIRED_FIELDS = ('id', 'input', 'expected_behavior')


def load_cases(path=DEFAULT_CASES_PATH):
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

    return cases
