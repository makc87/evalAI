"""Оценка готового ответа из examples/saved_answer.json."""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from eval_lib import EvalTestCase, evaluate
from eval_lib.metrics import AnswerRelevancyMetric, GEval, ToxicityMetric

SCENARIO_PATH = Path(__file__).resolve().parent / 'examples' / 'saved_answer.json'


async def main():
    scenario = json.loads(SCENARIO_PATH.read_text(encoding='utf-8'))
    settings = scenario['evaluation']
    metrics_config = scenario['metrics']

    load_dotenv()
    api_key = os.getenv('DOTPOIN_API_KEY')
    if not api_key:
        raise ValueError('DOTPOIN_API_KEY не найден в окружении или .env')

    base_url = os.getenv('OPENAI_BASE_URL', 'https://llms.dotpoin.com/v1')
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ['OPENAI_BASE_URL'] = base_url

    model = settings['model']
    test_case = EvalTestCase(**scenario['case'])
    metrics = [
        AnswerRelevancyMetric(model=model, **metrics_config['answer_relevancy']),
        ToxicityMetric(model=model, **metrics_config['toxicity']),
        GEval(model=model, **metrics_config['geval']),
    ]

    print('KEY LOADED:', bool(api_key))
    print('BASE URL:', base_url)
    results = await evaluate(
        test_cases=[test_case],
        metrics=metrics,
        show_dashboard=settings['show_dashboard'],
        session_name=settings['session_name'],
    )
    print(results)


if __name__ == '__main__':
    asyncio.run(main())
