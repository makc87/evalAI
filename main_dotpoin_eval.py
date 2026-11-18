import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from eval_lib import EvalTestCase, evaluate
from eval_lib.metrics import AnswerRelevancyMetric, GEval, ToxicityMetric

from evalai.case_loader import load_cases
from evalai.clients.dotpoin import DotpoinClient
from evalai.config import load_settings

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / 'prompt.txt'
CASES_PATH = BASE_DIR / 'golden_cases.json'


def build_test_cases(system_prompt, settings):
    cases = load_cases(CASES_PATH)
    client = DotpoinClient(settings)
    return [
        EvalTestCase(
            input=case['input'],
            actual_output=client.generate_answer(case['input'], system_prompt),
            expected_output=case['expected_behavior'],
        )
        for case in cases
    ]


async def main():
    load_dotenv()
    settings = load_settings()
    os.environ['OPENAI_API_KEY'] = settings.api_key
    os.environ['OPENAI_BASE_URL'] = settings.base_url
    test_cases = build_test_cases(PROMPT_PATH.read_text(encoding='utf-8').strip(), settings)
    model = settings.eval_model
    metrics = [
        AnswerRelevancyMetric(model=model, threshold=0.7),
        ToxicityMetric(model=model, threshold=0.9),
        GEval(name='Role and consultation quality', model=model, threshold=0.7, criteria=(
            'Оцени, насколько ответ соответствует роли консультанта по элитной недвижимости. '
            'Хороший ответ должен: '
            '1) вежливо отвечать клиенту; '
            '2) не обесценивать клиента; '
            '3) объяснять ограничения бюджета или сроков, если они есть; '
            '4) уточнять важные детали; '
            '5) предлагать следующий шаг; '
            '6) не навязывать прайс-лист без запроса клиента.'
        )),
        GEval(name='Expected behavior alignment', model=model, threshold=0.75, criteria=(
            'Оцени, насколько actual_output соответствует expected_output. '
            'Нужно проверить, выполнил ли агент ожидаемое поведение из тест-кейса. '
            'Снижать оценку, если ответ уходит в сторону, игнорирует ключевые требования '
            'или не покрывает важные элементы expected_output.'
        )),
        GEval(name='Clarification quality', model=model, threshold=0.7, criteria=(
            'Оцени качество уточняющих вопросов в ответе. '
            'Если запрос пользователя неполный, неоднозначный или требует дополнительного контекста, '
            'хороший ответ должен запросить только действительно важные уточнения. '
            'Снижать оценку, если уточнений нет, когда они нужны, '
            'или если вопросы лишние, слишком общие или не помогают продвинуть диалог.'
        )),
        GEval(name='Next step usefulness', model=model, threshold=0.7, criteria=(
            'Оцени, предлагает ли ответ понятный и полезный следующий шаг для пользователя. '
            'Хороший ответ должен завершаться конкретным продолжением: что нужно уточнить, '
            'что агент может сделать дальше, или как пользователь может продвинуться к решению. '
            'Снижать оценку, если ответ заканчивается размыто, пассивно или без практического действия.'
        )),
    ]
    results = await evaluate(
        test_cases=test_cases,
        metrics=metrics,
        show_dashboard=True,
        session_name='dotpoin-prompt-regression-eval',
    )
    print(results)


if __name__ == '__main__':
    asyncio.run(main())
