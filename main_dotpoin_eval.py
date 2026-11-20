import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from eval_lib import EvalTestCase, evaluate
from eval_lib.metrics import AnswerRelevancyMetric, GEval, ToxicityMetric

from evalai.case_loader import load_cases
from evalai.clients.dotpoin import DotpoinClient
from evalai.config import load_settings
from evalai.run_report import run_evaluation

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / 'prompts' / 'current.txt'
CASES_PATH = BASE_DIR / 'cases' / 'current.json'
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '3'))


def build_test_cases(system_prompt, settings, *, on_validated=None):
    cases = load_cases(CASES_PATH)
    if on_validated is not None:
        on_validated()
    client = DotpoinClient(settings)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def generate(case):
        async with semaphore:
            answer = await client.generate_answer_async(case['input'], system_prompt)
        return EvalTestCase(
            input=case['input'],
            actual_output=answer,
            expected_output=case['expected_behavior'],
        )

    return [asyncio.create_task(generate(case)) for case in cases]


async def main():
    load_dotenv()
    settings = load_settings()
    os.environ['OPENAI_API_KEY'] = settings.api_key
    os.environ['OPENAI_BASE_URL'] = settings.base_url
    system_prompt = PROMPT_PATH.read_text(encoding='utf-8').strip()
    model = settings.eval_model
    metrics = [
        AnswerRelevancyMetric(model=model, threshold=0.7),
        ToxicityMetric(model=model, threshold=0.9),
        GEval(
            name='Role and consultation quality', model=model, threshold=0.7,
            sampling_temperature=1.0,
            evaluation_steps=[
                'Проверь вежливость и уважительный тон ответа.',
                'Проверь, что ответ не обесценивает клиента или его запрос.',
                'Если ограничения бюджета или сроков актуальны, проверь их объяснение.',
                'Проверь уточнение важных деталей, если их не хватает в запросе.',
                'Проверь наличие конкретного и уместного следующего шага.',
                'Проверь, что прайс-лист не навязывается без запроса клиента.',
                'Выставь итоговую оценку от 0 до 1 по выполнению этих требований.',
            ],
            criteria=(
                'Оцени, насколько ответ соответствует роли консультанта по элитной недвижимости. '
                'Хороший ответ должен: '
                '1) вежливо отвечать клиенту; '
                '2) не обесценивать клиента; '
                '3) объяснять ограничения бюджета или сроков, если они есть; '
                '4) уточнять важные детали; '
                '5) предлагать следующий шаг; '
                '6) не навязывать прайс-лист без запроса клиента.'
            ),
        ),
        GEval(
            name='Expected behavior alignment', model=model, threshold=0.75,
            sampling_temperature=1.0,
            evaluation_steps=[
                'Выдели ключевые требования к поведению из expected_output.',
                'Сопоставь actual_output с каждым требованием с учётом input.',
                'Снизь оценку за пропущенные требования, противоречия и уход от запроса.',
                'Не требуй дословного совпадения: оцени смысл и выполненное поведение.',
                'Выставь итоговую оценку от 0 до 1 по полноте соответствия.',
            ],
            criteria=(
                'Оцени, насколько actual_output соответствует expected_output. '
                'Нужно проверить, выполнил ли агент ожидаемое поведение из тест-кейса. '
                'Снижать оценку, если ответ уходит в сторону, игнорирует ключевые требования '
                'или не покрывает важные элементы expected_output.'
            ),
        ),
        GEval(
            name='Clarification quality', model=model, threshold=0.7,
            sampling_temperature=1.0,
            evaluation_steps=[
                'Определи, каких важных данных не хватает в input для полезного ответа.',
                'Если уточнения нужны, проверь, запрашивает ли actual_output эти данные.',
                'Проверь конкретность вопросов и их пользу для продолжения диалога.',
                'Снизь оценку за лишние, слишком общие вопросы или отсутствие нужных уточнений.',
                'Не штрафуй отсутствие вопросов, если запрос уже достаточно полный.',
                'Выставь итоговую оценку от 0 до 1 по качеству уточнений.',
            ],
            criteria=(
                'Оцени качество уточняющих вопросов в ответе. '
                'Если запрос пользователя неполный, неоднозначный или требует дополнительного контекста, '
                'хороший ответ должен запросить только действительно важные уточнения. '
                'Снижать оценку, если уточнений нет, когда они нужны, '
                'или если вопросы лишние, слишком общие или не помогают продвинуть диалог.'
            ),
        ),
        GEval(
            name='Next step usefulness', model=model, threshold=0.7,
            sampling_temperature=1.0,
            evaluation_steps=[
                'Определи цель пользователя по input и найди предложенное продолжение в actual_output.',
                'Проверь, что следующий шаг конкретен, понятен и связан с запросом.',
                'Проверь, помогает ли этот шаг уточнить задачу или продвинуться к решению.',
                'Снизь оценку за размытое, пассивное или не относящееся к задаче завершение.',
                'Выставь итоговую оценку от 0 до 1 по полезности следующего шага.',
            ],
            criteria=(
                'Оцени, предлагает ли ответ понятный и полезный следующий шаг для пользователя. '
                'Хороший ответ должен завершаться конкретным продолжением: что нужно уточнить, '
                'что агент может сделать дальше, или как пользователь может продвинуться к решению. '
                'Снижать оценку, если ответ заканчивается размыто, пассивно или без практического действия.'
            ),
        ),
    ]

    async def evaluate_cases(test_case_tasks, session_name):
        test_cases = await asyncio.gather(*test_case_tasks)
        return await evaluate(
            test_cases=test_cases,
            metrics=metrics,
            show_dashboard=True,
            session_name=session_name,
        )

    results = await run_evaluation(
        system_prompt, settings, build_test_cases, evaluate_cases, BASE_DIR,
    )
    print(results)


if __name__ == '__main__':
    asyncio.run(main())
