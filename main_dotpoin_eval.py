import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from eval_lib import EvalTestCase, evaluate
from eval_lib.metrics import AnswerRelevancyMetric, ToxicityMetric, GEval


load_dotenv()

OPENAI_API_KEY = os.getenv("DOTPOIN_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://llms.dotpoin.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")
EVAL_MODEL = os.getenv("EVAL_MODEL", "openai:deepseek-chat")

if not OPENAI_API_KEY:
    raise ValueError("DOTPOIN_API_KEY не найден в .env")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

BASE_DIR = Path(__file__).parent
PROMPT_PATH = BASE_DIR / "prompt.txt"
CASES_PATH = BASE_DIR / "golden_cases.json"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def generate_answer(user_input: str, system_prompt: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def build_test_cases(system_prompt: str) -> list[EvalTestCase]:
    cases = load_cases()
    test_cases = []

    for case in cases:
        actual_output = generate_answer(case["input"], system_prompt)
        test_case = EvalTestCase(
            input=case["input"],
            actual_output=actual_output,
            expected_output=case["expected_behavior"],
        )
        test_cases.append(test_case)

    return test_cases


async def main():
    system_prompt = load_prompt()
    test_cases = build_test_cases(system_prompt)

    metrics = [
        AnswerRelevancyMetric(
            model=EVAL_MODEL,
            threshold=0.7,
        ),
        ToxicityMetric(
            model=EVAL_MODEL,
            threshold=0.9,
        ),
        GEval(
            name="Role and consultation quality",
            model=EVAL_MODEL,
            threshold=0.7,
            criteria=(
                "Оцени, насколько ответ соответствует роли консультанта по элитной недвижимости. "
                "Хороший ответ должен: "
                "1) вежливо отвечать клиенту; "
                "2) не обесценивать клиента; "
                "3) объяснять ограничения бюджета или сроков, если они есть; "
                "4) уточнять важные детали; "
                "5) предлагать следующий шаг; "
                "6) не навязывать прайс-лист без запроса клиента."
            ),
        ),
        GEval(
            name="Expected behavior alignment",
            model=EVAL_MODEL,
            threshold=0.75,
            criteria=(
                "Оцени, насколько actual_output соответствует expected_output. "
                "Нужно проверить, выполнил ли агент ожидаемое поведение из тест-кейса. "
                "Снижать оценку, если ответ уходит в сторону, игнорирует ключевые требования "
                "или не покрывает важные элементы expected_output."
            ),
        ),
        GEval(
            name="Clarification quality",
            model=EVAL_MODEL,
            threshold=0.7,
            criteria=(
                "Оцени качество уточняющих вопросов в ответе. "
                "Если запрос пользователя неполный, неоднозначный или требует дополнительного контекста, "
                "хороший ответ должен запросить только действительно важные уточнения. "
                "Снижать оценку, если уточнений нет, когда они нужны, "
                "или если вопросы лишние, слишком общие или не помогают продвинуть диалог."
            ),
        ),
        GEval(
            name="Next step usefulness",
            model=EVAL_MODEL,
            threshold=0.7,
            criteria=(
                "Оцени, предлагает ли ответ понятный и полезный следующий шаг для пользователя. "
                "Хороший ответ должен завершаться конкретным продолжением: что нужно уточнить, "
                "что агент может сделать дальше, или как пользователь может продвинуться к решению. "
                "Снижать оценку, если ответ заканчивается размыто, пассивно или без практического действия."
            ),
        ),
    ]

    print("KEY LOADED:", bool(OPENAI_API_KEY))
    print("BASE URL:", OPENAI_BASE_URL)
    print("CHAT MODEL:", CHAT_MODEL)
    print("EVAL MODEL:", EVAL_MODEL)
    print("PROMPT PATH:", PROMPT_PATH)
    print("CASES COUNT:", len(test_cases))
    print("METRICS COUNT:", len(metrics))

    results = await evaluate(
        test_cases=test_cases,
        metrics=metrics,
        show_dashboard=True,
        session_name="dotpoin-prompt-regression-eval",
    )
    print(results)


if __name__ == "__main__":
    asyncio.run(main())