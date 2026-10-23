import asyncio
import os

from dotenv import load_dotenv

from eval_lib import EvalTestCase, evaluate, FormatCheckMetric
from eval_lib.metrics import (
    AnswerRelevancyMetric,
    ToxicityMetric,
    GEval,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("DOTPOIN_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://llms.dotpoin.com/v1")

if not OPENAI_API_KEY:
    raise ValueError("DOTPOIN_API_KEY не найден в .env")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

test_case = EvalTestCase(
    input="планирую купить однушку за 5 млн.рублей в центре города. Сроки 2 дня. Возможно такое сделать?",
    actual_output=(
        "Здравствуйте! Благодарю за обращение."
"Вынужден сразу внести ясность: 5 млн рублей за квартиру в центре города — это, к сожалению, нереалистичная сумма для сегмента элитного жилья, в котором я работаю. В центре даже стандартные однокомнатные квартиры эконом-класса редко опускаются ниже 8–10 млн рублей, а элитный сегмент стартует от 20–25 млн и выше."
"Что касается сроков в 2 дня — с юридической точки зрения провести полноценную сделку (проверка объекта, подготовка документов, регистрация) за такой срок практически невозможно. Обычно только на подготовку и проверку уходит от 5 до 14 дней."
"Что я могу вам предложить:"
"Если бюджет гибкий — могу подобрать достойные варианты в центре из моего каталога элитной недвижимости."
"Если бюджет строгий — возможно, стоит рассмотреть районы рядом с центром или объекты чуть меньшей площади."
"Хотите, чтобы я показал вам актуальные варианты из моего прайс-листа? Также могу поделиться изображениями и видео некоторых объектов, чтобы вы могли оценить уровень."

    ),
    expected_output=(
        "Агент должен корректно объяснить, про сроки и бюджет "
        "для элитной вторички, не обесценивать клиента, уточнить район, тип жилья и валюту, "
        "а также предложить помощь в дальнейшем подборе."
    ),
)

test_cases = [test_case]

metrics = [
    AnswerRelevancyMetric(
        model="openai:deepseek-chat",
        threshold=0.7,
    ),
    ToxicityMetric(
        model="openai:deepseek-chat",
        threshold=0.9,
    ),
    GEval(
        name="Role and consultation quality",
        model="openai:deepseek-chat",
        threshold=0.7,
        criteria=(
            "Оцени, насколько ответ соответствует роли консультанта по элитной недвижимости. "
            "Хороший ответ должен: "
            "1) вежливо объяснить, что 1,5 млн рублей недостаточно для элитной вторички; "
            "2) не обесценивать клиента; "
            "3) уточнять район, тип жилья или валюту; "
            "4) предлагать конкретный следующий шаг или дальнейшую помощь; "
            "5) не навязывать прайс-лист без запроса клиента."
        ),
    ),

]

async def main():
    print("KEY LOADED:", bool(OPENAI_API_KEY))
    print("BASE URL:", OPENAI_BASE_URL)

    results = await evaluate(
        test_cases=test_cases,
        metrics=metrics,
        show_dashboard=True,
        session_name="real-estate-eval-1",
    )
    print(results)

if __name__ == "__main__":
    asyncio.run(main())