from __future__ import annotations

"""
Скрипт для загрузки примерных вакансий в бота.

Использует те же функции, что и бот:
- извлечение навыков и требуемого опыта;
- построение эмбеддингов через модель `cointegrated/rubert-tiny2`;
- сохранение вакансий в `data/vacancies.json`.

Запуск из корня проекта:

    python -m src.load_sample_vacancies
"""

from typing import List, Dict

from . import nlp, storage


def _sample_vacancies() -> List[Dict[str, str]]:
    """
    Возвращает список примерных вакансий (название + текст описания).
    """
    return [
        {
            "title": "Python-разработчик (backend)",
            "description": (
                "Ищем Python-разработчика для разработки backend-сервисов.\n\n"
                "Требования:\n"
                "- Уверенное знание Python\n"
                "- Опыт разработки веб-приложений на Django или FastAPI\n"
                "- Опыт работы с PostgreSQL и написания SQL-запросов\n"
                "- Опыт работы с Docker и Linux\n"
                "- Понимание принципов REST API\n"
                "- Опыт коммерческой разработки от 2 лет\n\n"
                "Будет плюсом:\n"
                "- Опыт работы с Pandas и NumPy\n"
                "- Опыт написания unit-тестов\n"
            ),
        },
        {
            "title": "Data Scientist / ML Engineer",
            "description": (
                "Ищем Data Scientist / ML инженера для разработки и поддержания моделей машинного обучения.\n\n"
                "Требования:\n"
                "- Уверенное знание Python\n"
                "- Опыт работы с библиотеками Pandas, NumPy, scikit-learn\n"
                "- Опыт подготовки и анализа данных\n"
                "- Понимание основных алгоритмов машинного обучения\n"
                "- Опыт коммерческой работы с ML от 1 года\n\n"
                "Будет плюсом:\n"
                "- Опыт построения REST API для моделей (Flask, FastAPI)\n"
                "- Опыт работы с Docker\n"
            ),
        },
        {
            "title": "Frontend-разработчик (React)",
            "description": (
                "Ищем Frontend-разработчика для разработки SPA-приложений.\n\n"
                "Требования:\n"
                "- Уверенное знание JavaScript и TypeScript\n"
                "- Опыт разработки на React\n"
                "- Базовые знания HTML, CSS\n"
                "- Опыт работы с Git\n"
                "- Опыт коммерческой разработки от 1 года\n\n"
                "Будет плюсом:\n"
                "- Опыт работы с REST API\n"
                "- Опыт работы с Docker и Linux\n"
            ),
        },
        {
            "title": "DevOps-инженер",
            "description": (
                "Ищем DevOps-инженера для сопровождения инфраструктуры и настройки CI/CD.\n\n"
                "Требования:\n"
                "- Хорошее знание Linux\n"
                "- Опыт работы с Docker и Kubernetes\n"
                "- Опыт настройки CI/CD (GitLab CI, GitHub Actions и т.п.)\n"
                "- Понимание сетевых протоколов и основ безопасности\n"
                "- Опыт коммерческой работы от 2 лет\n\n"
                "Будет плюсом:\n"
                "- Опыт работы с Python\n"
                "- Опыт мониторинга и логирования (Prometheus, Grafana и т.п.)\n"
            ),
        },
    ]


def load_samples() -> None:
    """
    Добавить примерные вакансии в JSON-хранилище.
    Существующие вакансии не удаляются.
    """
    vacancies_data = _sample_vacancies()

    for item in vacancies_data:
        title = item["title"]
        description = item["description"]

        skills = nlp.extract_skills(description)
        experience_years = nlp.extract_experience_years(description)
        embedding = nlp.text_embedding(f"{title}\n{description}")

        vacancy = storage.Vacancy(
            id=storage.next_vacancy_id(),
            title=title,
            description=description,
            skills=skills,
            min_experience_years=experience_years,
            embedding=embedding,
        )
        storage.add_vacancy(vacancy)
        print(f"Вакансия добавлена: {vacancy.id} — {vacancy.title}")


def main() -> None:
    load_samples()
    print("Все примерные вакансии успешно загружены.")


if __name__ == "__main__":
    main()

