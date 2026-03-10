from __future__ import annotations

from typing import Dict, List, Optional
from io import BytesIO
from pathlib import Path
import re

import joblib
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Document, Message

from . import nlp, storage


router = Router()


DISCLAIMER_TEXT = (
    "⚠️ Результаты носят рекомендательный характер и требуют обязательной "
    "проверки HR-специалистом."
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "candidate_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


_loaded_model = None
_loaded_scaler = None


def _load_ml_model() -> None:
    """
    Лениво загрузить модель и скейлер для числового прогноза пригодности кандидата.
    """
    global _loaded_model, _loaded_scaler
    if _loaded_model is not None and _loaded_scaler is not None:
        return

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise RuntimeError(
            "ML-модель не найдена. Сначала запустите обучение: python -m src.train_model"
        )

    _loaded_model = joblib.load(MODEL_PATH)
    _loaded_scaler = joblib.load(SCALER_PATH)


def _predict_fit_level(features: Dict[str, float]) -> Optional[Dict[str, object]]:
    """
    Сделать предсказание уровня пригодности кандидата по числовым признакам.

    Ожидаемые признаки:
        - experience_years
        - skills_count
        - age
        - education
        - match_score
    """
    _load_ml_model()
    assert _loaded_model is not None and _loaded_scaler is not None

    ordered_feature_names = [
        "experience_years",
        "skills_count",
        "age",
        "education",
        "match_score",
    ]

    try:
        x = [
            float(features.get(name, 0.0))
            for name in ordered_feature_names
        ]
    except (TypeError, ValueError):
        return None

    X = _loaded_scaler.transform([x])

    try:
        # Предсказание класса
        y_pred = _loaded_model.predict(X)
        # Предсказание вероятностей, если поддерживается
        if hasattr(_loaded_model, "predict_proba"):
            proba = _loaded_model.predict_proba(X)[0]
            max_proba = float(max(proba))
        else:
            max_proba = None
    except Exception:
        return None

    fit_class = int(y_pred[0])

    if fit_class == 0:
        text = "Кандидат не подходит"
        recommendation = (
            "Рекомендация: кандидат не соответствует требованиям текущей вакансии по уровню. "
            "Можно рассмотреть на позицию Junior или стажёра, если такая открыта."
        )
    elif fit_class == 1:
        text = "Кандидат сомнительный, нужно собеседование"
        recommendation = (
            "Рекомендация: пригласить кандидата на собеседование, чтобы уточнить опыт, "
            "мотивацию и проверить профильные навыки на практике."
        )
    elif fit_class == 2:
        text = "Кандидат отлично подходит"
        recommendation = (
            "Рекомендация: кандидат хорошо соответствует требованиям. "
            "Рекомендуется пригласить на техническое и финальное собеседования."
        )
    else:
        text = f"Неизвестный класс пригодности: {fit_class}"
        recommendation = "Рекомендация: перепроверьте введённые данные кандидата и повторите оценку."

    return {
        "class": fit_class,
        "text": text,
        "probability": max_proba,
        "recommendation": recommendation,
    }


class NewVacancyFSM(StatesGroup):
    title = State()
    description = State()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    text_lines: List[str] = [
        "Привет! Я бот для первичного подбора персонала.",
        "",
        "Я умею:",
        "• создавать и хранить вакансии;",
        "• анализировать текстовые резюме (.txt);",
        "• сопоставлять кандидатов с вакансиями и выдавать рейтинг соответствия.",
        "",
        "Основные команды:",
        "/new_vacancy — создать новую вакансию",
        "/list_vacancies — список вакансий",
        "/help — справка",
        "",
        "Просто пришлите файл резюме в формате .txt, и я его проанализирую.",
        "",
        "Либо отправьте строку вида:",
        "experience=5, skills=8, age=30, education=1, match_score=70",
        "и я оценю пригодность кандидата с помощью ML-модели.",
        "",
        DISCLAIMER_TEXT,
    ]
    await message.answer("\n".join(text_lines))


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    text_lines: List[str] = [
        "Справка по боту:",
        "",
        "/new_vacancy — пошаговое создание вакансии (название + требования).",
        "/list_vacancies — просмотр всех активных вакансий.",
        "",
        "Чтобы проанализировать кандидата:",
        "1. Сохраните резюме в формате .txt.",
        "2. Отправьте файл боту как документ.",
        "3. Я извлеку навыки и опыт, сравню с вакансиями и выдам рейтинг.",
        "",
        "Альтернативный режим: числовая диагностика.",
        "Отправьте текстовое сообщение в формате:",
        "experience=5, skills=8, age=30, education=1, match_score=70",
        "Поля:",
        "- experience — опыт в годах (0–15)",
        "- skills — количество навыков (1–20)",
        "- age — возраст (20–60)",
        "- education — 0 (нет высшего) или 1 (есть высшее)",
        "- match_score — процент совпадения с требованиями (0–100, опционально)",
        "",
        DISCLAIMER_TEXT,
    ]
    await message.answer("\n".join(text_lines))


@router.message(Command("list_vacancies"))
async def cmd_list_vacancies(message: Message) -> None:
    vacancies = storage.list_vacancies()
    if not vacancies:
        await message.answer("Пока нет ни одной вакансии. Используйте /new_vacancy.")
        return

    lines: List[str] = ["Активные вакансии:"]
    for v in vacancies:
        short_desc = (v.description[:150] + "...") if len(v.description) > 150 else v.description
        lines.append(f"{v.id}. {v.title}\n   Требуемый опыт: {v.min_experience_years} лет\n   {short_desc}")
    lines.append("")
    lines.append(DISCLAIMER_TEXT)
    await message.answer("\n".join(lines))


@router.message(Command("new_vacancy"))
async def cmd_new_vacancy(message: Message, state: FSMContext) -> None:
    await state.set_state(NewVacancyFSM.title)
    await message.answer("Введите название вакансии.")


@router.message(NewVacancyFSM.title)
async def vacancy_title_entered(message: Message, state: FSMContext) -> None:
    await state.update_data(title=message.text or "")
    await state.set_state(NewVacancyFSM.description)
    await message.answer(
        "Введите описание требований к вакансии (навыки, стек, обязанности, требуемый опыт)."
    )


@router.message(NewVacancyFSM.description)
async def vacancy_description_entered(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    title = str(data.get("title", "")).strip()
    description = (message.text or "").strip()

    if not title:
        await message.answer("Название вакансии не задано, начните заново с /new_vacancy.")
        await state.clear()
        return

    if not description:
        await message.answer("Описание не может быть пустым. Введите требования к вакансии.")
        return

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
    await state.clear()

    msg_lines = [
        "Вакансия сохранена ✅",
        f"ID: {vacancy.id}",
        f"Название: {vacancy.title}",
        f"Навыки (извлечены автоматически): {', '.join(skills) if skills else 'не найдены'}",
        f"Требуемый опыт (оценка): {experience_years:.1f} лет",
        "",
        DISCLAIMER_TEXT,
    ]
    await message.answer("\n".join(msg_lines))


def _candidate_name_from_text(text: str) -> str:
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not first_line:
        return "Кандидат"
    if len(first_line) > 80:
        return "Кандидат"
    return first_line


def _education_flag_from_text(text: str) -> int:
    """
    Простая эвристика: определить наличие высшего образования по тексту резюме.
    """
    lowered = text.lower()
    markers = [
        "высшее образование",
        "бакалавр",
        "магистр",
        "специалист",
        "университет",
        "институт",
    ]
    return 1 if any(m in lowered for m in markers) else 0


async def _download_document(message: Message, document: Document) -> str:
    """
    Скачать содержимое документа как текст, совместимо с aiogram 3.7+.
    """
    bot = message.bot
    file = await bot.get_file(document.file_id)
    buffer = BytesIO()
    await bot.download(file, destination=buffer)
    buffer.seek(0)
    return buffer.read().decode("utf-8", errors="ignore")


@router.message(F.document)
async def handle_resume_document(message: Message) -> None:
    document = message.document
    if not document:
        return

    if not document.file_name.lower().endswith(".txt"):
        await message.answer("Пожалуйста, отправьте резюме в виде текстового файла (.txt).")
        return

    await message.answer("Получил файл, анализирую резюме. Это может занять несколько секунд...")

    try:
        text = await _download_document(message, document)
    except Exception:
        await message.answer("Не удалось скачать или прочитать файл резюме.")
        return

    name = _candidate_name_from_text(text)
    skills = nlp.extract_skills(text)
    experience_years = nlp.extract_experience_years(text)
    embedding = nlp.text_embedding(text)

    candidate = storage.Candidate(
        id=storage.next_candidate_id(),
        name=name,
        raw_text=text,
        skills=skills,
        experience_years=experience_years,
        embedding=embedding,
    )
    storage.add_candidate(candidate)

    vacancies = storage.list_vacancies()
    if not vacancies:
        msg_lines = [
            f"Резюме кандидата «{name}» проанализировано.",
            "",
            "Сейчас нет сохранённых вакансий. Создайте их командой /new_vacancy.",
            "",
            f"Навыки кандидата: {', '.join(skills) if skills else 'не найдены'}",
            f"Опыт (оценка): {experience_years:.1f} лет",
            "",
            DISCLAIMER_TEXT,
        ]
        await message.answer("\n".join(msg_lines))
        return

    vacancies_data = [
        (v.id, v.title, v.skills, v.min_experience_years, v.embedding) for v in vacancies
    ]
    matches = nlp.match_candidate_to_vacancies(
        candidate_embedding=embedding,
        candidate_skills=skills,
        candidate_experience=experience_years,
        vacancies=vacancies_data,
    )

    top_matches = matches[:3]

    # Подготовка агрегированного match_score для ML-оценки: берём лучший матч или 50 по умолчанию.
    best_match_score = top_matches[0]["score"] if top_matches else 50.0

    # Попытка выполнить ML-оценку пригодности на основе извлечённых признаков.
    ml_lines: List[str] = []
    features_for_ml = {
        "experience_years": float(experience_years),
        "skills_count": float(len(skills)),
        "age": 30.0,  # возраст не извлекаем из текста, используем типовое значение
        "education": float(_education_flag_from_text(text)),
        "match_score": float(best_match_score),
    }

    try:
        ml_pred = _predict_fit_level(features_for_ml)
    except RuntimeError:
        ml_pred = None

    if ml_pred:
        prob = ml_pred["probability"]
        ml_lines = [
            "",
            "ML-оценка пригодности кандидата:",
            f"Класс: {ml_pred['class']} — {ml_pred['text']}",
            f"Рекомендация: {ml_pred['recommendation']}",
        ]
        if prob is not None:
            ml_lines.append(f"Уверенность модели: {prob * 100:.1f}%")

    lines: List[str] = [
        f"Результаты анализа для кандидата: {name}",
        "",
        f"Навыки кандидата: {', '.join(skills) if skills else 'не найдены'}",
        f"Опыт (оценка): {experience_years:.1f} лет",
        "",
        "Топ подходящих вакансий:",
    ]

    if not top_matches:
        lines.append("Подходящих вакансий не найдено.")
    else:
        for m in top_matches:
            common_str = ", ".join(m["common_skills"]) if m["common_skills"] else "нет совпадающих навыков"
            missing_str = ", ".join(m["missing_skills"]) if m["missing_skills"] else "нет"
            recommendation = _text_recommendation_for_match(m["score"])
            lines.extend(
                [
                    f"- {m['title']} (ID {m['vacancy_id']}): {m['score']:.1f}%",
                    f"  Совпадающие навыки: {common_str}",
                    f"  Не хватает навыков: {missing_str}",
                    f"  Рекомендация: {recommendation}",
                    "",
                ]
            )

    # Добавляем ML-блок, если он успешно посчитан.
    if ml_lines:
        lines.extend(ml_lines)

    lines.append(DISCLAIMER_TEXT)
    await message.answer("\n".join(lines))


def _parse_kv_text(text: str) -> Optional[Dict[str, float]]:
    """
    Распарсить строку формата "experience=5, skills=8, age=30, education=1, match_score=70".
    """
    if not text:
        return None

    parts = text.split(",")
    result: Dict[str, float] = {}
    for raw in parts:
        if "=" not in raw:
            continue
        key, value = raw.split("=", maxsplit=1)
        key = key.strip().lower()
        value = value.strip().replace(",", ".")
        if not key or not value:
            continue
        try:
            result[key] = float(value)
        except ValueError:
            continue

    if not result:
        return None

    # Переименование коротких ключей в используемые в модели
    mapped: Dict[str, float] = {}
    for k, v in result.items():
        if k == "experience":
            mapped["experience_years"] = v
        elif k == "skills":
            mapped["skills_count"] = v
        else:
            mapped[k] = v

    # Значения по умолчанию, если чего-то не хватает
    mapped.setdefault("experience_years", 0.0)
    mapped.setdefault("skills_count", 1.0)
    mapped.setdefault("age", 30.0)
    mapped.setdefault("education", 0.0)
    mapped.setdefault("match_score", 50.0)

    return mapped


def _text_recommendation_for_match(score: float) -> str:
    """
    Сгенерировать рекомендацию по результатам матчинга резюме и вакансии.
    Используется при загрузке резюме из файла.
    """
    if score >= 75:
        return (
            "Кандидат хорошо соответствует требованиям данной вакансии. "
            "Рекомендуется пригласить на техническое и финальное собеседования."
        )
    if score >= 45:
        return (
            "Соответствие средней силы. Имеет смысл провести собеседование, "
            "чтобы подробнее выяснить опыт, мотивацию и проверить ключевые навыки."
        )
    return (
        "Низкое соответствие требованиям вакансии. Можно рассмотреть кандидата "
        "на более junior-позиции или по другим открытым вакансиям."
    )


@router.message(F.text)
async def handle_numeric_profile(message: Message) -> None:
    """
    Обработка текстовых сообщений в формате "ключ=значение" для числовой диагностики.
    """
    text = (message.text or "").strip()
    if not text:
        return

    features = _parse_kv_text(text)
    if not features:
        # Не удалось распарсить как профиль — просто ничего не делаем,
        # чтобы не мешать другим хендлерам или сценариям.
        return

    try:
        prediction = _predict_fit_level(features)
    except RuntimeError as exc:
        await message.answer(
            f"Не удалось выполнить числовую оценку кандидата: {exc}"
        )
        return

    if not prediction:
        await message.answer(
            "Не удалось интерпретировать параметры или выполнить предсказание."
        )
        return

    prob = prediction["probability"]
    lines = [
        "Результат числовой диагностики кандидата:",
        "",
        f"Опыт (лет): {features['experience_years']:.1f}",
        f"Количество навыков: {features['skills_count']:.1f}",
        f"Возраст: {features['age']:.1f}",
        f"Высшее образование: {'да' if int(features['education']) == 1 else 'нет'}",
        f"Процент совпадения с требованиями: {features['match_score']:.1f}",
        "",
        f"Класс: {prediction['class']} — {prediction['text']}",
        f"Рекомендация: {prediction['recommendation']}",
    ]
    if prob is not None:
        lines.append(f"Уверенность модели: {prob * 100:.1f}%")
    lines.append("")
    lines.append(DISCLAIMER_TEXT)

    await message.answer("\n".join(lines))

