from __future__ import annotations

"""
Скрипт для обучения моделей оценки пригодности кандидатов.

Создаёт синтетический датасет, обучает несколько моделей,
выводит сравнительную таблицу метрик и сохраняет лучшую модель
и скейлер признаков в директорию models/.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]


def generate_synthetic_dataset(n_samples: int = 2000, random_state: int = 42) -> Dataset:
    """
    Сгенерировать синтетический датасет кандидатов.

    Признаки:
        - experience_years: стаж (0–15 лет)
        - skills_count: количество навыков (1–20)
        - age: возраст (20–60 лет)
        - education: 0 — нет высшего, 1 — есть высшее
        - match_score: процент совпадения с требованиями (0–100)

    Целевая переменная:
        - fit_level: 0 — не подходит, 1 — сомнительно, 2 — подходит
    """
    rng = np.random.default_rng(random_state)

    experience_years = rng.integers(0, 16, size=n_samples)
    skills_count = rng.integers(1, 21, size=n_samples)
    age = rng.integers(20, 61, size=n_samples)
    education = rng.integers(0, 2, size=n_samples)
    match_score = rng.integers(0, 101, size=n_samples)

    # Простое эвристическое правило для fit_level с шумом
    base_score = (
        0.4 * (experience_years / 15)
        + 0.3 * (skills_count / 20)
        + 0.1 * education
        + 0.2 * (match_score / 100)
    )

    noise = rng.normal(0, 0.05, size=n_samples)
    total = np.clip(base_score + noise, 0.0, 1.0)

    fit_level = np.zeros(n_samples, dtype=int)
    fit_level[total >= 0.6] = 2  # подходит
    fit_level[(total >= 0.35) & (total < 0.6)] = 1  # сомнительно

    X = np.column_stack(
        [experience_years, skills_count, age, education, match_score]
    )
    feature_names = [
        "experience_years",
        "skills_count",
        "age",
        "education",
        "match_score",
    ]
    return Dataset(X=X, y=fit_level, feature_names=feature_names)


def train_and_evaluate_models(
    dataset: Dataset, test_size: float = 0.25, random_state: int = 42
) -> Tuple[Dict[str, Dict[str, float]], StandardScaler, Dict[str, object]]:
    """
    Обучить и сравнить несколько моделей.

    Возвращает:
        - метрики по моделям,
        - обученный скейлер,
        - словарь {имя_модели: обученный_объект}.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models: Dict[str, object] = {
        # Параметр multi_class не используем, чтобы избежать конфликтов между версиями sklearn.
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.1,
            loss_function="MultiClass",
            verbose=False,
            random_state=random_state,
        ),
    }

    metrics: Dict[str, Dict[str, float]] = {}
    trained_models: Dict[str, object] = {}

    for name, model in models.items():
        try:
            # Обучение модели
            model.fit(X_train_scaled, y_train)

            # Базовые метрики
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            metrics[name] = {"accuracy": acc, "f1_macro": f1}
            trained_models[name] = model

            # Подробный отчёт по модели
            print(f"\n===== МОДЕЛЬ: {name} =====")
            print(f"Accuracy: {acc:.4f}")
            print(f"F1-macro: {f1:.4f}\n")

            print("Classification report:")
            # digits=3 для красивого форматирования, как в примере
            print(classification_report(y_test, y_pred, digits=3))

            print("Confusion matrix:")
            print(confusion_matrix(y_test, y_pred))
        except Exception as exc:
            print(f"[WARN] Ошибка при обучении модели {name}: {exc}", file=sys.stderr)

    return metrics, scaler, trained_models


def print_metrics_table(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Сводная таблица метрик по всем моделям (похожа на пример из курсовой).
    """
    if not metrics:
        print("Нет рассчитанных метрик.")
        return

    rows = []
    for name, vals in metrics.items():
        rows.append(
            {
                "model": name,
                "accuracy": vals["accuracy"],
                "f1_macro": vals["f1_macro"],
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)

    print("\nСводная таблица метрик:")
    # Форматирование чисел до 6 знаков после запятой, как в примере
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    best_row = df.iloc[0]
    print("\nЛучшая модель по F1-macro:")
    print(best_row)


def select_best_model(
    metrics: Dict[str, Dict[str, float]], trained_models: Dict[str, object]
) -> Tuple[str, object]:
    """
    Выбрать лучшую модель по F1-macro.
    """
    if not metrics:
        raise RuntimeError("Не удалось обучить ни одной модели.")

    best_name = None
    best_f1 = -1.0

    for name, vals in metrics.items():
        f1 = vals["f1_macro"]
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    if best_name is None or best_name not in trained_models:
        raise RuntimeError("Не удалось выбрать лучшую модель.")

    return best_name, trained_models[best_name]


def save_model_and_scaler(
    model: object, scaler: StandardScaler, model_name: str
) -> None:
    """
    Сохранить модель и скейлер в файлы models/candidate_model.pkl и models/scaler.pkl.
    """
    model_path = MODELS_DIR / "candidate_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    try:
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Модель ({model_name}) сохранена в: {model_path}")
        print(f"Скейлер сохранён в: {scaler_path}")
    except Exception as exc:
        raise RuntimeError(f"Ошибка при сохранении модели или скейлера: {exc}") from exc


def main() -> None:
    """
    Точка входа для обучения и сохранения модели.
    """
    print("Генерация синтетического датасета...")
    dataset = generate_synthetic_dataset()

    print("Обучение и оценка моделей...")
    metrics, scaler, trained_models = train_and_evaluate_models(dataset)

    print("\nСравнение моделей:")
    print_metrics_table(metrics)

    best_name, best_model = select_best_model(metrics, trained_models)
    print(f"\nЛучшая модель по F1-macro: {best_name}")

    save_model_and_scaler(best_model, scaler, best_name)


if __name__ == "__main__":
    main()

