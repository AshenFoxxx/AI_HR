from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VACANCIES_PATH = DATA_DIR / "vacancies.json"
CANDIDATES_PATH = DATA_DIR / "candidates.json"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    _ensure_data_dir()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _write_json(path: Path, data: Any) -> None:
    _ensure_data_dir()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@dataclass
class Vacancy:
    id: int
    title: str
    description: str
    skills: List[str]
    min_experience_years: float
    embedding: List[float]


@dataclass
class Candidate:
    id: int
    name: str
    raw_text: str
    skills: List[str]
    experience_years: float
    embedding: List[float]


def list_vacancies() -> List[Vacancy]:
    raw = _read_json(VACANCIES_PATH)
    return [Vacancy(**item) for item in raw]


def add_vacancy(vacancy: Vacancy) -> None:
    vacancies = _read_json(VACANCIES_PATH)
    vacancies.append(asdict(vacancy))
    _write_json(VACANCIES_PATH, vacancies)


def next_vacancy_id() -> int:
    vacancies = _read_json(VACANCIES_PATH)
    if not vacancies:
        return 1
    return max(v["id"] for v in vacancies) + 1


def list_candidates() -> List[Candidate]:
    raw = _read_json(CANDIDATES_PATH)
    return [Candidate(**item) for item in raw]


def add_candidate(candidate: Candidate) -> None:
    candidates = _read_json(CANDIDATES_PATH)
    candidates.append(asdict(candidate))
    _write_json(CANDIDATES_PATH, candidates)


def next_candidate_id() -> int:
    candidates = _read_json(CANDIDATES_PATH)
    if not candidates:
        return 1
    return max(c["id"] for c in candidates) + 1

