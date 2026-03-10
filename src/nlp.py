from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


@lru_cache(maxsize=1)
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    model.eval()
    return tokenizer, model


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def text_embedding(text: str) -> List[float]:
    tokenizer, model = _load_model()
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        model_output = model(**encoded)
    sentence_embeddings = _mean_pooling(model_output, encoded["attention_mask"])
    embedding = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)[0]
    return embedding.tolist()


SKILL_KEYWORDS = [
    "python",
    "django",
    "flask",
    "fastapi",
    "sql",
    "postgresql",
    "mysql",
    "mongodb",
    "docker",
    "kubernetes",
    "linux",
    "git",
    "ci/cd",
    "pandas",
    "numpy",
    "scikit-learn",
    "machine learning",
    "ml",
    "data analysis",
    "javascript",
    "typescript",
    "react",
    "vue",
    "angular",
]


def extract_skills(text: str) -> List[str]:
    lowered = text.lower()
    skills = []
    for kw in SKILL_KEYWORDS:
        if kw.lower() in lowered:
            skills.append(kw)
    return sorted(set(skills))


def extract_experience_years(text: str) -> float:
    pattern = re.compile(r"(\d+(\.\d+)?)\s*(год|года|лет)", re.IGNORECASE)
    matches = pattern.findall(text)
    if not matches:
        return 0.0
    years = [float(m[0].replace(",", ".")) for m in matches]
    return max(years) if years else 0.0


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def match_candidate_to_vacancies(
    candidate_embedding: List[float],
    candidate_skills: List[str],
    candidate_experience: float,
    vacancies: List[Tuple[int, str, List[str], float, List[float]]],
) -> List[dict]:
    results = []
    cand_skills_set = set(s.lower() for s in candidate_skills)
    for vac_id, title, vac_skills, vac_min_exp, vac_emb in vacancies:
        sim = cosine_similarity(candidate_embedding, vac_emb)
        vac_skills_set = set(s.lower() for s in vac_skills)
        common_skills = sorted(s for s in vac_skills if s.lower() in cand_skills_set)
        missing_skills = sorted(s for s in vac_skills if s.lower() not in cand_skills_set)
        if vac_skills:
            skills_score = len(common_skills) / len(vac_skills)
        else:
            skills_score = 0.5
        if vac_min_exp <= 0:
            exp_score = 0.5
        else:
            exp_score = min(candidate_experience / vac_min_exp, 1.0)

        sem_weight = 0.5
        skills_weight = 0.3
        exp_weight = 0.2

        total_score = (
            sem_weight * sim + skills_weight * skills_score + exp_weight * exp_score
        )
        results.append(
            {
                "vacancy_id": vac_id,
                "title": title,
                "score": max(0.0, min(total_score * 100, 100.0)),
                "common_skills": common_skills,
                "missing_skills": missing_skills,
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results

