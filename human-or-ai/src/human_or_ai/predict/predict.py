from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib

from human_or_ai.data_clean.clean import normalize_text


@dataclass(frozen=True)
class Prediction:
    label: str
    proba_ai: float
    proba_human: float


def _get_proba(pipe: Any, text: str) -> tuple[float, float]:
    if not hasattr(pipe, "predict_proba"):
        raise ValueError("Model does not support predict_proba")

    proba = pipe.predict_proba([text])[0]
    classes = list(pipe.classes_)
    idx_ai = classes.index("ai")
    idx_h = classes.index("human")
    return float(proba[idx_ai]), float(proba[idx_h])


def load_model(path: str):
    return joblib.load(path)


def predict_one(pipe: Any, text: str) -> Prediction:
    t = normalize_text(text)
    if len(t) < 20:
        raise ValueError("Text too short")

    proba_ai, proba_human = _get_proba(pipe, t)
    label = "ai" if proba_ai >= proba_human else "human"
    return Prediction(label=label, proba_ai=proba_ai, proba_human=proba_human)
