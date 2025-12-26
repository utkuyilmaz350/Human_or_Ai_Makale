from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_models(random_state: int) -> dict[str, Pipeline]:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=120_000, min_df=2)

    lr = LogisticRegression(max_iter=4000, n_jobs=None)
    nb = MultinomialNB()
    svc = CalibratedClassifierCV(LinearSVC(), method="sigmoid", cv=3)

    return {
        "logreg": Pipeline([("tfidf", tfidf), ("clf", lr)]),
        "nb": Pipeline([("tfidf", tfidf), ("clf", nb)]),
        "linear_svc_cal": Pipeline([("tfidf", tfidf), ("clf", svc)]),
    }


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("dataset must have columns: text,label")

    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    artifacts_dir = Path(args.artifacts)
    ensure_dir(artifacts_dir)

    models = build_models(random_state=args.random_state)
    metrics: dict[str, dict[str, str | float]] = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        report = classification_report(y_test, pred, digits=4)

        joblib.dump(pipe, artifacts_dir / f"{name}.joblib")

        metrics[name] = {"accuracy": acc, "classification_report": report}

        disp = ConfusionMatrixDisplay.from_predictions(y_test, pred)
        disp.figure_.suptitle(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(artifacts_dir / f"confusion_{name}.png", dpi=180)
        plt.close(disp.figure_)

    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
