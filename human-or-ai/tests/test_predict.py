import pytest

from human_or_ai.predict.predict import predict_one


class DummyModel:
    classes_ = ["ai", "human"]

    def predict_proba(self, X):
        return [[0.7, 0.3] for _ in X]


def test_predict_one_returns_probabilities() -> None:
    pred = predict_one(DummyModel(), "this is a sufficiently long text for testing")
    assert pred.label == "ai"
    assert 0.0 <= pred.proba_ai <= 1.0
    assert 0.0 <= pred.proba_human <= 1.0


def test_predict_one_raises_on_short_text() -> None:
    with pytest.raises(ValueError):
        predict_one(DummyModel(), "short")
