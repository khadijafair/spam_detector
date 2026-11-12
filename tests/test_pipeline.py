"""Tests for the shared NLP pipeline and the FastAPI endpoint.

The API tests need the trained artifacts in models/ (python -m research.train_model)
and are skipped otherwise.
"""
from pathlib import Path

import pytest

import core.config as config
from core.spam_utils import clean_text_advanced, is_borderline, sigmoid

HAS_MODELS = all(Path(p).exists() for p in (config.MODEL_PATH, config.VECTORIZER_PATH))


def test_cleaning_replaces_urls_emails_money_and_numbers():
    out = clean_text_advanced("Win $500 now at http://x.io or mail me@a.com, code 1234")
    for token in ("urltoken", "emailtoken", "moneytoken", "numbertoken"):
        assert token in out
    assert "http" not in out and "@" not in out


def test_cleaning_lowercases_and_stems():
    assert clean_text_advanced("CLICKING Winners") == "click winner"


def test_borderline_zone():
    assert is_borderline(55.0)
    assert not is_borderline(95.0)


def test_sigmoid_is_overflow_safe():
    assert sigmoid(10_000) == 1.0
    assert sigmoid(-10_000) == 0.0


@pytest.fixture(scope="module")
def client():
    if not HAS_MODELS:
        pytest.skip("models/ not trained")
    from fastapi.testclient import TestClient
    import api
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_flags_phishing(client):
    r = client.post("/predict", json={"text": (
        "URGENT: your account has been suspended. Verify your password and credit card "
        "within 24 hours at http://secure-login-update.com or it will be closed.")})
    assert r.status_code == 200
    body = r.json()
    assert body["is_spam"] is True
    assert 50 <= body["confidence_percent"] <= 100
    assert body["top_words"]


def test_predict_accepts_legit_email(client):
    r = client.post("/predict", json={"text": (
        "Hi team, attached are the notes from this morning's meeting. The next review "
        "of the project schedule is on Thursday at 10am in room B. Thanks, Sarah")})
    assert r.status_code == 200
    assert r.json()["is_spam"] is False


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
