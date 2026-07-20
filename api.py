"""
api.py — FastAPI production endpoint.

Shares the exact same pipeline as the Telegram bot and the Streamlit app
(core.spam_utils): language detection + translation, calibrated confidence,
explainability. Run with: uvicorn api:app --reload
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import core.config as config
from core.spam_utils import (
    load_artifacts, load_calibrator, predict_with_confidence,
    explain_prediction, detect_language, translate_to_english_sync,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sentinel-spam-api")

app = FastAPI(
    title="Spam & Phishing Detector API",
    description="Real-time classification powered by a calibrated Linear SVM, "
                 "with automatic translation and LLM-assisted arbitration on borderline scores.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
vectorizer = None
calibrator = None


@app.on_event("startup")
def load_model_artifacts():
    global model, vectorizer, calibrator
    try:
        model, vectorizer = load_artifacts(config.MODEL_PATH, config.VECTORIZER_PATH)
        logger.info("Model and vectorizer loaded from %s / %s", config.MODEL_PATH, config.VECTORIZER_PATH)
    except Exception as e:
        logger.error("Failed to load model artifacts: %s. Run train_model.py first.", e)
        raise

    calibrator = load_calibrator(config.CALIBRATOR_PATH)
    if calibrator is None:
        logger.warning("No confidence calibrator found -- run retrain_feedback.py once to generate one.")


class EmailPayload(BaseModel):
    text: str = Field(..., min_length=1, description="Raw message/email text to analyze")


class WordContribution(BaseModel):
    word: str
    contribution: float


class PredictionResponse(BaseModel):
    is_spam: bool
    verdict: str
    confidence_percent: float
    detected_language: str
    translated: bool
    top_words: list[WordContribution]


@app.get("/")
def read_root():
    return {"status": "online", "model": "Linear SVM (calibrated)" if calibrator else "Linear SVM", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_email(payload: EmailPayload):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet, please retry shortly.")

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")

    try:
        lang = detect_language(text)
        text_for_model = text
        translated = False
        if lang not in ("en", "unknown"):
            translation = translate_to_english_sync(text, config.OLLAMA_URL, config.MODEL_LLM)
            if translation:
                text_for_model = translation
                translated = True

        is_spam, confidence, _score = predict_with_confidence(text_for_model, model, vectorizer, calibrator)
        top_words = explain_prediction(text_for_model, model, vectorizer, top_n=8)

        return PredictionResponse(
            is_spam=is_spam,
            verdict="🔴 SPAM / PHISHING" if is_spam else "🟢 LEGITIMATE (HAM)",
            confidence_percent=confidence,
            detected_language=lang,
            translated=translated,
            top_words=[WordContribution(word=w, contribution=c) for w, c in top_words],
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")