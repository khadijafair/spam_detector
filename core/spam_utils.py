"""
spam_utils.py — Shared NLP pipeline used by every interface (API, Streamlit,
Telegram bot, training scripts). Keeping this logic in one place means a fix
here applies everywhere at once.
"""
import re
import math
import joblib
import requests
from functools import lru_cache
from nltk.stem.porter import PorterStemmer
from langdetect import detect, DetectorFactory, LangDetectException
import core.config as config

DetectorFactory.seed = 0  # deterministic langdetect results

stemmer = PorterStemmer()

_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_EMAIL = re.compile(r'\S+@\S+')
_RE_MONEY = re.compile(r'\$\d+(?:[\.,]\d+)?|£\d+(?:[\.,]\d+)?|\b\d+\s*(?:dollars|usd|euro|euros)\b')
_RE_PHONE = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
_RE_NUMBER = re.compile(r'\b\d+\b')
_RE_NONALPHA = re.compile(r'[^a-zA-Z\s]')


def clean_text_advanced(text: str) -> str:
    """Cleaning + tokenization + stemming, identical to what the model was trained on."""
    text = str(text).lower()
    text = _RE_URL.sub(' urltoken ', text)
    text = _RE_EMAIL.sub(' emailtoken ', text)
    text = _RE_MONEY.sub(' moneytoken ', text)
    text = _RE_PHONE.sub(' phonetoken ', text)
    text = _RE_NUMBER.sub(' numbertoken ', text)
    text = _RE_NONALPHA.sub(' ', text)
    words = text.split()
    return ' '.join(stemmer.stem(word) for word in words)


@lru_cache(maxsize=1)
def load_artifacts(model_path: str | None = None, vectorizer_path: str | None = None):
    model_path = model_path or config.MODEL_PATH
    vectorizer_path = vectorizer_path or config.VECTORIZER_PATH
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def sigmoid(x: float) -> float:
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def load_calibrator(calibrator_path: str):
    """Loads the Platt-scaling calibrator (trained by retrain_feedback.py).
    Returns None if it doesn't exist yet -> caller falls back to sigmoid()."""
    try:
        return joblib.load(calibrator_path)
    except Exception:
        return None


def predict_with_confidence(text: str, model, vectorizer, calibrator=None):
    """Returns (is_spam, confidence_pct, raw_score). Uses the calibrator for a
    properly calibrated confidence if available, otherwise a rough sigmoid."""
    clean = clean_text_advanced(text)
    vectorized = vectorizer.transform([clean])
    score = model.decision_function(vectorized)[0]
    is_spam = score > 0

    proba_spam = calibrator.predict_proba([[score]])[0][1] if calibrator is not None else sigmoid(score)
    confidence_pct = round((proba_spam if is_spam else 1 - proba_spam) * 100, 1)
    return is_spam, confidence_pct, score


def detect_language(text: str) -> str:
    """Returns an ISO 639-1 code, or 'unknown' if the text is too short (< 3 words)
    for langdetect to be reliable."""
    if len(text.split()) < 3:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


TRANSLATE_PROMPT_TEMPLATE = (
    "Translate the following text to English. Output ONLY the raw translation, "
    "with no preamble, no quotes, no explanation, no markdown.\n\n"
    "Text:\n\"\"\"\n{text}\n\"\"\""
)


def translate_to_english_sync(text: str, ollama_url: str, model_llm: str, timeout: float = 25.0):
    """Blocking Ollama call (used by the Streamlit app and the API)."""
    payload = {"model": model_llm, "prompt": TRANSLATE_PROMPT_TEMPLATE.format(text=text), "stream": False}
    try:
        resp = requests.post(ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        translated = resp.json().get("response", "").strip()
        return translated or None
    except Exception:
        return None


def is_borderline(confidence_pct: float, margin: float = 15.0) -> bool:
    """True if the score is too close to 50% to be trusted -- typically a
    text outside the model's training domain/language."""
    return confidence_pct <= (50 + margin)


def load_corrections(feedback_log_path: str) -> dict:
    """Reads feedback.jsonl and builds a {exact_text: corrected_label} cache
    for entries explicitly flagged as wrong. Not model retraining -- just a
    safety net against repeating an error on an identical text. Use
    retrain_feedback.py for actual learning."""
    import json
    import os

    corrections = {}
    if not os.path.exists(feedback_log_path):
        return corrections
    with open(feedback_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("marked_correct") is False:
                wrong_label = entry["predicted_label"]
                corrections[entry["text"].strip()] = "HAM" if wrong_label == "SPAM" else "SPAM"
    return corrections


def explain_prediction(text: str, model, vectorizer, top_n: int = 6):
    """Returns the (word, contribution) pairs that most influenced the verdict,
    computed as tfidf_weight * svm_coefficient, sorted by |contribution|."""
    clean = clean_text_advanced(text)
    vectorized = vectorizer.transform([clean])
    coef = model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()

    contributions = [
        (feature_names[idx], float(vectorized[0, idx] * coef[idx]))
        for idx in vectorized.nonzero()[1]
    ]
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)
    return contributions[:top_n]