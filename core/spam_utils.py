"""
spam_utils.py — Pipeline NLP partagé entre le bot Telegram et l'interface Streamlit.
Centraliser cette logique évite les divergences entre les deux interfaces
(un bug corrigé ici est corrigé partout).
"""
import re
import math
import joblib
import requests
from functools import lru_cache
from nltk.stem.porter import PorterStemmer
from langdetect import detect, DetectorFactory, LangDetectException
import core.config as config

DetectorFactory.seed = 0  # résultats déterministes (langdetect est sinon non-reproductible)

stemmer = PorterStemmer()

# Regex précompilées (perf : évite de les recompiler à chaque appel)
_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_EMAIL = re.compile(r'\S+@\S+')
_RE_MONEY = re.compile(r'\$\d+(?:[\.,]\d+)?|£\d+(?:[\.,]\d+)?|\b\d+\s*(?:dollars|usd|euro|euros)\b')
_RE_PHONE = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
_RE_NUMBER = re.compile(r'\b\d+\b')
_RE_NONALPHA = re.compile(r'[^a-zA-Z\s]')


def clean_text_advanced(text: str) -> str:
    """Nettoyage + tokenisation + stemming, identique à l'entraînement du modèle."""
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
def load_artifacts(model_path: str | None = None,
                    vectorizer_path: str | None = None):
    """Chargement mis en cache (une seule fois par process)."""
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
    """
    Charge le calibrateur de confiance (régression logistique de Platt scaling
    entraînée par retrain_from_feedback.py). Retourne None s'il n'existe pas
    encore (avant le premier ré-entraînement) -> fallback sur le sigmoid brut.
    """
    try:
        return joblib.load(calibrator_path)
    except Exception:
        return None


def predict_with_confidence(text: str, model, vectorizer, calibrator=None):
    """
    Retourne (is_spam, confidence_pct, raw_score).
    Si un calibrateur (Platt scaling) est fourni, la confiance vient d'une
    vraie probabilité apprise sur des données de validation -- beaucoup plus
    fiable que l'approximation sigmoid brute utilisée par défaut.
    """
    clean = clean_text_advanced(text)
    vectorized = vectorizer.transform([clean])
    score = model.decision_function(vectorized)[0]
    is_spam = score > 0

    if calibrator is not None:
        proba_spam = calibrator.predict_proba([[score]])[0][1]
    else:
        proba_spam = sigmoid(score)

    confidence_pct = round((proba_spam if is_spam else 1 - proba_spam) * 100, 1)
    return is_spam, confidence_pct, score


def detect_language(text: str) -> str:
    """
    Retourne un code langue ISO 639-1 ('en', 'fr', 'es', ...) ou 'unknown'
    si le texte est trop court pour être fiable (langdetect n'est pas fiable
    en dessous de ~3 mots).
    """
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
    """Version synchrone (utilisée par l'interface Streamlit) : appel bloquant à Ollama."""
    payload = {"model": model_llm, "prompt": TRANSLATE_PROMPT_TEMPLATE.format(text=text), "stream": False}
    try:
        resp = requests.post(ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        translated = resp.json().get("response", "").strip()
        return translated or None
    except Exception:
        return None


def is_borderline(confidence_pct: float, margin: float = 15.0) -> bool:
    """
    True si le score est trop proche de 50% pour être fiable (le modèle
    n'a quasiment aucun signal net sur ce texte -- typiquement un texte
    hors du domaine/langue d'entraînement). Un verdict borderline devrait
    être signalé comme incertain plutôt qu'affiché avec la même autorité
    qu'un 95%.
    """
    return confidence_pct <= (50 + margin)


def load_corrections(feedback_log_path: str) -> dict:
    """
    Relit le journal de feedback et construit un cache {texte_exact: label_corrigé}
    pour les cas explicitement marqués comme erronés par un utilisateur.
    Ce n'est PAS un ré-entraînement du modèle, juste un filet de sécurité pour
    ne pas répéter une erreur déjà signalée sur un texte identique. Le vrai
    correctif reste de ré-entraîner le modèle avec ces exemples.
    """
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
                corrected_label = "HAM" if wrong_label == "SPAM" else "SPAM"
                corrections[entry["text"].strip()] = corrected_label
    return corrections


def explain_prediction(text: str, model, vectorizer, top_n: int = 6):
    """
    Explicabilité "gratuite" : pour un SVM linéaire, chaque mot présent dans
    le message contribue au score via (poids_tfidf * coefficient_du_mot).
    On retourne les mots qui ont le plus tiré la décision vers SPAM ou HAM.
    Retourne une liste de tuples (mot, contribution) triée par |contribution| décroissante.
    """
    clean = clean_text_advanced(text)
    vectorized = vectorizer.transform([clean])
    coef = model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()

    contributions = []
    # vectorized est une matrice creuse : on ne regarde que les indices non nuls
    for idx in vectorized.nonzero()[1]:
        weight = vectorized[0, idx] * coef[idx]
        contributions.append((feature_names[idx], float(weight)))

    contributions.sort(key=lambda t: abs(t[1]), reverse=True)
    return contributions[:top_n]