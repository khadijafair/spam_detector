"""
config.py — Centralized configuration, loaded from .env. No secrets in code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # only required for telegram_bot.py

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_LLM = os.getenv("MODEL_LLM", "llama3.2:3b")

# Paths resolved relative to the project root, not the current working
# directory, so every entry point behaves the same regardless of where
# it's launched from.
MODEL_PATH = os.getenv("MODEL_PATH") or str(MODELS_DIR / "best_spam_model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH") or str(MODELS_DIR / "tfidf_vectorizer.pkl")
CALIBRATOR_PATH = os.getenv("CALIBRATOR_PATH") or str(MODELS_DIR / "confidence_calibrator.pkl")
FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH") or str(DATA_DIR / "feedback.jsonl")
DATASET_MASTER_PATH = DATA_DIR / "dataset_master_clean.csv"
RAW_DATASET_DIR = DATA_DIR / "dataset"

# Below this word count, a message with no fraud signal is treated as chat
# rather than sent for analysis (see telegram_bot.py).
SHORT_MSG_WORD_THRESHOLD = int(os.getenv("SHORT_MSG_WORD_THRESHOLD", "25"))