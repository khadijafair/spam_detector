"""
config.py — Toute la configuration sensible vient du .env, JAMAIS en dur dans le code.
Copie .env.example en .env et remplis tes propres valeurs.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # requis seulement pour telegram_bot.py, vérifié à son démarrage

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_LLM = os.getenv("MODEL_LLM", "llama3.2:3b")

# Chemins résolus par rapport à la racine du projet (et non au dossier
# depuis lequel le script est lancé) : évite les incohérences si le bot et
# l'interface sont démarrés depuis des terminaux/dossiers différents.
MODEL_PATH = os.getenv("MODEL_PATH") or str(PROJECT_ROOT / "models" / "best_spam_model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH") or str(PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl")
CALIBRATOR_PATH = os.getenv("CALIBRATOR_PATH") or str(PROJECT_ROOT / "models" / "confidence_calibrator.pkl")
FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH") or str(DATA_DIR / "feedback.jsonl")
DATASET_MASTER_PATH = DATA_DIR / "dataset_master_clean.csv"

# Nombre de mots en dessous duquel on considère le message "court" (SMS-like)
SHORT_MSG_WORD_THRESHOLD = int(os.getenv("SHORT_MSG_WORD_THRESHOLD", "25"))