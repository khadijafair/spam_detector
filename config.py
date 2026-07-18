"""
config.py — Toute la configuration sensible vient du .env, JAMAIS en dur dans le code.
Copie .env.example en .env et remplis tes propres valeurs.
"""
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_LLM = os.getenv("MODEL_LLM", "llama3.2:3b")

MODEL_PATH = os.getenv("MODEL_PATH", "best_spam_model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "tfidf_vectorizer.pkl")

# Nombre de mots en dessous duquel on considère le message "court" (SMS-like)
SHORT_MSG_WORD_THRESHOLD = int(os.getenv("SHORT_MSG_WORD_THRESHOLD", "25"))

FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH", "feedback.jsonl")