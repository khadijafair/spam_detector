"""
predict.py — Quick CLI prediction utility, using the shared pipeline
(core.spam_utils) instead of a duplicated cleaning function, so it always
matches the behaviour of the bot/API/Streamlit app exactly.
"""
import core.config as config
from core.spam_utils import (
    load_artifacts, load_calibrator, predict_with_confidence,
    explain_prediction, detect_language, translate_to_english_sync,
)

print("🧠 Loading serialized model and vectorizer...")
model, vectorizer = load_artifacts(config.MODEL_PATH, config.VECTORIZER_PATH)
calibrator = load_calibrator(config.CALIBRATOR_PATH)

# 📝 Put the raw email you want to test here!
email_a_tester = """
Subject: Urgent Security Notification: Unusual Activity Detected on your Corporate Account

Dear Valued Customer,

Our automated global fraud detection systems have flags a series of suspicious and unauthorized login attempts on your corporate online banking profile within the last few hours. These connection requests originated from multiple blacklisted IP addresses located outside your usual geographic operational area.

To protect your financial funds and prevent identity theft, we have temporarily placed a restricted hold on your outgoing transactions and online profile access. To resolve this security lock and verify your legitimate ownership of the account, you are required to complete our secure synchronization process within 24 hours of receiving this notification.

Please click the secure network link below to log in to your dashboard, confirm your personal identification numbers, and update your multi-factor security preferences. Failure to complete this mandatory verification will result in the permanent suspension of your corporate assets and a potential legal review.

http://secure-banking-update-portal-auth.com/login/auth-session

Thank you for your immediate attention to this critical security matter.
"""

lang = detect_language(email_a_tester)
text_for_model = email_a_tester
if lang not in ("en", "unknown"):
    translated = translate_to_english_sync(email_a_tester, config.OLLAMA_URL, config.MODEL_LLM)
    if translated:
        text_for_model = translated
        print(f"🌐 Translated from '{lang}' before analysis.")

is_spam, confidence, score = predict_with_confidence(text_for_model, model, vectorizer, calibrator)
top_words = explain_prediction(text_for_model, model, vectorizer, top_n=8)

print("\n🔮 PREDICTION RESULT:")
print(f"Verdict         : {'🔴 SPAM / PHISHING' if is_spam else '🟢 LEGITIMATE (HAM)'}")
print(f"Confidence      : {confidence}%")
print(f"Detected lang   : {lang}")
print(f"Top words       : {', '.join(f'{w}({c:+.2f})' for w, c in top_words)}")