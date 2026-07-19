import re
import joblib
from nltk.stem.porter import PorterStemmer
import core.config as config

stemmer = PorterStemmer()

def clean_text_advanced(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' urltoken ', text)
    text = re.sub(r'\S+@\S+', ' emailtoken ', text)
    text = re.sub(r'\$\d+(?:[\.,]\d+)?|£\d+(?:[\.,]\d+)?|\b\d+\s*(?:dollars|usd|euro|euros)\b', ' moneytoken ', text)
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', ' phonetoken ', text)
    text = re.sub(r'\b\d+\b', ' numbertoken ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    return ' '.join([stemmer.stem(word) for word in words])

# ⚡ Chargement instantané des cerveaux sauvegardés
print("🧠 Chargement du modèle et du vectoriseur sérialisés...")
model = joblib.load(config.MODEL_PATH)
vectorizer = joblib.load(config.VECTORIZER_PATH)

# 📝 ICI : Mets l'e-mail brut que tu veux tester en temps réel !
email_a_tester = """
Subject: Urgent Security Notification: Unusual Activity Detected on your Corporate Account

Dear Valued Customer,

Our automated global fraud detection systems have flags a series of suspicious and unauthorized login attempts on your corporate online banking profile within the last few hours. These connection requests originated from multiple blacklisted IP addresses located outside your usual geographic operational area. 

To protect your financial funds and prevent identity theft, we have temporarily placed a restricted hold on your outgoing transactions and online profile access. To resolve this security lock and verify your legitimate ownership of the account, you are required to complete our secure synchronization process within 24 hours of receiving this notification. 

Please click the secure network link below to log in to your dashboard, confirm your personal identification numbers, and update your multi-factor security preferences. Failure to complete this mandatory verification will result in the permanent suspension of your corporate assets and a potential legal review. 

http://secure-banking-update-portal-auth.com/login/auth-session

Thank you for your immediate attention to this critical security matter.
"""

# Pipeline de prédiction
clean_mail = clean_text_advanced(email_a_tester)
vectorized_mail = vectorizer.transform([clean_mail])
prediction = model.predict(vectorized_mail)[0]

print("\n🔮 RESULTAT DE LA PREDICTION EN TEMP RÉEL :")
print(f"Verdict : {'🔴 SPAM / PHISHING' if prediction == 1 else '🟢 SAIN (HAM)'}")