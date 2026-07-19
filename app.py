import re
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.stem.porter import PorterStemmer
import core.config as config

# Initialisation de l'API et des outils NLP
app = FastAPI(
    title="Spam & Phishing Detector API",
    description="API de classification en temps réel propulsée par un Linear SVM Balanced.",
    version="1.0.0"
)
stemmer = PorterStemmer()

# Chargement unique des artefacts au démarrage de l'application
print("🧠 Chargement du modèle de production...")
model = joblib.load(config.MODEL_PATH)
vectorizer = joblib.load(config.VECTORIZER_PATH)

# Modèle de données attendu en entrée (Validation par Pydantic)
class EmailPayload(BaseModel):
    text: str

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

@app.get("/")
def read_root():
    return {"status": "online", "model": "Linear SVM", "version": "1.0.0"}

@app.post("/predict")
def predict_email(payload: EmailPayload):
    # Sécurité si le texte envoyé est vide
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Le contenu du texte ne peut pas être vide.")
    
    try:
        # Pipeline de traitement et prédiction
        clean_mail = clean_text_advanced(payload.text)
        vectorized_mail = vectorizer.transform([clean_mail])
        prediction = int(model.predict(vectorized_mail)[0])
        
        # Réponse structurée propre
        return {
            "is_spam": bool(prediction),
            "label_num": prediction,
            "verdict": "🔴 SPAM / PHISHING" if prediction == 1 else "🟢 SAIN (HAM)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")