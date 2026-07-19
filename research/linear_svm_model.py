import os
import re
import pandas as pd
import joblib
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

stemmer = PorterStemmer()
FICHIER_CACHE = 'dataset_master_clean.csv'

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

# 1. Chargement des données propres
print("💾 Chargement des données depuis le cache...")
df_master = pd.read_csv(FICHIER_CACHE).dropna()

if df_master.shape[0] > 20000:
    df_master = df_master.sample(n=20000, random_state=42).reset_index(drop=True)

# 2. Vectorisation et entraînement du grand gagnant
print("🧮 Vectorisation TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50000)
X_tfidf = vectorizer.fit_transform(df_master['clean_text'])
y = df_master['label_num']

print("🤖 Entraînement final de notre champion (Linear SVM)...")
champion_model = LinearSVC(class_weight='balanced', random_state=42, dual=False)
champion_model.fit(X_tfidf, y)

# =====================================================================
# 💾 EXPORTATION SAUVEGARDE (MLOps)
# =====================================================================
print("\n📦 Sauvegarde des artefacts du modèle...")
joblib.dump(champion_model, 'best_spam_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("✨ Terminé ! 'best_spam_model.pkl' et 'tfidf_vectorizer.pkl' sont prêts.")