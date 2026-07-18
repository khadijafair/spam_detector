import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import ComplementNB 


# =====================================================================
# 1. CHARGEMENT DE TON FICHIER CSV
# =====================================================================
print("🔄 Chargement du dataset...")
df = pd.read_csv('spam_ham_dataset.csv')

# On ne garde que les colonnes utiles
df = df[['text', 'label_num']]


# =====================================================================
# 2. NETTOYAGE DU TEXTE (NLP)
# =====================================================================
def clean_text(text):
    # Conversion en chaîne pour éviter les erreurs de type, puis minuscules
    text = str(text).lower()
    # On retire la ponctuation, les chiffres et les caractères spéciaux
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

print("🧹 Nettoyage du texte en cours...")
df['clean_text'] = df['text'].apply(clean_text)


# =====================================================================
# 3. SÉPARATION ENTRAÎNEMENT / TEST (80% / 20%)
# =====================================================================
X = df['clean_text']
y = df['label_num'] # 0 = sain, 1 = spam

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# =====================================================================
# 4. VECTORISATION (TF-IDF)
# =====================================================================
print("🧮 Transformation du texte en nombres (TF-IDF)...")
vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =====================================================================
# 5. ENTRAÎNEMENT DU MODÈLE AMÉLIORÉ (COMPLEMENT NAVIE BAYES)
# =====================================================================
print("🤖 Entraînement du modèle ComplementNB (robuste au déséquilibre)...")
model = ComplementNB()
model.fit(X_train_tfidf, y_train)


# =====================================================================
# 6. ÉVALUATION DU MODÈLE
# =====================================================================
y_pred = model.predict(X_test_tfidf)

print("\n=== NOUVELLES PERFORMANCES ===")
print(f"Précision Globale (Accuracy) : {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['Ham (Sain)', 'Spam']))


# =====================================================================
# 7. RE-TEST DE L'E-MAIL PIÈGE
# =====================================================================
print("\n=== RE-TEST EN SITUATION RÉELLE ===")
custom_email = "Subject: we are pleased to inform you that you have been selected for the role of a software engineer at our company. "

custom_clean = [clean_text(custom_email)]
custom_tfidf = vectorizer.transform(custom_clean)
pred = model.predict(custom_tfidf)

resultat = "🔴 SPAM" if pred[0] == 1 else "🟢 SAIN"
print(f"E-mail testé : '{custom_email}'")
print(f"Résultat du modèle : {resultat}")