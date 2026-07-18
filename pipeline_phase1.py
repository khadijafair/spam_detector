import os
import re
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report

# On s'assure d'avoir la bibliothèque d'outils linguistiques de base
# PorterStemmer fonctionne de manière algorithmique (sans téléchargement lourd requis)
stemmer = PorterStemmer()

dossier_dataset = 'dataset'
fichiers_charges = []

def clean_label(val):
    val = str(val).lower().strip()
    if val in ['spam', 'phishing', 'fraud', '1', '1.0']:
        return 1
    return 0

# =====================================================================
# 🔥 NOUVELLE FONCTION NLP AVANCÉE (PHASE 1)
# =====================================================================
def clean_text_advanced(text):
    text = str(text).lower()
    
    # 1. Remplacement des URLs (liens web)
    text = re.sub(r'https?://\S+|www\.\S+', ' urltoken ', text)
    
    # 2. Remplacement des adresses e-mail
    text = re.sub(r'\S+@\S+', ' emailtoken ', text)
    
    # 3. Remplacement des montants d'argent ($500, £2000, 500 dollars...)
    text = re.sub(r'\$\d+(?:[\.,]\d+)?|£\d+(?:[\.,]\d+)?|\b\d+\s*(?:dollars|usd|euro|euros|eur|pounds|gbp)\b', ' moneytoken ', text)
    
    # 4. Remplacement des numéros de téléphone courants
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', ' phonetoken ', text)
    
    # 5. Remplacement des nombres génériques restants
    text = re.sub(r'\b\d+\b', ' numbertoken ', text)
    
    # 6. Suppression de la ponctuation résiduelle (on garde nos lettres et tokens)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 7. Racinisation (Stemming) de chaque mot
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    
    return ' '.join(stemmed_words)


print("🚀 CHARGEMENT ET CONSOLIDATION DU MASTER DATASET...")
for fichier in os.listdir(dossier_dataset):
    if fichier.endswith('.csv'):
        chemin = os.path.join(dossier_dataset, fichier)
        df_temp = pd.read_csv(chemin)
        df_clean = pd.DataFrame()
        
        if 'text_combined' in df_temp.columns:
            df_clean['text'] = df_temp['text_combined'].fillna('')
        elif 'subject' in df_temp.columns and 'body' in df_temp.columns:
            df_clean['text'] = "Subject: " + df_temp['subject'].fillna('') + " \n " + df_temp['body'].fillna('')
        else:
            continue
            
        df_clean['label_num'] = df_temp['label'].apply(clean_label)
        fichiers_charges.append(df_clean)

df_master = pd.concat(fichiers_charges, ignore_index=True).dropna().drop_duplicates(subset=['text'])
print(f"📊 Taille du Master Dataset combiné : {df_master.shape[0]} e-mails.")

# Application du nouveau nettoyage avancé
print("🧹 Application du Feature Engineering Avancé (Tokens + Stemming)...")
df_master['clean_text'] = df_master['text'].apply(clean_text_advanced)

# SÉPARATION
X = df_master['clean_text']
y = df_master['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# 🧮 TF-IDF AVEC N-GRAMS (1, 2)
# =====================================================================
print("🧮 Vectorisation TF-IDF (Unigrams + Bigrams)...")
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=60000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ENTRAÎNEMENT
print("🤖 Entraînement du modèle...")
model = ComplementNB()
model.fit(X_train_tfidf, y_train)

# ÉVALUATION
y_pred = model.predict(X_test_tfidf)
print("\n🏆 PERFORMANCES APRES PHASE 1 :")
print(f"Précision Globale (Accuracy) : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Ham (Sain)', 'Spam/Phishing']))

# TEST DE L'E-MAIL DE RECRUTEMENT PIÈGE
print("\n🎯 TEST DE VÉRIFICATION CONTEXTUELLE :")
test_email = "Subject: we are pleased to inform you that you have been selected for the role of a software engineer at our company."
test_clean = [clean_text_advanced(test_email)]
test_tfidf = vectorizer.transform(test_clean)
prediction = model.predict(test_tfidf)

resultat = "🔴 SPAM / PHISHING" if prediction[0] == 1 else "🟢 SAIN (Ham)"
print(f"E-mail traité : '{test_email}'")
print(f"Verdict du modèle enrichi : {resultat}")