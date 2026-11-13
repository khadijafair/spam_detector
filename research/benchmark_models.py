import os
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import des nouveaux modèles pour la Phase 2
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

stemmer = PorterStemmer()
import core.config as config
FICHIER_CACHE = config.DATASET_MASTER_PATH
dossier_dataset = config.RAW_DATASET_DIR

def clean_label(val):
    val = str(val).lower().strip()
    if val in ['spam', 'phishing', 'fraud', '1', '1.0']:
        return 1
    return 0

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

# =====================================================================
# 📦 CHARGEMENT AVEC SYSTÈME DE CACHE / ÉCHANTILLON
# =====================================================================
if os.path.exists(FICHIER_CACHE):
    print("💾 Fichier nettoyé trouvé en cache ! Chargement...")
    df_master = pd.read_csv(FICHIER_CACHE).dropna()
else:
    print("🧹 Aucun cache trouvé. Génération du Master Dataset...")
    fichiers_charges = []
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
    print("🧹 Application du traitement NLP avancé...")
    df_master['clean_text'] = df_master['text'].apply(clean_text_advanced)
    df_master[['clean_text', 'label_num']].to_csv(FICHIER_CACHE, index=False)

# 🏎️ MODE DEV : Échantillonnage pour que le calcul reste super rapide (max 20k lignes)
if df_master.shape[0] > 20000:
    print("🏎️ Échantillonnage à 20 000 lignes pour accélérer le benchmark...")
    df_master = df_master.sample(n=20000, random_state=42).reset_index(drop=True)

# PRÉPARATION DATA
X_train, X_test, y_train, y_test = train_test_split(
    df_master['clean_text'], df_master['label_num'], test_size=0.2, random_state=42
)

# VECTORISATION
print("🧮 Vectorisation TF-IDF (Unigrams + Bigrams)...")
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =====================================================================
# ⚔️ DUEL DES MODÈLES
# =====================================================================
modeles = {
    "Complement Naive Bayes": ComplementNB(),
    "Logistic Regression (Balanced)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Linear SVM (SVC Balanced)": LinearSVC(class_weight='balanced', random_state=42, dual=False)
}

resultats = []
email_test_recrutement = "Subject: STRICTLY CONFIDENTIAL: Urgent Corporate Acquisition Project - Wire Transfer Required . Hi, I am writing to you directly because I need your urgent assistance with a highly sensitive corporate transaction that we are finalizing today. We are currently in the final stages of acquiring a new international technology vendor, and this operation must remain strictly confidential until the official press release next week. Due to the nature of the non-disclosure agreement, do not discuss this matter with anyone else in your department or mention it via public corporate chat channels. The legal team has cleared the paperwork, and we need to process an initial deposit payment of forty-five thousand dollars immediately to secure the exclusivity rights.  I am currently stuck in an all-day executive board meeting and cannot take any external phone calls. Please reply to this message directly as soon as you receive it so I know you are at your desk. I will send over the external escrow account numbers, routing information, and invoice documents so you can execute the wire transfer before the close of business today. I am counting on your swift execution and absolute discretion on this project."
email_test_clean = [clean_text_advanced(email_test_recrutement)]
email_test_tfidf = vectorizer.transform(email_test_clean)

print("\n🏋️ Entraînement et évaluation des compétiteurs...")
for nom, model in modeles.items():
    # Entraînement
    model.fit(X_train_tfidf, y_train)
    # Prédiction
    y_pred = model.predict(X_test_tfidf)
    
    # Métriques
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    
    # Test de l'e-mail piège
    pred_piege = model.predict(email_test_tfidf)[0]
    verdict_piege = "🟢 SAIN" if pred_piege == 0 else "🔴 SPAM"
    
    resultats.append({
        "Modèle": nom,
        "Accuracy": f"{acc:.2f}%",
        "Precision": f"{prec:.2f}%",
        "Recall (Rappel)": f"{rec:.2f}%",
        "F1-Score": f"{f1:.2f}%",
        "Verdict E-mail Piège": verdict_piege
    })

# =====================================================================
# 🏆 TABLEAU COMPARATIF FINAL
# =====================================================================
df_res = pd.DataFrame(resultats)
print("\n🏆 TABLEAU COMPARATIF DES PERFORMANCES :")
print(df_res.to_markdown(index=False))