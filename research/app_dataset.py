import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report

dossier_dataset = 'dataset'
fichiers_charges = []

# Fonction pour standardiser les labels en 0 (Sain) et 1 (Spam/Phishing/Fraude)
def clean_label(val):
    val = str(val).lower().strip()
    if val in ['spam', 'phishing', 'fraud', '1', '1.0', 'spam']:
        return 1
    return 0

# Fonction de nettoyage de texte NLP de base
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

print("🚀 DÉMARRAGE DE LA FUSION DE LA COMPILATION...")

# Parcourir et traiter chaque fichier
for fichier in os.listdir(dossier_dataset):
    if fichier.endswith('.csv'):
        chemin = os.path.join(dossier_dataset, fichier)
        print(f"🔄 Traitement de : {fichier}...")
        
        df_temp = pd.read_csv(chemin)
        df_clean = pd.DataFrame()
        
        # CAS 1 : Le fichier a 'text_combined' (phishing_email.csv)
        if 'text_combined' in df_temp.columns:
            df_clean['text'] = df_temp['text_combined'].fillna('')
            
        # CAS 2 : Le fichier a 'subject' et 'body' (Tous les autres)
        elif 'subject' in df_temp.columns and 'body' in df_temp.columns:
            # On combine le sujet et le corps pour donner un max de contexte au modèle
            df_clean['text'] = "Subject: " + df_temp['subject'].fillna('') + " \n " + df_temp['body'].fillna('')
        else:
            print(f"⚠️ Colonnes inconnues dans {fichier}, ignoré.")
            continue
            
        # Standardisation des labels
        df_clean['label_num'] = df_temp['label'].apply(clean_label)
        
        # Sauvegarde temporaire dans notre liste
        fichiers_charges.append(df_clean)

# =====================================================================
# FUSION DU MASTER DATASET
# =====================================================================
print("\n🔗 Fusion de tous les datasets en un seul Master Dataset...")
df_master = pd.concat(fichiers_charges, ignore_index=True)

# Nettoyage des doublons et lignes vides créés par la fusion
df_master = df_master.dropna()
df_master = df_master.drop_duplicates(subset=['text'])

print(f"📊 Taille totale du Master Dataset : {df_master.shape[0]} e-mails.")
print(f"   ↳ 🟢 E-mails Sains (Ham) : {df_master[df_master['label_num'] == 0].shape[0]}")
print(f"   ↳ 🔴 Spams / Phishing    : {df_master[df_master['label_num'] == 1].shape[0]}")

# Application du nettoyage NLP sur l'ensemble
print("\n🧹 Nettoyage linguistique global (NLP)...")
df_master['clean_text'] = df_master['text'].apply(clean_text)

# =====================================================================
# MACHINE LEARNING PIPELINE
# =====================================================================
X = df_master['clean_text']
y = df_master['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧮 Vectorisation TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=50000) # Limite à 50k mots uniques pour éviter d'exploser la RAM
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("🤖 Entraînement de l'ultra-modèle ComplementNB...")
model = ComplementNB()
model.fit(X_train_tfidf, y_train)

# Évaluation
y_pred = model.predict(X_test_tfidf)
print("\n🏆 PERFORMANCES DU ULTRA-MODÈLE GÉNÉRALISÉ :")
print(f"Précision Globale (Accuracy) : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Ham (Sain)', 'Spam/Phishing']))

# =====================================================================
# RE-TEST DE NOTRE ANCIEN E-MAIL PIÈGE
# =====================================================================
print("\n🎯 RE-TEST DE L'E-MAIL DE RECRUTEMENT (L'ancien Faux Positif) :")
test_email = "Subject: hello Mr after reviewing your application , we are pleased to inform you that you have been selected to participate in the next step of the process and have an interview scheduled."
test_clean = [clean_text(test_email)]
test_tfidf = vectorizer.transform(test_clean)
prediction = model.predict(test_tfidf)

resultat = "🔴 SPAM / PHISHING" if prediction[0] == 1 else "🟢 SAIN (Ham)"
print(f"E-mail : '{test_email}'")
print(f"Verdict de la forteresse : {resultat}")