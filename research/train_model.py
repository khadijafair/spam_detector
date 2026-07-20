"""
train_model.py — Initial training script. Produces the artifacts every
interface loads (models/best_spam_model.pkl, models/tfidf_vectorizer.pkl).

Run this once to bootstrap the model; use retrain_feedback.py afterwards to
fold in user corrections over time.
"""
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import core.config as config

print("Loading cleaned dataset...")
df = pd.read_csv(config.DATASET_MASTER_PATH).dropna()

if df.shape[0] > 20000:
    df = df.sample(n=20000, random_state=42).reset_index(drop=True)

print("Vectorizing (TF-IDF, unigrams + bigrams)...")
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label_num']

print("Training Linear SVM (class_weight='balanced')...")
model = LinearSVC(class_weight='balanced', random_state=42, dual=False)
model.fit(X, y)

config.MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(model, config.MODEL_PATH)
joblib.dump(vectorizer, config.VECTORIZER_PATH)
print(f"Done. Saved to {config.MODEL_PATH} and {config.VECTORIZER_PATH}.")
print("Note: no confidence calibrator yet -- run retrain_feedback.py once feedback exists to generate one.")