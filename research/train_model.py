"""
train_model.py — Trains the production artifacts every interface loads:
  models/best_spam_model.pkl        Linear SVM (class_weight='balanced')
  models/tfidf_vectorizer.pkl       TF-IDF, unigrams + bigrams, 50k features
  models/confidence_calibrator.pkl  Platt scaling fitted on a held-out 15%

Uses the full de-duplicated dataset (no 20k sampling). Run once to bootstrap,
then use retrain_feedback.py to fold in user corrections over time.

Usage (from the project root):
    python -m research.train_model
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import core.config as config
from research.evaluate import SEED, load_master, vectorizer


def main():
    df = load_master()
    print(f"Dataset: {len(df)} emails")
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        df["clean_text"], df["label_num"], test_size=0.15, random_state=SEED, stratify=df["label_num"])

    vec = vectorizer()
    X = vec.fit_transform(X_fit)
    print(f"Training Linear SVM on {len(X_fit)} emails...")
    model = LinearSVC(class_weight="balanced", random_state=SEED, dual=False)
    model.fit(X, y_fit)

    print(f"Fitting Platt-scaling calibrator on {len(X_cal)} held-out emails...")
    calibrator = LogisticRegression()
    calibrator.fit(model.decision_function(vec.transform(X_cal)).reshape(-1, 1), y_cal)

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(vec, config.VECTORIZER_PATH)
    joblib.dump(calibrator, config.CALIBRATOR_PATH)
    print(f"Saved to {config.MODELS_DIR}/")


if __name__ == "__main__":
    main()
