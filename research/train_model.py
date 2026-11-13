"""
train_model.py — Trains the production artifacts every interface loads:
  models/best_spam_model.pkl        Linear SVM (class_weight='balanced')
  models/tfidf_vectorizer.pkl       TF-IDF, unigrams + bigrams, 50k features
  models/confidence_calibrator.pkl  Platt scaling fitted on a held-out 15%

Uses the full de-duplicated dataset (no 20k sampling). The run is logged to
MLflow and the model (TF-IDF + SVM pipeline) is registered in the MLflow Model
Registry as 'sentinel-spam-svm' with the alias 'champion'.

Usage (from the project root):
    python -m research.train_model
"""
import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import core.config as config
from core import tracking
from research.evaluate import SEED, load_master, scores, vectorizer


def main():
    tracking.setup()
    df = load_master()
    print(f"Dataset: {len(df)} emails")
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        df["clean_text"], df["label_num"], test_size=0.15, random_state=SEED, stratify=df["label_num"])

    with mlflow.start_run(run_name="train_production_model") as run:
        vec = vectorizer()
        X = vec.fit_transform(X_fit)
        print(f"Training Linear SVM on {len(X_fit)} emails...")
        model = LinearSVC(class_weight="balanced", random_state=SEED, dual=False)
        model.fit(X, y_fit)

        print(f"Fitting Platt-scaling calibrator on {len(X_cal)} held-out emails...")
        X_cal_vec = vec.transform(X_cal)
        calibrator = LogisticRegression()
        calibrator.fit(model.decision_function(X_cal_vec).reshape(-1, 1), y_cal)

        mlflow.log_params({"model": "LinearSVC", "C": model.C, "class_weight": "balanced",
                           "tfidf_ngram_range": "(1, 2)", "tfidf_max_features": 50000,
                           "train_emails": len(X_fit), "calibration_emails": len(X_cal), "seed": SEED})
        # The SVM never saw the calibration split, so it doubles as a held-out check.
        tracking.log_scores(scores(y_cal, model.predict(X_cal_vec)), prefix="holdout_")

        config.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)
        joblib.dump(vec, config.VECTORIZER_PATH)
        joblib.dump(calibrator, config.CALIBRATOR_PATH)
        mlflow.log_artifact(config.CALIBRATOR_PATH, artifact_path="calibrator")

        pipeline = Pipeline([("tfidf", vec), ("svm", model)])  # input: text already cleaned by spam_utils
        info = mlflow.sklearn.log_model(pipeline, name="model", input_example=[X_cal.iloc[0][:200]],
                                        registered_model_name=tracking.REGISTERED_MODEL)
        version = info.registered_model_version
        MlflowClient().set_registered_model_alias(tracking.REGISTERED_MODEL, "champion", version)
        print(f"Saved to {config.MODELS_DIR}/ — MLflow run {run.info.run_id}, "
              f"registered {tracking.REGISTERED_MODEL} v{version} (alias: champion)")


if __name__ == "__main__":
    main()
