"""
retrain_feedback.py — Periodic retraining that folds user corrections
(feedback.jsonl, shared by every interface) back into the model.

Beyond incorporating corrections, it also re-tunes the SVM's regularization
(C) via cross-validated grid search, and fits a Platt-scaling calibrator so
confidence scores reflect real reliability instead of a rough sigmoid
approximation. Old model files are backed up (models/model_backups/) before
being overwritten -- nothing is lost.

Usage:
    python retrain_feedback.py --dry-run    # preview metrics, overwrite nothing
    python retrain_feedback.py              # apply
    python retrain_feedback.py --dataset other.csv --text-col X --label-col Y
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score

import core.config as config
from core.spam_utils import clean_text_advanced
from core import tracking

TEXT_COL_CANDIDATES = ["clean_text", "text", "message", "v2", "email", "content"]
LABEL_COL_CANDIDATES = ["label_num", "label", "v1", "class", "category", "spam"]
CORRECTION_OVERSAMPLE = 5  # weight given to each corrected example vs the original dataset
C_GRID = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]


def autodetect_columns(df: pd.DataFrame, text_col: str | None, label_col: str | None):
    if text_col is None:
        text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Can't guess text/label columns in {list(df.columns)}. Use --text-col/--label-col.")
    return text_col, label_col


def normalize_label(value) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    v = str(value).strip().lower()
    if v in ("spam", "1", "true", "phishing"):
        return 1
    if v in ("ham", "0", "false", "legit", "legitimate"):
        return 0
    raise ValueError(f"Unrecognized label: {value!r}")


def load_feedback_corrections(feedback_path: str) -> list[tuple[str, int]]:
    import json
    corrections = []
    path = Path(feedback_path)
    if not path.exists():
        print(f"No feedback file at {path}, nothing to integrate.")
        return corrections
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("marked_correct") is False:
                corrected = 0 if entry["predicted_label"] == "SPAM" else 1
                corrections.append((entry["text"], corrected))
    return corrections


def main():
    tracking.setup()
    with mlflow.start_run(run_name="retrain_feedback"):
        _main()


def _main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(config.DATASET_MASTER_PATH))
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-tune", action="store_true", help="Skip the C grid search (faster)")
    args = parser.parse_args()

    print("Loading current artifacts to reuse their hyperparameters...")
    old_model = joblib.load(config.MODEL_PATH)
    old_vectorizer = joblib.load(config.VECTORIZER_PATH)
    vec_params = old_vectorizer.get_params()
    model_params = old_model.get_params()
    print(f"  Vectorizer: ngram_range={vec_params.get('ngram_range')}, max_features={vec_params.get('max_features')}")
    print(f"  Model: class_weight={model_params.get('class_weight')}, current C={model_params.get('C')}")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        sys.exit(f"Dataset not found: {dataset_path}. Use --dataset to specify a path.")
    df = pd.read_csv(dataset_path, encoding="latin-1", encoding_errors="ignore")
    text_col, label_col = autodetect_columns(df, args.text_col, args.label_col)
    print(f"Dataset: {len(df)} rows (columns: text='{text_col}', label='{label_col}')")
    print(f"  Note: '{text_col}' is treated as ALREADY cleaned (not re-cleaned).")

    before = len(df)
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]
    if before - len(df):
        print(f"  Dropped {before - len(df)} row(s) with missing/empty text or label.")

    dataset_texts_clean = df[text_col].tolist()
    dataset_labels = [normalize_label(v) for v in df[label_col]]

    corrections = load_feedback_corrections(config.FEEDBACK_LOG_PATH)
    print(f"Found {len(corrections)} correction(s) in {config.FEEDBACK_LOG_PATH}")
    if not corrections:
        print("Nothing to retrain (no pending corrections). Done.")
        return

    print("Cleaning correction texts (raw -> same pipeline as production)...")
    correction_texts_clean, correction_labels = [], []
    for raw_text, label in corrections:
        cleaned = clean_text_advanced(raw_text)
        correction_texts_clean.extend([cleaned] * CORRECTION_OVERSAMPLE)
        correction_labels.extend([label] * CORRECTION_OVERSAMPLE)

    all_texts = dataset_texts_clean + correction_texts_clean
    all_labels = dataset_labels + correction_labels

    # 3-way split: train / calibration (fits the Platt scaler) / test (final, unseen).
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        all_texts, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    calib_texts, test_texts, calib_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    print(f"Split: {len(train_texts)} train / {len(calib_texts)} calibration / {len(test_texts)} test")

    new_vectorizer = TfidfVectorizer(**{k: v for k, v in vec_params.items() if k not in ("input", "dtype")})
    X_train = new_vectorizer.fit_transform(train_texts)
    X_calib = new_vectorizer.transform(calib_texts)
    X_test = new_vectorizer.transform(test_texts)

    base_params = {k: v for k, v in model_params.items() if k != "C"}
    if args.no_tune:
        print(f"Training with C={model_params.get('C')} (tuning skipped)...")
        new_model = LinearSVC(**model_params)
        new_model.fit(X_train, train_labels)
    else:
        print(f"Searching best C among {C_GRID} (5-fold CV, scoring=F1)...")
        grid = GridSearchCV(LinearSVC(**base_params), param_grid={"C": C_GRID}, scoring="f1", cv=5, n_jobs=-1)
        grid.fit(X_train, train_labels)
        new_model = grid.best_estimator_
        print(f"  Best C: {grid.best_params_['C']} (CV F1 = {grid.best_score_:.4f})")

    print("Fitting confidence calibrator (Platt scaling) on the validation split...")
    calib_scores = new_model.decision_function(X_calib).reshape(-1, 1)
    calibrator = LogisticRegression()
    calibrator.fit(calib_scores, calib_labels)

    old_test_vec = old_vectorizer.transform(test_texts)
    old_preds = old_model.predict(old_test_vec)
    new_preds = new_model.predict(X_test)

    print("\n=== Current model (before retraining) ===")
    print(classification_report(test_labels, old_preds, target_names=["HAM", "SPAM"]))
    print("\n=== New model (corrections + tuned C + calibration) ===")
    print(classification_report(test_labels, new_preds, target_names=["HAM", "SPAM"]))

    old_f1, new_f1 = f1_score(test_labels, old_preds), f1_score(test_labels, new_preds)
    mlflow.log_params({"dataset": str(dataset_path), "corrections": len(corrections),
                       "correction_oversample": CORRECTION_OVERSAMPLE, "C": new_model.C, "dry_run": args.dry_run})
    mlflow.log_metrics({"f1_before": old_f1 * 100, "f1_after": new_f1 * 100,
                        "test_emails": len(test_texts)})
    print(f"\nF1-score: {old_f1:.4f} -> {new_f1:.4f} ({'+' if new_f1 >= old_f1 else ''}{new_f1 - old_f1:.4f})")

    test_scores = new_model.decision_function(X_test)
    test_proba_spam = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]
    confidences = np.where(new_preds == 1, test_proba_spam, 1 - test_proba_spam) * 100
    correct_mask = np.array(new_preds) == np.array(test_labels)
    print(f"Avg confidence (correct predictions): {confidences[correct_mask].mean():.1f}%")
    if (~correct_mask).sum() > 0:
        print(f"Avg confidence (wrong predictions):   {confidences[~correct_mask].mean():.1f}%")

    if args.dry_run:
        print("\n--dry-run: nothing was overwritten.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = config.MODELS_DIR / "model_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config.MODEL_PATH, backup_dir / f"best_spam_model_{timestamp}.pkl")
    shutil.copy(config.VECTORIZER_PATH, backup_dir / f"tfidf_vectorizer_{timestamp}.pkl")
    if Path(config.CALIBRATOR_PATH).exists():
        shutil.copy(config.CALIBRATOR_PATH, backup_dir / f"confidence_calibrator_{timestamp}.pkl")

    joblib.dump(new_model, config.MODEL_PATH)
    joblib.dump(new_vectorizer, config.VECTORIZER_PATH)
    joblib.dump(calibrator, config.CALIBRATOR_PATH)
    print(f"\nSaved. Previous files backed up in {backup_dir}/. Restart the bot/API/Streamlit app to pick up changes.")


if __name__ == "__main__":
    main()