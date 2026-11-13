"""
evaluate.py — Reproducible evaluation of the spam/phishing classifier.

Every number quoted in the README comes from this script. It:
  1. Builds the master dataset from data/dataset/*.csv (6 public corpora),
     de-duplicates it and records which corpus each email came from.
  2. Holds out a stratified 20% test set the model never sees during training.
  3. Benchmarks Complement NB / Logistic Regression / Linear SVM on that split.
  4. Measures single-message inference latency of the production pipeline
     (cleaning + TF-IDF + SVM + calibrated confidence).
  5. Runs a leave-one-corpus-out test: train on 5 corpora, test on the 6th,
     to show how the model behaves on emails from an unseen source.

Writes reports/metrics.json and reports/metrics.md, and logs everything to
MLflow (one parent run, one nested run per model; see core/tracking.py).

Usage (from the project root):
    python -m research.evaluate
"""
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

import core.config as config
from core import tracking
from core.spam_utils import clean_text_advanced, predict_with_confidence

SEED = 42
REPORTS_DIR = config.PROJECT_ROOT / "reports"
CACHE_PATH = config.DATASET_MASTER_PATH  # shared cache: clean_text, label_num, source


def vectorizer():
    # Same settings as research/train_model.py (production model).
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)


def models():
    return {
        "Complement Naive Bayes": ComplementNB(),
        "Logistic Regression (balanced)": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        "Linear SVM (balanced)": LinearSVC(class_weight="balanced", random_state=SEED, dual=False),
    }


def load_master() -> pd.DataFrame:
    if CACHE_PATH.exists():
        cached = pd.read_csv(CACHE_PATH).dropna()
        if "source" in cached.columns:
            return cached
    frames = []
    for csv in sorted(Path(config.RAW_DATASET_DIR).glob("*.csv")):
        raw = pd.read_csv(csv)
        if not {"subject", "body", "label"}.issubset(raw.columns):
            continue
        frames.append(pd.DataFrame({
            "text": "Subject: " + raw["subject"].fillna("") + " \n " + raw["body"].fillna(""),
            "label_num": raw["label"].astype(int),
            "source": csv.stem,
        }))
    if not frames:
        raise SystemExit(f"No corpus found in {config.RAW_DATASET_DIR}. See README > Dataset.")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Cleaning {len(df)} emails (stemming, URL/email/money/phone tokens)...")
    df["clean_text"] = df["text"].apply(clean_text_advanced)
    df = df[df["clean_text"].str.strip() != ""]
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["clean_text", "label_num", "source"]].to_csv(CACHE_PATH, index=False)
    return df[["clean_text", "label_num", "source"]]


def scores(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "false_positive_rate": round(fp / (fp + tn) * 100, 2) if (fp + tn) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    tracking.setup()
    with mlflow.start_run(run_name="evaluate"):
        run(time.time())


def run(t0):
    df = load_master()
    report = {"dataset": {
        "corpora": sorted(df["source"].unique().tolist()),
        "emails_after_dedup": int(len(df)),
        "spam_or_phishing": int(df["label_num"].sum()),
        "legitimate": int((df["label_num"] == 0).sum()),
        "per_corpus": {s: int(n) for s, n in df["source"].value_counts().sort_index().items()},
    }}
    print(json.dumps(report["dataset"], indent=2))

    # ---- 1. Held-out benchmark --------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean_text"], df["label_num"], test_size=0.2, random_state=SEED, stratify=df["label_num"])
    vec = vectorizer()
    t = time.time()
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)
    report["split"] = {"train": int(len(X_tr)), "test": int(len(X_te)), "stratified": True, "seed": SEED}
    mlflow.log_params({"dataset_emails": len(df), "corpora": ",".join(report["dataset"]["corpora"]),
                       "test_size": 0.2, "seed": SEED, "tfidf_ngram_range": "(1, 2)", "tfidf_max_features": 50000})

    report["benchmark"] = {}
    svm = None
    for name, m in models().items():
        t = time.time()
        m.fit(Xtr, y_tr)
        fit_s = time.time() - t
        report["benchmark"][name] = {**scores(y_te, m.predict(Xte)), "train_seconds": round(fit_s, 1)}
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params({"model": type(m).__name__, **{k: str(v) for k, v in m.get_params().items()}})
            tracking.log_scores(report["benchmark"][name])
        print(name, report["benchmark"][name])
        if "SVM" in name:
            svm = m

    # ---- 2. Inference latency of the production pipeline ------------------
    # Raw-ish text: the clean_text column is already preprocessed, so latency
    # is measured on it re-run through the full predict_with_confidence path.
    sample = X_te.sample(n=min(1000, len(X_te)), random_state=SEED).tolist()
    for s in sample[:20]:  # warm-up
        predict_with_confidence(s, svm, vec)
    timings = []
    for s in sample:
        t = time.perf_counter()
        predict_with_confidence(s, svm, vec)
        timings.append((time.perf_counter() - t) * 1000)
    timings = np.array(timings)
    report["latency_ms_per_message"] = {
        "messages": len(timings),
        "median": round(float(np.median(timings)), 2),
        "p95": round(float(np.percentile(timings, 95)), 2),
        "note": "single-threaded CPU, cleaning + TF-IDF + SVM + confidence, no LLM call",
    }
    print("Latency:", report["latency_ms_per_message"])
    tracking.log_scores(report["benchmark"]["Linear SVM (balanced)"], prefix="svm_")
    mlflow.log_metric("latency_median_ms", report["latency_ms_per_message"]["median"])
    mlflow.log_metric("latency_p95_ms", report["latency_ms_per_message"]["p95"])

    # ---- 3. Leave-one-corpus-out ------------------------------------------
    report["leave_one_corpus_out"] = {}
    for src in report["dataset"]["corpora"]:
        train, test = df[df["source"] != src], df[df["source"] == src]
        v = vectorizer()
        m = LinearSVC(class_weight="balanced", random_state=SEED, dual=False)
        m.fit(v.fit_transform(train["clean_text"]), train["label_num"])
        pred = m.predict(v.transform(test["clean_text"]))
        res = scores(test["label_num"], pred)
        res["emails"] = int(len(test))
        res["only_one_class"] = bool(test["label_num"].nunique() == 1)
        report["leave_one_corpus_out"][src] = res
        mlflow.log_metric(f"loco_{src}_accuracy", res["accuracy"])
        mlflow.log_metric(f"loco_{src}_f1", res["f1"])
        print("LOCO", src, res["accuracy"], res["f1"])

    report["runtime_seconds"] = round(time.time() - t0, 1)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    (REPORTS_DIR / "metrics.md").write_text(to_markdown(report))
    mlflow.log_artifact(str(REPORTS_DIR / "metrics.json"))
    mlflow.log_artifact(str(REPORTS_DIR / "metrics.md"))
    print(f"\nWritten to {REPORTS_DIR}/metrics.json and metrics.md")


def to_markdown(r: dict) -> str:
    d = r["dataset"]
    lines = [
        "# Evaluation report (generated by `python -m research.evaluate`)", "",
        f"**Dataset:** {d['emails_after_dedup']:,} unique emails from {len(d['corpora'])} public corpora "
        f"({d['spam_or_phishing']:,} spam/phishing, {d['legitimate']:,} legitimate).", "",
        f"**Split:** {r['split']['train']:,} train / {r['split']['test']:,} test, stratified, seed {r['split']['seed']}.", "",
        "## Held-out test set", "",
        "| Model | Accuracy | Precision | Recall | F1 | False-positive rate | Train time |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in r["benchmark"].items():
        lines.append(f"| {name} | {s['accuracy']}% | {s['precision']}% | {s['recall']}% | {s['f1']}% | "
                     f"{s['false_positive_rate']}% | {s['train_seconds']} s |")
    cm = r["benchmark"]["Linear SVM (balanced)"]["confusion_matrix"]
    lat = r["latency_ms_per_message"]
    lines += [
        "", f"Linear SVM confusion matrix: TN={cm['tn']:,} · FP={cm['fp']:,} · FN={cm['fn']:,} · TP={cm['tp']:,}", "",
        "## Inference latency", "",
        f"Median **{lat['median']} ms**, p95 {lat['p95']} ms per message over {lat['messages']} messages ({lat['note']}).", "",
        "## Leave-one-corpus-out (train on 5 corpora, test on the unseen 6th)", "",
        "| Unseen corpus | Emails | Accuracy | F1 |", "|---|---|---|---|",
    ]
    for src, s in r["leave_one_corpus_out"].items():
        f1 = "n/a (one class only)" if s["only_one_class"] and s["f1"] == 0 else f"{s['f1']}%"
        lines.append(f"| {src} | {s['emails']:,} | {s['accuracy']}% | {f1} |")
    lines += ["", "Scores on an unseen corpus are lower than on the random split: emails from the same "
              "corpus share vocabulary and formatting, so the random split is the optimistic case.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
