# 🛡️ Sentinel-Spam — Hybrid Spam & Phishing Detection

![tests](https://github.com/khadijafair/spam_detector/actions/workflows/tests.yml/badge.svg?branch=main)

**▶ Live demo: [sentinel-spam-khadija.streamlit.app](https://sentinel-spam-khadija.streamlit.app/)** (English text, SVM classifier with explanations)

Sentinel-Spam is an end-to-end spam and phishing detection system. It combines a fast **Linear SVM classifier (TF-IDF)** with a local **Ollama LLM** used for translation of non-English messages and arbitration of borderline cases. The same pipeline is exposed through a **FastAPI** service, a **Streamlit** dashboard and a **Telegram bot**, with a user-feedback loop that feeds periodic retraining. Experiments and models are tracked in **MLflow** and input drift is monitored with **Evidently**.

## 📈 Results (measured, reproducible)

All numbers below come from `python -m research.evaluate`, which writes [`reports/metrics.md`](reports/metrics.md) and `reports/metrics.json`.

**Dataset:** 82,486 emails from 6 public corpora — CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud, SpamAssassin (42,891 spam/phishing, 39,595 legitimate).
**Protocol:** stratified 80/20 split (65,988 train / 16,498 held-out test), seed 42.

| Model | Accuracy | Precision | Recall | F1 | False-positive rate |
|---|---|---|---|---|---|
| Complement Naive Bayes | 97.26% | 99.09% | 95.61% | 97.32% | 0.95% |
| Logistic Regression (balanced) | 98.62% | 98.64% | 98.72% | 98.68% | 1.48% |
| **Linear SVM (balanced)** — selected | **99.13%** | **99.15%** | **99.18%** | **99.17%** | **0.92%** |

- Linear SVM confusion matrix on the 16,498 test emails: 73 false positives, 70 missed spams.
- **Inference latency:** median **1.3 ms** per message (p95 5.9 ms) for cleaning + TF-IDF + SVM + calibrated confidence on one CPU core, without the LLM call.

**Generalization limit (honest view).** When the model is trained on 5 corpora and tested on the 6th, unseen one, accuracy ranges from **53% (Nazario)** to **99% (Nigerian_Fraud)**, and is 81–91% on the other four. The random-split score above is therefore the optimistic case: emails from a source the model has never seen are much harder. This is the main motivation for the LLM arbitration on borderline scores and for the feedback/retraining loop.

## 🔁 MLOps: experiment tracking, model registry, drift monitoring

**MLflow.** Every script logs to MLflow (`core/tracking.py`; local SQLite store by default, or `MLFLOW_TRACKING_URI`):
- `research.evaluate` — one parent run with dataset parameters, latency and leave-one-corpus-out metrics, plus one nested run per benchmarked model.
- `research.train_model` — parameters, held-out metrics (the 12,373-email calibration split the SVM never sees: **F1 99.18%, false-positive rate 0.91%**), and registration of the TF-IDF + SVM pipeline in the **Model Registry** as `sentinel-spam-svm` with the alias `champion`.
- `retrain_feedback.py` — number of user corrections, tuned `C`, F1 before/after retraining.
- `monitoring.drift_report` — drift results and the Evidently HTML reports as artifacts.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db       # browse runs and registered models
python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); \
  print(mlflow.sklearn.load_model('models:/sentinel-spam-svm@champion'))"
```

**Evidently drift monitoring** (`monitoring/drift_report.py`). Compares a window of incoming messages with a 3,000-email reference sample on 9 columns: the text itself, message length, URL / money / phone / number / email token counts, the SVM score and the predicted-spam flag. No labels are needed, so it can run on production traffic (`--current data/feedback.jsonl`). Reproducible demo with `--demo` ([`reports/drift_summary.md`](reports/drift_summary.md)):

| Scenario | Current window | Drifted columns | Alert | Predicted spam rate |
|---|---|---|---|---|
| Same corpora, new random sample | 3,000 emails | 1 / 9 | no | 50.4% → 51.0% |
| Unseen phishing corpus (Nazario) | 1,565 emails | 8 / 9 | **yes** | 50.4% → 99.6% |

The monitor stays quiet on data like the training set and raises an alert when the traffic changes source — the situation where the leave-one-corpus-out results show the model is least reliable.

## 🗺️ How it works

```text
📥 Input text (any language)
       │
       ▼
 [ Language detection ] ──► non-English? ──► [ Ollama translation → English ]
       │                                                │
       ▼ ◄──────────────────────────────────────────────┘
 [ Cleaning: URL / email / money / phone / number tokens + Porter stemming ]
       │
       ▼
 [ TF-IDF (1-2 grams, 50k features) → Linear SVM ] ──► raw margin score
       │
       ▼
 [ Platt-scaling calibrator ] ──► confidence %
       │
       ▼
 Borderline (≤ 65%)? ── yes ──► [ Ollama secondary check ] (Telegram bot)
       │
       ▼
 🚨 / ✅ Verdict + most influential words (TF-IDF weight × SVM coefficient)
```

- **One shared pipeline** (`core/spam_utils.py`): API, dashboard, bot and training scripts use the same cleaning and vectorization code, so predictions are identical everywhere.
- **Explainability:** each verdict lists the words that pushed it, computed as TF-IDF weight × SVM coefficient.
- **Calibrated confidence:** a Platt-scaling model maps the SVM margin to a probability, fitted on a 15% held-out split.
- **Feedback loop:** user corrections are appended to `data/feedback.jsonl`; `retrain_feedback.py` re-tunes `C` by 5-fold grid search, refits the calibrator, compares old vs new model on a test split and backs up previous models before overwriting.

## ⚡ Quick start

Requires **Python 3.10+**. Ollama is only needed for translation / LLM arbitration; the classifier works without it.

```bash
git clone https://github.com/khadijafair/spam_detector.git
cd spam_detector
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

The trained model (`models/*.pkl`, ~2.4 MB) is included, so the interfaces run directly:

```bash
uvicorn api:app --reload           # REST API → http://127.0.0.1:8000/docs
streamlit run streamlit_app.py     # web dashboard
streamlit run demo/streamlit_demo.py   # light public demo (no Ollama), deployable on Streamlit Community Cloud
python telegram_bot.py             # Telegram bot (needs TELEGRAM_TOKEN in .env)
```

Optional, for translation and arbitration: install [Ollama](https://ollama.com) and run `ollama pull llama3.2:3b`.

Example API call:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"text": "URGENT: your account is suspended, verify your password at http://secure-login-update.com"}'
```

With Docker (Streamlit dashboard on port 8501):

```bash
docker build -t sentinel-spam .
docker run -p 8501:8501 sentinel-spam
```

## 📊 Dataset & reproducing the results

The 6 corpora are the public *Phishing Email Dataset* (Kaggle, CC BY-SA 4.0; also mirrored at [rokibulroni/Phishing-Email-Dataset](https://github.com/rokibulroni/Phishing-Email-Dataset)). They are not versioned here because of their size (~150 MB).

```bash
# 1. put CEAS_08.csv, Enron.csv, Ling.csv, Nazario.csv, Nigerian_Fraud.csv, SpamAssasin.csv in data/dataset/
python -m research.evaluate        # benchmark + latency + leave-one-corpus-out → reports/
python -m research.train_model     # retrain the production model on the full dataset (+ MLflow registry)
python -m monitoring.drift_report --demo   # Evidently drift reports → reports/
python -m pytest -q tests          # 9 tests: cleaning pipeline, API endpoints, drift features
```

`research/benchmark_models.py` is the original exploration script (20k-email sample, non-stratified split). On that protocol the Linear SVM scores 98.31% F1.

## 📁 Repository structure

```text
spam_detector
├─ core/
│  ├─ config.py              # paths and settings, loaded from .env
│  ├─ i18n.py                # bot messages (EN/FR)
│  ├─ tracking.py            # MLflow setup (tracking store, experiment, registry name)
│  └─ spam_utils.py          # shared cleaning, prediction, calibration, explainability
├─ research/
│  ├─ evaluate.py            # reproducible evaluation → reports/metrics.{md,json}
│  ├─ train_model.py         # trains the production SVM + vectorizer + calibrator
│  ├─ benchmark_models.py    # original model comparison (exploration)
│  └─ inspect_dataset.py     # lists the columns of each corpus
├─ monitoring/
│  └─ drift_report.py        # Evidently data-drift monitoring
├─ demo/                     # public demo (Streamlit Community Cloud) + its light requirements
├─ models/                   # trained artifacts (SVM, TF-IDF, calibrator)
├─ reports/                  # generated evaluation and drift reports
├─ tests/                    # pytest suite (pipeline + API)
├─ api.py                    # FastAPI service
├─ streamlit_app.py          # Streamlit dashboard
├─ telegram_bot.py           # Telegram bot
├─ retrain_feedback.py       # retraining from user feedback
├─ Dockerfile
└─ requirements.txt
```

## ⚠️ Known limitations

- The classifier is trained on English emails; other languages rely on the Ollama translation step.
- Performance drops on email sources that differ from the training corpora (see leave-one-corpus-out above).
- The corpora date mostly from the 2000s; modern phishing styles (QR codes, SMS, short links) are under-represented.
