Ah ! Autant pour moi, je vois exactement ce que tu veux dire. Tu veux que le `README.md` raconte **toute l'histoire du projet**, depuis les scripts de base du tout début (exploration du dataset, comparaison des modèles mathématiques classiques) jusqu'à l'architecture hybride ultra-avancée d'aujourd'hui.

C'est une excellente idée, car cela montre ta démarche d'ingénieur : tu n'as pas juste sorti un LLM du chapeau, tu as analysé la donnée, benchmarké des modèles, et construit une solution incrémentale.

Voici le `README.md` global et définitif qui retrace **absolument tout**, en intégrant chaque fichier de ton arborescence (les scripts d'analyse, de benchmark, la phase 1, etc.).

---

```markdown
# 🛡️ Sentinel-Spam — Hybrid Anti-Fraud & Conversational AI System

Sentinel-Spam is an end-to-end phishing and spam detection ecosystem. It traces its evolution from a **classical data science exploration and benchmarking framework** to a highly advanced **hybrid production architecture** combining an ultra-fast **Linear SVM classifier (TF-IDF)** with a local **Ollama LLM** for semantic verification, translation, and live conversational routing.

---

## 🗺️ Project Evolution Framework (How We Built It)

The repository reflects a complete 4-phase engineering pipeline, starting from raw data up to an automated MLOps production environment:

### Phase 1: Exploration & Model Benchmarking
*   **Data Inspection:** Initial exploration of class distribution and vocabulary density using `inspect_dataset.py`.
*   **The Tournament:** Running `benchmark_models.py` to test multiple classical algorithms (Multinomial Naive Bayes, Random Forest, Logistic Regression, and SVM). 
*   **The Verdict:** Linear SVM was selected as the core engine due to its exceptional F1-Score (98.56%) and lightning-fast inference time (< 1ms).

### Phase 2: Core Engineering & Extraction
*   **Pipeline Setup:** Construction of `pipeline_phase1.py` and `spam_utils.py` to clean text, strip HTML/regex noise, and apply a Porter Stemmer.
*   **Vectorization:** Transforming text into numerical space using a custom-tuned TF-IDF matrix (`tfidf_vectorizer.pkl`).

### Phase 3: The Hybrid Shift (Adding LLM & Interfaces)
*   **Bilingual Capability:** Integrating `langdetect` to capture non-English inputs (French, Spanish, etc.) and routing them to Ollama for zero-shot translation before classification.
*   **Borderline Arbitration:** Deploying Ollama to act as a secondary human-like judge whenever the SVM confidence hovers near the ambiguous 50% boundary zone.
*   **Multi-Client Deployment:** Exposing the pipeline through a **Streamlit Web UI**, a **FastAPI REST API**, and a conversational **Telegram Bot**.

### Phase 4: Continuous MLOps Loop
*   **Active Feedback:** Real-time logging of user corrections into `feedback.jsonl`.
*   **Self-Correction Retraining:** Automating model fine-tuning and Platt scaling calibration via `retrain_from_feedback.py` to fix classification errors on the fly.

---

## 🏗️ Technical Architecture & Routing Logic


```

📥 Input text (any language)
│
▼
[ Language detection ] ──► non-English? ──► [ Ollama translation → English ]
│
▼
[ Linear SVM classifier ] ──► raw decision score
│
▼
[ Confidence calibrator ] ──► calibrated confidence %  (Platt scaling)
│
▼
┌─────────────────────────────┐
│ Score borderline (~50%)?    │──yes──► [ Ollama secondary check ]
└─────────────────────────────┘
│ no
▼
✅/🚨 Verdict + top influential words (explainability)

```

---

## 🚀 Key Features

*   **Shared NLP Pipeline (`spam_utils.py`):** Unified text cleaning, tokenization, stemming, and feature extraction identical across every interface to completely prevent logic drift.
*   **Automatic Translation:** Non-English inputs are instantly translated through Ollama before entering the core classifier, maximizing accuracy against the English-trained master dataset.
*   **Calibrated Confidence Scores:** A Platt-scaling calibrator converts raw SVM hyperplane distances into robust, well-calibrated confidence percentages, avoiding crude sigmoid approximations.
*   **Explainability Model:** Highlights exactly which words triggered a specific verdict by calculating the exact impact vector: Weight = TF-IDF value × SVM coefficient.
*   **Automated Production Retraining:** Running `retrain_from_feedback.py` ingests logged corrections, re-tunes the SVM's regularization parameter (C) via cross-validated Grid Search, re-fits the Platt calibrator, and creates safety backups of old weights.

---

## 📁 Project Structure (Every File Explained)

Here is the role of every component present in the workspace:

### ⚙️ Core System & Interfaces
| File | Role |
| :--- | :--- |
| `detector_app_interface.py` | **Streamlit Web UI** (Primary interface for single text, history, and CSV batch uploads). |
| `app.py` | **FastAPI Core** exposing REST API endpoints for external microservice integration. |
| `telegram_bot.py` | Asynchronous gateway managing the interactive **Telegram Chat Bot**. |
| `config.py` | Centralized strict configuration loading from `.env` environment variables. |
| `i18n.py` | Internationalization dictionary handling English/French UI strings for the bot. |

### 🧪 Historical Development & Benchmarking (Phase 1 & 2)
| File | Role |
| :--- | :--- |
| `inspect_dataset.py` | Script used to check class balance, missing values, and raw text statistics. |
| `benchmark_models.py` | Cross-validation tournament script comparing Naive Bayes vs Random Forest vs SVM. |
| `pipeline_phase1.py` | Baseline operational workflow script for early feature-engineering tracking. |
| `linear_svm_model.py` | Script dedicated to testing the initial standalone Linear Support Vector Machine parameters. |
| `predict.py` | Lightweight command-line interface (CLI) tool for quick model inference testing. |
| `app_dataset.py` / `app_spam_ham_data.py` | Historical data processing, ingestion, and splitting helper utilities. |

### 🧠 Data, Artifacts & MLOps (Continuous Learning)
| File | Role |
| :--- | :--- |
| `spam_utils.py` | Shared NLP pipeline: engineering features, translation, and explainability vectors. |
| `retrain_from_feedback.py` | MLOps orchestrator for grid searching, performance tracking, and model tuning. |
| `best_spam_model.pkl` | Serialized Linear Support Vector Machine class weights. |
| `tfidf_vectorizer.pkl` | Term Frequency-Inverse Document Frequency vocabulary matrix. |
| `confidence_calibrator.pkl` | Platt scaling coefficients matching outputs to explicit percentage probabilities. |
| `feedback.jsonl` | Append-only raw dataset recording real-time user classification feedback (👍/👎). |
| `dataset_master_clean.csv` / `spam_ham_dataset.csv` | Cleaned master validation reference data matrices. |
| `Dockerfile` | Image manifest compiling steps to containerize the application. |

---

## 🛠️ Installation & Setup

### 1. Environment Preparation
```bash
git clone [https://github.com/khadijafair/spam_detector.git](https://github.com/khadijafair/spam_detector.git)
cd spam_detector

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt

```

### 2. Fetch the Local Intelligence Core

Ensure Ollama is running locally, then pull the model used for translation and arbitration:

```bash
ollama pull llama3.2:3b

```

### 3. Configuration

Create your local environment file at the project root directory:

```bash
cp .env.example .env

```

Open `.env` and fill in your variables securely:

```env
TELEGRAM_TOKEN="your_bot_token_from_botfather"
OLLAMA_MODEL="llama3.2:3b"

```

---

## 🏃 Execution Guide

* **Run Streamlit Web App:** `streamlit run detector_app_interface.py`
* **Run FastAPI API:** `uvicorn app:app --reload`
* **Run Telegram Bot:** `python telegram_bot.py`
* **Trigger MLOps Retraining Loop:** `python retrain_from_feedback.py`

---

## 📈 Baseline Model Performance

The foundational classifier model achieved the following metrics during the Phase 1 benchmarking tournament on the core text dataset:

* **Accuracy:** 98.50%
* **Precision:** 98.23%
* **Recall:** 98.90%
* **F1-Score:** 98.56%

> **Safety Design Note:** The underlying SVM classifier uses `class_weight='balanced'` to explicitly minimize False Negatives, ensuring active fraud threats are aggressively filtered out.

```

```
```
spam_detector
├─ app.py
├─ app_dataset.py
├─ app_spam_ham_data.py
├─ benchmark_models.py
├─ config.py
├─ dataset
├─ detector_app_interface.py
├─ Dockerfile
├─ i18n.py
├─ inspect_dataset.py
├─ linear_svm_model.py
├─ model_backups
├─ pipeline_phase1.py
├─ predict.py
├─ Readme.md
├─ requirements.txt
├─ retrain_feedback.py
├─ spam_utils.py
└─ telegram_bot.py

```