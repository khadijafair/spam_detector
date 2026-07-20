
## 🛡️ Sentinel-Spam — Hybrid Anti-Fraud & Conversational AI System

Sentinel-Spam is an end-to-end phishing and spam detection ecosystem. The project bridges the gap between classical machine learning and modern generative AI by combining an ultra-fast **Linear SVM classifier (TF-IDF)** with a local **Ollama LLM** for semantic verification, translation, and advanced conversational routing.

---

## 📌 Prerequisites

Before running this project, ensure you have the following installed on your machine:

*   **Python 3.9 to 3.11** (Recommended for ML library stability)
*   **Git** (To clone the repository)
*   **Ollama** (For running local LLMs. Download it from [ollama.com](https://ollama.com))
*   *Optional:* A **Telegram Bot Token** (If you plan to deploy and test the Telegram bot interface)

---

## ⚡ Quick Start (Run the Project in 5 Minutes)

### 1. Clone & Prepare Environment
Open your terminal and run the following commands to clone the project and configure a virtual environment:

```bash
# Clone the repository
git clone [https://github.com/khadijafair/spam_detector.git](https://github.com/khadijafair/spam_detector.git)
cd spam_detector

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate  
# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat
# On Windows (PowerShell):
.\.venv\Scripts\activate.ps1

# Install all required project dependencies
pip install --upgrade pip
pip install -r requirements.txt
2. Launch local AI Core (Ollama)Make sure the Ollama application is running on your computer, then pull the required model from your terminal:Bashollama pull llama3.2:3b
3. Setup Environment VariablesCopy the example environment template to create your live configuration file:Bash# On macOS/Linux:
cp .env.example .env

# On Windows (Command Prompt):
copy .env.example .env
Open the newly created .env file in your text editor and add your configuration details:Extrait de codeTELEGRAM_TOKEN="your_bot_token_from_botfather"
OLLAMA_MODEL="llama3.2:3b"
4. Choose Your Interface & RunSentinel-Spam offers three different ways to interact with the system. You can run any (or all) of them simultaneously:Option A: Web Dashboard (Streamlit) — Best for manual testing and data analyticsBashstreamlit run detector_app_interface.py
Option B: Microservice API (FastAPI) — Best for integrating into other appsBashuvicorn app:app --reload
Once running, view interactive API documentation at: http://127.0.0.1:8000/docsOption C: Telegram Bot — Best for live conversational text classificationBashpython telegram_bot.py

🗺️ Project Evolution FrameworkThis repository documents a complete 4-phase production-grade MLOps engineering pipeline:

┌─────────────────────────────┐      ┌─────────────────────────────┐
│  Phase 1: Benchmarking      │ ───► │  Phase 2: Core Engineering  │
│  • Checked raw distributions│      │  • Unified cleaning utils  │
│  • SVM won (98.56% F1)      │      │  • Fit custom TF-IDF matrix │
└─────────────────────────────┘      └─────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────┐      ┌─────────────────────────────┐
│  Phase 4: Continuous MLOps  │ ◄─── │  Phase 3: The Hybrid Shift  │
│  • Log active feedback loop │      │  • Multilingual Ollama route│
│  • On-the-fly retraining    │      │  • Streamlit, API, & Bot UI │
└─────────────────────────────┘      └─────────────────────────────┘

Phase 1: Exploration & Model Benchmarking: Initial dataset exploration (inspect_dataset.py) led to a model tournament (benchmark_models.py). Linear SVM was selected as our primary core engine due to its exceptional F1-Score (98.56%) and sub-millisecond inference speeds.Phase 2: Core Engineering & Extraction: Built a deterministic text processing pipeline (spam_utils.py) that strips HTML/regex noise, cleans text strings, and applies Porter Stemming before applying a custom-tuned TF-IDF matrix.Phase 3: The Hybrid Shift: Introduced automatic language detection (langdetect). Non-English inputs are seamlessly translated via Ollama before classification. Furthermore, Ollama acts as an automated "human-like judge" if the SVM's prediction confidence lands inside a borderline 50% ambiguity zone.Phase 4: Continuous MLOps Loop: Real-time user corrections are logged to an append-only file (feedback.jsonl). Running retrain_from_feedback.py automatically initiates grid search tuning, recalculates Platt scaling coefficients, and updates weights safely.🏗️ Routing Logic ArchitectureThe flow diagram below displays how user inputs are safely evaluated across multilingual boundaries:

Plaintext 📥 Input Text (Any Language)
       │
       ▼
 [ Language Detection ] ──► Non-English? ──► [ Ollama Translation ➔ English ]
       │                                                    │
       ▼ ◄──────────────────────────────────────────────────┘
 [ Linear SVM Classifier ] ──► Raw Margin Score
       │
       ▼
 [ Confidence Calibrator ] ──► Calibrated Confidence % (Platt Scaling)
       │
       ▼
 ┌───────────────────────────────┐
 │ Score Borderline (~50% Zone)? │ ──► YES ──► [ Ollama Secondary Arbitration ]
 └───────────────────────────────┘                    │
       │ NO                                           ▼
       ▼ ◄────────────────────────────────────────────┘
 🚨/✅ Final Verdict + Key Influential Words (Explainability Matrix)


🎨 Key System FeaturesZero Logic Drift (spam_utils.py): The cleaning, tokenization, and vectorization process is entirely centralized. Every interface uses identical code blocks to guarantee matching predictions across all platforms.Explainable ML Core: Shows users why text was classified a certain way. The system extracts feature weights by computing the impact vector: $\text{Weight} = \text{TF-IDF Score} \times \text{SVM Coefficient}$.Calibrated Probabilities: Instead of using crude sigmoid functions, a Platt-scaling model maps raw hyper-plane mathematical distances to accurate, reliable real-world percentages.Automated Retraining Orchestrator: Instantly fine-tunes regularizations ($C$), recalculates calibration matrices, and structures date-stamped file backups when processing updated human feedback arrays.

📁 Repository StructurePlaintextspam_detector
├─ core/
│  ├─ config.py                 # Safe environment configuration loader
│  ├─ i18n.py                   # Multi-language dictionary engine (EN/FR)
│  └─ spam_utils.py             # Global NLP preprocessing & explainability utils
├─ research/                    
│  ├─ train_model.py           # Legacy historical ingestion scripts
│  ├─ benchmark_models.py      # Classifier comparison tournament runner
│  └─ inspect_dataset.py       # Basic dataset metrics & vocabulary balance
├─ data/
├─ models/                      # Saved Model Binaries
│  ├─ best_spam_model.pkl       # Serialized SVM weight matrices
│  ├─ confidence_calibrator.pkl # Saved Platt-scaling coefficients
│  └─ tfidf_vectorizer.pkl      # Serialized TF-IDF feature vocabulary
├─ api.py                       # FastAPI Production Endpoint Engine
├─ streamli_app.py              # Streamlit Interactive Web Application
├─ telegram_bot.py              # Asynchronous Telegram Gateway Interface
├─ retrain_feedback.py          # Continuous learning MLOps worker script
├─ Dockerfile                   # Containerization configuration manifest
└─ requirements.txt             # Project software dependencies manifest

📈 Baseline Model PerformanceDuring our initial model selection phase, the foundational Linear SVM model delivered the following metrics against our evaluation baseline data matrix:MetricScorePerformance TrajectoryAccuracy98.50%High structural generalizationPrecision98.23%Exceptionally low false-positive occurrencesRecall98.90%Comprehensive threat capture rateF1-Score98.56%Balanced precision-recall mathematical harmony💡 Design Architecture Note: The core SVM classifier explicitly initializes with class_weight='balanced'. This design choice artificially penalizes False Negatives, forcing the system to actively filter out subtle, malicious fraud schemes.


