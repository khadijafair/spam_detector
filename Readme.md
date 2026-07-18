Voici un `README.md` complet, moderne et structuré pour ton projet. Il met parfaitement en valeur l'architecture hybride (SVM + LLM) et l'aspect MLOps que tu as développés, ce qui en fait un excellent atout pour ton portfolio d'ingénieur.

---

```markdown
# 🛡️ Sentinel-Spam : Système Hybride Anti-Fraude & IA Conversationnelle

Sentinel-Spam est une solution de détection de phishing et de spams de bout en bout (End-to-End). Le projet combine l'efficacité mathématique du **Machine Learning classique (Linear SVM)** pour le traitement de masse ultra-rapide et la puissance contextuelle d'un **LLM local (Ollama)** pour l'analyse des messages courts (SMS) et l'interaction naturelle.

L'ensemble de l'écosystème est conçu pour être modulaire, hautement performant et industrialisé (MLOps).

---

## 🏗️ Architecture du Système

Le projet repose sur un routage sémantique intelligent pour optimiser les coûts de calcul, la latence et la fiabilité :


```

📥 Message Reçu (Telegram / UI)
│
▼
[ Routeur LLM (Ollama) ] ──► (Bavardage / Question) ──► 💬 Réponse fluide du LLM
│
▼ (Demande d'analyse de sécurité)
[ Analyse de Longueur ]
│
├─► LONG (E-mail) ──► [ Moteur SVM ] ──► 🚀 Verdict en 1ms (Précision : 98.5%)
│
└─► COURT (SMS)   ──► [ Moteur LLM ] ──► 🧠 Analyse sémantique fine des fraudes

```

### 📊 Comparatif des Composants

| Fonctionnalité | Moteur SVM (Machine Learning) | Moteur LLM (Ollama) |
| :--- | :--- | :--- |
| **Spécialité** | E-mails longs, structures textuelles denses | SMS courts, contexte flou, Chatbot |
| **Vitesse d'exécution**| `< 1 milliseconde` | ~500ms à quelques secondes |
| **Empreinte RAM** | Quelques Mo (`.pkl`) | ~3 à 4 Go (Modèle quantifié) |
| **Risque d'injection** | **Nul** (Frontière mathématique stricte) | Moyen (Sensible au prompt hacking) |

---

## 🚀 Fonctionnalités Clés

*   **Pipeline NLP Avancé :** Nettoyage automatique, extraction de tokens critiques (`urltoken`, `moneytoken`, `phonetoken`) et normalisation par racinisation (*Porter Stemmer*).
*   **Modèle ML Champion :** Classifieur *Linear SVM (Balanced)* entraîné sur un échantillon maître, atteignant un **F1-Score de 98.56%**.
*   **Sérialisation (MLOps) :** Sauvegarde et chargement instantané du modèle via `joblib` sans ré-entraînement.
*   **Multi-Interfaces :**
    *   **FastAPI :** API REST de production documentée sous Swagger UI.
    *   **Streamlit :** Interface web épurée et grand public pour tester des e-mails.
    *   **Telegram Bot :** Bot hybride et conversationnel exploitant Ollama (`llama3.2:3b`) pour aiguiller les requêtes.

---

## 🛠️ Installation et Configuration

### 1. Prérequis
*   Python 3.10 ou supérieur
*   Docker Desktop (Optionnel, pour la conteneurisation)
*   [Ollama](https://ollama.com/) installé et actif en local

### 2. Cloner le projet et installer les dépendances
```bash
git clone [https://github.com/ton-username/spam_detector.git](https://github.com/ton-username/spam_detector.git)
cd spam_detector

# Créer et activer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Sur Mac/Linux
.\.venv\Scripts\activate   # Sur Windows (PowerShell)

# Installer les packages nécessaires
pip install -r requirements.txt

```

### 3. Télécharger le modèle LLM local

Dans votre terminal, téléchargez le modèle optimisé pour le routeur :

```bash
ollama pull llama3.2:3b

```

---

## 🏃‍♂️ Guide d'Utilisation

### Option A : Lancer le Bot Telegram Hybride (Recommandé)

1. Modifiez la variable `TOKEN_TELEGRAM` dans `telegram_bot.py` avec le jeton fourni par `@BotFather`.
2. Lancez le script :

```bash
python telegram_bot.py

```

### Option B : Lancer l'Interface Web Streamlit

Pour une démonstration visuelle et interactive dans le navigateur :

```bash
streamlit run interface.py

```

*L'interface sera accessible sur `http://localhost:8501*`

### Option C : Déploiement de l'API de Production (FastAPI)

Pour exposer le modèle sous forme de micro-service :

```bash
uvicorn app:app --reload

```

*Découvrez la documentation interactive sur `http://127.0.0.1:8000/docs*`

---

## 🐋 Conteneurisation (Docker)

Pour figer l'environnement et déployer l'interface Streamlit sur n'importe quel serveur cloud :

```bash
# Construire l'image Docker
docker build -t spam-detector-app .

# Exécuter le conteneur sur le port 8501
docker run -d -p 8501:8501 spam-detector-app

```

---

## 📈 Performances du Classifieur ML

Évaluation finale obtenue lors du benchmark sur 20 000 lignes (Unigrams + Bigrams TF-IDF) :

* **Accuracy :** 98.50%
* **Precision :** 98.23%
* **Recall (Rappel) :** 98.90%
* **F1-Score :** 98.56%

> **Note de sécurité :** Le modèle utilise un ajustement des poids de classes (`class_weight='balanced'`) afin de minimiser drastiquement le taux de faux négatifs (un e-mail de phishing non détecté étant plus dangereux qu'un e-mail sain bloqué par erreur).

```

```