"""
retrain_from_feedback.py — Ré-entraînement périodique intégrant les corrections.

C'est LA vraie réponse à "le modèle n'apprend pas à long terme" : le cache
de corrections dans le bot/l'interface ne fait que masquer une erreur déjà
vue sur un texte identique. Ce script, lui, RÉ-ENTRAÎNE réellement le SVM en
ajoutant les exemples mal classés (et corrigés par un humain) au jeu
d'entraînement d'origine.

À lancer manuellement de temps en temps (ex: une fois par semaine, ou quand
feedback.jsonl a accumulé suffisamment de corrections), PAS en continu :
un SVM linéaire classique n'a pas d'apprentissage en ligne natif, on le
ré-entraîne par lots (batch).

Usage :
    python retrain_from_feedback.py
    python retrain_from_feedback.py --dataset dataset_master_clean.csv --text-col text --label-col label
    python retrain_from_feedback.py --dry-run   # évalue sans écraser les .pkl existants

Le script :
1. Charge le vectorizer + modèle ACTUELS pour réutiliser exactement les
   mêmes hyperparamètres (ngrams, class_weight, C, etc.) -- pas besoin de
   connaître le contenu exact de linear_svm_model.py.
2. Charge le dataset d'origine (auto-détection des colonnes texte/label si
   non précisées).
3. Charge feedback.jsonl, ne garde que les entrées marquées comme erronées
   (marked_correct=False), et reconstruit le label correct.
4. Fusionne dataset d'origine + corrections (les corrections sont dupliquées
   plusieurs fois pour leur donner plus de poids -- un SVM classique n'a pas
   de vrai sample_weight simple à ajuster sans ça).
5. Ré-entraîne, évalue sur un split de test, affiche les métriques avant/après.
6. Sauvegarde les nouveaux .pkl (en gardant une sauvegarde des anciens),
   sauf en --dry-run.
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import config
from spam_utils import clean_text_advanced

BASE_DIR = Path(__file__).resolve().parent

TEXT_COL_CANDIDATES = ["text", "message", "v2", "email", "content"]
LABEL_COL_CANDIDATES = ["label", "v1", "class", "category", "spam"]

# Combien de fois dupliquer chaque exemple corrigé, pour lui donner plus de
# poids face au dataset d'origine (souvent bien plus volumineux).
CORRECTION_OVERSAMPLE = 5


def autodetect_columns(df: pd.DataFrame, text_col: str | None, label_col: str | None):
    if text_col is None:
        text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"Impossible de deviner les colonnes texte/label dans {list(df.columns)}. "
            "Précise-les avec --text-col et --label-col."
        )
    return text_col, label_col


def normalize_label(value) -> int:
    """Convertit un label quelconque ('spam'/'ham', 1/0, True/False...) en 0/1."""
    if isinstance(value, (int, float)):
        return int(value)
    v = str(value).strip().lower()
    if v in ("spam", "1", "true", "phishing"):
        return 1
    if v in ("ham", "0", "false", "legit", "legitimate"):
        return 0
    raise ValueError(f"Label non reconnu : {value!r}")


def load_feedback_corrections(feedback_path: str) -> list[tuple[str, int]]:
    import json
    corrections = []
    path = Path(feedback_path)
    if not path.exists():
        print(f"⚠️  Aucun fichier de feedback trouvé à {path}, rien à intégrer.")
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
                wrong_label = entry["predicted_label"]
                corrected = 0 if wrong_label == "SPAM" else 1
                corrections.append((entry["text"], corrected))
    return corrections


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(BASE_DIR / "dataset_master_clean.csv"))
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Évalue sans écraser les fichiers .pkl")
    args = parser.parse_args()

    # 1. Récupérer les hyperparamètres du modèle/vectorizer actuels
    print("📦 Chargement des artefacts actuels pour réutiliser leurs hyperparamètres...")
    old_model = joblib.load(config.MODEL_PATH)
    old_vectorizer = joblib.load(config.VECTORIZER_PATH)
    vec_params = old_vectorizer.get_params()
    model_params = old_model.get_params()
    print(f"   Vectorizer : ngram_range={vec_params.get('ngram_range')}, max_features={vec_params.get('max_features')}")
    print(f"   Modèle     : class_weight={model_params.get('class_weight')}, C={model_params.get('C')}")

    # 2. Dataset d'origine
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        sys.exit(f"❌ Dataset introuvable : {dataset_path}. Précise le chemin avec --dataset.")
    df = pd.read_csv(dataset_path, encoding="latin-1", encoding_errors="ignore")
    text_col, label_col = autodetect_columns(df, args.text_col, args.label_col)
    print(f"📄 Dataset d'origine : {len(df)} lignes (colonnes détectées : text='{text_col}', label='{label_col}')")

    texts = df[text_col].astype(str).tolist()
    labels = [normalize_label(v) for v in df[label_col]]

    # 3. Corrections issues du feedback
    corrections = load_feedback_corrections(config.FEEDBACK_LOG_PATH)
    print(f"🔧 {len(corrections)} correction(s) trouvée(s) dans {config.FEEDBACK_LOG_PATH}")
    if not corrections:
        print("Rien à ré-entraîner (aucune correction en attente). Fin.")
        return

    for text, label in corrections:
        texts.extend([text] * CORRECTION_OVERSAMPLE)
        labels.extend([label] * CORRECTION_OVERSAMPLE)

    # 4. Nettoyage (même pipeline que la production, via spam_utils partagé)
    print("🧹 Nettoyage du texte (pipeline identique à la production)...")
    cleaned = [clean_text_advanced(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 5. Ré-entraînement avec les MÊMES hyperparamètres que le modèle en prod
    print("🏋️  Ré-entraînement...")
    new_vectorizer = TfidfVectorizer(**{
        k: v for k, v in vec_params.items() if k not in ("input", "dtype")
    })
    X_train_vec = new_vectorizer.fit_transform(X_train)
    X_test_vec = new_vectorizer.transform(X_test)

    new_model = LinearSVC(**{
        k: v for k, v in model_params.items()
    })
    new_model.fit(X_train_vec, y_train)

    # 6. Évaluation avant/après sur le même split de test
    old_test_vec = old_vectorizer.transform(X_test)
    old_preds = old_model.predict(old_test_vec)
    new_preds = new_model.predict(X_test_vec)

    print("\n=== Modèle ACTUEL (avant ré-entraînement) sur ce split de test ===")
    print(classification_report(y_test, old_preds, target_names=["HAM", "SPAM"]))
    print("\n=== Nouveau modèle (après intégration des corrections) ===")
    print(classification_report(y_test, new_preds, target_names=["HAM", "SPAM"]))

    old_f1 = f1_score(y_test, old_preds)
    new_f1 = f1_score(y_test, new_preds)
    print(f"\nF1-score : {old_f1:.4f} → {new_f1:.4f} ({'+' if new_f1 >= old_f1 else ''}{(new_f1 - old_f1):.4f})")

    if args.dry_run:
        print("\n--dry-run activé : rien n'a été écrasé.")
        return

    # 7. Sauvegarde avec backup des anciens fichiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BASE_DIR / "model_backups"
    backup_dir.mkdir(exist_ok=True)
    shutil.copy(config.MODEL_PATH, backup_dir / f"best_spam_model_{timestamp}.pkl")
    shutil.copy(config.VECTORIZER_PATH, backup_dir / f"tfidf_vectorizer_{timestamp}.pkl")

    joblib.dump(new_model, config.MODEL_PATH)
    joblib.dump(new_vectorizer, config.VECTORIZER_PATH)
    print(f"\n✅ Nouveaux modèle et vectorizer sauvegardés (anciens sauvegardés dans {backup_dir}/).")
    print("Redémarre le bot Telegram et l'interface Streamlit pour charger le nouveau modèle.")


if __name__ == "__main__":
    main()