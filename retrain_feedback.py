"""
retrain_from_feedback.py — Ré-entraînement périodique + calibration de confiance.

Deux choses distinctes que ce script fait :

1. RÉ-ENTRAÎNEMENT réel : intègre les corrections de feedback.jsonl (Telegram
   ET Streamlit, même fichier) au dataset d'origine, refait un vrai .fit(),
   et cherche automatiquement le meilleur paramètre C par validation croisée
   (GridSearchCV) -- une vraie amélioration de performance, pas juste un cache.

2. CALIBRATION DE CONFIANCE : le score brut de LinearSVC.decision_function()
   n'a pas d'échelle de probabilité fiable, ce qui donnait des scores de
   confiance artificiellement bas (50-66%) même sur des prédictions correctes.
   Ce script entraîne un petit modèle de calibration de Platt (régression
   logistique 1D sur les scores) sur un split de validation séparé, pour que
   les pourcentages affichés reflètent une vraie fiabilité.

Rien n'est perdu : les anciens .pkl sont sauvegardés dans model_backups/
avant d'être écrasés (sauf en --dry-run, où rien n'est écrasé du tout).

Usage :
    python retrain_from_feedback.py --dry-run    # évalue sans rien écraser
    python retrain_from_feedback.py              # ré-entraîne et sauvegarde
    python retrain_from_feedback.py --dataset autre.csv --text-col X --label-col Y
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score

import core.config as config
from core.spam_utils import clean_text_advanced

BASE_DIR = Path(__file__).resolve().parent

TEXT_COL_CANDIDATES = ["clean_text", "text", "message", "v2", "email", "content"]
LABEL_COL_CANDIDATES = ["label_num", "label", "v1", "class", "category", "spam"]

# Combien de fois dupliquer chaque exemple corrigé, pour lui donner plus de
# poids face au dataset d'origine (souvent bien plus volumineux).
CORRECTION_OVERSAMPLE = 5

# Grille de recherche pour le paramètre de régularisation C du SVM linéaire.
C_GRID = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]


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
    parser.add_argument("--dataset", default=str(config.DATASET_MASTER_PATH))
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Évalue sans écraser les fichiers .pkl")
    parser.add_argument("--no-tune", action="store_true", help="Désactive la recherche du meilleur C (plus rapide)")
    args = parser.parse_args()

    # 1. Récupérer les hyperparamètres du modèle/vectorizer actuels (sauf C, qu'on retune)
    print("📦 Chargement des artefacts actuels pour réutiliser leurs hyperparamètres...")
    old_model = joblib.load(config.MODEL_PATH)
    old_vectorizer = joblib.load(config.VECTORIZER_PATH)
    vec_params = old_vectorizer.get_params()
    model_params = old_model.get_params()
    print(f"   Vectorizer : ngram_range={vec_params.get('ngram_range')}, max_features={vec_params.get('max_features')}")
    print(f"   Modèle     : class_weight={model_params.get('class_weight')}, C actuel={model_params.get('C')}")

    # 2. Dataset d'origine (déjà nettoyé -> pas de re-nettoyage)
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        sys.exit(f"❌ Dataset introuvable : {dataset_path}. Précise le chemin avec --dataset.")
    df = pd.read_csv(dataset_path, encoding="latin-1", encoding_errors="ignore")
    text_col, label_col = autodetect_columns(df, args.text_col, args.label_col)
    print(f"📄 Dataset d'origine : {len(df)} lignes (colonnes détectées : text='{text_col}', label='{label_col}')")
    print(f"   ⚠️  Le texte de '{text_col}' est traité comme DÉJÀ nettoyé (pas de re-nettoyage appliqué).")

    before = len(df)
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]
    dropped = before - len(df)
    if dropped:
        print(f"   🧹 {dropped} ligne(s) supprimée(s) (texte ou label manquant/vide).")

    dataset_texts_clean = df[text_col].tolist()
    dataset_labels = [normalize_label(v) for v in df[label_col]]

    # 3. Corrections issues du feedback (texte BRUT -> nettoyage identique à la prod)
    corrections = load_feedback_corrections(config.FEEDBACK_LOG_PATH)
    print(f"🔧 {len(corrections)} correction(s) trouvée(s) dans {config.FEEDBACK_LOG_PATH}")
    if not corrections:
        print("Rien à ré-entraîner (aucune correction en attente). Fin.")
        return

    print("🧹 Nettoyage des textes de correction (pipeline identique à la production)...")
    correction_texts_clean, correction_labels = [], []
    for raw_text, label in corrections:
        cleaned = clean_text_advanced(raw_text)
        correction_texts_clean.extend([cleaned] * CORRECTION_OVERSAMPLE)
        correction_labels.extend([label] * CORRECTION_OVERSAMPLE)

    all_texts = dataset_texts_clean + correction_texts_clean
    all_labels = dataset_labels + correction_labels

    # 4. Split en 3 : train (fit du modèle) / calib (fit du calibrateur de
    #    confiance) / test (évaluation finale, jamais vu par les deux autres).
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        all_texts, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    calib_texts, test_texts, calib_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    print(f"📊 Split : {len(train_texts)} train / {len(calib_texts)} calibration / {len(test_texts)} test")

    # 5. Vectorisation (même hyperparamètres que le vectorizer en prod)
    new_vectorizer = TfidfVectorizer(**{
        k: v for k, v in vec_params.items() if k not in ("input", "dtype")
    })
    X_train = new_vectorizer.fit_transform(train_texts)
    X_calib = new_vectorizer.transform(calib_texts)
    X_test = new_vectorizer.transform(test_texts)

    # 6. Ré-entraînement, avec recherche du meilleur C par validation croisée
    #    (amélioration réelle de performance, pas juste l'ajout de corrections)
    base_params = {k: v for k, v in model_params.items() if k != "C"}
    if args.no_tune:
        print(f"🏋️  Entraînement avec C={model_params.get('C')} (recherche désactivée)...")
        new_model = LinearSVC(**model_params)
        new_model.fit(X_train, train_labels)
    else:
        print(f"🔍 Recherche du meilleur C parmi {C_GRID} (validation croisée, scoring=F1)...")
        grid = GridSearchCV(
            LinearSVC(**base_params), param_grid={"C": C_GRID}, scoring="f1", cv=5, n_jobs=-1,
        )
        grid.fit(X_train, train_labels)
        new_model = grid.best_estimator_
        print(f"   Meilleur C trouvé : {grid.best_params_['C']} (F1 CV = {grid.best_score_:.4f})")

    # 7. Calibration de confiance (Platt scaling) sur le split dédié
    print("🎯 Calibration des scores de confiance sur le split de validation...")
    calib_scores = new_model.decision_function(X_calib).reshape(-1, 1)
    calibrator = LogisticRegression()
    calibrator.fit(calib_scores, calib_labels)

    # 8. Évaluation finale (avant/après) sur le split de test, jamais vu
    old_test_vec = old_vectorizer.transform(test_texts)
    old_preds = old_model.predict(old_test_vec)
    new_preds = new_model.predict(X_test)

    print("\n=== Modèle ACTUEL (avant ré-entraînement) ===")
    print(classification_report(test_labels, old_preds, target_names=["HAM", "SPAM"]))
    print("\n=== Nouveau modèle (corrections + C optimisé + calibration) ===")
    print(classification_report(test_labels, new_preds, target_names=["HAM", "SPAM"]))

    old_f1 = f1_score(test_labels, old_preds)
    new_f1 = f1_score(test_labels, new_preds)
    print(f"\nF1-score : {old_f1:.4f} → {new_f1:.4f} ({'+' if new_f1 >= old_f1 else ''}{(new_f1 - old_f1):.4f})")

    # Sanity check de la calibration : la confiance moyenne sur les bonnes
    # prédictions doit être nettement plus haute que sur les erreurs.
    test_scores = new_model.decision_function(X_test)
    test_proba_spam = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]
    confidences = np.where(new_preds == 1, test_proba_spam, 1 - test_proba_spam) * 100
    correct_mask = np.array(new_preds) == np.array(test_labels)
    print(f"Confiance moyenne (prédictions correctes) : {confidences[correct_mask].mean():.1f}%")
    if (~correct_mask).sum() > 0:
        print(f"Confiance moyenne (prédictions erronées)  : {confidences[~correct_mask].mean():.1f}%")

    if args.dry_run:
        print("\n--dry-run activé : rien n'a été écrasé.")
        return

    # 9. Sauvegarde avec backup des anciens fichiers -- rien n'est perdu
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BASE_DIR / "models" / "model_backups"
    backup_dir.mkdir(exist_ok=True)
    shutil.copy(config.MODEL_PATH, backup_dir / f"best_spam_model_{timestamp}.pkl")
    shutil.copy(config.VECTORIZER_PATH, backup_dir / f"tfidf_vectorizer_{timestamp}.pkl")
    if Path(config.CALIBRATOR_PATH).exists():
        shutil.copy(config.CALIBRATOR_PATH, backup_dir / f"confidence_calibrator_{timestamp}.pkl")

    joblib.dump(new_model, config.MODEL_PATH)
    joblib.dump(new_vectorizer, config.VECTORIZER_PATH)
    joblib.dump(calibrator, config.CALIBRATOR_PATH)
    print(f"\n✅ Modèle, vectorizer et calibrateur sauvegardés (anciens dans {backup_dir}/).")
    print("Redémarre le bot Telegram et/ou l'interface Streamlit pour charger le nouveau modèle.")


if __name__ == "__main__":
    main()