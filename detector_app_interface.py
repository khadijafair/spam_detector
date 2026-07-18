"""
detector_app_interface.py — Interface Streamlit v2.

Améliorations vs v1 :
- Score de confiance affiché (pas juste un verdict binaire)
- Explicabilité : mots les plus influents dans la décision (graphique)
- Historique de session (multi-analyses dans la même session)
- Onglet analyse par lot (upload CSV, une colonne de texte)
- Boutons de feedback pour signaler les erreurs (log JSONL, réutilisable
  pour ré-entraîner le modèle plus tard)
- Pipeline NLP importé depuis spam_utils.py (plus de duplication de code
  avec le bot Telegram — un seul endroit à corriger si le nettoyage change)
"""
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import config
from spam_utils import (
    load_artifacts, predict_with_confidence, explain_prediction,
    is_borderline, load_corrections, detect_language, translate_to_english_sync,
)

st.set_page_config(page_title="Sentinel-Spam", page_icon="🛡️", layout="centered")

FEEDBACK_LOG_PATH = "feedback.jsonl"


@st.cache_resource
def get_artifacts():
    return load_artifacts("best_spam_model.pkl", "tfidf_vectorizer.pkl")


try:
    model, vectorizer = get_artifacts()
except Exception:
    st.error("⚠️ Fichiers modèle introuvables. Lance d'abord l'entraînement pour générer "
             "best_spam_model.pkl et tfidf_vectorizer.pkl.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []


def log_feedback(text, label, correct):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "predicted_label": label,
        "marked_correct": correct,
    }
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


st.title("🛡️ Sentinel-Spam")
st.caption("Analyseur d'e-mails / SMS avec score de confiance et explicabilité")

tab_single, tab_batch, tab_history = st.tabs(["🔍 Analyse simple", "📂 Analyse par lot", "🕒 Historique"])

# ---------------------------------------------------------------- Analyse simple
with tab_single:
    email_input = st.text_area(
        "Contenu à analyser :",
        height=220,
        placeholder="Copiez-collez le texte complet du message ici...",
    )

    if st.button("Lancer l'analyse", type="primary"):
        if not email_input.strip():
            st.warning("⚠️ Veuillez insérer du texte avant de lancer l'analyse.")
        else:
            with st.spinner("Analyse en cours..."):
                corrections = load_corrections(FEEDBACK_LOG_PATH)
                override_label = corrections.get(email_input.strip())

                lang = detect_language(email_input)
                text_for_model = email_input
                translated_shown = None
                if lang not in ("en", "unknown"):
                    translated = translate_to_english_sync(email_input, config.OLLAMA_URL, config.MODEL_LLM)
                    if translated:
                        text_for_model = translated
                        translated_shown = translated

                is_spam, confidence, score = predict_with_confidence(text_for_model, model, vectorizer)
                top_words = explain_prediction(text_for_model, model, vectorizer, top_n=8)
                if override_label is not None:
                    is_spam = override_label == "SPAM"

            st.session_state.history.append({
                "text": email_input, "is_spam": is_spam, "confidence": confidence,
            })

            st.markdown("---")
            if override_label is not None:
                st.info("ℹ️ Corrigé suite à un feedback précédent sur ce texte exact.")
            elif is_borderline(confidence):
                st.warning("⚠️ Score du modèle peu tranché (proche de 50%) — à vérifier manuellement.")
            if translated_shown:
                st.caption(f"🌐 Traduit ({lang.upper()}→EN) avant analyse")
                with st.expander("Voir le texte traduit utilisé pour l'analyse"):
                    st.text(translated_shown)

            if is_spam:
                st.error(f"🚨 **SPAM / PHISHING DÉTECTÉ** — confiance {confidence}%")
            else:
                st.success(f"✅ **MESSAGE SAIN (HAM)** — confiance {confidence}%")

            st.progress(confidence / 100)

            if top_words:
                st.subheader("Mots les plus déterminants")
                df_words = pd.DataFrame(top_words, columns=["mot", "contribution"])
                df_words["direction"] = df_words["contribution"].apply(
                    lambda c: "vers SPAM" if c > 0 else "vers HAM"
                )
                st.bar_chart(df_words.set_index("mot")["contribution"])
                st.caption("Contribution positive = pousse vers SPAM · négative = pousse vers HAM")

            col1, col2 = st.columns(2)
            label = "SPAM" if is_spam else "HAM"
            if col1.button("👍 Verdict correct"):
                log_feedback(email_input, label, True)
                st.toast("Merci, feedback enregistré ✅")
            if col2.button("👎 Verdict erroné"):
                log_feedback(email_input, label, False)
                st.toast("Merci, erreur signalée 📝")

# ---------------------------------------------------------------- Analyse par lot
with tab_batch:
    st.write("Charge un CSV avec une colonne de texte à analyser en masse.")
    uploaded = st.file_uploader("Fichier CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        col_name = st.selectbox("Colonne contenant le texte :", df.columns)
        if st.button("Analyser le fichier"):
            with st.spinner(f"Analyse de {len(df)} lignes..."):
                results = []
                for text in df[col_name].astype(str):
                    is_spam, confidence, _ = predict_with_confidence(text, model, vectorizer)
                    results.append({"verdict": "SPAM" if is_spam else "HAM", "confiance_%": confidence})
                out_df = pd.concat([df, pd.DataFrame(results)], axis=1)
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "Télécharger les résultats (CSV)",
                out_df.to_csv(index=False).encode("utf-8"),
                "resultats_analyse.csv",
                "text/csv",
            )

# ---------------------------------------------------------------- Historique
with tab_history:
    if not st.session_state.history:
        st.info("Aucune analyse effectuée pour l'instant dans cette session.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df["verdict"] = hist_df["is_spam"].map({True: "SPAM", False: "HAM"})
        st.dataframe(hist_df[["text", "verdict", "confidence"]], use_container_width=True)
        if st.button("Vider l'historique"):
            st.session_state.history = []
            st.rerun()