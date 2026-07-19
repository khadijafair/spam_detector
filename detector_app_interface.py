"""
detector_app_interface.py — Streamlit interface, v3.

Language design: the interface is centralized in English (consistent with
the English-trained dataset and the rest of the codebase). If pasted text
is detected as non-English, it's transparently translated to English via
Ollama before classification (the translated text is shown in an expander
for transparency). The UI labels themselves stay English regardless of the
analyzed text's language -- this keeps the tool simple and consistent,
unlike the Telegram bot, whose free-form chat mode is bilingual by design.

Other improvements kept from v2:
- Confidence score + explainability (most influential words)
- Borderline-score detection -> warning instead of a falsely confident verdict
- Correction cache from user feedback (👍/👎)
- Shared NLP pipeline via spam_utils.py (no duplication with the bot)
"""
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import core.config as config
from core.spam_utils import (
    load_artifacts, predict_with_confidence, explain_prediction,
    is_borderline, load_corrections, detect_language, translate_to_english_sync,
    load_calibrator,
)

st.set_page_config(page_title="Sentinel-Spam", page_icon="🛡️", layout="centered")

FEEDBACK_LOG_PATH = config.FEEDBACK_LOG_PATH


@st.cache_resource
def get_artifacts():
    model, vectorizer = load_artifacts(config.MODEL_PATH, config.VECTORIZER_PATH)
    calibrator = load_calibrator(config.CALIBRATOR_PATH)
    return model, vectorizer, calibrator


try:
    model, vectorizer, confidence_calibrator = get_artifacts()
    if confidence_calibrator is None:
        st.sidebar.warning(
            "No confidence calibrator found yet. Confidence scores use a rough "
            "approximation until you run `retrain_feedback.py` once."
        )
except Exception:
    st.error("⚠️ Model files not found. Run the training script first to generate "
             "best_spam_model.pkl and tfidf_vectorizer.pkl.")
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
st.caption("Email / SMS analyzer with confidence score and explainability")

tab_single, tab_batch, tab_history = st.tabs(["🔍 Single analysis", "📂 Batch analysis", "🕒 History"])

# ---------------------------------------------------------------- Single analysis
with tab_single:
    email_input = st.text_area(
        "Content to analyze:",
        height=220,
        placeholder="Paste the full message text here (any language)...",
    )

    if st.button("Run analysis", type="primary"):
        if not email_input.strip():
            st.warning("⚠️ Please enter some text before running the analysis.")
        else:
            with st.spinner("Analyzing..."):
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

                is_spam, confidence, score = predict_with_confidence(text_for_model, model, vectorizer, confidence_calibrator)
                top_words = explain_prediction(text_for_model, model, vectorizer, top_n=8)
                if override_label is not None:
                    is_spam = override_label == "SPAM"

            st.session_state.history.append({
                "text": email_input, "is_spam": is_spam, "confidence": confidence,
            })

            st.markdown("---")
            if override_label is not None:
                st.info("ℹ️ Corrected based on previous feedback on this exact text.")
            elif is_borderline(confidence):
                st.warning("⚠️ Model score inconclusive (near 50%) — please double-check manually.")
            if translated_shown:
                st.caption(f"🌐 Translated ({lang.upper()}→EN) before analysis")
                with st.expander("View the translated text used for analysis"):
                    st.text(translated_shown)

            if is_spam:
                st.error(f"🚨 **SPAM / PHISHING DETECTED** — confidence {confidence}%")
            else:
                st.success(f"✅ **LEGITIMATE MESSAGE (HAM)** — confidence {confidence}%")

            st.progress(confidence / 100)

            if top_words:
                st.subheader("Most influential words")
                df_words = pd.DataFrame(top_words, columns=["word", "contribution"])
                df_words["direction"] = df_words["contribution"].apply(
                    lambda c: "toward SPAM" if c > 0 else "toward HAM"
                )
                st.bar_chart(df_words.set_index("word")["contribution"])
                st.caption("Positive contribution = pushes toward SPAM · negative = pushes toward HAM")

            col1, col2 = st.columns(2)
            label = "SPAM" if is_spam else "HAM"
            if col1.button("👍 Correct verdict"):
                log_feedback(email_input, label, True)
                st.toast("Thanks, feedback logged ✅")
            if col2.button("👎 Wrong verdict"):
                log_feedback(email_input, label, False)
                st.toast("Thanks, error reported 📝")

# ---------------------------------------------------------------- Batch analysis
with tab_batch:
    st.write("Upload a CSV with a text column to analyze in bulk.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        col_name = st.selectbox("Column containing the text:", df.columns)
        if st.button("Analyze file"):
            with st.spinner(f"Analyzing {len(df)} rows..."):
                results = []
                for text in df[col_name].astype(str):
                    lang = detect_language(text)
                    text_for_model = text
                    if lang not in ("en", "unknown"):
                        translated = translate_to_english_sync(text, config.OLLAMA_URL, config.MODEL_LLM)
                        if translated:
                            text_for_model = translated
                    is_spam, confidence, _ = predict_with_confidence(text_for_model, model, vectorizer, confidence_calibrator)
                    results.append({
                        "verdict": "SPAM" if is_spam else "HAM",
                        "confidence_%": confidence,
                        "detected_lang": lang,
                    })
                out_df = pd.concat([df, pd.DataFrame(results)], axis=1)
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "Download results (CSV)",
                out_df.to_csv(index=False).encode("utf-8"),
                "analysis_results.csv",
                "text/csv",
            )

# ---------------------------------------------------------------- History
with tab_history:
    if not st.session_state.history:
        st.info("No analysis run yet in this session.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df["verdict"] = hist_df["is_spam"].map({True: "SPAM", False: "HAM"})
        st.dataframe(hist_df[["text", "verdict", "confidence"]], use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()