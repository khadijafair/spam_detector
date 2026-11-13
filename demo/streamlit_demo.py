"""
Sentinel-Spam — public demo for Streamlit Community Cloud.

Same production pipeline as the API / dashboard / Telegram bot
(core/spam_utils.py): cleaning + TF-IDF + Linear SVM + Platt-calibrated
confidence + word-level explanation. The Ollama translation / arbitration step
needs a local LLM, so this hosted demo expects English text.

Run locally:  streamlit run demo/streamlit_demo.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `core` importable

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import core.config as config  # noqa: E402
from core.spam_utils import (explain_prediction, is_borderline, load_artifacts,  # noqa: E402
                             load_calibrator, predict_with_confidence)

EXAMPLES = {
    "Phishing (account suspended)": "URGENT: your bank account has been suspended. Verify your password and card number within 24 hours at http://secure-login-update.com or it will be closed.",
    "Spam (prize)": "Congratulations! You have been selected to receive a $1,000 gift card. Click here to claim your prize now, offer expires today.",
    "Legitimate (meeting notes)": "Hi team, attached are the notes from this morning's meeting. The next project review is on Thursday at 10am in room B. Thanks, Sarah",
    "Legitimate (work request)": "Can you send me the updated version of the report before Friday? I want to check the numbers for the Q3 budget.",
}


@st.cache_resource
def artifacts():
    model, vec = load_artifacts(config.MODEL_PATH, config.VECTORIZER_PATH)
    return model, vec, load_calibrator(config.CALIBRATOR_PATH)


st.set_page_config(page_title="Sentinel-Spam", page_icon="🛡️")
st.title("🛡️ Sentinel-Spam — spam & phishing detector")
st.markdown(
    "Linear SVM + TF-IDF trained on **82,486 emails** from 6 public corpora. On a 16,498-email "
    "held-out test set: **F1 99.17%**, false-positive rate 0.92%. Scores are lower on email "
    "sources the model has never seen (see the README).  \n"
    "English text only in this demo · Code, evaluation, MLflow & Evidently: "
    "[github.com/khadijafair/spam_detector](https://github.com/khadijafair/spam_detector)"
)

choice = st.selectbox("Try an example", ["—"] + list(EXAMPLES))
text = st.text_area("Email or message", value=EXAMPLES.get(choice, ""), height=180,
                    placeholder="Paste the text of an email here…")

if st.button("Analyse", type="primary") and text.strip():
    model, vec, calibrator = artifacts()
    t0 = time.perf_counter()
    is_spam, confidence, _ = predict_with_confidence(text, model, vec, calibrator)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if is_spam:
        st.error(f"🚨 **Spam / phishing** — confidence {confidence}%")
    else:
        st.success(f"✅ **Legitimate** — confidence {confidence}%")
    st.caption(f"Analysed in {elapsed_ms:.1f} ms")
    if is_borderline(confidence):
        st.warning("Borderline score: in the full system this message is sent to a local LLM "
                   "(Ollama) for a second opinion.")
    words = explain_prediction(text, model, vec, top_n=8)
    if words:
        st.subheader("Most influential words")
        st.caption("TF-IDF weight × SVM coefficient (words are stemmed)")
        st.dataframe(pd.DataFrame(
            [(w, "→ spam" if c > 0 else "→ legitimate", round(c, 3)) for w, c in words],
            columns=["Word", "Pushes towards", "Contribution"]), hide_index=True)
