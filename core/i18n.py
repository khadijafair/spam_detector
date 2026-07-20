"""
i18n.py — User-facing strings for the Telegram bot.

Design: the app/API/code stay in English (matches the English-trained
dataset). The Telegram bot's verdict messages adapt to the language of the
analyzed text (en/fr here, English fallback otherwise). The bot's free-chat
mode doesn't use this file -- Ollama replies natively in whatever language
the user writes in.
"""

STRINGS = {
    "en": {
        "analysing": "🔍 Analysing...",
        "spam_header": "🚨 *VERDICT: PHISHING / SPAM*",
        "ham_header": "✅ *VERDICT: LEGITIMATE MESSAGE (HAM)*",
        "confidence_label": "Model confidence",
        "indicators_label": "Most influential indicators",
        "no_indicators": "(no significant lexical indicator isolated)",
        "recommendation": "➡️ Do not click any link, do not share any personal data.",
        "translated_note": "🌐 Automatically translated ({lang}→EN) before analysis.",
        "translation_unavailable": "⚠️ Translation unavailable, analysis run on the original text (less reliable).",
        "borderline_note": "⚠️ Model score inconclusive (near 50%) — verified with a secondary AI check.",
        "correction_note": "ℹ️ Corrected based on previous feedback on this exact text.",
        "correct_button": "👍 Correct",
        "wrong_button": "👎 Wrong",
        "feedback_logged": "Thanks, feedback logged ✅",
        "feedback_wrong": "Thanks, error reported 📝",
        "start_message": (
            "🛡️ *Sentinel-Spam — Hybrid Anti-Fraud System*\n\n"
            "Chat with me normally, or paste a suspicious message/SMS/email and "
            "I'll tell you if it's fraudulent, with a confidence score and the "
            "clues behind the decision.\n\nCommands: /help /stats"
        ),
        "help_message": (
            "ℹ️ *Help*\n"
            "- Send a normal message → I'll chat with you (English or French).\n"
            "- Send a suspicious text (email, SMS, link) → I'll analyse it and give a verdict.\n"
            "- After a verdict, use 👍/👎 to tell me if I was right: it feeds a log "
            "that can be used to retrain the model.\n"
            "/stats — see your usage statistics"
        ),
        "stats_message": "📊 *Your stats*\nMessages analysed: {analysed}\nSpam detected: {spam}",
    },
    "fr": {
        "analysing": "🔍 Analyse en cours...",
        "spam_header": "🚨 *VERDICT : PHISHING / SPAM*",
        "ham_header": "✅ *VERDICT : MESSAGE SAIN (HAM)*",
        "confidence_label": "Confiance du modèle",
        "indicators_label": "Indices les plus déterminants",
        "no_indicators": "(aucun indice lexical significatif isolé)",
        "recommendation": "➡️ Ne cliquez sur aucun lien, ne fournissez aucune donnée personnelle.",
        "translated_note": "🌐 Traduit automatiquement ({lang}→EN) avant analyse.",
        "translation_unavailable": "⚠️ Traduction indisponible, analyse effectuée sur le texte original (moins fiable).",
        "borderline_note": "⚠️ Score du modèle peu tranché (proche de 50%) — vérifié par une IA en second avis.",
        "correction_note": "ℹ️ Corrigé suite à un feedback précédent sur ce texte exact.",
        "correct_button": "👍 Correct",
        "wrong_button": "👎 Erreur",
        "feedback_logged": "Merci, feedback enregistré ✅",
        "feedback_wrong": "Merci, erreur signalée 📝",
        "start_message": (
            "🛡️ *Sentinel-Spam — Système Hybride Anti-Fraude*\n\n"
            "Discute normalement avec moi, ou colle un message/SMS/e-mail suspect "
            "et je te dis s'il s'agit d'une fraude, avec un score de confiance et "
            "les indices qui ont motivé la décision.\n\nCommandes : /help /stats"
        ),
        "help_message": (
            "ℹ️ *Aide*\n"
            "- Envoie un message normal → je discute avec toi (français ou anglais).\n"
            "- Envoie un texte suspect (mail, SMS, lien) → j'analyse et te donne un verdict.\n"
            "- Après un verdict, utilise 👍/👎 pour me dire si j'ai eu raison : "
            "ça alimente un journal utilisable pour ré-entraîner le modèle.\n"
            "/stats — voir tes statistiques d'utilisation"
        ),
        "stats_message": "📊 *Tes statistiques*\nMessages analysés : {analysed}\nSpams détectés : {spam}",
    },
}

SUPPORTED_VERDICT_LANGS = ("en", "fr")


def t(ui_lang: str, key: str, **kwargs) -> str:
    """Returns the translated string for `ui_lang` (falls back to English)."""
    ui_lang = ui_lang if ui_lang in STRINGS else "en"
    template = STRINGS[ui_lang].get(key, STRINGS["en"].get(key, key))
    return template.format(**kwargs) if kwargs else template