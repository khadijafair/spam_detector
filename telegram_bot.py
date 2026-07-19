"""
telegram_bot.py — Bot hybride anti-fraude, v3.

Changements v3 (langue) :
- Les messages système / verdicts sont pilotés par i18n.py : le verdict
  d'analyse s'affiche dans la langue DU MESSAGE ANALYSÉ (en/fr supportés,
  fallback anglais pour les autres langues détectées).
- Le mode "chat" ne dépend plus d'un prompt figé en français : Ollama
  répond nativement dans la langue utilisée par la personne (EN, FR, ou
  toute autre langue qu'il maîtrise), sans template imposé.
- /start, /help, /stats s'adaptent à la langue Telegram du client si
  disponible (fallback anglais).

Améliorations précédentes conservées :
- Token via config.py / .env (jamais en dur)
- Appels Ollama en async (httpx)
- Pré-routage rapide par heuristique avant d'appeler le LLM
- Traduction automatique vers l'anglais avant classification SVM (dataset
  d'entraînement anglais), détection de langue via langdetect
- Score de confiance + explicabilité (mots les plus influents)
- Détection de score "borderline" -> second avis LLM
- Cache de corrections basé sur le feedback 👍/👎
- Logging structuré
"""
import json
import logging
from datetime import datetime, timezone

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes,
)

import config
import i18n
from spam_utils import (
    load_artifacts, predict_with_confidence, explain_prediction,
    is_borderline, load_corrections, detect_language, TRANSLATE_PROMPT_TEMPLATE,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("sentinel-spam-bot")

model_svm, vectorizer = load_artifacts(config.MODEL_PATH, config.VECTORIZER_PATH)

# Mots-clés (bilingues) qui trahissent quasi-certainement une demande d'analyse
# plutôt qu'un bavardage. Ça évite d'appeler le LLM pour trancher les cas évidents.
ANALYSIS_HINTS = (
    "http://", "https://", "www.", "@", "urgent", "gagné", "gagne", "félicitation",
    "cliquez", "cliquer", "compte bloqué", "vérifiez votre", "code promo",
    "carte bancaire", "virement", "mot de passe", "gratuit", "offre limitée",
    "winner", "you won", "click here", "verify your account", "bank account",
    "free gift", "limited offer", "password", "wire transfer",
)
# Longueur en dessous de laquelle un message SANS indice de fraude explicite
# est traité comme du chat par défaut.
CHAT_DEFAULT_WORD_LIMIT = 8


def route_heuristic(text: str) -> str | None:
    """Retourne 'ANALYSE', 'CHAT', ou None si vraiment ambigu (le LLM tranche alors)."""
    words = text.split()
    lower = text.lower()

    if len(words) >= config.SHORT_MSG_WORD_THRESHOLD:
        return "ANALYSE"
    if any(h in lower for h in ANALYSIS_HINTS):
        return "ANALYSE"
    if len(words) <= CHAT_DEFAULT_WORD_LIMIT:
        return "CHAT"
    return None


async def ask_ollama(prompt: str, timeout: float = 30.0) -> str | None:
    payload = {"model": config.MODEL_LLM, "prompt": prompt, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(config.OLLAMA_URL, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except Exception as e:
        logger.error("Ollama error: %s", e)
        return None


def log_feedback(user_id: int, text: str, predicted_label: str, correct: bool):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "text": text,
        "predicted_label": predicted_label,
        "marked_correct": correct,
    }
    with open(config.FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def telegram_ui_lang(update: Update) -> str:
    """Langue pour les commandes fixes (/start, /help, /stats) : suit la langue
    Telegram du client si dispo (en/fr), sinon anglais par défaut."""
    code = (update.effective_user.language_code or "en").split("-")[0].lower()
    return code if code in i18n.SUPPORTED_VERDICT_LANGS else "en"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.setdefault("history", [])
    context.user_data.setdefault("stats", {"analysed": 0, "spam": 0})
    lang = telegram_ui_lang(update)
    await update.message.reply_text(i18n.t(lang, "start_message"), parse_mode="Markdown")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = telegram_ui_lang(update)
    await update.message.reply_text(i18n.t(lang, "help_message"), parse_mode="Markdown")


async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = telegram_ui_lang(update)
    stats = context.user_data.get("stats", {"analysed": 0, "spam": 0})
    await update.message.reply_text(
        i18n.t(lang, "stats_message", analysed=stats["analysed"], spam=stats["spam"]),
        parse_mode="Markdown",
    )


async def orchestrateur_intelligent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    context.user_data.setdefault("history", [])
    context.user_data.setdefault("stats", {"analysed": 0, "spam": 0})

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    decision = route_heuristic(user_text)

    if decision is None:
        recent = context.user_data["history"][-4:]
        history_block = "\n".join(f"{role}: {txt}" for role, txt in recent)
        routing_prompt = (
            "You are the router of a security bot. Reply with a single word: "
            "ANALYSE if the message is a suspicious email/SMS/link to analyse, "
            "or CHAT if it's normal conversation. One word only, nothing else.\n\n"
            f"Recent history:\n{history_block}\n\n"
            f"Message: \"{user_text}\""
        )
        verdict = await ask_ollama(routing_prompt, timeout=15.0)
        # Fallback CHAT (pas ANALYSE) : un faux positif "spam" gêne plus
        # l'utilisateur qu'une réponse de chat un peu générique.
        decision = "ANALYSE" if verdict and "ANALYSE" in verdict.upper() else "CHAT"

    context.user_data["history"].append(("user", user_text))

    if decision == "CHAT":
        chat_prompt = (
            "You are a friendly, concise assistant embedded in an anti-fraud bot. "
            "Reply in the SAME LANGUAGE the user just wrote in (English, French, "
            "or whichever language they used). Keep it to 1-3 sentences.\n\n"
            f"User message: \"{user_text}\""
        )
        reponse = await ask_ollama(chat_prompt) or (
            "I'm here! Send me a suspicious message if you'd like me to analyse it."
        )
        context.user_data["history"].append(("assistant", reponse))
        await update.message.reply_text(reponse)
        return

    # --- Branche analyse ---
    # Langue détectée du message : pilote à la fois la traduction pour le SVM
    # ET la langue d'affichage du verdict (parité UX avec le contenu analysé).
    lang = detect_language(user_text)
    display_lang = lang if lang in i18n.SUPPORTED_VERDICT_LANGS else "en"

    status_msg = await update.message.reply_text(i18n.t(display_lang, "analysing"))

    corrections = load_corrections(config.FEEDBACK_LOG_PATH)
    override_label = corrections.get(user_text.strip())

    # Le modèle SVM a été entraîné sur un dataset anglais : on traduit avant
    # classification si nécessaire (sinon signal quasi nul -> score ~50%).
    text_for_model = user_text
    translation_note = ""
    if lang not in ("en", "unknown"):
        translated = await ask_ollama(TRANSLATE_PROMPT_TEMPLATE.format(text=user_text), timeout=25.0)
        if translated:
            text_for_model = translated
            translation_note = "\n\n_" + i18n.t(display_lang, "translated_note", lang=lang.upper()) + "_"
        else:
            translation_note = "\n\n_" + i18n.t(display_lang, "translation_unavailable") + "_"

    is_spam, confidence, score = predict_with_confidence(text_for_model, model_svm, vectorizer)
    top_words = explain_prediction(text_for_model, model_svm, vectorizer, top_n=5)
    uncertain = is_borderline(confidence)

    if override_label is not None:
        is_spam = override_label == "SPAM"
        note = "\n\n_" + i18n.t(display_lang, "correction_note") + "_"
    elif uncertain:
        verify_prompt = (
            "Is this message a phishing/scam attempt, or a legitimate message "
            "(e.g. professional email, HR email, normal conversation)? "
            "Reply with only SPAM or HAM.\n\n"
            f"Message: \"{user_text}\""
        )
        llm_verdict = await ask_ollama(verify_prompt, timeout=20.0)
        if llm_verdict:
            is_spam = "SPAM" in llm_verdict.upper() and "HAM" not in llm_verdict.upper()
        note = "\n\n_" + i18n.t(display_lang, "borderline_note") + "_" + translation_note
    else:
        note = translation_note

    context.user_data["stats"]["analysed"] += 1
    if is_spam:
        context.user_data["stats"]["spam"] += 1

    indices = "\n".join(
        f"  • `{w}` ({'+' if c > 0 else ''}{c:.2f})" for w, c in top_words
    ) or f"  • {i18n.t(display_lang, 'no_indicators')}"

    header = i18n.t(display_lang, "spam_header" if is_spam else "ham_header")
    conf_label = i18n.t(display_lang, "confidence_label")
    ind_label = i18n.t(display_lang, "indicators_label")

    if is_spam:
        reponse = (
            f"{header}\n"
            f"{conf_label}: *{confidence}%*\n\n"
            f"{ind_label}:\n{indices}\n\n"
            f"{i18n.t(display_lang, 'recommendation')}{note}"
        )
    else:
        reponse = (
            f"{header}\n"
            f"{conf_label}: *{confidence}%*\n\n"
            f"{ind_label}:\n{indices}{note}"
        )

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(i18n.t(display_lang, "correct_button"), callback_data=f"fb:1:{int(is_spam)}"),
        InlineKeyboardButton(i18n.t(display_lang, "wrong_button"), callback_data=f"fb:0:{int(is_spam)}"),
    ]])
    context.user_data["last_analysed_text"] = user_text
    context.user_data["last_display_lang"] = display_lang
    await status_msg.edit_text(reponse, parse_mode="Markdown", reply_markup=keyboard)


async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _, correct_flag, predicted_spam = query.data.split(":")
    text = context.user_data.get("last_analysed_text", "")
    lang = context.user_data.get("last_display_lang", "en")
    label = "SPAM" if predicted_spam == "1" else "HAM"
    log_feedback(update.effective_user.id, text, label, correct_flag == "1")
    key = "feedback_logged" if correct_flag == "1" else "feedback_wrong"
    suffix = "\n\n_" + i18n.t(lang, key) + "_"
    try:
        await query.edit_message_text(query.message.text_markdown + suffix, parse_mode="Markdown")
    except Exception:
        pass


def main():
    if not config.TELEGRAM_TOKEN:
        raise RuntimeError(
            "TELEGRAM_TOKEN missing. Create a .env file (see .env.example) "
            "and NEVER commit it (check it's in .gitignore)."
        )
    application = Application.builder().token(config.TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("stats", stats_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, orchestrateur_intelligent))
    application.add_handler(CallbackQueryHandler(feedback_callback, pattern=r"^fb:"))

    logger.info("Sentinel-Spam Bot v3 online (heuristic routing + async Ollama + bilingual EN/FR)")
    application.run_polling()


if __name__ == "__main__":
    main()