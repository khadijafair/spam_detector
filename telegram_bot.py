"""
telegram_bot.py — Conversational interface. Run with: python telegram_bot.py

Free-form chat replies in whichever language the user writes in (Ollama
handles this natively). Analysis verdicts display in the language of the
analyzed text (English/French, English fallback for others). Shares the
same pipeline as api.py and streamlit_app.py (core.spam_utils).
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

import core.config as config
from core import i18n
from core.spam_utils import (
    load_artifacts, predict_with_confidence, explain_prediction,
    is_borderline, load_corrections, detect_language, TRANSLATE_PROMPT_TEMPLATE,
    load_calibrator,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("sentinel-spam-bot")

model_svm, vectorizer = load_artifacts(config.MODEL_PATH, config.VECTORIZER_PATH)
confidence_calibrator = load_calibrator(config.CALIBRATOR_PATH)
if confidence_calibrator is None:
    logger.warning("No confidence calibrator found -- run retrain_feedback.py once to generate one.")

# Bilingual keywords that clearly indicate an analysis request rather than chat,
# so obvious cases skip the LLM routing call entirely.
ANALYSIS_HINTS = (
    "http://", "https://", "www.", "@", "urgent", "gagné", "gagne", "félicitation",
    "cliquez", "cliquer", "compte bloqué", "vérifiez votre", "code promo",
    "carte bancaire", "virement", "mot de passe", "gratuit", "offre limitée",
    "winner", "you won", "click here", "verify your account", "bank account",
    "free gift", "limited offer", "password", "wire transfer",
)
CHAT_DEFAULT_WORD_LIMIT = 8  # short + no fraud signal -> treated as chat by default


def route_heuristic(text: str) -> str | None:
    """Returns 'ANALYSE', 'CHAT', or None if genuinely ambiguous (LLM decides then)."""
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
        "user_id": user_id, "text": text,
        "predicted_label": predicted_label, "marked_correct": correct,
    }
    with open(config.FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def telegram_ui_lang(update: Update) -> str:
    """Language for fixed commands (/start, /help, /stats): follows the
    client's Telegram language if supported, English otherwise."""
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
        i18n.t(lang, "stats_message", analysed=stats["analysed"], spam=stats["spam"]), parse_mode="Markdown",
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
            f"Recent history:\n{history_block}\n\nMessage: \"{user_text}\""
        )
        verdict = await ask_ollama(routing_prompt, timeout=15.0)
        decision = "ANALYSE" if verdict and "ANALYSE" in verdict.upper() else "CHAT"

    context.user_data["history"].append(("user", user_text))

    if decision == "CHAT":
        chat_prompt = (
            "You are a friendly, concise assistant embedded in an anti-fraud bot. "
            "Reply in the SAME LANGUAGE the user just wrote in. Keep it to 1-3 sentences.\n\n"
            f"User message: \"{user_text}\""
        )
        reponse = await ask_ollama(chat_prompt) or "I'm here! Send me a suspicious message if you'd like me to analyse it."
        context.user_data["history"].append(("assistant", reponse))
        await update.message.reply_text(reponse)
        return

    # --- Analysis branch ---
    lang = detect_language(user_text)
    display_lang = lang if lang in i18n.SUPPORTED_VERDICT_LANGS else "en"
    status_msg = await update.message.reply_text(i18n.t(display_lang, "analysing"))

    corrections = load_corrections(config.FEEDBACK_LOG_PATH)
    override_label = corrections.get(user_text.strip())

    text_for_model = user_text
    translation_note = ""
    if lang not in ("en", "unknown"):
        translated = await ask_ollama(TRANSLATE_PROMPT_TEMPLATE.format(text=user_text), timeout=25.0)
        if translated:
            text_for_model = translated
            translation_note = "\n\n_" + i18n.t(display_lang, "translated_note", lang=lang.upper()) + "_"
        else:
            translation_note = "\n\n_" + i18n.t(display_lang, "translation_unavailable") + "_"

    is_spam, confidence, score = predict_with_confidence(text_for_model, model_svm, vectorizer, confidence_calibrator)
    top_words = explain_prediction(text_for_model, model_svm, vectorizer, top_n=5)
    uncertain = is_borderline(confidence)

    if override_label is not None:
        is_spam = override_label == "SPAM"
        note = "\n\n_" + i18n.t(display_lang, "correction_note") + "_"
    elif uncertain:
        verify_prompt = (
            "Is this message a phishing/scam attempt, or a legitimate message "
            "(e.g. professional email, HR email, normal conversation)? "
            "Reply with only SPAM or HAM.\n\nMessage: \"{user_text}\"".format(user_text=user_text)
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
        reponse = f"{header}\n{conf_label}: *{confidence}%*\n\n{ind_label}:\n{indices}\n\n{i18n.t(display_lang, 'recommendation')}{note}"
    else:
        reponse = f"{header}\n{conf_label}: *{confidence}%*\n\n{ind_label}:\n{indices}{note}"

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


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Global safety net: logs the error and tells the user, instead of leaving
    them without a reply."""
    logger.error("Unhandled exception while processing update: %s", context.error, exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ Something went wrong while processing your message. Please try again "
                "in a moment. / Une erreur est survenue, réessaie dans un instant."
            )
        except Exception:
            pass


def main():
    if not config.TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN missing. Create a .env file (see .env.example).")
    application = Application.builder().token(config.TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("stats", stats_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, orchestrateur_intelligent))
    application.add_handler(CallbackQueryHandler(feedback_callback, pattern=r"^fb:"))
    application.add_error_handler(error_handler)

    logger.info("Sentinel-Spam Bot online")
    application.run_polling()


if __name__ == "__main__":
    main()