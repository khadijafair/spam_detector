"""
telegram_bot.py — Bot hybride anti-fraude, v2.

Principales améliorations vs la v1 :
- Plus de token en dur (voir config.py / .env)
- Appels Ollama en async (httpx) pour ne pas bloquer le bot pendant qu'il répond
  à d'autres utilisateurs
- Pré-routage rapide par heuristique (regex/mots-clés) AVANT d'appeler le LLM :
  on n'invoque le LLM que pour les cas ambigus, ce qui réduit fortement
  la latence moyenne et la dépendance à un LLM 3B pas toujours fiable pour
  respecter un format de sortie strict
- Mémoire de conversation courte (derniers échanges) pour un mode "chat" plus naturel
- Score de confiance + mots les plus influents affichés à l'utilisateur (explicabilité)
- Boutons de feedback (👍/👎) pour logger les erreurs de classification et
  constituer un jeu de données de ré-entraînement
- Logging structuré au lieu de print()
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

# Mots-clés qui trahissent quasi-certainement une demande d'analyse plutôt qu'un bavardage.
# Ça évite d'appeler le LLM pour trancher dans les cas évidents.
ANALYSIS_HINTS = (
    "http://", "https://", "www.", "@", "urgent", "gagné", "gagne", "félicitation",
    "cliquez", "cliquer", "compte bloqué", "vérifiez votre", "code promo",
    "carte bancaire", "virement", "mot de passe", "gratuit", "offre limitée",
)
# Longueur en dessous de laquelle un message SANS indice de fraude explicite
# est traité comme du chat par défaut. Un phishing/spam sans lien, sans
# urgence, sans mot-clé financier tient rarement en moins de 8 mots.
CHAT_DEFAULT_WORD_LIMIT = 8


def route_heuristic(text: str) -> str | None:
    """Retourne 'ANALYSE', 'CHAT', ou None si vraiment ambigu (auquel cas on demande au LLM)."""
    words = text.split()
    lower = text.lower()

    if len(words) >= config.SHORT_MSG_WORD_THRESHOLD:
        return "ANALYSE"  # un texte long est presque toujours un mail/SMS à analyser
    if any(h in lower for h in ANALYSIS_HINTS):
        return "ANALYSE"  # indice de fraude explicite, peu importe la longueur
    if len(words) <= CHAT_DEFAULT_WORD_LIMIT:
        return "CHAT"  # court + aucun indice suspect -> conversation par défaut
    return None  # zone grise (longueur moyenne, pas d'indice net) : le LLM tranche


async def ask_ollama(prompt: str, timeout: float = 30.0) -> str | None:
    payload = {"model": config.MODEL_LLM, "prompt": prompt, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(config.OLLAMA_URL, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except Exception as e:
        logger.error("Erreur Ollama: %s", e)
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.setdefault("history", [])
    context.user_data.setdefault("stats", {"analysed": 0, "spam": 0})
    await update.message.reply_text(
        "🛡️ *Sentinel-Spam — Système Hybride Anti-Fraude*\n\n"
        "Discute normalement avec moi, ou colle un message/SMS/e-mail suspect "
        "et je te dis s'il s'agit d'une fraude, avec un score de confiance et "
        "les indices qui ont motivé la décision.\n\n"
        "Commandes : /help /stats",
        parse_mode="Markdown",
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ *Aide*\n"
        "- Envoie un message normal → je discute avec toi.\n"
        "- Envoie un texte suspect (mail, SMS, lien) → j'analyse et te donne un verdict.\n"
        "- Après un verdict, utilise 👍/👎 pour me dire si j'ai eu raison : "
        "ça alimente un journal utilisable pour ré-entraîner le modèle.\n"
        "/stats — voir tes statistiques d'utilisation",
        parse_mode="Markdown",
    )


async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = context.user_data.get("stats", {"analysed": 0, "spam": 0})
    await update.message.reply_text(
        f"📊 *Tes statistiques*\n"
        f"Messages analysés : {stats['analysed']}\n"
        f"Spams détectés : {stats['spam']}",
        parse_mode="Markdown",
    )


async def orchestrateur_intelligent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    context.user_data.setdefault("history", [])
    context.user_data.setdefault("stats", {"analysed": 0, "spam": 0})

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    decision = route_heuristic(user_text)

    if decision is None:
        # Cas ambigu seulement : on sollicite le LLM, avec un peu d'historique
        # pour un routage plus contextuel.
        recent = context.user_data["history"][-4:]
        history_block = "\n".join(f"{r}: {t}" for r, t in recent)
        routing_prompt = (
            "Tu es le routeur d'un bot de sécurité. Réponds uniquement par le mot "
            "ANALYSE si le message est un e-mail/SMS/lien suspect à analyser, "
            "ou par le mot CHAT si c'est une discussion normale. Un seul mot, rien d'autre.\n\n"
            f"Historique récent :\n{history_block}\n\n"
            f"Message : \"{user_text}\""
        )
        verdict = await ask_ollama(routing_prompt, timeout=15.0)
        # Si le LLM ne répond pas / répond de façon inexploitable, on retombe sur CHAT :
        # une fausse analyse (faux positif spam) est plus gênante pour l'utilisateur
        # qu'une réponse de chat un peu trop générique.
        decision = "ANALYSE" if verdict and "ANALYSE" in verdict.upper() else "CHAT"

    context.user_data["history"].append(("user", user_text))

    if decision == "CHAT":
        chat_prompt = (
            "Tu es un assistant francophone amical et concis, intégré à un bot "
            "anti-fraude. Réponds en 1 à 3 phrases maximum.\n\n"
            f"Message de l'utilisateur : \"{user_text}\""
        )
        reponse = await ask_ollama(chat_prompt) or (
            "Je suis là ! Envoie-moi un message suspect si tu veux que je l'analyse."
        )
        context.user_data["history"].append(("assistant", reponse))
        await update.message.reply_text(reponse)
        return

    # --- Branche analyse ---
    status_msg = await update.message.reply_text("🔍 Analyse en cours...")

    # 1. Cache de corrections : un texte déjà signalé comme mal classé n'est
    #    pas re-signalé de la même façon (filet de sécurité, pas un ré-entraînement).
    corrections = load_corrections(config.FEEDBACK_LOG_PATH)
    override_label = corrections.get(user_text.strip())

    # Le modèle SVM a été entraîné sur un dataset anglais : un texte dans une
    # autre langue est traduit avant classification, sinon le score est peu fiable
    # (signal quasi nul -> confiance proche de 50%, cf. l'exemple du mail en français).
    lang = detect_language(user_text)
    text_for_model = user_text
    translation_note = ""
    if lang not in ("en", "unknown"):
        translated = await ask_ollama(TRANSLATE_PROMPT_TEMPLATE.format(text=user_text), timeout=25.0)
        if translated:
            text_for_model = translated
            translation_note = f"\n\n_🌐 Traduit automatiquement ({lang.upper()}→EN) avant analyse._"
        else:
            translation_note = "\n\n_⚠️ Traduction indisponible, analyse effectuée sur le texte original (moins fiable)._"

    is_spam, confidence, score = predict_with_confidence(text_for_model, model_svm, vectorizer)
    top_words = explain_prediction(text_for_model, model_svm, vectorizer, top_n=5)
    uncertain = is_borderline(confidence)

    if override_label is not None:
        is_spam = override_label == "SPAM"
        note = "\n\n_ℹ️ Corrigé suite à un feedback précédent sur ce texte exact._"
    elif uncertain:
        # Score trop proche de 50% pour être fiable : on demande un second avis au LLM
        # plutôt que d'afficher un verdict SVM peu fiable avec une fausse assurance.
        verify_prompt = (
            "Ce message est-il une tentative de phishing/arnaque, ou un message "
            "légitime (ex: email professionnel, mail RH, conversation normale) ? "
            "Réponds uniquement par SPAM ou HAM.\n\n"
            f"Message : \"{user_text}\""
        )
        llm_verdict = await ask_ollama(verify_prompt, timeout=20.0)
        if llm_verdict:
            is_spam = "SPAM" in llm_verdict.upper() and "HAM" not in llm_verdict.upper()
        note = "\n\n_⚠️ Score du modèle peu tranché (proche de 50%) — vérifié par IA sémantique._" + translation_note
    else:
        note = translation_note

    context.user_data["stats"]["analysed"] += 1
    if is_spam:
        context.user_data["stats"]["spam"] += 1

    indices = "\n".join(
        f"  • `{w}` ({'+' if c > 0 else ''}{c:.2f})" for w, c in top_words
    ) or "  • (aucun indice lexical significatif isolé)"

    if is_spam:
        reponse = (
            f"🚨 *VERDICT : PHISHING / SPAM*\n"
            f"Confiance modèle : *{confidence}%*\n\n"
            f"Indices les plus déterminants :\n{indices}\n\n"
            f"➡️ Ne cliquez sur aucun lien, ne fournissez aucune donnée.{note}"
        )
    else:
        reponse = (
            f"✅ *VERDICT : MESSAGE SAIN (HAM)*\n"
            f"Confiance modèle : *{confidence}%*\n\n"
            f"Indices les plus déterminants :\n{indices}{note}"
        )

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("👍 Correct", callback_data=f"fb:1:{int(is_spam)}"),
        InlineKeyboardButton("👎 Erreur", callback_data=f"fb:0:{int(is_spam)}"),
    ]])
    context.user_data["last_analysed_text"] = user_text
    await status_msg.edit_text(reponse, parse_mode="Markdown", reply_markup=keyboard)


async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _, correct_flag, predicted_spam = query.data.split(":")
    text = context.user_data.get("last_analysed_text", "")
    label = "SPAM" if predicted_spam == "1" else "HAM"
    log_feedback(update.effective_user.id, text, label, correct_flag == "1")
    suffix = "\n\n_Merci, feedback enregistré ✅_" if correct_flag == "1" else "\n\n_Merci, erreur signalée 📝_"
    try:
        await query.edit_message_text(query.message.text_markdown + suffix, parse_mode="Markdown")
    except Exception:
        pass


def main():
    if not config.TELEGRAM_TOKEN:
        raise RuntimeError(
            "TELEGRAM_TOKEN manquant. Crée un fichier .env (voir .env.example) "
            "et NE COMMIT JAMAIS ce fichier (vérifie qu'il est dans .gitignore)."
        )
    application = Application.builder().token(config.TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("stats", stats_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, orchestrateur_intelligent))
    application.add_handler(CallbackQueryHandler(feedback_callback, pattern=r"^fb:"))

    logger.info("⚡ Sentinel-Spam Bot v2 en ligne (heuristique + Ollama async)")
    application.run_polling()


if __name__ == "__main__":
    main()