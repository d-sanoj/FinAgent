"""
WhatsApp Bot – Flask Webhook Server
------------------------------------
Receives incoming WhatsApp messages via Meta Cloud API webhooks,
passes them to the Financial Chat Engine, and sends replies back.
"""

import os
import json
import logging
import requests
from flask import Flask, request, jsonify

from chat_engine import FinancialChatEngine
from data_loader import FinancialDataLoader

logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Globals (initialised in create_app / run.py) ─────────────────────
engine: FinancialChatEngine | None = None

# Track processed message IDs to deduplicate (Meta sends retries)
_processed_ids: set[str] = set()
_MAX_PROCESSED = 1000  # rolling window


def init_engine(chat_engine: FinancialChatEngine) -> None:
    """Inject the chat engine after startup."""
    global engine
    engine = chat_engine


# ── WhatsApp API helpers ─────────────────────────────────────────────

def send_whatsapp_message(to: str, body: str) -> bool:
    """Send a text message via WhatsApp Cloud API."""
    token = os.environ.get("WHATSAPP_TOKEN")
    phone_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")

    if not token or not phone_id:
        logger.error("Missing WHATSAPP_TOKEN or WHATSAPP_PHONE_NUMBER_ID")
        return False

    url = f"https://graph.facebook.com/v21.0/{phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # WhatsApp has a 4096 char limit per message
    # Split long messages if needed
    chunks = _split_message(body, max_len=4000)

    for chunk in chunks:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": chunk},
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            logger.info("Message sent to %s (status %d)", to, resp.status_code)
        except requests.RequestException as e:
            logger.error("Failed to send message to %s: %s", to, e)
            return False
    return True


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split a message into chunks that fit WhatsApp's character limit."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try to split at a newline
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos == -1:
            split_pos = max_len
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


# ── Webhook routes ───────────────────────────────────────────────────

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """
    Meta sends a GET request to verify the webhook URL.
    We respond with the hub.challenge if the token matches.
    """
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN", "finagent-verify")

    if mode == "subscribe" and token == verify_token:
        logger.info("Webhook verified successfully")
        return challenge, 200
    else:
        logger.warning("Webhook verification failed — token mismatch")
        return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def receive_message():
    """
    Receive and process incoming WhatsApp messages.
    Meta sends a JSON payload with the message details.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "no data"}), 400

    logger.debug("Webhook payload: %s", json.dumps(data, indent=2))

    try:
        # Navigate the nested Meta webhook structure
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})

                # Only process actual messages (not status updates)
                if "messages" not in value:
                    continue

                for message in value["messages"]:
                    msg_id = message.get("id", "")

                    # Deduplicate
                    if msg_id in _processed_ids:
                        logger.debug("Skipping duplicate message %s", msg_id)
                        continue
                    _processed_ids.add(msg_id)
                    # Keep the set bounded
                    if len(_processed_ids) > _MAX_PROCESSED:
                        _processed_ids.clear()

                    sender = message.get("from", "")
                    msg_type = message.get("type", "")

                    if msg_type == "text":
                        body = message["text"]["body"]
                        logger.info("Message from %s: %s", sender, body[:100])
                        _handle_text_message(sender, body)
                    else:
                        send_whatsapp_message(
                            sender,
                            "I can only handle text messages right now. "
                            "Please type your financial question! 💬",
                        )

    except Exception as e:
        logger.exception("Error processing webhook: %s", e)

    # Always return 200 to Meta (they retry on non-200)
    return jsonify({"status": "ok"}), 200


def _handle_text_message(sender: str, body: str) -> None:
    """Process a text message and send a reply."""
    if engine is None:
        send_whatsapp_message(sender, "⚠️ Bot is still starting up. Try again shortly!")
        return

    # Special commands
    lower = body.strip().lower()
    if lower == "/reload":
        result = engine.reload_data()
        send_whatsapp_message(sender, result)
        return

    if lower == "/help":
        help_text = (
            "💰 *FinAgent — Your Financial Assistant*\n\n"
            "Just ask me anything about your finances! Examples:\n\n"
            "• How much did I spend this month?\n"
            "• What's my bank balance?\n"
            "• Show my top spending categories\n"
            "• How much did I spend on food in February?\n"
            "• What are my recent transactions?\n\n"
            "Commands:\n"
            "• /reload — Refresh data from latest sync\n"
            "• /summary — Quick overview of your data\n"
            "• /help — Show this message"
        )
        send_whatsapp_message(sender, help_text)
        return

    if lower == "/summary":
        loader = engine._loader
        summary = loader.get_summary()
        send_whatsapp_message(sender, f"📊 *Data Summary*\n\n{summary}")
        return

    # Regular question → AI
    answer = engine.ask(body, user_id=sender)
    send_whatsapp_message(sender, answer)


# ── Health check ─────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify({
        "status": "ok",
        "engine_ready": engine is not None,
    })


# ── Direct run (for testing) ────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    loader = FinancialDataLoader()
    init_engine(FinancialChatEngine(data_loader=loader))

    print("\n🤖 FinAgent WhatsApp Bot running on http://localhost:5000")
    print("   Webhook URL: http://localhost:5000/webhook\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
