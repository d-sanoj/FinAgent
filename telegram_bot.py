"""
Telegram Bot – FinAgent
-----------------------
Receives messages via Telegram Bot API, passes them to the
Financial Chat Engine, and sends replies back.

Setup:
  1. Message @BotFather on Telegram → /newbot → get your token
  2. Add TELEGRAM_BOT_TOKEN to .env
  3. Run: python telegram_bot.py
"""

import os
import logging
import pandas as pd
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from data_loader import FinancialDataLoader
from chat_engine import FinancialChatEngine

logger = logging.getLogger(__name__)

# ── Globals ──────────────────────────────────────────────────────────
engine: FinancialChatEngine | None = None


def init_engine(chat_engine: FinancialChatEngine) -> None:
    """Inject the chat engine."""
    global engine
    engine = chat_engine


# ── Command handlers ─────────────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command — mini overview + command list."""
    overview_text = ""
    if engine is not None:
        try:
            df = engine._loader.get_df()
            if not df.empty:
                from datetime import datetime as dt
                now = dt.now()
                # Bank balance
                bank = df[(df["account"] == "bank") & df["balance"].notna()].sort_values("date", ascending=False)
                bal = f"${bank.iloc[0]['balance']:,.2f}" if not bank.empty else "N/A"
                # This month spending
                month_start = now.replace(day=1)
                tm = df[(df["date"] >= pd.Timestamp(month_start)) & (df["type"] == "debit")]
                spent = f"${tm['amount'].abs().sum():,.2f}"
                overview_text = (
                    f"\n\n"
                    f"🏦 Bank Balance: {bal}\n"
                    f"💸 Spent this month: {spent}\n"
                )
        except Exception:
            pass

    await update.message.reply_text(
        f"💰 FinAgent — Your Financial Assistant\n"
        f"{'━' * 28}"
        f"{overview_text}\n"
        f"💬 Just type any question about your finances!\n\n"
        f"Examples:\n"
        f"  • How much did I spend on food last month?\n"
        f"  • What's my bank balance?\n"
        f"  • Show me Amazon transactions this year\n\n"
        f"Commands:\n"
        f"  /overview — Full financial dashboard\n"
        f"  /reload — Sync fresh data from SimpleFin\n"
        f"  /help — Show this message"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await start_command(update, context)


async def reload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reload command — sync from SimpleFin + reload data."""
    if engine is None:
        await update.message.reply_text("⚠️ Bot is still starting up.")
        return

    await update.message.reply_text("🔄 Syncing from SimpleFin...")

    # Step 1: Run sync to fetch new transactions
    try:
        import subprocess
        result = subprocess.run(
            ["python", "sync_financial_data.py"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning("Sync stderr: %s", result.stderr[-500:] if result.stderr else "")
            await update.message.reply_text(
                f"⚠️ Sync finished with warnings.\n{result.stderr[-200:] if result.stderr else ''}"
            )
    except subprocess.TimeoutExpired:
        await update.message.reply_text("⚠️ Sync timed out after 60s. Try again later.")
        return
    except Exception as e:
        await update.message.reply_text(f"❌ Sync failed: {e}")
        return

    # Step 2: Reload DataFrame from updated parquet files
    reload_result = engine.reload_data()
    await update.message.reply_text(f"✅ {reload_result}")




async def overview_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /overview command — financial dashboard."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("🔒 Sorry, this bot is private.")
        return
    if engine is None:
        await update.message.reply_text("⚠️ Bot is still starting up.")
        return
    await update.message.chat.send_action("typing")
    overview = engine.get_overview()
    await update.message.reply_text(overview)


# ── Message handler ──────────────────────────────────────────────────

def _is_allowed(user_id: int) -> bool:
    """Check if a Telegram user is in the allowed list."""
    allowed = os.environ.get("TELEGRAM_ALLOWED_USERS", "")
    if not allowed:
        return True  # No restriction if not configured
    allowed_ids = {int(uid.strip()) for uid in allowed.split(",") if uid.strip()}
    return user_id in allowed_ids


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any text message — pass to the AI chat engine."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("🔒 Sorry, this bot is private.")
        return

    if engine is None:
        await update.message.reply_text("⚠️ Bot is still starting up. Try again shortly!")
        return

    user_id = str(update.effective_user.id)
    text = update.message.text

    logger.info("Message from %s: %s", user_id, text[:100])

    # Show "typing..." indicator while processing
    await update.message.chat.send_action("typing")

    # Get AI response
    answer = engine.ask(text, user_id=user_id)

    # Telegram has a 4096 char limit per message
    if len(answer) <= 4096:
        await update.message.reply_text(answer)
    else:
        # Split long messages
        for i in range(0, len(answer), 4096):
            await update.message.reply_text(answer[i : i + 4096])


# ── Main ─────────────────────────────────────────────────────────────

def main():
    """Start the Telegram bot."""
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not found in .env — get one from @BotFather on Telegram")
        return

    # 1. Load data & create engine
    logger.info("Loading financial data...")
    loader = FinancialDataLoader()
    logger.info("Data summary:\n%s", loader.get_summary())

    logger.info("Initialising chat engine (model: %s)...", os.environ.get("OLLAMA_MODEL", "llama3.2"))
    chat_engine = FinancialChatEngine(data_loader=loader)
    init_engine(chat_engine)

    # 2. Build Telegram app
    app = Application.builder().token(token).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reload", reload_command))
    app.add_handler(CommandHandler("overview", overview_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 3. Start polling (no webhook/ngrok needed!)
    logger.info("=" * 60)
    logger.info("🤖  FinAgent Telegram Bot is running!")
    logger.info("   Send a message to your bot on Telegram.")
    logger.info("=" * 60)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
