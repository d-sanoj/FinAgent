#!/usr/bin/env python3
"""
FinAgent — Entry Point
----------------------
Starts the WhatsApp Financial AI Agent:
1. Loads environment variables
2. Initialises the financial data loader
3. Creates the AI chat engine
4. Starts the Flask webhook server 
"""

import os
import logging
from dotenv import load_dotenv

# Load env FIRST so all modules pick up the variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("finagent")


def main():
    from data_loader import FinancialDataLoader
    from chat_engine import FinancialChatEngine
    from whatsapp_bot import app, init_engine

    # 1. Load financial data
    logger.info("Loading financial data...")
    loader = FinancialDataLoader()
    logger.info("Data summary:\n%s", loader.get_summary())

    # 2. Create chat engine
    logger.info("Initialising chat engine (model: %s)...", os.environ.get("OLLAMA_MODEL", "llama3.2"))
    chat_engine = FinancialChatEngine(data_loader=loader)
    init_engine(chat_engine)

    # 3. Start server
    port = int(os.environ.get("PORT", 5000))
    logger.info("=" * 60)
    logger.info("🤖  FinAgent WhatsApp Bot")
    logger.info("   Webhook URL:  http://localhost:%d/webhook", port)
    logger.info("   Health check: http://localhost:%d/health", port)
    logger.info("=" * 60)

    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
