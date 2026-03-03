#!/bin/bash
# FinAgent — One-Click Setup & Run
# Usage: ./start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "💰 FinAgent — Telegram Financial Bot"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install it first: https://python.org"
    exit 1
fi

# 2. Create virtual environment if missing
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# 3. Activate venv
source venv/bin/activate

# 4. Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# 5. Check .env
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Copy .env.example and fill in your tokens."
    exit 1
fi

# 6. Check Telegram token
if grep -q "your-telegram-bot-token-here" .env 2>/dev/null; then
    echo "⚠️  Set your TELEGRAM_BOT_TOKEN in .env first!"
    echo "   Get one from @BotFather on Telegram"
    exit 1
fi

# 7. Check Ollama
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "⚠️  Ollama not running. Starting it..."
    if command -v ollama &> /dev/null; then
        ollama serve &
        sleep 3
    else
        echo "❌ Ollama not installed. Get it: https://ollama.ai"
        exit 1
    fi
fi

# 8. Ensure model is pulled
MODEL=$(grep OLLAMA_MODEL .env 2>/dev/null | cut -d= -f2 || echo "llama3.2")
MODEL=${MODEL:-llama3.2}
echo "🧠 Checking model: $MODEL"
ollama pull "$MODEL" 2>/dev/null || true

# 9. Run the bot
echo ""
echo "🚀 Starting FinAgent Telegram Bot..."
echo "   Press Ctrl+C to stop"
echo ""
python telegram_bot.py
