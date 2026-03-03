# FinAgent — Your Personal Finance AI on Telegram

FinAgent is a Telegram bot that connects to your bank accounts through [SimpleFin](https://www.simplefin.org/) and lets you ask questions about your money in plain English. It supports two LLM backends — run it **fully locally** with [Ollama](https://ollama.ai/) for complete privacy, or use [Groq's free API](https://console.groq.com/) for faster responses and a more powerful model.

I built this because I wanted a quick way to check my spending without logging into five different bank apps. Now I just open Telegram and ask things like "how much did I spend on Amazon this year?" and get an answer in seconds.

**Here are some examples for the chat**

<br>

<table align="center">
<tr>
<td width="45%">
<img src="https://github.com/user-attachments/assets/be04fe00-4ea0-4de4-8f79-1a6eef770775" width="100%">
</td>
<td width="55%">
<img src="https://github.com/user-attachments/assets/35fddc22-0bfe-4b89-9060-8383874be8c1" width="100%">
<br><br>
<img src="https://github.com/user-attachments/assets/812dc429-32a4-408f-8ec5-b27325d6c368" width="100%">
</td>
</tr>
</table>

<br>


## What It Does

**Ask questions in plain English, get real answers from your actual bank data.**

- "How much did I spend on food last month?" → Scans your transactions, adds up all Food & Dining debits
- "Show me my Amazon spending in 2025" → Searches transaction descriptions for "Amazon" and totals them up
- "What's my bank balance?" → Pulls the most recent balance from your checking account
- "How much at Costco this year?" → Finds all Costco transactions across all your cards
- `/overview` → Shows a full financial dashboard: balance, monthly spending vs income, top categories, year-to-date savings

It works with both **category-based queries** (food, transportation, entertainment) and **merchant-specific searches** (Amazon, Netflix, Walmart). If you mention a known category, it filters by category. If you mention anything else, it searches your transaction descriptions. You don't have to think about it — just ask naturally.

## How It Works Under the Hood

The tricky part of building this was getting a small local LLM (llama3.2 — only 3B parameters) to reliably answer financial questions. Having the LLM generate Python code directly was way too error-prone at this model size. So I split the work into two phases:

1. **Phase 1 — The LLM extracts structured data.** When you ask a question, the LLM doesn't try to compute anything. It just figures out what you're asking for and outputs a simple JSON: `{ intent: "spending", search: "Amazon", date_start: "2026-01-01" }`. Small models are actually great at this kind of structured extraction.

2. **Phase 2 — Deterministic Python does the actual work.** Pre-written, tested Pandas code takes that JSON and runs the exact right query against your transaction data. No generated code, no hallucinated column names, no random errors. It just works.

There's also a bunch of post-processing that fixes common LLM mistakes — like when the model puts "Amazon" as a category instead of a search term, or gets the date wrong for "this year." The code catches these and corrects them automatically.

## Project Structure

```
FinAgent/
├── telegram_bot.py          # Telegram bot (main entry point)
├── chat_engine.py           # Two-phase AI query engine
├── data_loader.py           # Loads and merges transaction data from Parquet files
├── sync_financial_data.py   # Pulls new transactions from SimpleFin API
├── whatsapp_bot.py          # WhatsApp bot (optional alternative)
├── run.py                   # WhatsApp entry point
├── start.sh                 # One-click setup & run script
├── requirements.txt         # Python dependencies
├── .env                     # Your tokens and config (not committed)
└── gold/                    # Transaction data (Parquet files)
    ├── final_static_data/   # Historical transaction data
    └── simplefin_updates.parquet  # New transactions from sync
```

## Installation

### Prerequisites

- **Python 3.10+**
- **LLM backend** (pick one):
  - **Ollama** — Install from [ollama.ai](https://ollama.ai/) for fully local, private inference
  - **Groq API key** — Get a free key at [console.groq.com](https://console.groq.com/keys) for cloud inference (faster, no GPU needed)
- **A SimpleFin account** — For pulling bank transactions ([simplefin.org](https://www.simplefin.org/))
- **A Telegram account** — For the bot interface

### Quick Start (Recommended)

```bash
# Clone the repo
git clone <your-repo-url>
cd FinAgent

# Make the start script executable
chmod +x start.sh

# Run it — handles everything automatically
./start.sh
```

The script creates a virtual environment, installs dependencies, checks that Ollama is running, and starts the bot. That's it.

### Manual Setup

If you prefer to set things up step by step:

**1. Install Ollama and pull the model**

```bash
# Install Ollama (macOS)
brew install ollama

# Start the Ollama server
ollama serve

# Pull the model (in a separate terminal)
ollama pull llama3.2
```

**2. Set up Python environment**

```bash
cd FinAgent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Create a Telegram bot**

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the bot token it gives you (looks like `123456:ABC-DEF1234...`)

**4. Configure environment variables**

Copy the example and fill in your values:

```bash
cp .env.example .env   # or just edit .env directly
```

Your `.env` should have at minimum:

```env
# LLM — Option A: Local Ollama (default)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2

# LLM — Option B: Groq Cloud (uncomment to use instead of Ollama)
# If GROQ_API_KEY is set, the bot automatically switches to Groq.
# GROQ_API_KEY=gsk_your_key_here
# GROQ_MODEL=llama-3.3-70b-versatile

# Telegram
TELEGRAM_BOT_TOKEN=your-bot-token-from-botfather
TELEGRAM_ALLOWED_USERS=your-telegram-user-id

# SimpleFin (for bank data sync)
SIMPLEFIN_USERNAME=your-simplefin-username
SIMPLEFIN_PASSWORD=your-simplefin-password
```

> **Ollama vs Groq:** Ollama runs everything on your machine — fully private but requires a decent CPU/GPU. Groq is a free cloud API that runs a much larger model (70B vs 3B) so responses are more accurate and faster, but your queries leave your machine. Pick whichever you prefer — the bot auto-detects based on which key is set.

> **Finding your Telegram user ID:** Send a message to your bot, then check the terminal logs — it prints `Message from <YOUR_ID>: ...` You can add that ID to `TELEGRAM_ALLOWED_USERS` to lock it down.

**5. Sync your financial data**

```bash
python sync_financial_data.py
```

This pulls the last 90 days of transactions from SimpleFin and saves them as Parquet files in `gold/`.

**6. Start the bot**

```bash
python telegram_bot.py
```

Open Telegram, find your bot, and start chatting!

## Telegram Commands

| Command | What it does |
|---------|-------------|
| `/start` | Welcome message with usage examples |
| `/overview` | Full financial dashboard — balance, spending, income, savings |
| `/reload` | Syncs fresh data from SimpleFin and reloads |
| `/summary` | Quick data stats (transaction count, date range, accounts) |
| `/help` | Shows available commands |

Or just type any question naturally:

- "How much did I spend this month?"
- "Show me Netflix charges in the last 3 months"
- "What are my top spending categories this year?"
- "How much income did I get last month?"

## Privacy & Security

- **With Ollama:** Everything runs locally. Your financial data and queries never leave your machine.
- **With Groq:** Your transaction data stays local. Only the question text is sent to Groq's API for processing — no raw financial data is shared.
- **Bot is owner-locked.** Only your Telegram user ID can interact with the bot.
- **Credentials stay local.** All tokens and passwords live in `.env` which should be in `.gitignore`.

## Tech Stack

- **LLM:** Ollama (local, 3B model) or Groq (cloud, 70B model) — auto-detected from `.env`
- **Data:** Pandas + PyArrow for transaction analysis
- **Bot:** python-telegram-bot (polling mode — no server exposure needed)
- **Bank Data:** SimpleFin API for account aggregation
- **API Layer:** OpenAI-compatible client (works with both Ollama and Groq)

## Troubleshooting

**Bot says $0 for a merchant I know I've used**
- The LLM might be getting the dates wrong. Try being more specific: "Amazon spending from January to March 2026" instead of "Amazon spending this year"

**SimpleFin sync times out**
- SimpleFin's servers can be slow. Try again in a few minutes. The `/reload` command will retry automatically.

**Balance shows an old date**
- Balance data comes from your bank's feed. If your bank hasn't posted new transactions with balance data, the bot shows the most recent one available. Run `/reload` to check for updates.

**Ollama is slow**
- First response after startup is slower (model loading). Subsequent queries should be 1-3 seconds. If it's consistently slow, try a smaller model like `llama3.2:1b`.

## License

MIT
