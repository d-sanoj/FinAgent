"""
Financial Chat Engine
---------------------
AI-powered engine that answers natural-language questions about your
finances.  Uses a two-phase approach for reliability with small local LLMs:

Phase 1 — LLM extracts structured query parameters (intent, filters, dates)
Phase 2 — Deterministic Python code executes the actual Pandas query
"""

import os
import re
import json
import logging
import textwrap
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd

from data_loader import FinancialDataLoader

logger = logging.getLogger(__name__)

# ── Prompt for structured extraction ─────────────────────────────────

EXTRACTION_PROMPT = textwrap.dedent("""\
You are a financial query parser. Given a user message and available data categories,
extract the query intent as JSON.

Available categories: {categories}
Available accounts: {accounts}
Today's date: {today}

Respond with ONLY a JSON object (no markdown, no explanation). The JSON must have:
{{
  "intent": one of ["spending", "income", "balance", "transactions", "categories", "chitchat"],
  "category": category name or null,
  "search": keyword to search in transaction descriptions or null,
  "account": account name or null (use "bank" for checking/balance queries),
  "date_start": "YYYY-MM-DD" or null,
  "date_end": "YYYY-MM-DD" or null,
  "top_n": number or null (for "show top N" queries),
  "chitchat_response": string or null (friendly reply if intent is chitchat)
}}

Rules:
- "category" vs "search": Use "category" ONLY when the user's words match one of the
  available categories above (e.g. "food" → "Food & Dining", "gas" → "Transportation").
  For ANY other word like a merchant or store name (e.g. "Amazon", "Costco", "Netflix",
  "Walmart", "Starbucks", "AT&T"), set "search" to that keyword and leave "category" null.
- If no dates mentioned, default to current month: "{month_start}" to "{today}"
- "last month" = previous calendar month
- "this year" or "in 2026" = date_start "2026-01-01", date_end "{today}"
- "in 2025" or "last year" = date_start "2025-01-01", date_end "2025-12-31"
- "in YYYY" = date_start "YYYY-01-01", date_end "YYYY-12-31"
- For "balance" intent, always set account to "bank"
- For loose category mentions like "food" or "eating out", map to "Food & Dining"
- For "groceries" or "shopping", map to "Purchases and Refunds"
- For "gas" or "uber", map to "Transportation"
""")

FORMATTING_PROMPT = textwrap.dedent("""\
The user asked: "{question}"
Here is the computed result: {result}

Write a SHORT, friendly 1-2 sentence response summarising the answer.
Use $ for dollar amounts with 2 decimal places. Do NOT include any code or JSON.
""")


class FinancialChatEngine:
    """Answers financial questions using structured extraction + deterministic execution."""

    def __init__(self, data_loader: FinancialDataLoader | None = None):
        self._loader = data_loader or FinancialDataLoader()
        # Use Ollama's OpenAI-compatible API running locally
        self._client = OpenAI(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # Ollama doesn't need a real key
        )
        self._model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        # Per-user message history  (phone → list of messages)
        self._history: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_overview(self) -> str:
        """Generate a financial overview dashboard."""
        df = self._loader.get_df()
        if df.empty:
            return "No financial data available. Please sync your data first."

        now = datetime.now()
        this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month_end = this_month_start - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)

        this_month = df[df["date"] >= pd.Timestamp(this_month_start)]
        last_month = df[
            (df["date"] >= pd.Timestamp(last_month_start))
            & (df["date"] < pd.Timestamp(this_month_start))
        ]

        # Bank balance
        bank = df[(df["account"] == "bank") & df["balance"].notna()].sort_values(
            "date", ascending=False
        )
        bank_str = (
            f"${bank.iloc[0]['balance']:,.2f} (as of {bank.iloc[0]['date'].strftime('%b %d')})"
            if not bank.empty
            else "N/A"
        )

        # This month
        tm_spending = this_month[this_month["type"] == "debit"]["amount"].abs().sum()
        tm_income = this_month[this_month["type"] == "credit"]["amount"].sum()
        tm_net = tm_income - tm_spending

        # Last month
        lm_spending = last_month[last_month["type"] == "debit"]["amount"].abs().sum()
        lm_income = last_month[last_month["type"] == "credit"]["amount"].sum()
        lm_net = lm_income - lm_spending

        # Month-over-month spending change
        if lm_spending > 0:
            spending_change = ((tm_spending - lm_spending) / lm_spending) * 100
            change_emoji = "📈" if spending_change > 0 else "📉"
            change_str = f"{change_emoji} {spending_change:+.1f}% vs last month"
        else:
            change_str = ""

        # Top categories this month
        tm_debits = this_month[this_month["type"] == "debit"]
        top_cats = (
            tm_debits.groupby("category")["amount"]
            .apply(lambda x: x.abs().sum())
            .sort_values(ascending=False)
            .head(5)
        )
        cats_str = "\n".join(
            f"  {'🔴' if i == 0 else '🟡' if i == 1 else '⚪'} {cat}: ${amt:,.2f}"
            for i, (cat, amt) in enumerate(top_cats.items())
        )

        # Year-to-date
        ytd = df[df["date"] >= pd.Timestamp(f"{now.year}-01-01")]
        ytd_spending = ytd[ytd["type"] == "debit"]["amount"].abs().sum()
        ytd_income = ytd[ytd["type"] == "credit"]["amount"].sum()
        ytd_saved = ytd_income - ytd_spending

        month_name = now.strftime("%B")
        last_month_name = last_month_start.strftime("%B")

        return (
            f"💰 Financial Overview\n"
            f"{'━' * 28}\n"
            f"\n"
            f"🏦 Bank Balance: {bank_str}\n"
            f"\n"
            f"📅 {month_name} (so far)\n"
            f"  💸 Spent: ${tm_spending:,.2f}\n"
            f"  💵 Income: ${tm_income:,.2f}\n"
            f"  {'✅' if tm_net >= 0 else '⚠️'} Net: ${tm_net:,.2f}\n"
            f"  {change_str}\n"
            f"\n"
            f"📅 {last_month_name}\n"
            f"  💸 Spent: ${lm_spending:,.2f}\n"
            f"  💵 Income: ${lm_income:,.2f}\n"
            f"  {'✅' if lm_net >= 0 else '⚠️'} Net: ${lm_net:,.2f}\n"
            f"\n"
            f"📊 Top Spending ({month_name})\n"
            f"{cats_str if cats_str else '  No spending yet'}\n"
            f"\n"
            f"📈 Year-to-Date ({now.year})\n"
            f"  Total Spent: ${ytd_spending:,.2f}\n"
            f"  Total Income: ${ytd_income:,.2f}\n"
            f"  💰 Saved: ${ytd_saved:,.2f}\n"
        )

    def ask(self, question: str, user_id: str = "default") -> str:
        """Answer a user question.  Returns a plain-English response."""
        try:
            df = self._loader.get_df()
            if df.empty:
                return "I don't have any financial data loaded yet. Please sync your data first."

            # Phase 1: Extract structured query intent via LLM
            params = self._extract_intent(question, df)
            logger.info("Extracted params (raw): %s", params)

            if params is None:
                return "Sorry, I couldn't understand that question. Try asking about spending, income, or balances!"

            # Phase 1.5: Fix category vs search — if LLM put a non-existent
            # category (like "Amazon"), move it to search instead
            real_categories = set(df["category"].unique())
            cat = params.get("category")
            if cat and cat not in real_categories:
                logger.info("'%s' is not a real category — treating as description search", cat)
                params["search"] = cat
                params["category"] = None

            # Phase 1.6: Deterministic Date Correction
            # Small models often default to 'today' even for 'this year' queries
            q_lower = question.lower()
            today = datetime.now()
            
            if "this year" in q_lower or "in 2026" in q_lower:
                params["date_start"] = "2026-01-01"
                params["date_end"] = today.strftime("%Y-%m-%d")
            elif "last year" in q_lower or "in 2025" in q_lower:
                params["date_start"] = "2025-01-01"
                params["date_end"] = "2025-12-31"
            elif "last 3 months" in q_lower:
                three_months_ago = today - timedelta(days=90)
                params["date_start"] = three_months_ago.strftime("%Y-%m-%d")
                params["date_end"] = today.strftime("%Y-%m-%d")

            logger.info("Extracted params (fixed): %s", params)

            # Handle chitchat
            if params.get("intent") == "chitchat":
                response = params.get("chitchat_response", "Hey! Ask me about your finances 💰")
                self._update_history(user_id, question, response)
                return response

            # Phase 2: Execute deterministic query
            result = self._execute_query(df, params)
            logger.info("Query result: %s", str(result)[:300])

            # Phase 3: Format result into friendly message
            answer = self._format_result(question, result)
            self._update_history(user_id, question, answer)
            return answer

        except Exception as e:
            logger.exception("Error processing question: %s", question)
            return f"Sorry, I ran into an issue: {e}"

    def reload_data(self) -> str:
        """Reload financial data from disk."""
        self._loader.reload()
        return "✅ Financial data reloaded successfully!\n" + self._loader.get_summary()

    # ------------------------------------------------------------------
    # Phase 1: Intent Extraction
    # ------------------------------------------------------------------

    def _extract_intent(self, question: str, df: pd.DataFrame) -> dict | None:
        """Use the LLM to extract structured query parameters from the question."""
        now = datetime.now()
        month_start = now.replace(day=1).strftime("%Y-%m-%d")

        prompt = EXTRACTION_PROMPT.format(
            categories=", ".join(sorted(df["category"].unique())),
            accounts=", ".join(sorted(df["account"].unique())),
            today=now.strftime("%Y-%m-%d"),
            month_start=month_start,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]

        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,
                )
                reply = response.choices[0].message.content.strip()
                logger.info("Extraction reply (attempt %d): %s", attempt + 1, reply[:300])

                # Parse JSON from response (handle markdown-wrapped JSON)
                json_str = self._extract_json(reply)
                if json_str:
                    params = json.loads(json_str)
                    # Validate required field
                    if "intent" in params:
                        return params

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, e)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "Please respond with ONLY valid JSON, no other text."})

        return None

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract JSON from text that may contain markdown or other wrapping."""
        # Try to find JSON in code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to find a JSON object directly
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0).strip()

        return None

    # ------------------------------------------------------------------
    # Phase 2: Deterministic Query Execution
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_query(df: pd.DataFrame, params: dict) -> str:
        """Execute a structured financial query against the DataFrame."""
        intent = params.get("intent", "spending")
        category = params.get("category")
        account = params.get("account")
        date_start = params.get("date_start")
        date_end = params.get("date_end")
        top_n = params.get("top_n", 5)

        # Apply date filters
        filtered = df.copy()
        if date_start:
            filtered = filtered[filtered["date"] >= pd.Timestamp(date_start)]
        if date_end:
            filtered = filtered[filtered["date"] <= pd.Timestamp(date_end)]

        # Apply account filter
        if account:
            filtered = filtered[filtered["account"] == account]

        # Apply category filter (skip for 'categories' intent — we want all)
        if category and intent != "categories":
            filtered = filtered[filtered["category"] == category]

        # Apply description search (for merchants like Amazon, Costco, etc.)
        search = params.get("search")
        if search:
            filtered = filtered[
                filtered["description"].str.contains(search, case=False, na=False)
            ]

        # Execute based on intent
        if intent == "spending":
            debits = filtered[filtered["type"] == "debit"]
            total = debits["amount"].abs().sum()
            count = len(debits)

            if search:
                breakdown = f"matching '{search}'"
            elif category:
                breakdown = f"on {category}"
            else:
                breakdown = "total"

            date_range = ""
            if date_start and date_end:
                date_range = f" from {date_start} to {date_end}"
            elif date_start:
                date_range = f" since {date_start}"

            # Top categories breakdown
            top_cats = (
                debits.groupby("category")["amount"]
                .apply(lambda x: x.abs().sum())
                .sort_values(ascending=False)
                .head(5)
            )
            breakdown_str = "\n".join(
                f"  • {cat}: ${amt:.2f}" for cat, amt in top_cats.items()
            )

            return (
                f"Spending {breakdown}{date_range}: ${total:.2f} "
                f"across {count} transactions.\n"
                f"Top categories:\n{breakdown_str}"
            )

        elif intent == "income":
            credits = filtered[filtered["type"] == "credit"]
            total = credits["amount"].sum()
            count = len(credits)

            date_range = ""
            if date_start and date_end:
                date_range = f" from {date_start} to {date_end}"

            return f"Total income{date_range}: ${total:.2f} across {count} transactions."

        elif intent == "balance":
            bank = df[
                (df["account"] == "bank") & (df["balance"].notna())
            ].sort_values("date", ascending=False)

            if bank.empty:
                return "No balance data available for bank account."

            latest = bank.iloc[0]
            return (
                f"Bank balance: ${latest['balance']:.2f} "
                f"(as of {latest['date'].strftime('%Y-%m-%d')})"
            )

        elif intent == "transactions":
            n = min(top_n or 10, 20)
            recent = filtered.head(n)
            if recent.empty:
                return "No transactions found matching your criteria."

            lines = []
            for _, row in recent.iterrows():
                lines.append(
                    f"  {row['date'].strftime('%m/%d')} | "
                    f"{'−' if row['type'] == 'debit' else '+'}"
                    f"${abs(row['amount']):.2f} | "
                    f"{row['description'][:30]} | {row['category']}"
                )
            return f"Recent transactions ({len(recent)}):\n" + "\n".join(lines)

        elif intent == "categories":
            n = top_n or 5
            debits = filtered[filtered["type"] == "debit"]
            top_cats = (
                debits.groupby("category")["amount"]
                .apply(lambda x: x.abs().sum())
                .sort_values(ascending=False)
                .head(n)
            )
            lines = [f"  {i+1}. {cat}: ${amt:.2f}" for i, (cat, amt) in enumerate(top_cats.items())]
            return f"Top {n} spending categories:\n" + "\n".join(lines)

        return "I'm not sure how to answer that. Try asking about spending, income, or balances!"

    # ------------------------------------------------------------------
    # Phase 3: Natural Language Formatting
    # ------------------------------------------------------------------

    def _format_result(self, question: str, result: str) -> str:
        """Ask the LLM to turn the raw result into a friendly sentence."""
        prompt = FORMATTING_PROMPT.format(question=question, result=result)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            answer = response.choices[0].message.content.strip()
            # If the LLM just echoed back the data, return it as-is
            if len(answer) < 10:
                return result
            return answer
        except Exception:
            # Fallback: return raw result
            return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_history(self, user_id: str, question: str, answer: str) -> None:
        """Track conversation history per user."""
        history = self._history.setdefault(user_id, [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > 10:
            history[:] = history[-10:]


# ── Quick CLI test ────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    engine = FinancialChatEngine()

    print("Financial Chat Engine — type 'quit' to exit\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        answer = engine.ask(q)
        print(f"\n🤖 {answer}\n")
