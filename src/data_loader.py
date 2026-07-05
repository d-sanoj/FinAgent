"""
Financial Data Loader
---------------------
Loads and merges parquet files from the gold/ directory into a single
Pandas DataFrame for querying by the AI chat engine.
"""

import os
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "gold")
STATIC_PATH = os.path.join(DATA_DIR, "final_static_data")
UPDATES_PATH = os.path.join(DATA_DIR, "simplefin_updates.parquet")


class FinancialDataLoader:
    """Loads, merges, and normalises all financial transaction data."""

    def __init__(self):
        self._df: pd.DataFrame | None = None
        self.reload()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_df(self) -> pd.DataFrame:
        """Return the master DataFrame (read-only copy)."""
        if self._df is None:
            self.reload()
        return self._df.copy()

    def reload(self) -> None:
        """Re-read parquet files from disk and rebuild the DataFrame."""
        frames: list[pd.DataFrame] = []

        # 1. Static historical data (partitioned parquet directory)
        if os.path.exists(STATIC_PATH):
            logger.info("Loading static data from %s", STATIC_PATH)
            df_static = pd.read_parquet(STATIC_PATH)
            frames.append(df_static)
        else:
            logger.warning("Static data path not found: %s", STATIC_PATH)

        # 2. Incremental updates
        if os.path.exists(UPDATES_PATH):
            logger.info("Loading updates from %s", UPDATES_PATH)
            df_updates = pd.read_parquet(UPDATES_PATH)
            frames.append(df_updates)

        if not frames:
            logger.error("No data files found – DataFrame will be empty.")
            self._df = pd.DataFrame()
            return

        df = pd.concat(frames, ignore_index=True)

        # Normalise types
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["balance"] = pd.to_numeric(df["balance"], errors="coerce")

        # Ensure text columns are strings
        for col in ("description", "category", "type", "account"):
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Sort by date descending (newest first)
        df.sort_values("date", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Drop exact duplicates
        df.drop_duplicates(
            subset=["date", "description", "amount", "account"],
            keep="first",
            inplace=True,
        )

        self._df = df
        logger.info(
            "Loaded %d transactions  |  %s → %s",
            len(df),
            df["date"].min().date() if not df.empty else "N/A",
            df["date"].max().date() if not df.empty else "N/A",
        )

    # ------------------------------------------------------------------
    # Quick helpers
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        df = self.get_df()
        if df.empty:
            return "No financial data loaded."

        lines = [
            f"Total transactions: {len(df):,}",
            f"Date range: {df['date'].min().date()} → {df['date'].max().date()}",
            f"Accounts: {', '.join(sorted(df['account'].unique()))}",
            f"Categories: {', '.join(sorted(df['category'].unique()))}",
        ]
        return "\n".join(lines)

    def get_schema_description(self) -> str:
        """Return column info for LLM prompts."""
        df = self.get_df()
        parts = ["DataFrame `df` columns:"]
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = df[col].dropna().head(3).tolist()
            parts.append(f"  - {col} ({dtype}): e.g. {sample}")
        return "\n".join(parts)
