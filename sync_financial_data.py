"""
Financial Data Sync Service
--------------------------
Fetches new transactions from SimpleFin, categorizes them according to custom rules,
merges them with the historical dataset, and updates the master parquet file.

Usage:
    python3 sync_financial_data.py
"""

import os
import json
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
DATA_PATH = "gold/final_static_data"
SIMPLEFIN_URL = "https://beta-bridge.simplefin.org/simplefin"
# Credentials
SIMPLEFIN_USERNAME = os.environ.get("SIMPLEFIN_USERNAME")
SIMPLEFIN_PASSWORD = os.environ.get("SIMPLEFIN_PASSWORD")

if not SIMPLEFIN_USERNAME or not SIMPLEFIN_PASSWORD:
    logger.warning("SimpleFin credentials not found in environment variables.")

ACCOUNT_MAPPING = {
    "Chase Bank": "chase",
    "Capital One": "capital",
    "American Express": "amex",
    "Discover Credit Card": "discover",
    "MidFirst Bank": "bank",
}

def fetch_simplefin_data(days_back: int = 30) -> Dict:
    """Fetch recent transaction data from SimpleFin API."""
    start_date = int((datetime.now() - timedelta(days=days_back)).timestamp())
    url = f"{SIMPLEFIN_URL}/accounts?start-date={start_date}"
    
    logger.info(f"Fetching data from SimpleFin (last {days_back} days)...")
    try:
        response = requests.get(url, auth=(SIMPLEFIN_USERNAME, SIMPLEFIN_PASSWORD), timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data from SimpleFin: {e}")
        raise

def categorize_transaction(account: str, description: str, payee: str) -> str:
    """Determine category based on account type and transaction details."""
    text = (description + " " + payee).lower()
    
    # 1. Bank Account Logic
    if account == "bank":
        if "zelle" in text or "phone/online" in text:
            return "Zelle Transfers"
        if any(kw in text for kw in ["discover", "chase credit", "apple", "amex", "capital"]):
            return "Credit Card Payments"
        if "nissan" in text:
            return "Auto Payments"
        if any(kw in text for kw in ["rent", "twin", "resident", "propert", "billpay", "og&e", "city", "att"]):
            return "Housing and Utilities"
        if "tax" in text:
            return "Tax"
        if any(kw in text for kw in ["fid", "robin"]):
            return "Investment & Stocks"
        if "bursar" in text:
            return "Education"
        if any(kw in text for kw in ["merchant", "purchase", "paypal"]):
            return "Purchases and Refunds"
        if any(kw in text for kw in ["deposit", "check", "payroll"]):
            return "Income"
        return "Others"

    # 2. Credit Card Logic
    # Payments/Rewards/Credits
    if any(kw in text for kw in ["payment", "thank you", "autopay", "reward", "cashback", 
                                  "cash back", "statement credit", "mobile pymt", "capital one mobile"]):
        return "Credit Due Payments"
    
    # Food & Dining
    if any(kw in text for kw in ["restaurant", "cafe", "coffee", "pizza", "burger", "halal",
                                  "chipotle", "starbucks", "mcdonald", "taco", "food", 
                                  "doordash", "ubereats", "grubhub", "panda", "whataburger",
                                  "chick-fil", "subway", "wendy", "sonic", "panera",
                                  "wingstop", "domino", "papa john", "little caesar", 
                                  "jersey mike", "firehouse", "potbelly", "krispy", "dunkin",
                                  "noodle", "pho", "ramen", "sushi", "chinese", "thai", 
                                  "boba", "tea house", "kuppanna", "jollibee",
                                  "chaat", "indian", "desi district", "biryani", "curry",
                                  "mexican", "velvet taco", "torchy",
                                  "bakery", "deli", "grill", "kitchen", "eatery", "diner",
                                  "bistro", "pub", "bar", "brewery", "treats", "tiff's",
                                  "einstein", "ihop", "waffle", "pancake",
                                  "fruitea", "smoothie", "juice", "yogurt", "ice cream",
                                  "gyros", "stop-n-go", "red hot"]):
        return "Food & Dining"
    
    # Entertainment & Travel
    if any(kw in text for kw in ["netflix", "spotify", "youtube", "hulu", "disney",
                                  "hbo", "movie", "cinema", "amc", "gaming", "playstation",
                                  "xbox", "steam", "nintendo", "airline", "hotel", "airbnb",
                                  "expedia", "booking", "travel", "vacation", "frontier",
                                  "southwest", "american air", "delta", "united"]):
        return "Entertainment & Travel"
    
    # Transportation
    if any(kw in text for kw in ["uber", "lyft", "gas", "fuel", "racetrac", "shell", 
                                  "exxon", "chevron", "valero", "murphy", "qt ", "quiktrip",
                                  "parking", "toll", "dart", "transit", "costco gas"]):
        return "Transportation"
    
    # Purchases (Shopping)
    if any(kw in text for kw in ["amazon", "walmart", "wal-mart", "target", "costco",
                                  "bestbuy", "best buy", "home depot", "lowes", "ikea", 
                                  "store", "shop", "market", "grocery", "heb", "kroger", 
                                  "aldi", "trader joe", "walgreens", "cvs", "pharmacy", 
                                  "dollar", "ross", "marshalls", "tjmaxx", "nordstrom", 
                                  "macy", "kohls", "apple.com", "ebay", "etsy",
                                  "prime", "amzn"]):
        return "Purchases and Refunds"
    
    # Housing & Utilities
    if any(kw in text for kw in ["electric", "water", "internet", "utility", "spectrum",
                                  "comcast", "att", "verizon", "t-mobile", "sprint",
                                  "insurance", "geico", "allstate", "progressive",
                                  "rent", "apartment"]):
        return "Housing & Utilities"
    
    # Health
    if any(kw in text for kw in ["doctor", "hospital", "clinic", "medical", "dental",
                                  "health", "urgent care", "lab", "vision"]):
        return "Health & Insurance"
    
    # Education
    if any(kw in text for kw in ["university", "college", "school", "tuition", "book",
                                  "education", "course", "udemy", "coursera"]):
        return "Education"
        
    # Specific Known 'Others'
    if any(kw in text for kw in ["smoke", "vape", "liquor", "wine", "beer"]):
        return "Others"
        
    return "Others"

UPDATE_PATH = "gold/simplefin_updates.parquet"

def process_and_merge():
    # 1. Load Existing Data (Static + Updates)
    if not os.path.exists(DATA_PATH):
        logger.error(f"Static data not found at {DATA_PATH}")
        return

    logger.info("Loading existing dataset...")
    # Load static history
    df_static = pd.read_parquet(DATA_PATH)
    
    # Load any previous updates
    if os.path.exists(UPDATE_PATH):
        logger.info(f"Loading previous updates from {UPDATE_PATH}")
        df_updates = pd.read_parquet(UPDATE_PATH)
        df_combined = pd.concat([df_static, df_updates], ignore_index=True)
    else:
        df_updates = pd.DataFrame()
        df_combined = df_static

    # Type conversion
    df_combined['date'] = pd.to_datetime(df_combined['date'])
    df_combined['amount'] = pd.to_numeric(df_combined['amount'], errors='coerce')
    df_combined['balance'] = pd.to_numeric(df_combined['balance'], errors='coerce')

    # Per-account max dates for deduplication
    max_dates = df_combined.groupby('account')['date'].max().to_dict()
    logger.info("Per-account latest dates:")
    for acct, dt in sorted(max_dates.items()):
        logger.info(f"  {acct}: {dt.date()}")
    
    # 2. Fetch last 90 days from SimpleFin
    try:
        raw_data = fetch_simplefin_data(days_back=90)
    except Exception:
        return

    # 3. Convert to DataFrame — only keep transactions AFTER the last date per account
    new_records = []
    for acc in raw_data.get("accounts", []):
        org_name = acc["org"]["name"]
        account_name = ACCOUNT_MAPPING.get(org_name, org_name.lower())
        
        # Last date we have for this account (or epoch if new account)
        account_max_date = max_dates.get(account_name, pd.Timestamp("1970-01-01"))
        
        # Grab account-level balance if available
        acc_balance = acc.get("balance")
        
        for txn in acc.get("transactions", []):
            posted_ts = txn.get("posted") or txn.get("transacted_at")
            txn_date = datetime.fromtimestamp(posted_ts)
            
            # Only add if newer than what we already have for THIS account
            if txn_date <= account_max_date:
                continue

            amount = float(txn["amount"])
            desc = txn.get("description", "")
            payee = txn.get("payee", "")
            
            category = categorize_transaction(account_name, desc, payee)
            
            # Use transaction-level balance if available, else account balance for latest txn
            txn_balance = txn.get("balance")
            if txn_balance is not None:
                txn_balance = float(txn_balance)
            
            new_records.append({
                "date": txn_date,
                "description": desc,
                "amount": amount,
                "category": category,
                "type": "credit" if amount > 0 else "debit",
                "account": account_name,
                "balance": txn_balance,
            })
    
    if not new_records:
        logger.info("No new transactions found.")
        return

    df_new = pd.DataFrame(new_records)
    logger.info(f"Found {len(df_new)} new transactions.")

    # 4. Calculate Bank Balances
    bank_new = df_new[df_new['account'] == 'bank'].copy()
    
    if not bank_new.empty:
        # Get last known balance from combined history
        bank_history = df_combined[df_combined['account'] == 'bank']
        
        if not bank_history[bank_history['balance'].notna()].empty:
            last_valid_balance_row = bank_history[bank_history['balance'].notna()].sort_values('date', ascending=False).iloc[0]
            running_balance = float(last_valid_balance_row['balance'])
            logger.info(f"Starting balance calculation from: ${running_balance:.2f} (Date: {last_valid_balance_row['date'].date()})")
        else:
            logger.warning("No previous bank balance found! Using 0.00")
            running_balance = 0.00
        
        # Sort new transactions ascending to calculate forward
        bank_new = bank_new.sort_values('date', ascending=True)
        
        # Add slight time offset to ensure unique strictly increasing ordering
        # This guarantees that the last calculated balance has the latest timestamp
        for i in range(1, len(bank_new)):
            if bank_new.iloc[i]['date'] <= bank_new.iloc[i-1]['date']:
                bank_new.iloc[i, bank_new.columns.get_loc('date')] = bank_new.iloc[i-1]['date'] + timedelta(seconds=1)
        
        for idx, row in bank_new.iterrows():
            running_balance += row['amount']
            bank_new.at[idx, 'balance'] = round(running_balance, 2)
            
        # Update df_new with calculated balances
        df_new.update(bank_new)
        logger.info(f"Updated Bank balances. Newest balance: ${running_balance:.2f}")

    # 5. Save ONLY the updates (Previous Updates + New)
    # Ensure columns match
    columns = ['date', 'description', 'amount', 'category', 'type', 'account', 'balance']
    df_new = df_new[columns]
    
    # Concat with previous updates
    if not df_updates.empty:
        df_final_updates = pd.concat([df_updates, df_new], ignore_index=True)
    else:
        df_final_updates = df_new
        
    df_final_updates = df_final_updates.sort_values('date', ascending=False).reset_index(drop=True)
    
    # Save updates file
    df_final_updates.to_parquet(UPDATE_PATH, index=False)
    logger.info(f"Successfully updated {UPDATE_PATH}")
    logger.info(f"Total transactions in updates file: {len(df_final_updates)}")

if __name__ == "__main__":
    process_and_merge()
