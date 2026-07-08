"""
data_loader.py
--------------
Load and rename the UCI credit-card default dataset for the
Moroccan banking context. Converts TWD amounts to MAD (× 0.35).
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "default_of_credit_card_clients.csv"
PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "credit_data_cleaned.csv"

# TWD → MAD conversion rate (approximate)
TWD_TO_MAD = 0.35

RENAME_MAP = {
    "LIMIT_BAL":  "credit_limit_mad",
    "SEX":        "gender",
    "EDUCATION":  "education_level",
    "MARRIAGE":   "marital_status",
    "AGE":        "age",
    "PAY_0":      "payment_status_month_1",
    "PAY_2":      "payment_status_month_2",
    "PAY_3":      "payment_status_month_3",
    "PAY_4":      "payment_status_month_4",
    "PAY_5":      "payment_status_month_5",
    "PAY_6":      "payment_status_month_6",
    "BILL_AMT1":  "bill_amount_month_1",
    "BILL_AMT2":  "bill_amount_month_2",
    "BILL_AMT3":  "bill_amount_month_3",
    "BILL_AMT4":  "bill_amount_month_4",
    "BILL_AMT5":  "bill_amount_month_5",
    "BILL_AMT6":  "bill_amount_month_6",
    "PAY_AMT1":   "payment_amount_month_1",
    "PAY_AMT2":   "payment_amount_month_2",
    "PAY_AMT3":   "payment_amount_month_3",
    "PAY_AMT4":   "payment_amount_month_4",
    "PAY_AMT5":   "payment_amount_month_5",
    "PAY_AMT6":   "payment_amount_month_6",
    "default payment next month": "target",
}

BILL_COLS    = [f"bill_amount_month_{i}"    for i in range(1, 7)]
PAY_AMT_COLS = [f"payment_amount_month_{i}" for i in range(1, 7)]
PAY_STS_COLS = [f"payment_status_month_{i}" for i in range(1, 7)]
NUMERIC_MAD  = ["credit_limit_mad"] + BILL_COLS + PAY_AMT_COLS


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Load the raw CSV, drop the ID column, rename columns,
    and convert monetary columns from TWD to MAD.
    """
    df = pd.read_csv(path)
    df.drop(columns=["ID"], errors="ignore", inplace=True)
    df.rename(columns=RENAME_MAP, inplace=True)

    # Convert monetary amounts
    for col in NUMERIC_MAD:
        if col in df.columns:
            df[col] = (df[col] * TWD_TO_MAD).round(0).astype(int)

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Handle missing values
      2. Fix out-of-range categoricals
      3. Encode gender, education, marital status
      4. Cap outliers in credit_limit_mad at 99th percentile
    """
    df = df.copy()

    # --- Missing values ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # --- gender: 1=Male, 2=Female → 0/1 ---
    df["gender"] = df["gender"].map({1: 0, 2: 1})          # 0=Male, 1=Female

    # --- education_level: group 4,5,6 into 4 (Other) ---
    df["education_level"] = df["education_level"].replace({5: 4, 6: 4, 0: 4})

    # --- marital_status: map 0 (unknown) to 3 (Other) ---
    df["marital_status"] = df["marital_status"].replace({0: 3})

    # --- Outlier cap for credit_limit_mad ---
    p99 = df["credit_limit_mad"].quantile(0.99)
    df["credit_limit_mad"] = df["credit_limit_mad"].clip(upper=p99)

    return df


def load_and_clean(path: Path = RAW_PATH, save: bool = True) -> pd.DataFrame:
    """Convenience wrapper: load → clean → optionally save."""
    df = clean(load_raw(path))
    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
    return df


DATA_DICTIONARY = {
    "credit_limit_mad": {
        "description": "Total credit limit on the customer's card (MAD)",
        "banker_meaning": "Maximum exposure to the bank; higher limits typically given to lower-risk clients.",
        "risk_relevance": "Inversely correlated with default — higher-limit clients tend to be better payers.",
        "dtype": "int", "range": "3,500 – 350,000 MAD",
    },
    "gender": {
        "description": "Customer gender (0 = Male, 1 = Female)",
        "banker_meaning": "Demographic feature; used only for compliance monitoring.",
        "risk_relevance": "Slight statistical variation; must not be used as a primary decision driver (regulatory).",
        "dtype": "binary", "range": "0 or 1",
    },
    "education_level": {
        "description": "Highest education attained (1=Graduate, 2=University, 3=High School, 4=Other)",
        "banker_meaning": "Proxy for income stability and financial literacy.",
        "risk_relevance": "Graduate-educated customers show lower default rates.",
        "dtype": "ordinal", "range": "1–4",
    },
    "marital_status": {
        "description": "Marital status (1=Married, 2=Single, 3=Other)",
        "banker_meaning": "Affects household income pooling and financial obligations.",
        "risk_relevance": "Married customers tend to have slightly lower default rates.",
        "dtype": "nominal", "range": "1–3",
    },
    "age": {
        "description": "Customer age in years",
        "banker_meaning": "Older customers often have longer credit histories and more stable income.",
        "risk_relevance": "Non-linear relationship; very young and very old customers may show higher risk.",
        "dtype": "int", "range": "21–75",
    },
    "payment_status_month_1": {
        "description": "Repayment status in most recent month (-2=no consumption, -1=paid in full, 0=minimum paid, 1=1 month delay, …, 9=9+ months delay)",
        "banker_meaning": "The single strongest default predictor; any positive value signals trouble.",
        "risk_relevance": "Directly measures delinquency — the most important feature in every model.",
        "dtype": "ordinal", "range": "-2 to 9",
    },
    "bill_amount_month_1": {
        "description": "Statement balance in most recent month (MAD)",
        "banker_meaning": "Shows how much the customer owes right now.",
        "risk_relevance": "High balances relative to credit limit indicate stress; context-dependent.",
        "dtype": "int", "range": "0 – 350,000 MAD",
    },
    "payment_amount_month_1": {
        "description": "Amount paid in most recent month (MAD)",
        "banker_meaning": "Cash actually transferred by the customer; zero payments are a red flag.",
        "risk_relevance": "Low or zero payments signal inability or unwillingness to repay.",
        "dtype": "int", "range": "0 – 500,000 MAD",
    },
    "target": {
        "description": "Default in the following month (1 = defaulted, 0 = did not default)",
        "banker_meaning": "Binary outcome the model predicts.",
        "risk_relevance": "This is what we are predicting.",
        "dtype": "binary", "range": "0 or 1",
    },
}
