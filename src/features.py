"""
features.py
-----------
All feature engineering for the Credit Risk Morocco project.
Run after data_loader.clean().
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

BILL_COLS    = [f"bill_amount_month_{i}"    for i in range(1, 7)]
PAY_AMT_COLS = [f"payment_amount_month_{i}" for i in range(1, 7)]
PAY_STS_COLS = [f"payment_status_month_{i}" for i in range(1, 7)]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 7 domain-driven features.

    Returns a new DataFrame with original columns preserved
    plus the new engineered columns.
    """
    df = df.copy()

    total_bills    = df[BILL_COLS].sum(axis=1).replace(0, np.nan)
    total_payments = df[PAY_AMT_COLS].sum(axis=1)

    # 1. Average statement balance over 6 months
    df["avg_bill_6m"] = df[BILL_COLS].mean(axis=1)

    # 2. Average payment over 6 months
    df["avg_payment_6m"] = df[PAY_AMT_COLS].mean(axis=1)

    # 3. Payment-to-bill ratio (higher → better payer)
    df["payment_to_bill_ratio"] = (total_payments / total_bills).fillna(0).clip(0, 5)

    # 4. Worst payment delay across 6 months (max of payment status)
    df["max_payment_delay"] = df[PAY_STS_COLS].max(axis=1)

    # 5. Credit utilization: avg bill / credit limit
    df["credit_utilization"] = (
        df["avg_bill_6m"] / df["credit_limit_mad"].replace(0, np.nan)
    ).fillna(0).clip(0, 3)

    # 6. Payment consistency (lower std = more predictable payer)
    df["payment_consistency"] = df[PAY_AMT_COLS].std(axis=1).fillna(0)

    # 7. Number of months with any payment delay (payment_status > 0)
    df["months_delayed"] = (df[PAY_STS_COLS] > 0).sum(axis=1)

    return df


def get_feature_groups(df: pd.DataFrame) -> dict:
    """Return categorised lists of feature column names."""
    engineered = [
        "avg_bill_6m", "avg_payment_6m", "payment_to_bill_ratio",
        "max_payment_delay", "credit_utilization",
        "payment_consistency", "months_delayed",
    ]
    numeric = (
        ["credit_limit_mad", "age"]
        + BILL_COLS + PAY_AMT_COLS
        + engineered
    )
    categorical_ordinal = PAY_STS_COLS  # ordinal: -2 … 9
    categorical_nominal = ["gender", "education_level", "marital_status"]

    return {
        "numeric": numeric,
        "categorical_ordinal": categorical_ordinal,
        "categorical_nominal": categorical_nominal,
        "engineered": engineered,
        "all_features": numeric + categorical_ordinal + categorical_nominal,
    }


def build_preprocessor(feature_groups: dict) -> ColumnTransformer:
    """
    ColumnTransformer that:
      - StandardScales all numeric features
      - OrdinalEncodes payment-status columns (-2 … 9)
      - OrdinalEncodes nominal categoricals (gender, education, marital)
    """
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])

    ordinal_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(
            categories="auto",
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ))
    ])

    nominal_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(
            categories="auto",
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",     numeric_pipeline,  feature_groups["numeric"]),
            ("ord",     ordinal_pipeline,  feature_groups["categorical_ordinal"]),
            ("nominal", nominal_pipeline,  feature_groups["categorical_nominal"]),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(df: pd.DataFrame, feature_groups: dict, test_size: float = 0.20, seed: int = 42):
    """
    Stratified 80/20 train-test split.
    Returns: X_train, X_test, y_train, y_test
    """
    all_features = feature_groups["all_features"]
    X = df[all_features]
    y = df["target"]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,       # ← preserves class ratio in both splits
    )
