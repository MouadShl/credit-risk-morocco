"""
models.py
---------
Define, train, and compare all 7 models.
Hyperparameter tuning for XGBoost via RandomizedSearchCV.
"""

from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Positive-class weight for imbalance (≈ 22% default rate → ratio ≈ 3.5)
SCALE_POS_WEIGHT = 3.5

MODEL_REGISTRY: dict = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.1,
        solver="lbfgs",
        random_state=42,
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=15,
        metric="euclidean",
        weights="distance",
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        is_unbalance=True,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    ),
    "CatBoost": CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        random_state=42,
        verbose=0,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    ),
}

# ─────────────────────────────────────────────
#  XGBOOST HYPERPARAMETER SEARCH SPACE
# ─────────────────────────────────────────────
XGB_PARAM_DIST = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [3, 5, 7, 10],
    "learning_rate":     [0.01, 0.05, 0.1, 0.2],
    "subsample":         [0.6, 0.8, 1.0],
    "colsample_bytree":  [0.6, 0.8, 1.0],
    "scale_pos_weight":  [1, 2, 3, 4],
    "min_child_weight":  [1, 3, 5],
    "gamma":             [0, 0.1, 0.2],
}


def train_all_models(
    X_train_proc: np.ndarray,
    y_train: pd.Series,
) -> dict:
    """
    Train all 7 models on preprocessed training data.
    Returns a dict {model_name: fitted_estimator}.
    """
    fitted = {}
    for name, model in MODEL_REGISTRY.items():
        t0 = time.time()
        model.fit(X_train_proc, y_train)
        elapsed = time.time() - t0
        fitted[name] = model
        print(f"  ✓ {name:<22} trained in {elapsed:.1f}s")
    return fitted


def tune_xgboost(
    X_train_proc: np.ndarray,
    y_train: pd.Series,
    n_iter: int = 40,
    cv: int = 5,
    seed: int = 42,
) -> tuple[XGBClassifier, dict]:
    """
    RandomizedSearchCV over XGB_PARAM_DIST.
    Returns (best_estimator, best_params).
    """
    base_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
    )
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=XGB_PARAM_DIST,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv_strategy,
        n_jobs=-1,
        random_state=seed,
        verbose=1,
        refit=True,
    )
    search.fit(X_train_proc, y_train)
    return search.best_estimator_, search.best_params_
