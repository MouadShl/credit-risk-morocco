"""
evaluate.py
-----------
All evaluation utilities: metrics, comparison tables, ROC/PR curves,
confusion matrices, and SHAP explainability helpers.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay,
)

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = [
    "#C0392B", "#2980B9", "#27AE60", "#8E44AD",
    "#F39C12", "#16A085", "#2C3E50", "#E67E22",
]
MODEL_COLORS = {}   # filled in compute_all_metrics()

sns.set_theme(style="whitegrid", font_scale=1.05)


# ────────────────────────────────────────────────────────────────────────────
#  CORE METRICS
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(model, X_test_proc: np.ndarray, y_test: pd.Series,
                    threshold: float = 0.50) -> dict:
    """Compute 5 key metrics at a given classification threshold."""
    y_prob = model.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "AUC-ROC":   round(roc_auc_score(y_test, y_prob), 4),
    }


def compute_all_metrics(
    fitted_models: dict,
    X_test_proc: np.ndarray,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Return a tidy DataFrame of metrics for every model."""
    global MODEL_COLORS
    records = []
    for i, (name, model) in enumerate(fitted_models.items()):
        m = compute_metrics(model, X_test_proc, y_test, threshold)
        m["Model"] = name
        records.append(m)
        MODEL_COLORS[name] = PALETTE[i % len(PALETTE)]

    df = pd.DataFrame(records).set_index("Model")
    return df[["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]]


def highlight_best(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Highlight the maximum value in each column."""
    return df.style.highlight_max(
        axis=0,
        props="background-color: #27AE60; color: white; font-weight: bold;",
    ).format("{:.4f}")


# ────────────────────────────────────────────────────────────────────────────
#  CHARTS
# ────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str):
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_roc_curves(fitted_models: dict, X_test_proc: np.ndarray,
                    y_test: pd.Series, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, model in fitted_models.items():
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2,
                color=MODEL_COLORS.get(name, "#555"),
                label=f"{name}  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — All Models\nMoroccan Credit Default Portfolio",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    if save:
        _save(fig, "roc_curves.png")
    return fig


def plot_pr_curves(fitted_models: dict, X_test_proc: np.ndarray,
                   y_test: pd.Series, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline = y_test.mean()
    for name, model in fitted_models.items():
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(rec, prec, lw=2,
                color=MODEL_COLORS.get(name, "#555"), label=name)
    ax.axhline(baseline, linestyle="--", color="gray", lw=1.2,
               label=f"No-skill baseline ({baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    if save:
        _save(fig, "pr_curves.png")
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    n_models = len(metrics_df)
    x = np.arange(len(metrics))
    width = 0.10
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (model_name, row) in enumerate(metrics_df.iterrows()):
        bars = ax.bar(
            x + i * width,
            [row[m] for m in metrics],
            width,
            label=model_name,
            color=MODEL_COLORS.get(model_name, PALETTE[i]),
            edgecolor="white", linewidth=0.5,
        )
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — 5 Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    if save:
        _save(fig, "metrics_comparison.png")
    return fig


def plot_confusion_matrix(model, X_test_proc: np.ndarray, y_test: pd.Series,
                          model_name: str = "XGBoost",
                          threshold: float = 0.50, save: bool = True) -> plt.Figure:
    y_prob = model.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Default", "Default"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold = {threshold})",
                 fontsize=12, fontweight="bold")
    if save:
        fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        _save(fig, fname)
    return fig


def plot_xgb_feature_importance(model, feature_names: list,
                                 top_n: int = 15, save: bool = True) -> plt.Figure:
    """XGBoost built-in gain importance."""
    imp = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.Series(imp)
        .rename_axis("Feature")
        .reset_index(name="Gain")
        .sort_values("Gain", ascending=True)
        .tail(top_n)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(imp_df)))
    ax.barh(imp_df["Feature"], imp_df["Gain"], color=colors, edgecolor="white")
    ax.set_xlabel("Gain", fontsize=11)
    ax.set_title(f"XGBoost Feature Importance (Top {top_n}) — Gain",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    if save:
        _save(fig, "xgb_feature_importance.png")
    return fig


def plot_shap_summary(shap_values: np.ndarray, X_test_df: pd.DataFrame,
                      save: bool = True) -> plt.Figure:
    """SHAP beeswarm summary plot."""
    import shap
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False, plot_size=None, ax=ax)
    ax.set_title("SHAP Summary Plot — XGBoost\n(Impact on Default Probability)",
                 fontsize=12, fontweight="bold")
    if save:
        _save(fig, "shap_summary.png")
    return fig
