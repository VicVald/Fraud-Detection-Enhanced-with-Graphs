"""
metrics.py — Evaluation metrics and verification suite for the baseline model.

Implements the full verification suite from ADR-0003 Q7=C:
  1. Temporal leakage check
  2. Class balance preservation check
  3. Sanity baseline (model beats stratified-random)
  4. Routing coverage check
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the primary and secondary evaluation metrics (ADR-0003).

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0 = legitimate, 1 = fraud).
    y_score:
        Continuous fraud probability scores in [0, 1].
    threshold:
        Decision threshold for F1 and precision/recall at threshold.

    Returns
    -------
    dict with keys: pr_auc, roc_auc, f1_at_threshold, f1_best, precision_at_threshold,
    recall_at_threshold, best_threshold.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    f1_at_t = f1_score(y_true, y_pred, zero_division=0)

    # Best F1 across all thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    f1_best = float(f1_scores[best_idx])
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else threshold
    precision_best = float(precisions[best_idx])
    recall_best = float(recalls[best_idx])

    # Precision/recall at given threshold
    y_pred_t = (y_score >= threshold).astype(int)
    tp = ((y_pred_t == 1) & (y_true == 1)).sum()
    fp = ((y_pred_t == 1) & (y_true == 0)).sum()
    fn = ((y_pred_t == 0) & (y_true == 1)).sum()
    precision_at_t = tp / (tp + fp + 1e-10)
    recall_at_t = tp / (tp + fn + 1e-10)

    return {
        "pr_auc": round(float(pr_auc), 6),
        "roc_auc": round(float(roc_auc), 6),
        "f1_at_threshold": round(float(f1_at_t), 6),
        "f1_best": round(float(f1_best), 6),
        "best_threshold": round(float(best_threshold), 6),
        "precision_at_threshold": round(float(precision_at_t), 6),
        "recall_at_threshold": round(float(recall_at_t), 6),
        "precision_best": round(float(precision_best), 6),
        "recall_best": round(float(recall_best), 6),
    }


# ---------------------------------------------------------------------------
# Verification suite (ADR-0003, Q7=C)
# ---------------------------------------------------------------------------


def verification_suite(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    group_a_idx: pd.Index | None = None,
    group_b_idx: pd.Index | None = None,
    verbose: bool = True,
) -> dict:
    """Run the four-check verification suite defined in ADR-0003.

    Checks
    ------
    1. No temporal leakage: val TransactionDT min > train TransactionDT max
    2. Class balance preserved: val fraud rate ≈ train fraud rate (±0.5 pp)
    3. Model beats stratified-random baseline (PR-AUC > fraud prevalence)
    4. Routing coverage: group_a + group_b == full val set (no rows lost)

    Parameters
    ----------
    train_df, val_df:
        Train and validation DataFrames from the temporal split.
    y_true:
        Ground-truth labels for the validation set.
    y_score:
        Model fraud probability scores for the validation set.
    group_a_idx, group_b_idx:
        Index objects for group A and B within val_df.
        If None, check #4 is skipped.

    Returns
    -------
    dict with check results and computed metrics.
    """
    results: dict = {"checks": {}, "metrics": {}}
    passed = 0
    total = 0

    # ------------------------------------------------------------------
    # Check 1: Temporal leakage
    # ------------------------------------------------------------------
    total += 1
    train_max_time = train_df["timestamp"].max()
    val_min_time = val_df["timestamp"].min()
    ok1 = val_min_time > train_max_time
    results["checks"]["no_temporal_leakage"] = {
        "passed": bool(ok1),
        "train_max_dt": str(train_max_time),
        "val_min_dt": str(val_min_time),
    }
    if verbose:
        status = "✅" if ok1 else "❌"
        print(
            f"{status} Check 1 — No temporal leakage: "
            f"val_min_time={val_min_time} > train_max_time={train_max_time}"
        )
    if ok1:
        passed += 1

    # ------------------------------------------------------------------
    # Check 2: Class balance preserved
    # ------------------------------------------------------------------
    total += 1
    train_rate = train_df["isFraud"].mean()
    val_rate = val_df["isFraud"].mean()
    ok2 = abs(val_rate - train_rate) <= 0.005  # ±0.5 percentage points
    results["checks"]["class_balance_preserved"] = {
        "passed": bool(ok2),
        "train_fraud_rate": round(float(train_rate), 5),
        "val_fraud_rate": round(float(val_rate), 5),
        "delta": round(float(abs(val_rate - train_rate)), 5),
    }
    if verbose:
        status = "✅" if ok2 else "⚠️ "
        print(
            f"{status} Check 2 — Class balance: "
            f"train={train_rate:.3%} | val={val_rate:.3%} | Δ={abs(val_rate-train_rate):.3%}"
        )
    if ok2:
        passed += 1

    # ------------------------------------------------------------------
    # Check 3: Model beats stratified-random baseline
    # ------------------------------------------------------------------
    total += 1
    model_pr_auc = average_precision_score(np.asarray(y_true), np.asarray(y_score))
    # Stratified random PR-AUC ≈ fraud prevalence (Saito & Rehmsmeier 2015)
    random_pr_auc = float(np.asarray(y_true).mean())
    ok3 = model_pr_auc > random_pr_auc
    results["checks"]["beats_random_baseline"] = {
        "passed": bool(ok3),
        "model_pr_auc": round(float(model_pr_auc), 6),
        "random_pr_auc": round(float(random_pr_auc), 6),
    }
    if verbose:
        status = "✅" if ok3 else "❌"
        print(
            f"{status} Check 3 — Beats random: "
            f"model PR-AUC={model_pr_auc:.4f} > random={random_pr_auc:.4f}"
        )
    if ok3:
        passed += 1

    # ------------------------------------------------------------------
    # Check 4: Routing coverage (if indices provided)
    # ------------------------------------------------------------------
    if group_a_idx is not None and group_b_idx is not None:
        total += 1
        covered = len(group_a_idx) + len(group_b_idx)
        expected = len(val_df)
        overlap = len(group_a_idx.intersection(group_b_idx))
        ok4 = covered == expected and overlap == 0
        results["checks"]["routing_coverage"] = {
            "passed": bool(ok4),
            "group_a_size": len(group_a_idx),
            "group_b_size": len(group_b_idx),
            "total_covered": covered,
            "expected": expected,
            "overlap": overlap,
        }
        if verbose:
            status = "✅" if ok4 else "❌"
            print(
                f"{status} Check 4 — Routing coverage: "
                f"A={len(group_a_idx):,} + B={len(group_b_idx):,} = {covered:,} / {expected:,} | "
                f"overlap={overlap}"
            )
        if ok4:
            passed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    results["metrics"] = compute_metrics(y_true, y_score)
    results["summary"] = {"passed": passed, "total": total, "all_passed": passed == total}

    if verbose:
        print(f"\n{'='*50}")
        print(f"Verification: {passed}/{total} checks passed")
        print(f"  PR-AUC  : {results['metrics']['pr_auc']:.4f}")
        print(f"  ROC-AUC : {results['metrics']['roc_auc']:.4f}")
        print(f"  Precision (best): {results['metrics']['precision_best']:.4f}")
        print(f"  Recall (best)   : {results['metrics']['recall_best']:.4f}")
        print(f"  F1 (best): {results['metrics']['f1_best']:.4f} @ threshold={results['metrics']['best_threshold']:.3f}")
        print(f"{'='*50}")

    return results


def save_metrics(metrics: dict, path: str | Path = "results/baseline_metrics.json") -> None:
    """Persist the metrics dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] metrics saved → {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_pr_curve(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    label: str = "Model",
    ax: plt.Axes | None = None,
    color: str = "#6C63FF",
) -> plt.Axes:
    """Plot Precision-Recall curve with random baseline and PR-AUC annotation.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_score:
        Fraud probability scores.
    label:
        Legend label for the model curve.
    ax:
        Optional existing matplotlib Axes. Creates a new figure if None.
    color:
        Line colour for the model curve.

    Returns
    -------
    matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    precision, recall, _ = precision_recall_curve(np.asarray(y_true), np.asarray(y_score))
    pr_auc = average_precision_score(np.asarray(y_true), np.asarray(y_score))
    fraud_rate = float(np.asarray(y_true).mean())

    ax.plot(recall, precision, lw=2, color=color, label=f"{label} (PR-AUC = {pr_auc:.4f})")
    ax.axhline(y=fraud_rate, color="gray", linestyle="--", lw=1, label=f"Random (≈ {fraud_rate:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    return ax


def plot_roc_curve(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    label: str = "Model",
    ax: plt.Axes | None = None,
    color: str = "#FF6584",
) -> plt.Axes:
    """Plot ROC curve with diagonal random baseline and AUC annotation.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_score:
        Fraud probability scores.
    label:
        Legend label for the model curve.
    ax:
        Optional existing matplotlib Axes.
    color:
        Line colour.

    Returns
    -------
    matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_score))
    auc = roc_auc_score(np.asarray(y_true), np.asarray(y_score))

    ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    return ax
