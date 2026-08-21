"""
preprocessing.py — Feature cleaning and encoding for the IEEE-CIS baseline.

Decisions (see ADR-0001 / grilling session):
- Missing values: NOT imputed. XGBoost receives raw NaN and learns the
  optimal default branch direction natively (Q3 = Option D).
- Columns dropped: >90% null rate OR near-zero variance (Q5).
- Categorical encoding: Label encoding via sklearn LabelEncoder (Q4 = Option A).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Column dropping
# ---------------------------------------------------------------------------

#: Identity (account) columns.
IDENTITY_COLS: list[str] = [
    "src_balance",
    "src_risk_score",
    "dst_balance",
    "dst_risk_score",
]

#: Columns to always exclude from feature matrices regardless of split.
ALWAYS_EXCLUDE: list[str] = [
    "tx_id",
    "src_id",
    "dst_id",
    "timestamp",
    "description",
    "embedding",
    "isFraud",
    "src_customer_name",
    "dst_customer_name",
    "src_creation_date",
    "dst_creation_date",
]



def drop_high_null_cols(
    df: pd.DataFrame,
    threshold: float = 0.90,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns where the fraction of NaN values exceeds *threshold*.

    Parameters
    ----------
    df:
        Input DataFrame (training set — fit on train, apply to val/test).
    threshold:
        Null rate above which a column is dropped. Default 0.90 (90 %).
    verbose:
        If True, print a summary of dropped columns.

    Returns
    -------
    (cleaned_df, dropped_cols)
    """
    null_rates = df.isnull().mean()
    to_drop = null_rates[null_rates > threshold].index.tolist()
    # Never drop target, ID, or DT columns
    to_drop = [c for c in to_drop if c not in ALWAYS_EXCLUDE]
    if verbose:
        print(
            f"[preprocess] drop_high_null: {len(to_drop)} columns dropped "
            f"(null rate > {threshold:.0%})"
        )
    return df.drop(columns=to_drop), to_drop


def drop_near_zero_variance(
    df: pd.DataFrame,
    threshold: float = 0.01,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop numerical columns whose standard deviation / mean < *threshold*.

    Uses coefficient of variation (CV) for scale-invariant variance check.
    Columns with mean == 0 are checked by absolute std instead.

    Parameters
    ----------
    df:
        Input DataFrame. Only numerical columns are evaluated.
    threshold:
        CV below which a column is considered near-zero-variance.
    verbose:
        If True, print a summary.

    Returns
    -------
    (cleaned_df, dropped_cols)
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ALWAYS_EXCLUDE]

    to_drop: list[str] = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            to_drop.append(col)
            continue
        mean = series.mean()
        std = series.std()
        if mean == 0:
            if std < threshold:
                to_drop.append(col)
        else:
            if abs(std / mean) < threshold:
                to_drop.append(col)

    if verbose:
        print(
            f"[preprocess] drop_near_zero_variance: {len(to_drop)} columns "
            f"dropped (CV < {threshold})"
        )
    return df.drop(columns=to_drop), to_drop


# ---------------------------------------------------------------------------
# Categorical detection and encoding
# ---------------------------------------------------------------------------


def get_categorical_cols(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Return all object-dtype columns, minus any in *exclude*.

    Parameters
    ----------
    df:
        Input DataFrame.
    exclude:
        Additional columns to exclude (e.g. ALWAYS_EXCLUDE).

    Returns
    -------
    List of categorical column names.
    """
    exc = set(exclude or []) | set(ALWAYS_EXCLUDE)
    return [c for c in df.select_dtypes(include=["object"]).columns if c not in exc]


def label_encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Apply label encoding to *cat_cols* in *df*.

    NaN values are preserved: LabelEncoder cannot handle NaN natively, so we
    temporarily fill with the sentinel "__NaN__" and restore NaN afterwards.

    Parameters
    ----------
    df:
        Input DataFrame (modified in place on a copy).
    cat_cols:
        Columns to encode.
    encoders:
        Pre-fitted encoders dict (from a previous call). If None, fit fresh
        encoders from *df*. Pass the fitted dict at inference time.
    verbose:
        Print summary.

    Returns
    -------
    (encoded_df, encoders_dict)
    """
    df = df.copy()
    fitted: dict[str, LabelEncoder] = encoders or {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        # Replace NaN with sentinel so LabelEncoder can process
        null_mask = df[col].isnull()
        df[col] = df[col].fillna("__NaN__")

        if col in fitted:
            # Inference mode: handle unseen categories gracefully
            le = fitted[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__NaN__")
            df[col] = le.transform(df[col])
        else:
            # Fit mode
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            fitted[col] = le

        # Restore NaN as float NaN (XGBoost handles natively)
        df[col] = df[col].astype(float)
        df.loc[null_mask, col] = np.nan

    if verbose:
        print(f"[preprocess] label_encode: {len(cat_cols)} categorical columns encoded")

    return df, fitted


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def save_dropped_cols(
    dropped: dict[str, list[str]],
    path: str | Path = "results/dropped_columns.json",
) -> None:
    """Persist the lists of dropped columns to a JSON file for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dropped, f, indent=2)
    print(f"[preprocess] Dropped column lists saved → {path}")


def load_dropped_cols(path: str | Path = "results/dropped_columns.json") -> dict[str, list[str]]:
    """Load previously saved dropped-column lists."""
    with open(path) as f:
        return json.load(f)


def apply_v_pca(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_components: int = 15,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce V1-V339 feature columns to n_components using PCA.

    Backward-compatible: if no V columns are found, returns DataFrames unchanged.
    """
    v_cols = [c for c in train_df.columns if c.startswith("V") and c[1:].isdigit()]
    
    if len(v_cols) == 0:
        if verbose:
            print("[preprocess] No V columns found. Skipping PCA.")
        return train_df, val_df

    if verbose:
        print(f"[preprocess] Reducing {len(v_cols)} V-columns to {n_components} components using PCA...")

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    train_df = train_df.copy()
    val_df = val_df.copy()

    # Fit imputer and scaler on train
    medians = train_df[v_cols].median().fillna(0.0).to_dict()
    
    X_train_v = train_df[v_cols].fillna(medians)
    X_val_v = val_df[v_cols].fillna(medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_v)
    X_val_scaled = scaler.transform(X_val_v)

    # Fit PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    # Add PCA features
    for i in range(n_components):
        col_name = f"PCA_V_{i}"
        train_df[col_name] = X_train_pca[:, i]
        val_df[col_name] = X_val_pca[:, i]

    # Drop original V columns
    train_df = train_df.drop(columns=v_cols)
    val_df = val_df.drop(columns=v_cols)

    if verbose:
        print(f"[preprocess] PCA completed. Dropped {len(v_cols)} columns, added {n_components} PCA columns.")

    return train_df, val_df

