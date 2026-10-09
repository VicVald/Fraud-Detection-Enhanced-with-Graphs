"""
split.py — Data splitting and identity-based routing.

Decisions:
- Temporal split: Sort by numeric transaction ID (from tx_id) and split at 80%.
- Identity routing: Split by even/odd source account ID (src_id) to mock Group A/B.
"""

from __future__ import annotations

import pandas as pd

from sklearn.model_selection import train_test_split

from .preprocessing import ALWAYS_EXCLUDE, IDENTITY_COLS


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def random_split(
    df: pd.DataFrame,
    boundary: float = 0.8,
    seed: int = 42,
    stratify: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into training and validation sets randomly (optionally stratified by isFraud).

    Parameters
    ----------
    df:
        Full merged DataFrame.
    boundary:
        Percentile boundary to split on (default 0.80).
    seed:
        Random state seed.
    stratify:
        Whether to stratify by isFraud if present.
    verbose:
        Whether to print split statistics.

    Returns
    -------
    (train_df, val_df)
    """
    stratify_target = df["isFraud"] if (stratify and "isFraud" in df.columns) else None
    train_df, val_df = train_test_split(
        df,
        train_size=boundary,
        random_state=seed,
        stratify=stratify_target,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if verbose:
        print("[split] random stratified split:" if stratify_target is not None else "[split] random split:")
        _print_split_stats(train_df, val_df)

    return train_df, val_df


def temporal_split(
    df: pd.DataFrame,
    boundary: float = 0.8,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into training and validation sets by transaction timestamp.

    This preserves transaction order and prevents data leakage.

    Parameters
    ----------
    df:
        Full merged DataFrame.
    boundary:
        Percentile boundary to split on (default 0.80).

    Returns
    -------
    (train_df, val_df)
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * boundary)
    
    train_df = df_sorted.iloc[:split_idx].copy()
    val_df = df_sorted.iloc[split_idx:].copy()

    if verbose:
        _print_split_stats(train_df, val_df)

    return train_df, val_df


def _print_split_stats(train: pd.DataFrame, val: pd.DataFrame) -> None:
    n_train, n_val = len(train), len(val)
    n_total = n_train + n_val
    fraud_train = train["isFraud"].mean() if "isFraud" in train else float("nan")
    fraud_val = val["isFraud"].mean() if "isFraud" in val else float("nan")
    print(
        f"[split] temporal split:\n"
        f"  train: {n_train:,} ({n_train/n_total:.1%}) | fraud rate: {fraud_train:.3%}\n"
        f"  val:   {n_val:,} ({n_val/n_total:.1%}) | fraud rate: {fraud_val:.3%}"
    )


# ---------------------------------------------------------------------------
# Identity routing
# ---------------------------------------------------------------------------


def identity_split(
    df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into Group A (even src_id) and Group B (odd src_id).

    Routing signal: source account ID mod 2.
    - Group A: even src_id -> Model A (transaction features only)
    - Group B: odd src_id  -> Model B (transaction + identity/account features)

    Parameters
    ----------
    df:
        DataFrame containing transactions with src_id.

    Returns
    -------
    (group_a, group_b)
    """
    src_idx = df["src_id"].str.replace("acc_", "").astype(int)
    is_even = (src_idx % 2 == 0)
    group_a = df[is_even].copy()
    group_b = df[~is_even].copy()

    if verbose:
        n = len(df)
        fa = group_a["isFraud"].mean() if "isFraud" in group_a else float("nan")
        fb = group_b["isFraud"].mean() if "isFraud" in group_b else float("nan")
        print(
            f"[split] identity routing:\n"
            f"  Group A (even acc): {len(group_a):,} ({len(group_a)/n:.1%}) | fraud: {fa:.3%}\n"
            f"  Group B (odd acc) : {len(group_b):,} ({len(group_b)/n:.1%}) | fraud: {fb:.3%}"
        )

    return group_a, group_b


# ---------------------------------------------------------------------------
# Feature column selection
# ---------------------------------------------------------------------------


def get_feature_cols(
    df: pd.DataFrame,
    include_identity: bool = False,
) -> list[str]:
    """Return the list of feature columns for a given model.

    Excludes: tx_id, isFraud, timestamp, etc. (always).
    If *include_identity* is False, also excludes all IDENTITY_COLS.

    Parameters
    ----------
    df:
        Reference DataFrame.
    include_identity:
        True  → Model B feature set (transaction + account features)
        False → Model A feature set (transaction only)

    Returns
    -------
    List of column names to use as X (features).
    """
    exclude = set(ALWAYS_EXCLUDE)
    if not include_identity:
        exclude.update(IDENTITY_COLS)

    return [c for c in df.columns if c not in exclude]

