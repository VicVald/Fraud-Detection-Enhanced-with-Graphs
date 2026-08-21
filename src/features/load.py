"""
load.py — Load and merge gen-fraud-graph synthetic data files.

Provides:
    load_train() -> pd.DataFrame   merged transactions (normal + fraud) joined with accounts
    load_test()  -> pd.DataFrame   similar merged transactions without labels (or empty)
"""

from __future__ import annotations

from pathlib import Path
import glob
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_train(data_dir: Path | str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Load and merge normal and fraud transactions, and join with source/destination account data.

    Returns
    -------
    pd.DataFrame
        Merged training DataFrame with isFraud as the target column.
    """
    data_path = Path(data_dir) if data_dir is not None else _DATA_RAW
    if not data_path.exists():
        raise FileNotFoundError(
            f"[load] Raw data directory not found at {data_path}.\n"
            "Run: uv run python src/data/generate_data.py"
        )

    # 1. Load all accounts
    acc_files = glob.glob(str(data_path / "accounts" / "accounts_*.csv"))
    if not acc_files:
        raise FileNotFoundError(f"[load] No account files found in {data_path / 'accounts'}")
    accounts_df = pd.concat([pd.read_csv(f) for f in acc_files], ignore_index=True)
    if verbose:
        print(f"[load] Loaded {len(accounts_df):,} accounts.")

    # 2. Load all normal transactions (and decoys)
    tx_files = glob.glob(str(data_path / "transactions" / "transactions_*.csv"))
    if not tx_files:
        raise FileNotFoundError(f"[load] No transaction files found in {data_path / 'transactions'}")
    normal_tx_df = pd.concat([pd.read_csv(f) for f in tx_files], ignore_index=True)
    normal_tx_df["isFraud"] = 0

    # 3. Load fraud transactions
    fraud_tx_path = data_path / "fraud" / "transactions_fraud.csv"
    if not fraud_tx_path.exists():
        raise FileNotFoundError(f"[load] Fraud transactions file not found at {fraud_tx_path}")
    fraud_tx_df = pd.read_csv(fraud_tx_path)
    fraud_tx_df["isFraud"] = 1

    # Combine all transactions
    all_tx_df = pd.concat([normal_tx_df, fraud_tx_df], ignore_index=True)
    
    # Santander synthetic transactions are generated sequentially by worker processes,
    # meaning all fraud transactions appear at the very end of the index space.
    # We assign randomized timestamps over a 30-day window to simulate realistic time distribution.
    import numpy as np
    np.random.seed(seed=42)
    start_date = pd.to_datetime("2024-01-01T00:00:00")
    random_seconds = np.random.randint(0, 30 * 24 * 3600, size=len(all_tx_df))
    all_tx_df["timestamp"] = start_date + pd.to_timedelta(random_seconds, unit="s")

    if verbose:
        print(f"[load] Loaded {len(all_tx_df):,} total transactions (fraud rate: {all_tx_df['isFraud'].mean():.3%}).")

    # 4. Join transactions with source and destination account attributes
    # Rename columns to avoid collision
    src_accounts = accounts_df.rename(
        columns={
            "account_id": "src_id",
            "balance": "src_balance",
            "risk_score": "src_risk_score",
            "customer_name": "src_customer_name",
            "creation_date": "src_creation_date",
        }
    )
    dst_accounts = accounts_df.rename(
        columns={
            "account_id": "dst_id",
            "balance": "dst_balance",
            "risk_score": "dst_risk_score",
            "customer_name": "dst_customer_name",
            "creation_date": "dst_creation_date",
        }
    )

    merged = all_tx_df.merge(src_accounts, on="src_id", how="left")
    merged = merged.merge(dst_accounts, on="dst_id", how="left")

    if verbose:
        print(f"[load] Final merged transactions shape: {merged.shape}")

    return merged


def load_test(data_dir: Path | str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Mock for test dataset, returns empty DataFrame with same columns as train."""
    train_df = load_train(data_dir=data_dir, verbose=False)
    return pd.DataFrame(columns=train_df.columns)
