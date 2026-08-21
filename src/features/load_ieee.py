"""
load_ieee.py — Load and merge Kaggle IEEE-CIS Fraud Detection data files.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

_DATA_IEEE_RAW = Path(__file__).resolve().parents[2] / "data" / "ieee" / "raw"

def load_ieee_train(
    data_dir: Path | str | None = None,
    sample_fraction: float | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load and merge train_transaction and train_identity.

    Parameters
    ----------
    data_dir:
        Custom directory containing the CSV files.
    sample_fraction:
        Fraction of transactions to load (0.0 to 1.0) for rapid testing.
    verbose:
        Whether to print status.

    Returns
    -------
    pd.DataFrame
        Merged training DataFrame.
    """
    data_path = Path(data_dir) if data_dir is not None else _DATA_IEEE_RAW
    
    tx_path = data_path / "train_transaction.csv"
    id_path = data_path / "train_identity.csv"
    
    if not tx_path.exists():
        raise FileNotFoundError(
            f"[load_ieee] Train transactions file not found at {tx_path}.\n"
            "Run: uv run python src/data/download_ieee.py"
        )
        
    if verbose:
        print(f"[load_ieee] Loading transactions from {tx_path}...")
    tx_df = pd.read_csv(tx_path)
    
    if id_path.exists():
        if verbose:
            print(f"[load_ieee] Loading identity from {id_path}...")
        id_df = pd.read_csv(id_path)
        
        if verbose:
            print("[load_ieee] Merging transactions and identity tables...")
        merged = tx_df.merge(id_df, on="TransactionID", how="left")
    else:
        if verbose:
            print(f"[load_ieee] Warning: Identity file not found at {id_path}. Using transactions only.")
        merged = tx_df

    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        if verbose:
            print(f"[load_ieee] Subsampling dataset to {sample_fraction:.1%}...")
        merged = merged.sample(frac=sample_fraction, random_state=42)
        # Re-sort by TransactionDT to ensure temporal split works
        merged = merged.sort_values("TransactionDT").reset_index(drop=True)

    if verbose:
        print(f"[load_ieee] Loaded IEEE-CIS data with shape: {merged.shape} (fraud rate: {merged['isFraud'].mean():.3%})")

    return merged
