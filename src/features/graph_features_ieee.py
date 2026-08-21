"""
graph_features_ieee.py — Bipartite transaction-entity graph features for IEEE-CIS.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import numpy as np

def make_uid(df: pd.DataFrame) -> pd.Series:
    """Construct composite UID representing a unique customer profile."""
    c1 = df['card1'].fillna(0).astype(int).astype(str)
    c2 = df['card2'].fillna(0).astype(int).astype(str)
    addr = df['addr1'].fillna(0).astype(int).astype(str)
    email = df['P_emaildomain'].fillna('null').astype(str)
    return c1 + "_" + c2 + "_" + addr + "_" + email

def compute_rolling_count_group(group: pd.DataFrame) -> pd.Series:
    """Vectorized calculation of prior transactions in a sliding 24h window (86400s)."""
    times = group['timestamp'].values
    # For each transaction time t, find index of first transaction >= t - 86400
    left_indices = np.searchsorted(times, times - 86400, side='left')
    counts = np.arange(len(times)) - left_indices + 1
    return pd.Series(counts, index=group.index, dtype=float)

def compute_graph_features_ieee(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute composite entity graph features and temporal velocity on train_df and map to train_df/val_df.

    Parameters
    ----------
    train_df:
        Training transactions.
    val_df:
        Validation transactions.
    verbose:
        Whether to print status.

    Returns
    -------
    (train_augmented, val_augmented)
    """
    if verbose:
        print("\n--- Computing IEEE-CIS Bipartite Graph Features & Velocity ---")

    # Combine train and validation temporarily to compute rolling windows globally (prevent boundaries leakage)
    train_len = len(train_df)
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined['uid'] = make_uid(combined)

    # 1. Compute Temporal Velocity Features (Rolling 24h)
    if verbose:
        print("Computing rolling 24h transaction counts (velocity)...")
    
    # Sort combined by timestamp for rolling binary search
    combined = combined.sort_values('timestamp')
    
    # UID 24h Count
    uid_counts = combined.groupby('uid', group_keys=False).apply(compute_rolling_count_group)
    combined['uid_tx_count_24h'] = uid_counts
    
    # Device 24h Count (only for non-null DeviceInfo)
    combined['device_tx_count_24h'] = 1.0
    non_null_dev = combined[combined['DeviceInfo'].notnull()]
    if len(non_null_dev) > 0:
        dev_counts = non_null_dev.groupby('DeviceInfo', group_keys=False).apply(compute_rolling_count_group)
        combined.loc[dev_counts.index, 'device_tx_count_24h'] = dev_counts

    # Restore split DataFrames with the temporal velocity features mapped
    train_df_aug = combined.iloc[:train_len].copy()
    val_df_aug = combined.iloc[train_len:].copy()

    # 2. Compute Degrees (Frequencies) on Training Split only
    if verbose:
        print("Computing entity degrees...")
    uid_degrees = train_df_aug['uid'].value_counts().to_dict()
    email_degrees = train_df_aug['P_emaildomain'].value_counts().to_dict()
    device_degrees = train_df_aug['DeviceInfo'].value_counts().to_dict()

    # 3. Compute PageRank on Bipartite UID-Transaction Graph
    if verbose:
        print("Constructing UID-Transaction bipartite graph for PageRank...")
    G = nx.Graph()
    for _, row in train_df_aug.iterrows():
        tx_node = f"tx_{row['TransactionID']}"
        uid_node = f"uid_{row['uid']}"
        G.add_edge(tx_node, uid_node)

    if verbose:
        print(f"Bipartite graph has {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges. Computing PageRank...")
    
    try:
        pagerank_scores = nx.pagerank(G, alpha=0.85)
    except Exception as e:
        print(f"Warning: PageRank failed: {e}. Using default values.")
        pagerank_scores = {}

    uid_pagerank = {}
    for node, score in pagerank_scores.items():
        if node.startswith("uid_"):
            uid_val = node.replace("uid_", "")
            uid_pagerank[uid_val] = score

    # 4. Map features back to DataFrames
    default_degree = 0.0
    default_pagerank = 1.0 / max(G.number_of_nodes(), 1) if G.number_of_nodes() > 0 else 0.0

    # Behavior deviation calculations
    if verbose:
        print("Computing behavior deviation features (TransactionAmt relative to profile)...")
    uid_means = train_df_aug.groupby('uid')['TransactionAmt'].mean().to_dict()
    uid_stds = train_df_aug.groupby('uid')['TransactionAmt'].std().fillna(0.0).to_dict()
    
    default_mean = train_df_aug['TransactionAmt'].mean()
    default_std = 0.0

    # Train DF Mapping
    train_df_aug["uid_degree"] = train_df_aug["uid"].map(uid_degrees).fillna(default_degree)
    train_df_aug["email_degree"] = train_df_aug["P_emaildomain"].map(email_degrees).fillna(default_degree)
    train_df_aug["device_degree"] = train_df_aug["DeviceInfo"].map(device_degrees).fillna(default_degree)
    train_df_aug["uid_pagerank"] = train_df_aug["uid"].map(uid_pagerank).fillna(default_pagerank)
    
    train_df_aug["uid_amt_mean"] = train_df_aug["uid"].map(uid_means).fillna(default_mean)
    train_df_aug["uid_amt_std"] = train_df_aug["uid"].map(uid_stds).fillna(default_std)
    train_df_aug["TransactionAmt_to_mean_uid"] = train_df_aug["TransactionAmt"] / (train_df_aug["uid_amt_mean"] + 1e-5)
    train_df_aug["TransactionAmt_to_std_uid"] = train_df_aug["TransactionAmt"] / (train_df_aug["uid_amt_std"] + 1e-5)
    train_df_aug = train_df_aug.drop(columns=["uid_amt_mean", "uid_amt_std"])

    # Val DF Mapping
    val_df_aug["uid_degree"] = val_df_aug["uid"].map(uid_degrees).fillna(default_degree)
    val_df_aug["email_degree"] = val_df_aug["P_emaildomain"].map(email_degrees).fillna(default_degree)
    val_df_aug["device_degree"] = val_df_aug["DeviceInfo"].map(device_degrees).fillna(default_degree)
    val_df_aug["uid_pagerank"] = val_df_aug["uid"].map(uid_pagerank).fillna(default_pagerank)
    
    val_df_aug["uid_amt_mean"] = val_df_aug["uid"].map(uid_means).fillna(default_mean)
    val_df_aug["uid_amt_std"] = val_df_aug["uid"].map(uid_stds).fillna(default_std)
    val_df_aug["TransactionAmt_to_mean_uid"] = val_df_aug["TransactionAmt"] / (val_df_aug["uid_amt_mean"] + 1e-5)
    val_df_aug["TransactionAmt_to_std_uid"] = val_df_aug["TransactionAmt"] / (val_df_aug["uid_amt_std"] + 1e-5)
    val_df_aug = val_df_aug.drop(columns=["uid_amt_mean", "uid_amt_std"])

    # Clean up temporary combined columns we don't want as model features
    if "uid" in train_df_aug.columns:
        train_df_aug = train_df_aug.drop(columns=["uid"])
    if "uid" in val_df_aug.columns:
        val_df_aug = val_df_aug.drop(columns=["uid"])

    if verbose:
        print("IEEE-CIS Graph & Velocity feature enrichment complete.")

    return train_df_aug, val_df_aug
