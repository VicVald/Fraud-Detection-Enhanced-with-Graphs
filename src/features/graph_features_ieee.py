"""
graph_features_ieee.py — Bipartite transaction-entity graph features and Magic Feature engineering for IEEE-CIS.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import numpy as np


def make_uid(df: pd.DataFrame) -> pd.Series:
    """Construct composite UID representing a unique customer profile.
    
    Combines card1, card2, addr1, P_emaildomain and the normalized creation anchor (Day - D1).
    """
    c1 = df['card1'].fillna(0).astype(int).astype(str)
    c2 = df['card2'].fillna(0).astype(int).astype(str)
    addr = df['addr1'].fillna(0).astype(int).astype(str)
    email = df['P_emaildomain'].fillna('null').astype(str)
    
    # Normalized D1 anchor (approximate day of card/account creation)
    day = df['timestamp'] / 86400.0
    if 'D1' in df.columns:
        d1_anchor = (day - df['D1'].fillna(0)).astype(int).astype(str)
        return c1 + "_" + c2 + "_" + addr + "_" + email + "_" + d1_anchor
    
    return c1 + "_" + c2 + "_" + addr + "_" + email


def compute_rolling_count_group(group: pd.DataFrame) -> pd.Series:
    """Vectorized calculation of prior transactions in a sliding 24h window (86400s)."""
    times = group['timestamp'].values
    left_indices = np.searchsorted(times, times - 86400, side='left')
    counts = np.arange(len(times)) - left_indices + 1
    return pd.Series(counts, index=group.index, dtype=float)


def compute_graph_features_ieee(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute composite entity graph features, Magic normalized features, and temporal velocity.

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
        print("\n--- Computing IEEE-CIS Bipartite Graph Features & Magic Features ---")

    train_len = len(train_df)
    combined = pd.concat([train_df, val_df], ignore_index=True)
    
    # 1. Compute D Normalization (Time-consistent D features)
    day = combined['timestamp'] / 86400.0
    for d_col in ['D1', 'D2', 'D10', 'D15']:
        if d_col in combined.columns:
            combined[f'{d_col}n'] = day - combined[d_col]

    combined['uid'] = make_uid(combined)

    # 2. Compute Temporal Velocity Features (Rolling 24h)
    if verbose:
        print("Computing rolling 24h transaction counts (velocity)...")
    
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

    # Restore split DataFrames
    train_df_aug = combined.iloc[:train_len].copy()
    val_df_aug = combined.iloc[train_len:].copy()

    # 3. Compute Entity Degrees & Frequency Encodings on Training Split only
    if verbose:
        print("Computing entity degrees and frequency encodings on training split...")
    uid_degrees = train_df_aug['uid'].value_counts().to_dict()
    card1_counts = train_df_aug['card1'].value_counts().to_dict()
    email_degrees = train_df_aug['P_emaildomain'].value_counts().to_dict()
    device_degrees = train_df_aug['DeviceInfo'].value_counts().to_dict()

    # 4. Compute PageRank on Bipartite UID-Transaction Graph
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

    # 5. Behavior Deviation and Group Aggregations (fitted on Train only)
    if verbose:
        print("Computing Group Aggregations & deviations by UID...")

    # Group aggregations on key attributes: TransactionAmt, D15n, C1, C13
    agg_cols = ['TransactionAmt']
    if 'D15n' in train_df_aug.columns:
        agg_cols.append('D15n')
    if 'C1' in train_df_aug.columns:
        agg_cols.append('C1')
    if 'C13' in train_df_aug.columns:
        agg_cols.append('C13')

    group_stats = {}
    for col in agg_cols:
        group_stats[f'{col}_mean'] = train_df_aug.groupby('uid')[col].mean().to_dict()
        group_stats[f'{col}_std'] = train_df_aug.groupby('uid')[col].std().fillna(0.0).to_dict()

    default_degree = 0.0
    default_pagerank = 1.0 / max(G.number_of_nodes(), 1) if G.number_of_nodes() > 0 else 0.0

    for df_target in [train_df_aug, val_df_aug]:
        df_target["uid_degree"] = df_target["uid"].map(uid_degrees).fillna(default_degree)
        df_target["card1_count"] = df_target["card1"].map(card1_counts).fillna(default_degree)
        df_target["email_degree"] = df_target["P_emaildomain"].map(email_degrees).fillna(default_degree)
        df_target["device_degree"] = df_target["DeviceInfo"].map(device_degrees).fillna(default_degree)
        df_target["uid_pagerank"] = df_target["uid"].map(uid_pagerank).fillna(default_pagerank)
        
        # Map group aggregations and compute deviations
        for col in agg_cols:
            mean_map = df_target["uid"].map(group_stats[f'{col}_mean']).fillna(train_df_aug[col].mean())
            std_map = df_target["uid"].map(group_stats[f'{col}_std']).fillna(0.0)
            df_target[f"{col}_to_mean_uid"] = df_target[col] / (mean_map + 1e-5)
            df_target[f"{col}_to_std_uid"] = (df_target[col] - mean_map) / (std_map + 1e-5)

    # Clean up temporary combined columns
    if "uid" in train_df_aug.columns:
        train_df_aug = train_df_aug.drop(columns=["uid"])
    if "uid" in val_df_aug.columns:
        val_df_aug = val_df_aug.drop(columns=["uid"])

    if verbose:
        print("IEEE-CIS Graph & Magic Feature enrichment complete.")

    return train_df_aug, val_df_aug

