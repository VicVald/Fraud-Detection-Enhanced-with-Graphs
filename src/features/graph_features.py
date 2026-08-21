"""
graph_features.py — Extract graph-structured features using NetworkX.

Computes:
- PageRank
- Degree (in-degree and out-degree)
- Neighborhood Fraud Rate
Features are fitted on the training split graph to prevent data leakage.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import numpy as np


def compute_graph_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build transaction graph from train_df, compute metrics, and map them to train_df and val_df.

    Parameters
    ----------
    train_df:
        Training transactions DataFrame.
    val_df:
        Validation transactions DataFrame.
    verbose:
        Whether to print status.

    Returns
    -------
    (train_augmented, val_augmented)
    """
    if verbose:
        print("\n--- Constructing Training Graph ---")

    # 1. Build directed graph from training transactions
    G = nx.DiGraph()
    for _, row in train_df.iterrows():
        G.add_edge(row["src_id"], row["dst_id"])

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    if verbose:
        print(f"Graph constructed with {num_nodes:,} nodes and {num_edges:,} edges.")

    # 2. Compute PageRank
    if verbose:
        print("Computing PageRank...")
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
    except Exception as e:
        print(f"Warning: PageRank computation failed, using default values. Error: {e}")
        pagerank = {node: 1.0 / max(num_nodes, 1) for node in G.nodes()}

    # 3. Compute Degrees
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    # 4. Compute Neighborhood Fraud Rate
    # Track fraud counts and total counts per account in train_df
    if verbose:
        print("Computing neighborhood fraud rates...")
    
    account_fraud_counts = {}
    account_total_counts = {}
    
    for _, row in train_df.iterrows():
        src = row["src_id"]
        dst = row["dst_id"]
        is_fraud = int(row["isFraud"])
        
        for acc in (src, dst):
            account_total_counts[acc] = account_total_counts.get(acc, 0) + 1
            account_fraud_counts[acc] = account_fraud_counts.get(acc, 0) + is_fraud

    account_fraud_rate = {
        acc: account_fraud_counts[acc] / account_total_counts[acc]
        for acc in account_total_counts
    }

    # 4.1 Compute Strongly Connected Component (SCC) sizes
    if verbose:
        print("Computing Strongly Connected Component (SCC) sizes...")
    scc_sizes = {}
    for scc in nx.strongly_connected_components(G):
        size = len(scc)
        for node in scc:
            scc_sizes[node] = float(size)

    # 4.2 Compute Flow Balance (ratio of net degree flow)
    if verbose:
        print("Computing degree flow ratios...")
    flow_ratios = {}
    for node in G.nodes():
        ind = in_degree.get(node, 0)
        outd = out_degree.get(node, 0)
        flow_ratios[node] = float(abs(ind - outd) / (ind + outd + 1e-5))

    # 4.3 Compute standard deviation of transaction amounts per account
    if verbose:
        print("Computing transaction amount standard deviations...")
    df_src = train_df[['src_id', 'amount']].rename(columns={'src_id': 'acc_id'})
    df_dst = train_df[['dst_id', 'amount']].rename(columns={'dst_id': 'acc_id'})
    df_all = pd.concat([df_src, df_dst], ignore_index=True)
    amount_std = df_all.groupby('acc_id')['amount'].std().fillna(0.0).to_dict()

    # 5. Map features back to Train and Validation DataFrames
    # Default values for unseen nodes (e.g. in validation set)
    default_pagerank = 1.0 / max(num_nodes, 1)
    default_degree = 0.0
    default_fraud_rate = 0.0
    default_scc_size = 1.0
    default_flow_ratio = 1.0
    default_amount_std = 0.0

    def get_features_for_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Source features
        df["src_pagerank"] = df["src_id"].map(pagerank).fillna(default_pagerank)
        df["src_in_degree"] = df["src_id"].map(in_degree).fillna(default_degree)
        df["src_out_degree"] = df["src_id"].map(out_degree).fillna(default_degree)
        df["src_neigh_fraud_rate"] = df["src_id"].map(account_fraud_rate).fillna(default_fraud_rate)
        df["src_scc_size"] = df["src_id"].map(scc_sizes).fillna(default_scc_size)
        df["src_flow_ratio"] = df["src_id"].map(flow_ratios).fillna(default_flow_ratio)
        df["src_amount_std"] = df["src_id"].map(amount_std).fillna(default_amount_std)

        # Destination features
        df["dst_pagerank"] = df["dst_id"].map(pagerank).fillna(default_pagerank)
        df["dst_in_degree"] = df["dst_id"].map(in_degree).fillna(default_degree)
        df["dst_out_degree"] = df["dst_id"].map(out_degree).fillna(default_degree)
        df["dst_neigh_fraud_rate"] = df["dst_id"].map(account_fraud_rate).fillna(default_fraud_rate)
        df["dst_scc_size"] = df["dst_id"].map(scc_sizes).fillna(default_scc_size)
        df["dst_flow_ratio"] = df["dst_id"].map(flow_ratios).fillna(default_flow_ratio)
        df["dst_amount_std"] = df["dst_id"].map(amount_std).fillna(default_amount_std)

        return df

    if verbose:
        print("Mapping graph features back to DataFrames...")
    train_aug = get_features_for_df(train_df)
    val_aug = get_features_for_df(val_df)

    if verbose:
        print("Graph feature enrichment complete.")

    return train_aug, val_aug
