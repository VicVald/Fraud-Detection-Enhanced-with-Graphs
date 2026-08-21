"""
gnn_features.py — Bipartite GraphSAGE GNN feature extractor using PyTorch Geometric.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path

# Lazy import helpers to prevent importing torch_geometric at the module level
# (prevents errors if dependencies are still installing or not available)
def get_pyg_imports():
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv, to_hetero
    import torch_geometric.transforms as T
    return HeteroData, SAGEConv, to_hetero, T

class GNN(nn.Module):
    def __init__(self, hidden_channels: int, SAGEConv_class):
        super().__init__()
        self.conv1 = SAGEConv_class((-1, -1), hidden_channels)
        self.conv2 = SAGEConv_class((-1, -1), hidden_channels)
        
    def forward(self, x: dict, edge_index: dict) -> dict:
        x_dict = {}
        for node_type, x_val in x.items():
            # First layer
            h = self.conv1(x_val, edge_index)
            h = F.relu(h)
            # Second layer
            h = self.conv2(h, edge_index)
            x_dict[node_type] = h
        return x_dict

class EdgeClassifier(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        
    def forward(self, z_user: torch.Tensor, z_device: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        col, row = edge_index
        # Concatenate user and device embeddings for each edge
        x = torch.cat([z_user[col], z_device[row]], dim=-1)
        x = self.lin1(x).relu()
        return self.lin2(x).squeeze(-1)

def make_uid(df: pd.DataFrame) -> pd.Series:
    """Helper to generate composite UID matching graph_features_ieee.py."""
    c1 = df['card1'].fillna(0).astype(int).astype(str)
    c2 = df['card2'].fillna(0).astype(int).astype(str)
    addr = df['addr1'].fillna(0).astype(int).astype(str)
    email = df['P_emaildomain'].fillna('null').astype(str)
    return c1 + "_" + c2 + "_" + addr + "_" + email

def compute_gnn_embeddings(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 15,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a bipartite GraphSAGE GNN on training edges and output 32-d transaction embeddings.

    Parameters
    ----------
    train_df, val_df:
        Train and Validation DataFrames.
    epochs:
        Number of GNN training epochs. Default 15.
    verbose:
        Whether to print status.

    Returns
    -------
    (train_augmented, val_augmented)
    """
    if verbose:
        print("\n--- Training PyTorch Geometric Bipartite GraphSAGE ---")

    # Load PyG dependencies
    HeteroData, SAGEConv, to_hetero, T = get_pyg_imports()

    train_df = train_df.copy()
    val_df = val_df.copy()

    # Generate UIDs
    train_df['uid'] = make_uid(train_df)
    val_df['uid'] = make_uid(val_df)
    
    # Fill null device infos
    train_df['DeviceInfo_clean'] = train_df['DeviceInfo'].fillna('null_device').astype(str)
    val_df['DeviceInfo_clean'] = val_df['DeviceInfo'].fillna('null_device').astype(str)

    # Collect unique nodes
    unique_uids = sorted(list(set(train_df['uid']) | set(val_df['uid'])))
    unique_devices = sorted(list(set(train_df['DeviceInfo_clean']) | set(val_df['DeviceInfo_clean'])))

    uid_map = {uid: i for i, uid in enumerate(unique_uids)}
    device_map = {dev: i for i, dev in enumerate(unique_devices)}

    # Construct HeteroData
    data = HeteroData()
    
    # Node features (constant ones)
    num_users = len(unique_uids)
    num_devices = len(unique_devices)
    data['user'].x = torch.ones((num_users, 16), dtype=torch.float)
    data['device'].x = torch.ones((num_devices, 16), dtype=torch.float)

    # Training Edges
    train_users_idx = train_df['uid'].map(uid_map).values
    train_devices_idx = train_df['DeviceInfo_clean'].map(device_map).values
    
    edge_index = torch.tensor([train_users_idx, train_devices_idx], dtype=torch.long)
    edge_label = torch.tensor(train_df['isFraud'].values, dtype=torch.float)

    data['user', 'transacts', 'device'].edge_index = edge_index
    data['user', 'transacts', 'device'].edge_label = edge_label

    # Apply ToUndirected transform to add ('device', 'rev_transacts', 'user') edges
    data = T.ToUndirected()(data)

    # Initialize GNN and Edge Classifier
    hidden_channels = 16
    
    # We define a base GNN and wrap it to support heterogeneous nodes
    class HeteroGNN(nn.Module):
        def __init__(self):
            super().__init__()
            # SageConv needs to receive a tuple of dimensions for bipartite graphs
            self.conv1 = SAGEConv((-1, -1), hidden_channels)
            self.conv2 = SAGEConv((-1, -1), hidden_channels)
            
        def forward(self, x_dict, edge_index_dict):
            # First layer
            h_dict = {}
            # SAGEConv maps (source, target) to target node features
            # ('user', 'transacts', 'device') maps user to device, so it outputs device features
            h_dict['device'] = self.conv1(
                (x_dict['user'], x_dict['device']),
                edge_index_dict[('user', 'transacts', 'device')]
            ).relu()
            # ('device', 'rev_transacts', 'user') maps device to user, so it outputs user features
            h_dict['user'] = self.conv1(
                (x_dict['device'], x_dict['user']),
                edge_index_dict[('device', 'rev_transacts', 'user')]
            ).relu()
            
            # Second layer
            out_dict = {}
            out_dict['device'] = self.conv2(
                (h_dict['user'], h_dict['device']),
                edge_index_dict[('user', 'transacts', 'device')]
            )
            out_dict['user'] = self.conv2(
                (h_dict['device'], h_dict['user']),
                edge_index_dict[('device', 'rev_transacts', 'user')]
            )
            return out_dict

    gnn = HeteroGNN()
    classifier = EdgeClassifier(hidden_channels)

    optimizer = torch.optim.Adam(list(gnn.parameters()) + list(classifier.parameters()), lr=0.01)

    # Training loop
    gnn.train()
    classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Message passing
        z = gnn(data.x_dict, data.edge_index_dict)
        # Classify edges
        y_pred = classifier(z['user'], z['device'], data['user', 'transacts', 'device'].edge_index)
        loss = F.binary_cross_entropy_with_logits(y_pred, data['user', 'transacts', 'device'].edge_label)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  GNN Epoch {epoch+1:02d}/{epochs:02d} | BCE Loss: {loss.item():.4f}")

    # Extract final node embeddings
    gnn.eval()
    with torch.no_grad():
        z = gnn(data.x_dict, data.edge_index_dict)
        
    user_emb = z['user'].numpy()
    device_emb = z['device'].numpy()

    # Map embeddings back to DataFrame as features GNN_0 to GNN_31
    if verbose:
        print("Mapping GNN embeddings to transactions...")
        
    def map_embeddings(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Get mapped indices
        u_idx = df['uid'].map(uid_map).values
        d_idx = df['DeviceInfo_clean'].map(device_map).values
        
        # Extracted vectors
        u_vectors = user_emb[u_idx] # [num_rows, 16]
        d_vectors = device_emb[d_idx] # [num_rows, 16]
        
        # Concatenate user and device embeddings
        concat_emb = np.concatenate([u_vectors, d_vectors], axis=1) # [num_rows, 32]
        
        # Assign to DataFrame
        for i in range(hidden_channels * 2):
            df[f"GNN_{i}"] = concat_emb[:, i]
            
        # Clean temporary columns
        df = df.drop(columns=["uid", "DeviceInfo_clean"])
        return df

    train_df_aug = map_embeddings(train_df)
    val_df_aug = map_embeddings(val_df)

    if verbose:
        print(f"GNN feature extraction complete. Added 32 GNN dimensions.")

    return train_df_aug, val_df_aug


def train_pure_gnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 30,
    verbose: bool = True,
) -> pd.Series:
    """Train a bipartite GraphSAGE GNN and classify validation edges directly.

    Returns
    -------
    y_score:
        pandas Series of predicted probabilities for the validation dataset,
        aligned by validation index.
    """
    if verbose:
        print("\n--- Training Pure PyTorch Geometric Bipartite GraphSAGE (Edge Classification) ---")

    # Load PyG dependencies
    HeteroData, SAGEConv, to_hetero, T = get_pyg_imports()

    train_df = train_df.copy()
    val_df = val_df.copy()

    # Generate UIDs
    train_df['uid'] = make_uid(train_df)
    val_df['uid'] = make_uid(val_df)
    
    # Fill null device infos
    train_df['DeviceInfo_clean'] = train_df['DeviceInfo'].fillna('null_device').astype(str)
    val_df['DeviceInfo_clean'] = val_df['DeviceInfo'].fillna('null_device').astype(str)

    # Collect unique nodes
    unique_uids = sorted(list(set(train_df['uid']) | set(val_df['uid'])))
    unique_devices = sorted(list(set(train_df['DeviceInfo_clean']) | set(val_df['DeviceInfo_clean'])))

    uid_map = {uid: i for i, uid in enumerate(unique_uids)}
    device_map = {dev: i for i, dev in enumerate(unique_devices)}

    # Construct HeteroData
    data = HeteroData()
    
    num_users = len(unique_uids)
    num_devices = len(unique_devices)
    data['user'].x = torch.ones((num_users, 16), dtype=torch.float)
    data['device'].x = torch.ones((num_devices, 16), dtype=torch.float)

    # Training Edges
    train_users_idx = train_df['uid'].map(uid_map).values
    train_devices_idx = train_df['DeviceInfo_clean'].map(device_map).values
    
    edge_index = torch.tensor([train_users_idx, train_devices_idx], dtype=torch.long)
    edge_label = torch.tensor(train_df['isFraud'].values, dtype=torch.float)

    data['user', 'transacts', 'device'].edge_index = edge_index
    data['user', 'transacts', 'device'].edge_label = edge_label

    # Apply ToUndirected transform
    data = T.ToUndirected()(data)

    # Initialize GNN and Edge Classifier
    hidden_channels = 16
    
    class HeteroGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), hidden_channels)
            self.conv2 = SAGEConv((-1, -1), hidden_channels)
            
        def forward(self, x_dict, edge_index_dict):
            h_dict = {}
            # SAGEConv maps (source, target) to target node features
            h_dict['device'] = self.conv1(
                (x_dict['user'], x_dict['device']),
                edge_index_dict[('user', 'transacts', 'device')]
            ).relu()
            h_dict['user'] = self.conv1(
                (x_dict['device'], x_dict['user']),
                edge_index_dict[('device', 'rev_transacts', 'user')]
            ).relu()
            
            out_dict = {}
            out_dict['device'] = self.conv2(
                (h_dict['user'], h_dict['device']),
                edge_index_dict[('user', 'transacts', 'device')]
            )
            out_dict['user'] = self.conv2(
                (h_dict['device'], h_dict['user']),
                edge_index_dict[('device', 'rev_transacts', 'user')]
            )
            return out_dict

    gnn = HeteroGNN()
    classifier = EdgeClassifier(hidden_channels)

    optimizer = torch.optim.Adam(list(gnn.parameters()) + list(classifier.parameters()), lr=0.01)

    # Training loop
    gnn.train()
    classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = gnn(data.x_dict, data.edge_index_dict)
        y_pred = classifier(z['user'], z['device'], data['user', 'transacts', 'device'].edge_index)
        loss = F.binary_cross_entropy_with_logits(y_pred, data['user', 'transacts', 'device'].edge_label)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Pure GNN Epoch {epoch+1:02d}/{epochs:02d} | BCE Loss: {loss.item():.4f}")

    # Validation predictions
    gnn.eval()
    classifier.eval()
    
    val_users_idx = val_df['uid'].map(uid_map).values
    val_devices_idx = val_df['DeviceInfo_clean'].map(device_map).values
    val_edge_index = torch.tensor([val_users_idx, val_devices_idx], dtype=torch.long)
    
    with torch.no_grad():
        z = gnn(data.x_dict, data.edge_index_dict)
        y_score_logits = classifier(z['user'], z['device'], val_edge_index)
        y_score = torch.sigmoid(y_score_logits).numpy()
        
    return pd.Series(y_score, index=val_df.index)

