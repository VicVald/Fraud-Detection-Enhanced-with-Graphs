# ADR-0005 — Integrating PyTorch Geometric GNN and Advanced Tabular Features

**Status:** Accepted  
**Date:** 2026-08-18  
**Authors:** Victor (TCC) & Antigravity  
**Supersedes:** —  
**Superseded by:** —

---

## Context

To further improve the fraud detection performance on the IEEE-CIS dataset and capture complex topological structures, we want to integrate Graph Neural Networks (GNNs). However, training GNNs in real-time or end-to-end classification mode introduces significant engineering trade-offs:
1. **Inference Latency**: Real-time GNN node classification requires fetching the k-hop neighborhood graph, which is computationally expensive.
2. **Hybrid GNN-GBDT Generalization**: GBDTs (XGBoost/LightGBM) still outperform pure GNNs on tabular data due to their ability to split on raw categorical and numerical features.
3. **Memory Overhead**: Representing the entire transaction graph with transactions as nodes is $O(E^2)$ and causes Out-Of-Memory (OOM) errors.

---

## Decision

We implement a **hybrid GNN-GBDT architecture** using **PyTorch Geometric (PyG)**:
1. **Graph Representation**: We construct a bipartite network where nodes are **User Profiles (UIDs)** and **Devices (DeviceInfo)**, and transactions represent directed edges between them.
2. **GNN Model**: We use a **GraphSAGE** (Sample and Aggregate) network to learn 16-dimensional dense embedding vectors for both User and Device nodes.
3. **Feature Enrichment**: For each transaction, we concatenate the learned embedding of its source `uid` and its `DeviceInfo` to produce a 32-dimensional transaction embedding. These embeddings are passed to the baseline GBDT classifiers (XGBoost/LightGBM) as regular features.
4. **Tabular PCA**: We apply Principal Component Analysis (PCA) to Vesta's collinear V-features (V1–V339), reducing them to 15 principal components to save memory and accelerate training.

---

## Consequences

- **System dependencies**: We add `torch` and `torch-geometric` to `pyproject.toml`.
- **Generalization**: GBDTs retain their ability to split on tabulations while gaining deep representation embeddings of the account/device neighborhood.
- **Hardware safety**: The GNN feature extractor is optimized to run on CPU and support sampling (`--sample`) for memory-constrained development environments.
