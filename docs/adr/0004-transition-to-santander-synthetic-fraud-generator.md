# ADR-0004 — Transition to Santander gen-fraud-graph for Synthetic Data

**Status:** Accepted  
**Date:** 2026-08-17  
**Authors:** Victor (TCC)  
**Supersedes:** ADR-0001 (partially, regarding dataset choice)  
**Superseded by:** —

---

## Context

We were previously using the Kaggle IEEE-CIS Fraud Detection dataset. However, in order to guarantee full control over dataset scale, avoid data licensing issues, and facilitate deep graph topology experiments, we decided to switch to Santander's synthetic data generator: **gen-fraud-graph**.

### Benefits of gen-fraud-graph
1. **Dynamic Scaling**: Scale factor can adjust the number of accounts and transactions dynamically.
2. **Explicit Fraud Topology**: Fraud cases are explicitly generated as cycle patterns (money laundering rings) of configurable depths.
3. **No Private/PII Data**: Fully synthetic data means zero compliance or privacy leakage risk.
4. **Rich Features & Embeddings**: Built-in option to generate sentence embeddings for transaction descriptions.

---

## Decision

We replace the Kaggle IEEE-CIS dataset download/processing pipelines with a generation pipeline powered by Santander's `gen-fraud-graph`.

### Data Schema Mapping

| Entity | gen-fraud-graph Column | Description |
|---|---|---|
| **Account (Nodes)** | `account_id` | Unique account string identifier |
| | `balance` | Account balance |
| | `risk_score` | Tabular risk probability |
| **Transaction (Edges)** | `tx_id` | Unique transaction ID |
| | `src_id` | Source account |
| | `dst_id` | Destination account |
| | `amount` | Transaction amount |
| | `timestamp` | Time of transaction |
| | `description` | Text description |
| | `isFraud` | Target binary column (1 if from `transactions_fraud.csv`, else 0) |

---

## Consequences

- We can run local, end-to-end testing with a very small scale factor (e.g., `0.001` or `0.01`).
- Feature preprocessing must adjust to target tabular features (e.g., account balance, risk score, transaction amount) instead of the 433 Kaggle features.
- We implement NetworkX offline graph features:
  - Account node degrees
  - Account PageRank
  - Local clustering coefficients
  - Louvain communities
  - Neighborhood aggregation (historical fraud rate of source/destination accounts)
