# ADR-0001 — Graph Feature Engineering for XGBoost/LightGBM Baseline

**Status:** Accepted  
**Date:** 2026-08-11  
**Authors:** Victor (TCC)  
**Supersedes:** —  
**Superseded by:** —

---

## Context

### Problem Statement

Credit-card and e-commerce fraud is a highly adversarial, class-imbalanced classification problem. The IEEE-CIS Fraud Detection dataset (Kaggle, 2019) contains ~590 k transactions across 433 features, with a positive-fraud rate of roughly **3.5 %**. State-of-the-art tabular classifiers (XGBoost, LightGBM) already exploit the rich identity and behavioural signals in this dataset. The central research question of this TCC is:

> **Does enriching a gradient-boosted baseline with graph-derived features (node centrality, community membership, Node2Vec embeddings) produce a statistically significant lift in PR-AUC?**

To answer this cleanly we need an experimental design that holds the classification engine constant while varying only the feature set. We also need to justify why we chose this design over a full end-to-end Graph Neural Network approach.

### Dataset Characteristics Relevant to This Decision

| Property | Value |
|---|---|
| Total transactions | ~590,540 |
| Fraud prevalence | ~3.5 % |
| Identity table join rate | ~46 % of transactions have identity info |
| Temporal span | ~6 months (split required to respect time ordering) |
| Feature types | Numerical, ordinal-encoded categoricals, anonymised Vxxx/Cxxx features |

The strong class imbalance makes **PR-AUC** the primary evaluation metric: it penalises false negatives while acknowledging the cost of false positives, unlike ROC-AUC which is insensitive to class imbalance. A lift in PR-AUC of ≥ 0.01 (absolute) is considered a meaningful improvement for this dataset scale.

### Graph Construction Rationale

Transactions in the IEEE-CIS dataset are connected through shared entities: the same card number, email domain, device fingerprint, IP address, or billing address can appear in multiple transactions. These _transaction relations_ form a bipartite network between **transaction nodes** and **entity nodes** (cards, emails, IPs, devices, addresses). This network encodes behavioural patterns invisible to tabular models:

- A card involved in rapid successive small transactions across many merchants signals a card-testing ring.
- A cluster of transactions sharing an email domain with high fraud prevalence marks a synthetic-identity cluster.
- High-PageRank entity nodes act as fraud hubs that route large volumes of fraudulent transactions.

### Candidate Approaches Evaluated

#### Approach A — Tabular-Only Baseline (Pure XGBoost/LightGBM)

Train gradient-boosted trees directly on the 433 raw features without any graph information.

| Dimension | Assessment |
|---|---|
| **PR-AUC** | Strong — GBDT on IEEE-CIS routinely achieves 0.88–0.92 on public leaderboards |
| **Interpretability** | High — SHAP values, gain-based importances |
| **Implementation effort** | Low — scikit-learn compatible API |
| **Graph signal** | ❌ None — entity co-occurrence relationships ignored |
| **Deployment** | Straightforward — serialised model + feature pipeline |

This is our **control condition**.

#### Approach B — Graph Feature Engineering → XGBoost/LightGBM

Extract structural graph metrics from the transaction-entity bipartite graph and append them as new columns to the tabular feature matrix. The classifier remains a gradient-boosted model.

**Graph features planned:**

| Feature | Graph Algorithm | Captures |
|---|---|---|
| `entity_degree` | Node degree (bipartite) | Transaction volume through an entity |
| `entity_fraud_rate` | Neighbourhood aggregation | Fraud density around an entity |
| `pagerank_score` | PageRank (NetworkX) | Global hub-ness of an entity |
| `clustering_coeff` | Local clustering coefficient | Local triangles — synthetic-identity rings |
| `betweenness_centrality` | Betweenness centrality | Bridge entities between fraud clusters |
| `community_id` | Louvain / Label Propagation | Community membership |
| `node2vec_emb_0..31` | Node2Vec (32-dim walk-based) | Latent structural embedding |

The final feature matrix is: `[tabular_features ‖ graph_features]`, fed to the same GBDT.

| Dimension | Assessment |
|---|---|
| **PR-AUC** | Expected lift ≥ 0.01–0.03 (hypothesis to be validated) |
| **Interpretability** | High — SHAP still applicable; graph features individually explainable |
| **Implementation effort** | Medium — NetworkX pipeline, offline extraction |
| **Graph signal** | ✅ Structural + behavioural signal captured |
| **Deployment** | Medium — graph extraction must be integrated into inference pipeline |

This is our **treatment condition**.

#### Approach C — End-to-End Graph Neural Network (GraphSAGE / GAT)

Train a GNN directly on the transaction graph, learning node representations end-to-end.

| Dimension | Assessment |
|---|---|
| **PR-AUC** | Potentially higher, but highly sensitive to graph construction choices |
| **Interpretability** | Low — black-box message-passing layers; GNNExplainer adds overhead |
| **Implementation effort** | High — requires PyTorch Geometric / DGL, custom mini-batch sampling |
| **Inductive generalisation** | Harder — transductive GNNs cannot score unseen nodes without re-running the graph |
| **Real-time deployment** | Very hard — requires a live graph store + inference server |
| **Research risk** | High — graph construction, sampling strategy, and hyperparameters all interact |

GNNs conflate _two_ variables: the graph construction quality and the classification architecture. This makes it difficult to isolate whether a performance difference stems from graph signal or from the neural model capacity. For a TCC-scope project with a single researcher, this risk is unacceptable.

### Why Not Approach C?

The goal of this TCC is to **prove the value of graph-derived information** for fraud detection, not to propose a new GNN architecture. The cleanest experimental design to achieve this is:

```
Control:   P(fraud | tabular_features)              → XGBoost
Treatment: P(fraud | tabular_features + graph_feats) → XGBoost
```

This design holds the classifier constant, making any PR-AUC difference attributable solely to the graph features. A GNN bundles feature extraction and classification into a single model, making attribution impossible.

---

## Decision

We choose **Approach B — Graph Feature Engineering coupled with XGBoost/LightGBM** as the primary experimental strategy.

### Rationale Summary

1. **Isolability**: By keeping the GBDT classifier constant, any PR-AUC lift is directly attributable to graph-derived features. This gives the TCC a clean causal claim.
2. **Interpretability**: SHAP values can quantify the individual contribution of each graph feature (e.g., `pagerank_score`, `community_id`), supporting explainability requirements common in financial anti-fraud systems.
3. **Implementation tractability**: Within TCC scope, NetworkX + Node2Vec can be implemented and validated in a single research cycle without the engineering overhead of PyTorch Geometric.
4. **Production proximity**: Gradient-boosted models are the industry standard for real-time scoring (millisecond latency, serialisable). This decision keeps the deployment story realistic.
5. **Incremental path to Neo4j**: Graph features extracted offline via NetworkX can later be migrated to Neo4j GDS (see ADR-0002), enabling online feature computation without changing the classifier.

### Experimental Protocol

| Phase | Action | Metric |
|---|---|---|
| 1. Baseline | Train XGBoost on raw 433 tabular features | PR-AUC, ROC-AUC, F1@threshold |
| 2. Graph construction | Build bipartite graph from entity columns; compute centrality features | Graph stats (density, diameter) |
| 3. Node2Vec embeddings | Train Node2Vec; append 32-dim embeddings | Embedding quality (t-SNE visual) |
| 4. Augmented model | Train XGBoost on tabular + graph features | PR-AUC Δ vs. baseline |
| 5. Ablation | Train with each graph feature group independently | Feature group contribution |
| 6. SHAP analysis | Explain augmented model predictions | Ranked feature importances |

Temporal split: training on first 5 months, evaluation on last month (preserves real-world temporal leakage constraints).

### Accepted Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Graph features add noise (negative lift) | Medium | Ablation study; regularisation via `min_child_weight` / `feature_fraction` |
| Node2Vec training time prohibitive | Low–Medium | Limit walk length to 80 steps; use `node2vec` PyPI package with parallel workers |
| Betweenness centrality O(V·E) cost | Medium | Approximate via `k`-sample betweenness (NetworkX `betweenness_centrality_subset`) |
| Data leakage through graph construction | Medium | Graph is built only from training split entities; test nodes use look-up, not re-computation |

---

## Consequences

### Positive

- Clear, auditable comparison between tabular-only and graph-augmented performance.
- All graph features are human-interpretable (degree = volume, PageRank = hub-ness, community = cluster).
- SHAP-based explanations align with regulatory expectations for financial AI models.
- Pipeline is modular: graph extraction is a separate stage and can be swapped for Neo4j GDS in Phase 2.

### Negative

- Graph features are computed offline — they do not capture real-time relationship updates during the test window.
- Node2Vec embeddings are not inherently interpretable; they require t-SNE/UMAP visualisation for qualitative validation.
- Extending to streaming/online inference requires a live graph store (addressed in ADR-0002).

### Neutral / Open Questions

- The specific entity columns to use for bipartite graph construction (card1, card2, addr1, P_emaildomain, DeviceInfo, …) will be finalised during Dataset Profiling.
- Whether to use Louvain or Label Propagation for community detection is deferred to experimentation.

---

## References

- [IEEE-CIS Fraud Detection — Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection)
- Grover, A. & Leskovec, J. (2016). _node2vec: Scalable Feature Learning for Networks_. KDD 2016.
- Page, L. et al. (1999). _The PageRank Citation Ranking: Bringing Order to the Web_. Stanford Technical Report.
- Chen, T. & Guestrin, C. (2016). _XGBoost: A Scalable Tree Boosting System_. KDD 2016.
- Davis, J. & Goadrich, M. (2006). _The Relationship Between Precision-Recall and ROC Curves_. ICML 2006.
- Hamilton, W. et al. (2017). _Inductive Representation Learning on Large Graphs_ (GraphSAGE). NeurIPS 2017.
