# ADR-0003 — Evaluation Protocol and Model Comparison Methodology

**Status:** Accepted  
**Date:** 2026-08-11  
**Authors:** Victor (TCC)  
**Supersedes:** —  
**Superseded by:** —

---

## Context

### The Central Research Question

> **Does augmenting a gradient-boosted baseline with graph-derived features produce a statistically meaningful improvement in fraud detection performance, as measured by PR-AUC, on the IEEE-CIS dataset?**

To answer this question rigorously in a TCC context, we must specify:

1. Which **models** are being compared.
2. Which **metrics** govern the comparison.
3. How data is **split** to prevent leakage.
4. How **statistical significance** of differences will be assessed.
5. What constitutes a **meaningful improvement**.

This ADR records those decisions so that the experimental methodology is reproducible and auditable.

---

## Model Comparison Matrix

The experiment compares four configurations in a **controlled factorial design**:

| Model ID | Classifier | Feature Set | Role |
|---|---|---|---|
| `M1` | XGBoost | Tabular only (433 features) | Baseline / Control |
| `M2` | LightGBM | Tabular only (433 features) | Baseline variant |
| `M3` | XGBoost | Tabular + Graph features | Treatment |
| `M4` | LightGBM | Tabular + Graph features | Treatment variant |

**Rationale for including both XGBoost and LightGBM:**

The lift from graph features should be robust across GBDT implementations. If only XGBoost shows lift but LightGBM does not (or vice versa), this indicates a potential interaction between the feature set and the specific boosting algorithm, which would itself be a significant finding. Including both eliminates this confounder.

### Feature Set Definitions

#### Tabular Feature Set (M1, M2 baseline)

All 433 features from the IEEE-CIS dataset as provided:
- `TransactionAmt` — transaction amount
- `ProductCD` — product category
- `card1–card6` — payment card attributes
- `addr1, addr2` — billing address codes
- `dist1, dist2` — distance features
- `P_emaildomain, R_emaildomain` — payer/recipient email domains
- `C1–C14` — counting features (anonymised)
- `D1–D15` — time-delta features
- `M1–M9` — match features (boolean)
- `V1–V339` — Vesta-engineered features (anonymised)
- Identity features: `id_01–id_38`, `DeviceType`, `DeviceInfo`

No feature selection is applied at baseline — the full feature set is used with GBDT's built-in feature importance pruning via regularisation.

#### Graph Feature Set (additional features in M3, M4)

| Feature Name | Type | Algorithm | Graph Layer |
|---|---|---|---|
| `card_degree` | Integer | Degree centrality | Card–Transaction bipartite |
| `email_degree` | Integer | Degree centrality | Email–Transaction bipartite |
| `device_degree` | Integer | Degree centrality | Device–Transaction bipartite |
| `card_fraud_rate` | Float [0,1] | Neighbourhood aggregation | Card–Transaction bipartite |
| `email_fraud_rate` | Float [0,1] | Neighbourhood aggregation | Email–Transaction bipartite |
| `card_pagerank` | Float | PageRank (α=0.85, ε=1e-6) | Projected card graph |
| `entity_betweenness` | Float | Approx. betweenness (k=200) | Full entity graph |
| `local_clustering` | Float [0,1] | Clustering coefficient | Projected transaction graph |
| `community_id` | Integer | Louvain community detection | Full entity graph |
| `community_fraud_rate` | Float [0,1] | Neighbourhood aggregation | Community level |
| `n2v_0` … `n2v_31` | Float (×32) | Node2Vec (p=1, q=1, d=32) | Full entity graph |

Total graph features: **42** (10 structural/aggregation + 32 Node2Vec dimensions).

**Total augmented feature count:** 433 + 42 = **475 features**.

---

## Evaluation Metrics

### Primary Metric: PR-AUC (Precision-Recall Area Under the Curve)

PR-AUC is selected as the **primary metric** because:

1. **Class imbalance robustness**: With ~3.5 % fraud prevalence, ROC-AUC can be misleadingly optimistic (a model predicting 0% fraud achieves ROC-AUC ≈ 0.5 but PR-AUC ≈ 0). PR-AUC reflects performance specifically on the positive (fraud) class.
2. **Business relevance**: In fraud detection, precision (investigation cost) and recall (fraud missed) are the direct business levers. PR-AUC integrates both across all operating thresholds.
3. **Kaggle alignment**: The IEEE-CIS Kaggle competition used ROC-AUC, but for the TCC's research question — where the class imbalance is the central challenge — PR-AUC is more informative.

### Secondary Metrics

| Metric | Purpose |
|---|---|
| **ROC-AUC** | Comparability with published Kaggle results |
| **F1-score @0.5 threshold** | Harmonic mean at default decision boundary |
| **Precision @k%** | Precision when top-k% of scores are flagged (k=1,5,10) |
| **Recall @FPR=1%** | Fraud recall at a business-acceptable false-positive rate |
| **Training time (s)** | Engineering cost of graph features |
| **Inference latency (ms)** | Operational feasibility for real-time scoring |

### What Constitutes a Meaningful Improvement?

Following the practitioner convention for fraud detection:

- **Δ PR-AUC ≥ 0.005 (absolute)** — technically significant, worth noting.
- **Δ PR-AUC ≥ 0.010 (absolute)** — practically significant, supports the thesis claim.
- **Δ PR-AUC ≥ 0.020 (absolute)** — strong support; comparable to published graph-augmented results in literature.

---

## Data Splitting Strategy

### Temporal Split (Primary)

The dataset spans 6 months. A **temporal split** is mandatory because:

- Fraud patterns evolve over time; random splits would create temporal leakage.
- Real-world anti-fraud models are retrained on historical data and scored on future transactions.

```
|──────── Training (months 1–5) ─────────|── Validation (month 6) ──|
  ~490,000 transactions                     ~100,000 transactions
```

No data from the validation period is used for:
- Feature engineering (graph construction uses training graph only).
- Hyperparameter tuning (all tuning on training split via time-series cross-validation).
- Threshold calibration.

### Graph Construction Boundary

**Critical**: The bipartite graph is constructed **exclusively from training transactions**. For validation transactions, graph features are computed as:

1. **Known entity** (card/email/device seen in training): Look up pre-computed centrality value.
2. **Unknown entity** (not seen in training): Assign imputed value (mean of training distribution for that feature).

This simulates real-world inference where a new transaction from a never-seen card still receives graph features (via imputation), mirroring production behaviour.

### Cross-Validation (Hyperparameter Tuning Only)

Within the training split, hyperparameters are tuned using **time-series cross-validation** (sklearn `TimeSeriesSplit`, n_splits=5):

```
Split 1: train [T0, T1] → val [T2]
Split 2: train [T0, T2] → val [T3]
Split 3: train [T0, T3] → val [T4]
Split 4: train [T0, T4] → val [T5]
Split 5: train [T0, T5] → val [T6]
```

The temporal ordering is preserved in every fold. Random k-fold cross-validation is **explicitly prohibited** to prevent temporal leakage.

---

## Hyperparameter Tuning Protocol

Hyperparameters are tuned independently for each model configuration (M1–M4):

| Parameter | Search Space | Method |
|---|---|---|
| `n_estimators` | [200, 2000] | Bayesian optimisation (Optuna) |
| `max_depth` | [3, 10] | Bayesian optimisation |
| `learning_rate` | [0.01, 0.3] (log scale) | Bayesian optimisation |
| `subsample` | [0.6, 1.0] | Bayesian optimisation |
| `colsample_bytree` | [0.6, 1.0] | Bayesian optimisation |
| `min_child_weight` | [1, 50] | Bayesian optimisation |
| `scale_pos_weight` | `(n_negative / n_positive)` | Fixed (class imbalance correction) |

**Budget**: 50 Optuna trials per model configuration, maximising mean PR-AUC across CV folds.

---

## Ablation Study Design

To decompose which graph features are most valuable, an ablation study is conducted after training M3/M4:

| Ablation Group | Features Removed | Purpose |
|---|---|---|
| `M3-no-degree` | All `*_degree` features | Isolate degree contribution |
| `M3-no-fraud-rate` | All `*_fraud_rate` features | Isolate fraud rate contribution |
| `M3-no-pagerank` | `card_pagerank`, `entity_betweenness` | Isolate centrality contribution |
| `M3-no-community` | `community_id`, `community_fraud_rate` | Isolate community contribution |
| `M3-no-n2v` | `n2v_0`…`n2v_31` | Isolate Node2Vec contribution |
| `M3-struct-only` | All features except N2V | Structural features without embeddings |
| `M3-n2v-only` | All features except structural | Embeddings without structural features |

Each ablation group retrains with the same hyperparameters as M3. The ablation grid reveals which feature category drives the lift.

---

## Statistical Significance Assessment

Given that we report results on a single fixed test set (temporal hold-out), we use **bootstrap confidence intervals**:

- 10,000 bootstrap samples of the test set.
- Report 95 % CI for PR-AUC of each model.
- The lift Δ PR-AUC(M3 − M1) is considered statistically supported if the 95 % CI of the difference excludes zero.

Additionally, a **DeLong test** (for ROC-AUC) and a **permutation test** (for PR-AUC) will be computed to complement bootstrap results.

---

## Reproducibility Requirements

All experiments must be fully reproducible from a single command:

```bash
python main.py --seed 42 --model xgb --features tabular   # M1
python main.py --seed 42 --model lgbm --features tabular   # M2
python main.py --seed 42 --model xgb --features augmented  # M3
python main.py --seed 42 --model lgbm --features augmented # M4
```

Reproducibility constraints:
- Random seeds are fixed for: train/val split, GBDT training, Node2Vec random walks, Optuna sampler.
- Graph algorithm results (PageRank, clustering) are deterministic in NetworkX for a given graph construction.
- All results (PR-AUC, ROC-AUC, feature importances, SHAP values) are persisted to `results/` as JSON + CSV.

---

## Consequences

### Positive

- The 2×2 factorial design (classifier × feature set) separates the effect of the classifier architecture from the effect of graph features.
- The temporal split prevents inflated PR-AUC due to data leakage — a common failure mode in published fraud detection benchmarks.
- The ablation study provides actionable insight for the TCC discussion: not all graph features contribute equally, and the ablation reveals which representations matter.
- Bootstrap CIs provide an honest uncertainty estimate on a single test set without requiring multiple dataset splits.

### Negative

- Four full model trainings + ablation (8 additional) = 12 total training runs. With Optuna (50 trials × 5 folds each), this is computationally intensive. Estimated wall-clock time: 2–4 hours on a modern CPU.
- The single temporal test set means results may be sensitive to the specific fraud patterns in month 6. A rolling window evaluation would be more robust but is out of TCC scope.

### Neutral / Open Questions

- Whether to use `scale_pos_weight` (XGBoost) vs. `is_unbalance=True` (LightGBM) for class imbalance handling — to be validated in initial runs.
- Whether SHAP TreeExplainer can handle 475 features within reasonable time — may require background dataset sampling.

---

## References

- Davis, J. & Goadrich, M. (2006). _The Relationship Between Precision-Recall and ROC Curves_. ICML.
- Akiba, T. et al. (2019). _Optuna: A Next-generation Hyperparameter Optimization Framework_. KDD.
- Efron, B. & Hastie, T. (2016). _Computer Age Statistical Inference_. Cambridge University Press. [Bootstrap CI]
- DeLong, E. R. et al. (1988). _Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves_. Biometrics.
- Lundberg, S. & Lee, S-I. (2017). _A Unified Approach to Interpreting Model Predictions_ (SHAP). NeurIPS.
- Rao, X. et al. (2021). _xFraud: Explainable Fraud Transaction Detection_. VLDB — graph-augmented XGBoost in production fraud detection.
