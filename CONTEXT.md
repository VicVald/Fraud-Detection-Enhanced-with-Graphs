# TG-graph

A project dedicated to improving traditional machine learning systems for real-time fraud detection by enriching them with graph-structured features.

## Language

**Baseline Model**:
The traditional machine learning classifier (specifically XGBoost or LightGBM) trained solely on tabular account and transaction data, used as a benchmark for performance comparison.
_Avoid_: Tabular Classifier, Legacy Model

**Graph-Derived Features**:
Structural attributes extracted from the network representation of transactions (such as node degree, PageRank, bipartite projections, or Node2Vec embeddings) that are appended to the baseline model's features.
_Avoid_: Network Features, Graph Variables

**Transaction Relation**:
The connections between entities in the fraud dataset (specifically accounts) modeled as a graph where transactions represent directed edges between source and destination accounts.
_Avoid_: Transaction Link, Data Association

**PR-AUC**:
Precision-Recall Area Under the Curve, used as the primary evaluation metric due to the highly imbalanced nature of fraud datasets.
_Avoid_: ROC-AUC (as the primary metric), F1-Score (as the primary metric)

**Graph Database (Neo4j)**:
The persistent graph database system to be introduced after establishing a robust in-memory NetworkX model. Used to scale graph query capabilities and simulate a production-ready real-time anti-fraud system.
_Avoid_: Transaction DB, Relational DB

**Dataset Profiling**:
The initial systematic collection of statistical metrics (including class imbalance, distributions, null percentages, and temporal patterns) to map the domain characteristics before feature engineering begins.
_Avoid_: Data Summary, EDA (when referring specifically to the metrics report)

