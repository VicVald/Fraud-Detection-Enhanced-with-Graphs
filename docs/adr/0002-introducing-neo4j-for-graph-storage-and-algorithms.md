# ADR-0002 — Introducing Neo4j for Graph Storage and Real-Time Algorithm Execution

**Status:** Accepted  
**Date:** 2026-08-11  
**Authors:** Victor (TCC)  
**Supersedes:** —  
**Superseded by:** —

---

## Context

### Phase 1 Limitations of NetworkX (In-Memory)

ADR-0001 established that graph-derived features will be computed offline and appended to the tabular feature matrix. In Phase 1, this computation runs in-memory using **NetworkX**. NetworkX is appropriate for the research prototype because:

- It provides the full suite of required algorithms (PageRank, betweenness centrality, clustering coefficient, Louvain community detection).
- It requires zero infrastructure overhead (pure Python dependency, already in `pyproject.toml`).
- It is sufficient for a static dataset of ~590 k transactions.

However, NetworkX in-memory is **not production-ready** for three reasons:

| Limitation | Impact |
|---|---|
| Entire graph must fit in RAM | Fails for multi-year transaction history or streaming datasets |
| No native indexing or query engine | Feature look-up for a new transaction requires full graph traversal |
| No persistence | Graph must be rebuilt from raw CSVs on every run (~minutes) |
| No real-time update | New transactions cannot be inserted without rebuilding the graph |

A secondary objective of this TCC is to simulate an architecture that is **plausible in a real production anti-fraud system**. Demonstrating that graph features can be computed from a persistent graph database closes this gap.

### Why a Graph Database, Not a Relational DB or Feature Store?

| Option | Pro | Con |
|---|---|---|
| Relational DB (PostgreSQL) | Mature, well-understood | JOINs for multi-hop paths are expensive; PageRank requires external tooling |
| Feature Store (Feast, Tecton) | Purpose-built for ML feature serving | Does not compute structural features; still needs a graph source |
| Graph Database (Neo4j) | Native graph storage, built-in GDS algorithms, Cypher query language | Operational overhead; requires a running server |

Neo4j is the leading open-source graph database with a dedicated **Graph Data Science (GDS)** library that natively implements PageRank, node similarity, community detection, and Node2Vec — exactly the algorithms used in Phase 1. Migrating from NetworkX-computed features to Neo4j GDS-computed features changes only the _computation backend_, not the feature definitions or the downstream classifier.

### Alternative Graph Databases Considered

| Database | License | GDS / Algorithm Library | Why Rejected |
|---|---|---|---|
| **Neo4j Community** | GPL v3 | Neo4j GDS (APOC + GDS plugin) | **Selected** |
| Amazon Neptune | Proprietary | Limited — no PageRank native | Cloud-only; no local dev |
| ArangoDB | Apache 2 / BSL | Pregel-based graph algorithms | Smaller ecosystem; less documentation for fraud use cases |
| TigerGraph | Proprietary | GSQL analytics | Requires paid licence for most features |
| JanusGraph | Apache 2 | None native | Requires separate Gremlin + Spark for analytics |

Neo4j Community Edition is free for local development and research, has the most mature GDS library, and has extensive academic and industry fraud-detection documentation.

---

## Decision

We will introduce **Neo4j Community Edition** (+ **Neo4j GDS library**) in **Phase 2** of the project, after Phase 1 has validated the performance lift from graph features using NetworkX.

### Phase Sequencing

```
Phase 1 (In-Memory Prototype — NetworkX)
 ├─ Build bipartite graph from raw CSV
 ├─ Compute: degree, PageRank, betweenness, clustering, community, Node2Vec
 ├─ Append graph features → XGBoost training
 └─ Validate PR-AUC lift (research question answered)

Phase 2 (Persistent Graph — Neo4j)
 ├─ Load transaction & identity data into Neo4j (Bolt + CSV import)
 ├─ Replicate graph algorithms using Neo4j GDS (CALL gds.pageRank.stream, ...)
 ├─ Expose feature look-up via Bolt queries (neo4j-driver)
 └─ Simulate real-time inference: new transaction → Cypher look-up → feature vector → GBDT score
```

### Data Model in Neo4j

The graph is modelled as a **bipartite heterogeneous graph**:

```cypher
// Nodes
(:Transaction {TransactionID, isFraud, TransactionAmt, ...})
(:Card        {card_id})          // derived from card1+card2 composite key
(:Email       {email_domain})
(:Device      {device_fingerprint})
(:IP          {addr1, addr2})

// Relationships
(:Transaction)-[:USES_CARD]->(:Card)
(:Transaction)-[:USES_EMAIL]->(:Email)
(:Transaction)-[:USES_DEVICE]->(:Device)
(:Transaction)-[:FROM_IP]->(:IP)
```

This schema enables Cypher queries like:

```cypher
// All transactions sharing a card with a known fraudulent transaction
MATCH (t:Transaction {isFraud: 1})-[:USES_CARD]->(c:Card)<-[:USES_CARD]-(t2:Transaction)
RETURN t2.TransactionID, count(*) AS shared_card_count
```

### Neo4j GDS Algorithm Mapping

| Phase 1 (NetworkX) | Phase 2 (Neo4j GDS) |
|---|---|
| `nx.pagerank(G)` | `CALL gds.pageRank.stream('myGraph')` |
| `nx.betweenness_centrality(G)` | `CALL gds.betweenness.stream('myGraph')` |
| `nx.clustering(G)` | `CALL gds.localClusteringCoefficient.stream('myGraph')` |
| `community_louvain.best_partition(G)` | `CALL gds.louvain.stream('myGraph')` |
| `node2vec.Node2Vec(G).fit()` | `CALL gds.node2vec.stream('myGraph', {dimensions: 32})` |

The Python inference layer uses the official `neo4j` driver:

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def get_graph_features(transaction_id: str) -> dict:
    with driver.session() as session:
        result = session.run(
            """
            MATCH (t:Transaction {TransactionID: $tid})
            RETURN t.pagerank AS pagerank,
                   t.community AS community,
                   t.degree AS degree
            """,
            tid=transaction_id
        )
        return result.single().data()
```

### Scope Boundary for TCC

Phase 2 is **exploratory and demonstrative**, not a full production deployment. Specifically:

- Neo4j will be run **locally** (Docker or native install) — no cloud deployment required.
- Feature parity with Phase 1 (same algorithm outputs) will be verified numerically before any re-training.
- The GBDT model weights are **not retrained** in Phase 2 — the focus is on showing that the same features can be served from a live graph database.
- Real-time inference simulation will use a small held-out sample, not a full streaming pipeline.

---

## Consequences

### Positive

- Demonstrates a **production-plausible architecture** for real-time graph feature serving in anti-fraud systems.
- Neo4j GDS algorithms are optimised for large graphs (parallel execution, native memory management) — closer to production-grade performance than NetworkX.
- Cypher queries support complex multi-hop pattern matching (e.g., fraud ring detection) beyond what is feasible with simple feature aggregation.
- The project gains a second infrastructure contribution suitable for TCC presentation.

### Negative

- Neo4j introduces operational overhead: a running database server, CSV bulk import, and driver dependency.
- GDS algorithm results may differ slightly from NetworkX due to implementation differences (e.g., convergence tolerance in PageRank) — requires numerical equivalence testing.
- Neo4j Community Edition does not support clustering or role-based access control — acceptable for a research prototype, not for production.

### Neutral / Open Questions

- Whether to use **Docker Compose** (preferred for reproducibility) or a **native Neo4j install** for local development.
- Whether the Node2Vec GDS implementation produces embeddings comparable to the `node2vec` PyPI package used in Phase 1 — to be validated empirically.
- The specific Cypher projection syntax (`gds.graph.project`) may differ between Neo4j GDS versions; the implementation should pin the GDS version.

---

## References

- [Neo4j Graph Data Science Library Documentation](https://neo4j.com/docs/graph-data-science/current/)
- [Neo4j GDS — PageRank](https://neo4j.com/docs/graph-data-science/current/algorithms/page-rank/)
- [Neo4j GDS — Node2Vec](https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/node2vec/)
- [Neo4j GDS — Louvain](https://neo4j.com/docs/graph-data-science/current/algorithms/louvain/)
- [IEEE-CIS Fraud Detection — Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection)
- Akoglu, L., Tong, H. & Koutra, D. (2015). _Graph-based Anomaly Detection and Description: A Survey_. DMKD.
