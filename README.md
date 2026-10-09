# TG-graph — Engenharia de Grafos Bipartidos e Compressão Espectral na Detecção de Fraudes em E-Commerce

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: IEEE-CIS](https://img.shields.io/badge/Dataset-IEEE--CIS%20Fraud%20Detection-orange.svg)](https://www.kaggle.com/c/ieee-fraud-detection)

Este repositório contém o código-fonte, a suíte experimental de ablação, os notebooks e a monografia completa/artigo do Trabalho de Conclusão de Curso (TCC) intitulado:

> **"Engenharia de Grafos Bipartidos e Compressão Espectral na Detecção de Fraudes em E-Commerce: Um Estudo de Ablação e Decomposição Estrutural"**  
> **Autor:** Victor Valadares  
> **Área:** Inteligência Artificial, Ciência de Dados e Engenharia de Software  

---

## 📌 Documentos e Monografia

* 📄 **Monografia Completa / Artigo (PDF):** [`LOCAL_COMPLETE_PAPER.pdf`](LOCAL_COMPLETE_PAPER.pdf) *(3,13 MB, com diagramação acadêmica, fórmulas MathJax e figuras em alta resolução)*
* 📝 **Artigo em Markdown:** [`LOCAL_COMPLETE_PAPER.md`](LOCAL_COMPLETE_PAPER.md)
* 🌐 **Artigo em HTML Renderizado:** [`LOCAL_COMPLETE_PAPER.html`](LOCAL_COMPLETE_PAPER.html)
* 📊 **Relatório Metodológico dos Notebooks:** [`docs/RELATORIO_METODOLOGICO_NOTEBOOKS.md`](docs/RELATORIO_METODOLOGICO_NOTEBOOKS.md)

---

## 🔬 Visão Geral do Trabalho

Classificadores tabulares padrão baseados em *Gradient Boosted Decision Trees* (XGBoost/LightGBM) operam sob a premissa de que as transações são **Independentes e Identicamente Distribuídas (IID)**. No entanto, o cibercrime contemporâneo atua por meio de **ataques coordenados em rede** (*fraud rings*, *botnets* e fazendas de emuladores que reutilizam o mesmo hardware físico para testar múltiplos cartões roubados).

Este projeto desenvolve uma arquitetura híbrida de detecção de fraudes em esteiras de baixa latência, combinando:
1. **Resolução Determinística de Entidades (UID):** Ancoragem no calendário civil $\text{UID} = \text{card1} \;\Vert\; \text{addr1} \;\Vert\; \lfloor \text{timestamp}/86.400 - D1 \rfloor$, expandindo o espaço amostral de $12.730$ nós bancários artificiais para **$182.347$ portadores únicos reais** ($14,3\times$ em granularidade).
2. **Grafo Bipartido Portador $\leftrightarrow$ Dispositivo:** Formalizado como $\mathcal{G} = (\mathcal{U}, \mathcal{V}, \mathcal{E})$ com poda de nós sentinela nulos e projeção unipartida $\mathbf{A}_{\mathcal{U}} = \mathbf{B}\mathbf{B}^\top$, onde caminhos de comprimento 2 ($u_i \leftrightarrow v_k \leftrightarrow u_j$) identificam conluios de hardware compartilhado.
3. **Algoritmo de Velocidade Temporal em $\mathcal{O}(N \log K)$:** Contador de transações em janelas deslizantes de 24h via busca binária causal vetorizada (`np.searchsorted`), processando as **590.540 transações integrais em apenas 1,13 segundo**.
4. **Compressão Espectral PCA:** Redução das 339 variáveis colineares Vesta ($V$) para 15 componentes ortogonais, reduzindo o tempo de treino em **55,4%** (de 81,6s para 36,4s) e melhorando a generalização (ROC-AUC de 0,8641 para 0,8922 no modelo integrado).
5. **Roteamento Dual de Identidade (Model A / Model B):** Desacoplamento entre compras sem dados de hardware (Model A: fallback operacional com cobertura de 100%) e compras com identidade (Model B: especialista com métricas de PageRank e graus de rede).
6. **Matrizes Assimétricas e Risco Mínimo de Bayes:** Otimização financeira da regra de corte $\tau^*$ contra custos reais de *chargeback* ($C_{\text{chargeback}} = \text{US\$} 15,00$) e atrito de revisão manual ($C_{\text{revisão}} = \text{US\$} 3,00$).

---

## 🏛️ Arquitetura do Sistema

```
                      INGESTÃO INTEGRAL (N = 590.540)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       Treino Histórico (80%)                  Validação Temporal (20%)
    N = 472.432 (Dias 1 a 140)               N = 118.108 (Dias 141 a 182)
                 │                                       │
                 ├───────────────────────────────────────┤
                 ▼                                       ▼
     [Resolução de Entidades]                [Grafo Retrospectivo Estrito]
       UID Composto Calendário                  Poda de Nós Sentinela Nulos
       182.347 Portadores Reais                 Isolamento Anti-Topology Leakage
                 │                                       │
                 ├───────────────────────────────────────┤
                 ▼                                       ▼
     [Compressão Espectral PCA]              [Velocidade Causal O(N log K)]
       339 Vesta V ──> 15 PCA                  Janelas Deslizantes de 24h
       -55,4% Tempo de Treino                  Busca Binária Vetorizada
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                      ROTEAMENTO DUAL DE IDENTIDADE
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
     [Model A: Sem Identidade]               [Model B: Com Identidade]
      Fallback Transacional                   Especialista em Grafos
      Cobertura: 100% da População            PageRank & Graus de Hardware
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                          DECISÃO DE RISCO DE BAYES
                          Otimização de Limiar tau*
                          Maximização de Economia Liq.
```

---

## 📊 Resultados do Estudo de Ablação (100% dos Dados)

Experimentos conduzidos sobre a partição temporal do benchmark IEEE-CIS ($N_{\text{total}} = 590.540$; $N_{\text{val}} = 118.108$):

| ID | Configuração Metodológica | Features (B) | Treino (s) | PR-AUC | ROC-AUC | F1-Score | Perda Financeira ($L(\tau^*)$) | Recall (Rede) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | Sem Modelo (Aprovação Cega) | — | — | — | — | — | US$ 670.894,34 | 0,00% |
| **1A** | Baseline Raw (Sem PCA, 339 V) | 430 | 81,6s | 0,5100 | **0,8915** | 0,5351 | US$ 325.073,39 | 74,47% |
| **1B** | Baseline Com PCA (15-d) | 106 | **36,4s** | 0,4950 | 0,8912 | 0,5159 | US$ 325.726,95 | 75,37% |
| **2** | Baseline + Topologia de Grafo | 108 | 37,2s | 0,4964 | 0,8911 | 0,5232 | **US$ 321.913,83** | 76,01% |
| **3** | Baseline + Velocidade Causal 24h | 108 | 38,1s | 0,5039 | 0,8914 | 0,5266 | US$ 324.960,63 | 75,47% |
| **4** | Baseline + Grafos Completos | 110 | 39,0s | 0,5050 | 0,8913 | 0,5279 | US$ 322.479,44 | 76,04% |
| **5** | Baseline + Feature Engineering Tabular | 121 | 43,8s | 0,5526 | 0,8888 | 0,5513 | US$ 375.409,24 | 78,54% |
| **6A** | **All-in-One Proposto (Com PCA 15-d)** | **125** | **44,6s** | **0,5563** | **0,8922** | **0,5553** | US$ 378.925,44 | **79,18%** |
| **6B** | All-in-One (Sem PCA / 339 V Brutas) | 449 | 91,9s | 0,5550 | 0,8901 | 0,5547 | US$ 359.010,90 | 78,89% |

### Principais Destaques Empíricos:
* **Ganhos em Redes de Conluio:** O modelo *All-in-One* elevou a taxa de captura (*Recall*) sobre conluios de hardware compartilhado de **75,37% para 79,18% (+3,81 p.p.)**.
* **Impacto da Compressão PCA:** A redução dimensional cortou **55,4% do tempo de treino** e elevou o ROC-AUC global para **0,8922**.
* **Economia Operacional:** Os modelos baseados em grafos reduziram as perdas operacionais para US$ 321.913, gerando economia líquida de mais de **US$ 348 mil** sobre a aprovação cega.

---

## 🖼️ Galeria de Figuras

Todas as figuras foram geradas na pasta [`figures/`](figures/) em 300 DPI, obedecendo ao padrão estrito de 1 gráfico por figura:

* [`fig1_temporal_fraud_raw_volume.png`](figures/fig1_temporal_fraud_raw_volume.png) — Volume financeiro bruto e taxa temporal de fraude.
* [`fig2_v_collinearity_pca.png`](figures/fig2_v_collinearity_pca.png) — Matriz de correlação densa das 339 variáveis Vesta ($V$).
* [`fig3_entity_resolution_card_vs_uid.png`](figures/fig3_entity_resolution_card_vs_uid.png) — Distribuição log-log (CCDF) da resolução de entidades (`card1` vs UID).
* [`fig4_device_sharing_fraud_rings.png`](figures/fig4_device_sharing_fraud_rings.png) — Taxa de fraude vs quantidade de UIDs por dispositivo físico.
* [`fig5_velocity_burstiness_kde.png`](figures/fig5_velocity_burstiness_kde.png) — Curvas KDE de intervalo $\Delta t$ entre transações (legítimas vs rajadas criminosas).
* [`fig6_structuring_and_productcd.png`](figures/fig6_structuring_and_productcd.png) — Fracionamento de valores e taxas de fraude por `ProductCD`.
* [`fig7_bipartite_pagerank_dist.png`](figures/fig7_bipartite_pagerank_dist.png) — Densidade de probabilidade estacionária do PageRank bipartido.
* [`fig8_financial_dispersion_boxplots.png`](figures/fig8_financial_dispersion_boxplots.png) — Boxplots de desvio relativo do ticket ($Z$-Score) por portador.
* [`fig9_ablation_pr_roc_curves.png`](figures/fig9_ablation_pr_roc_curves.png) — Curvas Precision-Recall comparativas do estudo de ablação.
* [`fig10_cost_and_typology_breakdown.png`](figures/fig10_cost_and_typology_breakdown.png) — Curvas de perda operacional $L(\tau)$ sob matriz assimétrica de Bayes.

---

## 💻 Instalação e Execução

### 1. Pré-requisitos e Ambiente
Recomenda-se o uso do gerenciador [`uv`](https://github.com/astral-sh/uv) para velocidade e reprodutibilidade:

```bash
# Clonar o repositório
git clone https://github.com/VictorValadares/TG-graph.git
cd TG-graph

# Criar ambiente virtual e instalar dependências
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt # ou pyproject.toml
```

### 2. Download do Benchmark IEEE-CIS
Coloque os arquivos `train_transaction.csv` e `train_identity.csv` na pasta `data/raw/ieee-fraud-detection/`.

### 3. Execução do Pipeline Completo (100% dos Dados)
Para rodar a ablação completa de 8 configurações e persistir os resultados em `results/100percent/`:
```bash
python3 scripts/run_100percent_pipeline.py
```

### 4. Regenerar Figuras em Alta Resolução
```bash
python3 scripts/generate_all_single_figures.py
```

### 5. Compilar a Monografia em PDF
Requer Node.js e Puppeteer para renderização fiel de MathJax e Markdown:
```bash
uv run --with "markdown,pymdown-extensions" python3 scripts/export_paper_to_pdf.py
```

---

## 📓 Suíte de Jupyter Notebooks (`notebooks/`)

1. **`01_eda_and_feature_selection.ipynb`**: Análise exploratória e colinearidade das variáveis Vesta.
2. **`02_model_and_verification.ipynb`**: Baseline tabular e suite de verificação de métricas.
3. **`03_synthetic_graph_features.ipynb`**: Prototipação de métricas de rede em grafos sintéticos.
4. **`04_ablation_with_vs_without_graphs_colab.ipynb`**: Estudo comparativo de ablação de grafos vs tabular.
5. **`05_fraud_typology_cost_function_analysis.ipynb`**: Análise por tipologia de ataque e otimização de custo de Bayes.
6. **`06_walk_forward_temporal_ablation_colab.ipynb`**: Validação temporal *walk-forward* em múltiplas janelas.
7. **`07_hyperparameter_tuning_and_significance_colab.ipynb`**: Ajuste fino e testes de significância estatística.

---

## 📁 Estrutura do Repositório

```
TG-graph/
├── figures/              # 10 Figuras em alta resolução (300 DPI) para o artigo
├── results/100percent/   # CSVs de métricas, custos e predições com 100% dos dados
├── notebooks/            # Suíte de 7 Jupyter Notebooks experimentais
├── src/                  # Módulos Python (ingestão, features de grafo, modelos, métricas)
│   ├── data/             # Ingestão e resolução de entidades UID
│   ├── features/         # Features tabulares, grafos NetworkX e PCA
│   ├── models/           # XGBoost / LightGBM e Roteamento Dual
│   └── evaluation/       # PR-AUC, ROC-AUC, curvas de custo e matrizes de confusão
├── scripts/              # Scripts de automação, pipelines e exportação de PDF
├── docs/                 # Relatórios metodológicos e decisões de arquitetura
├── LOCAL_COMPLETE_PAPER.md   # Artigo completo em Markdown
├── LOCAL_COMPLETE_PAPER.pdf  # Monografia / Artigo compilado em PDF (3,13 MB)
├── pyproject.toml        # Metadados e dependências do projeto
└── README.md             # Este documento
```

---

## 📜 Licença

Distribuído sob a licença **MIT**. Consulte `LICENSE` para mais detalhes.