# Relatório Metodológico e Documentação Arquitetural dos Notebooks Experimentais

**Título da Pesquisa:** O Impacto Estrutural e Econômico de Grafos na Detecção de Fraude em Pagamentos Digitais: Um Estudo de Ablação e Modelagem Sensível ao Custo  
**Autor:** Victor Valadares  
**Dataset Oficial:** IEEE-CIS Fraud Detection Benchmark (590.540 transações reais de e-commerce)

---

## Sumário Executivo & Mapa da Arquitetura Experimental

Este documento estabelece o registro formal e detalhado de todas as decisões científicas, arquiteturais e de código implementadas nos notebooks da pasta [`notebooks/`](file:///home/victor/Documents/TG-graph/notebooks). Cada notebook cumpre um papel modular e estritamente auditável dentro da jornada experimental que fundamenta o [COMPLETE_PAPER.md](file:///home/victor/Documents/TG-graph/COMPLETE_PAPER.md).

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 MAPA DA JORNADA EXPERIMENTAL                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
            ┌────────────────────────────────────┴────────────────────────────────────┐
            ▼                                                                         ▼
 [01_eda_and_feature_selection]                                           [02_model_and_verification]
 Análise Diagnóstica de Tipologias                                         Verificação Unitária & Baseline
 e Quebra da Hipótese IID                                                  Auditoria Local de Pipeline
            │                                                                         │
            └────────────────────────────────────┬────────────────────────────────────┘
                                                 │
                                                 ▼
                                     [03_full_sota_pipeline_colab]
                                     Pipeline de Produção Integrado
                                     (Ensemble XGBoost + LightGBM)
                                                 │
                                                 ▼
                                  [04_ablation_with_vs_without_graphs]
                                  ⭐ CORAÇÃO EXPERIMENTAL DO TCC ⭐
                                  • Com vs. Sem PCA nas 339 colunas V
                                  • Decomposição Topologia vs. Velocidade
                                  • Pós-Processamento Causal vs. Bruto
                                  • Avaliação de Custo Financeiro Real
                                  • Exporta: val_predictions.csv
                                                 │
            ┌────────────────────────────────────┼────────────────────────────────────┐
            ▼                                    ▼                                    ▼
[05_fraud_typology_cost_function]   [06_walk_forward_temporal_ablation]   [07_hyperparameter_tuning]
Impacto Financeiro Real &           Robustez ao Concept Drift             Otimização Bayesiana (Optuna)
Taxonomia Estrutural de Fraudes     em 4 Janelas Expansivas Mensais       & Teste Bootstrap (1.000 iterações)
```

---

## 1. Mapeamento dos Notebooks e Papel no Processo Experimental

| Notebook | Nome do Arquivo | Seção Correspondente no Paper | Pergunta de Pesquisa (RQ) | Decisão Metodológica Central |
|---|---|---|---|---|
| **01** | [`01_eda_and_feature_selection.ipynb`](file:///home/victor/Documents/TG-graph/notebooks/01_eda_and_feature_selection.ipynb) | Seção 3 (Taxonomia Estrutural) | Contexto & Hipótese de Trabalho | Diagnóstico empírico de que fraudes não são IID, demonstrando emuladores, hardware compartilhado e rajadas. |
| **02** | [`02_model_and_verification.ipynb`](file:///home/victor/Documents/TG-graph/notebooks/02_model_and_verification.ipynb) | Seção 5 (Auditoria de Leakage) | Sanidade Experimental | Teste de sanidade em subamostra com verificações unitárias estritas contra vazamento temporal. |
| **03** | [`03_full_sota_pipeline_colab.ipynb`](file:///home/victor/Documents/TG-graph/notebooks/03_full_sota_pipeline_colab.ipynb) | Seção 4 (Metodologia) & 8.2 | Modelo Completo de Produção | Implementação do pipeline unificado com *Ensemble* (XGBoost + LightGBM) e roteamento dual de identidade. |
| **04** | [`04_ablation_with_vs_without_graphs_colab.ipynb`](file:///home/victor/Documents/TG-graph/notebooks/04_ablation_with_vs_without_graphs_colab.ipynb) | **Seção 6 (Estudo de Ablação)** | **RQ1 (Valor Estrutural Puro)** | **Ablação rigorosa isolando PCA, Topologia pura, Velocidade pura, Grafos completos e FE tabular sob hiperparâmetros fixos.** |
| **05** | [`05_fraud_typology_cost_function_analysis.ipynb`](file:///home/victor/Documents/TG-graph/notebooks/05_fraud_typology_cost_function_analysis.ipynb) | **Seção 7 (Modelagem de Custo)** & Seção 3 | **RQ2 & RQ3 (Impacto Econômico)** | **Otimização da Função de Custo sob predições reais dos modelos, eliminando simulações artificiais.** |
| **06** | [`06_walk_forward_temporal_ablation_colab.ipynb`](file:///home/victor/Documents/TG-graph/notebooks/06_walk_forward_temporal_ablation_colab.ipynb) | Seção 6.3 & Seção 8.3 | RQ1 (Estabilidade Temporal) | Validação temporal cruzada em 4 dobras expansivas (*Expanding Window Walk-Forward*), provando robustez ao *drift*. |
| **07** | [`07_hyperparameter_tuning_and_significance_colab.ipynb`](file:///home/victor/Documents/TG-graph/notebooks/07_hyperparameter_tuning_and_significance_colab.ipynb) | Seção 6 & Seção 8.1 | RQ1 (Rigor Estatístico) | Busca Bayesiana (Optuna) com orçamento idêntico + Teste Bootstrap pareado por blocos temporais (1.000 iterações). |

---

## 2. Detalhamento Bloco a Bloco do Código e Importância Metodológica

Abaixo disseca-se a fundamentação teórica, a formulação matemática e o código exato dos componentes críticos implementados nos notebooks.

### 2.1 Orçamento Experimental (`EXECUTION_BUDGET`)
* **Código:**
  ```python
  EXECUTION_BUDGET = 'FULL'  # Opções: 'FULL' ou 'ESSENTIAL'
  ```
* **Onde entra:** Célula de configuração inicial dos Notebooks 04, 05, 06 e 07.
* **Importância Metodológica:**
  Garante a viabilidade operacional em qualquer ambiente.
  - No modo `'ESSENTIAL'`, o código roda em **25 a 35 minutos** sobre uma janela de 20% das transações e 200 árvores.
  - No modo `'FULL'`, o código consome todas as **590.540 transações** com 450 árvores em profundidade 9 (~1h30 a 2h), gerando os números de escala total do trabalho.
  - **Ajuste para Google Colab Gratuito (`SAMPLE_FRACTION = 0.5`):** Devido ao limite estrito de ~12 GB de RAM no Colab padrão, os notebooks foram padronizados com `SAMPLE_FRACTION = 0.5` (~295.270 transações) e downcasting numérico (`reduce_mem_usage`). Essa amostragem cobre integralmente todo o horizonte de 182 dias e preserva a ordem causal estrita, eliminando travamentos por *Out-Of-Memory* (OOM).

---

### 2.2 Resolução de Entidade: O UID Mágico Padronizado
* **Formulação Matemática:**
  $$\text{Day}(t) = \left\lfloor \frac{t}{86.400} \right\rfloor$$
  $$\text{D1}_{\text{anchor}} = \left\lfloor \text{Day}(t) - D1(t) \right\rfloor$$
  $$\text{UID}_{\text{padronizado}} = \text{card1} \;\Vert\; \text{addr1} \;\Vert\; \text{D1}_{\text{anchor}}$$
* **Código Implementado:**
  ```python
  def build_magic_uid(df):
      c1 = df['card1'].fillna(0).astype(int).astype(str)
      addr = df['addr1'].fillna(0).astype(int).astype(str)
      day = df['timestamp'] / 86400.0
      d1_anchor = (day - df['D1'].fillna(0)).astype(int).astype(str)
      return c1 + "_" + addr + "_" + d1_anchor
  ```
* **Importância Metodológica:**
  No dataset IEEE-CIS, a coluna $D1$ representa os dias transcorridos desde a emissão daquele cartão específico. Como o tempo avança linearmente, subtrair $D1$ do dia corrente da transação ($\text{Day} - D1$) ancora o evento em uma coordenada invariante no calendário civil. Ao concatenar com o BIN/cartão (`card1`) e a região de cobrança (`addr1`), sintetiza-se uma entidade portadora unívoca sem qualquer vazamento de alvo (*leakage-free*).
* **Blindagem para a Banca:**
  Elimina qualquer inconsistência entre notebooks: a mesma fórmula de UID é utilizada em todos os notebooks (01 a 07).

---

### 2.2.1 Natureza das Features Tabulares de Portador ($\mu, \sigma$) e Diagnóstico Causal
* **Formulação e Código Implementado:**
  ```python
  # Agregações de Grupo por UID ajustadas no Treino
  agg_cols = ['TransactionAmt', 'D15n', 'C1', 'C13']
  for col in agg_cols:
      mean_map = train_df.groupby('uid')[col].mean().to_dict()
      std_map = train_df.groupby('uid')[col].std().fillna(0.0).to_dict()
      global_mean = train_df[col].mean()
      for d in [train_df, val_df]:
          d_mean = d['uid'].map(mean_map).fillna(global_mean)
          d_std = d['uid'].map(std_map).fillna(0.0)
          d[f'{col}_to_mean_uid'] = d[col] / (d_mean + 1e-5)
          d[f'{col}_to_std_uid'] = (d[col] - d_mean) / (d_std + 1e-5)
  ```
* **Diagnóstico de Causalidade (Treino vs. Validação):**
  1. **Não são expansivas:** As variáveis utilizam a média e o desvio padrão estáticos calculados sobre a totalidade das transações de cada `uid` no conjunto de treino.
  2. **Comportamento no Treino (*In-Sample Lookahead*):** No treino, uma transação ocorrida no Dia 1 recebe a média de compras calculada considerando também as transações dos Dias 60 ou 120 daquele portador. Ocorre, portanto, um *lookahead* interno no treino análogo ao do grafo estático.
  3. **Comportamento na Validação (*Sem Vazamento Futuro de Validação*):** Como o dicionário `mean_map` é derivado estritamente do treino, a base de validação não enxerga transações futuras da própria validação. Contudo, para **65,76% dos UIDs da validação que são inéditos** (abrangendo 61,47% das transações de validação), `d['uid'].map(mean_map)` resulta em `NaN` e é imputado com `global_mean` (razão = 1,0 e escore Z = 0,0).
  4. **Por que o FE Tabular ainda assim domina o lift:** Mesmo com UIDs inéditos recebendo valores neutros, variáveis como $D_n$ (deltas temporais ancorados no calendário) e contadores $C_n$ possuem generalização causal contínua e não sofrem a mesma dependência de conexões de hardware que penaliza os grafos bipartidos.

---

### 2.3 Compressão Espectral PCA vs. Dados Brutos (Com vs. Sem PCA)
* **Formulação Matemática:**
  $$\mathbf{\Sigma}_V = \frac{1}{N-1} \mathbf{X}_V^\top \mathbf{X}_V = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^\top$$
  $$\mathbf{Z}_V = \mathbf{X}_V \mathbf{U}_{:, 1:15}, \quad \mathbf{Z}_V \in \mathbb{R}^{N \times 15}$$
* **Código Implementado:**
  ```python
  # Decomposição PCA ajustada estritamente sobre o Treino
  pca = PCA(n_components=15, random_state=42)
  pca_train = pca.fit_transform(X_v_train)
  pca_val = pca.transform(X_v_val)
  ```
* **Importância Metodológica:**
  A família $V1$ a $V339$ contém 339 atributos fornecidos pela adquirente Vesta, com colinearidade extrema e matriz esparsa. A compressão PCA em 15 dimensões reduz a largura da base em mais de 75%, acelerando drasticamente o treino das árvores.
* **O Teste Metodológico de Ablação (Config 1A vs 1B e 6A vs 6B):**
  Para responder cientificamente se o PCA descarta sinal preditivo ou remove ruído:
  - **Configuração 1A**: Baseline mantendo as 339 colunas $V$ brutas.
  - **Configuração 1B**: Baseline substituindo as 339 colunas $V$ pelos 15 componentes PCA.
  - **Configuração 6A**: All-in-One com PCA.
  - **Configuração 6B**: All-in-One sem PCA (Raw V).
  Isso documenta formalmente o *trade-off* de compressão dimensional.

---

### 2.4 Topologia de Grafo Pura (Grafo Bipartido, PageRank e Graus Estruturais)
* **Formulação Matemática:**
  $$\mathcal{G}_{\text{train}} = (\mathcal{V}_{\text{uid}} \cup \mathcal{V}_{\text{dev}}, \mathcal{E}_{\text{train}}), \quad e = (u, d) \iff u \text{ operou em } d$$
  $$\mathbf{p}^{(k+1)} = \alpha \mathbf{M} \mathbf{p}^{(k)} + \frac{1 - \alpha}{|\mathcal{V}|} \mathbf{e}, \quad \alpha = 0,85$$
  $$\deg_{\text{distinct}}(d) = |\mathcal{N}(d)| = \text{UIDs distintos conectados ao hardware } d$$
* **Código Implementado:**
  ```python
  # Construção restrita ao conjunto de treino e exclusão explícita de sentinelas nulas
  train_dev_edges = train_df[train_df['DeviceInfo_clean'] != 'null_device'][['uid', 'DeviceInfo_clean']].drop_duplicates()
  for _, row in train_dev_edges.iterrows():
      G_train.add_edge(f"uid_{row['uid']}", f"dev_{row['DeviceInfo_clean']}")

  pr_dict = nx.pagerank(G_train, alpha=0.85)

  dev_distinct_uids = train_dev_edges.groupby('DeviceInfo_clean')['uid'].nunique().to_dict()
  uid_distinct_devs = train_dev_edges.groupby('uid')['DeviceInfo_clean'].nunique().to_dict()
  ```
* **Importância Metodológica:**
  1. **Exclusão de Hubs Artificiais**: O valor nulo (`null_device`) foi rigorosamente excluído do grafo. Caso estivesse presente, formaria um supernó artificial conectando mais de 100.000 clientes não relacionados, colapsando a centralidade.
  2. **Imunidade a Vazamento Temporal**: O grafo e o PageRank são construídos **estritamente sobre as arestas do conjunto de treino**. Na validação, os escores são apenas consultados; portadores ou dispositivos inéditos recebem o valor padrão $1/|\mathcal{V}|$.
  3. **Graus Reais**: Mede-se o número de entidades *distintas* (vizinhos únicos na topologia bipartida), diferenciando clientes com muitas compras normais daqueles que operam em fazendas de dispositivos compartilhados.

---

### 2.5 Velocidade Causal 24h por UID (Busca Binária Vetorizada)
* **Formulação Matemática:**
  $$\text{velocity}_{24h}(i) = \Big| \big\{ k \in T(u_i) \;\mid\; t_i - 86.400 \le t_k \le t_i \big\} \Big|$$
* **Código Implementado:**
  ```python
  def compute_causal_velocity(df):
      df_sorted = df.sort_values('timestamp').copy()
      times = df_sorted['timestamp'].values
      counts = np.ones(len(df_sorted), dtype=np.int32)
      secs_prev = np.full(len(df_sorted), np.nan, dtype=np.float32)
      
      # Agrupamento estrito por UID com busca binária interna
      uid_indices = df_sorted.groupby('uid', sort=False).indices
      for uid, idxs in uid_indices.items():
          t_uid = times[idxs]
          left = np.searchsorted(t_uid, t_uid - 86400, side='left')
          counts[idxs] = np.arange(len(t_uid)) - left + 1
          if len(t_uid) > 1:
              secs_prev[idxs[1:]] = np.diff(t_uid)
              
      df_sorted['uid_tx_count_24h'] = counts
      df_sorted['uid_secs_since_prev'] = secs_prev
      return df_sorted.sort_index()
  ```
* **Importância Metodológica:**
  - Corrige a falha anterior onde a busca binária ocorria sobre toda a tabela global de transações.
  - Executa a busca binária causal dentro dos timestamps daquele `uid` específico.
  - O algoritmo é estritamente $O(N \log K)$ (onde $K$ é o número de transações do cliente, tipicamente $K \le 50$), processando 590.000 linhas em apenas **1,13 segundo**.
  - Não há vazamento de dados futuros (`searchsorted` com `side='left'` olha apenas para $t \le t_i$).

---

### 2.6 Separação de Topologia vs. Velocidade na Ablação
* **Código das Configurações na Célula 6 do Notebook 04:**
  - **Configuração 2 (Apenas Topologia)**: `include_topology=True, include_velocity=False`
  - **Configuração 3 (Apenas Velocidade)**: `include_topology=False, include_velocity=True`
  - **Configuração 4 (Grafos Completos)**: `include_topology=True, include_velocity=True`
* **Importância Metodológica & Defesa da Tese:**
  Esta separação atende frontalmente à crítica clássica de bancas acadêmicas: *"o ganho veio da velocidade temporal ou da topologia do grafo?"*.
  Ao criar a linha 2 (Apenas PageRank + Graus) e a linha 3 (Apenas Velocidade 24h), prova-se matematicamente quanto cada dimensão adiciona ao modelo tabular.

---

### 2.7 Pós-Processamento Causal por Portador e Correção de Alinhamento Temporal
* **Formulação Matemática:**
  $$\mu_{\text{exp}}(t_i) = \frac{1}{|\{k \in T(u_i) \mid t_k \le t_i\}|} \sum_{t_k \le t_i} \hat{y}(k)$$
  $$\hat{y}_{\text{final}}(i) = 0,7 \cdot \hat{y}_{\text{GBDT}}(i) + 0,3 \cdot \mu_{\text{exp}}(t_i)$$
* **Código Implementado e Blindado:**
  ```python
  # Média expansiva causal estritamente alinhada por índice temporal
  val_temp = val_df[['uid', 'timestamp']].copy()
  val_temp['raw_score'] = y_score_raw
  val_temp_sorted = val_temp.sort_values('timestamp')
  exp_series = val_temp_sorted.groupby('uid')['raw_score'].expanding().mean().reset_index(level=0, drop=True)
  val_temp_sorted['score_pp'] = 0.7 * val_temp_sorted['raw_score'] + 0.3 * exp_series
  y_score_pp = val_temp_sorted.sort_index()['score_pp'].values
  ```
* **Diagnóstico e Quantificação Numérica do Pitfall do Pandas (`.values` vs `.reset_index`):**
  - O método `df.groupby('uid')['raw_score'].expanding().mean()` devolve uma `pd.Series` com índice multinível `(uid, original_index)`, ordenada internamente pela chave alfabética do grupo (`uid`), e não pela ordem temporal de `val_temp_sorted`.
  - O emprego de `.values` descartava o índice e realizava atribuição puramente posicional sobre o DataFrame ordenado cronologicamente, provocando uma mistura espúria de escores entre portadores distintos.
  - A tabela abaixo documenta o impacto numérico exato (**Antes com `.values` $\to$ Depois com `.reset_index`**) nas 8 configurações de ablação sobre as 59.054 transações de validação:

| Configuração Avaliada | PR-AUC Bruto (Sem Pós-Proc) | PR-AUC com Bug (`.values`) | $\Delta$ Bug | PR-AUC Alinhado Causal (`.reset_index`) | $\Delta$ Causal Real |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1A. Baseline Raw** | 0,5038 | 0,4629 | **-0,0409** | 0,5034 | -0,0004 |
| **1B. Baseline PCA (15-d)** | 0,5028 | 0,4658 | **-0,0370** | 0,5016 | -0,0012 |
| **2. Apenas Topologia** | 0,4918 | 0,4513 | **-0,0405** | 0,4906 | -0,0012 |
| **3. Apenas Velocidade** | 0,4943 | 0,4536 | **-0,0407** | 0,4922 | -0,0021 |
| **4. Grafos Completos** | 0,4997 | 0,4600 | **-0,0397** | 0,4980 | -0,0017 |
| **5. Apenas FE Tabular** | 0,5304 | 0,4902 | **-0,0402** | 0,5299 | -0,0005 |
| **6A. All-in-One PCA** | 0,5344 | 0,4917 | **-0,0427** | 0,5329 | -0,0015 |
| **6B. All-in-One Raw V** | 0,5395 | 0,5005 | **-0,0390** | **0,5434** | **+0,0039** |

* **Conclusões Numéricas do Pós-Processamento:**
  1. A falha de alinhamento com `.values` causava uma degradação artificial severa e uniforme de **-0,0370 a -0,0427** em PR-AUC em todas as configurações, explicando a anomalia em que o pós-processamento degradava todos os modelos.
  2. Com a preservação dos índices via `.reset_index(level=0, drop=True)`, o pós-processamento causal estabiliza o PR-AUC e gera ganho genuíno no modelo completo **6B (+0,0039)**, elevando-o de 0,5395 para **0,5434** e o F1 de 0,5646 para **0,5705**.
* **Superação do Leakage Bidirecional do `transform('mean')` (Notebook 03):**
  - No Notebook 03 preliminar, utilizava-se `val_df.groupby('uid')['raw_pred'].transform('mean')`, que calculava a média estática incorporando eventos futuros da validação (*lookahead temporal*), gerando um ganho artificial de PR-AUC (0,5544 $\to$ 0,5597 e F1 0,5573).
  - A conversão para média expansiva causal alinhada (`expanding().mean().reset_index()`) elimina o lookahead: o modelo sem pós-processamento registra PR-AUC 0,5544 / F1 0,5524, e com pós-processamento causal atinge PR-AUC 0,5544 e F1 0,5573 (sem inflação artificial de PR-AUC).
  - **Apresentação Transparente**: Reportam-se ambos os valores (Bruto e Pós-Processado) em todo o relatório.

---

### 2.8 Avaliação da Função de Custo Financeiro Real (Sem Simulações)
* **Formulação Matemática (Bayes Minimum Risk):**
  $$L(\tau) = \sum_{i=1}^N \Big[ y_i (1 - \hat{y}_i(\tau)) \cdot (\text{amt}_i + C_{\text{chargeback}}) + (1 - y_i) \hat{y}_i(\tau) \cdot (C_{\text{revisão}} + \alpha \cdot \text{amt}_i) + y_i \hat{y}_i(\tau) \cdot C_{\text{operacional}} \Big]$$
  $$\tau^* = \arg\min_{\tau \in [0, 1]} L(\tau)$$
* **Código Implementado (Notebooks 04 e 05):**
  ```python
  def compute_financial_loss(y_true, y_score, amounts, threshold, C_chargeback=15.0, C_review=3.0, alpha=0.05, C_op=0.10):
      y_pred = (y_score >= threshold).astype(int)
      fn_mask = (y_true == 1) & (y_pred == 0)
      loss_fn = (amounts[fn_mask] + C_chargeback).sum()
      fp_mask = (y_true == 0) & (y_pred == 1)
      loss_fp = (C_review + alpha * amounts[fp_mask]).sum()
      tp_mask = (y_true == 1) & (y_pred == 1)
      loss_tp = tp_mask.sum() * C_op
      return loss_fn + loss_fp + loss_tp, loss_fn, loss_fp, loss_tp
  ```
* **Correção Metodológica Histórica:**
  No notebook 05 anterior, o modelo com grafos era artificialmente simulado multiplicando-se a probabilidade das fraudes por 1,35 (`y_score_with_graph[is_coord_mask] *= 1.35`).  
  **No código atual, isso foi 100% extirpado.** A função de custo agora é alimentada exclusivamente pelos vetores de probabilidade **reais** inferidos pelo XGBoost Baseline e pelo XGBoost com Grafos gerados no Notebook 04 (via arquivo `val_predictions.csv`).

---

### 2.9 Análise de Sensibilidade e Recall por Tipologia Estrutural (Notebook 05)
* **Matriz de Cenários Operacionais:**
  1. *Conservador*: $\alpha = 0,01$ (fricção baixa), $C_{\text{cb}} = \$15$.
  2. *Padrão de Mercado*: $\alpha = 0,05$ (fricção padrão de 5%), $C_{\text{cb}} = \$15$.
  3. *Rigoroso*: $\alpha = 0,05$, $C_{\text{cb}} = \$25$ (bandeiras aplicam multas punitivas por alto índice de disputa).
  4. *Sensível ao Cliente*: $\alpha = 0,10$ (alto risco de churn de clientes VIP), $C_{\text{cb}} = \$15$.
* **Importância Científica:**
  Demonstra que o modelo aumentado por grafos é economicamente superior independentemente do perfil de apetite ao risco da instituição financeira, provando a robustez da política ótima de corte $\tau^*$.
* **Metodologia de Avaliação por Tipologia Estrutural:**
  A partição de fraude é segmentada em duas tipologias:
  - **Fraudes Coordenadas em Rede**: Transações em que o dispositivo é compartilhado por $\ge 2$ UIDs ou cuja velocidade acumulada em 24h atinge $\ge 2$ transações ($N = 1.260$ fraudes na validação, respondendo por 46,65% das perdas financeiras brutas, totalizando US$ 146.605,41);
  - **Fraudes Isoladas**: Transações avulsas sem compartilhamento de dispositivo nem cadência acelerada em 24h ($N = 846$ fraudes, respondendo por 53,35% das perdas, totalizando US$ 167.657,80).
* **Rigor Metodológico: Avaliação a Falsa Positiva Fixa (FPR Controlado) vs. Artefato de Limiar:**
  - *O Artefato de Limiar*: No Notebook 05, a comparação ingênua de recall entre Baseline ($\tau^* = 0,35$, recall 57,84%) e Modelo com Grafos ($\tau^* = 0,23$, recall 62,21%, aparente lift de +4,37%) decorre em grande parte da redução do limiar de corte, que gera mais alertas gerais.
  - *O Protocolo Metodológico Controlado (FPR Fixo)*: Para mensurar o ganho estrutural real livre de distorção de corte, os modelos são avaliados a uma taxa de falsos positivos idêntica e fixa ($\text{FPR} = 1,0\%$, correspondente à capacidade de análise manual das equipes de antifraude). Sob esse protocolo estrito:
    - O recall em fraudes coordenadas cresce de **50,87%** (Baseline 1B) para **61,27%** (All-in-One 6B com Pós-Processamento Causal), totalizando um expressivo ganho de **+10,40 pontos percentuais** (IC 95% Bootstrap: **[+8,02, +12,78]** p.p., $p < 0,001$).
    - **Decomposição Honesta da Causalidade**: Desses +10,40 p.p., **+7,54 p.p.** derivam da Engenharia Tabular de Entidade (Configuração 5 atinge 58,41%, IC 95%: [+5,08, +9,68] p.p.), **+2,22 p.p.** derivam dos Grafos Bipartidos (6B atinge 60,63%, IC 95%: [+1,03, +3,73] p.p.) e **+0,64 p.p.** adicionais decorrem do Pós-Processamento Causal (efeito marginal, IC 95%: [+0,00, +1,43] p.p.).
* **Ressalva Metodológica de Circularidade (*Surrogate Label*):**
  A definição de fraude coordenada ($\text{is\_shared\_device} \ge 2 \lor \text{velocity}_{24h} \ge 2$) é um rótulo substituto operacional devido à ausência de dados externos de anotação de anéis de fraude no benchmark. Como os modelos utilizam exatamente atributos de conectividade e velocidade temporal, existe uma circularidade operacional inerente: os ganhos medem a eficácia do pipeline em detectar os padrões para os quais foi desenhado sob restrição de falsos alarmes.

---

### 2.10 Interpretabilidade Global e Estudo de Caso de Subgrafo Real
* **TreeSHAP / Gain no XGBoost Model B:**
  Na célula 9 do Notebook 04, extrai-se o ganho médio por divisão (*Gain*) de cada nó das árvores, exibindo onde as métricas de rede (`device_degree`, `uid_pagerank`, `uid_tx_count_24h`) se posicionam no ranking das mais de 80 colunas.
* **Estudo de Caso Qualitativo:**
  O notebook inspeciona automaticamente na partição de validação temporal um dispositivo que se conectou a múltiplos portadores e registrou fraudes confirmadas, exibindo:
  - `TransactionAmt` (frequentemente valores normais, simulando compras cotidianas);
  - `Score Baseline` (baixo, pois o modelo tabular não enxerga a máquina);
  - `Score com Grafos` (elevado drasticamente devido a `device_degree >= 3` e PageRank).
  Isso traduz o rigor quantitativo em evidência visual e palpável para a banca examinadora.

---

### 2.11 Validação Walk-Forward e Teste de Significância Bootstrap (Notebooks 06 e 07)
* **Walk-Forward (Notebook 06):**
  Avalia a generalização temporal em 4 dobras expansivas com janelas mensais:
  - Fold 1: Treino Meses 1–2 $\rightarrow$ Validação Mês 3;
  - Fold 2: Treino Meses 1–3 $\rightarrow$ Validação Mês 4;
  - Fold 3: Treino Meses 1–4 $\rightarrow$ Validação Mês 5;
  - Fold 4: Treino Meses 1–5 $\rightarrow$ Validação Mês 6.
  Elimina o viés de uma única divisão de corte e relata $\mu \pm \sigma$.
* **Bootstrap Pareado por Blocos (Notebook 07):**
  Realiza 1.000 reamostragens pareadas sobre a base de validação, calculando a distribuição do delta de performance:
  $$\Delta \text{PR-AUC} = \text{PR-AUC}_{\text{Grafos}} - \text{PR-AUC}_{\text{Baseline}}$$
  Gerando o Intervalo de Confiança a 95% ($IC_{95\%}$) e o p-valor empírico, garantindo que o ganho não é ruído amostral.

---

## 3. Roteiro Passo a Passo para Execução no Google Colab

Para executar os experimentos no Google Colab e gerar os artefatos de dados reais, siga a ordem sequencial:

```
PASSO 1: Fazer upload dos dados do IEEE-CIS no Colab (ou montar o Google Drive em /content/drive/MyDrive/ieee-data)
         - train_transaction.csv
         - train_identity.csv

PASSO 2: Abrir e Executar o notebook:
         notebooks/04_ablation_with_vs_without_graphs_colab.ipynb
         • Definir EXECUTION_BUDGET = 'FULL' (ou 'ESSENTIAL' para teste rápido)
         • O notebook treinará as configurações (1A, 1B, 2, 3, 4, 5, 6A, 6B)
         • Gerará os gráficos de Ablação, PCA, Custo e Estudo de Caso
         • Exportará automaticamente o arquivo: val_predictions.csv

PASSO 3: Abrir e Executar o notebook:
         notebooks/05_fraud_typology_cost_function_analysis.ipynb
         • O notebook detectará automaticamente o 'val_predictions.csv'
         • Gerará as curvas de custo real, análise de sensibilidade e o breakdown por tipologia
         • Exportará os gráficos e o sumário financeiro

PASSO 4 (Opcional para Robustez Adicional):
         Executar os notebooks:
         • notebooks/06_walk_forward_temporal_ablation_colab.ipynb
         • notebooks/07_hyperparameter_tuning_and_significance_colab.ipynb
```

---

## 4. Conclusão Metodológica & Síntese da Narrativa Científica

Com a refatoração e auditoria rigorosa de todos os 7 notebooks, estabelece-se a narrativa científica autêntica, matematicamente precisa e plenamente blindada contra questionamentos de banca:

1. **A Engenharia de Atributos Tabular Carrega a Maior Parte do Ganho (Cadeia PCA: 87,3% do Lift):**
   - Na cadeia de ablação estritamente consistente sob compressão dimensional PCA ($1\text{B} \to 5 \to 6\text{A}$):
     - **Baseline PCA (1B)**: PR-AUC Bruto = **0,5028**
     - **Apenas FE Tabular (5)**: PR-AUC Bruto = **0,5304** (Lift de **+0,0276**)
     - **All-in-One PCA (6A)**: PR-AUC Bruto = **0,5344** (Lift de **+0,0040** vindo de Grafos)
     - **Lift Total Bruto da Cadeia PCA**: $+0,0316$ (de 0,5028 a 0,5344).
     - **Decomposição Relativa do Lift**: A Engenharia Tabular ($5 - 1\text{B}$) responde por **87,3%** de todo o ganho obtido sobre o baseline, enquanto os atributos de rede bipartida ($6\text{A} - 5$) agregam os **12,7%** restantes.
   - Nenhuma técnica de rede ou GNN substitui a necessidade de resolução de entidades e ancoragem temporal de calendário no domínio de fraudes financeiras.

2. **A Modelagem em Grafos Atua como um Refinamento Incremental Ortogonal (+0,0040 a +0,0091 em PR-AUC Bruto):**
   - O papel dos atributos de grafo (graus de conectividade bipartida e PageRank de perfis de dispositivo) não é substituir o classificador tabular, mas fornecer uma dimensão estrutural relacional que linhas tabulares isoladas não sintetizam.
   - Na comparação com atributos brutos sem compressão PCA ($1\text{A} \to 6\text{B}$):
     - **Baseline Raw V (1A)**: PR-AUC Bruto = **0,5038**
     - **All-in-One Raw V (6B)**: PR-AUC Bruto = **0,5395** (Lift bruto de **+0,0357** sobre a baseline 1A, e **+0,0091** sobre a Configuração 5).
     - **Contribuição do Pós-Processamento Causal**: O filtro de média expansiva alinhada no 6B adiciona **+0,0039**, elevando o PR-AUC final para **0,5434** (e o F1 para 0,5705).
   - **Tuning Bayesiano Separado (Notebook 07)**: A busca Optuna eleva o Baseline para 0,5555 e o modelo com Grafos para 0,5622 (**+0,0067**). Este ganho decorre da sintonização de hiperparâmetros e é documentado isoladamente, não se somando aos deltas de hiperparâmetros fixos da ablação.
   - Embora incremental no cômputo agregado de PR-AUC, a modelagem em grafos exibe **alta especialização em topologias complexas**:
     - No protocolo com taxa de falsos positivos controlada ($\text{FPR} = 1,0\%$), eleva o recall de fraudes coordenadas de 58,41% para **61,27%** (+2,86 p.p. adicionais ao FE tabular);
     - Proporciona estabilidade temporal sustentada contra *concept drift*, gerando uma **economia líquida média de US\$ 6.722,08 por mês** em validação *Walk-Forward* de 4 janelas expansivas (Notebook 06).

3. **Reprodutibilidade e Blindagem Científica Plena:**
   - O PCA foi demonstrado lado a lado com os atributos brutos $V$ (reduzindo tempo de treino em **51,4%**, de 105,8s para 51,4s na baseline, com retenção da separabilidade).
   - O desalinhamento por `.values` no pós-processamento temporal foi corrigido via `.reset_index(level=0, drop=True)`, substituindo a queda artificial de -0,0409 por estabilização rigorosa.
   - A assimetria do grafo estático intra-treino (inflação de 7,39x) e a desconexão na validação (84,15% de nós nulos ou inéditos) foram quantificadas empiricamente na Seção 6.

---

## 5. Resultados Oficiais Consolidados da Execução Sequencial (Amostragem Causal 50% / 295.270 Transações)

Abaixo estão registrados os resultados empíricos reais obtidos com a execução sequencial isolada (um a um, com liberação completa de memória RAM entre os processos) de todos os 7 notebooks experimentais:

### 📊 5.1 Sumário por Notebook

| Notebook | Escopo & Arquitetura | Tempo de Execução | Métricas Principais | Figuras Geradas |
|---|---|---|---|---|
| **01 EDA** | 295.270 transações (236.216 treino / 59.054 validação) | 0.58 min | Taxa de Fraude: 3.50% — 46.65% das perdas associadas a fraudes coordenadas em rede | 12 figuras (`nb01_fig_01` a `12`) |
| **02 Verificação** | Subamostra sintética auditável | 0.21 min | Testes de Sanidade Temporal: 4/4 verificações aprovadas (Sem data leakage) | 3 figuras (`nb02_fig_01` a `03`) |
| **03 Full SOTA Pipeline** | Ensemble de Produção (XGBoost + LightGBM com Roteamento Dual) | 4.15 min | **Sem Pós-Proc**: PR-AUC 0.5544 — ROC-AUC 0.8931 — F1 0.5524<br>**Com Pós-Proc Causal Alinhado**: PR-AUC 0.5544 — ROC-AUC 0.8976 — F1 0.5573 | 1 figura (`nb03_fig_01`) |
| **04 Ablação Estrutural** | 8 configurações comparativas isolando PCA, Topologia, Velocidade e FE Tabular | 9.85 min | **Baseline Raw (1A)**: PR-AUC 0.5038 (PP: 0.5034) — F1: 0.5061 (PP: 0.5095)<br>**Baseline PCA 15-d (1B)**: PR-AUC 0.5028 (PP: 0.5016) — F1: 0.4979 (PP: 0.5023)<br>**Apenas Topologia (2)**: PR-AUC 0.4918 (PP: 0.4906) — F1: 0.4917 (PP: 0.4984)<br>**Apenas Velocidade (3)**: PR-AUC 0.4943 (PP: 0.4922) — F1: 0.4936 (PP: 0.4957)<br>**Grafos Completos (4)**: PR-AUC 0.4997 (PP: 0.4980) — F1: 0.4983 (PP: 0.5093)<br>**Apenas FE Tabular (5)**: PR-AUC 0.5304 (PP: 0.5299) — F1: 0.5443 (PP: 0.5472)<br>**All-in-One PCA (6A)**: PR-AUC 0.5344 (PP: 0.5329) — F1: 0.5481 (PP: 0.5524)<br>**All-in-One Raw V (6B)**: PR-AUC 0.5395 (PP: 0.5434) — F1: 0.5646 (PP: 0.5705) | 3 figuras (`nb04_fig_01` a `03`) + `val_predictions.csv` |
| **05 Função de Custo Real** | Avaliação financeira orientada por `val_predictions.csv` | 0.05 min | Fraudes coordenadas = 46.65% das perdas ($146.605).<br>**Custo Mínimo**: Baseline $175.432,64 (@ τ=0.35) vs Grafos $171.183,63 (@ τ=0.23).<br>**Economia Líquida**: $4.249,01 (-2.42%).<br>**Ganho de Recall em Fraudes Coordenadas**: +4.37%. | 2 figuras (`nb05_fig_01` a `02`) |
| **06 Walk-Forward (4 Dobras)** | Avaliação temporal contínua em 4 dobras expansivas mensais | 9.26 min | **Prejuízo Médio Mensal**:<br>• Baseline Tabular: $139.775,31/mês<br>• Apenas Grafos: $133.053,23/mês<br>**Economia Média Mensal Comprovada**: **$6.722,08/mês**.<br>Grafos superaram o baseline em PR-AUC e ROC-AUC em todas as 4 dobras temporais. | 1 figura (`nb06_fig_01`) |
| **07 Otimização & Significância** | Busca Bayesiana (Optuna 25 trials/config) + Bootstrap 1.000 iterações | 32.11 min | **Melhores Modelos Otimizados**:<br>• Baseline Otimizado: PR-AUC 0.5555<br>• Apenas Grafos Otimizado: PR-AUC 0.5622 (+0.0067)<br>• All-in-One Otimizado: PR-AUC 0.5519<br>**Custo Operacional Mínimo**:<br>• Baseline: $147.350,56 (@ τ=0.06)<br>• Apenas Grafos: $145.698,64 (@ τ=0.15)<br>**Economia Líquida Provada**: $1.651,92 (com limiar operacional muito mais equilibrado contra falsos positivos). | 1 figura (`nb07_fig_01`) |

Total de Figuras de Alta Resolução Salvas em `figures_notebooks/`: **23 figuras**.

---

## 6. Discussão Crítica e Limitações Metodológicas

### ⚠️ 6.1 Assimetria e Não-Causalidade dos Grafos Estáticos no Treino (Demonstração Empírica)

Uma das contribuições metodológicas mais ricas deste trabalho consiste em transformar o que seria apenas uma "limitação teórica" em um **resultado científico quantitativo comprovado**:

1. **A Não-Causalidade Intra-Treino e o Lookahead de Conectividade:**
   - O grafo bipartido de treino $\mathcal{G}_{\text{train}}$ é construído agregando todas as arestas do bloco histórico de 4 a 5 meses. Uma transação no início do treino recebe métricas de rede que já computaram as arestas que só ocorrerão meses adiante (*in-sample static lookahead*).
   - Para quantificar esse fenômeno, computamos o `device_degree` causal exato (contagem estrita de UIDs distintos observados no dispositivo até o timestamp $t_i$) e comparamos com o `device_degree` estático sobre as 49.815 transações com dispositivo no treino:

| Métrica Avaliada no Treino | `device_degree` Estático | `device_degree` Causal (até $t$) | Impacto / Distorção |
| :--- | :---: | :---: | :--- |
| **Média de Conectividade por Transação** | 7.575,6 UIDs | 3.963,5 UIDs | **+3.612,2 UIDs futuros** de lookahead médio |
| **Mediana** | 7.184,0 UIDs | 2.621,0 UIDs | Inflação de 2,74x na mediana |
| **Transações onde Estático > Causal** | — | — | **96,71% das transações** têm grau inflado |
| **Razão Média de Inflação ($\text{Estático} / \text{Causal}$)** | — | — | **7,39x de superestimação média** |
| **Correlação entre Versões** | — | — | $r = 0,7494$ |

* **Inflação Temporal por Decis Cronológicos no Treino:**
  - **Decil 0 (Início do Treino):** Grau causal médio = 503,3 vs. Grau estático médio = 8.177,3 $\to$ **Inflação de 16,25x** (99,2% das transações distorcidas).
  - **Decil 2 (1º Mês):** Grau causal médio = 2.462,4 vs. Grau estático médio = 8.073,8 $\to$ **Inflação de 3,28x**.
  - **Decil 5 (Meio do Treino):** Grau causal médio = 4.283,0 vs. Grau estático médio = 7.063,0 $\to$ **Inflação de 1,65x**.
  - **Decil 9 (Final do Treino):** Grau causal médio = 6.427,8 vs. Grau estático médio = 6.722,9 $\to$ **Inflação de 1,05x** (convergência natural no fim do treino).

2. **A Desconexão Estrutural na Validação Temporal (Nós Inéditos):**
   - Na validação ($N = 59.054$), o grafo é congelado do passado para prevenir vazamentos. A auditoria de conectividade revela:
     - **Ausência de Dispositivo (`null_device`):** 49.459 transações (**83,75%** da validação não possuem metadados de hardware).
     - **Dispositivos Inéditos na Validação:** Dos 653 dispositivos distintos na validação, **133 (20,37%) nunca haviam sido vistos no treino**, recebendo grau 0 e PageRank neutro uniforme $1/|\mathcal{V}|$.
     - **Transações Desconectadas:** Somando nulos e inéditos, **84,15% das transações da validação** não possuem conexão prévia conhecida na topologia de treino.
     - **Portadores Inéditos:** Dos 36.201 UIDs na validação, **23.804 (65,76%) são UIDs inéditos**, correspondendo a **61,47% de todas as transações da validação**.

3. **Por que a Topologia Isolada (Configuração 2) não superou o Baseline:**
   - Este resultado empírico soluciona o mistério da ablação: as árvores de decisão aprendem limiares no treino baseados em nós hiper-maduros (graus inflados em até 16x); quando chegam na validação temporal, deparam-se com uma massa de 84,15% de transações desprovidas de grafo maduro.
   - Isso explica por que a topologia isolada obtém PR-AUC de 0,4918 (inferior ao baseline de 0,5028), enquanto a velocidade causal 24h (`uid_tx_count_24h`), por ser estritamente retrospectiva em ambos os lados, não sofre essa distorção e atinge 0,4943.

4. **Implicação para Pesquisas Futuras:**
   - A superação dessa assimetria exige a migração para *Continuous-Time Dynamic Graph Neural Networks* (CTDGNNs, e.g., TGN - Temporal Graph Networks), onde as arestas e os embeddings de nós são atualizados dinamicamente a cada transação em streaming.

---

### ⚠️ 6.2 Granularidade de `DeviceInfo`: Perfil de Modelo vs. Dispositivo Físico e a Não-Monotonicidade do Risco

1. **A Natureza de `DeviceInfo` (Assinatura de Modelo vs. Hardware Físico Único):**
   - No benchmark IEEE-CIS, a coluna `DeviceInfo` não é um identificador biométrico ou de hardware unívoco (como IMEI, MAC Address ou *Device Fingerprint* criptográfico da ThreatMetrix/FingerprintJS).
   - Trata-se de uma string de texto extraída do cabeçalho User-Agent HTTP ou da compilação do sistema operacional (ex.: `Windows`, `iOS Device`, `MacOS`, `Trident/7.0`, `SM-G935F Build/NRD90M`).
   - Consequentemente, múltiplos usuários legítimos distintos compartilham exatamente o mesmo valor de `DeviceInfo`.

2. **Os Verdadeiros Megahubs da Rede (Popularidade de SO):**
   - Ao inspecionar os nós de maior grau na partição bipartida de dispositivos do treino, constata-se a dominância absoluta de plataformas de mercado:
     - `Windows`: conectou-se a **27.939 UIDs distintos** (taxa de fraude: **6,54%**)
     - `iOS Device`: conectou-se a **14.729 UIDs distintos** (taxa de fraude: **6,27%**)
     - `MacOS`: conectou-se a **9.577 UIDs distintos** (taxa de fraude: **2,21%**)
     - `Trident/7.0`: conectou-se a **5.668 UIDs distintos** (taxa de fraude: **1,29%**)
   - O grau elevado desses nós reflete a **popularidade massiva do sistema operacional**, e não conluio ou operação centralizada de fraudadores.

3. **O Fenômeno da Não-Monotonicidade da Taxa de Fraude:**
   - A correlação entre o grau de compartilhamento e o risco de fraude **não é estritamente monotônica**. Ao analisar a taxa de fraude por faixas de conectividade de portadores por `DeviceInfo`:
     - **1 UID (Exclusivo):** Taxa de fraude de ~3,8% a 5,7%
     - **2 UIDs:** Taxa de fraude de ~2,8% a 5,9%
     - **3–5 UIDs:** Taxa de fraude de ~7,5% a 9,5%
     - **6–10 UIDs:** Taxa de fraude de ~12,7% a 13,0%
     - **11–100 UIDs:** Taxa de fraude atinge o pico de **~14,9% a 15,1%** (modelos específicos de celulares ou builds incomuns de emuladores operados em campanhas coordenadas)
     - **100+ UIDs (Megahubs como Windows/iOS/MacOS):** A taxa de fraude despenca de volta para **5,8%** (diluída por dezenas de milhares de compras legítimas de e-commerce).
   - Essa inversão no topo explica por que o grupo mais densamente conectado não apresenta a maior taxa de fraude: nós com conectividade astronômica são, em sua esmagadora maioria, plataformas legítimas de grande porte.

4. **Revisão do Estudo de Caso (O Exemplo do `SM-G935F`):**
   - O código `SM-G935F` corresponde ao smartphone comercial **Samsung Galaxy S7 Edge**, um dos aparelhos mais vendidos no mundo no período de coleta do dataset (2016–2018).
   - No benchmark, a string `SM-G935F Build/NRD90M` possui 334 transações e conecta-se a 221 UIDs distintos. Interpretar esse nó como um "único hardware físico compartilhado por fraudadores em conluio" é um equívoco conceitual: seu grau elevado decorre principalmente do sucesso de vendas do aparelho no mercado consumidor.
   - **Blindagem Teórica:** O ganho preditivo do algoritmo de árvores de decisão (XGBoost/LightGBM) não decorre da premissa ingênua de que "quanto maior o grau, maior a fraude", mas sim da sua capacidade de realizar divisões não-lineares ortogonais — isolando perfis anômalos de média conectividade e cruzando `device_degree` e PageRank com atributos temporais de velocidade (`uid_tx_count_24h`) e dispersão financeira.

---

### ⚠️ 6.3 Auditoria Metodológica de Recall em Fraudes Coordenadas: FPR Fixo vs. Distorção de Limiar

1. **A Fragilidade da Comparação em Limiares Variáveis:**
   - No relatório preliminar do Notebook 05, reportou-se um ganho aparente de **+4,37%** em recall sobre fraudes coordenadas ao comparar o modelo Baseline com o modelo aumentado por Grafos.
   - Contudo, uma auditoria rigorosa de sensibilidade revela que o Baseline operava sob seu ponto financeiro ótimo $\tau^* = 0,35$ (disparando alertas em 4,02% das transações), enquanto o modelo com grafos operava em $\tau^* = 0,23$ (disparando alertas em 5,47% das transações).
   - Parte substancial do ganho aparente provinha, portanto, do aumento no volume total de alertas (gerando mais falsos positivos operacionais), e não de uma pura capacidade seletiva de rede.

2. **O Protocolo Metodológico de Avaliação sob FPR Constante:**
   - Para mensurar a real capacidade discriminatória estrutural, fixou-se a taxa de falsos positivos em limites operacionais industriais estritos ($\text{FPR} = 1,0\%$, $2,0\%$, $3,0\%$ e $5,0\%$).
   - A tabela abaixo apresenta o recall real obtido sobre as 59.054 transações de validação sob $\text{FPR} = 1,0\%$ (exatamente 570 falsos positivos permitidos na esteira de produção):

| Configuração Avaliada | Recall Geral (FPR=1%) | Recall Fraudes Coordenadas ($N=1.260$) | Recall Fraudes Isoladas ($N=846$) | Contribuição Metodológica do Lift Coordenado |
| :--- | :---: | :---: | :---: | :--- |
| **1B. Baseline PCA (15-d)** | 39,51% | 50,87% | 28,01% | *Ponto de Controle Inicial* |
| **5. Apenas FE Tabular** | 44,25% | 58,41% | 29,91% | **+7,54 p.p.** (Motor Principal: Entidade & Tempo) |
| **6B. All-in-One Raw V** | 46,47% | 60,63% | 32,15% | **+2,22 p.p.** adicionais (Grafos Bipartidos) |
| **6B PP. All-in-One + Pós-Proc Causal** | **47,15%** | **61,27%** | **32,86%** | **+0,64 p.p.** adicionais (Filtro Histórico Causal) |
| **Lift Total (6B PP vs 1B)** | **+7,64 p.p.** | **+10,40 p.p.** | **+4,85 p.p.** | **72,5% do lift vem de FE Tabular; 27,5% de Grafos/PP** |

3. **Intervalos de Confiança via Bootstrap (1.000 iterações) para os Deltas de Recall:**

| Estágio Metodológico | Recall Coordenadas (%) | IC 95% Bootstrap | $\Delta$ Recall (p.p.) | IC 95% do $\Delta$ | $p$-valor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline PCA (1B)** | 50,87% | [48,10%, 53,65%] | — | — | — |
| **+ FE Tabular (5)** | 58,41% | [55,56%, 61,11%] | **+7,54 p.p.** | [+5,08, +9,68] | $p < 0,001$ |
| **+ Grafos Bipartidos (6B bruto)** | 60,63% | [57,78%, 63,26%] | **+2,22 p.p.** | [+1,03, +3,73] | $p < 0,001$ |
| **+ Pós-Processamento Causal (6B PP)** | 61,27% | [58,49%, 63,97%] | **+0,64 p.p.** | [+0,00, +1,43] | $p = 0,048$ |
| **Ganho Acumulado Total ($1\text{B} \to 6\text{B PP}$)** | **61,27%** | [58,49%, 63,97%] | **+10,40 p.p.** | **[+8,02, +12,78]** | **$p < 0,001$** |

4. **Análise de Significância do Pós-Processamento (+0,64 p.p.):**
   - O ganho de +0,64 p.p. proporcionado pelo filtro causal expansivo corresponde à captura adicional de exatamente **8 transações coordenadas** (em um universo de 1.260 fraudes coordenadas na partição de validação).
   - Como o limite inferior do IC 95% Bootstrap toca exatamente zero ([+0,00, +1,43] p.p.), trata-se de um efeito marginal e limítrofe no nível de 5%.
   - Por outro lado, os ganhos da Engenharia Tabular (+7,54 p.p., IC 95%: [+5,08, +9,68]) e dos Grafos Bipartidos (+2,22 p.p., IC 95%: [+1,03, +3,73]) são estritamente positivos e robustamente significativos ($p < 0,001$).

5. **Ressalva Metodológica de Circularidade e Definição de Coordenada:**
   - A rotulagem de "fraude coordenada" como $(\text{is\_shared\_device} \ge 2 \lor \text{velocity}_{24h} \ge 2)$ é um *surrogate label* (rótulo operacional aproximado), dada a inexistência de rótulos externos de quadrilhas (*fraud rings*) no benchmark IEEE-CIS.
   - Como o modelo inclui como preditores exatamente `device_degree` e `uid_tx_count_24h`, há uma circularidade intencional na avaliação: mede-se a capacidade do classificador de recuperar instâncias que apresentam os mesmos comportamentos estruturais e temporais que as features representam, sob restrição de falsos alarmes ($\text{FPR} = 1\%$).

6. **Síntese Metodológica Definitiva:**
   - O ganho em fraudes coordenadas existe e é estatisticamente demonstrável (**+10,40 p.p.** de recall a $\text{FPR} = 1,0\%$, IC 95%: [+8,02, +12,78] p.p.).
   - Entretanto, a decomposição causal estabelece com clareza científica: a ancoragem de entidade tabular ($D_n$, UID Mágico e agregações de portador) é o motor primário do ganho (**+7,54 p.p.** ou 72,5% do lift de recall), enquanto a topologia de grafos bipartidos agrega **+2,22 p.p.** (21,3% do lift), e o pós-processamento causal adiciona um refinamento marginal de **+0,64 p.p.** (6,2% do lift).



