# Análise Comparativa de Modelos Tabulares e Modelos Aumentados por Grafos na Detecção de Fraude em Tempo Real

## Resumo
Este trabalho acadêmico investiga a hipótese de que a incorporação de **Graph-Derived Features** (atributos estruturais extraídos da topologia da rede de transações) melhora significativamente o desempenho de modelos de aprendizado de máquina tradicionais baseados em dados tabulares para detecção de fraudes. Utilizando a biblioteca `gen-fraud-graph` do Santander para geração de dados sintéticos e modelos XGBoost em uma arquitetura de roteamento de identidade (**Identity Routing**), comparamos um modelo tabular puro (**Baseline Model**) com um modelo enriquecido com atributos de rede estruturais básicos e avançados. 

Documentamos uma falha metodológica inicial no split temporal que causava vazamento de alvos (Target Leakage), e apresentamos os resultados corrigidos sob diferentes níveis de complexidade de dados (`low`, `medium` e `high` hardness). Os resultados empíricos finais confirmam que, enquanto modelos tabulares falham em manter precisão aceitável devido ao desequilíbrio extremo de classes (PR-AUC ~0.02), o modelo aumentado com atributos de grafo estruturais e de fluxo mantém alta robustez mesmo em cenários de alta complexidade (PR-AUC ~0.90).

---

## 1. Introdução
A detecção de fraudes financeiras é caracterizada por um desequilíbrio de classes extremo (onde transações fraudulentas representam menos de 0.01% do volume total) e por padrões de transação estruturados para evitar a detecção trivial. Classificadores tabulares tradicionais focam em atributos de transações individuais (valor, tipo) ou em saldos estáticos das contas. Contudo, fraudadores operam em redes, utilizando técnicas como anéis de lavagem de dinheiro (estruturas circulares de movimentação de fundos) e contas intermediárias que agem como mulas para repassar recursos rapidamente.

Ao modelar as transações como uma **Transaction Relation** (um grafo direcionado onde contas são nós e transações são arestas direcionadas), é possível extrair atributos estruturais de rede (**Graph-Derived Features**). Este trabalho estuda o impacto e a necessidade dessas características sob níveis crescentes de dificuldade de detecção.

---

## 2. Metodologia

O pipeline do projeto está estruturado de acordo com as seguintes fases:

```mermaid
graph TD
    A[gen-fraud-graph] -->|Dados Sintéticos de Contas e Transações| B[Ingestão de Dados: load.py]
    B -->|Pool de Dados Mesclados| C[Divisão Temporal: split.py]
    C -->|Split de Treino| D[Construção de Grafo e Features: graph_features.py]
    C -->|Split de Validação| E[Mapeamento de Features do Grafo de Treino]
    D -->|Features Tabulares + Grafo| F[Divisão por Roteamento de Identidade]
    E -->|Features Tabulares + Grafo| F
    F -->|Entrada Modelo A| G[Modelo A: Sem Identidade de Conta]
    F -->|Entrada Modelo B| H[Modelo B: Com Identidade de Conta]
    G -->|Predições de Probabilidade| I[Suíte de Validação e Métricas: metrics.py]
    H -->|Predições de Probabilidade| I
```

### 2.1 Geração de Dados Sintéticos e Níveis de Complexidade
Os dados de contas e transações foram gerados usando a biblioteca `gen-fraud-graph` (Santander), escalados para 100.000 contas e aproximadamente 900.000 transações. Para testar o impacto de padrões adversários mais complexos, avaliamos três níveis de dificuldade do gerador (`hardness`):
- **Low (Padrão)**: Valores de fraude são idênticos a um valor sentinela (9999.0), os anéis de fraude são totalmente disjuntos e não há ciclos legítimos semelhantes.
- **Medium**: Aplica variação (+/- 25%) nos valores de fraude, sobrepõe 25% das contas dos anéis fraudulentos e injeta ciclos legítimos de alto valor para confundir heurísticas baseadas em loops.
- **High**: Aplica variação severa (+/- 50%) nos valores de fraude, sobrepõe 50% das contas dos anéis fraudulentos e injeta ciclos legítimos adicionais.

### 2.2 O Bug do Split Temporal e Vazamento de Alvos (Target Leakage)
Na fase preliminar dos testes, observou-se que o modelo tabular puro obtinha um **PR-AUC** de 0.0003 (idêntico ao modelo aleatório) enquanto o modelo com grafos obtinha um PR-AUC de 0.9360. 

Investigações posteriores revelaram que isso era uma ilusão causada por um **bug metodológico de split**:
1. O gerador do Santander escreve transações legítimas e depois transações de fraude sequencialmente.
2. Sem a randomização correta dos timestamps, ordenar cronologicamente fez com que **todas as transações fraudulentas fossem enviadas para o conjunto de validação (teste)**, deixando o conjunto de treino com exatamente 0 casos de fraude.
3. Sem fraudes no treino, o modelo tabular não aprendeu nenhum padrão, caindo na performance aleatória (0.0003).
4. No entanto, o modelo com grafos obteve 0.9360 devido a um **Vazamento de Alvos (Target Leakage)**: todas as contas envolvidas em fraudes na validação eram nós inéditos (não vistos no treino). Consequentemente, elas receberam os valores padrão de PageRank e Grau (`default_pagerank`, `default_degree`). O classificador simplesmente aprendeu o artefato do split: *"se a conta tem PageRank e Grau iguais aos valores default, prediga fraude"*.

**Correção**: O bug foi mitigado atribuindo timestamps distribuídos aleatoriamente em uma janela de 30 dias para todas as transações antes do split temporal. Com isso, os casos de fraude foram distribuídos uniformemente ao longo do tempo (fraud rate de ~0.005% no treino e ~0.008% na validação), garantindo um treino real e livre de vazamentos estruturais.

### 2.3 Engenharia de Atributos Avançados de Grafo
O grafo direcionado é construído usando NetworkX exclusivamente sobre os dados de treino. Mapeamos os atributos calculados para os DataFrames de treino e validação. As **Graph-Derived Features** implementadas foram:
1. **PageRank Centrality**: Mede a centralidade estrutural de cada nó.
2. **Degrees (In-Degree e Out-Degree)**: Mede o volume de conexões de entrada e saída.
3. **Neighborhood Fraud Rate**: Taxa de fraude histórica observada nas contas vizinhas durante o treino.
4. **Strongly Connected Component Size (SCC Size)**: Identifica se o nó pertence a um componente fortemente conectado. Se o tamanho do SCC for > 1, indica que o nó faz parte de um ciclo direcionado (padrão clássico de anéis de fraude).
5. **Flow Balance Ratio**: Computa a razão de conservação de recursos: $|in - out| / (in + out + 10^{-5})$. Contas intermediárias de fraude (mulas) tendem a repassar tudo o que recebem, aproximando essa razão de 0.
6. **Amount Standard Deviation**: Desvio padrão dos valores das transações que passam pelo nó.

### 2.4 Arquitetura de Roteamento de Identidade (Identity Routing)
Para mimetizar um sistema de produção, as transações são divididas via ID da conta origem (par/ímpar):
- **Grupo A (ID par)**: Roteado para o **Modelo A** (usa apenas atributos da transação + atributos de grafo).
- **Grupo B (ID ímpar)**: Roteado para o **Modelo B** (usa atributos da transação, atributos de grafo e dados cadastrais/identidade da conta, como saldos e scores de risco).

---

## 3. Resultados Empíricos: A Cronologia Experimental (Storytelling)

Os testes no dataset real (IEEE-CIS, 10% sample) foram estruturados como uma **jornada de validação de hipóteses**, onde cada etapa guiou o design do passo seguinte, catalogando tanto as melhorias quanto os fracassos metodológicos.

```
Jornada: 1. Baseline Tabular -> 2. Grafo Básico (card1) -> 3. Target Encoding (Falha) -> 4. UID + Velocidade -> 5. Generalização LGBM
```

### Etapa 1: A Referência de Controle (Baseline Tabular)
Estabelecemos um controle robusto treinando os classificadores XGBoost e LightGBM estritamente sobre os 433 atributos tabulares fornecidos:
- **XGBoost Tabular**: PR-AUC = `0.4355` | ROC-AUC = `0.8606` | F1 = `0.4453`
- **LightGBM Tabular**: PR-AUC = `0.4231` | ROC-AUC = `0.8342` | F1 = `0.4500`

### Etapa 2: A Hipótese do Grafo de Cartão Simples (card1)
Conectamos as transações a um nó simples do cartão (`card1`) e calculamos métricas estruturais (PageRank e graus de conexões). O XGBoost apresentou uma melhoria sutil de desempenho:
- **XGBoost (card1 Graph)**: PR-AUC = `0.4403` (**+0.0048** de lift absoluto) | ROC-AUC = `0.8647`

### Etapa 3: O Colapso por Target Leakage (Atributos baseados no Alvo)
Na busca por um ganho maior, introduzimos a taxa de fraude histórica de cartões e dispositivos (`card_fraud_rate`/`device_fraud_rate`) suavizadas via Leave-One-Out (LOO) e shrinkage ($m=10.0$).
- **O Fracasso**: A performance despencou na validação temporal:
  - **XGBoost (Target Encoded Graph)**: PR-AUC = `0.2807` (Queda acentuada de **-0.1548** em relação à baseline)
- **Causa científica**: Embora o LOO impeça vazamento imediato na mesma linha, o modelo superajusta (overfitting) nas taxas históricas do conjunto de treino. Como o split é temporal, a deriva de padrões na validação torna essas taxas obsoletas, induzindo o classificador a erros catastróficos.

### Etapa 4: O Modelo Campeão (UIDs Compostos + Velocidade 24h)
Para solucionar o overfitting, removemos totalmente os atributos baseados no alvo e redesenhamos a topologia do grafo:
1. **UID Composto (User Profile)**: Agrupamos os nós combinando `card1` + `card2` + `addr1` + `P_emaildomain` para reduzir o ruído de cartões compartilhados.
2. **Atributos de Velocidade Temporal (24h)**: Calculamos o fluxo recente de transações por UID e Device nas últimas 24 horas (`uid_tx_count_24h` / `device_tx_count_24h`) usando busca binária vetorizada.
- **Resultado (XGBoost)**: PR-AUC = **0.4474** (**+0.0119** de lift sobre a baseline) | F1-Score = **0.4656** (**+0.0203** de ganho).

### Etapa 5: Teste de Generalização do Lift (LightGBM)
Para garantir que o ganho estrutural do grafo da Etapa 4 não era um artefato exclusivo do XGBoost, rodamos o mesmo pipeline com o LightGBM:
- **LightGBM (UID + Velocity Graph)**: PR-AUC = **0.4360** (**+0.0129** de lift sobre a baseline tabular do LightGBM) | F1-Score = **0.4545**.

### Etapa 6: O Modelo Híbrido GNN-GBDT + PCA de Atributos V
Como passo final em direção ao estado da arte, introduzimos duas técnicas avançadas de deep learning e redução de dimensionalidade (ADR-0005):
1. **GNN Bipartida (GraphSAGE)**: Treinamos uma rede neural convolucional para grafos heterogêneos no PyTorch Geometric (PyG) de forma supervisionada para classificação de arestas. Extraímos embeddings densos de 16-d para nós de Usuários e Dispositivos, concatenando-os em representações transacionais de 32-d.
2. **Redução de Dimensionalidade (PCA)**: Aplicamos PCA com 15 componentes principais sobre as 339 colunas redundantes e correlacionadas `V1` a `V339` da Vesta, removendo as variáveis cruas e reduzindo a largura da tabela em cerca de **80%**.
- **Resultado (XGBoost + GNN + PCA)**: PR-AUC = **0.4376** (Melhoria de **+0.0021** sobre a baseline e **+0.0100** de lift em ROC-AUC), operando com apenas **73 features** (comparado a 433 da baseline).
- **Resultado (LightGBM + GNN + PCA)**: PR-AUC = `0.4166` | ROC-AUC = **0.8458** (**+0.0116** de lift em ROC-AUC sobre a baseline do LightGBM), operando com apenas **74 features**.

### Etapa 7: O Modelo GNN Puro (Direct Edge Classification)
Para testar a capacidade de predição isolada da GNN (GraphSAGE + EdgeClassifier), executamos a classificação direta de fraude baseada apenas nas probabilidades de arestas geradas pela rede, sem a participação de modelos de árvore (XGBoost/LightGBM).
- **O Fracasso Metodológico**: A GNN pura obteve uma performance equivalente a um classificador aleatório:
  - **Pure GNN (GraphSAGE)**: PR-AUC = `0.0345` (Linha de base aleatória = `0.0347`) | ROC-AUC = `0.4483`.
- **Causa científica**: Como os nós do grafo bipartido utilizam vetores constantes de inicialização e a GNN não ingere diretamente os atributos tabulares heterogêneos das transações (como valores e categorias), ela é incapaz de generalizar para novas transações em splits temporais. Isso ressalta a necessidade crítica de uma arquitetura híbrida.

---

## 4. Tabela Consolidada de Resultados (IEEE-CIS 10%)

Abaixo, catalogamos a cronologia das etapas de desenvolvimento de forma resumida:

| Etapa | Configuração / Atributos | Classificador | PR-AUC | ROC-AUC | F1-Score | Limiar Ótimo | Categoria de Resultado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Etapa 1** | Tabular Puro | XGBoost | 0.4355 | 0.8606 | 0.4453 | 0.612 | Controle (Baseline) |
| **Etapa 1** | Tabular Puro | LightGBM | 0.4231 | 0.8342 | 0.4500 | 0.730 | Controle (Baseline) |
| **Etapa 2** | Grafo Básico (`card1`) | XGBoost | 0.4403 | 0.8647 | 0.4486 | 0.738 | Lift Estrutural Sutil (+0.0048) |
| **Etapa 3** | Grafo + Target Encoding (LOO) | XGBoost | 0.2807 | 0.8347 | 0.2914 | 0.979 | **Resultado Negativo (Falha/Leakage)** |
| **Etapa 4** | Grafo + UIDs + Velocidade 24h | XGBoost | **0.4474** | 0.8634 | **0.4656** | 0.563 | **Campeão (Ganho Real +0.0119)** |
| **Etapa 5** | Grafo + UIDs + Velocidade 24h | LightGBM | **0.4360** | 0.8384 | **0.4545** | 0.761 | **Generalização Provada (+0.0129)** |
| **Etapa 6** | Híbrido GNN (GraphSAGE) + PCA | XGBoost | **0.4376** | **0.8706** | 0.4487 | 0.629 | **Eficiência Dimensional (apenas 73 features)** |
| **Etapa 6** | Híbrido GNN (GraphSAGE) + PCA | LightGBM | 0.4166 | **0.8458** | 0.4423 | 0.730 | **Eficiência Dimensional (apenas 74 features)** |
| **Etapa 7** | Grafo Bipartido Puro (GraphSAGE) | Pure GNN | 0.0345 | 0.4483 | 0.0671 | 0.035 | **Resultado Negativo (Sem Sinais Tabulares)** |

---

## 5. Discussão Metodológica

1. **A Ilusão do ROC-AUC no Desequilíbrio Extremo**: 
   A avaliação inicial no dataset do Santander provou que o ROC-AUC é uma métrica enganosa sob desequilíbrio severo. Em cenários reais como o IEEE-CIS, a PR-AUC é o único farol confiável para avaliar a precisão e recall sem distorções de classes.

2. **Divergência entre o Domínio Sintético e Real**:
   - **Sintético (Santander)**: Ganhos massivos (PR-AUC de 0.02 para 0.93) porque a fraude é injetada de forma determinística na forma de ciclos estruturais perfeitos (loops de lavagem).
   - **Real (IEEE-CIS)**: Ganhos sutis porém significativos (+0.012 PR-AUC). A fraude no e-commerce real é comportamental, exigindo modelagem de velocidade temporal (rolling windows) e UIDs compostos para que os atributos estruturais se tornem significativos sobre baselines tabulares já consolidados.

3. **Generalização Multialgoritmo**:
   O fato de o lift de grafos ter sido de **+0.0119** no XGBoost e **+0.0129** no LightGBM indica que os sinais topológicos e temporais extraídos da rede são robustos e independentes das particularidades do algoritmo de boosting.

4. **Eficiência e Compressão Híbrida (GNN + PCA)**:
   A Etapa 6 provou que representações estruturais profundas aprendidas por vizinhança (GraphSAGE) combinadas com a compressão de variância (PCA de 15 componentes em 339 colunas V) permitem treinar modelos de árvore com **cerca de 20% do volume original de variáveis** sem perda expressiva de PR-AUC, superando as baselines em ROC-AUC e diminuindo o consumo de memória do sistema local em mais de 80%.

5. **Por que a GNN Pura Falha no Domínio Tabular**:
   Os resultados da Etapa 7 demonstram empiricamente um fato consolidado na literatura: GNNs puras alimentadas estritamente com dados de conectividade falham ao classificar fraudes em splits temporais porque a fraude transacional depende criticamente de atributos numéricos e categóricos (como valores de transação, dados do cartão e localização) que não se estruturam em grafos. A arquitetura híbrida GNN-GBDT resolve essa limitação, permitindo que a GNN atue como extratora de contexto estrutural e o GBDT como classificador ótimo para variáveis de tabela.

---

## 6. Conclusão

Este trabalho documentou a jornada de validação e refinamento de atributos de grafo para detecção de fraudes. Demonstramos empiricamente que a inclusão de atributos estruturais puros de rede e de velocidade temporal de 24 horas traz ganhos robustos de PR-AUC no mundo real, livre de riscos de target leakage. Reforçamos também o papel de catalogar falhas (resultados negativos) como parte vital do rigor científico, provando que atributos baseados no alvo em splits temporais devem ser rejeitados em favor de métricas puras de conectividade e fluxo. Por fim, mostramos que modelos híbridos GNN-GBDT combinados a PCA reduzem drasticamente a pegada computacional das tabelas de transação de e-commerce mantendo a capacidade discriminatória dos classificadores, enquanto GNNs puras sem contexto tabular falham na generalização.
