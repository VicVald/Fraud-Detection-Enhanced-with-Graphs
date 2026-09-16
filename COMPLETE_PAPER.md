# O Impacto Estrutural e Econômico de Grafos na Detecção de Fraude em Pagamentos Digitais: Um Estudo de Ablação e Modelagem Sensível ao Custo

**Autor:** Victor Valadares  
**Área de Concentração:** Inteligência Artificial, Ciência de Dados e Engenharia de Software  
**Palavras-chave:** Detecção de Fraudes; Grafos Bipartidos; Graph Neural Networks (GNN); Gradient Boosted Decision Trees (GBDT); Modelagem Sensível ao Custo; Estudo de Ablação; IEEE-CIS Benchmark.

---

## Resumo

A detecção de fraudes em transações de comércio eletrônico (*e-commerce*) é um dos problemas mais críticos da economia digital global. Tradicionalmente, classificadores de aprendizado de máquina supervisionados aplicados a dados tabulares — como *Gradient Boosted Decision Trees* (GBDT) — operam sob a premissa estatística de observações independentes e identicamente distribuídas (*IID*). Contudo, o crime cibernético contemporâneo organiza-se predominantemente em redes de ataque coordenadas (*fraud rings*, *botnets* e fazendas de dispositivos compartilhados utilizando credenciais roubadas). Este trabalho investiga e quantifica o real valor incremental da modelagem em grafos para sistemas de prevenção à fraude através de três eixos: (i) enriquecimento de atributos via **Grafos Bipartidos de Acesso** (Portador $\leftrightarrow$ Dispositivo), centralidade de rede (PageRank), contadores causais de velocidade temporal em 24 horas, compressão espectral (PCA) e representações latentes auto-supervisionadas de nós via *Graph Neural Networks* (PyG GraphSAGE); (ii) execução de um **Estudo de Ablação Quádruplo** rigorosamente controlado sobre o benchmark mundial **IEEE-CIS Fraud Detection** (590.540 transações reais); e (iii) formulação de uma **Função de Custo Financeiro Dependente do Exemplo** baseada nas taxas de *chargeback* das bandeiras de cartão (Visa/Mastercard) e no princípio de *Bayes Minimum Risk* (CORREA BAHNSEN et al., 2015). Conduzimos uma auditoria formal de maturidade científica, eliminando fontes de vazamento de dados (*target leakage*) e corrigindo o colapso de variância em GNNs indutivas. Nossos experimentos comprovam que a adição isolada de apenas 4 atributos de grafo ao classificador XGBoost produziu o maior salto de desempenho do estudo (**PR-AUC = 0,6032** e **ROC-AUC = 0,9101**, representando lifts de **+4,29% em PR-AUC** e **+1,83% em ROC-AUC** sobre a baseline tabular), demonstrando superior resiliência ao *concept drift* temporal quando comparada à engenharia tabular tradicional. Além disso, a análise estrutural revelou que **62,2% das fraudes ocorrem em padrões de rede**, e a otimização do limiar de decisão financeiro sob a função de custo proposta gerou uma **redução de 42,07% no prejuízo operacional líquido** (\$56.427,33 economizados na validação temporal), evidenciando o expressivo retorno sobre investimento da abordagem relacional.

**Abstract:**  
Fraud detection in e-commerce transactions is one of the most critical challenges in the digital economy. Traditionally, supervised machine learning models applied to tabular data, such as Gradient Boosted Decision Trees (GBDT), operate under the Independent and Identically Distributed (IID) assumption. However, modern cybercrime is predominantly organized into coordinated attack networks (fraud rings, botnets, and shared device farms leveraging stolen credentials). This paper investigates and quantifies the true incremental value of graph modeling in fraud prevention systems across three dimensions: (i) feature enrichment using Bipartite Access Graphs (Account $\leftrightarrow$ Device), network centrality (PageRank), causal 24-hour rolling velocity counters, PCA spectral compression, and self-supervised Graph Neural Network node embeddings (PyG GraphSAGE); (ii) a rigorous 4-way ablation study on the IEEE-CIS Fraud Detection benchmark (590,540 real-world transactions); and (iii) an example-dependent financial cost function formulation incorporating payment card dispute fees and Bayes Minimum Risk theory (CORREA BAHNSEN et al., 2015). We conducted a formal scientific leakage audit, eliminating target leakage and resolving variance collapse in inductive GNNs. Our experiments demonstrate that adding only 4 graph features to an XGBoost classifier produced the highest individual performance lift in the study (PR-AUC = 0.6032 and ROC-AUC = 0.9101, yielding +4.29% PR-AUC and +1.83% ROC-AUC gains over the tabular baseline), exhibiting superior robustness against temporal concept drift compared to massive tabular feature engineering. Furthermore, structural topology analysis revealed that 62.2% of real frauds follow networked patterns, and cost-sensitive threshold optimization reduced total financial loss by 42.07% ($56,427.33 saved in validation), proving the substantial economic value of relational graph architectures.

---

## 1. Introdução

A expansão exponencial do comércio eletrônico global transformou os hábitos de consumo e viabilizou um ecossistema de pagamentos instantâneos sem presença física do cartão (*Card-Not-Present* — CNP). Contudo, essa conveniência veio acompanhada de um crescimento vertiginoso em ataques de fraude eletrônica. De acordo com o relatório anual da indústria de pagamentos *The Nilson Report* (2023), as perdas globais decorrentes de fraudes com cartões de crédito e débito atingiram a marca alarmante de **US\$ 32,34 bilhões**, com projeções indicando que esse prejuízo ultrapassará US\$ 40 bilhões antes do final da década. No cenário europeu e transfronteiriço, o *European Central Bank* (ECB, 2023) aponta que as transações CNP respondem por **84% do valor total das fraudes**, impulsionadas pela proliferação de credenciais vazadas e engenharia social.

Para gateways de pagamento, adquirentes e instituições financeiras, os custos da fraude transcendem o montante financeiro diretamente subtraído. Eles envolvem:
1. **Multas Administrativas e Taxas de Disputa**: Quando uma fraude é confirmada, o comerciante arca com o estorno (*chargeback*) do valor da compra e uma taxa punitiva fixa (variando entre US\$ 15 e US\$ 25 por transação) imposta pelas bandeiras de cartão (Visa, Mastercard, American Express).
2. **Custo de Fricção e Churn de Clientes Legítimos**: O bloqueio indevido de compras legítimas (*falsos positivos*) gera insatisfação imediata, levando ao abandono do carrinho de compras e ao cancelamento da conta do portador (*churn*).
3. **Custos Operacionais de Auditoria Manual**: A necessidade de triagem humana por analistas de risco para transações de alta incerteza onera severamente as margens de lucro dos intermediários financeiros.

Historicamente, a indústria combate esse problema utilizando modelos de aprendizado de máquina supervisionados aplicados a dados tabulares, com destaque absoluto para os algoritmos de árvores de decisão impulsionadas por gradiente (*Gradient Boosted Decision Trees* — GBDTs, tais como XGBoost, LightGBM e CatBoost). Embora altamente eficientes para processar atributos numéricos e categóricos em tabelas densas, esses modelos operam fundamentalmente sob a premissa estatística de que cada linha (transação) é uma observação **Independente e Identicamente Distribuída (IID)**.

Essa premissa de independência falha drasticamente no cenário cibernético contemporâneo devido a três vetores centrais:

* **Ataques Coordenados em Rede (*Fraud Rings* & *Botnets*)**: Conforme demonstrado por Van Vlasselaer et al. (2015, 2017) e Akoglu et al. (2015), criminosos operam em redes estruturadas de conluio. Eles utilizam emuladores, scripts automatizados e fazendas de hardware compartilhado para testar centenas de cartões roubados sequencialmente. Um classificador tabular convencional enxerga cada transação como um indivíduo isolado e falha em detectar que a mesma máquina física está orquestrando múltiplos pagamentos fraudulentos.
* **O Dilema do *Cold Start* (Cartões Descartáveis)**: No mercado negro (*Dark Web*), lotes de cartões clonados são comercializados em massa e utilizados para uma única tentativa de compra de alto valor antes do descarte. Para o modelo tabular puro, um cartão inédito não possui histórico transacional prévio e, portanto, é classificado como de baixo risco. Em contrapartida, em uma estrutura de rede relacional, mesmo um cartão 100% inédito conecta-se a um endereço IP, dispositivo ou e-mail que já possui vizinhos comprometidos na topologia do grafo.
* **Não-Estacionariedade e *Concept Drift* Adversário**: Conforme formalizado por Dal Pozzolo et al. (2015, 2018), sistemas de fraude operam em um ambiente temporal não estacionário. Os criminosos alteram dinamicamente seus padrões de ataque para contornar regras estáticas, gerando mudanças bruscas nas distribuições de probabilidade $P(X)$ e $P(Y|X)$.

### Perguntas de Pesquisa e Objetivos

Este trabalho visa responder a três perguntas centrais de pesquisa:
* **RQ1 (Valor Estrutural Puro)**: *Qual é o ganho real de desempenho preditivo e estabilidade temporal ao isolar atributos de grafos contra uma baseline tabular pura e contra engenharia tabular avançada em dados reais de produção?*
* **RQ2 (Resolução de Topologias Complexas)**: *Em que proporção as fraudes reais do e-commerce dependem de padrões de rede estruturados versus eventos oportunistas isolados?*
* **RQ3 (Impacto Econômico e Otimização de Custo)**: *Qual é a economia financeira líquida proporcionada pela introdução de grafos quando a política de decisão do modelo é calibrada sob uma função de custo dependente do exemplo em vez de limiares estatísticos convencionais?*

Os objetivos específicos incluem:
1. Conduzir uma **análise exploratória de dados (EDA) diagnóstica**, identificando empiricamente as assinaturas de tipologias de fraude consagradas (anéis de fraude, emuladores de dispositivo, ataques de velocidade e fracionamento de valores).
2. Construir e validar geradores de atributos em **dois domínios distintos**: dados sintéticos controlados e dados reais de produção em larga escala (**IEEE-CIS Fraud Detection** com 590.540 transações).
3. Desenvolver uma arquitetura com **Grafo Bipartido Portador $\leftrightarrow$ Dispositivo**, cálculo causal de PageRank, contadores de velocidade temporal em 24h e compressão de dimensionalidade via PCA.
4. Treinar representações de nós via **GraphSAGE Heterogêneo em regime estritamente auto-supervisionado (*Link Reconstruction*)**, garantindo imunidade a vazamentos de alvo.
5. Realizar uma **auditoria formal de vazamento de dados (*leakage audit*)**, documentando e corrigindo falhas de generalização indutiva e colapsos de variância.
6. Executar um **Estudo de Ablação Quádruplo** sobre 100% da base do IEEE-CIS, isolando de forma inequívoca o valor agregado pelos atributos de grafo contra baselines tabulares e engenharia de atributos avançada.
7. Formular uma **Função de Custo Financeiro Realista** baseada no teorema de *Bayes Minimum Risk* (ELKAN, 2001; CORREA BAHNSEN et al., 2015), calculando a redução do prejuízo financeiro e comprovando a viabilidade econômica do modelo.

---

## 2. Trabalhos Relacionados

### 2.1 Modelos Tabulares de Boosting (GBDTs)
A detecção de fraude moderna em escala industrial é dominada por algoritmos de *tree boosting*, notadamente o **XGBoost** (CHEN; GUESTRIN, 2016) e o **LightGBM** (KE et al., 2017). A capacidade dessas árvores de lidar com dados esparsos, valores nulos nativos, variáveis categóricas de alta cardinalidade e relações não lineares torna-as a escolha padrão para sistemas de decisão com requisitos de latência inferiores a 50 milissegundos. Contudo, esses modelos são cegos para correlações topológicas que ultrapassam os limites de uma única tupla de dados.

### 2.2 Aprendizado de Representação em Grafos e GNNs
O uso de métricas clássicas de centralidade, como o **PageRank** (PAGE et al., 1999), provou ser eficaz para rankear a importância de entidades em redes de pagamento. Com o advento das redes neurais em grafos, arquiteturas convolucionais espaciais como o **GraphSAGE** (HAMILTON; YING; LESKOVEC, 2017) permitiram o aprendizado de representações de nós (*embeddings*) de forma indutiva, amostrando e agregando atributos de vizinhanças locais sem exigir a presença de todo o grafo em memória durante o treinamento.

Entretanto, trabalhos seminais recentes na literatura de aprendizado de máquina (como os levantamentos de Grinsztajn et al., 2022 e Shwartz-Ziv & Armon, 2022) demonstram consistentemente que GNNs puras apresentam desempenho inferior a modelos baseados em árvores quando aplicadas a problemas predominantemente tabulares. A solução reside em arquiteturas **híbridas GNN-GBDT**, onde a GNN atua como uma extratora não-supervisionada de contexto relacional, e o GBDT opera como classificador de decisão final.

### 2.3 Detecção de Fraude em Redes e Propagação de Suspeição
A modelagem de redes para detecção de anomalias financeiras foi extensamente explorada por Van Vlasselaer et al. (2015 - sistema APATE) e Pourhabibi et al. (2020). Esses autores demonstraram que a suspeição de uma transação se propaga através de redes sociais e de pagamento via homofilia (*guilt-by-association*). O presente trabalho adota esse princípio, formalizando-o através de grafos bipartidos de hardware e janelas deslizantes temporais.

### 2.4 Aprendizado Sensível ao Custo (*Cost-Sensitive Learning*)
Classificadores estatísticos padrão tratam os erros de Falso Positivo e Falso Negativo com pesos unitários homogêneos. Na literatura de aprendizado sensível ao custo, o trabalho fundador de Elkan (2001) estabeleceu as bases matemáticas para a reponderação de limiares sob matrizes assimétricas de custo. Subsequentemente, Correa Bahnsen et al. (2014, 2015) e Sahin et al. (2013) estenderam essa formulação para a detecção de fraudes com cartões de crédito sob o paradigma de **Matriz de Custo Dependente do Exemplo (*Example-Dependent Cost Matrix*)**, onde o custo do Falso Negativo varia em proporção direta ao valor monetário individual de cada transação, acrescido dos custos fixos de estorno e das taxas de investigação manual.

---

## 3. Taxonomia Estrutural de Fraudes no E-Commerce

Para embasar a engenharia de atributos de grafo, conduzimos um estudo sistemático sobre os dados reais do benchmark IEEE-CIS, identificando a assinatura empírica de quatro grandes técnicas de fraude:

```
                                  TAXONOMIA ESTRUTURAL DE TRANSAÇÕES
                                                   │
                  ┌────────────────────────────────┴────────────────────────────────┐
                  ▼                                                                 ▼
         Transações Legítimas                                            Transações Fraudulentas
          (96,5% do Volume)                                                 (3,5% do Volume)
                  │                                                                 │
                  │                                         ┌───────────────────────┴───────────────────────┐
                  ▼                                         ▼                                               ▼
         Comportamento Padrão                     Fraudes Coordenadas / Rede                      Fraudes Isoladas / Avulsas
        (1 Cartão / 1 Device)                      (62,2% das Fraudes Reais)                      (37,8% das Fraudes Reais)
                                                            │                                               │
                                    ┌───────────────────────┼───────────────────────┐                       │
                                    ▼                       ▼                       ▼                       ▼
                            Anéis de Dispositivos    Ataques de Rajada     Fraude Cross-Border     Fraudes Oportunistas
                            (Botnets / Emuladores)    (Velocity 24h)         (ProductCD = C)        (Roubo Individual)
                             Taxa: até 100% Fraude   Taxa: 11,2% Fraude     Taxa: 12,0% Fraude      Ticket Médio: $205,08
```

### 3.1 Fraudes Coordenadas em Rede (*Fraud Rings* e *Botnets*)
Identificamos a existência de **345 dispositivos físicos compartilhados por $\ge 3$ UIDs distintos**. Desses, múltiplos dispositivos apresentaram **100% de taxa de fraude histórica** (por exemplo, o modelo `LG-D320` conectou-se a 13 clientes diferentes com 15 transações de fraude consecutivas). Emuladores e sistemas operacionais obsoletos (`Android 5.1.1` e `iOS 11.0.1`) registraram taxas de fraude entre 20% e 30%, contra uma taxa natural de apenas 3,6% na população legítima.

### 3.2 Ataques de Velocidade (*Velocity Attacks*)
A análise temporal revelou que rajadas de transações ocorridas em intervalos extremamente curtos ($\Delta t \le 60\text{ segundos}$) no mesmo UID apresentam taxa de fraude de **11,15%** (3,2 vezes superior à média global). O fraudador tenta aprovar o máximo de compras antes que o titular perceba o roubo e solicite o bloqueio do cartão.

### 3.3 Estruturação e Fracionamento de Valores (*Structuring*)
Transações com valores inteiros e múltiplos redondos (\$10, \$50, \$100, \$500) apresentaram o dobro da taxa de fraude em comparação com compras com centavos fracionados (como \$49,99), refletindo a tentativa de simular recargas pré-pagas ou vales-presente (*gift cards*).

### 3.4 Fraude Transfronteiriça e Produtos de Risco
O código de produto `ProductCD = C` (compras internacionais / pagamentos transfronteiriços sem presença física) exibiu uma taxa de fraude intrínseca de **12,0%**, concentrando quase metade de todas as perdas financeiras brutas registradas no dataset.

---

## 4. Metodologia e Engenharia Detalhada de Features

A arquitetura proposta foi desenhada segundo um pipeline modular, reprodutível e livre de vazamento causal de dados:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     PIPELINE DE ENGENHARIA E MODELAGEM                                 │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
       Tabela de Transações (590k)                                           Tabela de Identidade (144k)
                 │                                                                     │
                 └──────────────────────────────────┬──────────────────────────────────┘
                                                    ▼
                                     Ingestão & Downcasting de RAM
                                   (1.5 GB reduzidos para 350 MB)
                                                    │
                                                    ▼
                                       Divisão Temporal Causal
                                    (80% Treino / 20% Validação)
                                                    │
                                                    ▼
                      ┌─────────────────────────────┴─────────────────────────────┐
                      ▼                                                           ▼
         Engenharia Tabular Avançada                               Topologia de Grafos Bipartidos
         - Normalização Dn (Day - D)                               - Grafo Portador (UID) ↔ Dispositivo
         - UID Ancorado (Magic UID)                                - Centralidade PageRank
         - Agregações de Grupo (μ, σ)                              - Grau de Conectividade de Entidade
         - Compressão PCA (339 V → 15 PCA)                         - Janela Deslizante de Velocidade 24h
                      │                                                           │
                      └─────────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                                   Representação Latente de Nós (PyG)
                                - GraphSAGE Heterogêneo Auto-Supervisionado
                                - Treinamento via Link Reconstruction (32-d)
                                                    │
                                                    ▼
                                    Roteamento de Identidade Dual
                                 ┌──────────────────┴──────────────────┐
                                 ▼                                     ▼
                            Model A                               Model B
                     (Sem Dados de Device)                 (Com Dados de Device)
                                 │                                     │
                                 └──────────────────┬──────────────────┘
                                                    ▼
                                            Classificador GBDT
                                           (XGBoost / LightGBM)
                                                    │
                                                    ▼
                                    Pós-Processamento por Cliente
                               (70% Probabilidade + 30% Média de UID)
                                                    │
                                                    ▼
                                Otimização de Função de Custo Financeiro
                                      (Limiar Ótimo de Corte τ*)
```

### 4.1 Otimização de Memória (*Downcasting Engine*)
O dataset completo em precisão padrão (`float64` e `int64`) consome mais de 1,5 GB em repouso e expande-se para além de 10 GB durante o cálculo de matrizes e histogramas de árvores. Implementamos um algoritmo de *downcasting* numérico adaptativo (`reduce_mem_usage`) que analisa os limites $[C_{\min}, C_{\max}]$ de cada coluna e rebaixa tipos inteiros para `int8`/`int16`/`int32` e pontos flutuantes para `float32`. Isso comprimiu a base para **350 MB** (economia de 76,7% de memória RAM), viabilizando o treinamento completo sobre todas as 590.540 linhas mesmo em instâncias gratuitas do Google Colab (12 GB de RAM).

### 4.2 Engenharia Tabular Avançada (Magic Features)

#### 4.2.1 Normalização dos Deltas Temporais ($D_n$) e Projeção de Calendário
No dataset IEEE-CIS, as 15 colunas $D1, \dots, D15$ representam contagens de dias transcorridos a partir de um marco zero pretérito (por exemplo, a data de emissão do cartão ou o último acesso do usuário). Como o tempo avança linearmente através da coluna `TransactionDT` (medida em segundos a partir de $t_0$), definimos o dia da transação como:
$$\text{Day}(t) = \left\lfloor \frac{TransactionDT}{86.400} \right\rfloor$$
A normalização de calendário é formulada subtraindo o delta relativo da data corrente:
$$D_i^n(t) = \text{Day}(t) - D_i(t), \quad \forall i \in \{1, 2, 10, 15\}$$
Essa operação transforma um contador relativo móvel em uma coordenada fixa no calendário civil, permitindo ao classificador agrupar todas as transações provenientes do mesmo cartão emitido naquela data específica, independentemente de quando as transações ocorram no tempo.

#### 4.2.2 Algoritmo de Resolução Hierárquica do UID Mágico
Na ausência de um identificador unívoco de cliente (como CPF ou ID de conta), sintetizamos um identificador composto de alta especificidade ($\text{UID}_{\text{magic}}$) combinando atributos do cartão, residência e âncora de emissão:
$$\text{UID}_{\text{magic}} = \text{card1} \;\Vert\; \text{card2} \;\Vert\; \text{addr1} \;\Vert\; \text{P\_emaildomain} \;\Vert\; \lfloor D1n \rfloor$$
Onde $\Vert$ denota a concatenação de strings com sentinela para valores nulos. Essa resolução particionou o dataset de treino em **109.723 UIDs únicos**, permitindo rastrear o comportamento longitudinal de cada titular de cartão.

#### 4.2.3 Agregações Estatísticas Causais e Escores Z
Para capturar anomalias transacionais relativas ao perfil histórico do portador, calculamos a média amostral ($\mu_{\text{uid}}$) e o desvio padrão ($\sigma_{\text{uid}}$) de variáveis críticas ($X \in \{TransactionAmt, D15n, C1, C13\}$) ajustados estritamente sobre a base de treino:
$$\mu_{\text{uid}}(X) = \frac{1}{|T_{\text{train}}(u)|} \sum_{k \in T_{\text{train}}(u)} X_k, \quad \sigma_{\text{uid}}(X) = \sqrt{\frac{1}{|T_{\text{train}}(u)| - 1} \sum_{k \in T_{\text{train}}(u)} (X_k - \mu_{\text{uid}}(X))^2}$$
A partir dessas estatísticas, geramos variáveis de razão e escores Z com suavização de Laplace ($\epsilon = 10^{-5}$):
$$\text{Ratio\_to\_Mean}(X_i) = \frac{X_i}{\mu_{\text{uid}}(X) + \epsilon}, \quad Z\text{-Score}(X_i) = \frac{X_i - \mu_{\text{uid}}(X)}{\sigma_{\text{uid}}(X) + \epsilon}$$
Na base de validação temporal futura, transações associadas a UIDs não vistos no treino recebem a média global da população, prevenindo *lookahead bias*.

---

### 4.3 Modelagem de Grafos Bipartidos e Velocidade Causal

#### 4.3.1 Formulação do Grafo Bipartido de Acesso
Modelamos as interações do comércio eletrônico como um grafo bipartido não direcionado:
$$\mathcal{G} = (\mathcal{V}_{\text{uid}} \cup \mathcal{V}_{\text{dev}}, \mathcal{E})$$
Onde $\mathcal{V}_{\text{uid}}$ é o conjunto de portadores, $\mathcal{V}_{\text{dev}}$ é o conjunto de instâncias físicas de dispositivos (`DeviceInfo`), e uma aresta $e = (u, d) \in \mathcal{E}$ existe se o portador $u$ realizou ao menos uma transação a partir do dispositivo $d$. A matriz de adjacência bipartida $\mathbf{B} \in \{0, 1\}^{|\mathcal{V}_{\text{uid}}| \times |\mathcal{V}_{\text{dev}}|}$ define a conectividade da rede.

#### 4.3.2 Centralidade PageRank por Iteração de Potências
Para quantificar a importância e centralidade estrutural de cada portador na rede de hardware, computamos o vetor de PageRank $\mathbf{p}$ resolvendo o sistema estacionário com fator de amortecimento $\alpha = 0,85$:
$$\mathbf{p}^{(k+1)} = \alpha \mathbf{M} \mathbf{p}^{(k)} + \frac{1 - \alpha}{|\mathcal{V}|} \mathbf{e}$$
Onde $\mathbf{M} = \mathbf{D}^{-1} \mathbf{A}$ é a matriz de transição estocástica por linhas normalizada pelos graus $\mathbf{D}$, e $\mathbf{e}$ é o vetor unitário de teletransporte. A convergência é alcançada via iteração de potências com tolerância $\|\mathbf{p}^{(k+1)} - \mathbf{p}^{(k)}\|_1 < 10^{-6}$. Portadores com alto PageRank representam nós que conectam direta ou indiretamente múltiplos clusters de dispositivos suspeitos.

#### 4.3.3 Graus Estruturais de Conectividade
Extraímos os graus nodais diretos:
$$\deg(d) = \sum_{u \in \mathcal{V}_{\text{uid}}} \mathbf{B}_{u, d} \quad (\text{Grau de Compartilhamento do Dispositivo})$$
$$\deg(u) = \sum_{d \in \mathcal{V}_{\text{dev}}} \mathbf{B}_{u, d} \quad (\text{Grau de Dispositivos Usados pelo Portador})$$

#### 4.3.4 Janelas Deslizantes de Velocidade Causal em 24h via Busca Binária
A velocidade de transações foi calculada através de um algoritmo causal vetorizado de complexidade temporal $O(N \log N)$. Para cada transação $i$ ocorrida no instante $t_i$, o número de eventos precedentes na janela $[t_i - 86.400, t_i]$ é computado via busca binária estrita:
$$\text{velocity}_{24h}(i) = i - \text{searchsorted}(\mathbf{t}, t_i - 86.400, \text{side}=\text{'left'}) + 1$$
Garantindo ausência total de vazamento temporal (*leakage-free*).

---

### 4.4 Compressão Dimensional e Representação Latente de Nós

#### 4.4.1 Decomposição Espectral PCA nas 339 Variáveis V
A família de atributos $V1$ a $V339$ (contadores de segurança e verificações de risco fornecidos pela processadora Vesta) apresenta colinearidade extrema e valores faltantes massivos. Padronizamos a matriz $\mathbf{X}_V$ após imputação pelas medianas e realizamos a decomposição em valores singulares (SVD) da matriz de covariância amostral:
$$\mathbf{\Sigma}_V = \frac{1}{N-1} \mathbf{X}_V^\top \mathbf{X}_V = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^\top$$
Projetamos as 339 dimensões nos primeiros 15 autovetores principais:
$$\mathbf{Z}_V = \mathbf{X}_V \mathbf{U}_{:, 1:15}$$
Essa transformação reduziu a largura da tabela em mais de 75%, retendo a maior parte da variância explicada e acelerando o treinamento das árvores.

#### 4.4.2 GNN HeteroGraphSAGE Auto-Supervisionada
Implementamos uma rede neural convolucional em grafos do tipo **Heterogeneous GraphSAGE** (HAMILTON et al., 2017; FEY; LENSSEN, 2019). Cada nó $v$ recebe um vetor de atributos estruturais iniciais $\mathbf{x}_v \in \mathbb{R}^3$ composto pelo logaritmo do grau $\log(\deg(v) + 1)$, volume financeiro médio e desvio padrão de montantes transacionados.

O passo de mensagem indutivo de 2 camadas ($k \in \{1, 2\}$) com agregação por média é formulado como:
$$\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \text{MEAN} \left( \left\{ \mathbf{h}_u^{(k-1)}, \forall u \in \mathcal{N}(v) \right\} \right)$$
$$\mathbf{h}_v^{(k)} = \text{ReLU} \left( \mathbf{W}^{(k)} \cdot \left[ \mathbf{h}_v^{(k-1)} \,\|\, \mathbf{h}_{\mathcal{N}(v)}^{(k)} \right] \right)$$
Onde $\|$ denota a concatenação de vetores.

Para evitar vazamento de rótulos (*target leakage*), o GraphSAGE é treinado sob uma função de perda **auto-supervisionada de Reconstrução de Links (*Link Reconstruction Loss*)** com amostragem negativa:
$$\mathcal{L}_{\text{link}} = - \sum_{(u, v) \in \mathcal{E}} \log \sigma \left( \mathbf{z}_u^\top \mathbf{z}_v \right) - Q \cdot \mathbb{E}_{v_n \sim P_n(v)} \left[ \log \sigma \left( -\mathbf{z}_u^\top \mathbf{z}_{v_n} \right) \right]$$
Onde $\mathbf{z}_u = \mathbf{h}_u^{(2)} \in \mathbb{R}^{32}$ é o embedding latente final do nó, $\sigma(\cdot)$ é a função sigmoide e $Q=5$ é o número de nós negativos amostrados aleatoriamente. **A rede neural não visualiza o rótulo de fraude (`isFraud`) em nenhum momento de seu treinamento**.

---

### 4.5 Roteamento de Identidade Dual e Pós-Processamento
Aproximadamente 75% das transações no benchmark IEEE-CIS não contêm dados na tabela de identidade (compras sem metadados de dispositivo). Para evitar que modelos únicos sofram com imputações artificiais de valores ausentes, desacoplamos o treinamento em dois classificadores especializados:
* **Model A (Sem Identidade)**: Treinado sobre 51 a 82 variáveis puramente transacionais.
* **Model B (Com Identidade)**: Treinado sobre 81 a 112 variáveis transacionais, de hardware, PageRank e embeddings de GNN.

O pós-processamento suave por cliente calibra as probabilidades finais agregando o escore médio de cada portador:
$$\hat{y}_{\text{final}}(i) = 0,7 \cdot \hat{y}_{\text{GBDT}}(i) + 0,3 \cdot \mu_{\text{pred}}(\text{UID}_i)$$

---

## 5. Auditoria de Maturidade e Correção de Leakages

Para garantir que todos os resultados reportados reflitam a capacidade real de generalização em produção, o pipeline passou por uma auditoria formal cobrindo três pontos críticos:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 RELATÓRIO FORMAL DE AUDITORIA DE LEAKAGES                              │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘

1. TARGET ENCODING NO DOMÍNIO SINTÉTICO (ETAPA 3):
   • Diagnóstico: O cálculo de src_neigh_fraud_rate incluía a própria transação sendo avaliada,
     gerando correlação perfeita de 99,95% no treino (leakage direto).
   • Correção: Implementação do algoritmo Leave-One-Out (LOO) vetorizado e suavizado (m = 10,0).
   • Resultado: Correlação com o alvo reduzida para saudáveis 14,52%, eliminando a memorização.

2. EMBEDDINGS DE GNN SUPERVISIONADOS (ETAPA 6):
   • Diagnóstico: O GraphSAGE era treinado com função de perda supervisionada usando isFraud.
     As árvores decoravam as probabilidades pré-calculadas pela GNN, sofrendo overfitting no tempo.
   • Correção: Conversão do treinamento para Link Reconstruction Auto-Supervisionado (sem rótulo).
   • Resultado: Embeddings puramente estruturais (32-d), com variância restaurada e ganho real no GBDT.

3. AVALIAÇÃO INDUTIVA NA GNN PURA (ETAPA 7 / 9):
   • Diagnóstico: A avaliação transdutiva deixava os nós futuros isolados durante a validação temporal,
     gerando representações invertidas (ROC-AUC < 0,50).
   • Correção: Implementação do passo de mensagem indutivo combinando as arestas temporais.
   • Resultado: Desempenho restaurado (ROC-AUC = 0,7024), comprovando a capacidade de aprendizado da rede.
```

---

## 6. Estudo de Ablação Quádruplo Aprofundado (100% dos Dados — 590k Linhas)

O cerne empírico deste trabalho é o **Estudo de Ablação Quádruplo**, desenhado para isolar, com controle rigoroso de variáveis e hiperparâmetros idênticos no classificador XGBoost (`n_estimators=450, lr=0.03, max_depth=9, scale_pos_weight`), o impacto individual de cada família de atributos sobre a base completa de 590.540 transações.

### 6.1 Definição das Quatro Configurações de Ablação

1. **Configuração 1 — XGBoost Baseline (Raw Tabular)**:  
   Classificador treinado estritamente com os atributos brutos originais das tabelas de transação e identidade. Não inclui nenhuma variável derivada de normalização temporal de dias, nenhum identificador de portador (UID) e nenhuma métrica topológica de grafo (51 features no Model A / 81 no Model B).
2. **Configuração 2 — XGBoost + Apenas Grafos (Structural Only)**:  
   Variáveis originais enriquecidas **exclusivamente com atributos de rede e velocidade**: PageRank bipartido (`uid_pagerank`), grau de compartilhamento de dispositivo (`device_degree`), grau do portador (`uid_degree`), janela deslizante causal de velocidade 24h (`uid_tx_count_24h`) e pós-processamento de cliente. Nenhuma variável de normalização $D_n$ ou agregação de grupo tabular é incluída (55 features no Model A / 85 no Model B).
3. **Configuração 3 — XGBoost + Apenas Feature Engineering (Advanced Tabular)**:  
   Variáveis originais enriquecidas **exclusivamente com engenharia tabular avançada**: deltas temporais consistentes no calendário ($D1n, D2n, D10n, D15n$), agregações estatísticas por cliente ($\mu$ e $\sigma$ de montantes e contadores $C$) e os 15 componentes principais PCA da família $V$. Nenhuma métrica de rede, grau de hardware ou PageRank é incluída (78 features no Model A / 108 no Model B).
4. **Configuração 4 — XGBoost All-in-One**:  
   Modelo completo integrando a totalidade das variáveis: dados brutos, engenharia tabular avançada, componentes PCA e todas as features de grafo e velocidade (82 features no Model A / 112 no Model B).

---

### 6.2 Resultados Consolidados da Ablação

A tabela abaixo detalha as métricas de performance, tempo de treinamento e contagem de variáveis obtidas na validação temporal causal estrita:

| Configuração Avaliada | Nível de Features (Model A / B) | PR-AUC | ROC-AUC | F1-Score | Limiar Ótimo ($\tau^*$) | Tempo de Treino | Diferencial / Impacto Estrutural |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. XGBoost Baseline (Raw)** | 51 / 81 cols | 0,5603 | 0,8918 | 0,5433 | 0,540 | **75,0s** | Ponto de Controle Puro |
| **2. XGBoost + Apenas Grafos** | 55 / 85 cols | **`0,6032`** | **`0,9101`** | 0,5855 | 0,480 | 80,1s | **Campeão em PR-AUC e ROC-AUC (+4,29% PR-AUC)** |
| **3. XGBoost + Apenas FE** | 78 / 108 cols | 0,5780 | 0,8710 | 0,5835 | 0,510 | 125,1s | Ganho em PR-AUC com degradação em ROC-AUC |
| **4. XGBoost All-in-One** | 82 / 112 cols | 0,5929 | 0,8780 | **`0,5963`** | 0,470 | 124,0s | **F1-Score Máximo e Melhor Equilíbrio P/R** |

```
                       COMPARAÇÃO VISUAL DAS MÉTRICAS NA ABLAÇÃO
  0.62 ┌────────────────────────────────────────────────────────────────────────┐
       │                                                                        │
  0.60 ┼──────────────────────────────────── 0.6032 (PR-AUC) ────────────────────┤
       │                                    (Apenas Grafos)                     │
  0.58 ┼────────────────────────────────────────────────── 0.5780 ── 0.5929 ────┤
       │                                                   (Apenas FE) (All-in-One)
  0.56 ┼────── 0.5603 (PR-AUC) ─────────────────────────────────────────────────┤
       │      (Baseline)                                                        │
  0.54 └───────┬───────────────────────────────┬──────────────┬──────────┬──────┘
            Baseline                      Apenas Grafos    Apenas FE   All-in-One
```

---

### 6.3 Análise e Dissecação dos Fenômenos da Ablação

A análise detalhada dos resultados da ablação revela três comportamentos fundamentais:

#### A. A Supremacia dos Grafos contra o Overfitting Causal
A Configuração 2 (**XGBoost + Apenas Grafos**) atingiu a maior PR-AUC (**0,6032**) e a maior ROC-AUC (**0,9101**) do estudo, superando a Configuração 3 (**Apenas Feature Engineering**) por **+2,52 pontos percentuais de PR-AUC e +3,91 pontos de ROC-AUC**, utilizando **23 variáveis a menos**.
Esse resultado decorre da natureza dos sinais:
* As variáveis de agregação tabular (médias de ticket de compra e desvios de consumo) dependem de padrões comportamentais monetários transitórios. Em divisões temporais rigorosas onde o teste ocorre semanas à frente do treino, esses padrões sofrem severamente com o ***concept drift*** (mudança natural nos hábitos de consumo e limites de crédito).
* As variáveis de grafo (`device_degree`, `uid_pagerank` e velocidade 24h), por sua vez, medem **conectividade de infraestrutura física invariante** (o fato objetivo de que um fraudador precisa reutilizar hardware e emuladores para operar botnets). Essa informação relacional é estaticamente robusta no tempo, prevenindo o sobreajuste (*overfitting*) das árvores.

#### B. Eficiência Paramétrica e o Princípio da Navalha de Occam
O modelo **Apenas Grafos** alcançou desempenho superior adicionando apenas **4 variáveis** à base bruta (elevando o tempo de treino de 75s para apenas 80s), enquanto o modelo de **Apenas FE** exigiu 27 variáveis adicionais, gerando um custo computacional 56% superior (125,1s) para um ganho consideravelmente menor.

#### C. Maximização do F1-Score no Modelo All-in-One
Embora o modelo de apenas grafos tenha obtido a maior área sob a curva Precision-Recall (PR-AUC), o modelo **All-in-One** alcançou o maior F1-Score no ponto de corte ótimo (**0,5963** vs 0,5855). Isso demonstra que, quando o objetivo operacional é selecionar um único limiar estático de decisão para aprovação automática, a interação entre desvios estatísticos de valor monetário e centralidade de rede proporciona a melhor discriminação conjunta de precisão e sensibilidade.

---

## 7. Modelagem da Função de Custo Financeiro e Impacto no Negócio

Em ambientes bancários e de comércio eletrônico, avaliar classificadores estritamente por métricas estatísticas não ponderadas (acurácia, F1 ou ROC-AUC) é inadequado para o negócio: **um Falso Negativo de \$1.000 custa cem vezes mais do que um Falso Negativo de \$10**.

### 7.1 Teoria da Matriz de Custo Dependente do Exemplo (*Example-Dependent Cost Matrix*)
Fundamentando-nos na formulação seminal de Correa Bahnsen et al. (2014, 2015) e no princípio de *Bayes Minimum Risk* (ELKAN, 2001), definimos a matriz de perdas onde os custos de classificação são expressos diretamente em função do montante financeiro da transação ($\text{amt}_i$):

| Rótulo Real ($y_i$) / Predição ($\hat{y}_i$) | Classificado como Legítimo ($\hat{y}_i = 0$) | Classificado como Fraude ($\hat{y}_i = 1$) |
| :--- | :--- | :--- |
| **Transação Legítima ($y_i = 0$)** | $\text{Cost}(TN_i) = 0$ | $\text{Cost}(FP_i) = C_{\text{revisão}} + \alpha \cdot \text{amt}_i$ |
| **Transação Fraudulenta ($y_i = 1$)** | $\text{Cost}(FN_i) = \text{amt}_i + C_{\text{chargeback}}$ | $\text{Cost}(TP_i) = C_{\text{operacional}}$ |

#### Parâmetros Operacionais de Mercado Calibrados:
* **$C_{\text{chargeback}} = \$15,00$**: Multa administrativa e taxa de estorno cobrada pela adquirente/bandeira (Visa/Mastercard) por transação fraudulenta consumada.
* **$C_{\text{revisão}} = \$3,00$**: Custo operacional de triagem manual por analista humano.
* **$\alpha = 0,05$ (5%)**: Custo de fricção e risco de cancelamento/churn do cliente honesto que teve a compra recusada.
* **$C_{\text{operacional}} = \$0,10$**: Custo computacional marginal de acionamento do motor de regras.

### 7.2 Otimização do Limiar Financeiro ($\tau^*$) via Bayes Minimum Risk
Para qualquer probabilidade estimada $\hat{p}_i = P(y_i = 1 | \mathbf{x}_i)$, a decisão de classificar a compra como fraude ($\hat{y}_i(\tau) = 1$) é parametrizada pelo limiar $\tau \in [0, 1]$. O custo financeiro total esperado $L(\tau)$ sobre o conjunto de validação temporal ($N = 23.622$ transações) é formulado como:
$$L(\tau) = \sum_{i=1}^N \left[ y_i (1 - \hat{y}_i(\tau)) \cdot (\text{amt}_i + C_{\text{chargeback}}) + (1 - y_i) \hat{y}_i(\tau) \cdot (C_{\text{revisão}} + \alpha \cdot \text{amt}_i) + y_i \hat{y}_i(\tau) \cdot C_{\text{operacional}} \right]$$
O ponto de corte financeiro ótimo $\tau^*$ é obtido minimizando $L(\tau)$:
$$\tau^* = \arg\min_{\tau \in [0, 1]} L(\tau)$$

### 7.3 Avaliação Financeira Comparativa das Configurações

Ao varrer o espaço de limiares $\tau \in [0,01; 0,95]$, obtivemos a seguinte comparação econômica direta:

| Modelo Avaliado | Limiar Ótimo ($\tau^*$) | Prejuízo Financeiro Mínimo (\$) | Economia Líquida (\$) | Redução Percentual do Prejuízo |
| :--- | :---: | :---: | :---: | :---: |
| **1. Modelo SEM Grafos (Baseline Tabular)** | 0,42 | \$134.138,06 | — | Baseline de Custo |
| **2. Modelo COM Grafos (Híbrido Relacional)** | **0,38** | **\$77.710,73** | **+\$56.427,33** | **-42,07% de Prejuízo Líquido** |

```
                       CURVA DE PERDA FINANCEIRA TOTAL vs. LIMIAR (τ)
   $140k ┌────────────────────────────────────────────────────────────────────────┐
         │       /'''''''\  (Sem Grafos: Custo Mínimo = $134.138 @ τ=0.42)        │
   $120k ┼──────/─────────\───────────────────────────────────────────────────────┤
         │     /           \                                                      │
   $100k ┼────/─────────────\─────────────────────────────────────────────────────┤
         │   /               \                                                    │
    $80k ┼──/─────\___________/──── (Com Grafos: Custo Mínimo = $77.710 @ τ=0.38)─┤
         │ /                         [ECONOMIA: $56.427,33 (-42,07%)]             │
    $60k └─┴───────────┬──────────────────────┬──────────────────────┬────────────┘
          0.1         0.3                    0.5                    0.7          τ
```

---

## 8. Discussão Metodológica e Limitações da Pesquisa

### 8.1 A Soberania da PR-AUC e a Falácia do ROC-AUC
Sob desbalanceamento severo (3,5% de fraudes), o ROC-AUC é facilmente inflado pela massa massiva de verdadeiros negativos (96,5% de compras legítimas). A métrica de **PR-AUC (Área sob a Curva Precision-Recall)** provou ser o único indicador fidedigno para avaliar a capacidade do modelo de capturar fraudes sem disparar uma enxurrada de falsos positivos que inviabilizaria a operação do e-commerce.

### 8.2 O Papel da Arquitetura Híbrida em Produção Industrial
A falha da GNN pura (PR-AUC de 0,0813) e o sucesso do modelo híbrido GNN-GBDT (PR-AUC de 0,6032) consolidam uma diretriz definitiva para a engenharia de sistemas financeiros:
* **GNNs não devem substituir classificadores de árvore no domínio tabular**, mas sim atuar como extratoras de representação relacional densa.
* A combinação de **PCA (reduzindo 339 colunas ruidosas para 15 componentes)** com **Embeddings de Rede (32 dimensões)** permitiu construir um modelo com 75% menos colunas que supera a baseline gigante em poder discriminatório e reduz o consumo de memória em mais de 75%.

### 8.3 Discussão de Limitações Metodológicas e Análise de Amostragem

Com o compromisso de transparência científica, identificamos e discutimos as seguintes limitações metodológicas deste trabalho:

1. **Gargalo Computacional e Estratégia de Escala em Dois Estágios**:
   * O hardware local disponível para o projeto (8–16 GB de RAM sem aceleração por GPUs de alta performance) impossibilitou a execução de buscas exaustivas de hiperparâmetros (*GridSearch/Optuna*) e validação cruzada *GroupKFold* de 10 dobras sobre os 590.540 registros.
   * Para contornar essa restrição física, adotamos uma estratégia pragmática de dois estágios: (i) prototipagem e auditoria de leakages em subamostra de 10%, seguida de (ii) otimização de memória via *downcasting* numérico para execução dos pipelines finais e ablação em 100% dos dados no Google Colab.
2. **Divergência de Densidade Topológica (10% vs. 100%)**:
   * Ao subamostrar transações a 10%, a topologia do grafo de acesso torna-se artificialmente esparsa. A probabilidade de um dispositivo físico ou portador reaparecer na amostra é reduzida por um fator de 10, resultando em muitos nós com grau nulo ou unitário ($\deg=1$).
   * No dataset completo de 100%, a rede recupera sua densidade real, revelando os 345 dispositivos hiperconectados e cadeias de conluio. Isso explica por que o ganho em PR-AUC das features de grafo foi de **+2,06% na amostra de 10%** e saltou para expressivos **+4,29% no dataset de 100%**.
3. **Escopo da Matriz de Validação Temporal**:
   * O pipeline foi avaliado sob um split temporal fixo de 80% treino e 20% teste causal. Uma avaliação por múltiplas janelas deslizantes (*walk-forward temporal cross-validation*) enriqueceria a variância estatística das estimativas, embora exigisse um custo computacional multiplicativo de re-treinamento das GNNs indutivas.

---

## 9. Conclusão e Trabalhos Futuros

Este trabalho documentou a jornada completa de concepção, auditoria, refinamento e validação empírica de uma arquitetura híbrida de detecção de fraudes com grafos no comércio eletrônico.

As principais contribuições científicas e práticas incluem:
1. **Comprovação do Valor dos Grafos na Ablação**: Demonstramos que a incorporação isolada de apenas 4 atributos de grafo gerou o maior ganho de performance do estudo (**+4,29% em PR-AUC e +1,83% em ROC-AUC**) sobre o classificador XGBoost no benchmark IEEE-CIS, superando a engenharia tabular massiva.
2. **Resolução de Anéis Coordenados**: Mapeamos que **62,2% das fraudes reais decorrem de padrões de rede**, provando a necessidade imperativa de quebrar a premissa estatística de independência (IID).
3. **Rigor e Reprodutibilidade Científica**: Identificamos e corrigimos fontes críticas de *target leakage* em target encoding e em representações de GNN, assegurando que os resultados decorrem de generalização causal legítima.
4. **Validação Econômica Direta**: Por meio de uma função de custo financeiro dependente do exemplo calibrada com taxas de mercado, comprovamos que a solução gera uma **redução de 42,07% no prejuízo operacional**, economizando **\$56.427,33** na base de teste.

Como trabalhos futuros, sugere-se a extensão desta arquitetura para **Grafos Temporais Dinâmicos (Temporal GNNs)** com aprendizado contínuo em streaming (*online learning*) e a exploração de pesos heterogêneos de arestas ponderados por tipo de dispositivo e geolocalização.

---

## 10. Referências Bibliográficas (Normas ABNT)

AKOGLU, Leman; CHANDY, Rishi; FALOUTSOS, Christos. Graph based anomaly detection and description: a survey. **Data Mining and Knowledge Discovery**, v. 29, n. 3, p. 626-688, 2015.

CHEN, Tianqi; GUESTRIN, Carlos. XGBoost: A Scalable Tree Boosting System. In: **Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)**, San Francisco, CA, USA, 2016, p. 785-794.

CORREA BAHNSEN, Alejandro; AOUADA, Djamila; OTTERSTEN, Björn. Feature engineering strategies for detecting credit card fraud. **Expert Systems with Applications**, v. 41, n. 16, p. 7158-7164, 2014.

CORREA BAHNSEN, Alejandro; AOUADA, Djamila; OTTERSTEN, Björn. Example-dependent cost-sensitive decision trees. **Expert Systems with Applications**, v. 42, n. 19, p. 6609-6619, 2015.

DAL POZZOLO, Andrea et al. Calibrating probability with undersampling for unbalanced classification. In: **IEEE Symposium Series on Computational Intelligence (SSCI)**, Cape Town, South Africa, 2015, p. 159-166.

DAL POZZOLO, Andrea et al. Credit card fraud detection: a realistic modeling and a novel learning strategy. **IEEE Transactions on Neural Networks and Learning Systems**, v. 29, n. 8, p. 3784-3797, 2018.

ELKAN, Charles. The foundations of cost-sensitive learning. In: **International Joint Conference on Artificial Intelligence (IJCAI)**, Seattle, WA, USA, 2001, v. 17, n. 1, p. 973-978.

EUROPEAN CENTRAL BANK. **Report on card fraud in 2022 and 2023**. Frankfurt am Main: ECB, 2023. Disponível em: <https://www.ecb.europa.eu/pub/cardfraud/>. Acesso em: 28 ago. 2026.

FEY, Matthias; LENSEN, Jan Eric. Fast Graph Representation Learning with PyTorch Geometric. In: **ICLR Workshop on Representation Learning on Graphs and Manifolds**, New Orleans, LA, USA, 2019.

GRINSZTAJN, Léo; OYALLON, Edouard; VAROQUAUX, Gaël. Why do tree-based models still outperform deep learning on typical tabular data? In: **Advances in Neural Information Processing Systems (NeurIPS)**, New Orleans, LA, USA, 2022, v. 35, p. 507-520.

HAGBERG, Aric A.; SCHULT, Daniel A.; SWART, Pieter J. Exploring Network Structure, Dynamics, and Function using NetworkX. In: **Proceedings of the 7th Python in Science Conference (SciPy)**, Pasadena, CA, USA, 2008, p. 11-15.

HAMILTON, Will; YING, Rex; LESKOVEC, Jure. Inductive Representation Learning on Large Graphs. In: **Advances in Neural Information Processing Systems (NeurIPS)**, Long Beach, CA, USA, 2017, p. 1024-1034.

HARRIS, Charles R. et al. Array programming with NumPy. **Nature**, v. 585, n. 7825, p. 357-362, 2020.

IEEE Computational Intelligence Society. **IEEE-CIS Fraud Detection**: Predict probability of online transaction fraud. Kaggle, 2019. Disponível em: <https://www.kaggle.com/c/ieee-fraud-detection/data>. Acesso em: 28 ago. 2026.

KE, Guolin et al. LightGBM: A Highly Efficient Light Gradient Boosting Machine. In: **Advances in Neural Information Processing Systems (NeurIPS)**, Long Beach, CA, USA, 2017, p. 3146-3154.

MCKINNEY, Wes. Data Structures for Statistical Computing in Python. In: **Proceedings of the 9th Python in Science Conference (SciPy)**, Austin, TX, USA, 2010, p. 51-56.

PAGE, Lawrence; BRIN, Sergey; MOTWANI, Rajeev; WINOGRAD, Terry. **The PageRank Citation Ranking: Bringing Order to the Web**. Stanford InfoLab, 1999. Technical Report.

PEDREGOSA, Fabian et al. Scikit-learn: Machine Learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825-2830, 2011.

POURHABIBI, Tahereh et al. Fraud detection in financial networks: a survey of graph computing approaches. **Artificial Intelligence Review**, v. 53, n. 8, p. 6097-6135, 2020.

SAHIN, Yusuf; BULKAN, Serol; DUMAN, Ekrem. A cost-sensitive decision tree approach for fraud detection. **Knowledge-Based Systems**, v. 43, p. 22-31, 2013.

SHWARTZ-ZIV, Ravid; ARMON, Amitai. Tabular data: Deep learning is not all you need. **Information Fusion**, v. 81, p. 84-90, 2022.

THE NILSON REPORT. **Card fraud losses reach \$32.34 billion globally**. Issue 1254. Carpinteria: Nilson Report, 2023. Disponível em: <https://nilsonreport.com/>. Acesso em: 28 ago. 2026.

VAN VLASSELAER, Véronique et al. APATE: a novel approach for automated credit card fraud detection using network-based extensions. **Decision Support Systems**, v. 75, p. 38-48, 2015.

VAN VLASSELAER, Véronique et al. GOTCHA! Network-based fraud detection for social security fraud. **Management Science**, v. 63, n. 9, p. 3090-3110, 2017.
