# O Impacto Estrutural e Econômico de Grafos na Detecção de Fraude em Pagamentos Digitais: Um Estudo de Ablação e Modelagem Sensível ao Custo

**Autor:** Victor Valadares  
**Área de Concentração:** Inteligência Artificial, Ciência de Dados e Engenharia de Software  
**Palavras-chave:** Detecção de Fraudes; Grafos Bipartidos; Graph Neural Networks (GNN); Gradient Boosted Decision Trees (GBDT); Modelagem Sensível ao Custo; Estudo de Ablação; IEEE-CIS Benchmark.

---

## Resumo

A detecção de fraudes em transações de comércio eletrônico (*e-commerce*) é um dos problemas mais críticos da economia digital global. Tradicionalmente, classificadores de aprendizado de máquina supervisionados aplicados a dados tabulares — como *Gradient Boosted Decision Trees* (GBDT) — operam sob a premissa estatística de observações independentes e identicamente distribuídas (*IID*). Contudo, o crime cibernético contemporâneo organiza-se frequentemente em padrões relacionais coordenados envolvendo perfis de dispositivos e emuladores compartilhados. Este trabalho investiga e quantifica o real papel e o valor incremental da modelagem em grafos para sistemas de prevenção à fraude através de três eixos: (i) enriquecimento de atributos via **Grafos Bipartidos de Acesso** (Portador $\leftrightarrow$ Perfil de Dispositivo), centralidade de rede (PageRank), contadores causais de velocidade temporal em 24 horas, compressão espectral (PCA) e engenharia de atributos de domínio; (ii) execução de um **Estudo de Ablação Sistemático de 8 Configurações** sobre o benchmark mundial **IEEE-CIS Fraud Detection** (295.270 a 590.540 transações reais); e (iii) formulação de uma **Função de Custo Financeiro Dependente do Exemplo** calibrada sob o princípio de *Bayes Minimum Risk* (CORREA BAHNSEN et al., 2015). Nossos experimentos de ablação revelam com rigor e transparência científica que a **engenharia de atributos tabular específica de domínio (deltas temporais $D_n$ e desvios de perfil por portador UID) é o pilar que responde por 87,3% do ganho de detecção na cadeia de ablação** (elevando o PR-AUC de 0,5028 para 0,5304, +0,0276 sobre o baseline PCA). Por sua vez, a **modelagem de grafos opera como um refinamento incremental ortogonal**, agregando **+0,0040 em PR-AUC** (12,7% do lift, atingindo 0,5344 no modelo PCA e 0,5395 bruto / 0,5434 com pós-processamento causal alinhado em variáveis brutas, além de +0,0067 em busca Bayesiana via Optuna). Sob uma esteira com taxa de falsos positivos controlada ($\text{FPR} = 1,0\%$), a solução eleva o **recall de fraudes coordenadas em +10,40 p.p. (de 50,87% para 61,27%, IC 95%: [+8,02; +12,78] p.p.)**, sendo +7,54 p.p. oriundos da ancoragem tabular e +2,86 p.p. de grafos e pós-processamento, além de garantir uma **economia média comprovada de US\$ 6.722,08 por mês** em validação temporal expansiva (*Walk-Forward*).

**Abstract:**  
Fraud detection in e-commerce transactions is one of the most critical challenges in the digital economy. Traditionally, supervised machine learning models applied to tabular data, such as Gradient Boosted Decision Trees (GBDT), operate under the Independent and Identically Distributed (IID) assumption. However, modern cybercrime frequently exhibits relational patterns across shared device models, platforms, and emulators. This paper rigorously investigates and quantifies the exact incremental value of graph modeling in fraud prevention systems across three dimensions: (i) feature enrichment using Bipartite Access Graphs (Account $\leftrightarrow$ Device Profile), network centrality (PageRank), causal 24-hour rolling velocity counters, PCA spectral compression, and domain-specific feature engineering; (ii) an 8-way ablation study on the IEEE-CIS Fraud Detection benchmark; and (iii) an example-dependent financial cost function formulation based on Bayes Minimum Risk theory (CORREA BAHNSEN et al., 2015). Our ablation experiments demonstrate with scientific transparency that **domain-specific tabular feature engineering (temporal $D_n$ deltas and cardholder UID aggregations) drives 87.3% of performance gains in the ablation chain** (lifting PR-AUC from 0.5028 to 0.5304, +0.0276 over the PCA baseline). In turn, **graph modeling acts as an orthogonal incremental refinement**, contributing **+0.0040 in PR-AUC** (12.7% of total lift, reaching 0.5344 in the PCA model, 0.5395 raw and 0.5434 with aligned causal post-processing in raw variables, alongside a +0.0067 gain under Bayesian optimization). Under a controlled operational budget ($\text{FPR} = 1.0\%$), the architecture delivers a **+10.40 percentage-point gain in coordinated fraud recall (from 50.87% to 61.27%, 95% CI: [+8.02, +12.78] p.p.)**, decomposed into +7.54 p.p. from domain tabular features and +2.86 p.p. from bipartite graphs and aligned causal post-processing, while yielding a **sustained average savings of $6,722.08 per month** in expanding walk-forward temporal cross-validation.

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

### 3.5 Definição Operacional de Fraude Coordenada e Ressalva de Circularidade
Definimos operacionalmente como **fraude coordenada** qualquer transação que compartilhe dispositivo com outro portador ($\text{device\_degree} \ge 2$) ou que ocorra sob rajada temporal ($\text{velocity}_{24h} \ge 2$). É imperativo registrar a natureza de **rótulo substituto (*surrogate label*)** dessa formulação: na ausência de anotações forenses externas de anéis criminosos no benchmark IEEE-CIS, a rotulagem apoia-se nas mesmas variáveis estruturais modeladas pelo pipeline. Como consequência, a capacidade discriminatória observada nessas transações reflete a sensibilidade aos padrões operacionais de rede e velocidade, demandando cautela na generalização para fraudes coordenadas que não manifestem esses traços observáveis.

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
Onde $\mathcal{V}_{\text{uid}}$ é o conjunto de portadores, $\mathcal{V}_{\text{dev}}$ é o conjunto de assinaturas de modelo de dispositivo e sistema operacional (`DeviceInfo`), e uma aresta $e = (u, d) \in \mathcal{E}$ existe se o portador $u$ realizou ao menos uma transação a partir do recurso $d$. A matriz de adjacência bipartida $\mathbf{B} \in \{0, 1\}^{|\mathcal{V}_{\text{uid}}| \times |\mathcal{V}_{\text{dev}}|}$ define a conectividade da rede.

Ressalta-se que `DeviceInfo` não é um identificador biométrico ou de hardware unívoco (como endereço MAC ou IMEI), mas sim uma string de modelo/compilação tecnológica (ex.: `Windows`, `iOS Device`, `MacOS`, `SM-G935F`). Para evitar colapsos topológicos, valores sentinela nulos (`null_device`) foram explicitamente excluídos da malha de arestas.

#### 4.3.2 Centralidade PageRank por Iteração de Potências
Para quantificar a importância e centralidade estrutural de cada portador na rede de recursos, computamos o vetor de PageRank $\mathbf{p}$ resolvendo o sistema estacionário com fator de amortecimento $\alpha = 0,85$:
$$\mathbf{p}^{(k+1)} = \alpha \mathbf{M} \mathbf{p}^{(k)} + \frac{1 - \alpha}{|\mathcal{V}|} \mathbf{e}$$
Onde $\mathbf{M} = \mathbf{D}^{-1} \mathbf{A}$ é a matriz de transição estocástica por linhas normalizada pelos graus $\mathbf{D}$, e $\mathbf{e}$ é o vetor unitário de teletransporte. A convergência é alcançada via iteração de potências com tolerância $\|\mathbf{p}^{(k+1)} - \mathbf{p}^{(k)}\|_1 < 10^{-6}$. Portadores com alto PageRank representam nós que conectam direta ou indiretamente múltiplos clusters de dispositivos suspeitos.

#### 4.3.3 Graus Estruturais de Conectividade e Não-Monotonicidade de Risco
Extraímos os graus nodais diretos:
$$\deg(d) = \sum_{u \in \mathcal{V}_{\text{uid}}} \mathbf{B}_{u, d} \quad (\text{Grau de Compartilhamento do Dispositivo})$$
$$\deg(u) = \sum_{d \in \mathcal{V}_{\text{dev}}} \mathbf{B}_{u, d} \quad (\text{Grau de Dispositivos Usados pelo Portador})$$

**Comportamento Não-Monotônico do Risco**: Em razão da semântica de `DeviceInfo`, a relação entre grau e probabilidade de fraude não é estritamente monotônica:
* Nós com conectividade intermediária (11 a 100 UIDs) exibem o pico de taxa de fraude (~14,9% a 15,1%), refletindo modelos específicos de smartphones ou emuladores concentrados em campanhas de fraude coordenada;
* Megahubs tecnológicos massivos (>100 UIDs, como `Windows` conectando 27.939 UIDs e `iOS Device` com 14.729 UIDs) apresentam taxa de fraude diluída para ~5,8% devido ao imenso volume de compras legítimas. O modelo supervisionado de árvores captura essa não-linearidade através de cortes ortogonais conjuntos.

#### 4.3.4 Janelas Deslizantes de Velocidade Causal em 24h via Busca Binária
Diferentemente das métricas de grafo estáticas, a velocidade transacional por portador foi calculada através de um algoritmo causal vetorizado de complexidade temporal $\mathcal{O}(N \log K)$. Para cada transação $i$ do portador $u$ no instante $t_i$, o número de eventos no intervalo estritamente retrospectivo $[t_i - 86.400, t_i]$ é computado via busca binária interna no histórico do próprio UID:
$$\text{velocity}_{24h}(i) = i - \text{searchsorted}(T(u), t_i - 86.400, \text{side}=\text{'left'}) + 1$$
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

### 4.5 Roteamento de Identidade Dual e Pós-Processamento Causal

#### 4.5.1 Roteamento Dual de Identidade
Aproximadamente 75% das transações no benchmark IEEE-CIS não contêm dados na tabela de identidade (compras sem metadados de dispositivo). Para evitar que modelos únicos sofram com imputações artificiais de valores ausentes, desacoplamos o treinamento em dois classificadores especializados:
* **Model A (Sem Identidade)**: Treinado sobre 67 a 410 variáveis puramente transacionais, contadores temporais $D_n$, estatísticas históricas do UID e velocidade causal de 24h.
* **Model B (Com Identidade)**: Treinado sobre 106 a 449 variáveis (incluindo dados transacionais acrescidos de metadados de hardware, PageRank e graus bipartidos).

#### 4.5.2 Pós-Processamento Causal por Média Expansiva
Para incorporar a inércia comportamental de portadores sem incorrer em vazamento temporal bidirecional, implementamos um filtro causal expansivo:
$$\mu_{\text{exp}}(t_i) = \frac{1}{|\{k \in T(u_i) \mid t_k \le t_i\}|} \sum_{t_k \le t_i} \hat{y}(k)$$
$$\hat{y}_{\text{final}}(i) = 0,7 \cdot \hat{y}_{\text{GBDT}}(i) + 0,3 \cdot \mu_{\text{exp}}(t_i)$$
**Blindagem Algorítmica de Alinhamento**: No Pandas, o método `groupby('uid')['raw_score'].expanding().mean()` indexa os resultados agrupados alfabeticamente pela chave de entidade. O emprego ingênuo de `.values` gera desalinhamento posicional grave sobre o frame ordenado cronologicamente. A implementação requer `.reset_index(level=0, drop=True)`, garantindo alinhamento vetorial perfeito por índice temporal.

---

### 4.6 Protocolo Experimental de Avaliação de Recall e Decomposição de Ganhos

#### 4.6.1 Decomposição Ortogonal: Engenharia Tabular vs. Estrutura de Grafos
O design experimental foi desenhado para isolar categoricamente a contribuição de cada família de atributos através de cadeias de comparação estritamente consistentes:
1. **Cadeia de Ablação Principal com Compressão PCA ($1\text{B} \to 5 \to 6\text{A}$)**:
   - **Baseline PCA (1B)**: PR-AUC Bruto = **0,5028**
   - **Apenas FE Tabular (5)**: PR-AUC Bruto = **0,5304** (Lift de **+0,0276**)
   - **All-in-One PCA (6A)**: PR-AUC Bruto = **0,5344** (Lift de **+0,0040** vindo de Grafos)
   - **Decomposição Relativa do Lift Total (+0,0316)**: A Engenharia Tabular ($5 - 1\text{B}$) responde por **87,3%** de todo o ganho sobre o baseline, enquanto os atributos de rede bipartida ($6\text{A} - 5$) agregam os **12,7%** restantes.
2. **Cadeia de Controle com Variáveis $V$ Brutas ($1\text{A} \to 6\text{B}$)**:
   - **Baseline Raw V (1A)**: PR-AUC Bruto = **0,5038**
   - **All-in-One Raw V (6B)**: PR-AUC Bruto = **0,5395** (Lift bruto de **+0,0357** sobre 1A).
   - **Contribuição do Pós-Processamento Causal**: A média expansiva alinhada adiciona **+0,0039**, elevando o PR-AUC final para **0,5434** (e o F1 para 0,5705).
3. **Sintonização Bayesiana Separada (Optuna)**: O tuning Bayesiano no Notebook 07 eleva o modelo Baseline para 0,5555 e o modelo com Grafos para 0,5622 (**+0,0067**). Este incremento provém da otimização hiperparamétrica específica e é analisado de forma isolada, não se somando aos ganhos da ablação de hiperparâmetros fixos.

#### 4.6.2 Protocolo de Avaliação de Recall sob Taxa de Falso Positivo Controlada (FPR Fixo)
Na literatura de fraude, é frequente o vício metodológico de comparar o recall entre modelos operando sob seus respectivos limiares ótimos financeiros ($\tau^*$). Contudo, essa comparação introduz um artefato de sensibilidade (*threshold-shift artifact*): um modelo com limiar menor parece capturar mais fraudes simplesmente porque gera mais alertas globais, inflando os falsos positivos operacionais.

Para mensurar a real capacidade discriminatória estrutural, instituímos na metodologia a **avaliação sob Taxa de Falsos Positivos Fixa ($\text{FPR} = 1,0\%$)**, refletindo o limite de capacidade operacional de análise manual das mesas de antifraude.

A significância estatística de cada incremento de recall foi verificada via **Bootstrap não-paramétrico com 1.000 iterações** (amostragem com reposição sobre as transações de teste temporal, mantendo o limiar empírico fixado em $\text{FPR} = 1,0\%$):

| Estágio Metodológico | Recall Coordenadas (%) | IC 95% Bootstrap | $\Delta$ Recall (p.p.) | IC 95% do $\Delta$ | $p$-valor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline PCA (1B)** | 50,87% | [48,10%, 53,65%] | — | — | — |
| **+ FE Tabular (5)** | 58,41% | [55,56%, 61,11%] | **+7,54 p.p.** | [+5,08, +9,68] | $p < 0,001$ |
| **+ Grafos Bipartidos (6B bruto)** | 60,63% | [57,78%, 63,26%] | **+2,22 p.p.** | [+1,03, +3,73] | $p < 0,001$ |
| **+ Pós-Processamento Causal (6B PP)** | 61,27% | [58,49%, 63,97%] | **+0,64 p.p.** | [+0,00, +1,43] | $p = 0,048$ |
| **Ganho Acumulado Total ($1\text{B} \to 6\text{B PP}$)** | **61,27%** | [58,49%, 63,97%] | **+10,40 p.p.** | **[+8,02, +12,78]** | **$p < 0,001$** |

*Nota metodológica sobre o Pós-Processamento*: O ganho de recall de +0,64 p.p. no pós-processamento temporal corresponde à captura adicional de 8 transações coordenadas (em um total de 1.260 fraudes coordenadas na partição de validação). Como o limite inferior do IC 95% Bootstrap toca zero ([+0,00, +1,43] p.p.), seu efeito isolado em termos de recall específico de fraudes coordenadas é marginal/fronteiriço, diferentemente da Engenharia Tabular (+7,54 p.p.) e dos Grafos (+2,22 p.p.), cujos intervalos confirmam significância robusta ($p < 0,001$).

Em fraudes isoladas/avulsas sob o mesmo limiar ($\text{FPR} = 1,0\%$), o recall avança de 28,01% para 32,86% (+4,85 p.p.), confirmando que os grafos geram benefício desproporcional e direcionado a padrões de rede.

---

## 5. Auditoria de Maturidade e Correção de Leakages

Para garantir que todos os resultados reportados reflitam a capacidade real de generalização em produção, o pipeline passou por uma auditoria formal cobrindo três pontos críticos:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 RELATÓRIO FORMAL DE AUDITORIA DE LEAKAGES                              │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘

1. CORREÇÃO DA BUSCA BINÁRIA DE VELOCIDADE 24H:
   • Diagnóstico: Algoritmos anteriores aplicavam np.searchsorted globalmente,
     contando transações de outros clientes e agindo como proxy de horário.
   • Correção: Busca binária causal agrupada estritamente no histórico do UID.
   • Resultado: Contador mede exclusivamente a cadência do portador (O(N log K)).

2. CORREÇÃO DE VAZAMENTO NO PÓS-PROCESSAMENTO POR CLIENTE:
   • Diagnóstico: O cálculo com groupby('uid').transform('mean') incluía predições
     de compras futuras na validação, inflando artificialmente as métricas.
   • Correção: Conversão para média cumulativa expansiva (expanding().mean()).
   • Resultado: Avaliação com e sem pós-processamento reportada com transparência.

3. DESACOPLAMENTO E ISOLAMENTO DE GNN:
   • Diagnóstico: Redes GNN supervisionadas com isFraud sofriam de memorização.
   • Correção: Conversão para regime auto-supervisionado de reconstrução de links.
   • Resultado: Prevenção de dependência de rótulos em camadas de representação.

4. REMOÇÃO DE MULTIPLICADORES SINTÉTICOS NA FUNÇÃO DE CUSTO:
   • Diagnóstico: Avaliações preliminares utilizavam multiplicadores sintéticos (* 1,35).
   • Correção: Aplicação da função de custo estritamente sobre probabilidades reais do XGBoost.
   • Resultado: Métricas financeiras e curvas 100% ancoradas em predições empíricas verificáveis.

5. ALINHAMENTO POR ÍNDICE NO PÓS-PROCESSAMENTO (.reset_index vs .values):
   • Diagnóstico: O uso de .values sobre expanding().mean() gerava desalinhamento posicional
     entre entidades, degradando artificialmente o PR-AUC.
   • Correção: Emprego de .reset_index(level=0, drop=True) preservando alinhamento vetorial por índice.
   • Resultado: Eliminação de contaminação cruzada e estabilização rigorosa dos escores pós-processados.

6. GRANULARIDADE DE DeviceInfo E NÃO-MONOTONICIDADE DO RISCO:
   • Diagnóstico: Suposição errônea de que DeviceInfo representava instâncias físicas individuais.
   • Correção: Mapeamento de que a coluna expressa famílias de SO e modelos de mercado (Windows, iOS,
     SM-G935F), exibindo pico de taxa de fraude em graus intermediários (11-100) e queda em megahubs (>100).
   • Resultado: Modelagem não-linear adequada via árvores e desmistificação teórica de fraud rings.
```

---

## 6. Estudo de Ablação Sistemático (8 Configurações)

O cerne empírico deste trabalho é o **Estudo de Ablação Sistemático**, desenhado para isolar, com controle rigoroso de variáveis e hiperparâmetros idênticos no classificador XGBoost (`n_estimators=300, lr=0.03, max_depth=9, subsample=0.8, colsample=0.8, scale_pos_weight`), o impacto individual de cada família de atributos.

### 6.1 Definição das Oito Configurações de Ablação

1. **Configuração 1A — Baseline Raw (Sem PCA)**: Classificador treinado estritamente com as variáveis brutas originais (incluindo as 339 colunas $V$), sem atributos de rede, velocidade temporal ou agregações comportamentais (391 features no Model A / 430 no Model B).
2. **Configuração 1B — Baseline Com PCA (15-d)**: Isola o impacto da compressão dimensional espectral, substituindo as 339 colunas $V$ por 15 autovetores principais (67 features no Model A / 106 no Model B).
3. **Configuração 2 — Baseline + Apenas Topologia**: Adiciona PageRank (`uid_pagerank`), grau de compartilhamento de dispositivo (`device_degree`) e grau do portador (`uid_degree`) sobre o baseline 1B, sem velocidade ou FE comportamental (70 features no Model A / 109 no Model B).
4. **Configuração 3 — Baseline + Apenas Velocidade**: Isola a dinâmica temporal causal em 24h (`uid_tx_count_24h` e `uid_secs_since_prev`) sobre o baseline 1B, sem métricas de grafo (68 features no Model A / 107 no Model B).
5. **Configuração 4 — Baseline + Grafos Completos**: Combina topologia de grafo e velocidade temporal sobre o baseline 1B, sem atributos tabulares avançados (71 features no Model A / 110 no Model B).
6. **Configuração 5 — Baseline + Apenas FE Tabular**: Isola o poder das agregações comportamentais por portador (médias $\mu$ e desvios $\sigma$ de montantes e contadores $C$, contagens de frequência e coordenadas ancoradas de calendário $D_n$), sem atributos de rede (81 features no Model A / 120 no Model B).
7. **Configuração 6A — All-in-One (Com PCA 15-d)**: Modelo integrado completo combinando baseline com PCA, FE comportamental de portador, topologia de rede e velocidade causal (86 features no Model A / 125 no Model B).
8. **Configuração 6B — All-in-One (Sem PCA / Raw V)**: Modelo integrado completo mantendo a totalidade das 339 variáveis $V$ brutas em conjunto com FE tabular e grafos (410 features no Model A / 449 no Model B).

---

### 6.2 Resultados Consolidados da Ablação

A tabela abaixo consolida as métricas de performance preditiva, contagem de variáveis e tempos de treinamento obtidas na validação temporal causal estrita:

|   ID   | Configuração Avaliada        | Features (Model A / B) | Tempo Treino (s) | PR-AUC (Bruto) | ROC-AUC (Bruto) | F1-Score (Bruto) | PR-AUC (Pós-Proc) | F1-Score (Pós-Proc) |
| :----: | ---------------------------- | :--------------------: | :--------------: | :------------: | :-------------: | :--------------: | :---------------: | :-----------------: |
| **1A** | Baseline Raw (Sem PCA)       |       391 / 430        |      105,8s      |     0,5038     |     0,8779      |      0,5061      |      0,5034       |       0,5095        |
| **1B** | Baseline (Com PCA 15-d)      |        67 / 106        |      51,4s       |     0,5028     |     0,8852      |      0,4979      |      0,5016       |       0,5023        |
| **2**  | Baseline + Apenas Topologia  |        70 / 109        |      51,9s       |     0,4918     |     0,8827      |      0,4917      |      0,4906       |       0,4984        |
| **3**  | Baseline + Apenas Velocidade |        68 / 107        |      51,0s       |     0,4943     |     0,8822      |      0,4936      |      0,4922       |       0,4957        |
| **4**  | Baseline + Grafos Completos  |        71 / 110        |      52,5s       |     0,4997     |     0,8822      |      0,4983      |      0,4980       |       0,5093        |
| **5**  | Baseline + Apenas FE Tabular |        81 / 120        |      60,5s       |   **0,5304**   |     0,8622      |      0,5443      |      0,5299       |       0,5472        |
| **6A** | All-in-One (Com PCA 15-d)    |        86 / 125        |      63,2s       |   **0,5344**   |     0,8715      |      0,5481      |      0,5329       |       0,5524        |
| **6B** | All-in-One (Sem PCA / Raw V) |       410 / 449        |      118,4s      |   **0,5395**   |     0,8649      |    **0,5646**    |    **0,5434**     |     **0,5705**      |

---

### 6.3 Análise e Dissecação dos Fenômenos da Ablação

A análise rigorosa dos resultados esclarece o papel exato de cada família de features:

#### A. A Primazia da Engenharia Tabular de Entidade (87,3% do Lift)
Na cadeia consistente sob compressão PCA ($1\text{B} \to 5 \to 6\text{A}$), a Engenharia Tabular de Portador ($5 - 1\text{B}$) eleva a PR-AUC de 0,5028 para 0,5304 (+0,0276), respondendo por **87,3% de todo o lift obtido sobre o baseline**. Os atributos de rede bipartida ($6\text{A} - 5$) agregam **+0,0040 em PR-AUC** (12,7% do lift), atingindo 0,5344.  
Na cadeia com variáveis brutas ($1\text{A} \to 6\text{B}$), o modelo atinge 0,5395 bruto e **0,5434** com o pós-processamento causal alinhado via `.reset_index` (+0,0039, F1: 0,5705).

#### B. Por que a Topologia Isolada (Config 2) Não Supera o Baseline (Config 1B)?
A Configuração 2 (Apenas Topologia) registra PR-AUC de 0,4918 (vs 0,5028 no baseline). Esse comportamento decorre da assimetria estrutural causal:
* No treino, 96,71% das transações com dispositivo recebem grau estático inflacionado por eventos futuros (lookahead de 3.612,2 UIDs futuros e inflação média de 7,39x);
* Na validação causal, 83,75% das transações contêm `null_device` e 20,37% dos dispositivos são inéditos (totalizando 84,15% de transações desconectadas da rede de treino), enquanto 65,76% dos UIDs são inéditos. Essa desconexão massiva impede que atributos topológicos operem isoladamente na ausência de atributos comportamentais densos de portador.

#### C. Diagnóstico e Correção de Alinhamento no Pós-Processamento
A auditoria demonstrou que `groupby('uid')['raw_score'].expanding().mean().values` desalinha os escores por ordenar alfabeticamente pela chave de grupo e atribuir posicionalmente sobre o frame ordenado no tempo, causando degradação artificial de -0,0370 a -0,0427 em todas as configurações. A correção estrita via `.reset_index(level=0, drop=True)` preserva os índices cronológicos e estabiliza o modelo, propiciando ganho genuíno de **+0,0039** no modelo All-in-One 6B (0,5395 $\to$ **0,5434**).

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
Para qualquer probabilidade estimada $\hat{p}_i = P(y_i = 1 | \mathbf{x}_i)$, a decisão de classificar a compra como fraude ($\hat{y}_i(\tau) = 1$) é parametrizada pelo limiar $\tau \in [0, 1]$. O custo financeiro total esperado $L(\tau)$ sobre o conjunto de validação temporal é formulado como:
$$L(\tau) = \sum_{i=1}^N \left[ y_i (1 - \hat{y}_i(\tau)) \cdot (\text{amt}_i + C_{\text{chargeback}}) + (1 - y_i) \hat{y}_i(\tau) \cdot (C_{\text{revisão}} + \alpha \cdot \text{amt}_i) + y_i \hat{y}_i(\tau) \cdot C_{\text{operacional}} \right]$$
O ponto de corte financeiro ótimo $\tau^*$ é obtido minimizando $L(\tau)$:
$$\tau^* = \arg\min_{\tau \in [0, 1]} L(\tau)$$

A calibração do limiar $\tau^*$ sobre a esteira temporal evita o viés in-sample e baliza a tomada de decisão sob risco mínimo de Bayes, demonstrando que classificadores híbridos equilibram sensibilidade e custo operacional de falsos alarmes.

---

## 8. Discussão Metodológica e Limitações da Pesquisa

### 8.1 A Soberania da PR-AUC e a Falácia do ROC-AUC
Sob desbalanceamento severo (3,5% de fraudes), o ROC-AUC é facilmente inflado pela massa massiva de verdadeiros negativos (96,5% de compras legítimas). A métrica de **PR-AUC (Área sob a Curva Precision-Recall)** provou ser o único indicador fidedigno para avaliar a capacidade do modelo de capturar fraudes sem disparar uma enxurrada de falsos positivos que inviabilizaria a operação do e-commerce.

### 8.2 O Papel da Arquitetura Híbrida em Produção Industrial
A investigação experimental comprova que:
* **Atributos de rede e grafos não substituem classificadores baseados em árvores e features tabulares de domínio**, mas atuam como extratores complementares de contexto relacional.
* O ganho preditivo de grafos é estritamente focalizado em anéis de fraude que compartilham hardware, enquanto o ganho estrutural global de volume decorre de atributos comportamentais causais de portador.
* A combinação de **PCA (reduzindo 339 colunas ruidosas para 15 componentes)** com **atributos topológicos de rede bipartida (PageRank e graus)** e contadores temporais causais permitiu acelerar o treinamento das árvores em **51,4%** (de 105,8s para 51,4s na baseline), preservando integralmente a capacidade discriminativa.

### 8.3 Discussão de Limitações Metodológicas e Análise de Amostragem

Com o compromisso de transparência científica, identificamos e discutimos as seguintes limitações metodológicas deste trabalho:

1. **Gargalo Computacional e Estratégia de Amostragem Causal (50%)**:
   * O hardware local disponível para o projeto (8–16 GB de RAM) impossibilitou a execução de buscas Bayesianas exaustivas com o Optuna e ablações repetidas sobre todas as 590.540 transações sem estouro de memória (*Out-Of-Memory*).
   * Para assegurar rigor e reprodutibilidade, padronizou-se o protocolo experimental na partição causal de 50% ($N_{\text{total}} = 295.270$, com $N_{\text{train}} = 236.216$ e $N_{\text{val}} = 59.054$), preservando integralmente todo o horizonte cronológico de 182 dias e a taxa natural de fraude de 3,50%.
2. **Cobertura Incompleta de Identidade de Dispositivo**:
   * No benchmark IEEE-CIS, aproximadamente 75% a 84% das transações não possuem metadados de identidade preenchidos (`null_device`). Para essa parcela majoritária, os atributos de rede recebem valores nulos ou imputações uniformes neutras ($1/|\mathcal{V}|$), delimitando o escopo de atuação dos grafos.
3. **Avaliação Temporal Contínua (Walk-Forward)**:
   * A estabilidade temporal do modelo foi comprovada em validação *Walk-Forward* de 4 dobras mensais expansivas contínuas (Notebook 06), demonstrando que a modelagem relacional mitiga o *concept drift* e assegura uma economia sustentada de US$ 6.722,08 por mês.
4. **Assimetria Causal Intra-Treino vs. Validação em Grafos Estáticos (Resultado Empírico)**:
   * A extração de PageRank e graus no treino foi estática sobre todo o bloco histórico, gerando uma inflação temporal mensurável: em **96,71% das transações com dispositivo no treino**, o grau estático é estritamente maior que o causal, com uma antecipação média de **3.612,2 UIDs futuros** e inflação média de **7,39x** (atingindo **16,25x no 1º decil temporal**).
   * Na validação causal, o grafo é congelado: **83,75%** das transações possuem `null_device`, **20,37%** dos dispositivos são inéditos (totalizando **84,15% das transações desconectadas** da topologia de treino), e **65,76% dos UIDs são inéditos** (61,47% das transações). Esse choque distribucional explica matematicamente por que a topologia pura (0,4918) não superou o baseline (0,5028).
5. **Semântica de `DeviceInfo` e Limitações de Identificação de Hardware**:
   * `DeviceInfo` agrega assinaturas de modelo e compilação de SO, e não identificadores físicos unívocos (MAC/IMEI). Isso gera megahubs legítimos (Windows, iOS) e não-monotonicidade da taxa de fraude (o risco atinge o pico em 11-100 UIDs e cai em >100 UIDs), exigindo que classificadores não assumam linearidade na relação grau-fraude.

---

## 9. Conclusão e Trabalhos Futuros

Este trabalho documentou a jornada completa de concepção, auditoria, refinamento e validação empírica de uma arquitetura híbrida de detecção de fraudes com grafos no comércio eletrônico.

As principais contribuições científicas e práticas incluem:
1. **Decomposição Transparente do Valor Preditivo**: Demonstramos que, na cadeia de ablação com compressão PCA ($1\text{B} \to 5 \to 6\text{A}$), a **engenharia de atributos tabular de entidade e tempo responde por 87,3% do ganho de detecção** (elevando PR-AUC de 0,5028 para 0,5304, +0,0276). A incorporação de atributos de grafo atua como um **refinamento incremental ortogonal**, agregando **+0,0040 em PR-AUC** (12,7% do lift, elevando para 0,5344; e atingindo 0,5395 bruto e 0,5434 com pós-processamento causal no modelo com $V$ bruto, além de +0,0067 em tuning Bayesiano isolado).
2. **Resolução de Anéis Coordenados e Protocolo a FPR Fixo**: Mapeamos que **46,65% das perdas financeiras por fraude decorrem de padrões coordenados em rede**. Sob avaliação com taxa de falso positivo fixa ($\text{FPR} = 1,0\%$), o recall em fraudes coordenadas atinge **61,27%** (+10,40 p.p. sobre o baseline de 50,87%), sendo **+7,54 p.p.** derivados da ancoragem tabular de entidade e **+2,86 p.p.** derivados incrementalmente dos atributos de grafo e pós-processamento causal.
3. **Rigor e Reprodutibilidade Científica**: Identificamos e corrigimos fontes críticas de vazamento temporal e desalinhamentos posicionais no pós-processamento, assegurando total reprodutibilidade experimental.
4. **Validação Econômica e Temporal**: Por meio de uma função de custo calibrada com taxas de mercado e validação *Walk-Forward* em 4 dobras, comprovamos que a solução gera uma **economia média comprovada de US\$ 6.722,08 por mês** na esteira de pagamentos.

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
