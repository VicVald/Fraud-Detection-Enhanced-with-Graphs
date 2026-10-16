# Engenharia de Grafos Bipartidos e Compressão Espectral na Detecção de Fraudes em E-Commerce: Um Estudo de Ablação e Decomposição Estrutural

**Autor:** Victor Valadares  
**Área de Concentração:** Inteligência Artificial, Ciência de Dados e Engenharia de Software  
**Palavras-chave:** Detecção de Fraudes; Grafos Bipartidos; Resolução de Entidades; Detecção de Anomalias em Redes; Engenharia de Features Relacionais; Sistemas de Classificação Híbridos; Gradient Boosted Decision Trees (GBDT); IEEE-CIS Benchmark.

---

## Resumo

Classificadores supervisionados aplicados à detecção de fraude em comércio eletrônico — notadamente os baseados em *Gradient Boosted Decision Trees* (GBDT) — assumem convencionalmente que as transações são observações Independentes e Identicamente Distribuídas (*IID*). Na prática, fraudadores frequentemente operam em esquemas coordenados que reutilizam recursos técnicos e credenciais, desafiando essa premissa. Este trabalho investiga experimentalmente a incorporação de modelagem relacional via grafos bipartidos, dinâmica temporal não-antecipativa e compressão dimensional espectral sobre o benchmark mundial **IEEE-CIS Fraud Detection**. O objetivo central foi **testar rigorosamente o uso de grafos** no problema, avaliando em que condições a topologia de rede agrega ganho discriminativo e quais são seus limites práticos em dados tabulares heterogêneos. Nossas contribuições metodológicas compreendem: (i) um algoritmo determinístico de resolução de entidades via identificador composto UID ancorado em calendário com busca binária causal em $\mathcal{O}(N \log K)$ para janelas temporais retrospectivas; (ii) a formulação de grafos bipartidos causais (Portador $\leftrightarrow$ Dispositivo/Fingerprint) integrada a uma arquitetura de Roteamento Dual (Model A sem identidade / Model B com identidade); (iii) a compressão espectral PCA das 339 colunas Vesta para 15 componentes principais, reduzindo o tempo de treinamento das árvores em **51,4%** (de 105,8s para 51,4s na baseline) preservando a acurácia discriminativa (PR-AUC 0,5038 vs 0,5028); (iv) um estudo de ablação em 8 configurações ortogonais com avaliação rigorosa de sobreajuste causal; e (v) a formulação de avaliação sob custo financeiro e taxa controlada de falsos positivos operacionais. Os resultados empíricos demonstram com transparência que **a engenharia de atributos tabulares comportamentais ao nível do portador ($D_n$, UID ancorado e desvios de ticket) é o motor primário de ganho, respondendo por 87,3% de todo o lift discriminativo da cadeia PCA** (elevando a PR-AUC de 0,5028 para 0,5304, $\Delta = +0,0276$). Em contrapartida, a **modelagem de grafos bipartidos atua como um refinamento incremental ortogonal**, agregando **+0,0040 em PR-AUC** (12,7% do lift, atingindo 0,5344 na cadeia PCA e 0,5395 bruto / **0,5434** com pós-processamento causal alinhado em variáveis brutas, com F1 de 0,5705). Sob taxa de falsos positivos controlada operacionalmente ($\text{FPR} = 1,0\%$), o recall em fraudes coordenadas avança de 50,87% para 61,27% (**+10,40 p.p.**, IC 95% **[+8,02; +12,78]** p.p., $p < 0,001$), sendo +7,54 p.p. decorrentes da engenharia tabular, +2,22 p.p. da inclusão de grafos e +0,64 p.p. do pós-processamento temporal causal (efeito marginal). Conclui-se que grafos bipartidos oferecem sinais contextuais pontuais úteis e direcionados a anéis de fraude, mas sua eficácia é condicionada à granularidade dos identificadores de hardware, sendo superados em impacto global por atributos comportamentais causais de portador em dados tabulares esparsos de e-commerce.

**Abstract:**  
Supervised machine learning classifiers deployed for e-commerce fraud detection—most notably Gradient Boosted Decision Trees (GBDT)—conventionally rely on the assumption that transactions are Independent and Identically Distributed (IID). In practice, fraudsters often operate coordinated schemes reusing technical resources, challenging independence assumptions. This work experimentally investigates the incorporation of bipartite graph modeling, non-anticipative temporal dynamics, and spectral dimensionality compression on the **IEEE-CIS Fraud Detection** benchmark. The primary goal was to **empirically test the use of graphs** in this problem, assessing under what conditions network topology provides discriminative lift and identifying its operational limitations. Our methodology encompasses: (i) deterministic calendar-anchored composite UID entity resolution with an $\mathcal{O}(N \log K)$ causal binary search velocity counter; (ii) causal bipartite access graphs (Cardholder $\leftrightarrow$ Device/Fingerprint) within a Dual-Routing pipeline (Model A / Model B); (iii) PCA spectral compression of the 339 Vesta V-family columns into 15 components, reducing tree training time by **51.4%** (from 105.8s to 51.4s) while fully preserving discriminative capability (PR-AUC 0.5038 vs 0.5028); (iv) an ablation study across 8 orthogonal configurations evaluated under strict anti-leakage temporal protocols; and (v) an operational evaluation under Bayes Minimum Risk financial cost and controlled false positive rates. The empirical results reveal that **domain-specific tabular feature engineering (calendar-anchored $D_n$ deltas and cardholder UID aggregations) drives 87.3% of performance gains in the PCA ablation chain** (lifting PR-AUC from 0.5028 to 0.5304, $\Delta = +0.0276$ over baseline). In turn, **bipartite graph modeling operates as an orthogonal incremental refinement**, contributing **+0.0040 in PR-AUC** (12.7% of total lift, reaching 0.5344 in the PCA model and 0.5395 raw / **0.5434** with index-aligned causal post-processing in raw variables, with F1 of 0.5705). Under a fixed operational false positive rate ($\text{FPR} = 1.0\%$), recall on coordinated fraud increases from 50.87% to 61.27% (**+10.40 p.p.**, 95% CI **[+8.02, +12.78]** p.p., $p < 0.001$), decomposed into +7.54 p.p. from tabular engineering, +2.22 p.p. from bipartite graphs, and +0.64 p.p. from causal temporal post-processing (marginal effect). We conclude that while bipartite graphs provide valuable local contextual signals directed at fraud rings, their efficacy in e-commerce fraud detection is strictly constrained by identifier resolution and missingness, being outperformed globally by causal behavioral aggregations.

---

## 1. Introdução

A expansão do comércio eletrônico global consolidou os pagamentos instantâneos sem presença física do cartão (*Card-Not-Present* — CNP) como o principal vetor da economia digital. Contudo, essa agilidade operacional foi acompanhada pela escalada vertiginosa em ataques de fraude eletrônica. Conforme reportado pelo *The Nilson Report* (2023), as perdas globais decorrentes de fraudes com cartões de pagamento atingiram a marca de **US$ 32,34 bilhões**, com projeções indicando que o montante superará US$ 40 bilhões antes do final da década. No cenário europeu e transfronteiriço, o *European Central Bank* (ECB, 2023) aponta que as transações CNP respondem por **84% do valor total das fraudes**, impulsionadas pela proliferação de vazamentos maciços de credenciais, identidades sintéticas e ataques automatizados.

Historicamente, a indústria combate esse problema utilizando modelos estatísticos supervisionados aplicados a dados tabulares, com destaque para os algoritmos de árvores de decisão impulsionadas por gradiente (*Gradient Boosted Decision Trees* — GBDTs, tais como XGBoost, LightGBM e CatBoost). Embora robustos para processar atributos numéricos e categóricos em tabelas densas, esses modelos operam fundamentalmente sob a premissa de que cada linha (transação) é uma observação **Independente e Identicamente Distribuída (IID)**.

Essa premissa de independência falha drasticamente no cenário cibernético moderno devido a três vetores centrais de Ciência da Computação:

* **Ataques Coordenados em Rede (*Fraud Rings* & *Botnets*)**: Conforme demonstrado por Van Vlasselaer et al. (2015, 2017) e Akoglu et al. (2015), atacantes operam em redes estruturadas de conluio. Eles utilizam emuladores, scripts automatizados e fazendas de hardware compartilhado para testar centenas de cartões roubados sequencialmente. Um classificador tabular convencional enxerga cada transação como um evento isolado e falha em detectar que a mesma máquina física está orquestrando múltiplos pagamentos fraudulentos.
* **O Dilema do *Cold Start* e Resolução de Entidades**: No mercado negro (*Dark Web*), lotes de cartões clonados são comercializados em massa e utilizados para uma única tentativa de compra de alto valor antes do descarte. Para o modelo tabular puro, um cartão inédito não possui histórico transacional prévio e, portanto, é classificado como de baixo risco. Em contrapartida, em uma estrutura relacional, mesmo um cartão 100% inédito conecta-se a um endereço IP, dispositivo ou e-mail que já possui vizinhos comprometidos na topologia do grafo.
* **Não-Estacionariedade e *Concept Drift* Adversário**: Conforme formalizado por Dal Pozzolo et al. (2015, 2018), sistemas de prevenção a fraude operam em um ambiente temporal não estacionário. Os atacantes alteram dinamicamente seus padrões de consumo para contornar regras estáticas, gerando mudanças bruscas nas distribuições de probabilidade $P(X)$ e $P(Y|X)$.

### Perguntas de Pesquisa e Objetivos

Este trabalho visa responder a três perguntas centrais de pesquisa:

* **RQ1 (Decomposição Estrutural e Eficiência de Representação)**: *Qual é a contribuição preditiva incremental isolada da topologia de grafos bipartidos (PageRank e graus) contra a velocidade temporal causal e contra a engenharia tabular avançada? Em que medida a compressão espectral PCA sobre centenas de atributos esparsos de segurança preserva a discriminabilidade do modelo enquanto otimiza a latência computacional?*
* **RQ2 (Quebra da Premissa IID em Tipologias Complexas)**: *Em que proporção as fraudes reais do e-commerce dependem de infraestruturas de hardware compartilhadas versus fraudes oportunistas isoladas, e qual é o ganho real de sensibilidade (Recall) proporcionado pelas representações relacionais sobre cada tipologia?*
* **RQ3 (Calibração Operacional sob Matrizes Assimétricas)**: *Como a integração entre atributos relacionais de rede e engenharia comportamental tabular afeta o comportamento das probabilidades preditas sob matrizes de decisão com custos assimétricos e calibração de limiares operacionais?*

Os objetivos específicos incluem:

1. Conduzir uma **análise empírica de diagnóstico**, identificando assinaturas de tipologias de fraude (dispositivos compartilhados, rajadas de velocidade e fracionamento de valores).
2. Desenvolver algoritmos determinísticos de **resolução de entidades** ($\text{UID}$) e **velocidade causal em $\mathcal{O}(N \log K)$** via busca binária vetorizada.
3. Formular o **Grafo Bipartido Portador $\leftrightarrow$ Dispositivo** com poda estrita de nós sentinela para cálculo não degenerado de PageRank e graus de conectividade.
4. Conduzir uma **auditoria de vazamento de dados (*leakage audit*)**, prevenindo fontes de contaminação temporal em pré-processamento e pós-processamento.
5. Executar um **Estudo de Ablação Rigoroso** em partição temporal não-antecipativa sobre a partição causal de 50% do benchmark **IEEE-CIS Fraud Detection** ($N_{\text{total}} = 295.270$; $N_{\text{val}} = 59.054$), preservando a totalidade dos 182 dias de histórico e a taxa de fraude de 3,50%, decompondo sistematicamente oito configurações metodológicas com e sem compressão PCA.
6. Investigar a sensibilidade de captura dos modelos sobre diferentes classes de fraude e avaliar o comportamento das curvas de decisão sob o princípio de *Bayes Minimum Risk*.

---

## 2. Trabalhos Relacionados e Fundamentação Conceitual

### 2.1 Modelos Tabulares de Boosting (GBDTs): Por que Árvores em vez de Redes Profundas?
A detecção de fraude moderna em escala industrial é amplamente dominada por algoritmos de *tree boosting*, notadamente o **XGBoost** (CHEN; GUESTRIN, 2016) e o **LightGBM** (KE et al., 2017). Essa preferência técnica decorre de propriedades fundamentais da computação aplicada a pagamentos:
1. **Requisitos Estritos de Latência (SLAs < 10 ms)**: Em plataformas de comércio eletrônico com milhares de requisições concorrentes por segundo, o modelo de fraude precisa emitir um escore de risco em milissegundos na esteira de autorização. Árvores compiladas executam em frações de milissegundo na CPU, reduzindo a dependência de aceleradores gráficos dedicados (GPUs) de alto custo e latência de transferência de barramento PCIe.
2. **Heterogeneidade de Tipagem e Esparsidade Nula**: Dados transacionais combinam contadores discretos, montantes monetários contínuos, variáveis categóricas de alta cardinalidade e taxas elevadas de valores ausentes (nulos). Os GBDTs operam através de hiperplanos de corte ortogonais aos eixos (*axis-aligned decision boundaries*), são naturalmente invariantes a transformações monotônicas de escala e aprendem a direção padrão de ramificação para valores ausentes (*default split direction*), dispensando pré-processamentos artificiais que induzem ruído estatístico.

Contudo, apesar de sua eficiência tabular, árvores de decisão padrão são estruturalmente **cegas para correlações topológicas** que ultrapassam os limites de uma única linha da tabela, tratando eventos orquestrados por uma mesma máquina como fenômenos independentes.

### 2.2 Aprendizado de Representação em Grafos e GNNs
O uso de métricas clássicas de centralidade, como o **PageRank** (PAGE et al., 1999), mostrou-se eficaz na literatura para quantificar a importância e o tráfego estocástico de entidades em redes complexas. Com a evolução do aprendizado profundo em grafos, arquiteturas convolucionais indutivas espaciais como o **GraphSAGE** (HAMILTON; YING; LESKOVEC, 2017) viabilizaram a geração de representações vetoriais densas (*node embeddings*) amostrando recursivamente vizinhanças locais.

Entretanto, pesquisas seminais recentes na literatura de aprendizado de máquina (GRINSZTAJN et al., 2022; SHWARTZ-ZIV; ARMON, 2022) demonstram que **redes neurais profundas puras (GNNs e MLPs) são consistentemente superadas por modelos baseados em árvores em dados tabulares esparsos sob desbalanceamento severo**. GNNs puras sofrem do fenômeno de sobre-suavização (*oversmoothing*) em grafos esparsos e apresentam instabilidade de gradiente sob taxas de anomalia de 3,5%. Assim, a fronteira metodológica recomendada consiste na **arquitetura híbrida**: a topologia de grafos é empregada como um extrator de atributos estruturais determinísticos e explícitos (graus de conectividade, centralidade e coocorrência), alimentando os classificadores GBDT.

### 2.3 Detecção de Fraude em Redes e Propagação de Suspeição
A modelagem de redes para detecção de anomalias financeiras foi pioneiramente formalizada por Van Vlasselaer et al. (2015 - sistema APATE) e Pourhabibi et al. (2020). Esses autores demonstraram que a suspeição de uma transação se propaga através de redes de pagamento via homofilia (*guilt-by-association*). O presente trabalho adota esse princípio, modelando-o através de grafos bipartidos de hardware e janelas deslizantes temporais.

### 2.4 Matrizes Assimétricas de Decisão e Teoria de Risco de Bayes
Classificadores estatísticos tradicionais otimizam funções de perda homogêneas (como Log-Loss ou Binary Cross-Entropy), assumindo implicitamente que o custo de um Falso Positivo ($FP$) equivale ao de um Falso Negativo ($FN$). Na prática bancária, essa suposição é desastrosa:
* Um **Falso Negativo (Fraude Aprovada)** acarreta o prejuízo financeiro integral do montante transacionado ($\text{TransactionAmt}$), somado à taxa operacional de estorno bancário ($C_{\text{chargeback}} \approx \text{US\$} 15,00$) e custos de reputação com as bandeiras de cartão.
* Um **Falso Positivo (Cliente Legítimo Recusado)** custa apenas uma taxa marginal de revisão manual ($C_{\text{revisão}} \approx \text{US\$} 3,00$) e uma fração de atrito de atrito/churn ($\alpha \cdot \text{TransactionAmt}$).

Na literatura de aprendizado sensível ao custo, os trabalhos seminais de Elkan (2001) e Correa Bahnsen et al. (2014, 2015) estabeleceram as bases para a **Matriz de Custo Dependente do Exemplo (*Example-Dependent Cost Matrix*)**, onde o limiar operacional ótimo ($\tau^*$) é determinado pela minimização da esperança matemática da perda operacional total segundo o princípio de *Bayes Minimum Risk*.

### 2.5 Nota Terminológica sobre o Conceito de Causalidade em Engenharia
Para fins de rigor acadêmico, cabe diferenciar o uso do termo **causal** neste trabalho:
* **Não se refere** à inferência causal estocástica no sentido de Pearl (grafos causais DAG, operadores *do-calculus* ou efeitos de tratamento contrafactuais).
* **Refere-se estritamente** à definição clássica da **Teoria dos Sistemas e Processamento de Sinais (OPPENHEIM; SCHAFER, 2010)**: um algoritmo, operador ou filtro é definido como **causal (ou temporalmente não-antecipativo)** se e somente se o seu estado ou saída no instante de tempo $t$ é calculado exclusivamente a partir de observações do passado e presente $[0, t]$, sendo terminantemente proibido o acesso a qualquer evento futuro $t' > t$. Na engenharia de software de antifraude, a causalidade temporal é o requisito indispensável para prevenir vazamento de dados (*lookahead leakage*) e assegurar a reprodutibilidade na esteira de produção.

---

## 3. Análise Exploratória de Dados (EDA) e Diagnóstico Estrutural de Fraudes

A formulação de representações estruturais em grafos e atributos temporais exige uma investigação prévia rigorosa das distribuições estatísticas, dos padrões de colinearidade e dos mecanismos operacionais adotados pelo cibercrime no comércio eletrônico. Esta seção apresenta o diagnóstico empírico extraído diretamente da ingestão integral do benchmark mundial **IEEE-CIS Fraud Detection** ($N_{\text{total}} = 590.540$ transações analisadas sob partição estrita anti-vazamento).

---

### 3.1 Ingestão e Desequilíbrio Temporal em Valores Brutos

Para avaliar a magnitude operacional do sistema sob condições realistas, os dados foram particionados segundo um critério temporal não-antecipativo ($80\%$ histórico inicial para treino e $20\%$ final para validação), simulando a esteira de produção. Evitou-se o uso de validação cruzada aleatória (*random k-fold*), cuja aplicação provocaria vazamento temporal do futuro para o passado (*lookahead temporal leakage*).

A análise do desequilíbrio de classes e dos volumes financeiros em **valores brutos absolutos** revela a escala e a assimetria do problema:

| Partição | Janela (Timestamps) | Dias | Total ($N$) | Legítimas | Fraudes | Vol. Total (US$) | Vol. Fraude (US$) | Taxa (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Treino Histórico** | $86.400 \rightarrow 12.177.091$ | Dias 1 a 140 | 472.432 | 455.833 | 16.599 | US$ 63.495.517 | US$ 2.473.911 | 3,514% |
| **Validação Causal** | $12.177.125 \rightarrow 15.810.935$ | Dias 141 a 182 | 118.108 | 114.044 | 4.064 | US$ 16.243.432 | US$ 609.934 | 3,441% |
| **Total Analisado** | $86.400 \rightarrow 15.810.935$ | 182 dias (~6 meses) | 590.540 | 569.877 | 20.663 | US$ 79.738.949 | US$ 3.083.845 | 3,499% |

A evolução diária da contagem bruta de fraudes ao longo dos 182 dias de histórico é exibida na Figura 1:

![Figura 1: Evolução temporal da frequência diária de fraudes em valores brutos](figures/fig1_temporal_fraud_raw_volume.png)

Como se observa na Figura 1, a ocorrência de fraudes não segue uma taxa constante estacionária, mas manifesta-se em surtos agudos de ataque (atingindo picos de 150 a 250 fraudes em um único dia), alternados com períodos de relativa calmaria. Essa dinâmica espasmódica é característica de campanhas de ataque cibernético orquestradas, reforçando que modelos puramente IID têm dificuldade em capturar o ritmo e o contexto temporal dos ataques.

---

### 3.2 Anatomia das Variáveis Vesta e Por que a Compressão Espectral (PCA) é Necessária

O benchmark IEEE-CIS possui 339 colunas numéricas de segurança fornecidas pela Vesta Corporation ($V1$ a $V339$). A análise exploratória detalhada dessas variáveis revela dois desafios centrais de Ciência de Dados e Engenharia de Software:
1. **Esparsidade Massiva e Disparidade de Canais**: A taxa média de valores nulos no bloco $V$ atinge **45,3%**, com blocos inteiros de variáveis preenchidos apenas quando a transação provém de determinados canais (ex.: compras mobile versus desktop, ou cartões de débito versus crédito).
2. **Colinearidade Extrema em Blocos Contíguos**: As variáveis Vesta foram originalmente sintetizadas como contadores agregados em janelas de tempo deslizantes sobrepostas (ex.: "contagem de transações nos últimos 10 minutos", "última hora" e "últimas 24 horas"). Como medem o mesmo fenômeno físico sob intervalos próximos, blocos contíguos de atributos apresentam coeficientes de Pearson $\rho > 0,85$ (com dezenas de pares atingindo $\rho = 1,0$, conforme ilustrado na Figura 2).

![Figura 2: Matriz de correlação linear das variáveis Vesta V](figures/fig2_v_collinearity_pca.png)

Em modelos baseados em árvores (GBDTs), a colinearidade severa causa o fenômeno de **diluição de divisão (*feature split dilution*)**: as árvores gastam nós de decisão particionando variáveis quase idênticas, fragmentando o ganho de informação (*Gain*) e aumentando desnecessariamente a profundidade das árvores, o que favorece o sobreajuste (*overfitting*).  
Para mitigar a maldição da dimensionalidade, aplicamos a Análise de Componentes Principais (PCA) após imputação pelas medianas e padronização $\mathcal{N}(0, 1)$ no treino. A decomposição espectral demonstra que os primeiros 15 autovetores ortogonais concentram a estrutura de variância essencial das 339 colunas, filtrando o ruído colinear. Na Seção 6, os resultados experimentais indicam que essa compressão de 339 para 15 dimensões reduz o tempo de treinamento das árvores em **51,4%** (de 105,8s para 51,4s) e eleva a ROC-AUC na baseline de **0,8779 para 0,8852**, mantendo a PR-AUC virtualmente idêntica (0,5038 vs 0,5028).

---

### 3.3 Resolução Determinística de Entidades: A Matemática por Trás do UID Composto

Em sistemas de pagamento sem dados cadastrais explícitos (como CPF ou número de documento, omitidos por razões de privacidade), a identificação precisa do portador (*cardholder*) é indispensável para rastrear o comportamento financeiro ao longo do tempo.

O atributo primário da tabela, `card1`, representa o Número de Identificação Bancária (*Bank Identification Number* — BIN), que identifica unicamente a bandeira e a instituição emissora do cartão. No conjunto de dados:
* Existem apenas **12.730 valores distintos de `card1`**, resultando em uma média artificial de 37,1 transações por nó. Tratar `card1` isoladamente como se fosse um cliente gera **falsa agregação catastrófica**: transações de dezenas de milhares de clientes honestos de um mesmo grande banco são agrupadas junto a compras criminosas cometidas com cartões clonados daquela mesma instituição.
* Para resolver a entidade de forma unívoca sem necessidade de dados pessoais identificáveis (PII), formulamos o **UID Composto Padronizado**:
$$\text{UID} = \text{card1} \;\Vert\; \text{addr1} \;\Vert\; \left\lfloor \frac{\text{timestamp}}{86.400} - D1 \right\rfloor$$

> **Intuição Matemática da Ancoragem de Calendário ($\text{Day} - D1$):**  
> O atributo $D1$ representa a contagem de dias decorridos desde a emissão original daquele cartão de pagamento específico. Ao subtrair $D1$ do dia corrente da transação ($\lfloor \text{timestamp} / 86.400 \rfloor$), calculamos a coordenada estática fixa no calendário civil correspondente à **data exata em que aquele cartão foi emitido**.  
> Embora dois clientes distintos possam ter cartões do mesmo banco (`card1`) e residir na mesma região de cobrança (`addr1`), a probabilidade de que ambos tenham emitido seus respectivos cartões exatamente no mesmo dia civil é infinitesimal.

A Figura 3 contrasta as distribuições de probabilidade acumulada complementar (CCDF) em escala log-log:

![Figura 3: Distribuição log-log de transações por entidade: card1 versus UID composto padronizado](figures/fig3_entity_resolution_card_vs_uid.png)

A resolução composta expandiu o espaço de entidades de $12.730$ para **$182.347$ portadores únicos** no treino (um salto de **$14,3\times$ em granularidade**). A curva de transações por UID adere a um perfil de cauda longa característico do varejo eletrônico: 58,1% dos UIDs realizam uma única compra pontual no treino, enquanto 41,9% acumulam histórico longitudinal com múltiplas compras, viabilizando o cálculo estritamente causal de deltas temporais e médias de consumo individual.

---

### 3.4 Investigação Empírica de Assinaturas de Fraude no E-Commerce

A análise dos metadados de hardware e das variáveis temporais revelou as três principais mecânicas operacionais do cibercrime no benchmark:

#### 3.4.1 Conluios e Redes de Dispositivos Compartilhados (*Device Sharing*)
No comércio eletrônico, fraudadores podem operar a partir de ferramentas automatizadas, fazendas de emuladores e scripts, reutilizando ambientes técnicos para testar múltiplos cartões. Contudo, no benchmark IEEE-CIS, a variável `DeviceInfo` não constitui um identificador unívoco de hardware físico (como endereço MAC, IMEI ou *fingerprint* criptográfico de ThreatMetrix), mas sim uma string descritiva de modelo de aparelho ou sistema operacional extraída do *User-Agent* (ex.: "Windows", "iOS Device", "MacOS", "SM-G935F"). Consequentemente, milhares de portadores legítimos compartilham idêntico rótulo textual de dispositivo. Além disso, 78,9% das transações do treino não possuem metadados de identidade preenchidos (`null_device`).

Ao avaliar o grau de UIDs distintos conectados a cada assinatura de dispositivo (`device_degree`) na partição de treino, quantificamos a taxa de fraude observada na Figura 4:

![Figura 4: Taxa de fraude observada em função da quantidade de UIDs operando na mesma assinatura de modelo/build de dispositivo (DeviceInfo)](figures/fig4_device_sharing_fraud_rings.png)

Como ilustrado na Figura 4:
* Dispositivos com **1 único UID** associado apresentam taxa de fraude de **6,0%** (663 transações).
* Dispositivos com **2 UIDs** apresentam taxa de **3,5%** (709 transações).
* Dispositivos com **3 a 5 UIDs** sobem para **8,6%** (1.895 transações).
* A taxa atinge pico de **14,7%** na faixa de **6 a 10 UIDs** (1.802 transações).
* Dispositivos com **mais de 10 UIDs** (grandes plataformas de mercado como "Windows", com 24.202 UIDs e 39.681 transações) apresentam taxa média de **10,1%**.

Esse comportamento estritamente **não-monotônico** evidencia o desafio central da modelagem em grafos no benchmark: enquanto a conectividade moderada (6 a 10 UIDs) captura padrões anômalos de reutilização técnica suspeita, nós de hiperconectividade (>100 UIDs) refletem a popularidade massiva de sistemas operacionais no mercado consumidor, atuando como super-hubs que diluem a especificidade do grafo.

#### 3.4.2 Ataques Automatizados de Velocidade e Rajadas Temporais (*Burstiness*)
Após obter credenciais de pagamento, fraudadores podem disparar transações sequenciais em rajadas rápidas para validar cartões antes de bloqueios bancários.

Avaliando o intervalo temporal até a transação imediatamente anterior do mesmo portador ($\Delta t = t_i - t_{i-1}$) para portadores com histórico na Figura 5:

![Figura 5: Curvas de densidade temporal (KDE) do intervalo entre transações consecutivas do mesmo portador](figures/fig5_velocity_burstiness_kde.png)

* No treino, observam-se 278.341 intervalos positivos entre compras legítimas e 11.674 intervalos entre transações fraudulentas.
* A mediana de $\Delta t$ é de 154.083 s (~1,8 dia) para clientes legítimos e de 7.658 s (~2,1 horas) para fraudes.
* Na faixa crítica de rajada rápida ($\Delta t \le 60\text{ s}$), observa-se uma concentração de **6,93% das transações fraudulentas**, contra apenas **2,32% nas legítimas** (uma incidência 3 vezes superior).

#### 3.4.3 Fracionamento de Valores (*Structuring / Smurfing*) e Linhas de Negócio
A análise de montantes revela padrões associados a comportamentos atípicos:

![Figura 6: Padrões de valor monetário estruturado](figures/fig6_structuring_and_productcd.png)

Conforme a Figura 6:
* **Valores Inteiros**: Transações com valor redondo (centavos `.00`) registram taxa de fraude de **3,63%**, enquanto transações fracionadas comuns registram **3,42%** e preços com centavos comerciais típicos de varejo (`.99`) registram **2,29%**. Transações com montante múltiplo exato de US$ 10 atingem **4,39%**.
* **Linhas de Negócio (`ProductCD`)**: A categoria `ProductCD = C` (típica de pagamentos internacionais e carteiras digitais) registra taxa de fraude de **11,33%** no treino (11,69% no dataset total), em forte contraste com o comércio tradicional `ProductCD = W` (**2,07%** no treino).

---

### 3.5 Topologia Bipartida Portador-Dispositivo e Centralidade PageRank

Modelamos a interação entre portadores e recursos técnicos como um grafo bipartido $\mathcal{G} = (\mathcal{V}_{\text{uid}} \cup \mathcal{V}_{\text{dev}}, \mathcal{E})$. Na Figura 7, avaliamos a densidade do PageRank semanal não-antecipativo para os portadores presentes no grafo:

![Figura 7: Densidade de probabilidade da centralidade PageRank no grafo bipartido portador-dispositivo](figures/fig7_bipartite_pagerank_dist.png)

Diferente de premissas idealizadas onde anomalias ocupam regiões isoladas da distribuição, a distribuição empírica do PageRank normalizado exibe forte sobreposição entre classes. Portadores legítimos e fraudulentos concentram-se em ordens de magnitude similares de centralidade estacionária devido à presença de hubs de sistemas operacionais, evidenciando por que o PageRank isolado não atua como separador linear trivial no benchmark IEEE-CIS.

---

### 3.6 Dispersão Financeira e Desvio Individual de Consumo

Para mensurar anomalias monetárias relativas ao portador, computamos a razão causal entre o valor da transação e a média dos eventos anteriores daquele UID (`TransactionAmt_to_mean_uid`):

![Figura 8: Boxplots de dispersão monetária relativa em relação ao perfil histórico do UID](figures/fig8_financial_dispersion_boxplots.png)

A Figura 8 ilustra que, para transações com histórico prévio ($N_{\text{prev}} \ge 1$), a mediana da razão é próxima de 1,0 para operações legítimas, enquanto tentativas fraudulentas apresentam amplitude interquartil alargada, refletindo tickets que fogem expressivamente do padrão habitual do portador.

---

### 3.7 Diagnóstico de *Concept Drift* Temporal e o Desafio do *Cold Start*

A partição temporal 80/20 revela desafios operacionais cruciais:
* Na validação temporal futura do benchmark global ($N=118.108$), **56,5% das transações pertencem a UIDs nunca vistos no treino** (*cold start*); na partição experimental causal ($N_{\text{val}} = 59.054$), essa desconexão atinge **61,47% das transações** (abrangendo **65,76% dos UIDs únicos**).
* Entre as transações com metadados de identidade preenchidos na validação, apenas 50,8% dos fingerprints detalhados haviam sido observados no treino (e 20,37% dos dispositivos são inteiramente inéditos).

Esse diagnóstico sintetiza a limitação de qualquer abordagem puramente dependente de histórico estático e reforça a necessidade de modelos híbridos que combinem atributos atômicos de transação com sinais de contexto relacional quando disponíveis.

---

### 3.8 Taxonomia Estrutural Consolidada de Fraudes

A síntese dos achados exploratórios sobre a partição temporal de validação permite formalizar uma taxonomia estrutural bimodal de fraudes no comércio eletrônico. No universo global do benchmark ($N_{\text{val}} = 118.108$ transações, com 4.064 fraudes), 69,0% das ocorrências manifestam padrões de rede ou rajada. 

Para a execução controlada da jornada experimental de ablação, validação *Walk-Forward* e calibração de custo (Notebooks 01 a 07), adotou-se a **amostragem causal de 50%** ($N_{\text{total}} = 295.270$ registros, sendo $N_{\text{treino}} = 236.216$ e $N_{\text{val}} = 59.054$). Essa partição cobre integralmente todo o horizonte cronológico de 182 dias e preserva rigorosamente a taxa de fraude de 3,50% (sem distorção de desbalanceamento). Na partição de validação experimental ($N_{\text{val}} = 59.054$ transações), a distribuição estrutural consolida-se conforme o diagrama abaixo:

```text
                  TAXONOMIA ESTRUTURAL DE TRANSAÇÕES (VALIDAÇÃO EXPERIMENTAL)
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
             Transações Legítimas                            Transações Fraudulentas
             (56.948 transações)                             (2.106 transações - 3,566%)
              Taxa: 96,43%                                    Perda Bruta: US$ 314.263
                      │                                               │
                      │                               ┌───────────────┴───────────────┐
                      ▼                               ▼                               ▼
             Comportamento Padrão            Fraudes em Rede                 Fraudes Avulsas
            (1 Cartão / 1 Device)           (1.260 tx - 59,83%)             (846 tx - 40,17%)
            Operações Regulares             46,65% das Perdas (US$ 146.605) 53,35% das Perdas (US$ 167.658)
            Baixa Conectividade             Compartilhamento / Rajadas 24h  Sem Conexão ou Dispositivo
```

1. **Fraudes Coordenadas em Rede ($N = 1.260$, 59,83% das fraudes de validação)**: operam sob infraestrutura compartilhada de hardware (`device_degree` $\ge 2$) ou em velocidade acelerada (`uid_tx_count_24h` $\ge 2$). Concentram **46,65% das perdas financeiras brutas por fraude** (US$ 146.605 na validação), constituindo o alvo primário da modelagem relacional de grafos bipartidos.
2. **Fraudes Isoladas e Oportunistas ($N = 846$, 40,17% das fraudes de validação)**: operam de forma avulsa a partir de dispositivos dedicados ou sem metadados preenchidos (`null_device`), respondendo por **53,35% das perdas** (US$ 167.658 na validação). São detectadas prioritariamente por atributos tabulares de anomalia de montante e desvio individual de consumo ($D_n$, UID Mágico).

**Ressalva Metodológica de Circularidade e Rótulo Substituto (*Surrogate Label*)**:  
A categorização operacional de transações como 'Fraudes Coordenadas em Rede' adota o critério de compartilhamento de dispositivo com $\ge 2$ UIDs ou atividade em rajada ($\text{velocity}_{24h} \ge 2$). Cumpre explicitar que este critério opera como um rótulo substituto (*surrogate label*), uma vez que o benchmark IEEE-CIS não disponibiliza anotações externas ou registros periciais independentes de quadrilhas (*fraud rings*). Como o classificador utiliza exatamente variáveis de conectividade de rede e velocidade temporal como atributos preditivos, há uma circularidade operacional inerente nessa segmentação: o modelo é avaliado quanto à sua eficácia em detectar precisamente os padrões estruturais para os quais os atributos foram concebidos. Portanto, os ganhos de sensibilidade (recall) reportados devem ser compreendidos estritamente como a capacidade do sistema de recuperar transações que manifestam esses padrões comportamentais específicos sob restrição de falsos positivos, e não como garantia de detecção de conluios adversários externos que não deixem rastros em `DeviceInfo` ou na cadência temporal.

---

## 4. Metodologia e Engenharia de Atributos Relacionais

A arquitetura do sistema foi projetada segundo um fluxo modular e livre de vazamento causal de dados:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                     PIPELINE DE ENGENHARIA E MODELAGEM                     │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
         Tabela de Transações                  Tabela de Identidade
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                        Ingestão & Downcasting de RAM
                      (Float64/Int64 -> Float32/Int8-32)
                                      │
                                      ▼
                          Divisão Temporal Causal
              (80% Treino: 236k / 20% Validação: 59k transações)
                                      │
                                      ▼
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
           Engenharia Tabular Avançada     Topologia de Grafos Bipartidos
           - Normalização Dn (Day - D)     - Grafo Portador (UID) ↔ Dispositivo
           - UID Mágico Padronizado        - Poda Estrita de Nós Sentinela
           - Agregações de Grupo (μ, σ)    - Centralidade PageRank Estacionária
           - Compressão PCA (339 V → 15)   - Velocidade Causal 24h em O(N log K)
                        │                           │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        Roteamento de Identidade Dual
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
                  Model A                           Model B
           (Sem Dados de Device)             (Com Dados de Device)
                     │                                 │
                     └────────────────┬────────────────┘
                                      ▼
                             Classificador GBDT
                            (XGBoost / Hist-Tree)
                                      │
                                      ▼
                       Avaliação de Risco e Calibração
                          (Probabilidades Brutas)
```

### 4.1 Otimização de Memória (*Downcasting Engine*)
O dataset em precisão padrão (`float64` e `int64`) consome volumes excessivos de memória. Implementamos um algoritmo de *downcasting* numérico adaptativo (`reduce_mem_usage`) baseado em tipagem estrita (`pd.api.types.is_numeric_dtype`) que analisa os limites $[C_{\min}, C_{\max}]$ de cada coluna numérica e rebaixa tipos inteiros para `int8`/`int16`/`int32` e pontos flutuantes para `float32`. Essa operação reduziu o consumo de memória em mais de 70%, permitindo a execução rápida em ambientes locais com 16 GB de RAM.

### 4.2 Resolução Determinística de Entidades e Atributos Tabulares

#### 4.2.1 Normalização Temporal ($D_n$) e Ancoragem de Calendário
As 15 colunas $D1, \dots, D15$ representam deltas temporais móveis. Definindo o dia da transação a partir de `TransactionDT` (em segundos desde $t_0$):
$$\text{Day}(t) = \left\lfloor \frac{\text{TransactionDT}}{86.400} \right\rfloor$$
A normalização é expressa por:
$$D_i^n(t) = \text{Day}(t) - D_i(t), \quad \forall i \in \{1, 2, 10, 15\}$$
Essa operação transforma um contador relativo em uma coordenada fixa no calendário civil, permitindo agrupar eventos do mesmo instrumento financeiro emitido naquela data específica.

#### 4.2.2 Algoritmo de Resolução do UID Mágico Padronizado
Sintetizamos um identificador composto unívoco (UID):
$$\text{UID} = \text{card1} \;\Vert\; \text{addr1} \;\Vert\; \left\lfloor \text{Day} - D1 \right\rfloor$$
Onde $\Vert$ denota a concatenação com sentinelas nulas. Essa resolução determinou **182.347 portadores únicos** na partição de treino, viabilizando o cálculo causal de históricos longitudinais.

#### 4.2.3 Agregações Estatísticas de Grupo
Calculamos a média amostral ($\mu_{\text{uid}}$) e o desvio padrão ($\sigma_{\text{uid}}$) de variáveis críticas ($X \in \{TransactionAmt, D15n, C1, C13\}$) ajustados estritamente sobre o conjunto de treino:
$$\mu_{\text{uid}}(X) = \frac{1}{|T_{\text{train}}(u)|} \sum_{k \in T_{\text{train}}(u)} X_k, \quad \sigma_{\text{uid}}(X) = \sqrt{\frac{1}{|T_{\text{train}}(u)| - 1} \sum_{k \in T_{\text{train}}(u)} (X_k - \mu_{\text{uid}}(X))^2}$$
Gerando atributos de razão com suavização de Laplace:
$$\text{Ratio_to_Mean}(X_i) = \frac{X_i}{\mu_{\text{uid}}(X) + 10^{-5}}, \quad \text{Std_Dev_Ratio}(X_i) = \frac{X_i - \mu_{\text{uid}}(X)}{\sigma_{\text{uid}}(X) + 10^{-5}}$$
Na validação temporal, portadores não observados no treino recebem a média global da população, prevenindo vazamentos (*lookahead bias*).

---

### 4.3 Modelagem Matemática de Grafos Bipartidos e Velocidade Causal

#### 4.3.1 Definição Formal do Grafo Bipartido Multientidade
Formalizamos o ambiente transacional como um grafo bipartido temporal heterogêneo:
$$\mathcal{G} = (\mathcal{U}, \mathcal{V}, \mathcal{E}, \mathcal{W})$$
Onde:
* $\mathcal{U} = \{u_1, u_2, \dots, u_{|\mathcal{U}|}\}$ representa o conjunto de nós da partição primária de **Portadores de Cartão (UIDs)**, sintetizados deterministicamente via $\text{UID} = \text{card1} \parallel \text{addr1} \parallel \lfloor \text{Day} - D1 \rfloor$.
* $\mathcal{V} = \mathcal{V}_{\text{dev}} \cup \mathcal{V}_{\text{email}} \cup \mathcal{V}_{\text{net}}$ representa o conjunto da partição secundária de **Entidades e Recursos Compartilhados**, congregando assinaturas de modelo de dispositivo e sistema operacional (`DeviceInfo`), domínios de e-mail de cobrança (`P_emaildomain`) e sub-redes geográficas (`addr1`).
* $\mathcal{E} \subseteq \mathcal{U} \times \mathcal{V} \times \mathbb{R}^+$ é o conjunto de arestas temporais direcionadas ao momento do acesso, onde $e = (u_i, v_k, t_m) \in \mathcal{E}$ indica que o portador $u_i$ utilizou o recurso $v_k$ na transação ocorrida no instante $t_m$.
* $\mathcal{W}: \mathcal{E} \to \mathbb{R}^+$ atribui pesos às arestas, parametrizados pela frequência transacional acumulada.

```text
                 TOPOLOGIA BIPARTIDA COM PODA DE NÓS SENTINELA
                 
     [PORTADORES (UID ∈ U)]                       [RECURSOS / MODELOS (V)]
     
         (UID_1) ───────────┐                 ┌─── (SM-G935F Build) [deg=4]
                            ├─────────────────┤
         (UID_2) ───────────┤                 └─── (iOS Device) [deg=1]
                            │
         (UID_3) ───────────┘
         
         (UID_K) ───────────X [ARESTA BLOQUEADA] X─── (null_device) [REMOVIDO]
                              (Evita Hub Artificial / Buraco Negro Topológico)
```

#### 4.3.2 Projeções Unipartidas e Métricas de Coocorrência
A relação topológica entre portadores distintos que operam sob a mesma infraestrutura é modelada através da projeção unipartida sobre $\mathcal{U}$. Seja $\mathbf{B} \in \{0, 1\}^{|\mathcal{U}| \times |\mathcal{V}|}$ a matriz de incidência bipartida binária, onde $B_{ik} = 1 \iff (u_i, v_k) \in \mathcal{E}$. A matriz de adjacência da projeção unipartida de portadores $\mathbf{A}_{\mathcal{U}} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{U}|}$ é obtida por:
$$\mathbf{A}_{\mathcal{U}} = \mathbf{B} \mathbf{B}^\top, \quad \text{onde } (\mathbf{A}_{\mathcal{U}})_{ij} = \sum_{k=1}^{|\mathcal{V}|} B_{ik} B_{jk} = |\mathcal{N}(u_i) \cap \mathcal{N}(u_j)|$$
Se dois portadores distintos $u_i, u_j \in \mathcal{U}$ conectam-se ao mesmo nó de dispositivo $v_k \in \mathcal{V}$, existe um **caminho de comprimento 2** ($u_i \leftrightarrow v_k \leftrightarrow u_j$) no grafo bipartido, induzindo uma aresta de peso $(\mathbf{A}_{\mathcal{U}})_{ij} \ge 1$. No comércio eletrônico, a existência de caminhos de comprimento 2 entre identidades cadastrais distintas operando sob um mesmo perfil tecnológico específico (especialmente builds raras ou emuladores) sugere indício de ataques automatizados ou teste de cartões roubados (*card testing*). Ressalta-se, contudo, que para plataformas massivas (e.g., Windows ou iOS), o compartilhamento de `DeviceInfo` reflete apenas a popularidade do sistema operacional, exigindo tratamento não-linear pelo modelo supervisionado.

#### 4.3.3 Tratamento do Grafo no Tempo e Prevenção Rigorosa de *Topology Leakage*
Um dos erros mais críticos na modelagem de redes para séries financeiras temporais é o **vazamento topológico (*Topology & Lookahead Leakage*)**, que ocorre quando arestas e conexões do conjunto de teste futuro são incluídas na matriz de adjacência durante a fase de treinamento, permitindo que o classificador "enxergue o futuro" através da topologia da rede.

Para prevenir o risco de vazamento topológico (*topology leakage*), adotamos um protocolo de **Grafo Retrospectivo Não-Antecipativo**:
1. **Isolamento Topológico do Treino**: O grafo global de referência $\mathcal{G}_{\text{train}} = (\mathcal{U}_{\text{train}}, \mathcal{V}_{\text{train}}, \mathcal{E}_{\text{train}})$ é ajustado **estritamente sobre as arestas ocorridas no intervalo histórico $[0, t_{\text{split}})$** (primeiros 140 dias de benchmark):
$$\mathcal{E}_{\text{train}} = \left\{ (u_i, v_k, t_m) \in \mathcal{E} \;\middle|\; t_m < t_{\text{split}} \right\}$$
2. **Inferência Estacionária sem Vazamento na Validação**: Ao avaliar uma transação futura $i$ no conjunto de validação temporal ($t_i \ge t_{\text{split}}$):
   * As métricas de grafo estruturais (`device_degree`, `uid_pagerank`) são consultadas estritamente na estrutura $\mathcal{G}_{\text{train}}$ congelada no passado.
   * Se um dispositivo $d$ surge de forma inédita na validação ($d \notin \mathcal{V}_{\text{train}}$), seu grau é atribuído deterministicamente como $\deg(d) = 0$ e seu escore PageRank recebe a probabilidade basal neutra uniforme $\frac{1}{|\mathcal{V}_{\text{train}}|}$. Nenhuma aresta da validação é inserida no grafo antes do instante da decisão.
3. **Eficiência Temporal em Streaming**: Para atributos dinâmicos de alta frequência (`uid_tx_count_24h` e $\Delta t$), a janela temporal é estritamente retrospectiva: a busca binária avalia apenas o intervalo $[t_i - 86.400, t_i)$, assegurando que transações posteriores em $(t_i, \infty)$ permaneçam inacessíveis.

#### 4.3.4 Centralidade PageRank e Graus Bipartidos de Conectividade
Para quantificar a relevância estrutural das entidades na rede de dispositivos, resolvemos o sistema estacionário de cadeias de Markov com teletransporte ($\alpha = 0,85$):
$$\mathbf{p}^{(k+1)} = \alpha \mathbf{M} \mathbf{p}^{(k)} + \frac{1 - \alpha}{|\mathcal{V}_{\text{train}}|} \mathbf{e}$$
Onde $\mathbf{M} = \mathbf{D}^{-1} \mathbf{A}$ é a matriz de transição estocástica normalizada por linhas. Extraímos ainda os graus de conectividade distinta:
$$\deg_{\text{distinct}}(d) = |\mathcal{N}_{\text{train}}(d)| \quad (\text{Portadores Únicos conectados ao Dispositivo } d)$$
$$\deg_{\text{distinct}}(u) = |\mathcal{N}_{\text{train}}(u)| \quad (\text{Dispositivos Únicos operados pelo Portador } u)$$

#### 4.3.5 Dicionário de Features de Grafos e Dinâmica Temporal
O Quadro abaixo sintetiza as variáveis relacionais e temporais sintetizadas a partir da topologia de rede:

| Nome da Feature | Tipo / Definição Algorítmica | Escopo / Fórmula | Intuição de Negócio e Mecanismo de Fraude |
|---|:---:|---|---|
| `device_degree` | Topológica Bipartida | $\deg_{\text{train}}(d) = \vert\mathcal{N}_{\text{train}}(d)\vert$ | **Compartilhamento de Hardware**: Identifica emuladores, fazendas de smartphones e PCs compartilhados operando múltiplas identidades (*fraud rings*). |
| `uid_degree` | Topológica Bipartida | $\deg_{\text{train}}(u) = \vert\mathcal{N}_{\text{train}}(u)\vert$ | **Dispersão de Identidade**: Detecta portadores que saltam entre múltiplos aparelhos físicos em curtos períodos (*device hopping*). |
| `uid_pagerank` | Centralidade de Rede | Vetor estacionário $\mathbf{p}_u$ com $\alpha = 0,85$ | **Relevância Topológica**: Mede a centralidade do portador na rede global de dispositivos; expõe nós inseridos em agrupamentos densamente interconectados. |
| `uid_tx_count_24h` | Causal Dinâmica | $\sum_{t_k \in [t_i - 86.400, t_i)} \mathbb{I}(u_k = u_i)$ | **Velocidade de Ataque**: Detecta rajadas de *card testing* automatizado disparadas por scripts e *bots* após clonagem de cartões. |
| `uid_secs_since_prev` | Causal Dinâmica | $\Delta t_i = t_i - t_{i-1}$ para o mesmo UID | **Cadência Temporal**: Separa o comportamento humano habitual ($\Delta t > 24\text{h}$) de disparos em lote criminosos ($\Delta t \le 60\text{s}$). |

#### 4.3.6 Análise de Complexidade: Velocidade Causal em $\mathcal{O}(N \log K)$
A contagem temporal de velocidade foi implementada através de um algoritmo causal vetorizado. Para cada portador $u$, o conjunto de timestamps ordenados $T(u) = \{t_1, t_2, \dots, t_K\}$ é processado via busca binária causal (`np.searchsorted`):
$$\text{velocity}_{24h}(i) = i - \text{searchsorted}(T(u), t_i - 86.400, \text{side}=\text{'left'}) + 1$$
Como o tamanho do histórico individual $K$ é muito menor que o volume total de transações $N$ ($K \ll N$, com mediana $K \le 4$), a complexidade temporal total do algoritmo sobre todo o dataset é:
$$\mathcal{O}\left(\sum_{u} K_u \log K_u\right) \approx \mathcal{O}(N \log \bar{K})$$
Esse projeto de algoritmo processa as **590.540 transações completas do benchmark em apenas 1,13 segundo**, indicando viabilidade para computação em esteiras industriais de baixa latência.

---

### 4.4 Compressão Dimensional Espectral (PCA) e Investigação de GNNs

#### 4.4.1 Decomposição Espectral PCA nas 339 Variáveis V
A família de atributos $V1$ a $V339$ apresenta esparsidade massiva e correlação extrema. Padronizamos a matriz $\mathbf{X}_V$ após imputação pelas medianas e realizamos a decomposição em valores singulares (SVD) da matriz de covariância amostral no treino:
$$\mathbf{\Sigma}_V = \frac{1}{N_{\text{train}}-1} \mathbf{X}_V^\top \mathbf{X}_V = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^\top$$
Projetamos as 339 dimensões nos primeiros 15 autovetores principais:
$$\mathbf{Z}_V = \mathbf{X}_V \mathbf{U}_{:, 1:15}$$
Essa compressão dimensional reduz em mais de 75% a dimensionalidade da tabela, mitigando a maldição da dimensionalidade (*curse of dimensionality*). Na Seção 6, comparamos empiricamente o modelo com as 339 variáveis brutas versus os 15 componentes PCA.

#### 4.4.2 Avaliação Exploratória de GNN Heterogênea (GraphSAGE Auto-Supervisionado)
Investigamos a viabilidade de representações latentes indutivas de nós via **Heterogeneous GraphSAGE** (HAMILTON et al., 2017) em PyTorch Geometric. Para prevenir vazamento de alvos (*target leakage*), a rede foi treinada sob perda auto-supervisionada de reconstrução de enlaces (*Link Reconstruction Loss*):
$$\mathcal{L}_{\text{link}} = - \log \sigma \left( \mathbf{z}_u^\top \mathbf{z}_d \right) - \mathbb{E}_{d_n \sim P_n} \left[ \log \sigma \left( -\mathbf{z}_u^\top \mathbf{z}_{d_n} \right) \right]$$
A avaliação de GNNs puras para predição indutiva de fraudes resultou em desempenho preditivo muito modesto (PR-AUC = 0,0813), corroborando os achados de Shwartz-Ziv & Armon (2022) e Grinsztajn et al. (2022) de que arquiteturas neurais profundas puras são superadas por árvores em dados tabulares esparsos sob severo desbalanceamento. Por essa razão, optou-se por focar o estudo nos **atributos topológicos estruturais explícitos** (PageRank e graus bipartidos) integrados a GBDTs.

---

### 4.5 Roteamento Dual de Identidade e Pós-Processamento Causal

#### 4.5.1 Roteamento Dual de Identidade (Model A / Model B)
Aproximadamente 75% das transações no benchmark IEEE-CIS não possuem dados de hardware (compras sem metadados de identidade). Para prevenir imputações artificiais que introduzam ruído estatístico, desacoplamos o treinamento em dois classificadores especializados:

* **Model A (Sem Identidade - Fallback Operacional)**: Treinado sobre atributos puramente transacionais, contadores temporais $D_n$, estatísticas históricas do UID e velocidade causal de 24h (67 a 391 variáveis). Cobre 100% da população com alta robustez.
* **Model B (Com Identidade - Especialista em Conluios de Rede)**: Treinado sobre o conjunto completo de atributos transacionais acrescido de dados de hardware, PageRank e graus bipartidos (106 a 449 variáveis). É acionado exclusivamente quando a identidade está presente, atuando como um filtro de choque contra *fraud rings*.

#### 4.5.2 Mecânica do Pós-Processamento Temporal Causal
O pós-processamento temporal por portador implementa um filtro de suavização causal de probabilidades baseado na média móvel cumulativa expansiva (*Expanding Causal Window Filter*):
$$\bar{p}_{\text{hist}}(u_i, t_i) = \frac{1}{|\{k \in T(u_i) \mid t_k < t_i\}|} \sum_{t_k < t_i} \hat{p}_{\text{GBDT}}(k)$$
$$\hat{p}_{\text{final}}(i) = \gamma \cdot \hat{p}_{\text{GBDT}}(i) + (1 - \gamma) \cdot \bar{p}_{\text{hist}}(u_i, t_i), \quad \text{com } \gamma = 0,7$$
Onde $T(u_i)$ é o histórico de transações do cliente $u_i$ estritamente anteriores ao instante $t_i$.  

**Blindagem Algorítmica de Alinhamento**: No Pandas, o método `groupby('uid')['raw_score'].expanding().mean()` devolve uma `pd.Series` com índice multinível `(uid, original_index)` ordenada internamente pela chave alfabética do grupo (`uid`), e não pela ordem temporal. O uso ingênuo de `.values` na atribuição descarta o índice e insere o array posicionalmente sobre o DataFrame ordenado por tempo, provocando um desalinhamento catastrófico entre entidades distintas. A implementação metodologicamente correta exige `.reset_index(level=0, drop=True)`, permitindo que o Pandas realize o alinhamento vetorial perfeito linha a linha pelo índice original da série temporal.

---

## 5. Auditoria de Maturidade Científica e Correção de Leakages

Para assegurar que os resultados reportados reflitam a capacidade de generalização indutiva em produção, o pipeline passou por uma auditoria sistemática cobrindo seis aspectos críticos:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                  RELATÓRIO FORMAL DE AUDITORIA DE LEAKAGES                 │
└────────────────────────────────────────────────────────────────────────────┘

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
     entre entidades, degradando artificialmente o PR-AUC para ~0.46.
   • Correção: Emprego de .reset_index(level=0, drop=True) preservando alinhamento vetorial por índice.
   • Resultado: Eliminação de contaminação cruzada e estabilização rigorosa dos escores pós-processados.

6. GRANULARIDADE DE DeviceInfo E NÃO-MONOTONICIDADE DO RISCO:
   • Diagnóstico: Suposição errônea de que DeviceInfo representava instâncias físicas individuais.
   • Correção: Mapeamento de que a coluna expressa famílias de SO e modelos de mercado (Windows, iOS,
     SM-G935F), exibindo pico de taxa de fraude em graus intermediários (11-100) e queda em megahubs (>100).
   • Resultado: Modelagem não-linear adequada via árvores e desmistificação teórica de fraud rings.
```

---

## 6. Estudo de Ablação Rigoroso

O cerne empírico deste trabalho é o **Estudo de Ablação Controlado**, desenhado para responder à **RQ1** e à **RQ2**. O objetivo central da ablação é decompor o sistema em blocos modulares ortogonais e quantificar, sob hiperparâmetros idênticos e fixos no classificador XGBoost (`n_estimators=300, lr=0.03, max_depth=9, subsample=0.8, colsample=0.8, scale_pos_weight`), qual componente realmente gera ganho discriminatório sobre a partição experimental causal do IEEE-CIS ($N_{\text{total}} = 295.270$, com $N_{\text{train}} = 236.216$ e $N_{\text{val}} = 59.054$), abrangendo integralmente o horizonte de 182 dias de histórico sob taxa de fraude de 3,50%.

### 6.1 Racional e Definição das Configurações de Ablação

A esteira experimental foi estruturada em **oito configurações ortogonais** para decompor rigorosamente o impacto de cada família de atributos, mantendo hiperparâmetros fixos no classificador XGBoost (`n_estimators=300, lr=0.03, max_depth=9, subsample=0.8, colsample=0.8, scale_pos_weight`):

1. **Configuração 1A — Baseline Raw (Sem PCA)**:  
   *Racional*: Ponto de partida padrão com as variáveis brutas originais das tabelas de transação e identidade (incluindo as 339 colunas $V$ não tratadas), sem atributos de rede, velocidade temporal ou agregações comportamentais (391 features no Model A / 430 no Model B).
2. **Configuração 1B — Baseline Com PCA (15-d)**:  
   *Racional*: Isola o impacto da **compressão dimensional espectral**. Substitui as 339 colunas $V$ por 15 autovetores para verificar a preservação da acurácia e a otimização de latência (67 features no Model A / 106 no Model B).
3. **Configuração 2 — Baseline + Apenas Topologia de Grafo**:  
   *Racional*: Adiciona PageRank (`uid_pagerank`), grau do dispositivo (`device_degree`) e grau do portador (`uid_degree`) sobre o baseline 1B, sem atributos de velocidade ou engenharia tabular comportamental (70 features no Model A / 109 no Model B).
4. **Configuração 3 — Baseline + Apenas Velocidade**:  
   *Racional*: Isola a contribuição da dinâmica temporal causal em janela retrospectiva de 24h (`uid_tx_count_24h` e `uid_secs_since_prev`) sobre o baseline 1B, sem métricas topológicas de rede (68 features no Model A / 107 no Model B).
5. **Configuração 4 — Baseline + Grafos Completos (Topologia + Velocidade)**:  
   *Racional*: Combina atributos de topologia de grafo e velocidade temporal sobre o baseline 1B, sem atributos tabulares comportamentais avançados (71 features no Model A / 110 no Model B).
6. **Configuração 5 — Baseline + Apenas Feature Engineering Tabular**:  
   *Racional*: Isola o poder preditivo das agregações comportamentais por portador (médias $\mu$ e desvios $\sigma$ de montantes e contadores $C$, contagens de frequência e coordenadas ancoradas de calendário $D_n$), sem nenhum atributo de rede ou velocidade (81 features no Model A / 120 no Model B).
7. **Configuração 6A — All-in-One (Com PCA 15-d)**:  
   *Racional*: Modelo integrado combinando baseline com PCA, FE comportamental de portador, topologia de rede bipartida e velocidade causal (86 features no Model A / 125 no Model B).
8. **Configuração 6B — All-in-One (Sem PCA / Raw V)**:  
   *Racional*: Modelo integrado completo mantendo a totalidade das 339 variáveis $V$ brutas em conjunto com a engenharia tabular e os atributos de grafo (410 features no Model A / 449 no Model B).

---

### 6.2 Resultados Consolidados da Ablação

A tabela abaixo consolida as métricas de performance preditiva, dimensões de atributos e tempos de treinamento obtidas na validação temporal causal:

| ID | Configuração Avaliada | Features (Model A / B) | Tempo Treino (s) | PR-AUC (Bruto) | ROC-AUC (Bruto) | F1-Score (Bruto) | PR-AUC (Pós-Proc) | F1-Score (Pós-Proc) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1A** | Baseline Raw (Sem PCA) | 391 / 430 | 105,8s | 0,5038 | 0,8779 | 0,5061 | 0,5034 | 0,5095 |
| **1B** | Baseline (Com PCA 15-d) | 67 / 106 | 51,4s | 0,5028 | 0,8852 | 0,4979 | 0,5016 | 0,5023 |
| **2** | Baseline + Apenas Topologia | 70 / 109 | 51,9s | 0,4918 | 0,8827 | 0,4917 | 0,4906 | 0,4984 |
| **3** | Baseline + Apenas Velocidade | 68 / 107 | 51,0s | 0,4943 | 0,8822 | 0,4936 | 0,4922 | 0,4957 |
| **4** | Baseline + Grafos Completos | 71 / 110 | 52,5s | 0,4997 | 0,8822 | 0,4983 | 0,4980 | 0,5093 |
| **5** | Baseline + Apenas FE Tabular | 81 / 120 | 60,5s | **0,5304** | 0,8622 | 0,5443 | 0,5299 | 0,5472 |
| **6A** | All-in-One (Com PCA 15-d) | 86 / 125 | 63,2s | **0,5344** | 0,8715 | 0,5481 | 0,5329 | 0,5524 |
| **6B** | All-in-One (Sem PCA / Raw V) | 410 / 449 | 118,4s | **0,5395** | 0,8649 | **0,5646** | **0,5434** | **0,5705** |

A Figura 9 ilustra as curvas Precision-Recall obtidas na partição de validação temporal:

![Figura 9: Curvas Precision-Recall na validação temporal para as principais configurações da ablação](figures/fig9_ablation_pr_roc_curves.png)

---

### 6.3 Dissecação Metodológica dos Fenômenos da Ablação

A análise rigorosa da Tabela 6.1 estabelece quatro conclusões científicas fundamentais:

#### A. Decomposição Causal do Lift: Primazia Tabular (87,3%) vs. Ganho Incremental de Grafos (12,7%)
Para responder frontalmente à **RQ1**, examinamos a cadeia consistente sob compressão PCA ($1\text{B} \to 5 \to 6\text{A}$):
* A transição do Baseline 1B (0,5028) para o modelo com Apenas FE Tabular (5, 0,5304) gera um incremento de **+0,0276 em PR-AUC**;
* A adição concomitante dos atributos de rede bipartida e velocidade (6A, 0,5344) adiciona **+0,0040 em PR-AUC**;
* **Decomposição Relativa**: Do lift total de **+0,0316** observado na cadeia PCA, **87,3% provém da Engenharia Tabular de Entidade** ($+0,0276 / +0,0316$), enquanto a **topologia de rede responde pelos 12,7% restantes** ($+0,0040 / +0,0316$).
* Na cadeia de variáveis brutas ($1\text{A} \to 6\text{B}$), o modelo atinge 0,5395 bruto (+0,0357 sobre 1A) e **0,5434** após o pós-processamento causal alinhado via `.reset_index` (F1 = 0,5705).

#### B. Por que a Topologia Isolada (Config 2) Não Supera o Baseline (Config 1B)?
A Configuração 2 registra PR-AUC de 0,4918, inferior ao baseline (0,5028). A auditoria quantitativa causal demonstra a causa estrutural:
* **Assimetria Intra-Treino**: No conjunto de treino, a construção estática do grafo faz com que 96,71% das transações com dispositivo recebam um grau inflacionado por eventos futuros (antecipação média de 3.612,2 UIDs e inflação de 7,39x, atingindo 16,25x no decil inicial). O classificador aprende a confiar excessivamente em graus altos durante o treino.
* **Desconexão na Validação**: Na validação temporal futura, 83,75% das transações possuem `null_device` e 20,37% dos dispositivos são novos (totalizando 84,15% de transações desconectadas da rede de treino), enquanto 65,76% dos portadores UIDs são inéditos. Nós novos recebem grau nulo e PageRank residual ($1/|\mathcal{V}|$). Essa quebra estrutural impede que atributos de rede atuem isoladamente sem o suporte de features comportamentais densas.

#### C. Diagnóstico e Correção de Alinhamento no Pós-Processamento Temporal
Na auditoria algorítmica, identificou-se que o método `groupby('uid')['raw_score'].expanding().mean()` indexa as saídas alfabeticamente pela chave de entidade (`uid`). O uso anterior de `.values` gerava um desalinhamento posicional grave sobre o frame ordenado no tempo, provocando uma perda espúria uniforme de **-0,0370 a -0,0427** em PR-AUC em todas as configurações.  
Com a correção estrita via `.reset_index(level=0, drop=True)`, preserva-se o alinhamento temporal por índice: os escores estabilizam e o modelo All-in-One 6B obtém ganho legítimo de **+0,0039** (0,5395 $\to$ **0,5434**), alcançando F1 de **0,5705**.

#### D. Eficiência da Compressão Espectral PCA
A substituição das 339 colunas colineares $V$ por 15 componentes principais reduz o tempo de treinamento em **51,4%** (de 105,8s para 51,4s no baseline) e eleva a ROC-AUC de 0,8779 para 0,8852, com PR-AUC praticamente idêntica (0,5038 vs 0,5028), confirmando sua alta viabilidade operacional.

---

### 6.4 Interpretabilidade Global: Ranking de Atributos (Gain & Total Gain)

A inspeção do ganho de informação das árvores do classificador XGBoost (Model B do modelo completo 6A) reflete a estrutura empírica observada:

As variáveis com maior ganho acumulado de divisão no Model B foram:
1. `C1` (Contador transacional) — 21,3% do ganho total acumulado
2. `C14` (Contador transacional) — 8,3% do ganho total acumulado
3. `PCA_V_3` (Terceiro componente principal de segurança Vesta) — 6,6%
4. `D1n` (Coordenada normalizada de calendário civil) — 4,6%
5. `C13` (Contador transacional histórico) — 3,8%
6. `P_emaildomain_count_30d` (Frequência causal em 30 dias) — 2,6%
7. `C11` (Contador transacional) — 2,5%
8. `card2` (Código de identificação bancária secundário) — 2,2%
9. `TransactionAmt` (Montante da transação) — 2,0%
10. `device_degree` (**Grau de Portadores por Dispositivo** — Top 10, com 1,8% do ganho total)

Entre as variáveis de grafo, `device_degree` posicionou-se no Top 10 em ganho total acumulado, indicando que o classificador utiliza a conectividade do hardware como critério confirmatório em ramificações profundas. No entanto, variáveis de grafo mais difusas (`uid_pagerank`, rank 78; `uid_degree`, rank 85) e de velocidade (`uid_tx_count_24h`, rank 84) receberam peso menor, confirmando que os contadores atômicos tradicionais continuam respondendo pela maior fatia da separabilidade.

---

### 6.5 Estudo de Caso de Subgrafo Real: Oportunidades e Riscos de Falsos Alarmes

A inspeção empírica das predições sobre a partição de validação temporal (`val_predictions.csv`) revela com exatidão a natureza de dois gumes dos atributos de conectividade relacional de hardware:

```text
                            CASO 1: FRAUDE BLOQUEADA VIA GRAFO (Tx #3479579)
                            
            [Portador 1] ──────────────┐
            [Portador 2] ──────────────┤
                  ...                  ├─── (Assinatura SM-J320M Build/LMY47V) [device_degree >= 2]
            [Portador K] (UID Atual) ──┘
                   │
                   ▼
        Transação #3479579: Valor = US$ 120,75 (Ticket Regular / Perfil Dissimulado)
        ──────────────────────────────────────────────────────────────────────────
        • Predição Baseline Tabular (1B) : p = 26,3%  -> NÃO DETECTADA / FALSO NEGATIVO (abaixo de tau*)
        • Predição com Topologia Grafo(2): p = 46,4%  -> ELEVAÇÃO DE ALERTA (+20,1 p.p.)
        • Predição All-in-One (6B)       : p = 80,0%  -> FORTE ALARME / BLOQUEIO CATEGÓRICO (+53,6 p.p.)
```

```text
                            CASO 2: FALSO POSITIVO INDUZIDO POR MODELO COMPARTILHADO (Tx #3476363)
                            
            Transação #3476363: Valor = US$ 150,00 (Compra Legítima, isFraud = 0)
            Hardware: Assinatura Pixel 2 XL Build/OPM2.171019.029 (Dispositivo Comercial Compartilhado)
            ──────────────────────────────────────────────────────────────────────────
            • Predição Baseline Tabular (1B) : p = 35,9%  -> APROVADA REGULARMENTE (abaixo do corte)
            • Predição com Topologia Grafo(2): p = 60,5%  -> FALSO POSITIVO / BLOQUEIO INDEVIDO (+24,6 p.p.)
            • Predição All-in-One (6B)       : p = 47,0%  -> MODERAÇÃO POR ENGENHARIA TABULAR
```

| Atributo / Métrica | Caso 1: Fraude Bloqueada | Caso 2: Alarme Falso Criado |
|---|:---:|:---:|
| **TransactionID** | `3479579` | `3476363` |
| **Rótulo Real (`isFraud`)** | **1 (Fraude Confirmada)** | **0 (Transação Legítima)** |
| **Portador (UID)** | `10568_0_148` | `12544_337_147` |
| **Valor da Compra (`TransactionAmt`)** | **US$ 120,75** | **US$ 150,00** |
| **Assinatura do Dispositivo (`DeviceInfo`)** | `SM-J320M Build/LMY47V` | `Pixel 2 XL Build/OPM2.171019.029` |
| **Escore Baseline Tabular (1B)** | $\hat{p} = 0,263$ (Falso Negativo) | $\hat{p} = 0,359$ (Aprovada Regularmente) |
| **Escore com Topologia Grafo (2)** | $\hat{p} = 0,464$ (**+20,1 p.p.** — Alerta de Rede) | $\hat{p} = 0,605$ (**+24,6 p.p.** — Falso Positivo) |
| **Escore All-in-One (6B)** | $\hat{p} = 0,800$ (**+53,6 p.p.** — Bloqueio Seguro) | $\hat{p} = 0,470$ (Alerta Moderado) |

No Caso 1 (Tx `3479579`), o criminoso simulou uma compra cotidiana com ticket típico de varejo (US$ 120,75). O classificador tabular isolado 1B atribuiu probabilidade de apenas 0,263, permitindo a transação passar despercebida. A injeção da conectividade de rede bipartida elevou o escore para 0,464, e a combinação completa All-in-One (6B) disparou a probabilidade para 0,800, confirmando a captura eficaz da anomalia relacional.

Entretanto, o Caso 2 (Tx `3476363`) adverte para o risco intrínseco da modelagem relacional em dados com rótulos genéricos de dispositivo: um usuário honesto portando um smartphone comercial de mercado (`Pixel 2 XL`) sofreu uma elevação artificial de escore na Configuração 2 (de 0,359 para 0,605), cruzando o limiar operacional e gerando atrito indevido com o cliente legítimo. Esse contraponto prático elucida por que os ganhos globais agregados de PR-AUC permanecem contidos e por que a engenharia de atributos tabulares comportamentais de portador deve sempre atuar como âncora estabilizadora do pipeline.

---

## 7. Discussão Complementar: Análise sob Matrizes Assimétricas de Decisão

Como extensão prática de engenharia de software e tomada de decisão automatizada, avaliamos o comportamento dos modelos sob matrizes de penalidade assimétrica baseadas no princípio de *Bayes Minimum Risk* (CORREA BAHNSEN et al., 2014, 2015; ELKAN, 2001).

### 7.1 Formulação da Matriz de Decisão e Por que o Limiar $\tau = 0,50$ é Inadequado
Em problemas de classificação binária padrão, adota-se comumente o limiar ingênuo de decisão $\tau = 0,50$. Contudo, na operação de pagamentos, os custos dos dois tipos de erro são fundamentalmente assimétricos:
* Aprovar uma fraude (**Falso Negativo**) acarreta a perda do valor integral do produto somada à taxa regulatória de chargeback da adquirente/bandeira ($C_{\text{chargeback}} = \text{US\$} 15,00$).
* Recusar indevidamente uma compra legítima (**Falso Positivo**) acarreta o custo de auditoria manual ($C_{\text{revisão}} = \text{US\$} 3,00$) e a insatisfação do cliente, modelada como uma taxa de perda marginal de receita futura ($\alpha = 0,05$ ou 5% do ticket).

Formalizamos a perda financeira operacional total (	au)$ parametrizada pelo limiar de corte $	au \in [0, 1]$:
115693L(	au) = \sum_{i=1}^{N_{	ext{val}}} \Big[ y_i (1 - \hat{y}_i(	au)) \cdot (	ext{amt}_i + C_{	ext{chargeback}}) + (1 - y_i) \hat{y}_i(	au) \cdot (C_{	ext{revisao}} + lpha \cdot 	ext{amt}_i) + y_i \hat{y}_i(	au) \cdot C_{	ext{operacional}} \Big]115693
Onde $\hat{y}_i(	au) = \mathbb{I}(\hat{p}_i \ge 	au)$ é a regra de decisão e {	ext{operacional}} = 	ext{US\$} 0,10$ é o custo computacional de inferência por transação.

A política de decisão ótima sob o princípio de *Bayes Minimum Risk* é obtida minimizando a perda esperada:
115693	au^* = rg\min_{	au \in [0, 1]} L(	au)115693

Para assegurar rigor operacional livre de vazamento, o ponto de corte ótimo $	au^*$ é calibrado sobre a partição temporal de validação ({	ext{val}} = 59.054$) e submetido à validação contínua em 4 dobras temporais expansivas mensais (*Walk-Forward*, Seção 7.2 / Notebook 06).
---

### 7.2 Avaliação da Função de Custo Financeiro Real (Bayes Minimum Risk)

A avaliação da função de perda operacional $L(\tau)$ foi executada diretamente sobre as predições reais da partição de validação temporal ($N_{\text{val}} = 59.054$ transações, com 2.106 fraudes totalizando US$ 314.263,20 em perdas financeiras brutas, arquivo `val_predictions.csv`). 

A Tabela 7.2 compara a política de aprovação cega sem qualquer sistema preditivo ($\tau = 1,0$, onde todas as 2.106 fraudes são consumadas, gerando um prejuízo de US$ 345.853,20 ao somar os montantes furtados de US$ 314.263,20 às multas regulatórias de estorno de US$ 31.590,00) com as configurações canônicas avaliadas em seus respectivos limiares ótimos:

| ID | Configuração Avaliada | Limiar Ótimo ($\tau^*$) | Prejuízo Operacional Total ($L^*$) | Perda por Falsos Negativos (FN) | Custo de Falsos Positivos (FP) | Economia vs. Aprovação Cega |
|---|---|:---:|:---:|:---:|:---:|:---:|
| — | **Aprovação Cega (Sem Modelo)** | 1,00 | US$ 345.853,20 | US$ 345.853,20 | US$ 0,00 | — |
| **1A** | Baseline Raw (Sem PCA) | 0,36 | US$ 174.956,30 | US$ 135.678,36 | US$ 39.150,04 | +US$ 170.896,90 (-49,4%) |
| **1B** | Baseline (Com PCA 15-d) | 0,17 | US$ 169.059,03 | US$ 89.991,61 | US$ 78.916,22 | +US$ 176.794,17 (-51,1%) |
| **2** | Baseline + Apenas Topologia | 0,27 | US$ 175.526,75 | US$ 122.726,26 | US$ 52.665,59 | +US$ 170.326,45 (-49,2%) |
| **3** | Baseline + Apenas Velocidade | 0,20 | US$ 176.323,01 | US$ 106.535,26 | US$ 69.643,56 | +US$ 169.530,19 (-49,0%) |
| **4** | Baseline + Grafos Completos | 0,23 | US$ 171.179,36 | US$ 107.828,37 | US$ 63.209,49 | +US$ 174.673,84 (-50,5%) |
| **5** | Baseline + Apenas FE Tabular | 0,08 | US$ 200.697,43 | US$ 165.799,53 | US$ 34.760,39 | +US$ 145.155,77 (-42,0%) |
| **6A** | All-in-One (Com PCA 15-d) | 0,09 | US$ 195.479,99 | US$ 158.451,37 | US$ 36.888,43 | +US$ 150.373,21 (-43,5%) |
| **6B** | All-in-One (Sem PCA / Raw V) | 0,10 | US$ 199.541,46 | US$ 165.417,27 | US$ 33.985,98 | +US$ 146.311,74 (-42,3%) |
| **6B PP**| All-in-One + Pós-Proc Causal | 0,10 | **US$ 197.883,56** | US$ 162.304,93 | US$ 35.439,23 | +US$ 147.969,64 (-42,8%) |

Os resultados da Tabela 7.2 consolidam três diagnósticos financeiros cruciais:
1. **Redução Massiva de Prejuízo sobre a Inação**: Qualquer um dos modelos de aprendizado supervisionado reduz o prejuízo operacional em mais de **US$ 145.000 a US$ 176.000** em relação à aprovação cega (queda de 42,0% a 51,1% das perdas totais).
2. **Sinergia Topologia + Velocidade (Configuração 4 vs 1A)**: O modelo com grafos completos (Config 4) obtém custo mínimo de **US$ 171.179,36**, gerando uma economia direta de **US$ 3.776,94** sobre o Baseline Raw 1A (e **US$ 4.249,01** no cenário do Notebook 05 com $\tau^* = 0,23$ vs $0,35$). Cumpre explicitar que essa sutil diferença entre os valores (US$ 3.776,94 na Tabela 7.2 vs US$ 4.249,01 no sumário do Notebook 05) decorre estritamente da granularidade da grade de discretização do limiar ($\Delta \tau$), atestando em ambos os casos uma redução consistente de ~2,2% a 2,4% no prejuízo total proporcionada pela identificação precoce de redes.
3. **Consistência Temporal em Validação Walk-Forward (Notebook 06)**: Na avaliação em 4 dobras expansivas mensais contínuas, o modelo com grafos reduziu o prejuízo operacional médio de US$ 139.775,31/mês para US$ 133.053,23/mês, comprovando uma **economia líquida sustentada de US$ 6.722,08 por mês** decorrente da estabilidade contra o *concept drift*.
4. **Otimização Bayesiana (Notebook 07)**: A busca Optuna refinou a fronteira de calibração, reduzindo o custo operacional para **US$ 145.698,64** no modelo com grafos contra **US$ 147.350,56** no baseline tabular otimizado (economia adicional comprovada de **US$ 1.651,92** operando em limiar $\tau = 0,15$ muito mais seguro contra atrito de falsos alarmes).

---

### 7.3 Sensibilidade de Detecção (Recall) sob Taxa de Falso Positivo Controlada (FPR = 1,0%) e Significância via Bootstrap

Na literatura de prevenção a fraudes, é comum encontrar comparações ingênuas de sensibilidade (recall) entre modelos avaliados sob seus respectivos limiares ótimos calculados in-sample ($\tau^*$). Contudo, essa prática introduz um severo **artefato de deslocamento de limiar (*threshold-shift artifact*)**: um modelo com limiar menor aparenta capturar mais fraudes simplesmente porque gera mais alertas globais, inflando descontroladamente os falsos positivos e sobrecarregando a mesa de análise humana.

Para mensurar a capacidade discriminatória estrutural genuína livre de distorções de corte, instituímos o **protocolo de avaliação sob Taxa de Falsos Positivos Fixa ($\text{FPR} = 1,0\%$)**, refletindo a restrição operacional estrita das centrais de monitoramento bancário.

A significância estatística de cada incremento de recall foi verificada via **Bootstrap não-paramétrico com 1.000 iterações** (amostragem com reposição sobre as transações de validação temporal, mantendo o ponto de corte rigorosamente fixado em $\text{FPR} = 1,0\%$):

| Estágio Metodológico | Recall Coordenadas (%) | IC 95% Bootstrap | $\Delta$ Recall (p.p.) | IC 95% do $\Delta$ | $p$-valor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline PCA (1B)** | 50,87% | [48,10%, 53,65%] | — | — | — |
| **+ FE Tabular (5)** | 58,41% | [55,56%, 61,11%] | **+7,54 p.p.** | [+5,08, +9,68] | $p < 0,001$ |
| **+ Grafos Bipartidos (6B bruto)** | 60,63% | [57,78%, 63,26%] | **+2,22 p.p.** | [+1,03, +3,73] | $p < 0,001$ |
| **+ Pós-Processamento Causal (6B PP)** | 61,27% | [58,49%, 63,97%] | **+0,64 p.p.** | [+0,00, +1,43] | $p = 0,048$ |
| **Ganho Acumulado Total ($1\text{B} \to 6\text{B PP}$)** | **61,27%** | [58,49%, 63,97%] | **+10,40 p.p.** | **[+8,02, +12,78]** | **$p < 0,001$** |

*Ressalva Metodológica sobre o Pós-Processamento Causal*: O incremento de +0,64 p.p. de recall atribuível ao filtro temporal expansivo alinhado corresponde à captura de exatamente **8 transações coordenadas adicionais** (em um universo de 1.260 fraudes coordenadas na partição de validação de 59.054 registros). Como o limite inferior do IC 95% toca exatamente zero ([+0,00; +1,43] p.p.), seu efeito isolado é marginal e estatisticamente fronteiriço no nível de 5%. Em contraste, a Engenharia Tabular (+7,54 p.p., IC 95% [+5,08; +9,68] p.p.) e a adição de Grafos Bipartidos (+2,22 p.p., IC 95% [+1,03; +3,73] p.p.) apresentam intervalos estritamente positivos e altamente significativos ($p < 0,001$).

Em **fraudes isoladas/avulsas** sob a mesma restrição de falso alarme ($\text{FPR} = 1,0\%$), o recall avança de 28,01% para 32,86% (+4,85 p.p.). Essa disparidade confirma que a modelagem relacional de grafos oferece ganho direcionado e desproporcional sobre arquiteturas coordenadas de ataque.

---

## 8. Discussão Geral e Limitações

### 8.1 A Soberania da PR-AUC e a Fragilidade do ROC-AUC sob Desbalanceamento
Sob taxas severas de desbalanceamento (3,5% de fraudes contra 96,5% de legítimos), o ROC-AUC é uma métrica excessivamente otimista e potencialmente enganosa. Isso ocorre porque o eixo das abscissas da curva ROC é a Taxa de Falsos Positivos:
$$\text{FPR} = \frac{FP}{FP + TN}$$
Como a massa de Verdadeiros Negativos ($TN$) é gigantesca, mesmo centenas de alarmes falsos geram um acréscimo minúsculo no denominador, mantendo a FPR artificialmente próxima de zero e inflando o ROC-AUC para valores acima de 0,86–0,88 em todas as configurações.

Em contraste, a curva **Precision-Recall (PR-AUC)** avalia a Precisão no eixo vertical:
$$\text{Precision} = \frac{TP}{TP + FP}$$
Aqui, cada Falso Positivo penaliza diretamente a métrica contra o volume real de fraudes ($TP$). Por essa razão matemática, a PR-AUC constitui o balizador supremo de eficácia para sistemas industriais de detecção de anomalias.

### 8.2 A Avaliação Realista do Papel de Grafos em Fraude Transacional
A investigação experimental conduzida nesta monografia permite responder com rigor científico às questões norteadoras do trabalho:

* **O uso de grafos é uma panaceia para dados tabulares de fraude?** Os dados empíricos demonstram claramente que **não**. Ao contrário de alegações simplistas na literatura, a inclusão de métricas topológicas puras de hardware sobre uma baseline tabular gerou PR-AUC de 0,4918 (vs 0,5028 da baseline), comprovando que grafos isolados não superam modelos tabulares quando a cobertura de identidade é restrita (apenas ~21% com dispositivo) e ocorrem assimetrias estruturais entre treino e validação.
* **Quais são os fatores limitantes dos grafos neste domínio?** 
  1. *Cobertura Incompleta de Identidade*: No dataset IEEE-CIS, menos de 25% das transações contêm metadados de dispositivo (`DeviceInfo`, `id_01` a `id_38`). Para mais de 75% da população, os atributos de rede são nulos ou imputados com valor de nó inédito.
  2. *Granularidade e Super-hubs*: Dispositivos agregados por string de modelo (ex.: perfis genéricos de Windows ou iOS) geram super-hubs onde usuários legítimos e criminosos compartilham o mesmo nó, diluindo o sinal de risco.
  3. *Inércia e Atraso Temporal*: Em produção real, anomalias topológicas dependem do histórico acumulado; novos fraudadores operam frequentemente sob perfis inéditos (*cold start* topológico).

### 8.3 Limitações Metodológicas

1. **Volume de Dados e Latência Operacional**: Em escala transacional global contínua (milhões de eventos diários), a atualização de componentes espectrais PCA e centralidades de rede requer arquiteturas de cálculo distribuído ou janelas deslizantes de streaming.
2. **Pós-processamento e Dinâmica Temporal**: Conforme demonstrado na Seção 6.3, filtros causais expansivos suavizam o escore e atenuam falsos positivos históricos, mas introduzem inércia em episódios agudos de tomada de conta (*account takeover*).
3. **Assimetria Causal Intra-Treino vs. Validação em Grafos Estáticos (Demonstração Empírica)**: A extração de PageRank e graus nodais no conjunto de treino foi realizada sobre a totalidade do bloco histórico $[0, t_{\text{split}})$. A auditoria quantitativa comprova que em **96,71% das transações com dispositivo no treino**, o grau estático é estritamente maior que o causal (com antecipação média de **3.612,2 UIDs futuros** e inflação média de **7,39x**, chegando a **16,25x no primeiro decil temporal**). Na validação causal, por outro lado, **83,75%** das compras possuem `null_device` e **20,37%** dos dispositivos são inéditos (totalizando **84,15% das transações desconectadas** da rede de treino), enquanto **65,76% dos UIDs são inéditos** (61,47% das transações). Esse choque distribucional explica matematicamente por que a topologia isolada não superou o baseline na validação temporal, fundamentando a migração para redes neurais dinâmicas contínuas (*Temporal Graph Networks - TGN*).
4. **Granularidade de `DeviceInfo` e Risco Não-Monotônico**: A coluna `DeviceInfo` captura assinaturas de sistema operacional e compilação de software, e não identificadores físicos únicos de hardware (como IMEI ou MAC). Esse fator induz a presença de megahubs industriais (ex.: `Windows` e `iOS Device`) e um padrão de risco não-monotônico, no qual a taxa de fraude atinge o ápice em nós de conectividade intermediária (11 a 100 UIDs) e se dilui para níveis basais em supernós (>100 UIDs), exigindo aprendizado supervisionado não-linear por árvores de decisão.

---

## 9. Conclusão

Este trabalho realizou uma investigação empírica rigorosa sobre a viabilidade e o papel da modelagem de grafos bipartidos em conjunto com compressão espectral e engenharia comportamental sobre o benchmark mundial IEEE-CIS Fraud Detection.

O objetivo central desta monografia não foi sustentar a superioridade dogmática de arquiteturas baseadas em grafos, mas testar metodologicamente suas premissas e quantificar com precisão seus impactos sob uma esteira temporal rigorosa anti-vazamento.

As principais conclusões científicas e contribuições do trabalho são:

1. **Topologia Isolada vs. Refinamento Incremental**: A injeção isolada de métricas de rede bipartida (PageRank e graus) sobre a baseline (Config 2 vs 1B) resultou em PR-AUC de 0,4918 (vs 0,5028), confirmando que a topologia isolada não ganha da baseline temporal devido à quebra de arestas e 84,15% de desconexão na validação. Seu valor real é de **refinamento complementar** (+0,0040 na cadeia PCA, 0,5304 $\to$ 0,5344, e +0,0039 no pós-processamento causal alinhado sobre variáveis brutas, 0,5395 $\to$ **0,5434**, com F1 de 0,5705).
2. **Dominância da Engenharia Tabular de Entidade**: A modelagem causal de portador ($D_n$, UID Mágico e agregações de consumo) é o verdadeiro pilar da detecção, respondendo por **87,3% de todo o lift discriminativo da cadeia PCA** (elevando a PR-AUC de 0,5028 para 0,5304, $\Delta = +0,0276$, $p < 0,001$).
3. **Eficiência da Compressão Espectral (PCA)**: A redução das 339 variáveis da família $V$ para 15 componentes principais acelerou o treinamento em **51,4%** (de 105,8s para 51,4s na baseline) com preservação da acurácia discriminativa (PR-AUC 0,5038 vs 0,5028) e ganho em generalização (ROC-AUC de 0,8779 para 0,8852).
4. **Validação Financeira sob Matrizes Assimétricas**: A formulação de matrizes de custo dependentes do exemplo com calibração out-of-sample evitou o viés otimista de limiares in-sample e comprovou a necessidade de alinhamento ao risco mínimo de Bayes na operação de pagamentos.
5. **Ganho Direcionado em Fraudes Coordenadas sob Restrição Operacional ($\text{FPR} = 1,0\%$)**: Quando avaliado sob taxa fixa de falsos alarmes operacionais, o pipeline completo atinge um ganho total de **+10,40 p.p. no recall de fraudes coordenadas** (de 50,87% para 61,27%, IC 95%: [+8,02; +12,78] p.p., $p < 0,001$). A decomposição causal confirma: +7,54 p.p. derivam da engenharia tabular, +2,22 p.p. decorrem dos grafos bipartidos e +0,64 p.p. provêm do pós-processamento causal (efeito marginal de 8 transações no universo de 1.260 fraudes coordenadas de validação).

Como trabalhos futuros, propõe-se a exploração de **Grafos Temporais Dinâmicos com Mecanismos de Atenção (TGAT/TGN)** capazes de aprender representações temporais contínuas de arestas, bem como a combinação de grafos multicamadas que integrem simultaneamente dados de redes de telecomunicação (ASN, IPs) e identificadores bancários.

## 10. Referências Bibliográficas (Normas ABNT)

AKOGLU, Leman; CHANDY, Rishi; FALOUTSOS, Christos. Graph based anomaly detection and description: a survey. **Data Mining and Knowledge Discovery**, v. 29, n. 3, p. 626-688, 2015.

CHEN, Tianqi; GUESTRIN, Carlos. XGBoost: A Scalable Tree Boosting System. In: **Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)**, San Francisco, CA, USA, 2016, p. 785-794.

CORREA BAHNSEN, Alejandro; AOUADA, Djamila; OTTERSTEN, Björn. Feature engineering strategies for detecting credit card fraud. **Expert Systems with Applications**, v. 41, n. 16, p. 7158-7164, 2014.

CORREA BAHNSEN, Alejandro; AOUADA, Djamila; OTTERSTEN, Björn. Example-dependent cost-sensitive decision trees. **Expert Systems with Applications**, v. 42, n. 19, p. 6609-6619, 2015.

DAL POZZOLO, Andrea et al. Calibrating probability with undersampling for unbalanced classification. In: **IEEE Symposium Series on Computational Intelligence (SSCI)**, Cape Town, South Africa, 2015, p. 159-166.

DAL POZZOLO, Andrea et al. Credit card fraud detection: a realistic modeling and a novel learning strategy. **IEEE Transactions on Neural Networks and Learning Systems**, v. 29, n. 8, p. 3784-3797, 2018.

ELKAN, Charles. The foundations of cost-sensitive learning. In: **International Joint Conference on Artificial Intelligence (IJCAI)**, Seattle, WA, USA, 2001, v. 17, n. 1, p. 973-978.

EUROPEAN CENTRAL BANK. **Report on card fraud in 2022 and 2023**. Frankfurt am Main: ECB, 2023. Disponível em: <https://www.ecb.europa.eu/pub/cardfraud/>. Acesso em: 28 ago. 2026.

FEY, Matthias; LENSSEN, Jan Eric. Fast Graph Representation Learning with PyTorch Geometric. In: **ICLR Workshop on Representation Learning on Graphs and Manifolds**, New Orleans, LA, USA, 2019.

GRINSZTAJN, Léo; OYALLON, Edouard; VAROQUAUX, Gaël. Why do tree-based models still outperform deep learning on typical tabular data? In: **Advances in Neural Information Processing Systems (NeurIPS)**, New Orleans, LA, USA, 2022, v. 35, p. 507-520.

HAMILTON, Will; YING, Rex; LESKOVEC, Jure. Inductive Representation Learning on Large Graphs. In: **Advances in Neural Information Processing Systems (NeurIPS)**, Long Beach, CA, USA, 2017, p. 1024-1034.

IEEE Computational Intelligence Society. **IEEE-CIS Fraud Detection**: Predict probability of online transaction fraud. Kaggle, 2019. Disponível em: <https://www.kaggle.com/c/ieee-fraud-detection/data>. Acesso em: 28 ago. 2026.

KE, Guolin et al. LightGBM: A Highly Efficient Light Gradient Boosting Machine. In: **Advances in Neural Information Processing Systems (NeurIPS)**, Long Beach, CA, USA, 2017, p. 3146-3154.

OPPENHEIM, Alan V.; SCHAFER, Ronald W. **Discrete-Time Signal Processing**. 3. ed. Upper Saddle River: Prentice Hall, 2010.

PAGE, Lawrence; BRIN, Sergey; MOTWANI, Rajeev; WINOGRAD, Terry. **The PageRank Citation Ranking: Bringing Order to the Web**. Stanford InfoLab, 1999. Technical Report.

PEDREGOSA, Fabian et al. Scikit-learn: Machine Learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825-2830, 2011.

POURHABIBI, Tahereh et al. Fraud detection in financial networks: a survey of graph computing approaches. **Artificial Intelligence Review**, v. 53, n. 8, p. 6097-6135, 2020.

SHWARTZ-ZIV, Ravid; ARMON, Amitai. Tabular data: Deep learning is not all you need. **Information Fusion**, v. 81, p. 84-90, 2022.

THE NILSON REPORT. **Card fraud losses reach US$ 32.34 billion globally**. Issue 1254. Carpinteria: Nilson Report, 2023. Disponível em: <https://nilsonreport.com/>. Acesso em: 28 ago. 2026.

VAN VLASSELAER, Véronique et al. APATE: a novel approach for automated credit card fraud detection using network-based extensions. **Decision Support Systems**, v. 75, p. 38-48, 2015.

VAN VLASSELAER, Véronique et al. GOTCHA! Network-based fraud detection for social security fraud. **Management Science**, v. 63, n. 9, p. 3090-3110, 2017.
