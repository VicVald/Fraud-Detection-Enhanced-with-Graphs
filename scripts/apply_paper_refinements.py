#!/usr/bin/env python3
import re

with open("LOCAL_COMPLETE_PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Diagram: Taxonomy in Section 3.8
old_taxonomy = """<pre>
                                  TAXONOMIA ESTRUTURAL DE TRANSAÇÕES
                                                   │
                  ┌────────────────────────────────┴────────────────────────────────┐
                  ▼                                                                 ▼
         Transações Legítimas                                            Transações Fraudulentas
         (22.784 transações)                                                (838 transações)
          Volume: $3.130.984                                                Volume: $121.568
                  │                                                                 │
                  │                                         ┌───────────────────────┴───────────────────────┐
                  ▼                                         ▼                                               ▼
         Comportamento Padrão                     Fraudes Coordenadas / Rede                      Fraudes Isoladas / Avulsas
        (1 Cartão / 1 Device)                      (412 fraudes - 49,2%)                          (426 fraudes - 50,8%)
        Ticket Médio: $137,42                      Volume: $43.443,30 (35,7%)                     Volume: $78.124,76 (64,3%)
                                                   Ticket Médio: $105,44                          Ticket Médio: $183,39
                                                   Mediana: $52,72                                Mediana: $92,00
</pre>"""

new_taxonomy = """```text
                          TAXONOMIA ESTRUTURAL DE TRANSAÇÕES
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         Transações Legítimas                            Transações Fraudulentas
         (22.784 transações)                             (838 transações)
          Volume: US$ 3.130.984                           Volume: US$ 121.568
                  │                                               │
                  │                               ┌───────────────┴───────────────┐
                  ▼                               ▼                               ▼
         Comportamento Padrão            Fraudes em Rede                 Fraudes Avulsas
        (1 Cartão / 1 Device)           (412 tx - 49,2%)                (426 tx - 50,8%)
        Ticket Médio: US$ 137,42        Vol: US$ 43.443 (35,7%)         Vol: US$ 78.125 (64,3%)
                                        Ticket Médio: US$ 105,44        Ticket Médio: US$ 183,39
                                        Mediana: US$ 52,72              Mediana: US$ 92,00
```"""

if old_taxonomy in text:
    text = text.replace(old_taxonomy, new_taxonomy)
    print("Taxonomy diagram replaced successfully!")
else:
    print("WARNING: old_taxonomy not matched exactly")

# Also fix the currency in items 1 and 2 under taxonomy
text = text.replace("ticket médio reduzido (\\$105,44) e mediana baixa (\\$52,72)", "ticket médio reduzido (US$ 105,44) e mediana baixa (US$ 52,72)")
text = text.replace("ticket médio elevado (\\$183,39)", "ticket médio elevado (US$ 183,39)")

# 2. Diagram: Pipeline in Section 4
old_pipeline = """<pre>
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     PIPELINE DE ENGENHARIA E MODELAGEM                                 │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
       Tabela de Transações                                                  Tabela de Identidade
                 │                                                                     │
                 └──────────────────────────────────┬──────────────────────────────────┘
                                                    ▼
                                     Ingestão & Downcasting de RAM
                                   (Float64/Int64 -> Float32/Int8-32)
                                                    │
                                                    ▼
                                       Divisão Temporal Causal
                                (80% Treino: 94k / 20% Validação: 23k)
                                                    │
                                                    ▼
                      ┌─────────────────────────────┴─────────────────────────────┐
                      ▼                                                           ▼
         Engenharia Tabular Avançada                               Topologia de Grafos Bipartidos
         - Normalização Dn (Day - D)                               - Grafo Portador (UID) ↔ Dispositivo
         - UID Mágico Padronizado                                  - Poda Estrita de Nós Sentinela
         - Agregações de Grupo (μ, σ)                              - Centralidade PageRank Estacionária
         - Compressão PCA (339 V → 15 PCA)                         - Velocidade Causal 24h em O(N log K)
                      │                                                           │
                      └─────────────────────────────┬─────────────────────────────┘
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
                                           (XGBoost / Hist-Tree)
                                                    │
                                                    ▼
                                    Avaliação de Risco e Calibração
                                       (Probabilidades Brutas e PP)
</pre>"""

new_pipeline = """```text
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
                   (80% Treino: 94k / 20% Validação: 23k)
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
```"""

if old_pipeline in text:
    text = text.replace(old_pipeline, new_pipeline)
    print("Pipeline diagram replaced successfully!")
else:
    print("WARNING: old_pipeline not matched exactly")

# 3. Diagram: Bipartite Graph in Section 4.3.1
old_bipartite = """<pre>
                TOPOLOGIA BIPARTIDA COM PODA DE NÓS SENTINELA
                
    [PORTADORES (UID)]                           [DISPOSITIVOS (DeviceInfo)]
    
        (UID_1) ───────────┐                 ┌─── (SM-G935F) [deg=4]
                           ├─────────────────┤
        (UID_2) ───────────┤                 └─── (iPhone 8) [deg=1]
                           │
        (UID_3) ───────────┘
        
        (UID_K) ───────────X [ARESTA BLOQUEADA] X─── (null_device) [REMOVIDO]
                             (Evita Hub Artificial / Buraco Negro Topológico)
</pre>"""

new_bipartite = """```text
                TOPOLOGIA BIPARTIDA COM PODA DE NÓS SENTINELA
                
    [PORTADORES (UID)]                           [DISPOSITIVOS (DeviceInfo)]
    
        (UID_1) ───────────┐                 ┌─── (SM-G935F) [deg=4]
                           ├─────────────────┤
        (UID_2) ───────────┤                 └─── (iPhone 8) [deg=1]
                           │
        (UID_3) ───────────┘
        
        (UID_K) ───────────X [ARESTA BLOQUEADA] X─── (null_device) [REMOVIDO]
                             (Evita Hub Artificial / Buraco Negro Topológico)
```"""

if old_bipartite in text:
    text = text.replace(old_bipartite, new_bipartite)
    print("Bipartite diagram replaced successfully!")
else:
    print("WARNING: old_bipartite not matched exactly")

# 4. Diagram: Leakage Audit in Section 4.6
old_leakage = """<pre>
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 RELATÓRIO FORMAL DE AUDITORIA DE LEAKAGES                              │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘

1. CORREÇÃO DA BUSCA BINÁRIA DE VELOCIDADE 24H:
   • Diagnóstico: Algoritmos anteriores aplicavam np.searchsorted sobre o vetor temporal global,
     contando transações de outros clientes descorrelacionados e agindo como proxy de horário do dia.
   • Correção: Implementação da busca binária causal agrupada estritamente dentro do histórico do UID.
   • Resultado: O contador mede exclusivamente a cadência transacional do portador avaliado (O(N log K)).

2. CORREÇÃO DE VAZAMENTO NO PÓS-PROCESSAMENTO POR CLIENTE:
   • Diagnóstico: O cálculo com groupby('uid')['raw_score'].transform('mean') incluía predições
     de compras futuras do mesmo cliente na validação, inflando artificialmente as métricas.
   • Correção: Conversão para média cumulativa expansiva (expanding().mean()), utilizando apenas o passado.
   • Resultado: Avaliação com e sem pós-processamento reportada lado a lado de forma transparente.

3. DESACOPLAMENTO E ISOLAMENTO DE GNN:
   • Diagnóstico: Redes GNN supervisionadas treinadas com isFraud sofriam de memorização e vazamento.
   • Correção: Conversão para regime estritamente auto-supervisionado de reconstrução de links.
   • Resultado: Eliminação de dependência de rótulos em camadas de representação relacional.

4. ELIMINAÇÃO DE SIMULAÇÃO ARTIFICIAL NA FUNÇÃO DE CUSTO:
   • Diagnóstico: Avaliações preliminares de custo financeiro utilizavam multiplicadores sintéticos (* 1.35).
   • Correção: Aplicação da função de custo estritamente sobre as probabilidades reais inferidas pelo XGBoost.
   • Resultado: Métricas financeiras e curvas de custo 100% ancoradas em predições empíricas verificáveis.
</pre>"""

new_leakage = """```text
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
   • Resultado: Eliminação de dependência de rótulos em camadas de representação.

4. ELIMINAÇÃO DE SIMULAÇÃO ARTIFICIAL NA FUNÇÃO DE CUSTO:
   • Diagnóstico: Avaliações preliminares utilizavam multiplicadores sintéticos (* 1,35).
   • Correção: Aplicação da função de custo estritamente sobre probabilidades reais do XGBoost.
   • Resultado: Métricas financeiras e curvas 100% ancoradas em predições empíricas verificáveis.
```"""

if old_leakage in text:
    text = text.replace(old_leakage, new_leakage)
    print("Leakage report replaced successfully!")
else:
    print("WARNING: old_leakage not matched exactly")

# 5. Diagram: Subgraph in Section 6.5
old_subgraph = """<pre>
                           SUBGRAFO REAL DO DISPOSITIVO SM-G935F
                           
           [Portador 1] (UID_A) ──────────┐
           [Portador 2] (UID_B) ──────────┤
                                          ├─── (Dispositivo SM-G935F) [device_degree = 4]
           [Portador 3] (UID_C) ──────────┤
           [Portador 4] (UID_D) ──────────┘
                  │
                  ▼
       Transação #3428190: Valor = $48,50 (Ticket Baixo / Comportamento Comum)
       ──────────────────────────────────────────────────────────────────────────
       • Predição Baseline Tabular (IID) : p = 14,2%  -> FALSO NEGATIVO (APROVADO)
       • Predição com Grafos Bipartidos  : p = 58,9%  -> VERDADEIRO POSITIVO (BLOQUEADO)
</pre>"""

new_subgraph = """```text
                           SUBGRAFO REAL DO DISPOSITIVO SM-G935F
                           
           [Portador 1] (UID_A) ──────────┐
           [Portador 2] (UID_B) ──────────┤
                                          ├─── (Dispositivo SM-G935F) [device_degree = 4]
           [Portador 3] (UID_C) ──────────┤
           [Portador 4] (UID_D) ──────────┘
                  │
                  ▼
       Transação #3428190: Valor = US$ 48,50 (Ticket Baixo / Comportamento Comum)
       ──────────────────────────────────────────────────────────────────────────
       • Predição Baseline Tabular (IID) : p = 14,2%  -> FALSO NEGATIVO (APROVADO)
       • Predição com Grafos Bipartidos  : p = 58,9%  -> VERDADEIRO POSITIVO (BLOQUEADO)
```"""

if old_subgraph in text:
    text = text.replace(old_subgraph, new_subgraph)
    print("Subgraph diagram replaced successfully!")
else:
    print("WARNING: old_subgraph not matched exactly")

# 6. List formatting fixes (ensure blank line before list items)
# Section 4.5: Model A & Model B
old_s45 = """desacoplamos o treinamento em dois classificadores especializados:
* **Model A (Sem Identidade - Fallback Operacional)**:"""
new_s45 = """desacoplamos o treinamento em dois classificadores especializados:

* **Model A (Sem Identidade - Fallback Operacional)**:"""
text = text.replace(old_s45, new_s45)

# Section 6.3 - Item A: Tabular vs Graphs
old_s63 = """A resposta reside na **complementaridade funcional** entre os dois eixos de dados:
* O Feature Engineering Tabular"""
new_s63 = """A resposta reside na **complementaridade funcional** entre os dois eixos de dados:

* O Feature Engineering Tabular"""
text = text.replace(old_s63, new_s63)

# Section 6.4: Ranking
old_s64 = """as variáveis de rede posicionaram-se destacadamente entre as mais discriminatórias:
1. `TransactionAmt`"""
new_s64 = """as variáveis de rede posicionaram-se destacadamente entre as mais discriminatórias:

1. `TransactionAmt`"""
text = text.replace(old_s64, new_s64)

# Section 8.2: Hybrid superior
old_s82 = """A investigação experimental consolida uma diretriz para a engenharia de sistemas em produção:
* **GNNs puras end-to-end"""
new_s82 = """A investigação experimental consolida uma diretriz para a engenharia de sistemas em produção:

* **GNNs puras end-to-end"""
text = text.replace(old_s82, new_s82)

# Section 8.3: Limitações
old_s83 = """### 8.3 Limitações Metodológicas
1. **Amostragem Representativa Temporal (20%)**:"""
new_s83 = """### 8.3 Limitações Metodológicas

1. **Amostragem Representativa Temporal (20%)**:"""
text = text.replace(old_s83, new_s83)

# Section 9: Conclusão
old_s9 = """As principais contribuições científicas e de engenharia de software incluem:
1. **Quebra Comprovada da Premissa IID**:"""
new_s9 = """As principais contribuições científicas e de engenharia de software incluem:

1. **Quebra Comprovada da Premissa IID**:"""
text = text.replace(old_s9, new_s9)

with open("LOCAL_COMPLETE_PAPER.md", "w", encoding="utf-8") as f:
    f.write(text)

print("LOCAL_COMPLETE_PAPER.md successfully refined!")
