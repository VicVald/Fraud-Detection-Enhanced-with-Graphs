# 1. Update LOCAL_COMPLETE_PAPER.md
with open('LOCAL_COMPLETE_PAPER.md', 'r', encoding='utf-8') as f:
    local_p = f.read()

# Taxonomy block replacement
local_p = local_p.replace('1.040 tx - 49,38%', '1.260 tx - 59,83%')
local_p = local_p.replace('1.028 tx - 50,62%', '846 tx - 40,17%')
local_p = local_p.replace('Perda Bruta: US$ 361.770', 'Perda Bruta: US$ 314.263')
local_p = local_p.replace(
    '46,65% das Perdas (US$ 168.767) 53,35% das Perdas (US$ 193.003)',
    '46,65% das Perdas (US$ 146.605) 53,35% das Perdas (US$ 167.658)'
)

old_t1 = '1. **Fraudes Coordenadas em Rede ($N = 1.040$, 49,38% das fraudes de validação)**: operam sob infraestrutura compartilhada de hardware (`device_degree` $\\ge 2$) ou em velocidade acelerada (`uid_tx_count_24h` $\\ge 2$). Concentram **46,65% das perdas financeiras brutas por fraude** (US$ 168.767 na validação)'
new_t1 = '1. **Fraudes Coordenadas em Rede ($N = 1.260$, 59,83% das fraudes de validação)**: operam sob infraestrutura compartilhada de hardware (`device_degree` $\\ge 2$) ou em velocidade acelerada (`uid_tx_count_24h` $\\ge 2$). Concentram **46,65% das perdas financeiras brutas por fraude** (US$ 146.605 na validação)'
assert old_t1 in local_p, 'old_t1 not found'
local_p = local_p.replace(old_t1, new_t1)

old_t2 = '2. **Fraudes Isoladas e Oportunistas ($N = 1.028$, 50,62% das fraudes de validação)**: operam de forma avulsa a partir de dispositivos dedicados ou sem metadados preenchidos (`null_device`), respondendo por 53,35% das perdas.'
new_t2 = '2. **Fraudes Isoladas e Oportunistas ($N = 846$, 40,17% das fraudes de validação)**: operam de forma avulsa a partir de dispositivos dedicados ou sem metadados preenchidos (`null_device`), respondendo por **53,35% das perdas** (US$ 167.658 na validação).'
assert old_t2 in local_p, 'old_t2 not found'
local_p = local_p.replace(old_t2, new_t2)

# Table 7.2 intro
old_intro_loss = 'com 2.106 fraudes totalizando US$ 361.770 em perdas financeiras brutas, arquivo `val_predictions.csv`)'
new_intro_loss = 'com 2.106 fraudes totalizando US$ 314.263,20 em perdas financeiras brutas, arquivo `val_predictions.csv`)'
assert old_intro_loss in local_p, 'old_intro_loss not found'
local_p = local_p.replace(old_intro_loss, new_intro_loss)

old_intro_blind = 'gerando um prejuízo de US$ 383.536,66 ao somar os montantes furtados e as multas regulatórias de estorno)'
new_intro_blind = 'gerando um prejuízo de US$ 345.853,20 ao somar os montantes furtados de US$ 314.263,20 às multas regulatórias de estorno de US$ 31.590,00)'
assert old_intro_blind in local_p, 'old_intro_blind not found'
local_p = local_p.replace(old_intro_blind, new_intro_blind)

# Table 7.2 rows
old_table_72 = """| — | **Aprovação Cega (Sem Modelo)** | 1,00 | US$ 383.536,66 | US$ 383.536,66 | US$ 0,00 | — |
| **1A** | Baseline Raw (Sem PCA) | 0,36 | US$ 174.956,30 | US$ 135.678,36 | US$ 39.150,04 | +US$ 208.580,36 (-54,4%) |
| **1B** | Baseline (Com PCA 15-d) | 0,17 | US$ 169.059,03 | US$ 89.991,61 | US$ 78.916,22 | +US$ 214.477,63 (-55,9%) |
| **2** | Baseline + Apenas Topologia | 0,27 | US$ 175.526,75 | US$ 122.726,26 | US$ 52.665,59 | +US$ 208.009,91 (-54,2%) |
| **3** | Baseline + Apenas Velocidade | 0,20 | US$ 176.323,01 | US$ 106.535,26 | US$ 69.643,56 | +US$ 207.213,65 (-54,0%) |
| **4** | Baseline + Grafos Completos | 0,23 | US$ 171.179,36 | US$ 107.828,37 | US$ 63.209,49 | +US$ 212.357,30 (-55,4%) |
| **5** | Baseline + Apenas FE Tabular | 0,08 | US$ 200.697,43 | US$ 165.799,53 | US$ 34.760,39 | +US$ 182.839,23 (-47,7%) |
| **6A** | All-in-One (Com PCA 15-d) | 0,09 | US$ 195.479,99 | US$ 158.451,37 | US$ 36.888,43 | +US$ 188.056,67 (-49,0%) |
| **6B** | All-in-One (Sem PCA / Raw V) | 0,10 | US$ 199.541,46 | US$ 165.417,27 | US$ 33.985,98 | +US$ 183.995,20 (-48,0%) |
| **6B PP**| All-in-One + Pós-Proc Causal | 0,10 | **US$ 197.883,56** | US$ 162.304,93 | US$ 35.439,23 | +US$ 185.653,10 (-48,4%) |"""

new_table_72 = """| — | **Aprovação Cega (Sem Modelo)** | 1,00 | US$ 345.853,20 | US$ 345.853,20 | US$ 0,00 | — |
| **1A** | Baseline Raw (Sem PCA) | 0,36 | US$ 174.956,30 | US$ 135.678,36 | US$ 39.150,04 | +US$ 170.896,90 (-49,4%) |
| **1B** | Baseline (Com PCA 15-d) | 0,17 | US$ 169.059,03 | US$ 89.991,61 | US$ 78.916,22 | +US$ 176.794,17 (-51,1%) |
| **2** | Baseline + Apenas Topologia | 0,27 | US$ 175.526,75 | US$ 122.726,26 | US$ 52.665,59 | +US$ 170.326,45 (-49,2%) |
| **3** | Baseline + Apenas Velocidade | 0,20 | US$ 176.323,01 | US$ 106.535,26 | US$ 69.643,56 | +US$ 169.530,19 (-49,0%) |
| **4** | Baseline + Grafos Completos | 0,23 | US$ 171.179,36 | US$ 107.828,37 | US$ 63.209,49 | +US$ 174.673,84 (-50,5%) |
| **5** | Baseline + Apenas FE Tabular | 0,08 | US$ 200.697,43 | US$ 165.799,53 | US$ 34.760,39 | +US$ 145.155,77 (-42,0%) |
| **6A** | All-in-One (Com PCA 15-d) | 0,09 | US$ 195.479,99 | US$ 158.451,37 | US$ 36.888,43 | +US$ 150.373,21 (-43,5%) |
| **6B** | All-in-One (Sem PCA / Raw V) | 0,10 | US$ 199.541,46 | US$ 165.417,27 | US$ 33.985,98 | +US$ 146.311,74 (-42,3%) |
| **6B PP**| All-in-One + Pós-Proc Causal | 0,10 | **US$ 197.883,56** | US$ 162.304,93 | US$ 35.439,23 | +US$ 147.969,64 (-42,8%) |"""

assert old_table_72 in local_p, 'Old Table 7.2 not found'
local_p = local_p.replace(old_table_72, new_table_72)

# Diagnosis 1 below table
old_t72_diag = '1. **Redução Massiva de Prejuízo sobre a Inação**: Qualquer um dos modelos de aprendizado supervisionado reduz o prejuízo operacional em mais de **US$ 182.000 a US$ 214.000** em relação à aprovação cega (queda superior a 50% das perdas totais).'
new_t72_diag = '1. **Redução Massiva de Prejuízo sobre a Inação**: Qualquer um dos modelos de aprendizado supervisionado reduz o prejuízo operacional em mais de **US$ 145.000 a US$ 176.000** em relação à aprovação cega (queda de 42,0% a 51,1% das perdas totais).'
assert old_t72_diag in local_p, 'T72 diag not found'
local_p = local_p.replace(old_t72_diag, new_t72_diag)

# Section 7.3 and conclusion
old_res_73 = '*Ressalva Metodológica sobre o Pós-Processamento Causal*: O incremento de +0,64 p.p. de recall atribuível ao filtro temporal expansivo alinhado corresponde à captura de cerca de **7 transações coordenadas adicionais** (em um universo formal de 1.040 fraudes coordenadas na partição de validação de 59.054 registros).'
new_res_73 = '*Ressalva Metodológica sobre o Pós-Processamento Causal*: O incremento de +0,64 p.p. de recall atribuível ao filtro temporal expansivo alinhado corresponde à captura de exatamente **8 transações coordenadas adicionais** (em um universo de 1.260 fraudes coordenadas na partição de validação de 59.054 registros).'
assert old_res_73 in local_p, 'Ressalva 7.3 not found'
local_p = local_p.replace(old_res_73, new_res_73)

old_c5 = '+0,64 p.p. provêm do pós-processamento causal (efeito marginal de cerca de 7 transações no universo formal de 1.040 fraudes coordenadas de validação).'
new_c5 = '+0,64 p.p. provêm do pós-processamento causal (efeito marginal de 8 transações no universo de 1.260 fraudes coordenadas de validação).'
assert old_c5 in local_p, 'Conclusao 5 not found'
local_p = local_p.replace(old_c5, new_c5)

with open('LOCAL_COMPLETE_PAPER.md', 'w', encoding='utf-8') as f:
    f.write(local_p)
print('✓ LOCAL_COMPLETE_PAPER.md updated successfully!')

# 2. Update COMPLETE_PAPER.md
with open('COMPLETE_PAPER.md', 'r', encoding='utf-8') as f:
    comp_p = f.read()

old_comp_note = 'corresponde à captura adicional de cerca de 7 transações coordenadas (em um total formal de 1.040 fraudes coordenadas na partição de validação).'
new_comp_note = 'corresponde à captura adicional de 8 transações coordenadas (em um total de 1.260 fraudes coordenadas na partição de validação).'
assert old_comp_note in comp_p, 'comp_note not found'
comp_p = comp_p.replace(old_comp_note, new_comp_note)

old_comp_sec5 = """1. TARGET ENCODING NO DOMÍNIO SINTÉTICO (ETAPA 3):
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

4. CORREÇÃO DE ALINHAMENTO NO PÓS-PROCESSAMENTO TEMPORAL (AUDITORIA OPERACIONAL):
   • Diagnóstico: groupby('uid')['raw_score'].expanding().mean().values ordenava alfabeticamente
     por portador e aplicava posicionalmente sobre o frame ordenado no tempo, desalinhandos predições.
   • Correção: Emprego de .reset_index(level=0, drop=True) preservando alinhamento por índice temporal.
   • Resultado: Eliminação de contaminação cruzada entre portadores, estabilizando os ganhos pós-proc.

5. GRANULARIDADE DE DeviceInfo E DESMISTIFICAÇÃO DE FRAUD RINGS:
   • Diagnóstico: Identificação de que DeviceInfo representa modelo/SO (ex.: Windows, SM-G935F) e não MAC/IMEI único.
   • Correção: Revisão da leitura de megahubs (taxa cai de 14,9% em 11-100 UIDs para 5,8% em >100 UIDs).
   • Resultado: Interpretação não-linear robusta e rigor conceitual na modelagem de redes bipartidas."""

new_comp_sec5 = """1. CORREÇÃO DA BUSCA BINÁRIA DE VELOCIDADE 24H:
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
   • Resultado: Modelagem não-linear adequada via árvores e desmistificação teórica de fraud rings."""

assert old_comp_sec5 in comp_p, 'Old Section 5 in COMPLETE_PAPER not found'
comp_p = comp_p.replace(old_comp_sec5, new_comp_sec5)

with open('COMPLETE_PAPER.md', 'w', encoding='utf-8') as f:
    f.write(comp_p)
print('✓ COMPLETE_PAPER.md updated successfully!')

# 3. Update RELATORIO_METODOLOGICO_NOTEBOOKS.md
with open('docs/RELATORIO_METODOLOGICO_NOTEBOOKS.md', 'r', encoding='utf-8') as f:
    rel_p = f.read()

old_r1 = '($N = 1.040$ fraudes na validação, respondendo por 46,65% das perdas financeiras brutas);'
new_r1 = '($N = 1.260$ fraudes na validação, respondendo por 46,65% das perdas financeiras brutas, totalizando US$ 146.605,41);'
assert old_r1 in rel_p, 'old_r1 not found'
rel_p = rel_p.replace(old_r1, new_r1)

old_r2 = 'Fraudes Isoladas**: Transações avulsas sem compartilhamento de dispositivo nem cadência acelerada em 24h ($N = 1.028$ fraudes).'
new_r2 = 'Fraudes Isoladas**: Transações avulsas sem compartilhamento de dispositivo nem cadência acelerada em 24h ($N = 846$ fraudes, respondendo por 53,35% das perdas, totalizando US$ 167.657,80).'
assert old_r2 in rel_p, 'old_r2 not found'
rel_p = rel_p.replace(old_r2, new_r2)

old_r3 = '| Configuração Avaliada | Recall Geral (FPR=1%) | Recall Fraudes Coordenadas ($N=1.040$) | Recall Fraudes Isoladas ($N=1.028$) | Contribuição Metodológica do Lift Coordenado |'
new_r3 = '| Configuração Avaliada | Recall Geral (FPR=1%) | Recall Fraudes Coordenadas ($N=1.260$) | Recall Fraudes Isoladas ($N=846$) | Contribuição Metodológica do Lift Coordenado |'
assert old_r3 in rel_p, 'old_r3 not found'
rel_p = rel_p.replace(old_r3, new_r3)

old_r4 = 'corresponde à captura adicional de cerca de **7 transações coordenadas** (em um universo formal de 1.040 fraudes coordenadas na partição de validação).'
new_r4 = 'corresponde à captura adicional de exatamente **8 transações coordenadas** (em um universo de 1.260 fraudes coordenadas na partição de validação).'
assert old_r4 in rel_p, 'old_r4 not found'
rel_p = rel_p.replace(old_r4, new_r4)

old_r5 = '46.65% das perdas ($168.767)'
new_r5 = '46.65% das perdas ($146.605)'
assert old_r5 in rel_p, 'old_r5 not found'
rel_p = rel_p.replace(old_r5, new_r5)

with open('docs/RELATORIO_METODOLOGICO_NOTEBOOKS.md', 'w', encoding='utf-8') as f:
    f.write(rel_p)
print('✓ RELATORIO_METODOLOGICO_NOTEBOOKS.md updated successfully!')
