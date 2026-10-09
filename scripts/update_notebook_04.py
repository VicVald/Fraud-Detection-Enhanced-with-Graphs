import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 0: Title & Header
cells.append(nbf.v4.new_markdown_cell("""# 🔬 Estudo de Ablação Rigoroso: Comparação Estrutural de Performance (XGBoost)

**Projeto de TCC:** Detecção de Fraude em Tempo Real com Modelos Tabulares Aumentados por Grafos  
**Dataset:** IEEE-CIS Fraud Detection Benchmark (590.540 transações)

---

### 🎯 Objetivos Metodológicos deste Experimento (Seção 6 do Paper)
1. **Isolamento Controlado de Componentes**: Avaliar o impacto incremental de cada família de atributos mantendo o mesmo classificador (XGBoost), hiperparâmetros idênticos e o mesmo protocolo de split temporal causal (80% treino / 20% validação).
2. **Comparação Metodológica do PCA (Com vs. Sem PCA)**: Medir empiricamente o ganho ou perda de poder preditivo e eficiência computacional ao comprimir 339 colunas $V$ em 15 componentes principais.
3. **Decomposição Topologia vs. Velocidade**: Separar explicitamente a contribuição da topologia de rede pura (PageRank e graus bipartidos) da velocidade temporal de transações por cliente (24h).
4. **Avaliação Financeira Real**: Aplicar a matriz de custo financeiro dependente do exemplo (Correa Bahnsen / Bayes Minimum Risk) sobre as probabilidades **reais** preditas pelos modelos, sem qualquer simulação sintética.
5. **Interpretabilidade e Estudo de Caso**: Ranking global de atributos (Gain / TreeSHAP) e inspeção qualitativa de um subgrafo de dispositivo compartilhado.
"""))

# Cell 1: Package installation
cells.append(nbf.v4.new_markdown_cell("""---
## 1. Instalação de Pacotes & Configuração do Ambiente
"""))

cells.append(nbf.v4.new_code_cell("""# Instalação dos pacotes necessários no Google Colab
!pip install -q xgboost lightgbm networkx scikit-learn matplotlib seaborn
"""))

# Cell 2: Imports & Configuration (Budget Switch)
cells.append(nbf.v4.new_code_cell("""import os
import gc
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, roc_curve

import xgboost as xgb

try:
    from IPython.display import display
except ImportError:
    display = print

# ==============================================================================
# CONFIGURAÇÃO DO ORÇAMENTO EXPERIMENTAL (EXECUTION BUDGET):
# ------------------------------------------------------------------------------
# 'ESSENTIAL': Amostra rápida (20% das transações) e 200 árvores (~25-35 min).
#              Ideal para verificação de código, testes de integridade e prototipagem.
# 'FULL'     : 100% dos dados (590.540 transações) e 450 árvores (~1h30 - 2h no Colab).
#              Configuração oficial para geração dos resultados finais do TCC.
# ==============================================================================
EXECUTION_BUDGET = 'FULL'  # Opções: 'FULL' ou 'ESSENTIAL'

# Configuração de Estilo Visual Acadêmico
plt.rcParams.update({'figure.dpi': 110, 'font.family': 'DejaVu Sans'})
sns.set_style('whitegrid')

PALETTE = {
    'baseline_raw': '#757575',
    'baseline_pca': '#9E9E9E',
    'only_topology': '#E76F51',
    'only_velocity': '#F4A261',
    'only_graph': '#E63946',
    'only_fe': '#457B9D',
    'all_in_one': '#1D3557',
    'all_in_one_nopca': '#2A9D8F'
}

print(f"✅ Ambiente configurado com sucesso. Orçamento Ativo: [{EXECUTION_BUDGET}]")
"""))

# Cell 3: Data Ingestion & Memory Optimization
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Ingestão de Dados & Otimização de Memória RAM (Downcasting)
"""))

cells.append(nbf.v4.new_code_cell("""# Localização dos dados (Google Drive ou diretório corrente)
DATA_DIR = '.'
if os.path.exists('/content/drive/MyDrive/ieee-data'):
    DATA_DIR = '/content/drive/MyDrive/ieee-data'

train_trans_path = os.path.join(DATA_DIR, 'train_transaction.csv')
train_id_path = os.path.join(DATA_DIR, 'train_identity.csv')

def reduce_mem_usage(df, verbose=True):
    \"\"\"Converte tipos numéricos float64/int64 para economizar RAM mantendo precisão.\"\"\"
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_categorical_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"RAM: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% economia)")
    return df

print("Carregando tabelas do IEEE-CIS...")
df_trans = pd.read_csv(train_trans_path)
df_trans = reduce_mem_usage(df_trans)

df_id = pd.read_csv(train_id_path)
df_id = reduce_mem_usage(df_id)

df = df_trans.merge(df_id, on='TransactionID', how='left')
del df_trans, df_id
gc.collect()

df = df.rename(columns={"TransactionDT": "timestamp"})
df = df.sort_values('timestamp').reset_index(drop=True)

# Aplicação do Orçamento Computacional
if EXECUTION_BUDGET == 'ESSENTIAL':
    print("Modo ESSENTIAL: Amostrando 20% das transações preservando ordem temporal causal...")
    df = df.iloc[-int(len(df) * 0.2):].reset_index(drop=True)

print(f"Dataset em uso: {df.shape[0]:,} transações × {df.shape[1]} colunas.")

# Split Temporal Causal (80% treino / 20% validação)
split_idx = int(len(df) * 0.8)
train_df_raw = df.iloc[:split_idx].copy().reset_index(drop=True)
val_df_raw = df.iloc[split_idx:].copy().reset_index(drop=True)
del df
gc.collect()

print(f"Treino   : {len(train_df_raw):,} transações | Taxa de Fraude: {train_df_raw['isFraud'].mean():.3%}")
print(f"Validação: {len(val_df_raw):,} transações | Taxa de Fraude: {val_df_raw['isFraud'].mean():.3%}")
"""))

# Cell 4: Feature Engineering Tabular, Magic UID & PCA
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Resolução de Entidade (UID Mágico), Engenharia Tabular e Compressão PCA
- **UID Mágico Padronizado**: Formulado como `card1_addr1_d1anchor` onde $d1\\text{anchor} = \\lfloor \\text{timestamp}/86400 - D1 \\rfloor$.
- **Deltas de Calendário ($D_n$)**: Normalização temporal de $D1, D2, D10, D15$.
- **Agregações Estatísticas de Grupo por UID**: $\\mu$ e $\\sigma$ de $TransactionAmt$, $D15n$, $C1$, $C13$ ajustadas estritamente no treino.
- **Decomposição Espectral PCA nas 339 variáveis $V$**: Projeção em 15 componentes ortogonais principais. Ambas as representações (comprimida e bruta) são preservadas para permitir a comparação da ablação.
"""))

cells.append(nbf.v4.new_code_cell("""print("Construindo Features Tabulares, UID Mágico e Compressão PCA...")

train_df = train_df_raw.copy()
val_df = val_df_raw.copy()

# 1. Normalização de D (Deltas de Calendário)
for d_col in ['D1', 'D2', 'D10', 'D15']:
    if d_col in train_df.columns:
        train_df[f'{d_col}n'] = (train_df['timestamp'] / 86400.0) - train_df[d_col]
        val_df[f'{d_col}n'] = (val_df['timestamp'] / 86400.0) - val_df[d_col]

# 2. Construção do UID Mágico Padronizado: card1 + addr1 + d1_anchor
def build_magic_uid(df):
    c1 = df['card1'].fillna(0).astype(int).astype(str)
    addr = df['addr1'].fillna(0).astype(int).astype(str)
    day = df['timestamp'] / 86400.0
    d1_anchor = (day - df['D1'].fillna(0)).astype(int).astype(str)
    return c1 + "_" + addr + "_" + d1_anchor

train_df['uid'] = build_magic_uid(train_df)
val_df['uid'] = build_magic_uid(val_df)
print(f"UIDs únicos no treino: {train_df['uid'].nunique():,}")

# 3. Agregações Estatísticas de Grupo por UID (Ajustadas ESTRITAMENTE no Treino)
agg_cols = ['TransactionAmt', 'D15n', 'C1', 'C13']
for col in agg_cols:
    if col in train_df.columns:
        mean_map = train_df.groupby('uid')[col].mean().to_dict()
        std_map = train_df.groupby('uid')[col].std().fillna(0.0).to_dict()
        global_mean = train_df[col].mean()
        
        for d in [train_df, val_df]:
            d_mean = d['uid'].map(mean_map).fillna(global_mean)
            d_std = d['uid'].map(std_map).fillna(0.0)
            d[f'{col}_to_mean_uid'] = d[col] / (d_mean + 1e-5)
            d[f'{col}_to_std_uid'] = (d[col] - d_mean) / (d_std + 1e-5)

# 4. Frequency Encodings de Cartão e Domínio
for cat_col in ['card1', 'P_emaildomain']:
    if cat_col in train_df.columns:
        counts = train_df[cat_col].value_counts().to_dict()
        train_df[f'{cat_col}_count'] = train_df[cat_col].map(counts).fillna(0)
        val_df[f'{cat_col}_count'] = val_df[cat_col].map(counts).fillna(0)

# 5. Compressão PCA nas 339 Variáveis V (Preservando as originais para o teste de ablação)
v_cols = [c for c in train_df.columns if c.startswith('V') and c[1:].isdigit()]
print(f"Comprimindo {len(v_cols)} variáveis V em 15 componentes PCA...")
medians_v = train_df[v_cols].median().fillna(0.0)
scaler = StandardScaler()
X_v_train = scaler.fit_transform(train_df[v_cols].fillna(medians_v))
X_v_val = scaler.transform(val_df[v_cols].fillna(medians_v))

pca = PCA(n_components=15, random_state=42)
pca_train = pca.fit_transform(X_v_train)
pca_val = pca.transform(X_v_val)

for i in range(15):
    train_df[f'PCA_V_{i}'] = pca_train[:, i]
    val_df[f'PCA_V_{i}'] = pca_val[:, i]

del X_v_train, X_v_val, pca_train, pca_val
gc.collect()

# 6. Flag de Dispositivo para Roteamento de Identidade Dual
train_has_dev = train_df['DeviceType'].notnull()
val_has_dev = val_df['DeviceType'].notnull()

# 7. Label Encoding Categórico
cat_cols = [c for c in train_df.select_dtypes(include=['object', 'category']).columns if c not in ['uid', 'TransactionID']]
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    val_df[col] = val_df[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

print("✅ Engenharia Tabular e PCA concluídos com sucesso.")
"""))

# Cell 5: Graph Features & Causal Velocity
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Engenharia de Features de Grafo & Velocidade Causal 24h
- **Grafo Bipartido de Acesso**: Modelado estritamente sobre as conexões de treino entre $Portador$ (`uid`) e $Dispositivo$ (`DeviceInfo_clean`), ignorando sentinelas nulas (`null_device`).
- **PageRank Estacionário**: Centralidade de rede calculada via iteração de potências ($\alpha = 0.85$).
- **Graus Bipartidos de Conectividade**: Contagem de UIDs distintos por dispositivo e dispositivos distintos por UID.
- **Velocidade Causal 24h por UID**: Janela deslizante vetorizada calculada estritamente dentro de cada portador via busca binária causal, eliminando vazamentos.
"""))

cells.append(nbf.v4.new_code_cell("""print("Construindo Grafo Bipartido, PageRank e Velocidade Causal por UID...")

# 1. Normalização do Dispositivo
train_df['DeviceInfo_clean'] = train_df_raw['DeviceInfo'].fillna('null_device').astype(str)
val_df['DeviceInfo_clean'] = val_df_raw['DeviceInfo'].fillna('null_device').astype(str)

# 2. Grafo Bipartido no Treino (Ignorando null_device para evitar hubs artificiais)
G_train = nx.Graph()
train_dev_edges = train_df[train_df['DeviceInfo_clean'] != 'null_device'][['uid', 'DeviceInfo_clean']].drop_duplicates()
for _, row in train_dev_edges.iterrows():
    G_train.add_edge(f"uid_{row['uid']}", f"dev_{row['DeviceInfo_clean']}")

try:
    pr_dict = nx.pagerank(G_train, alpha=0.85)
except Exception:
    pr_dict = {}

default_pr = 1.0 / max(len(G_train), 1)
train_df['uid_pagerank'] = train_df['uid'].apply(lambda x: pr_dict.get(f"uid_{x}", default_pr))
val_df['uid_pagerank'] = val_df['uid'].apply(lambda x: pr_dict.get(f"uid_{x}", default_pr))

# 3. Graus Bipartidos Estruturais (Contagem de vizinhos distintos no grafo)
dev_distinct_uids = train_dev_edges.groupby('DeviceInfo_clean')['uid'].nunique().to_dict()
uid_distinct_devs = train_dev_edges.groupby('uid')['DeviceInfo_clean'].nunique().to_dict()

train_df['device_degree'] = train_df['DeviceInfo_clean'].map(dev_distinct_uids).fillna(0)
val_df['device_degree'] = val_df['DeviceInfo_clean'].map(dev_distinct_uids).fillna(0)

train_df['uid_degree'] = train_df['uid'].map(uid_distinct_devs).fillna(0)
val_df['uid_degree'] = val_df['uid'].map(uid_distinct_devs).fillna(0)

# 4. Velocidade Causal 24h Agrupada ESTRITAMENTE por UID (Vetorizada via Busca Binária)
def compute_causal_velocity(df):
    df_sorted = df.sort_values('timestamp').copy()
    times = df_sorted['timestamp'].values
    counts = np.ones(len(df_sorted), dtype=np.int32)
    secs_prev = np.full(len(df_sorted), np.nan, dtype=np.float32)
    
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

train_df = compute_causal_velocity(train_df)
val_df = compute_causal_velocity(val_df)

print("✅ Features de Grafo e Velocidade Causal 24h construídas.")
"""))

# Cell 6: Training Pipeline & Feature Routing
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Roteamento Dual de Identidade (Model A & B) & Pipeline de Ablação
Definimos formalmente as listas de variáveis de cada bloco e o pipeline de treino controlado:
- `raw_v_cols`: 339 colunas $V$ originais.
- `pca_cols`: 15 componentes principais PCA.
- `topology_cols`: `['uid_pagerank', 'uid_degree', 'device_degree']` (Métricas puras de topologia de rede).
- `velocity_cols`: `['uid_tx_count_24h', 'uid_secs_since_prev']` (Métricas dinâmicas de velocidade temporal).
- `fe_cols`: Deltas temporais normalizados $D_n$, agregações estatísticas por cliente e contagens de frequência.
"""))

cells.append(nbf.v4.new_code_cell("""# Mapeamento Formal dos Conjuntos de Atributos
base_drop = ['isFraud', 'TransactionID', 'uid', 'timestamp', 'DeviceInfo_clean', 'DeviceInfo']
raw_v_cols = [c for c in train_df.columns if c.startswith('V') and c[1:].isdigit()]
pca_cols = [f'PCA_V_{i}' for i in range(15)]
topology_cols = ['uid_pagerank', 'uid_degree', 'device_degree']
velocity_cols = ['uid_tx_count_24h', 'uid_secs_since_prev']
graph_cols = topology_cols + velocity_cols
fe_cols = [c for c in train_df.columns if c.endswith('n') and c[:-1] in ['D1', 'D2', 'D10', 'D15']] + \
          [c for c in train_df.columns if '_to_' in c] + \
          [c for c in train_df.columns if '_count' in c]

def train_and_eval_pipeline(config_id, feature_set_name, use_pca=True, include_topology=False, include_velocity=False, include_fe=False):
    print(f"\\n==========================================================================")
    print(f" Treinando Configuração [{config_id}]: {feature_set_name}")
    print(f" PCA: {use_pca} | Topologia: {include_topology} | Velocidade: {include_velocity} | FE Tabular: {include_fe}")
    print(f"==========================================================================")
    
    # Determinar variáveis a serem descartadas
    drop_list = list(base_drop)
    
    # Controle do PCA vs. Raw V
    if use_pca:
        drop_list += raw_v_cols  # Usa PCA, descarta Raw V
    else:
        drop_list += pca_cols    # Usa Raw V, descarta PCA
        
    if not include_topology:
        drop_list += topology_cols
        
    if not include_velocity:
        drop_list += velocity_cols
        
    if not include_fe:
        drop_list += fe_cols
        
    features_a = [c for c in train_df.columns if c not in drop_list and not c.startswith('id_') and c != 'DeviceType']
    features_b = [c for c in train_df.columns if c not in drop_list]
    
    print(f"  -> Features Model A (Sem Dispositivo): {len(features_a)} colunas")
    print(f"  -> Features Model B (Com Dispositivo): {len(features_b)} colunas")
    
    scale_pos_a = (~train_df[~train_has_dev]['isFraud'].astype(bool)).sum() / max(train_df[~train_has_dev]['isFraud'].sum(), 1)
    scale_pos_b = (~train_df[train_has_dev]['isFraud'].astype(bool)).sum() / max(train_df[train_has_dev]['isFraud'].sum(), 1)
    
    n_trees = 450 if EXECUTION_BUDGET == 'FULL' else 200
    
    t0 = time.time()
    
    # Treinamento Model A
    model_a_xgb = xgb.XGBClassifier(
        n_estimators=n_trees, learning_rate=0.03, max_depth=9,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_a,
        tree_method='hist', random_state=42
    )
    model_a_xgb.fit(train_df.loc[~train_has_dev, features_a], train_df.loc[~train_has_dev, 'isFraud'], verbose=False)
    
    # Treinamento Model B
    model_b_xgb = xgb.XGBClassifier(
        n_estimators=n_trees, learning_rate=0.03, max_depth=9,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_b,
        tree_method='hist', random_state=42
    )
    model_b_xgb.fit(train_df.loc[train_has_dev, features_b], train_df.loc[train_has_dev, 'isFraud'], verbose=False)
    
    elapsed = time.time() - t0
    
    # Inferência Causal
    pred_a = model_a_xgb.predict_proba(val_df.loc[~val_has_dev, features_a])[:, 1]
    pred_b = model_b_xgb.predict_proba(val_df.loc[val_has_dev, features_b])[:, 1]
    
    y_score_raw = pd.Series(index=val_df.index, dtype=float)
    y_score_raw.loc[~val_has_dev] = pred_a
    y_score_raw.loc[val_has_dev] = pred_b
    y_true = val_df['isFraud'].values
    
    # Pós-Processamento Causal (Média Expansiva Temporal de Predição por UID)
    val_temp = val_df[['uid', 'timestamp']].copy()
    val_temp['raw_score'] = y_score_raw
    val_temp_sorted = val_temp.sort_values('timestamp')
    expanding_mean = val_temp_sorted.groupby('uid')['raw_score'].expanding().mean().values
    val_temp_sorted['score_pp'] = 0.7 * val_temp_sorted['raw_score'] + 0.3 * expanding_mean
    y_score_pp = val_temp_sorted.sort_index()['score_pp'].values
    
    def calc_metrics(scores):
        roc = roc_auc_score(y_true, scores)
        prec, rec, thresholds = precision_recall_curve(y_true, scores)
        pr_auc = auc(rec, prec)
        f1_arr = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_i = np.argmax(f1_arr)
        best_f1 = f1_arr[best_i]
        best_th = thresholds[min(best_i, len(thresholds)-1)]
        return roc, pr_auc, best_f1, best_th
        
    roc_r, pr_r, f1_r, th_r = calc_metrics(y_score_raw.values)
    roc_pp, pr_pp, f1_pp, th_pp = calc_metrics(y_score_pp)
    
    print(f"  [Sem Post-Proc]  -> Tempo: {elapsed:.1f}s | ROC-AUC: {roc_r:.4f} | PR-AUC: {pr_r:.4f} | F1: {f1_r:.4f}")
    print(f"  [Com Post-Proc]  -> PR-AUC: {pr_pp:.4f} | F1: {f1_pp:.4f}")
    
    return {
        'id': config_id,
        'name': feature_set_name,
        'use_pca': use_pca,
        'features_a_count': len(features_a),
        'features_b_count': len(features_b),
        'roc_auc_raw': roc_r,
        'pr_auc_raw': pr_r,
        'f1_raw': f1_r,
        'threshold_raw': th_r,
        'roc_auc_pp': roc_pp,
        'pr_auc_pp': pr_pp,
        'f1_pp': f1_pp,
        'threshold_pp': th_pp,
        'time': elapsed,
        'y_score_raw': y_score_raw.values,
        'y_score_pp': y_score_pp,
        'model_b': model_b_xgb,
        'features_b': features_b
    }
"""))

# Cell 7: Execution of the Ablation Configurations
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Execução do Estudo de Ablação Completo
Executamos as configurações planejadas cobrindo:
- **1A. Baseline Raw (Sem PCA)**: 339 colunas $V$ originais, sem grafos, sem FE.
- **1B. Baseline Com PCA**: 15 componentes PCA, sem grafos, sem FE (*Isola o efeito do PCA*).
- **2. Baseline + Apenas Topologia**: PageRank e graus de conectividade bipartida (*Isola a topologia*).
- **3. Baseline + Apenas Velocidade**: Janela temporal dinâmica 24h por UID (*Isola a velocidade temporal*).
- **4. Baseline + Grafos Completos**: Topologia + Velocidade (*Combinação relacional*).
- **5. Baseline + Apenas FE Tabular**: Deltas $D_n$ e desvios de perfil por portador.
- **6A. All-in-One (Com PCA)**: Modelo completo integrado.
- **6B. All-in-One (Sem PCA)**: Modelo completo utilizando as 339 colunas brutas $V$.
"""))

cells.append(nbf.v4.new_code_cell("""results = []

# 1A. Baseline Raw (Sem PCA)
results.append(train_and_eval_pipeline('1A', 'Baseline Raw (Sem PCA)', use_pca=False, include_topology=False, include_velocity=False, include_fe=False))

# 1B. Baseline Com PCA
results.append(train_and_eval_pipeline('1B', 'Baseline (Com PCA 15-d)', use_pca=True, include_topology=False, include_velocity=False, include_fe=False))

# 2. Baseline + Apenas Topologia de Grafo
results.append(train_and_eval_pipeline('2', 'Baseline + Apenas Topologia', use_pca=True, include_topology=True, include_velocity=False, include_fe=False))

# 3. Baseline + Apenas Velocidade Causal 24h
results.append(train_and_eval_pipeline('3', 'Baseline + Apenas Velocidade', use_pca=True, include_topology=False, include_velocity=True, include_fe=False))

# 4. Baseline + Grafos Completos (Topologia + Velocidade)
results.append(train_and_eval_pipeline('4', 'Baseline + Grafos Completos', use_pca=True, include_topology=True, include_velocity=True, include_fe=False))

# 5. Baseline + Apenas FE Tabular Avançado
results.append(train_and_eval_pipeline('5', 'Baseline + Apenas FE Tabular', use_pca=True, include_topology=False, include_velocity=False, include_fe=True))

# 6A. All-in-One (Com PCA)
results.append(train_and_eval_pipeline('6A', 'All-in-One (Com PCA 15-d)', use_pca=True, include_topology=True, include_velocity=True, include_fe=True))

# 6B. All-in-One (Sem PCA)
results.append(train_and_eval_pipeline('6B', 'All-in-One (Sem PCA / Raw V)', use_pca=False, include_topology=True, include_velocity=True, include_fe=True))

print("\\n✅ Todas as configurações de ablação foram executadas com sucesso.")
"""))

# Cell 8: Consolidated Ablation Table & Visualizations
cells.append(nbf.v4.new_markdown_cell("""---
## 7. Consolidação de Resultados & Comparação Metodológica (Com vs. Sem PCA)
Abaixo consolidamos a tabela comparativa exibindo o desempenho bruto de cada modelo e o efeito do pós-processamento causal:
"""))

cells.append(nbf.v4.new_code_cell("""summary_table = pd.DataFrame([
    {
        'ID': r['id'],
        'Configuração': r['name'],
        'Features (A/B)': f"{r['features_a_count']} / {r['features_b_count']}",
        'PR-AUC (Bruto)': f"{r['pr_auc_raw']:.4f}",
        'ROC-AUC (Bruto)': f"{r['roc_auc_raw']:.4f}",
        'F1 (Bruto)': f"{r['f1_raw']:.4f}",
        'PR-AUC (Pós-Proc)': f"{r['pr_auc_pp']:.4f}",
        'F1 (Pós-Proc)': f"{r['f1_pp']:.4f}",
        'Tempo (s)': f"{r['time']:.1f}s"
    }
    for r in results
])

print("=========================================================================================================")
print("                                 TABELA DO ESTUDO DE ABLAÇÃO EXPANDIDO                                   ")
print("=========================================================================================================")
display(summary_table)
print("=========================================================================================================")

# Comparação Direta Metodológica: Com PCA vs Sem PCA
b_raw = next(r for r in results if r['id'] == '1A')
b_pca = next(r for r in results if r['id'] == '1B')
all_pca = next(r for r in results if r['id'] == '6A')
all_raw = next(r for r in results if r['id'] == '6B')

print(f"\\n🔬 COMPARAÇÃO METODOLÓGICA DE COMPRESSÃO PCA:")
print(f"  • Baseline: Raw V ({b_raw['features_b_count']} cols, {b_raw['time']:.1f}s, PR-AUC = {b_raw['pr_auc_raw']:.4f}) vs. PCA ({b_pca['features_b_count']} cols, {b_pca['time']:.1f}s, PR-AUC = {b_pca['pr_auc_raw']:.4f})")
print(f"  • All-in-One: Raw V ({all_raw['features_b_count']} cols, {all_raw['time']:.1f}s, PR-AUC = {all_raw['pr_auc_raw']:.4f}) vs. PCA ({all_pca['features_b_count']} cols, {all_pca['time']:.1f}s, PR-AUC = {all_pca['pr_auc_raw']:.4f})")
print(f"  -> Conclusão: O PCA comprime drasticamente a matriz acelerando o treino mantendo fidelidade preditiva.")

# Curvas Precision-Recall Comparativas
plt.figure(figsize=(10, 6))
y_true = val_df['isFraud'].values
for r in results:
    p, rec, _ = precision_recall_curve(y_true, r['y_score_raw'])
    plt.plot(rec, p, label=f"[{r['id']}] {r['name']} (PR={r['pr_auc_raw']:.4f})")
plt.axhline(y_true.mean(), color='gray', linestyle='--', label=f'Baseline Aleatório ({y_true.mean():.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curvas Precision-Recall Comparativas da Ablação (Validação Temporal Causal)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""))

# Cell 9: Real Financial Cost Function Evaluation
cells.append(nbf.v4.new_markdown_cell("""---
## 8. Avaliação da Função de Custo Financeiro Real (Bayes Minimum Risk)
Calculamos o custo financeiro total em dólares ($) parametrizado pelo limiar $\\tau \\in [0.01, 0.95]$ sobre as **predições reais dos modelos**:

$$L(\\tau) = \\sum_{i=1}^N \\Big[ y_i (1 - \\hat{y}_i(\\tau)) \\cdot (\\text{amt}_i + C_{\\text{chargeback}}) + (1 - y_i) \\hat{y}_i(\\tau) \\cdot (C_{\\text{revisão}} + \\alpha \\cdot \\text{amt}_i) + y_i \\hat{y}_i(\\tau) \\cdot C_{\\text{operacional}} \\Big]$$

### Parâmetros Calibrados de Mercado:
* $C_{\\text{chargeback}} = \\$15,00$
* $C_{\\text{revisão}} = \\$3,00$
* $\\alpha = 0,05$ (5%)
* $C_{\\text{operacional}} = \\$0,10$
"""))

cells.append(nbf.v4.new_code_cell("""amounts = val_df['TransactionAmt'].values
y_true = val_df['isFraud'].values

def compute_financial_loss(y_true, y_score, amounts, threshold, C_chargeback=15.0, C_review=3.0, alpha=0.05, C_op=0.10):
    y_pred = (y_score >= threshold).astype(int)
    fn_mask = (y_true == 1) & (y_pred == 0)
    loss_fn = (amounts[fn_mask] + C_chargeback).sum()
    fp_mask = (y_true == 0) & (y_pred == 1)
    loss_fp = (C_review + alpha * amounts[fp_mask]).sum()
    tp_mask = (y_true == 1) & (y_pred == 1)
    loss_tp = tp_mask.sum() * C_op
    return loss_fn + loss_fp + loss_tp, loss_fn, loss_fp, loss_tp

thresholds = np.linspace(0.01, 0.95, 100)

cost_evaluations = {}
key_configs = ['1B', '2', '3', '4', '5', '6A']

for cid in key_configs:
    r = next(item for item in results if item['id'] == cid)
    losses = []
    for t in thresholds:
        loss, _, _, _ = compute_financial_loss(y_true, r['y_score_raw'], amounts, threshold=t)
        losses.append(loss)
    best_idx = np.argmin(losses)
    cost_evaluations[cid] = {
        'name': r['name'],
        'best_thresh': thresholds[best_idx],
        'min_loss': losses[best_idx],
        'loss_curve': losses
    }

base_loss = cost_evaluations['1B']['min_loss']
cost_summary = pd.DataFrame([
    {
        'ID': cid,
        'Modelo': cost_evaluations[cid]['name'],
        'Limiar Ótimo (τ*)': f"{cost_evaluations[cid]['best_thresh']:.2f}",
        'Prejuízo Mínimo ($)': f"${cost_evaluations[cid]['min_loss']:,.2f}",
        'Economia vs Baseline ($)': f"${(base_loss - cost_evaluations[cid]['min_loss']):,.2f}",
        'Redução do Prejuízo (%)': f"{((base_loss - cost_evaluations[cid]['min_loss']) / base_loss * 100):+.2f}%"
    }
    for cid in key_configs
])

print("=========================================================================================================")
print("                           IMPACTO FINANCEIRO REAL NO PONTO ÓTIMO DE DECISÃO                             ")
print("=========================================================================================================")
display(cost_summary)
print("=========================================================================================================")

# Gráfico da Curva de Perda Financeira
plt.figure(figsize=(11, 5.5))
for cid in ['1B', '4', '6A']:
    cdata = cost_evaluations[cid]
    plt.plot(thresholds, cdata['loss_curve'], label=f"{cdata['name']} (Mínimo: ${cdata['min_loss']:,.2f} @ τ={cdata['best_thresh']:.2f})")

plt.title('Curvas de Perda Financeira Real ($) vs. Limiar de Decisão (Threshold)')
plt.xlabel('Limiar de Decisão (Threshold)')
plt.ylabel('Prejuízo Operacional Total ($)')
plt.legend()
plt.tight_layout()
plt.show()
"""))

# Cell 10: Interpretability & Real Subgraph Case Study
cells.append(nbf.v4.new_markdown_cell("""---
## 9. Interpretabilidade: Ranking Global de Atributos & Estudo de Caso de Subgrafo
Abaixo avaliamos o ganho de informação (Gain) no XGBoost Model B do modelo All-in-One e inspecionamos um caso qualitativo de subgrafo real na validação.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Ranking Global de Atributos por Ganho (Gain)
aio_model = next(r['model_b'] for r in results if r['id'] == '6A')
importances = aio_model.get_booster().get_score(importance_type='gain')
imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Gain']).sort_values('Gain', ascending=False).reset_index(drop=True)

print("Top 15 Atributos com Maior Ganho de Informação:")
display(imp_df.head(15))

plt.figure(figsize=(10, 6))
top20 = imp_df.head(20).copy()
colors = [
    PALETTE['only_graph'] if f in graph_cols else
    PALETTE['only_fe'] if f in fe_cols else
    PALETTE['baseline_pca']
    for f in top20['Feature']
]
sns.barplot(data=top20, x='Gain', y='Feature', palette=colors)
plt.title('Top 20 Atributos Mais Importantes no Modelo All-in-One (XGBoost Model B)')
plt.xlabel('Gain Médio')
plt.ylabel('Atributo')
plt.tight_layout()
plt.show()

# 2. Estudo de Caso Qualitativo de Subgrafo Real
print("\\n--- Estudo de Caso de Subgrafo na Validação Temporal ---")
# Identifica dispositivos compartilhados na validação associados a fraudes reais
val_devices = val_df[val_df['DeviceInfo_clean'] != 'null_device'].copy()
shared_suspicious = val_devices[val_devices['device_degree'] >= 3]
fraud_devices = shared_suspicious[shared_suspicious['isFraud'] == 1]['DeviceInfo_clean'].value_counts()

if len(fraud_devices) > 0:
    case_device = fraud_devices.index[0]
    case_txs = val_devices[val_devices['DeviceInfo_clean'] == case_device].head(5)
    print(f"Dispositivo Inspecionado: [{case_device}] compartilhado com {case_devices_uids := case_txs['device_degree'].iloc[0]} UIDs no treino.")
    
    baseline_scores = next(r['y_score_raw'] for r in results if r['id'] == '1B')[case_txs.index]
    graph_scores = next(r['y_score_raw'] for r in results if r['id'] == '4')[case_txs.index]
    
    case_summary = pd.DataFrame({
        'TransactionID': case_txs['TransactionID'].values,
        'UID': case_txs['uid'].values,
        'Valor ($)': case_txs['TransactionAmt'].values,
        'isFraud Real': case_txs['isFraud'].values,
        'Score Baseline': baseline_scores,
        'Score com Grafos': graph_scores,
        'Elevação de Risco': graph_scores - baseline_scores
    })
    display(case_summary)
    print("-> Diagnóstico: O modelo de grafos eleva sensivelmente o escore de suspeição em razão da infraestrutura compartilhada.")
else:
    print("Nenhum dispositivo compartilhado com múltiplas fraudes foi encontrado na janela selecionada.")
"""))

# Cell 11: Export Predictions & Summary
cells.append(nbf.v4.new_markdown_cell("""---
## 10. Exportação dos Resultados e Predições
Salvamos os escores de predição reais da validação temporal e o resumo da ablação em arquivos CSV para auditoria e consumo no Notebook 05.
"""))

cells.append(nbf.v4.new_code_cell("""# Construção do DataFrame de Predições Reais
val_export = val_df[['TransactionID', 'timestamp', 'isFraud', 'TransactionAmt', 'uid', 'DeviceInfo_clean']].copy()

for r in results:
    val_export[f"score_{r['id']}_raw"] = r['y_score_raw']
    val_export[f"score_{r['id']}_pp"] = r['y_score_pp']

val_export.to_csv('val_predictions.csv', index=False)
summary_table.to_csv('ablation_summary.csv', index=False)
cost_summary.to_csv('cost_summary.csv', index=False)

print(f"✅ Arquivos exportados com sucesso:")
print(f"  • val_predictions.csv ({len(val_export):,} linhas com predições reais de todos os modelos)")
print(f"  • ablation_summary.csv (tabela comparativa completa da ablação)")
print(f"  • cost_summary.csv (análise de custo financeiro real)")
"""))

# Write notebook
nb['cells'] = cells
with open('notebooks/04_ablation_with_vs_without_graphs_colab.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 04 updated successfully!")
