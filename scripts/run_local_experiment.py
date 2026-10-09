import os
import gc
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, roc_curve
import xgboost as xgb

SAMPLE_FRACTION = 0.20
N_ESTIMATORS = 300
N_JOBS = 8

print(f"=== INICIANDO EXPERIMENTO LOCAL (20% DO DATASET / 16GB RAM) ===")
print(f"Amostra: {SAMPLE_FRACTION:.0%} | Estimadores: {N_ESTIMATORS} | Threads: {N_JOBS}")

# 1. Carregamento dos dados
DATA_DIR = 'data/ieee/raw' if os.path.exists('data/ieee/raw') else '.'
train_trans_path = os.path.join(DATA_DIR, 'train_transaction.csv')
train_id_path = os.path.join(DATA_DIR, 'train_identity.csv')

def reduce_mem_usage(df, verbose=False):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            elif pd.api.types.is_float_dtype(df[col]):
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

t0 = time.time()
print("Carregando transações e identidade...")
df_trans = pd.read_csv(train_trans_path)
df_trans = reduce_mem_usage(df_trans)

df_id = pd.read_csv(train_id_path)
df_id = reduce_mem_usage(df_id)

df = df_trans.merge(df_id, on='TransactionID', how='left')
del df_trans, df_id
gc.collect()

df = df.rename(columns={"TransactionDT": "timestamp"})
print(f"Dataset completo: {len(df):,} linhas.")

if SAMPLE_FRACTION is not None and 0.0 < SAMPLE_FRACTION < 1.0:
    print(f"Subamostrando para {SAMPLE_FRACTION:.0%}...")
    df = df.sample(frac=SAMPLE_FRACTION, random_state=42)
    df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Dataset em uso: {len(df):,} linhas carregadas em {time.time()-t0:.1f}s.")

# Split Temporal Causal (80% treino / 20% validação)
split_idx = int(len(df) * 0.8)
train_df_raw = df.iloc[:split_idx].copy().reset_index(drop=True)
val_df_raw = df.iloc[split_idx:].copy().reset_index(drop=True)
del df
gc.collect()

print(f"Treino   : {len(train_df_raw):,} linhas (Fraude: {train_df_raw['isFraud'].mean():.3%})")
print(f"Validação: {len(val_df_raw):,} linhas (Fraude: {val_df_raw['isFraud'].mean():.3%})")

# 2. Features Tabulares, UID & PCA
train_df = train_df_raw.copy()
val_df = val_df_raw.copy()

for d_col in ['D1', 'D2', 'D10', 'D15']:
    if d_col in train_df.columns:
        train_df[f'{d_col}n'] = (train_df['timestamp'] / 86400.0) - train_df[d_col]
        val_df[f'{d_col}n'] = (val_df['timestamp'] / 86400.0) - val_df[d_col]

def build_magic_uid(df):
    c1 = df['card1'].fillna(0).astype(int).astype(str)
    addr = df['addr1'].fillna(0).astype(int).astype(str)
    day = df['timestamp'] / 86400.0
    d1_anchor = (day - df['D1'].fillna(0)).astype(int).astype(str)
    return c1 + "_" + addr + "_" + d1_anchor

train_df['uid'] = build_magic_uid(train_df)
val_df['uid'] = build_magic_uid(val_df)
print(f"UIDs únicos no treino: {train_df['uid'].nunique():,}")

# Agregações de Grupo por UID
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

for cat_col in ['card1', 'P_emaildomain']:
    if cat_col in train_df.columns:
        counts = train_df[cat_col].value_counts().to_dict()
        train_df[f'{cat_col}_count'] = train_df[cat_col].map(counts).fillna(0)
        val_df[f'{cat_col}_count'] = val_df[cat_col].map(counts).fillna(0)

# PCA nas variáveis V
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

train_has_dev = train_df['DeviceType'].notnull()
val_has_dev = val_df['DeviceType'].notnull()

cat_cols = [c for c in train_df.select_dtypes(include=['object', 'category']).columns if c not in ['uid', 'TransactionID']]
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    val_df[col] = val_df[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

# 3. Features de Grafo & Velocidade Causal 24h
print("Construindo Grafo Bipartido e PageRank...")
train_df['DeviceInfo_clean'] = train_df_raw['DeviceInfo'].fillna('null_device').astype(str)
val_df['DeviceInfo_clean'] = val_df_raw['DeviceInfo'].fillna('null_device').astype(str)

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

dev_distinct_uids = train_dev_edges.groupby('DeviceInfo_clean')['uid'].nunique().to_dict()
uid_distinct_devs = train_dev_edges.groupby('uid')['DeviceInfo_clean'].nunique().to_dict()

train_df['device_degree'] = train_df['DeviceInfo_clean'].map(dev_distinct_uids).fillna(0)
val_df['device_degree'] = val_df['DeviceInfo_clean'].map(dev_distinct_uids).fillna(0)

train_df['uid_degree'] = train_df['uid'].map(uid_distinct_devs).fillna(0)
val_df['uid_degree'] = val_df['uid'].map(uid_distinct_devs).fillna(0)

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

print("Calculando velocidade causal 24h por UID...")
train_df = compute_causal_velocity(train_df)
val_df = compute_causal_velocity(val_df)
print("Features estruturais construídas com sucesso.")

# 4. Pipeline de Treinamento da Ablação
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
    t_start = time.time()
    drop_list = list(base_drop)
    if use_pca:
        drop_list += raw_v_cols
    else:
        drop_list += pca_cols
    if not include_topology:
        drop_list += topology_cols
    if not include_velocity:
        drop_list += velocity_cols
    if not include_fe:
        drop_list += fe_cols
        
    features_a = [c for c in train_df.columns if c not in drop_list and not c.startswith('id_') and c != 'DeviceType']
    features_b = [c for c in train_df.columns if c not in drop_list]
    
    scale_pos_a = (~train_df[~train_has_dev]['isFraud'].astype(bool)).sum() / max(train_df[~train_has_dev]['isFraud'].sum(), 1)
    scale_pos_b = (~train_df[train_has_dev]['isFraud'].astype(bool)).sum() / max(train_df[train_has_dev]['isFraud'].sum(), 1)
    
    # Model A
    model_a_xgb = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=0.03, max_depth=9,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_a,
        tree_method='hist', n_jobs=N_JOBS, random_state=42
    )
    model_a_xgb.fit(train_df.loc[~train_has_dev, features_a], train_df.loc[~train_has_dev, 'isFraud'], verbose=False)
    
    # Model B
    model_b_xgb = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=0.03, max_depth=9,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_b,
        tree_method='hist', n_jobs=N_JOBS, random_state=42
    )
    model_b_xgb.fit(train_df.loc[train_has_dev, features_b], train_df.loc[train_has_dev, 'isFraud'], verbose=False)
    
    elapsed = time.time() - t_start
    
    pred_a = model_a_xgb.predict_proba(val_df.loc[~val_has_dev, features_a])[:, 1]
    pred_b = model_b_xgb.predict_proba(val_df.loc[val_has_dev, features_b])[:, 1]
    
    y_score_raw = pd.Series(index=val_df.index, dtype=float)
    y_score_raw.loc[~val_has_dev] = pred_a
    y_score_raw.loc[val_has_dev] = pred_b
    y_true = val_df['isFraud'].values
    
    # Pós-processamento Causal
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
    
    print(f"[{config_id:>2}] {feature_set_name:<30} ({len(features_a)}/{len(features_b)} cols) | {elapsed:>4.1f}s | PR: {pr_r:.4f} (pp: {pr_pp:.4f}) | ROC: {roc_r:.4f}")
    
    return {
        'id': config_id,
        'name': feature_set_name,
        'use_pca': use_pca,
        'cols_a': len(features_a),
        'cols_b': len(features_b),
        'roc_raw': roc_r,
        'pr_raw': pr_r,
        'f1_raw': f1_r,
        'th_raw': th_r,
        'roc_pp': roc_pp,
        'pr_pp': pr_pp,
        'f1_pp': f1_pp,
        'th_pp': th_pp,
        'time': elapsed,
        'y_score_raw': y_score_raw.values,
        'y_score_pp': y_score_pp
    }

print("\n--- EXECUTANDO TABELA DE ABLAÇÃO ---")
results = []
results.append(train_and_eval_pipeline('1A', 'Baseline Raw (Sem PCA)', use_pca=False, include_topology=False, include_velocity=False, include_fe=False))
results.append(train_and_eval_pipeline('1B', 'Baseline (Com PCA 15-d)', use_pca=True, include_topology=False, include_velocity=False, include_fe=False))
results.append(train_and_eval_pipeline('2',  'Baseline + Apenas Topologia', use_pca=True, include_topology=True, include_velocity=False, include_fe=False))
results.append(train_and_eval_pipeline('3',  'Baseline + Apenas Velocidade', use_pca=True, include_topology=False, include_velocity=True, include_fe=False))
results.append(train_and_eval_pipeline('4',  'Baseline + Grafos Completos', use_pca=True, include_topology=True, include_velocity=True, include_fe=False))
results.append(train_and_eval_pipeline('5',  'Baseline + Apenas FE Tabular', use_pca=True, include_topology=False, include_velocity=False, include_fe=True))
results.append(train_and_eval_pipeline('6A', 'All-in-One (Com PCA 15-d)', use_pca=True, include_topology=True, include_velocity=True, include_fe=True))
results.append(train_and_eval_pipeline('6B', 'All-in-One (Sem PCA / Raw V)', use_pca=False, include_topology=True, include_velocity=True, include_fe=True))

# 5. Salvar Resultados & Predições Reais
os.makedirs('results', exist_ok=True)

summary_table = pd.DataFrame([
    {
        'ID': r['id'],
        'Configuracao': r['name'],
        'Features_B': r['cols_b'],
        'PR_AUC_Raw': r['pr_raw'],
        'ROC_AUC_Raw': r['roc_raw'],
        'F1_Raw': r['f1_raw'],
        'PR_AUC_PP': r['pr_pp'],
        'F1_PP': r['f1_pp'],
        'Tempo_s': r['time']
    }
    for r in results
])
summary_table.to_csv('results/ablation_summary.csv', index=False)

# Exportação do val_predictions.csv
val_export = val_df[['TransactionID', 'timestamp', 'isFraud', 'TransactionAmt', 'uid', 'DeviceInfo_clean']].copy()
for r in results:
    val_export[f"score_{r['id']}_raw"] = r['y_score_raw']
    val_export[f"score_{r['id']}_pp"] = r['y_score_pp']
val_export.to_csv('results/val_predictions.csv', index=False)
val_export.to_csv('val_predictions.csv', index=False)

# 6. Avaliação da Função de Custo Financeiro Real
amounts = val_df['TransactionAmt'].values
y_true = val_df['isFraud'].values

def compute_financial_loss(y_true, y_score, amounts, threshold, C_cb=15.0, C_rev=3.0, alpha=0.05, C_op=0.10):
    y_pred = (y_score >= threshold).astype(int)
    fn = ((amounts + C_cb) * (y_true == 1) * (y_pred == 0)).sum()
    fp = ((C_rev + alpha * amounts) * (y_true == 0) * (y_pred == 1)).sum()
    tp = (C_op * (y_true == 1) * (y_pred == 1)).sum()
    return fn + fp + tp

thresholds = np.linspace(0.01, 0.95, 100)
cost_records = []
base_loss = None

for cid in ['1A', '1B', '2', '3', '4', '5', '6A', '6B']:
    r = next(item for item in results if item['id'] == cid)
    losses = [compute_financial_loss(y_true, r['y_score_raw'], amounts, t) for t in thresholds]
    best_i = np.argmin(losses)
    min_l = losses[best_i]
    best_th = thresholds[best_i]
    if cid == '1B':
        base_loss = min_l
    cost_records.append({
        'id': cid,
        'name': r['name'],
        'min_loss': min_l,
        'best_th': best_th,
        'losses': losses
    })

cost_table = pd.DataFrame([
    {
        'ID': cr['id'],
        'Modelo': cr['name'],
        'Limiar_Otimo': cr['best_th'],
        'Prejuizo_Minimo': cr['min_loss'],
        'Economia_vs_Base': base_loss - cr['min_loss'],
        'Reducao_Pct': (base_loss - cr['min_loss']) / base_loss * 100
    }
    for cr in cost_records
])
cost_table.to_csv('results/cost_summary.csv', index=False)

print("\n=== RESUMO CONSOLIDADO DA ABLAÇÃO REAL ===")
print(summary_table.to_string(index=False))

print("\n=== RESUMO FINANCEIRO REAL (BAYES MINIMUM RISK) ===")
print(cost_table[['ID', 'Modelo', 'Limiar_Otimo', 'Prejuizo_Minimo', 'Economia_vs_Base', 'Reducao_Pct']].to_string(index=False))

print("\n✅ Execução concluída com sucesso! Arquivos gerados:")
print("  • results/ablation_summary.csv")
print("  • results/cost_summary.csv")
print("  • results/val_predictions.csv (e ./val_predictions.csv)")
