#!/usr/bin/env python3
"""
scripts/run_100percent_pipeline.py

Pipeline experimental completo sobre 100% do IEEE-CIS (590.540 transações) para TESTAR o
uso de atributos de grafo na detecção de fraude (o objetivo não é provar superioridade).

Princípios de projeto (todos verificáveis no código):
  * Partição temporal 80/20 por ordem de TransactionDT; nada da validação é usado para ajustar
    transformações (PCA, medianas, codificação categórica) nem para treinar.
  * TODOS os atributos derivados são causais (não antecipativos): para a transação i só usam
    transações com índice temporal < i (ou a própria i, quando explicitado). Por isso são
    calculados uma única vez sobre a linha do tempo completa, sem assimetria treino/validação.
      - agregações por UID: média/desvio das transações ANTERIORES do mesmo UID;
      - contagens de frequência: janela retrospectiva de 30 dias;
      - velocidade: janela retrospectiva de 24 h;
      - grafo: graus calculados incrementalmente; PageRank recalculado em snapshots semanais
        usando apenas arestas observadas antes do início da semana.
  * Dois grafos bipartidos portador (UID) <-> dispositivo:
      - "fingerprint": DeviceInfo | id_30 (SO) | id_31 (navegador) | id_33 (resolução);
      - "DeviceInfo bruto" (controle): apenas a string de modelo/SO, que forma hubs
        gigantes (ex.: "Windows"), e por isso é avaliado separadamente.
  * O limiar operacional é escolhido na 1ª metade temporal da validação (calibração) e o custo é
    reportado na 2ª metade (teste), evitando otimizar e avaliar no mesmo conjunto.
  * Incerteza: 3 sementes de treino + bootstrap pareado por blocos de dia (IC 95%).

Saídas em results/100percent/ (as figuras são geradas por scripts/generate_all_single_figures.py).
"""

import os
import gc
import json
import time
import platform
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import psutil
import xgboost as xgb
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

warnings.filterwarnings('ignore')

OUT_DIR = 'results/100percent'
os.makedirs(OUT_DIR, exist_ok=True)

N_ESTIMATORS = 300
N_JOBS = int(os.environ.get('N_JOBS', 8))
SEEDS = [42, 7, 2024][: int(os.environ.get('N_SEEDS', 3))]
N_BOOT = int(os.environ.get('N_BOOT', 500))
ALERT_BUDGET = 0.03          # fração de transações sinalizadas na análise por tipologia
PR_SNAPSHOT_DAYS = 7         # periodicidade dos snapshots de PageRank
COST = dict(C_cb=15.0, C_rev=3.0, alpha=0.05, C_op=0.10)
THRESHOLDS = np.round(np.linspace(0.01, 0.99, 99), 2)


def ram_gb():
    return psutil.Process().memory_info().rss / 1024 ** 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} | {ram_gb():5.2f} GB] {msg}", flush=True)


t_global = time.time()
meta = {'seeds': SEEDS, 'n_estimators': N_ESTIMATORS, 'n_boot': N_BOOT,
        'alert_budget': ALERT_BUDGET, 'pagerank_snapshot_days': PR_SNAPSHOT_DAYS, 'cost': COST,
        'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                     'xgboost': xgb.__version__, 'networkx': nx.__version__,
                     'sklearn': sklearn.__version__}}

# ---------------------------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------------------------
DATA_DIR = 'data/ieee/raw'


def load(path):
    d = pd.read_csv(path)
    for c in d.select_dtypes(include=['float64']).columns:
        d[c] = d[c].astype(np.float32)
    for c in d.select_dtypes(include=['int64']).columns:
        d[c] = pd.to_numeric(d[c], downcast='integer')
    return d


log("Carregando train_transaction + train_identity...")
df = load(os.path.join(DATA_DIR, 'train_transaction.csv')).merge(
    load(os.path.join(DATA_DIR, 'train_identity.csv')), on='TransactionID', how='left')
df = df.rename(columns={'TransactionDT': 'timestamp'})
df = df.sort_values(['timestamp', 'TransactionID'], kind='stable').reset_index(drop=True)
N = len(df)
log(f"Dataset: {N:,} linhas x {df.shape[1]} colunas")

# ---------------------------------------------------------------------------------------------
# 2. PARTIÇÃO TEMPORAL
# ---------------------------------------------------------------------------------------------
split_idx = int(N * 0.8)
is_train = np.zeros(N, dtype=bool)
is_train[:split_idx] = True
is_val = ~is_train
val_pos = np.where(is_val)[0]
half_idx = val_pos[len(val_pos) // 2]
is_calib = is_val & (np.arange(N) < half_idx)   # 1ª metade temporal da validação
is_test = is_val & (np.arange(N) >= half_idx)   # 2ª metade temporal da validação

ts = df['timestamp'].values.astype(np.int64)
y_all = df['isFraud'].values.astype(np.int8)
amt_all = df['TransactionAmt'].values.astype(np.float64)

meta['split'] = {
    'n_train': int(is_train.sum()), 'n_val': int(is_val.sum()),
    'n_calib': int(is_calib.sum()), 'n_test': int(is_test.sum()),
    'fraud_train': int(y_all[is_train].sum()), 'fraud_val': int(y_all[is_val].sum()),
    'fraud_calib': int(y_all[is_calib].sum()), 'fraud_test': int(y_all[is_test].sum()),
    'amt_train': float(amt_all[is_train].sum()), 'amt_val': float(amt_all[is_val].sum()),
    'fraud_amt_train': float(amt_all[is_train & (y_all == 1)].sum()),
    'fraud_amt_val': float(amt_all[is_val & (y_all == 1)].sum()),
    'ts_train': [int(ts[0]), int(ts[split_idx - 1])], 'ts_val': [int(ts[split_idx]), int(ts[-1])],
    'ts_test_start': int(ts[half_idx]),
}
log(f"Split: {meta['split']}")

# ---------------------------------------------------------------------------------------------
# 3. ENGENHARIA DE ATRIBUTOS (todas causais)
# ---------------------------------------------------------------------------------------------
day_float = ts / 86400.0

# 3.1 Normalização de D (coordenada de calendário)
for d_col in ['D1', 'D2', 'D10', 'D15']:
    df[f'{d_col}n'] = (day_float - df[d_col]).astype(np.float32)

# 3.2 UID composto
df['uid'] = (df['card1'].fillna(0).astype(int).astype(str) + '_' +
             df['addr1'].fillna(0).astype(int).astype(str) + '_' +
             (day_float - df['D1'].fillna(0)).astype(int).astype(str))
uid = df['uid']


def group_window_count(keys, times, window):
    """Nº de transações com a mesma chave em (t - window, t], incluindo a atual. Causal."""
    out = np.full(len(keys), np.nan, dtype=np.float32)
    for _, idx in keys.groupby(keys, sort=False).indices.items():
        t = times[idx]
        out[idx] = np.arange(len(idx)) - np.searchsorted(t, t - window, side='left') + 1
    return out


def past_stats(x, key):
    """Média, desvio e contagem das observações ANTERIORES da mesma chave (exclui a atual)."""
    x = x.astype('float64')
    valid = x.notna().astype('float64')
    xf = x.fillna(0.0)
    n_prev = valid.groupby(key).cumsum() - valid
    s_prev = xf.groupby(key).cumsum() - xf
    ss_prev = (xf ** 2).groupby(key).cumsum() - xf ** 2
    mean = s_prev / n_prev.where(n_prev > 0)
    var = (ss_prev - n_prev * mean ** 2) / (n_prev - 1).where(n_prev > 1)
    return mean, np.sqrt(var.clip(lower=0)), n_prev


# 3.3 Agregações comportamentais por UID (passado apenas)
log("Agregações causais por UID...")
for col in ['TransactionAmt', 'D15n', 'C1', 'C13']:
    mean, std, n_prev = past_stats(df[col], uid)
    df[f'{col}_to_mean_uid'] = (df[col] / (mean + 1e-5)).astype(np.float32)
    df[f'{col}_to_std_uid'] = ((df[col] - mean) / (std + 1e-5)).astype(np.float32)
    if col == 'TransactionAmt':
        df['uid_prev_tx_count'] = n_prev.astype(np.float32)

# 3.4 Contagens de frequência em janela retrospectiva de 30 dias
for cat in ['card1', 'P_emaildomain']:
    df[f'{cat}_count_30d'] = group_window_count(df[cat], ts, 30 * 86400)

# 3.5 Velocidade causal 24 h (cronometrada)
log("Velocidade causal 24h...")
t0 = time.time()
df['uid_tx_count_24h'] = group_window_count(uid, ts, 86400)
df['uid_secs_since_prev'] = df.groupby('uid')['timestamp'].diff().astype(np.float32)
meta['velocity_seconds'] = round(time.time() - t0, 2)
log(f"  velocidade em {meta['velocity_seconds']} s")

# 3.6 Grafos bipartidos causais
fp_parts = [df[c].astype(str).where(df[c].notna(), '?') for c in ['DeviceInfo', 'id_30', 'id_31', 'id_33']]
dev_fp = (fp_parts[0] + '|' + fp_parts[1] + '|' + fp_parts[2] + '|' + fp_parts[3]).where(df['DeviceInfo'].notna())
dev_coarse = df['DeviceInfo'].astype('object')


def causal_degrees(dev, uid):
    """device_degree: nº de OUTROS UIDs que usaram o dispositivo antes de t.
       uid_degree: nº de OUTROS dispositivos usados pelo UID antes de t."""
    has = dev.notna().values
    sub = pd.DataFrame({'dev': dev[has].values, 'uid': uid[has].values})
    first = (~sub.duplicated(['dev', 'uid'])).values.astype(np.int64)
    seen = 1 - first
    dev_deg = np.full(len(dev), np.nan, dtype=np.float32)
    f = pd.Series(first)
    dev_deg[has] = (f.groupby(sub['dev']).cumsum().values - first) - seen
    f_all = np.zeros(len(dev), dtype=np.int64)
    s_all = np.zeros(len(dev), dtype=np.int64)
    f_all[has] = first
    s_all[has] = seen
    uid_deg = (pd.Series(f_all).groupby(uid.values).cumsum().values - f_all) - s_all
    return dev_deg, uid_deg.astype(np.float32)


def snapshot_pagerank(dev, uid, step_days):
    """PageRank do UID no grafo formado SOMENTE por arestas anteriores ao início da semana.
       Normalizado por |V| (média 1) para ser comparável entre snapshots de tamanhos diferentes.
       UIDs ausentes do grafo recebem NaN (o XGBoost trata ausência explicitamente)."""
    days = (ts // 86400).astype(np.int64)
    has = dev.notna().values
    pairs = pd.DataFrame({'u': 'u_' + uid[has].values, 'd': 'd_' + dev[has].values, 'day': days[has]})
    pairs = pairs.drop_duplicates(['u', 'd'], keep='first')
    out = np.full(len(dev), np.nan, dtype=np.float32)
    G = nx.Graph()
    uid_keys = 'u_' + uid
    stats = []
    prev_start = -10 ** 9
    for start in range(int(days.min()), int(days.max()) + 1, step_days):
        new = pairs[(pairs['day'] < start) & (pairs['day'] >= prev_start)]
        prev_start = start
        G.add_edges_from(zip(new['u'].values, new['d'].values))
        if G.number_of_nodes() == 0:
            continue
        pr = pd.Series(nx.pagerank(G, alpha=0.85))
        rows = (days >= start) & (days < start + step_days)
        out[rows] = (uid_keys[rows].map(pr) * G.number_of_nodes()).values
        stats.append((start, G.number_of_nodes(), G.number_of_edges()))
    return out, stats


graph_meta = {}
for tag, dev in [('', dev_fp), ('_coarse', dev_coarse)]:
    log(f"Grafo bipartido{tag or ' (fingerprint)'}: graus causais + PageRank semanal...")
    t0 = time.time()
    dd, ud = causal_degrees(dev, uid)
    df[f'device_degree{tag}'] = dd
    df[f'uid_degree{tag}'] = ud
    df[f'uid_pagerank{tag}'], snaps = snapshot_pagerank(dev, uid, PR_SNAPSHOT_DAYS)
    tr_dev = pd.DataFrame({'d': dev[is_train].values, 'u': uid[is_train].values}).dropna()
    deg_train = tr_dev.groupby('d')['u'].nunique().sort_values(ascending=False)
    graph_meta[tag or '_fp'] = {
        'seconds': round(time.time() - t0, 1),
        'n_devices_train': int(len(deg_train)),
        'n_uids_with_device_train': int(tr_dev['u'].nunique()),
        'max_device_degree_train': int(deg_train.iloc[0]),
        'top5_devices_train': {str(k): int(v) for k, v in deg_train.head(5).items()},
        'share_devices_degree1_train': float((deg_train == 1).mean()),
        'final_snapshot_nodes_edges': snaps[-1][1:] if snaps else None,
        'val_device_tx_with_device_seen_in_train': float(
            pd.Series(dev[is_val].values).dropna().isin(set(dev[is_train].dropna())).mean()),
        'val_device_tx_with_device_degree_ge1': float((dd[is_val][~np.isnan(dd[is_val])] >= 1).mean()),
    }
meta['graphs'] = graph_meta
meta['share_tx_with_deviceinfo'] = float(df['DeviceInfo'].notna().mean())

# 3.7 PCA sobre V (ajustado no treino)
v_cols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
log(f"PCA em {len(v_cols)} colunas V...")
medians_v = df.loc[is_train, v_cols].median().fillna(0.0)
scaler = StandardScaler().fit(df.loc[is_train, v_cols].fillna(medians_v))
pca = PCA(n_components=15, random_state=42).fit(scaler.transform(df.loc[is_train, v_cols].fillna(medians_v)))
Z = pca.transform(scaler.transform(df[v_cols].fillna(medians_v)))
for i in range(15):
    df[f'PCA_V_{i}'] = Z[:, i].astype(np.float32)
meta['pca_explained_variance_15'] = float(pca.explained_variance_ratio_.sum())
del Z
gc.collect()

# 3.8 Roteamento e codificação categórica (ajustada no treino)
has_dev = df['DeviceType'].notna().values
non_feature_str = {'uid'}
cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c not in non_feature_str]
for col in cat_cols:
    s = df[col].astype(str)
    mapping = {v: i for i, v in enumerate(sorted(s[is_train].unique(), key=lambda x: str(x)))}
    df[col] = s.map(mapping).fillna(-1).astype(np.int32)
log(f"Categóricas codificadas: {len(cat_cols)}")

# ---------------------------------------------------------------------------------------------
# 4. ABLAÇÃO
# ---------------------------------------------------------------------------------------------
raw_v_cols = v_cols
pca_cols = [f'PCA_V_{i}' for i in range(15)]
topo_fp = ['uid_pagerank', 'uid_degree', 'device_degree']
topo_coarse = ['uid_pagerank_coarse', 'uid_degree_coarse', 'device_degree_coarse']
velocity_cols = ['uid_tx_count_24h', 'uid_secs_since_prev']
fe_cols = ['D1n', 'D2n', 'D10n', 'D15n', 'uid_prev_tx_count', 'card1_count_30d', 'P_emaildomain_count_30d'] + \
          [c for c in df.columns if c.endswith('_to_mean_uid') or c.endswith('_to_std_uid')]
always_drop = ['isFraud', 'TransactionID', 'uid', 'timestamp', 'DeviceInfo']
engineered = set(pca_cols + topo_fp + topo_coarse + velocity_cols + fe_cols)
base_cols = [c for c in df.columns if c not in always_drop and c not in engineered and c not in raw_v_cols]

CONFIGS = [
    # id, nome, usa_pca, blocos extras
    ('1A', 'Baseline (V brutas)',              False, []),
    ('1B', 'Baseline (PCA 15-d)',              True,  []),
    ('2',  '1B + Topologia (fingerprint)',     True,  [topo_fp]),
    ('2B', '1B + Topologia (DeviceInfo bruto)', True, [topo_coarse]),
    ('3',  '1B + Velocidade',                  True,  [velocity_cols]),
    ('4',  '1B + Topologia + Velocidade',      True,  [topo_fp, velocity_cols]),
    ('5',  '1B + FE comportamental',           True,  [fe_cols]),
    ('6A', 'Completo (PCA 15-d)',              True,  [fe_cols, topo_fp, velocity_cols]),
    ('6B', 'Completo (V brutas)',              False, [fe_cols, topo_fp, velocity_cols]),
]

tr_a, tr_b = is_train & ~has_dev, is_train & has_dev
va_a, va_b = is_val & ~has_dev, is_val & has_dev
y_val = y_all[is_val]
amt_val = amt_all[is_val]
calib_v = is_calib[is_val]
test_v = is_test[is_val]
hasdev_v = has_dev[is_val]
uid_val = uid[is_val].reset_index(drop=True)


def feature_lists(use_pca, blocks):
    cols = base_cols + (pca_cols if use_pca else raw_v_cols)
    for b in blocks:
        cols = cols + b
    feats_b = list(dict.fromkeys(cols))
    feats_a = [c for c in feats_b if not c.startswith('id_') and c != 'DeviceType']
    return feats_a, feats_b


def fit_xgb(mask, feats, seed):
    y = y_all[mask]
    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=0.03, max_depth=9, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=(y == 0).sum() / max(y.sum(), 1),
        tree_method='hist', n_jobs=N_JOBS, random_state=seed)
    model.fit(df.loc[mask, feats], y, verbose=False)
    return model


def causal_postprocess(raw):
    """p_final = 0.7 p_i + 0.3 média(p_k das transações ANTERIORES do mesmo UID); sem histórico -> p_i.
       A validação está em ordem temporal, logo cumsum por grupo respeita a causalidade."""
    s = pd.Series(raw)
    g = s.groupby(uid_val.values)
    n_prev = g.cumcount().values
    hist = (g.cumsum().values - raw) / np.where(n_prev > 0, n_prev, 1)
    return np.where(n_prev > 0, 0.7 * raw + 0.3 * hist, raw)


def best_f1_threshold(y, s):
    return max(THRESHOLDS, key=lambda t: f1_score(y, s >= t, zero_division=0))


scores = {}          # scores[cid] = semente principal (SEEDS[0])
seed_metrics = []    # métricas por semente
keep_models = {}
for cid, name, use_pca, blocks in CONFIGS:
    feats_a, feats_b = feature_lists(use_pca, blocks)
    for seed in SEEDS:
        t0 = time.time()
        m_a = fit_xgb(tr_a, feats_a, seed)
        m_b = fit_xgb(tr_b, feats_b, seed)
        elapsed = time.time() - t0
        raw = np.empty(is_val.sum())
        raw[~hasdev_v] = m_a.predict_proba(df.loc[va_a, feats_a])[:, 1]
        raw[hasdev_v] = m_b.predict_proba(df.loc[va_b, feats_b])[:, 1]
        pp = causal_postprocess(raw)
        th_f1 = best_f1_threshold(y_val[calib_v], raw[calib_v])
        rec = {
            'ID': cid, 'Configuracao': name, 'seed': seed,
            'n_feats_A': len(feats_a), 'n_feats_B': len(feats_b),
            'PR_AUC': average_precision_score(y_val, raw),
            'ROC_AUC': roc_auc_score(y_val, raw),
            'PR_AUC_com_device': average_precision_score(y_val[hasdev_v], raw[hasdev_v]),
            'PR_AUC_sem_device': average_precision_score(y_val[~hasdev_v], raw[~hasdev_v]),
            'PR_AUC_PP': average_precision_score(y_val, pp),
            'F1_teste': f1_score(y_val[test_v], raw[test_v] >= th_f1),
            'tau_F1_calib': th_f1,
            'Tempo_treino_s': elapsed,
        }
        seed_metrics.append(rec)
        log(f"[{cid:>2}] seed={seed:<4} A={len(feats_a):>3} B={len(feats_b):>3} | {elapsed:5.1f}s | "
            f"PR={rec['PR_AUC']:.4f} (dev {rec['PR_AUC_com_device']:.4f}) ROC={rec['ROC_AUC']:.4f} PP={rec['PR_AUC_PP']:.4f}")
        if seed == SEEDS[0]:
            scores[cid] = {'raw': raw, 'pp': pp}
            if cid == '6A':
                keep_models = {'A': (m_a, feats_a), 'B': (m_b, feats_b)}
        del m_a, m_b
        gc.collect()

sm = pd.DataFrame(seed_metrics)
sm.to_csv(os.path.join(OUT_DIR, 'ablation_by_seed.csv'), index=False)
metric_cols = ['PR_AUC', 'ROC_AUC', 'PR_AUC_com_device', 'PR_AUC_sem_device', 'PR_AUC_PP', 'F1_teste', 'Tempo_treino_s']
summary = sm.groupby(['ID', 'Configuracao', 'n_feats_A', 'n_feats_B'], sort=False)[metric_cols].agg(['mean', 'std'])
summary.columns = [f'{a}_{b}' for a, b in summary.columns]
summary = summary.reset_index()
summary.to_csv(os.path.join(OUT_DIR, 'ablation_summary.csv'), index=False)
log("ablation_summary.csv salvo")

# Latência de inferência (Model B do 6A, 1 transação por chamada)
m_b, feats_b = keep_models['B']
one = df.loc[va_b, feats_b].iloc[[0]]
lat = []
for _ in range(300):
    t0 = time.perf_counter()
    m_b.predict_proba(one)
    lat.append((time.perf_counter() - t0) * 1000)
meta['latency_ms_single_row_modelB_6A'] = {'median': float(np.median(lat)), 'p95': float(np.percentile(lat, 95))}

# Importância de atributos (6A, semente principal)
imp_rows = []
for part, (m, feats) in keep_models.items():
    booster = m.get_booster()
    gain = booster.get_score(importance_type='gain')
    tgain = booster.get_score(importance_type='total_gain')
    tot = sum(tgain.values())
    for f in feats:
        imp_rows.append({'modelo': part, 'feature': f, 'gain': gain.get(f, 0.0),
                         'total_gain': tgain.get(f, 0.0), 'total_gain_share': tgain.get(f, 0.0) / tot})
imp = pd.DataFrame(imp_rows)
imp['rank_total_gain'] = imp.groupby('modelo')['total_gain'].rank(ascending=False, method='min').astype(int)
imp['rank_gain'] = imp.groupby('modelo')['gain'].rank(ascending=False, method='min').astype(int)
imp['grupo'] = np.select(
    [imp.feature.isin(topo_fp), imp.feature.isin(velocity_cols), imp.feature.isin(fe_cols), imp.feature.isin(pca_cols)],
    ['grafo', 'velocidade', 'FE comportamental', 'PCA V'], 'original')
imp.sort_values(['modelo', 'rank_total_gain']).to_csv(os.path.join(OUT_DIR, 'feature_importance_6A.csv'), index=False)
del keep_models, m_b
gc.collect()

# ---------------------------------------------------------------------------------------------
# 5. EXPORTAÇÃO DAS PREDIÇÕES (semente principal)
# ---------------------------------------------------------------------------------------------
export_cols = ['TransactionID', 'timestamp', 'isFraud', 'TransactionAmt', 'uid'] + topo_fp + topo_coarse + \
              velocity_cols + ['TransactionAmt_to_mean_uid', 'uid_prev_tx_count']
val_export = df.loc[is_val, export_cols].reset_index(drop=True)
val_export['has_device'] = hasdev_v
val_export['half'] = np.where(calib_v, 'calib', 'test')
for cid in scores:
    val_export[f'score_{cid}_raw'] = scores[cid]['raw']
    val_export[f'score_{cid}_pp'] = scores[cid]['pp']

# ---------------------------------------------------------------------------------------------
# 6. BOOTSTRAP PAREADO POR BLOCOS DE DIA
# ---------------------------------------------------------------------------------------------
log(f"Bootstrap ({N_BOOT} réplicas, blocos de dia)...")
COMPARISONS = [('1A', '1B'), ('2', '1B'), ('2B', '1B'), ('3', '1B'), ('4', '1B'), ('5', '1B'),
               ('6A', '5'), ('6A', '1B'), ('6B', '6A')]
day_v = (val_export['timestamp'].values // 86400)
boot_rows = []
for scope, mask in [('todas', np.ones(len(y_val), bool)), ('com_device', hasdev_v)]:
    d_idx = {d: np.where((day_v == d) & mask)[0] for d in np.unique(day_v[mask])}
    days = np.array(list(d_idx))
    rng = np.random.default_rng(0)
    obs = {c: average_precision_score(y_val[mask], scores[c]['raw'][mask]) for c in scores}
    reps = {c: [] for c in scores}
    for _ in range(N_BOOT):
        samp = np.concatenate([d_idx[d] for d in rng.choice(days, len(days), replace=True)])
        for c in scores:
            reps[c].append(average_precision_score(y_val[samp], scores[c]['raw'][samp]))
    for a, b in COMPARISONS:
        dlt = np.array(reps[a]) - np.array(reps[b])
        boot_rows.append({'escopo': scope, 'comparacao': f'{a} - {b}', 'A': a, 'B': b,
                          'delta_obs': obs[a] - obs[b], 'ic95_inf': np.percentile(dlt, 2.5),
                          'ic95_sup': np.percentile(dlt, 97.5), 'p_delta_le_0': float(np.mean(dlt <= 0)),
                          'significativo': bool(np.percentile(dlt, 2.5) > 0 or np.percentile(dlt, 97.5) < 0)})
boot = pd.DataFrame(boot_rows)
boot.to_csv(os.path.join(OUT_DIR, 'bootstrap_ci.csv'), index=False)
print(boot.to_string(index=False))

# ---------------------------------------------------------------------------------------------
# 7. CUSTO FINANCEIRO (limiar escolhido na calibração, custo medido no teste)
# ---------------------------------------------------------------------------------------------
log("Análise de custo...")


def cost_parts(y, s, a, t):
    pred = s >= t
    fn = (y == 1) & ~pred
    fp = (y == 0) & pred
    tp = (y == 1) & pred
    return {'FN_n': int(fn.sum()), 'FN_usd': float((a[fn] + COST['C_cb']).sum()),
            'FP_n': int(fp.sum()), 'FP_usd': float((COST['C_rev'] + COST['alpha'] * a[fp]).sum()),
            'TP_n': int(tp.sum()), 'TP_usd': float(COST['C_op'] * tp.sum())}


def total(p):
    return p['FN_usd'] + p['FP_usd'] + p['TP_usd']


yc, ac = y_val[calib_v], amt_val[calib_v]
yt, at = y_val[test_v], amt_val[test_v]
blind_test = float(((at + COST['C_cb']) * (yt == 1)).sum())
cost_rows, curves = [], {'tau': THRESHOLDS}
for cid in scores:
    s = scores[cid]['raw']
    calib_losses = [total(cost_parts(yc, s[calib_v], ac, t)) for t in THRESHOLDS]
    test_losses = [total(cost_parts(yt, s[test_v], at, t)) for t in THRESHOLDS]
    tau = THRESHOLDS[int(np.argmin(calib_losses))]
    p = cost_parts(yt, s[test_v], at, tau)
    curves[cid] = test_losses
    cost_rows.append({'ID': cid, 'tau_calib': tau, 'Perda_teste': total(p),
                      'Perda_teste_oraculo': min(test_losses), 'tau_oraculo': THRESHOLDS[int(np.argmin(test_losses))],
                      'Economia_vs_aprovacao_cega': blind_test - total(p),
                      'Recall_teste': p['TP_n'] / max(p['TP_n'] + p['FN_n'], 1),
                      'Taxa_alerta_teste': (p['TP_n'] + p['FP_n']) / len(yt), **p})
cost = pd.DataFrame(cost_rows)
base = cost.loc[cost.ID == '1B', 'Perda_teste'].iloc[0]
cost['Delta_vs_1B'] = base - cost['Perda_teste']
cost['Delta_vs_1B_pct'] = cost['Delta_vs_1B'] / base * 100
cost.to_csv(os.path.join(OUT_DIR, 'cost_summary.csv'), index=False)
pd.DataFrame(curves).to_csv(os.path.join(OUT_DIR, 'cost_curves_test.csv'), index=False)
meta['blind_loss_test'] = blind_test
print(cost.to_string(index=False))

# ---------------------------------------------------------------------------------------------
# 8. TIPOLOGIA RELACIONAL E RECALL COM ORÇAMENTO DE ALERTAS IGUAL
# ---------------------------------------------------------------------------------------------
log("Análise por tipologia...")
relational = ((val_export['device_degree'].fillna(0) >= 1) | (val_export['uid_tx_count_24h'] >= 2)).values
val_export['relacional'] = relational
k = int(round(ALERT_BUDGET * len(y_val)))
typ_rows = []
for cid in scores:
    s = scores[cid]['raw']
    flag = s >= np.sort(s)[-k]
    for label, m in [('Fraude com vínculo relacional', (y_val == 1) & relational),
                     ('Fraude sem vínculo relacional', (y_val == 1) & ~relational)]:
        typ_rows.append({'ID': cid, 'tipologia': label, 'n': int(m.sum()), 'usd': float(amt_val[m].sum()),
                         'recall_orcamento': float(flag[m].mean())})
typ = pd.DataFrame(typ_rows)
typ.to_csv(os.path.join(OUT_DIR, 'recall_by_typology.csv'), index=False)
meta['typology'] = {
    'alert_budget_k': k,
    'share_fraud_relational': float(relational[y_val == 1].mean()),
    'share_legit_relational': float(relational[y_val == 0].mean()),
    'fraud_rate_relational': float(y_val[relational].mean()),
    'fraud_rate_non_relational': float(y_val[~relational].mean()),
}
print(typ.pivot(index='ID', columns='tipologia', values='recall_orcamento').to_string())

# ---------------------------------------------------------------------------------------------
# 9. ESTUDOS DE CASO EXTRAÍDOS DOS DADOS (efeito isolado da topologia: config 2 vs 1B)
# ---------------------------------------------------------------------------------------------
tau_1b = cost.loc[cost.ID == '1B', 'tau_calib'].iloc[0]
tau_2 = cost.loc[cost.ID == '2', 'tau_calib'].iloc[0]
ve = val_export
ve['delta_2_vs_1B'] = ve['score_2_raw'] - ve['score_1B_raw']
cand = ve[ve.has_device & ve.device_degree.notna()]
cases = pd.concat([
    cand[cand.isFraud == 1].nlargest(3, 'delta_2_vs_1B').assign(tipo_caso='fraude: grafo elevou o escore'),
    cand[cand.isFraud == 1].nsmallest(3, 'delta_2_vs_1B').assign(tipo_caso='fraude: grafo reduziu o escore'),
    cand[cand.isFraud == 0].nlargest(3, 'delta_2_vs_1B').assign(tipo_caso='legítima: grafo elevou o escore'),
])
cases[['tipo_caso', 'TransactionID', 'isFraud', 'TransactionAmt', 'uid', 'device_degree', 'uid_degree',
       'uid_pagerank', 'device_degree_coarse', 'uid_tx_count_24h', 'score_1B_raw', 'score_2_raw', 'delta_2_vs_1B']] \
    .to_csv(os.path.join(OUT_DIR, 'case_studies.csv'), index=False)
fr = ve.isFraud == 1
meta['graph_flip_counts_test_half'] = {
    'tau_1B': float(tau_1b), 'tau_2': float(tau_2),
    'fraudes_capturadas_so_com_grafo': int((fr & (ve.half == 'test') & (ve.score_2_raw >= tau_2) & (ve.score_1B_raw < tau_1b)).sum()),
    'fraudes_perdidas_com_grafo': int((fr & (ve.half == 'test') & (ve.score_2_raw < tau_2) & (ve.score_1B_raw >= tau_1b)).sum()),
}

val_export.drop(columns=['delta_2_vs_1B']).to_csv(os.path.join(OUT_DIR, 'val_predictions.csv'), index=False)
meta['total_minutes'] = round((time.time() - t_global) / 60, 1)
with open(os.path.join(OUT_DIR, 'run_metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False, default=float)
log(f"Concluído em {meta['total_minutes']} min. Saídas em {OUT_DIR}/")
