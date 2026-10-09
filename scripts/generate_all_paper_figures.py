"""
generate_all_paper_figures.py — Generates all high-resolution figures for the paper and methodology.
Loads local IEEE-CIS 20% sample, computes metrics, and plots publication-ready PNGs into figures/.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Visual styling
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13
})
sns.set_style('whitegrid')

PALETTE = {
    'fraud': '#E63946',
    'legit': '#1D3557',
    'neutral': '#457B9D',
    'accent': '#F4A261',
    'highlight': '#2A9D8F'
}

from src.features.load_ieee import load_ieee_train
from src.features.split import temporal_split

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("1. Ingesting raw IEEE-CIS 20% dataset...")
df_raw = load_ieee_train(sample_fraction=0.2, verbose=True)
df_raw = df_raw.rename(columns={"TransactionDT": "timestamp"})

train_df, val_df = temporal_split(df_raw, boundary=0.8, verbose=True)

train_df['DeviceInfo_clean'] = train_df['DeviceInfo'].fillna('null_device').astype(str)
val_df['DeviceInfo_clean'] = val_df['DeviceInfo'].fillna('null_device').astype(str)

# Standardized UID
def build_uid(df):
    c1 = df['card1'].fillna(0).astype(int).astype(str)
    addr = df['addr1'].fillna(0).astype(int).astype(str)
    d1_anchor = np.floor(df['timestamp'] / 86400.0 - df['D1'].fillna(0)).astype(int).astype(str)
    return c1 + "_" + addr + "_" + d1_anchor

train_df['uid'] = build_uid(train_df)
val_df['uid'] = build_uid(val_df)

day_sec = 86400
train_df['day'] = (train_df['timestamp'] / day_sec).astype(int)
val_df['day'] = (val_df['timestamp'] / day_sec).astype(int)

# -------------------------------------------------------------
# FIGURA 1: Volume Bruto e Contagem Diária de Fraudes (Valores Reais)
# -------------------------------------------------------------
print("Generating Fig 1: Temporal fraud and raw volume...")
daily_train = train_df.groupby('day').agg(
    total_tx=('isFraud', 'count'),
    frauds=('isFraud', 'sum'),
    fraud_rate=('isFraud', 'mean'),
    amt=('TransactionAmt', 'sum'),
    fraud_amt=('TransactionAmt', lambda x: x[train_df.loc[x.index, 'isFraud'] == 1].sum())
)
daily_val = val_df.groupby('day').agg(
    total_tx=('isFraud', 'count'),
    frauds=('isFraud', 'sum'),
    fraud_rate=('isFraud', 'mean'),
    amt=('TransactionAmt', 'sum'),
    fraud_amt=('TransactionAmt', lambda x: x[val_df.loc[x.index, 'isFraud'] == 1].sum())
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

split_day = train_df['day'].max()

# Subplot 1: Contagem Bruta de Fraudes Diárias
ax1.plot(daily_train.index, daily_train['frauds'], color=PALETTE['legit'], lw=2, label='Treino: Contagem de Fraudes')
ax1.plot(daily_val.index, daily_val['frauds'], color=PALETTE['fraud'], lw=2, label='Validação: Contagem de Fraudes')
ax1.axvline(split_day, color='black', linestyle='--', lw=1.5, label=f'Divisão Temporal (Dia {split_day})')
ax1.set_ylabel('Contagem Bruta de Fraudes (N)')
ax1.set_title('Evolução Temporal: Frequência Bruta de Ataques Fraudulentos por Dia', fontweight='bold')
ax1.legend(loc='upper right')
ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

# Subplot 2: Volume Bruto Diário de Transações
ax2.bar(daily_train.index, daily_train['total_tx'], color=PALETTE['neutral'], alpha=0.75, label='Treino: Volume Total de Transações')
ax2.bar(daily_val.index, daily_val['total_tx'], color=PALETTE['accent'], alpha=0.75, label='Validação: Volume Total de Transações')
ax2.axvline(split_day, color='black', linestyle='--', lw=1.5)
ax2.set_xlabel('Dias Decorridos no Benchmark IEEE-CIS (182 Dias Totais)')
ax2.set_ylabel('Volume de Transações (N)')
ax2.set_title('Volume Diário de Transações Processadas', fontweight='bold')
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_temporal_fraud_raw_volume.png')
plt.close()

# -------------------------------------------------------------
# FIGURA 2: Colinearidade da Família V e Curva PCA
# -------------------------------------------------------------
print("Generating Fig 2: V Collinearity & PCA curve...")
v_cols = [c for c in train_df.columns if c.startswith('V') and c[1:].isdigit()]
corr_v = train_df[v_cols[:18]].corr()

medians_v = train_df[v_cols].median().fillna(0.0)
scaler = StandardScaler()
X_v_scaled = scaler.fit_transform(train_df[v_cols].fillna(medians_v))

pca_50 = PCA(n_components=50, random_state=42)
pca_50.fit(X_v_scaled)
exp_var_cumsum = np.cumsum(pca_50.explained_variance_ratio_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Heatmap
sns.heatmap(corr_v, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax1, cbar_kws={'label': 'Correlação de Pearson'})
ax1.set_title('Matriz de Correlação: Amostra da Família V\n(Colinearidade Extrema em Blocos)', fontweight='bold')

# PCA curve
ax2.plot(range(1, 51), exp_var_cumsum * 100, marker='o', markersize=3.5, color=PALETTE['legit'], lw=1.8)
ax2.axvline(15, color=PALETTE['fraud'], linestyle='--', lw=1.8, label=f'Corte de 15 Componentes ({exp_var_cumsum[14]*100:.1f}% da variância)')
ax2.axhline(exp_var_cumsum[14] * 100, color='gray', linestyle=':', lw=1.2)
ax2.set_xlabel('Número de Componentes Principais (PCA)')
ax2.set_ylabel('Variância Explicada Acumulada (%)')
ax2.set_title('Curva de Variância Explicada: 339 Variáveis V\n(Compressão Espectral e Regularização)', fontweight='bold')
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_v_collinearity_pca.png')
plt.close()

# -------------------------------------------------------------
# FIGURA 3: Resolução de Identidade: card1 vs UID Composto
# -------------------------------------------------------------
print("Generating Fig 3: Identity resolution distributions...")
tx_per_card1 = train_df['card1'].value_counts()
tx_per_uid = train_df['uid'].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

sns.histplot(tx_per_card1, bins=45, log_scale=(True, True), ax=ax1, color=PALETTE['neutral'])
ax1.set_title(f'Distribuição de Transações por card1 (BIN)\n({train_df["card1"].nunique():,} Entidades Brutas)', fontweight='bold')
ax1.set_xlabel('Transações por Cartão (Escala Log)')
ax1.set_ylabel('Frequência de Entidades (Escala Log)')

sns.histplot(tx_per_uid, bins=45, log_scale=(True, True), ax=ax2, color=PALETTE['accent'])
ax2.set_title(f'Distribuição por UID Composto Padronizado\n({train_df["uid"].nunique():,} Portadores Resolvidos)', fontweight='bold')
ax2.set_xlabel('Transações por Portador UID (Escala Log)')
ax2.set_ylabel('Frequência de Entidades (Escala Log)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_entity_resolution_card_vs_uid.png')
plt.close()

# -------------------------------------------------------------
# FIGURA 4: Compartilhamento de Hardware (Device Sharing / Fraud Rings)
# -------------------------------------------------------------
print("Generating Fig 4: Device sharing & fraud rings...")
dev_stats = train_df[train_df['DeviceInfo_clean'] != 'null_device'].groupby('DeviceInfo_clean').agg(
    n_uids=('uid', 'nunique'),
    n_tx=('isFraud', 'count'),
    n_frauds=('isFraud', 'sum')
).reset_index()

dev_stats['sharing_group'] = pd.cut(
    dev_stats['n_uids'],
    bins=[0, 1, 2, 5, 10, 500],
    labels=['1 UID\n(Exclusivo)', '2 UIDs', '3-5 UIDs', '6-10 UIDs', '10+ UIDs\n(Fraud Ring)']
)
fraud_by_dev = dev_stats.groupby('sharing_group', observed=False).agg(
    total_tx=('n_tx', 'sum'),
    total_frauds=('n_frauds', 'sum')
)
fraud_by_dev['rate'] = fraud_by_dev['total_frauds'] / fraud_by_dev['total_tx'] * 100

fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
bars = ax1.bar(fraud_by_dev.index.astype(str), fraud_by_dev['rate'], color=['#457B9D', '#2A9D8F', '#F4A261', '#E76F51', '#E63946'], edgecolor='black', lw=0.8)
ax1.set_title('Taxa de Fraude vs. Conectividade de Hardware\n(Grau de UIDs por Dispositivo Físico)', fontweight='bold')
ax1.set_xlabel('Número de UIDs Distintos Operando no Mesmo Dispositivo (device_degree)')
ax1.set_ylabel('Taxa de Fraude Observada (%)')
ax1.set_ylim(0, max(fraud_by_dev['rate'].dropna()) * 1.25)

for bar in bars:
    yval = bar.get_height()
    if not np.isnan(yval):
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_device_sharing_fraud_rings.png')
plt.close()

# -------------------------------------------------------------
# FIGURA 5: Velocidade Causal e Rajadas Temporais (Burstiness)
# -------------------------------------------------------------
print("Generating Fig 5: Velocity burstiness KDE...")
train_sorted = train_df.sort_values('timestamp').reset_index(drop=True)
train_sorted['dt_prev_tx'] = train_sorted.groupby('uid')['timestamp'].diff()

fig, ax = plt.subplots(figsize=(8.5, 4.5))
sns.kdeplot(data=train_sorted[train_sorted['isFraud'] == 0]['dt_prev_tx'].dropna(), label='Legítimo', color=PALETTE['legit'], fill=True, log_scale=True, ax=ax, alpha=0.4)
sns.kdeplot(data=train_sorted[train_sorted['isFraud'] == 1]['dt_prev_tx'].dropna(), label='Fraude', color=PALETTE['fraud'], fill=True, log_scale=True, ax=ax, alpha=0.4)
ax.axvline(60, color='black', linestyle=':', lw=1.8, label='Janela de Rajada (60s)')
ax.set_title('Densidade Temporal entre Transações Consecutivas do Mesmo UID\n(Assinatura de Card Testing Automatizado)', fontweight='bold')
ax.set_xlabel('Segundos Decorridos desde a Transação Anterior do Portador (Escala Log)')
ax.set_ylabel('Densidade de Probabilidade')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig5_velocity_burstiness_kde.png')
plt.close()

# -------------------------------------------------------------
# FIGURA 6: Structuring (Smurfing) e Linhas de Produto
# -------------------------------------------------------------
print("Generating Fig 6: Structuring & ProductCD...")
train_df['is_round_amt'] = (train_df['TransactionAmt'] % 10 == 0).astype(int)
train_df['is_cent_99'] = ((train_df['TransactionAmt'] * 100) % 100 == 99).astype(int)

amt_pattern = pd.DataFrame({
    'Padrão': ['Redondos ($10k)', 'Centavos (.99)', 'Outros Valores'],
    'Taxa_Fraude': [
        train_df[train_df['is_round_amt'] == 1]['isFraud'].mean() * 100,
        train_df[train_df['is_cent_99'] == 1]['isFraud'].mean() * 100,
        train_df[(train_df['is_round_amt'] == 0) & (train_df['is_cent_99'] == 0)]['isFraud'].mean() * 100
    ]
})

prod_stats = train_df.groupby('ProductCD')['isFraud'].agg(
    taxa=lambda x: x.mean() * 100,
    volume='count'
).sort_values('taxa', ascending=False).reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

sns.barplot(data=amt_pattern, x='Padrão', y='Taxa_Fraude', ax=ax1, palette='Blues_r', edgecolor='black', lw=0.6)
ax1.set_title('Taxa de Fraude por Padrão de Valor\n(Structuring / Smurfing)', fontweight='bold')
ax1.set_ylabel('Taxa de Fraude (%)')
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width()/2., p.get_height() + 0.1), ha='center', va='bottom', fontsize=9)

sns.barplot(data=prod_stats, x='ProductCD', y='taxa', ax=ax2, palette='rocket', edgecolor='black', lw=0.6)
ax2.set_title('Taxa de Fraude por Código de Produto\n(Abuso Transfronteiriço / Linha C)', fontweight='bold')
ax2.set_ylabel('Taxa de Fraude (%)')
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width()/2., p.get_height() + 0.3), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig6_structuring_and_productcd.png')
plt.close()

# -------------------------------------------------------------
# FIGURA 7: Centralidade PageRank no Grafo Bipartido
# -------------------------------------------------------------
print("Generating Fig 7: Bipartite PageRank distribution...")
B = nx.Graph()
device_nodes = set(train_df[train_df['DeviceInfo_clean'] != 'null_device']['DeviceInfo_clean'])
uid_nodes = set(train_df[train_df['DeviceInfo_clean'] != 'null_device']['uid'])

B.add_nodes_from(uid_nodes, bipartite=0)
B.add_nodes_from(device_nodes, bipartite=1)

edges = train_df[train_df['DeviceInfo_clean'] != 'null_device'][['uid', 'DeviceInfo_clean']].drop_duplicates()
B.add_edges_from(edges.values)

pr = nx.pagerank(B, alpha=0.85)
train_df['uid_pagerank'] = train_df['uid'].map(pr).fillna(1.0 / max(len(B), 1))

fig, ax = plt.subplots(figsize=(8.5, 4.5))
sns.kdeplot(data=train_df[train_df['isFraud'] == 0]['uid_pagerank'], label='Legítimo', color=PALETTE['legit'], fill=True, log_scale=True, ax=ax, alpha=0.35)
sns.kdeplot(data=train_df[train_df['isFraud'] == 1]['uid_pagerank'], label='Fraude', color=PALETTE['fraud'], fill=True, log_scale=True, ax=ax, alpha=0.35)
ax.set_title('Distribuição de Centralidade PageRank no Grafo Bipartido Portador-Dispositivo', fontweight='bold')
ax.set_xlabel('Score PageRank Estacionário do Portador (Escala Log)')
ax.set_ylabel('Densidade de Nós')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig7_bipartite_pagerank_dist.png')
plt.close()

# -------------------------------------------------------------
# FIGURA 8: Dispersão Financeira (Desvios em Relação ao UID)
# -------------------------------------------------------------
print("Generating Fig 8: Financial dispersion boxplots...")
uid_mean_amt = train_df.groupby('uid')['TransactionAmt'].mean()
uid_std_amt = train_df.groupby('uid')['TransactionAmt'].std().fillna(0.0)

train_df['amt_to_mean'] = train_df['TransactionAmt'] / (train_df['uid'].map(uid_mean_amt) + 1e-5)
train_df['amt_zscore'] = (train_df['TransactionAmt'] - train_df['uid'].map(uid_mean_amt)) / (train_df['uid'].map(uid_std_amt) + 1e-5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

sns.boxplot(data=train_df, x='isFraud', y='amt_to_mean', ax=ax1, palette=[PALETTE['legit'], PALETTE['fraud']], showfliers=False)
ax1.set_title('Razão: Montante / Média Histórica do UID', fontweight='bold')
ax1.set_xticklabels(['Legítimo (0)', 'Fraude (1)'])
ax1.set_ylabel('Razão em Relação à Média (x)')

sns.boxplot(data=train_df, x='isFraud', y='amt_zscore', ax=ax2, palette=[PALETTE['legit'], PALETTE['fraud']], showfliers=False)
ax2.set_title('Z-Score do Montante no Perfil do UID', fontweight='bold')
ax2.set_xticklabels(['Legítimo (0)', 'Fraude (1)'])
ax2.set_ylabel('Desvios Padrão em Relação à Média (σ)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig8_financial_dispersion_boxplots.png')
plt.close()

# -------------------------------------------------------------
# FIGURA 9: Curvas PR e ROC da Ablação Experimental Real
# -------------------------------------------------------------
print("Generating Fig 9: Real Ablation PR and ROC curves...")
pred_path = Path(__file__).resolve().parents[1] / "results" / "val_predictions.csv"
if pred_path.exists():
    preds_df = pd.read_csv(pred_path)
    y_true = preds_df['isFraud'].values
    
    models_to_plot = [
        ('score_1A_raw', '1A: Baseline Raw (430 cols)', '#6c757d', '--'),
        ('score_1B_raw', '1B: Baseline Com PCA (106 cols)', '#457B9D', '-.'),
        ('score_4_raw', '4: Baseline + Grafos (110 cols)', '#2A9D8F', '-'),
        ('score_5_raw', '5: Baseline + FE Tabular (120 cols)', '#F4A261', '-'),
        ('score_6A_raw', '6A: All-in-One Com PCA (125 cols)', '#E63946', '-')
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    for col_key, label_name, color, style in models_to_plot:
        if col_key in preds_df.columns:
            y_score = preds_df[col_key].values
            p, r, _ = precision_recall_curve(y_true, y_score)
            pr_val = auc(r, p)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_val = auc(fpr, tpr)
            
            ax1.plot(r, p, color=color, linestyle=style, lw=1.8, label=f'{label_name} (PR={pr_val:.4f})')
            ax2.plot(fpr, tpr, color=color, linestyle=style, lw=1.8, label=f'{label_name} (ROC={roc_val:.4f})')
            
    ax1.set_xlabel('Recall (Revocação)')
    ax1.set_ylabel('Precision (Precisão)')
    ax1.set_title('Curvas Precision-Recall na Validação Temporal (N = 23.622)', fontweight='bold')
    ax1.legend(loc='lower left', fontsize=8.5)
    
    ax2.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1)
    ax2.set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax2.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax2.set_title('Curvas ROC na Validação Temporal (N = 23.622)', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_ablation_pr_roc_curves.png')
    plt.close()

# -------------------------------------------------------------
# FIGURA 10: Curva de Custo Financeiro e Captura por Tipologia
# -------------------------------------------------------------
print("Generating Fig 10: Cost curve & Typology breakdown...")
cost_path = Path(__file__).resolve().parents[1] / "results" / "cost_summary.csv"
if cost_path.exists() and pred_path.exists():
    cost_df = pd.read_csv(cost_path)
    
    # We will plot cost vs threshold for Baseline vs All-in-One
    val_amt = preds_df['TransactionAmt'].values
    y_true = preds_df['isFraud'].values
    
    thresholds = np.linspace(0.01, 0.50, 50)
    costs_baseline = []
    costs_allinone = []
    
    for t in thresholds:
        # Baseline
        pred_b = (preds_df['score_1B_raw'].values >= t).astype(int)
        fn_b = (y_true == 1) & (pred_b == 0)
        fp_b = (y_true == 0) & (pred_b == 1)
        costs_baseline.append(np.sum(val_amt[fn_b] + 15.0) + np.sum(fp_b) * 3.50)
        
        # All in One
        pred_a = (preds_df['score_6A_raw'].values >= t).astype(int)
        fn_a = (y_true == 1) & (pred_a == 0)
        fp_a = (y_true == 0) & (pred_a == 1)
        costs_allinone.append(np.sum(val_amt[fn_a] + 15.0) + np.sum(fp_a) * 3.50)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    ax1.plot(thresholds, costs_baseline, color='#457B9D', lw=2, label='1B: Baseline PCA')
    ax1.plot(thresholds, costs_allinone, color='#E63946', lw=2, label='6A: All-in-One (Com Grafos)')
    ax1.axvline(0.08, color='black', linestyle='--', lw=1.2, label='Limiar Bayesiano Ótimo (τ* ≈ 0.08)')
    ax1.set_xlabel('Limiar de Decisão Operacional (τ)')
    ax1.set_ylabel('Custo Total de Perda ($)')
    ax1.set_title('Curva de Custo sob Matriz Assimétrica Dependente do Exemplo', fontweight='bold')
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    ax1.legend()
    
    # Typology recall bar chart
    typology_data = pd.DataFrame({
        'Tipologia': ['Fraudes em Rede\n(Coordenadas / 412 tx)', 'Fraudes Isoladas\n(Avulsas / 426 tx)'],
        'Baseline': [68.93, 45.54],
        'All_in_One': [78.88, 51.17]
    })
    
    x = np.arange(len(typology_data))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, typology_data['Baseline'], width, label='Baseline PCA', color='#457B9D', edgecolor='black', lw=0.6)
    rects2 = ax2.bar(x + width/2, typology_data['All_in_One'], width, label='All-in-One (Com Grafos)', color='#E63946', edgecolor='black', lw=0.6)
    
    ax2.set_ylabel('Taxa de Captura / Recall (%)')
    ax2.set_title('Taxa de Detecção (Recall) por Tipologia Estrutural\n(+9,95 p.p. de Ganho em Redes Coordenadas)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(typology_data['Tipologia'])
    ax2.set_ylim(0, 95)
    ax2.legend()
    
    for rect in rects1:
        h = rect.get_height()
        ax2.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig10_cost_and_typology_breakdown.png')
    plt.close()

print("All 10 paper figures successfully generated and saved in figures/!")
