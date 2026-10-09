#!/usr/bin/env python3
"""
scripts/generate_all_single_figures.py
Generates the 10 standalone figures (300 DPI) for LOCAL_COMPLETE_PAPER.md
Strict rule: Exactly 1 plot per figure, clear bold titles, 100% faithful to the data.
"""

import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'figure.titlesize': 13
})
sns.set_style('whitegrid')

PALETTE = {
    'fraud': '#E63946',
    'legit': '#1D3557',
    'neutral': '#457B9D',
    'accent': '#F4A261',
    'highlight': '#2A9D8F',
    'dark': '#2B2D42'
}

OUTPUT_DIRS = ['figures', 'results/100percent/figures']
for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)

def save_fig(fig, filename):
    for d in OUTPUT_DIRS:
        path = os.path.join(d, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")

DATA_DIR = 'data/ieee/raw'
print("1. Ingesting raw data for figures...")
df_trans = pd.read_csv(os.path.join(DATA_DIR, 'train_transaction.csv'),
                       usecols=['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt',
                                'ProductCD', 'card1', 'addr1', 'D1', 'V1', 'V3', 'V5', 'V7',
                                'V9', 'V12', 'V14', 'V16', 'V20', 'V24', 'V28', 'V32', 'V36',
                                'V40', 'V44', 'V48', 'V52', 'V56', 'V60', 'V64'])
df_id = pd.read_csv(os.path.join(DATA_DIR, 'train_identity.csv'),
                    usecols=['TransactionID', 'DeviceInfo', 'id_30', 'id_31', 'id_33'])
df = df_trans.merge(df_id, on='TransactionID', how='left')
del df_trans, df_id
gc.collect()

df = df.rename(columns={"TransactionDT": "timestamp"})
day_float = df['timestamp'] / 86400.0
df['day'] = day_float.astype(int)
df['uid'] = (df['card1'].fillna(0).astype(int).astype(str) + '_' +
             df['addr1'].fillna(0).astype(int).astype(str) + '_' +
             (day_float - df['D1'].fillna(0)).astype(int).astype(str))

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
val_df = df.iloc[split_idx:].copy().reset_index(drop=True)

# -------------------------------------------------------------------------
# FIGURA 1: Evolução Temporal da Frequência de Fraudes
# -------------------------------------------------------------------------
print("Generating Fig 1: Temporal fraud frequency...")
daily_train = train_df.groupby('day').agg(frauds=('isFraud', 'sum'))
daily_val = val_df.groupby('day').agg(frauds=('isFraud', 'sum'))
split_day = train_df['day'].max()

fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.plot(daily_train.index, daily_train['frauds'], color=PALETTE['legit'], lw=2, label='Treino Histórico (Contagem Diária)')
ax.plot(daily_val.index, daily_val['frauds'], color=PALETTE['fraud'], lw=2, label='Validação Causal (Contagem Diária)')
ax.axvline(split_day, color='black', linestyle='--', lw=1.5, label=f'Divisão Temporal Causal (Dia {split_day})')
ax.set_xlabel('Dias Decorridos no Benchmark IEEE-CIS (182 Dias Totais)')
ax.set_ylabel('Frequência Bruta de Fraudes (N)')
ax.set_title('Evolução Temporal: Frequência Diária de Fraudes em Valores Brutos', fontweight='bold')
ax.legend(loc='upper right')
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
save_fig(fig, 'fig1_temporal_fraud_raw_volume.png')

# -------------------------------------------------------------------------
# FIGURA 2: Matriz de Correlação das Variáveis Vesta
# -------------------------------------------------------------------------
print("Generating Fig 2: Representative V-family correlation matrix...")
v_sample_cols = [f'V{i}' for i in [1, 3, 5, 7, 9, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]]
v_sample_df = train_df[v_sample_cols].fillna(train_df[v_sample_cols].median())
corr_v = v_sample_df.corr()

fig, ax = plt.subplots(figsize=(8.5, 6.5))
sns.heatmap(corr_v, cmap='coolwarm', vmin=-1.0, vmax=1.0, center=0, ax=ax,
            cbar_kws={'label': 'Correlação Linear de Pearson (ρ)'}, linewidths=0.5, linecolor='white')
ax.set_title('Matriz de Correlação Linear: Amostra da Família Vesta V\n(Colinearidade Extrema em Blocos Estruturais)', fontweight='bold')
save_fig(fig, 'fig2_v_collinearity_pca.png')

# -------------------------------------------------------------------------
# FIGURA 3: Resolução de Entidades: card1 vs UID Composto (CCDF Log-Log)
# -------------------------------------------------------------------------
print("Generating Fig 3: Entity resolution CCDF log-log distribution...")
def calc_ccdf(series):
    counts = series.value_counts().values
    sorted_counts = np.sort(counts)
    ccdf = 1.0 - (np.arange(len(sorted_counts)) / float(len(sorted_counts)))
    unique_x, unique_idx = np.unique(sorted_counts, return_index=True)
    unique_y = ccdf[unique_idx]
    return unique_x, unique_y

x_c1, y_c1 = calc_ccdf(train_df['card1'])
x_uid, y_uid = calc_ccdf(train_df['uid'])

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.loglog(x_c1, y_c1, label='card1 Isolado (Agrupamento Ingênuo: BIN / Emissor)', color=PALETTE['neutral'], lw=2.2)
ax.loglog(x_uid, y_uid, label='UID Composto Padronizado (card1 + addr1 + D1_anchor)', color=PALETTE['fraud'], lw=2.4)
ax.set_title('Distribuição de Cauda Longa de Transações por Entidade (CCDF)\n(card1 versus UID Composto Padronizado)', fontweight='bold')
ax.set_xlabel('Número de Transações Associadas à Entidade (k) [Escala Logarítmica]')
ax.set_ylabel('P(X ≥ k) [Probabilidade Acumulada Complementar]')
ax.legend(loc='upper right')
save_fig(fig, 'fig3_entity_resolution_card_vs_uid.png')

# -------------------------------------------------------------------------
# FIGURA 4: Taxa de Fraude vs. Grau de Compartilhamento de Dispositivo
# -------------------------------------------------------------------------
print("Generating Fig 4: Device degree vs fraud rate...")
d_sub = train_df[train_df['DeviceInfo'].notna()]
g_dev = d_sub.groupby('DeviceInfo').agg(n=('uid', 'nunique'), tx=('isFraud', 'size'), fr=('isFraud', 'sum'))
g_dev['bin'] = pd.cut(g_dev['n'], [0, 1, 2, 5, 10, 1e9],
                      labels=['1 UID (Exclusivo)', '2 UIDs', '3–5 UIDs', '6–10 UIDs', '10+ UIDs (Hubs)'])
agg_dev = g_dev.groupby('bin', observed=False).agg(tx=('tx', 'sum'), fr=('fr', 'sum'))
agg_dev['rate'] = agg_dev['fr'] / agg_dev['tx'] * 100

fig, ax = plt.subplots(figsize=(8.5, 4.8))
bars = ax.bar(agg_dev.index.astype(str), agg_dev['rate'],
              color=['#457B9D', '#2A9D8F', '#F4A261', '#E76F51', '#E63946'], edgecolor='black', lw=0.8)
ax.set_title('Taxa Ponderada de Fraude vs. Grau de UIDs por Dispositivo\n(Partição de Treino — Rótulo DeviceInfo)', fontweight='bold')
ax.set_xlabel('Grau de UIDs Conectados ao Dispositivo (device_degree)')
ax.set_ylabel('Taxa Média de Fraude (%)')
ax.set_ylim(0, max(agg_dev['rate']) * 1.25)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9.5)
save_fig(fig, 'fig4_device_sharing_fraud_rings.png')

# -------------------------------------------------------------------------
# FIGURA 5: Velocidade Causal e Rajadas Temporais (Legítimo vs Fraude)
# -------------------------------------------------------------------------
print("Generating Fig 5: Velocity burstiness KDE...")
train_sorted = train_df.sort_values('timestamp').reset_index(drop=True)
train_sorted['dt_prev_tx'] = train_sorted.groupby('uid')['timestamp'].diff()

dt_legit = train_sorted[(train_sorted['isFraud'] == 0) & (train_sorted['dt_prev_tx'] > 0)]['dt_prev_tx']
dt_fraud = train_sorted[(train_sorted['isFraud'] == 1) & (train_sorted['dt_prev_tx'] > 0)]['dt_prev_tx']
n_legit = len(dt_legit)
n_fraud = len(dt_fraud)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
sns.kdeplot(dt_legit, label=f'Legítimo (N = {n_legit:,} intervalos)', color=PALETTE['legit'],
            fill=True, log_scale=True, common_norm=False, ax=ax, alpha=0.35, lw=2)
sns.kdeplot(dt_fraud, label=f'Fraude (N = {n_fraud:,} intervalos)', color=PALETTE['fraud'],
            fill=True, log_scale=True, common_norm=False, ax=ax, alpha=0.45, lw=2.2)
ax.axvline(60, color='black', linestyle=':', lw=1.8, label='Janela Crítica de Rajada (60s)')
ax.set_title('Densidade Temporal entre Transações Consecutivas do Mesmo Portador UID\n(Assinatura de Testes Automatizados vs. Consumo Espaçado)', fontweight='bold')
ax.set_xlabel('Segundos Decorridos desde a Transação Anterior do UID (Escala Logarítmica)')
ax.set_ylabel('Densidade Normalizada de Probabilidade')
ax.legend(loc='upper right')
save_fig(fig, 'fig5_velocity_burstiness_kde.png')

# -------------------------------------------------------------------------
# FIGURA 6: Structuring / Smurfing e Linhas de Negócio ProductCD
# -------------------------------------------------------------------------
print("Generating Fig 6: Structuring and ProductCD distributions...")
# Painel único comparando fracionamento monetário e categorias de produto
cents = (train_df['TransactionAmt'] % 1).round(2)
is_round = cents == 0.00
is_99 = cents == 0.99
other = (~is_round) & (~is_99)

labels_6 = ['Inteiro (.00)', 'Fracionado', 'Centavos (.99)', 'Prod: C', 'Prod: W', 'Prod: S']
rates_6 = [
    train_df.loc[is_round, 'isFraud'].mean() * 100,
    train_df.loc[other, 'isFraud'].mean() * 100,
    train_df.loc[is_99, 'isFraud'].mean() * 100,
    train_df.loc[train_df['ProductCD'] == 'C', 'isFraud'].mean() * 100,
    train_df.loc[train_df['ProductCD'] == 'W', 'isFraud'].mean() * 100,
    train_df.loc[train_df['ProductCD'] == 'S', 'isFraud'].mean() * 100
]

fig, ax = plt.subplots(figsize=(9, 4.8))
colors_6 = [PALETTE['fraud'], PALETTE['neutral'], PALETTE['highlight'], '#E76F51', '#2A9D8F', '#F4A261']
bars = ax.bar(labels_6, rates_6, color=colors_6, edgecolor='black', lw=0.8)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
ax.set_ylabel('Taxa de Incidência de Fraude (%)')
ax.set_title('Taxa de Fraude por Padrão de Fracionamento e Linha de Negócio (ProductCD)', fontweight='bold')
save_fig(fig, 'fig6_structuring_and_productcd.png')

del df, train_df, val_df, train_sorted
gc.collect()

# -------------------------------------------------------------------------
# FIGURA 7: Centralidade PageRank Causal (a partir de val_predictions.csv)
# -------------------------------------------------------------------------
print("Generating Fig 7: PageRank distribution from validation predictions...")
val_preds = pd.read_csv('results/100percent/val_predictions.csv',
                        usecols=['isFraud', 'uid_pagerank', 'score_1B_raw', 'score_2_raw', 'score_4_raw',
                                 'score_5_raw', 'score_6A_raw', 'TransactionAmt'])

fig, ax = plt.subplots(figsize=(8.5, 4.8))
pr_valid = val_preds[val_preds['uid_pagerank'].notna() & (val_preds['uid_pagerank'] > 0)]
sns.kdeplot(data=pr_valid[pr_valid['isFraud'] == 0]['uid_pagerank'], label='Legítimo',
            color=PALETTE['legit'], fill=True, log_scale=True, common_norm=False, ax=ax, alpha=0.35, lw=2)
sns.kdeplot(data=pr_valid[pr_valid['isFraud'] == 1]['uid_pagerank'], label='Fraude',
            color=PALETTE['fraud'], fill=True, log_scale=True, common_norm=False, ax=ax, alpha=0.45, lw=2.2)
ax.set_title('Densidade de Probabilidade do PageRank Semanal Causal (UIDs no Grafo)', fontweight='bold')
ax.set_xlabel('PageRank Normalizado por |V| (Escala Logarítmica)')
ax.set_ylabel('Densidade Normalizada de Nós')
ax.legend(loc='upper right')
save_fig(fig, 'fig7_bipartite_pagerank_dist.png')

# -------------------------------------------------------------------------
# FIGURA 8: Dispersão Financeira (Ratio to Mean)
# -------------------------------------------------------------------------
print("Generating Fig 8: Financial dispersion boxplots...")
ve_stats = pd.read_csv('results/100percent/val_predictions.csv',
                       usecols=['isFraud', 'TransactionAmt_to_mean_uid', 'uid_prev_tx_count'])
multi = ve_stats[ve_stats['uid_prev_tx_count'] >= 1].dropna()

fig, ax = plt.subplots(figsize=(8.5, 4.8))
sns.boxplot(
    data=multi, x='isFraud', y='TransactionAmt_to_mean_uid',
    palette=[PALETTE['legit'], PALETTE['fraud']],
    showfliers=False, ax=ax, width=0.45
)
ax.set_title('Dispersão do Montante em Relação ao Histórico Causal do Portador\n(Transações com Histórico Prévio ≥ 1)', fontweight='bold')
ax.set_xticklabels(['Legítimo (0)', 'Fraude (1)'], fontweight='bold')
ax.set_xlabel('Classe da Transação (isFraud)')
ax.set_ylabel('Razão: Montante / Média Anterior do Portador (TransactionAmt_to_mean_uid)')
save_fig(fig, 'fig8_financial_dispersion_boxplots.png')

# -------------------------------------------------------------------------
# FIGURA 9: Curvas Precision-Recall da Ablação
# -------------------------------------------------------------------------
print("Generating Fig 9: PR Curves of the Ablation...")
y_true = val_preds['isFraud'].values
models_to_plot = [
    ('score_1B_raw', '1B: Baseline (PCA 15-d)', '#457B9D', '-.'),
    ('score_2_raw',  '2: 1B + Topologia', '#2A9D8F', '-'),
    ('score_4_raw',  '4: 1B + Grafos Completos', '#3D5A80', ':'),
    ('score_5_raw',  '5: 1B + FE Comportamental', '#F4A261', '-'),
    ('score_6A_raw', '6A: Completo (PCA 15-d)', '#E63946', '-')
]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
for col_key, label_name, color, style in models_to_plot:
    if col_key in val_preds.columns:
        y_score = val_preds[col_key].values
        p, r, _ = precision_recall_curve(y_true, y_score)
        pr_val = auc(r, p)
        ax.plot(r, p, color=color, linestyle=style, lw=1.8, label=f'{label_name} (PR-AUC={pr_val:.4f})')

ax.set_xlabel('Recall (Revocação)')
ax.set_ylabel('Precision (Precisão)')
ax.set_title('Curvas Precision-Recall na Validação Temporal Causal (100% dos Dados)', fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
save_fig(fig, 'fig9_ablation_pr_roc_curves.png')

# -------------------------------------------------------------------------
# FIGURA 10: Curva de Custo Operacional (na Partição de Teste Independente)
# -------------------------------------------------------------------------
print("Generating Fig 10: Operational Cost Curves on Test Half...")
curves_df = pd.read_csv('results/100percent/cost_curves_test.csv')
cost_summary = pd.read_csv('results/100percent/cost_summary.csv').set_index('ID')

tau = curves_df['tau'].values
fig, ax = plt.subplots(figsize=(8.5, 4.8))

for cid, label, col, style in [
    ('1B', '1B: Baseline (PCA)', '#457B9D', '--'),
    ('2',  '2: 1B + Topologia', '#2A9D8F', '-'),
    ('5',  '5: 1B + FE Comportamental', '#F4A261', '-'),
    ('6A', '6A: Completo', '#E63946', '-')
]:
    loss = curves_df[cid].values
    tau_calib = cost_summary.loc[cid, 'tau_calib']
    perda_calib = cost_summary.loc[cid, 'Perda_teste']
    ax.plot(tau, loss, label=f'{label} (τ_calib={tau_calib:.2f} -> US$ {perda_calib:,.0f})',
            color=col, linestyle=style, lw=2.0)

ax.set_xlabel('Limiar Operacional de Decisão (τ)')
ax.set_ylabel('Perda Operacional Estimada no Teste (US$)')
ax.set_title('Curva de Custo Operacional Sob Risco de Bayes (Metade de Teste)', fontweight='bold')
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('US$ {x:,.0f}'))
ax.legend(loc='upper right', fontsize=8.5)
save_fig(fig, 'fig10_cost_and_typology_breakdown.png')

print("\n✅ Todas as 10 figuras individuais foram regeneradas com 100% de consistência!")
