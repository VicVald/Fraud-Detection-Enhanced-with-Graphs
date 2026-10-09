import os
import pandas as pd
import numpy as np
import networkx as nx

val_df = pd.read_csv('results/val_predictions.csv')
print(f"Predições carregadas: {len(val_df):,} linhas.")

# 1. Classificação de Tipologia
df_sorted = val_df.sort_values('timestamp').reset_index(drop=True)

B = nx.Graph()
dev_edges = df_sorted[df_sorted['DeviceInfo_clean'] != 'null_device'][['uid', 'DeviceInfo_clean']].drop_duplicates()
for _, r in dev_edges.iterrows():
    B.add_edge(f"uid_{r['uid']}", f"dev_{r['DeviceInfo_clean']}")

shared_devs = set([d.replace('dev_', '') for d in B.nodes if d.startswith('dev_') and B.degree(d) >= 2])
df_sorted['is_shared_device'] = df_sorted['DeviceInfo_clean'].isin(shared_devs)

times = df_sorted['timestamp'].values
counts = np.ones(len(df_sorted), dtype=np.int32)
uid_indices = df_sorted.groupby('uid', sort=False).indices
for uid, idxs in uid_indices.items():
    t_uid = times[idxs]
    left = np.searchsorted(t_uid, t_uid - 86400, side='left')
    counts[idxs] = np.arange(len(t_uid)) - left + 1
df_sorted['uid_velocity_24h'] = counts

is_network = (df_sorted['is_shared_device'] == True) | (df_sorted['uid_velocity_24h'] >= 2)

df_sorted['fraud_typology'] = 'Legítimo'
df_sorted.loc[(df_sorted['isFraud'] == 1) & (is_network), 'fraud_typology'] = 'Fraude Coordenada (Grafo/Rede)'
df_sorted.loc[(df_sorted['isFraud'] == 1) & (~is_network), 'fraud_typology'] = 'Fraude Isolada (Avulsa)'

print("\n--- DISTRIBUIÇÃO POR TIPOLOGIA ---")
print(df_sorted['fraud_typology'].value_counts())

total_loss = df_sorted[df_sorted['isFraud'] == 1]['TransactionAmt'].sum()
finance_summary = df_sorted.groupby('fraud_typology').agg(
    Volume=('TransactionAmt', 'count'),
    Total_Dolar=('TransactionAmt', 'sum'),
    Ticket_Medio=('TransactionAmt', 'mean'),
    Mediana=('TransactionAmt', 'median')
).reset_index()
finance_summary['Pct_Prejuizo'] = np.where(
    finance_summary['fraud_typology'] == 'Legítimo',
    0.0,
    finance_summary['Total_Dolar'] / total_loss * 100
)
print("\n--- RESUMO FINANCEIRO POR TIPOLOGIA ---")
print(finance_summary.to_string(index=False))

# 2. Recall por Tipologia no Limiar Ótimo
th_base = 0.237879  # de cost_summary para 1B
th_graph = 0.228384 # de cost_summary para 4
th_aio = 0.085960   # de cost_summary para 6A

pred_base = (df_sorted['score_1B_raw'] >= th_base).astype(int)
pred_graph = (df_sorted['score_4_raw'] >= th_graph).astype(int)
pred_aio = (df_sorted['score_6A_raw'] >= th_aio).astype(int)

df_sorted['pred_base'] = pred_base
df_sorted['pred_graph'] = pred_graph
df_sorted['pred_aio'] = pred_aio

det_records = []
for typ in ['Fraude Coordenada (Grafo/Rede)', 'Fraude Isolada (Avulsa)']:
    sub = df_sorted[df_sorted['fraud_typology'] == typ]
    det_records.append({
        'Tipologia': typ,
        'N_Fraudes': len(sub),
        'Recall_Baseline': sub['pred_base'].mean() * 100,
        'Recall_Grafos': sub['pred_graph'].mean() * 100,
        'Recall_All_in_One': sub['pred_aio'].mean() * 100,
    })
det_df = pd.DataFrame(det_records)
print("\n--- RECALL (%) NO LIMIAR FINANCEIRO ÓTIMO ---")
print(det_df.to_string(index=False))
