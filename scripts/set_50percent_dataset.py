import nbformat as nbf
import os

alert_banner = """> ⚠️ **Ajuste de Escala Operacional (50% do Dataset):**  
> Para garantir estabilidade absoluta e evitar estouro de memória RAM (*Out-Of-Memory* / OOM) no Google Colab gratuito (~12 GB de RAM), este notebook foi configurado para operar com uma amostra representativa de **50% do dataset IEEE-CIS (~295.270 transações)**. A amostragem preserva integralmente a distribuição temporal de 182 dias e a ordem cronológica causal estrita (sem *data leakage*).
"""

# ==============================================================================
# 1. NOTEBOOK 01
# ==============================================================================
with open('notebooks/01_eda_and_feature_selection.ipynb') as f:
    nb1 = nbf.read(f, as_version=4)

# Update Cell 0 markdown with alert banner
if alert_banner not in nb1.cells[0].source:
    nb1.cells[0].source = nb1.cells[0].source + "\n\n" + alert_banner

# Update Cell 1 with memory-efficient load_ieee_train
cell1_updated = """import sys
sys.path.insert(0, '..')   # Permitir imports da raiz do projeto

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from IPython.display import display
except ImportError:
    display = print

# Configuração de estilo visual acadêmico
plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})
sns.set_style('whitegrid')

PALETTE = {'fraud': '#E63946', 'legit': '#1D3557', 'neutral': '#457B9D', 'accent': '#F4A261', 'highlight': '#2A9D8F'}

def reduce_mem_usage(df, verbose=False):
    \"\"\"Converte tipos numéricos float64/int64 para economizar RAM mantendo precisão.\"\"\"
    start_mem = df.memory_usage().sum() / 1024**2
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"RAM reduzida de {start_mem:.2f} MB para {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% economia)")
    return df

def load_ieee_train(data_dir=None, sample_fraction=0.5, verbose=True):
    \"\"\"Carrega e mescla as tabelas do IEEE-CIS com otimização severa de RAM.\"\"\"
    tx_path = os.path.join(data_dir, 'train_transaction.csv')
    id_path = os.path.join(data_dir, 'train_identity.csv')
    
    if verbose:
        print(f'[load_ieee] Carregando transações de {tx_path}...')
    tx_df = pd.read_csv(tx_path)
    tx_df = reduce_mem_usage(tx_df, verbose=verbose)
    
    if os.path.exists(id_path):
        if verbose:
            print(f'[load_ieee] Carregando identidades de {id_path}...')
        id_df = pd.read_csv(id_path)
        id_df = reduce_mem_usage(id_df, verbose=verbose)
        if verbose:
            print('[load_ieee] Mesclando tabelas...')
        merged = tx_df.merge(id_df, on='TransactionID', how='left')
        del tx_df, id_df
        gc.collect()
    else:
        merged = tx_df
        
    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        if verbose:
            print(f'[load_ieee] Amostrando {sample_fraction:.0%} do dataset ({int(len(merged)*sample_fraction):,} linhas)...')
        merged = merged.sample(frac=sample_fraction, random_state=42).sort_values('TransactionDT').reset_index(drop=True)
        
    if verbose:
        print(f'[load_ieee] Pronto: {merged.shape[0]:,} transações × {merged.shape[1]} colunas | Taxa de Fraude: {merged[\"isFraud\"].mean():.3%}')
    return merged

def temporal_split(df, boundary=0.8, verbose=True):
    \"\"\"Split temporal estrito anti-leakage (80% treino / 20% validação).\"\"\"
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df_sorted) * boundary)
    train_df = df_sorted.iloc[:split_idx].copy()
    val_df = df_sorted.iloc[split_idx:].copy()
    if verbose:
        n_train, n_val = len(train_df), len(val_df)
        n_total = n_train + n_val
        print(f'[split] Temporal split estrito:')
        print(f'  Treino   : {n_train:,} ({n_train/n_total:.1%}) | Fraude: {train_df[\"isFraud\"].mean():.3%}')
        print(f'  Validação: {n_val:,} ({n_val/n_total:.1%}) | Fraude: {val_df[\"isFraud\"].mean():.3%}')
    return train_df, val_df

print("✅ Ambiente e funções de memória configurados com sucesso.")
"""
nb1.cells[1].source = cell1_updated

# Update Cell 3 to sample 50% and clean up df_raw
cell3_text = nb1.cells[3].source
cell3_text = cell3_text.replace("sample_fraction=1.0", "sample_fraction=0.5")
cell3_text = cell3_text.replace(
    "train_df, val_df = temporal_split(df_raw, boundary=0.8, verbose=True)",
    "train_df, val_df = temporal_split(df_raw, boundary=0.8, verbose=True)\ndel df_raw\ngc.collect()"
)
nb1.cells[3].source = cell3_text

with open('notebooks/01_eda_and_feature_selection.ipynb', 'w') as f:
    nbf.write(nb1, f)
print("Notebook 01 atualizado com sucesso!")


# ==============================================================================
# 2. NOTEBOOK 03
# ==============================================================================
with open('notebooks/03_full_sota_pipeline_colab.ipynb') as f:
    nb3 = nbf.read(f, as_version=4)

if alert_banner not in nb3.cells[0].source:
    nb3.cells[0].source = nb3.cells[0].source + "\n\n" + alert_banner

for cell in nb3.cells:
    if "df = df_trans.merge(df_id, on='TransactionID', how='left')" in cell.source:
        sampling_snippet = """df = df.rename(columns={"TransactionDT": "timestamp"})

# Amostragem de 50% para viabilidade de RAM no Colab
SAMPLE_FRACTION = 0.5
if SAMPLE_FRACTION < 1.0:
    print(f"Amostrando {SAMPLE_FRACTION:.0%} do dataset ({int(len(df)*SAMPLE_FRACTION):,} transações) com ordenação temporal...")
    df = df.sample(frac=SAMPLE_FRACTION, random_state=42).sort_values('timestamp').reset_index(drop=True)

print(f"Dataset consolidado (50%): {df.shape[0]:,} transações × {df.shape[1]} colunas.")"""
        
        # Replace the part after del df_trans, df_id; gc.collect()
        if "df = df.rename(columns={\"TransactionDT\": \"timestamp\"})" in cell.source:
            parts = cell.source.split("df = df.rename(columns={\"TransactionDT\": \"timestamp\"})")
            cell.source = parts[0] + sampling_snippet
            print("Notebook 03 atualizado com amostragem de 50%!")
            break

with open('notebooks/03_full_sota_pipeline_colab.ipynb', 'w') as f:
    nbf.write(nb3, f)


# ==============================================================================
# 3. NOTEBOOK 04
# ==============================================================================
with open('notebooks/04_ablation_with_vs_without_graphs_colab.ipynb') as f:
    nb4 = nbf.read(f, as_version=4)

if alert_banner not in nb4.cells[0].source:
    nb4.cells[0].source = nb4.cells[0].source + "\n\n" + alert_banner

# In cell 5, apply 50% sampling
for cell in nb4.cells:
    if "df = df_trans.merge(df_id, on='TransactionID', how='left')" in cell.source:
        old_part = """df = df.rename(columns={"TransactionDT": "timestamp"})
df = df.sort_values('timestamp').reset_index(drop=True)

# Aplicação do Orçamento Computacional
if EXECUTION_BUDGET == 'ESSENTIAL':
    print("Modo ESSENTIAL: Amostrando 20% das transações preservando ordem temporal causal...")
    df = df.iloc[-int(len(df) * 0.2):].reset_index(drop=True)"""

        new_part = """df = df.rename(columns={"TransactionDT": "timestamp"})
df = df.sort_values('timestamp').reset_index(drop=True)

# Aplicação do Orçamento Computacional (Configurado para 50% das transações para estabilidade de RAM)
SAMPLE_FRACTION = 0.5
print(f"Amostrando {SAMPLE_FRACTION:.0%} do dataset ({int(len(df)*SAMPLE_FRACTION):,} transações) preservando ordem temporal estrita...")
df = df.sample(frac=SAMPLE_FRACTION, random_state=42).sort_values('timestamp').reset_index(drop=True)"""
        cell.source = cell.source.replace(old_part, new_part)
        print("Notebook 04 atualizado com amostragem de 50%!")
        break

with open('notebooks/04_ablation_with_vs_without_graphs_colab.ipynb', 'w') as f:
    nbf.write(nb4, f)


# ==============================================================================
# 4. NOTEBOOK 05
# ==============================================================================
with open('notebooks/05_fraud_typology_cost_function_analysis.ipynb') as f:
    nb5 = nbf.read(f, as_version=4)

if alert_banner not in nb5.cells[0].source:
    nb5.cells[0].source = nb5.cells[0].source + "\n\n" + alert_banner

for cell in nb5.cells:
    if "df = df_trans.merge(df_id, on='TransactionID', how='left')" in cell.source:
        cell.source = cell.source.replace(
            "df = df.rename(columns={'TransactionDT': 'timestamp'})",
            "df = df.rename(columns={'TransactionDT': 'timestamp'})\n    df = df.sample(frac=0.5, random_state=42).sort_values('timestamp').reset_index(drop=True)"
        )
        print("Notebook 05 atualizado com amostragem de 50%!")
        break

with open('notebooks/05_fraud_typology_cost_function_analysis.ipynb', 'w') as f:
    nbf.write(nb5, f)


# ==============================================================================
# 5. NOTEBOOK 06
# ==============================================================================
with open('notebooks/06_walk_forward_temporal_ablation_colab.ipynb') as f:
    nb6 = nbf.read(f, as_version=4)

if alert_banner not in nb6.cells[0].source:
    nb6.cells[0].source = nb6.cells[0].source + "\n\n" + alert_banner

for cell in nb6.cells:
    if "df = df_trans.merge(df_id, on='TransactionID', how='left')" in cell.source:
        old_p = """df = df.rename(columns={"TransactionDT": "timestamp"})
df = df.sort_values('timestamp').reset_index(drop=True)"""
        new_p = """df = df.rename(columns={"TransactionDT": "timestamp"})
df = df.sort_values('timestamp').reset_index(drop=True)

# Amostragem de 50% para viabilidade de RAM no Colab
SAMPLE_FRACTION = 0.5
print(f"Amostrando {SAMPLE_FRACTION:.0%} do dataset ({int(len(df)*SAMPLE_FRACTION):,} transações) preservando ordem temporal...")
df = df.sample(frac=SAMPLE_FRACTION, random_state=42).sort_values('timestamp').reset_index(drop=True)"""
        cell.source = cell.source.replace(old_p, new_p)
        print("Notebook 06 atualizado com amostragem de 50%!")
        break

with open('notebooks/06_walk_forward_temporal_ablation_colab.ipynb', 'w') as f:
    nbf.write(nb6, f)


# ==============================================================================
# 6. NOTEBOOK 07
# ==============================================================================
with open('notebooks/07_hyperparameter_tuning_and_significance_colab.ipynb') as f:
    nb7 = nbf.read(f, as_version=4)

if alert_banner not in nb7.cells[0].source:
    nb7.cells[0].source = nb7.cells[0].source + "\n\n" + alert_banner

for cell in nb7.cells:
    if "df = df_trans.merge(df_id, on='TransactionID', how='left')" in cell.source:
        old_p = """df = df.rename(columns={"TransactionDT": "timestamp"})
df = df.sort_values('timestamp').reset_index(drop=True)"""
        new_p = """df = df.rename(columns={"TransactionDT": "timestamp"})
df = df.sort_values('timestamp').reset_index(drop=True)

# Amostragem de 50% para viabilidade de RAM no Colab
SAMPLE_FRACTION = 0.5
print(f"Amostrando {SAMPLE_FRACTION:.0%} do dataset ({int(len(df)*SAMPLE_FRACTION):,} transações) preservando ordem temporal...")
df = df.sample(frac=SAMPLE_FRACTION, random_state=42).sort_values('timestamp').reset_index(drop=True)"""
        cell.source = cell.source.replace(old_p, new_p)
        print("Notebook 07 atualizado com amostragem de 50%!")
        break

with open('notebooks/07_hyperparameter_tuning_and_significance_colab.ipynb', 'w') as f:
    nbf.write(nb7, f)

print("Todas as alterações nos notebooks foram concluídas!")
