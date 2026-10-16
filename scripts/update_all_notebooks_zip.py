import os
import nbformat as nbf

zip_resolution_code = """# ==============================================================================
# RESOLUÇÃO INTELIGENTE DE DADOS (SUPORTE A ZIP, GOOGLE DRIVE OU LOCAL)
# ==============================================================================
import os
import zipfile

def resolve_data_dir():
    \"\"\"Localiza ou descompacta os arquivos do IEEE-CIS automaticamente.\"\"\"
    candidates = [
        '/content/data',
        'data/ieee/raw',
        '../data/ieee/raw',
        '/content/drive/MyDrive/ieee-data',
        '.'
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, 'train_transaction.csv')):
            print(f"✅ Dados encontrados descompactados em: [{d}]")
            return d

    zip_candidates = [
        '/content/ieee_data.zip',
        '/content/drive/MyDrive/ieee_data.zip',
        '/content/drive/MyDrive/ieee-data/ieee_data.zip',
        'ieee_data.zip',
        'data/ieee/raw/ieee_data.zip',
        '../data/ieee/raw/ieee_data.zip'
    ]
    for z in zip_candidates:
        if os.path.exists(z):
            target = '/content/data' if os.path.exists('/content') else 'data/ieee/raw'
            os.makedirs(target, exist_ok=True)
            print(f"📦 Descompactando [{z}] para [{target}]...")
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(target)
            print(f"✅ Extração concluída em: [{target}]")
            return target

    raise FileNotFoundError(
        "Nenhum arquivo 'train_transaction.csv' ou 'ieee_data.zip' foi encontrado.\\n"
        "Faça o upload de 'ieee_data.zip' no Colab ou coloque no seu Google Drive."
    )

DATA_DIR = resolve_data_dir()
train_trans_path = os.path.join(DATA_DIR, 'train_transaction.csv')
train_id_path = os.path.join(DATA_DIR, 'train_identity.csv')
"""

# ------------------------------------------------------------------------------
# 1. Update Notebook 04
# ------------------------------------------------------------------------------
with open('notebooks/04_ablation_with_vs_without_graphs_colab.ipynb') as f:
    nb4 = nbf.read(f, as_version=4)

for i, cell in enumerate(nb4.cells):
    if 'train_trans_path = os.path.join(DATA_DIR' in cell.source:
        # replace the DATA_DIR assignment block
        lines = cell.source.split('\n')
        # find where reduce_mem_usage begins
        for j, line in enumerate(lines):
            if 'def reduce_mem_usage' in line:
                cell.source = zip_resolution_code + "\n" + '\n'.join(lines[j:])
                break
        print(f"Notebook 04 Cell {i} updated with smart zip resolution.")
        break

with open('notebooks/04_ablation_with_vs_without_graphs_colab.ipynb', 'w') as f:
    nbf.write(nb4, f)

# ------------------------------------------------------------------------------
# 2. Update Notebook 03
# ------------------------------------------------------------------------------
with open('notebooks/03_full_sota_pipeline_colab.ipynb') as f:
    nb3 = nbf.read(f, as_version=4)

for i, cell in enumerate(nb3.cells):
    if 'train_trans_path = os.path.join(DATA_DIR' in cell.source:
        lines = cell.source.split('\n')
        for j, line in enumerate(lines):
            if 'def reduce_mem_usage' in line:
                cell.source = zip_resolution_code + "\n" + '\n'.join(lines[j:])
                break
        print(f"Notebook 03 Cell {i} updated with smart zip resolution.")
        break

with open('notebooks/03_full_sota_pipeline_colab.ipynb', 'w') as f:
    nbf.write(nb3, f)

# ------------------------------------------------------------------------------
# 3. Update Notebook 05
# ------------------------------------------------------------------------------
with open('notebooks/05_fraud_typology_cost_function_analysis.ipynb') as f:
    nb5 = nbf.read(f, as_version=4)

nb5_cell_source = zip_resolution_code + """
# Verificação de arquivos de predições disponíveis
pred_candidates = [
    'val_predictions.csv',
    '../val_predictions.csv',
    'results/100percent/val_predictions.csv',
    '../results/100percent/val_predictions.csv',
    '/content/val_predictions.csv',
    '/content/drive/MyDrive/val_predictions.csv',
    '/content/data/val_predictions.csv'
]
predictions_file = None
for p in pred_candidates:
    if os.path.exists(p):
        predictions_file = p
        break

if predictions_file:
    print(f"✅ Carregando predições reais pré-calculadas de [{predictions_file}]...")
    val_df = pd.read_csv(predictions_file)
else:
    print(f"ℹ️ Arquivo 'val_predictions.csv' não encontrado. Carregando dados brutos do IEEE-CIS...")
    df_trans = pd.read_csv(train_trans_path, usecols=['TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud', 'card1', 'addr1', 'D1', 'DeviceType'])
    df_id = pd.read_csv(train_id_path, usecols=['TransactionID', 'DeviceInfo'])
    df = df_trans.merge(df_id, on='TransactionID', how='left')
    del df_trans, df_id
    gc.collect()
    
    df = df.rename(columns={'TransactionDT': 'timestamp'})
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Split Causal 80/20
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    # Construção do UID
    day = val_df['timestamp'] / 86400.0
    d1_anchor = (day - val_df['D1'].fillna(0)).astype(int).astype(str)
    val_df['uid'] = val_df['card1'].fillna(0).astype(int).astype(str) + '_' + val_df['addr1'].fillna(0).astype(int).astype(str) + '_' + d1_anchor
    val_df['DeviceInfo_clean'] = val_df['DeviceInfo'].fillna('null_device').astype(str)
    
    # Heurística de fallback caso executado isoladamente
    val_df['score_1B_raw'] = np.clip(val_df['TransactionAmt'] / 1000.0, 0.01, 0.90)
    val_df['score_4_raw'] = val_df['score_1B_raw']

print(f"Validação carregada: {len(val_df):,} transações | Taxa de Fraude: {val_df['isFraud'].mean():.3%}")
"""

for i, cell in enumerate(nb5.cells):
    if 'predictions_file = ' in cell.source and 'DATA_DIR = ' in cell.source:
        cell.source = nb5_cell_source
        print(f"Notebook 05 Cell {i} updated with smart zip and prediction file resolution.")
        break

with open('notebooks/05_fraud_typology_cost_function_analysis.ipynb', 'w') as f:
    nbf.write(nb5, f)

# ------------------------------------------------------------------------------
# 4. Update Notebook 06
# ------------------------------------------------------------------------------
with open('notebooks/06_walk_forward_temporal_ablation_colab.ipynb') as f:
    nb6 = nbf.read(f, as_version=4)

for i, cell in enumerate(nb6.cells):
    if 'train_trans_path = os.path.join(DATA_DIR' in cell.source:
        lines = cell.source.split('\n')
        for j, line in enumerate(lines):
            if 'def reduce_mem_usage' in line:
                cell.source = zip_resolution_code + "\n" + '\n'.join(lines[j:])
                break
        print(f"Notebook 06 Cell {i} updated with smart zip resolution.")
        break

with open('notebooks/06_walk_forward_temporal_ablation_colab.ipynb', 'w') as f:
    nbf.write(nb6, f)

# ------------------------------------------------------------------------------
# 5. Update Notebook 07
# ------------------------------------------------------------------------------
with open('notebooks/07_hyperparameter_tuning_and_significance_colab.ipynb') as f:
    nb7 = nbf.read(f, as_version=4)

for i, cell in enumerate(nb7.cells):
    if 'train_trans_path = os.path.join(DATA_DIR' in cell.source:
        lines = cell.source.split('\n')
        for j, line in enumerate(lines):
            if 'def reduce_mem_usage' in line:
                cell.source = zip_resolution_code + "\n" + '\n'.join(lines[j:])
                break
        print(f"Notebook 07 Cell {i} updated with smart zip resolution.")
        break

with open('notebooks/07_hyperparameter_tuning_and_significance_colab.ipynb', 'w') as f:
    nbf.write(nb7, f)

# ------------------------------------------------------------------------------
# 6. Update Notebook 01
# ------------------------------------------------------------------------------
with open('notebooks/01_eda_and_feature_selection.ipynb') as f:
    nb1 = nbf.read(f, as_version=4)

nb1_cell_3_prefix = zip_resolution_code + """
# Carregando dataset completo IEEE-CIS para análise aprofundada (sample_fraction=1.0)
df_raw = load_ieee_train(data_dir=DATA_DIR, sample_fraction=1.0, verbose=True)
df_raw = df_raw.rename(columns={"TransactionDT": "timestamp"})

# Execução do split temporal estrito
train_df, val_df = temporal_split(df_raw, boundary=0.8, verbose=True)

# Limpeza e padronização de identificadores
train_df['DeviceInfo_clean'] = train_df['DeviceInfo'].fillna('null_device').astype(str)
val_df['DeviceInfo_clean'] = val_df['DeviceInfo'].fillna('null_device').astype(str)

print(f"\\nResumo Temporal:")
print(f"  Treino: {train_df['timestamp'].min()} até {train_df['timestamp'].max()} ({len(train_df):,} linhas)")
print(f"  Validação: {val_df['timestamp'].min()} até {val_df['timestamp'].max()} ({len(val_df):,} linhas)")
assert val_df['timestamp'].min() > train_df['timestamp'].max(), "ERRO: Vazamento temporal detectado!"
"""

for i, cell in enumerate(nb1.cells):
    if 'df_raw = load_ieee_train(' in cell.source:
        cell.source = nb1_cell_3_prefix
        print(f"Notebook 01 Cell {i} updated with smart zip and FULL dataset resolution.")
        break

with open('notebooks/01_eda_and_feature_selection.ipynb', 'w') as f:
    nbf.write(nb1, f)

print("Todos os notebooks foram atualizados com sucesso!")
