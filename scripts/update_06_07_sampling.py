import nbformat as nbf

zip_code_prefix = """# ==============================================================================
# RESOLUÇÃO INTELIGENTE DE DADOS (SUPORTE A ZIP, GOOGLE DRIVE OU LOCAL)
# ==============================================================================
import os
import zipfile

def resolve_data_dir():
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

    raise FileNotFoundError("Arquivos CSV ou 'ieee_data.zip' não encontrados.")

DATA_DIR = resolve_data_dir()
train_trans_path = os.path.join(DATA_DIR, 'train_transaction.csv')
train_id_path = os.path.join(DATA_DIR, 'train_identity.csv')
"""

sample_code = """df = df.rename(columns={"TransactionDT": "timestamp"})
df = df.sort_values('timestamp').reset_index(drop=True)

# Amostragem Causal de 50% do dataset (~295k transações) para estabilidade no Colab
SAMPLE_FRACTION = 0.5
print(f"Amostrando {SAMPLE_FRACTION:.0%} do dataset ({int(len(df)*SAMPLE_FRACTION):,} transações) com ordenação temporal causal...")
df = df.sample(frac=SAMPLE_FRACTION, random_state=42).sort_values('timestamp').reset_index(drop=True)
"""

# Update 06
with open('notebooks/06_walk_forward_temporal_ablation_colab.ipynb', 'r', encoding='utf-8') as f:
    nb6 = nbf.read(f, as_version=4)

for cell in nb6.cells:
    if 'def resolve_data_dir():' in cell.source:
        # replace the resolve_data_dir block
        parts = cell.source.split('def reduce_mem_usage')
        cell.source = zip_code_prefix + '\ndef reduce_mem_usage' + parts[1]
    if 'df = df.rename(columns={"TransactionDT": "timestamp"})' in cell.source:
        old_part = 'df = df.rename(columns={"TransactionDT": "timestamp"})\ndf = df.sort_values(\'timestamp\').reset_index(drop=True)'
        if old_part in cell.source:
            cell.source = cell.source.replace(old_part, sample_code)
            print("Notebook 06 sampling updated!")

with open('notebooks/06_walk_forward_temporal_ablation_colab.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb6, f)

# Update 07
with open('notebooks/07_hyperparameter_tuning_and_significance_colab.ipynb', 'r', encoding='utf-8') as f:
    nb7 = nbf.read(f, as_version=4)

for cell in nb7.cells:
    if 'def resolve_data_dir():' in cell.source:
        parts = cell.source.split('def reduce_mem_usage')
        cell.source = zip_code_prefix + '\ndef reduce_mem_usage' + parts[1]
    if 'df = df.rename(columns={"TransactionDT": "timestamp"})' in cell.source:
        old_part = 'df = df.rename(columns={"TransactionDT": "timestamp"})\ndf = df.sort_values(\'timestamp\').reset_index(drop=True)'
        if old_part in cell.source:
            cell.source = cell.source.replace(old_part, sample_code)
            print("Notebook 07 sampling updated!")

with open('notebooks/07_hyperparameter_tuning_and_significance_colab.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb7, f)

print("Notebooks 06 e 07 atualizados com sucesso com 50% de dados e zip resolution!")
