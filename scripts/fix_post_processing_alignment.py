import nbformat

# 1. Fix Notebook 04
with open('notebooks/04_ablation_with_vs_without_graphs_colab.ipynb', 'r', encoding='utf-8') as f:
    nb04 = nbformat.read(f, as_version=4)

fixed_nb04 = False
for cell in nb04.cells:
    if "expanding_mean = val_temp_sorted.groupby('uid')['raw_score'].expanding().mean().values" in cell.source:
        cell.source = cell.source.replace(
            "expanding_mean = val_temp_sorted.groupby('uid')['raw_score'].expanding().mean().values\n    val_temp_sorted['score_pp'] = 0.7 * val_temp_sorted['raw_score'] + 0.3 * expanding_mean",
            "expanding_mean = val_temp_sorted.groupby('uid')['raw_score'].expanding().mean().reset_index(level=0, drop=True)\n    val_temp_sorted['score_pp'] = 0.7 * val_temp_sorted['raw_score'] + 0.3 * expanding_mean"
        )
        fixed_nb04 = True
        print("✓ Notebook 04 post-processing line fixed!")

if fixed_nb04:
    with open('notebooks/04_ablation_with_vs_without_graphs_colab.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb04, f)

# 2. Fix Notebook 03
with open('notebooks/03_full_sota_pipeline_colab.ipynb', 'r', encoding='utf-8') as f:
    nb03 = nbformat.read(f, as_version=4)

fixed_nb03 = False
old_pp_nb03 = """# Pós-processamento de Cliente (FraudSquad Technique)
# Substitui a probabilidade pela média do UID quando há múltiplas transações no mesmo portador
val_df['raw_pred'] = y_score
uid_pred_mean = val_df.groupby('uid')['raw_pred'].transform('mean')
y_score_pp = 0.7 * y_score + 0.3 * uid_pred_mean"""

new_pp_nb03 = """# Pós-processamento Causal de Cliente (Expanding Mean estritamente temporal)
# Suaviza a probabilidade apenas pelo histórico passado e presente observado do UID (sem lookahead futuro)
val_temp_pp = val_df[['uid', 'timestamp']].copy()
val_temp_pp['raw_score'] = y_score
val_temp_pp_sorted = val_temp_pp.sort_values('timestamp')
exp_mean_causal = val_temp_pp_sorted.groupby('uid')['raw_score'].expanding().mean().reset_index(level=0, drop=True)
val_temp_pp_sorted['score_pp'] = 0.7 * val_temp_pp_sorted['raw_score'] + 0.3 * exp_mean_causal
y_score_pp = val_temp_pp_sorted.sort_index()['score_pp'].values"""

for cell in nb03.cells:
    if "uid_pred_mean = val_df.groupby('uid')['raw_pred'].transform('mean')" in cell.source:
        cell.source = cell.source.replace(old_pp_nb03, new_pp_nb03)
        fixed_nb03 = True
        print("✓ Notebook 03 post-processing causal alignment fixed!")

if fixed_nb03:
    with open('notebooks/03_full_sota_pipeline_colab.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb03, f)
