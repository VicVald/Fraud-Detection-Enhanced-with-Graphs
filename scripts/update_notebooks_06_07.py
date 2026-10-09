import nbformat as nbf

# ----------------------------------------------------
# 1. Update Notebook 06
# ----------------------------------------------------
with open('notebooks/06_walk_forward_temporal_ablation_colab.ipynb') as f:
    nb6 = nbf.read(f, as_version=4)

# In cell 3 (config), add EXECUTION_BUDGET
budget_code = """# ==============================================================================
# CONFIGURAÇÃO DO ORÇAMENTO EXPERIMENTAL (EXECUTION BUDGET):
# - 'ESSENTIAL': Executa com amostra temporal e 200 árvores (~30 min).
# - 'FULL'     : 100% dos dados e 450 árvores nas 4 dobras (~1h30 - 2h).
# ==============================================================================
EXECUTION_BUDGET = 'FULL'  # Opções: 'FULL' ou 'ESSENTIAL'
"""
nb6.cells[3].source = nb6.cells[3].source + "\n\n" + budget_code

# In cell 8, replace add_velocity with causal UID-grouped velocity
old_vel_code = """    def add_velocity(d, col_name):
        d_sorted = d.sort_values('timestamp')
        times = d_sorted['timestamp'].values
        left_idx = np.searchsorted(times, times - 86400, side='left')
        d[col_name] = np.arange(len(times)) - left_idx + 1
        return d
        
    train_df = add_velocity(train_df, 'uid_tx_count_24h')
    val_df = add_velocity(val_df, 'uid_tx_count_24h')"""

new_vel_code = """    # 4. Velocidade Causal 24h Agrupada ESTRITAMENTE por UID (Sem vazamento)
    def compute_causal_velocity(d):
        d_sorted = d.sort_values('timestamp').copy()
        times = d_sorted['timestamp'].values
        counts = np.ones(len(d_sorted), dtype=np.int32)
        uid_indices = d_sorted.groupby('uid', sort=False).indices
        for uid_key, idxs in uid_indices.items():
            t_uid = times[idxs]
            left = np.searchsorted(t_uid, t_uid - 86400, side='left')
            counts[idxs] = np.arange(len(t_uid)) - left + 1
        d_sorted['uid_tx_count_24h'] = counts
        return d_sorted.sort_index()

    train_df = compute_causal_velocity(train_df)
    val_df = compute_causal_velocity(val_df)"""

nb6.cells[8].source = nb6.cells[8].source.replace(old_vel_code, new_vel_code)

with open('notebooks/06_walk_forward_temporal_ablation_colab.ipynb', 'w') as f:
    nbf.write(nb6, f)
print("Notebook 06 updated!")

# ----------------------------------------------------
# 2. Update Notebook 07
# ----------------------------------------------------
with open('notebooks/07_hyperparameter_tuning_and_significance_colab.ipynb') as f:
    nb7 = nbf.read(f, as_version=4)

# In cell 3, add EXECUTION_BUDGET
nb7.cells[3].source = nb7.cells[3].source + "\n\n" + budget_code

# In cell 7, replace add_velocity with causal UID-grouped velocity
old_vel_code_7 = """    def add_velocity(d, col_name):
        d_sorted = d.sort_values('timestamp')
        times = d_sorted['timestamp'].values
        left_idx = np.searchsorted(times, times - 86400, side='left')
        d[col_name] = np.arange(len(times)) - left_idx + 1
        return d
        
    t_df = add_velocity(t_df, 'uid_tx_count_24h')
    v_df = add_velocity(v_df, 'uid_tx_count_24h')"""

new_vel_code_7 = """    # 4. Velocidade Causal 24h Agrupada ESTRITAMENTE por UID (Sem vazamento)
    def compute_causal_velocity(d):
        d_sorted = d.sort_values('timestamp').copy()
        times = d_sorted['timestamp'].values
        counts = np.ones(len(d_sorted), dtype=np.int32)
        uid_indices = d_sorted.groupby('uid', sort=False).indices
        for uid_key, idxs in uid_indices.items():
            t_uid = times[idxs]
            left = np.searchsorted(t_uid, t_uid - 86400, side='left')
            counts[idxs] = np.arange(len(t_uid)) - left + 1
        d_sorted['uid_tx_count_24h'] = counts
        return d_sorted.sort_index()

    t_df = compute_causal_velocity(t_df)
    v_df = compute_causal_velocity(v_df)"""

nb7.cells[7].source = nb7.cells[7].source.replace(old_vel_code_7, new_vel_code_7)

with open('notebooks/07_hyperparameter_tuning_and_significance_colab.ipynb', 'w') as f:
    nbf.write(nb7, f)
print("Notebook 07 updated!")
