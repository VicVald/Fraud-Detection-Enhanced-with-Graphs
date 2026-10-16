import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

df = pd.read_csv('val_predictions.csv')
y_true = df['isFraud'].values
val_temp = df[['uid', 'timestamp']].copy()

configs = [
    ('1A', 'Baseline Raw (Sem PCA)', 'score_1A_raw'),
    ('1B', 'Baseline (Com PCA 15-d)', 'score_1B_raw'),
    ('2', 'Baseline + Apenas Topologia', 'score_2_raw'),
    ('3', 'Baseline + Apenas Velocidade', 'score_3_raw'),
    ('4', 'Baseline + Grafos Completos', 'score_4_raw'),
    ('5', 'Baseline + Apenas FE Tabular', 'score_5_raw'),
    ('6A', 'All-in-One (Com PCA 15-d)', 'score_6A_raw'),
    ('6B', 'All-in-One (Sem PCA / Raw V)', 'score_6B_raw'),
]

ablation_rows = []

# Read current ablation summary to preserve metadata like features and time
ablation_old = pd.read_csv('ablation_summary.csv')

for idx, (cid, cname, col) in enumerate(configs):
    raw_s = df[col].values
    
    # Correct causal expanding mean
    val_temp['raw_score'] = raw_s
    val_temp_sorted = val_temp.sort_values('timestamp')
    exp_series = val_temp_sorted.groupby('uid')['raw_score'].expanding().mean().reset_index(level=0, drop=True)
    val_temp_sorted['score_pp'] = 0.7 * val_temp_sorted['raw_score'] + 0.3 * exp_series
    cor_s = val_temp_sorted.sort_index()['score_pp'].values
    
    # Overwrite column in df
    df[f'score_{cid}_pp'] = cor_s
    
    # Metrics
    p_r, r_r, _ = precision_recall_curve(y_true, raw_s)
    pr_raw = auc(r_r, p_r)
    f1_r = np.max(2 * (p_r * r_r) / (p_r + r_r + 1e-8))
    roc_raw = roc_auc_score(y_true, raw_s)
    
    p_c, r_c, _ = precision_recall_curve(y_true, cor_s)
    pr_cor = auc(r_c, p_c)
    f1_c = np.max(2 * (p_c * r_c) / (p_c + r_c + 1e-8))
    
    time_str = ablation_old.loc[ablation_old['ID'] == cid, 'Tempo (s)'].values[0] if cid in ablation_old['ID'].values else '50.0s'
    feats_str = ablation_old.loc[ablation_old['ID'] == cid, 'Features (A/B)'].values[0] if cid in ablation_old['ID'].values else '-'
    
    ablation_rows.append({
        'ID': cid,
        'Configuração': cname,
        'Features (A/B)': feats_str,
        'PR-AUC (Bruto)': f'{pr_raw:.4f}',
        'ROC-AUC (Bruto)': f'{roc_raw:.4f}',
        'F1 (Bruto)': f'{f1_r:.4f}',
        'PR-AUC (Pós-Proc)': f'{pr_cor:.4f}',
        'F1 (Pós-Proc)': f'{f1_c:.4f}',
        'Tempo (s)': time_str
    })

ablation_new = pd.DataFrame(ablation_rows)
ablation_new.to_csv('ablation_summary.csv', index=False)
ablation_new.to_csv('notebooks/ablation_summary.csv', index=False)
df.to_csv('val_predictions.csv', index=False)

print("✓ val_predictions.csv e ablation_summary.csv atualizados com sucesso com o alinhamento correto!")
print(ablation_new.to_string())
