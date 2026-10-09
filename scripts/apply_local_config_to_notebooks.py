import nbformat as nbf

# 1. Notebook 04
with open('notebooks/04_ablation_with_vs_without_graphs_colab.ipynb') as f:
    nb4 = nbf.read(f, as_version=4)

# Update budget and data dir in nb4
nb4.cells[2].source = nb4.cells[2].source.replace("EXECUTION_BUDGET = 'FULL'", "EXECUTION_BUDGET = 'ESSENTIAL'")

# Fix reduce_mem_usage in cell 3
cell3_old = """def reduce_mem_usage(df, verbose=True):
    \"\"\"Converte tipos numéricos float64/int64 para economizar RAM mantendo precisão.\"\"\"
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_categorical_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)"""

cell3_new = """def reduce_mem_usage(df, verbose=True):
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
                    df[col] = df[col].astype(np.float64)"""

nb4.cells[3].source = nb4.cells[3].source.replace(cell3_old, cell3_new)

# Add local path support to cell 3
nb4.cells[3].source = nb4.cells[3].source.replace(
    "if os.path.exists('/content/drive/MyDrive/ieee-data'):\n    DATA_DIR = '/content/drive/MyDrive/ieee-data'",
    "if os.path.exists('/content/drive/MyDrive/ieee-data'):\n    DATA_DIR = '/content/drive/MyDrive/ieee-data'\nelif os.path.exists('data/ieee/raw'):\n    DATA_DIR = 'data/ieee/raw'"
)

# Use n_jobs=8 in xgb in cell 6
nb4.cells[6].source = nb4.cells[6].source.replace("random_state=42", "n_jobs=8, random_state=42")

with open('notebooks/04_ablation_with_vs_without_graphs_colab.ipynb', 'w') as f:
    nbf.write(nb4, f)
print("Notebook 04 updated with local execution defaults!")

# 2. Notebook 05
with open('notebooks/05_fraud_typology_cost_function_analysis.ipynb') as f:
    nb5 = nbf.read(f, as_version=4)

nb5.cells[2].source = nb5.cells[2].source.replace(
    "if os.path.exists('/content/drive/MyDrive/ieee-data'):\n    DATA_DIR = '/content/drive/MyDrive/ieee-data'",
    "if os.path.exists('/content/drive/MyDrive/ieee-data'):\n    DATA_DIR = '/content/drive/MyDrive/ieee-data'\nelif os.path.exists('data/ieee/raw'):\n    DATA_DIR = 'data/ieee/raw'"
)

with open('notebooks/05_fraud_typology_cost_function_analysis.ipynb', 'w') as f:
    nbf.write(nb5, f)
print("Notebook 05 updated with local execution defaults!")
