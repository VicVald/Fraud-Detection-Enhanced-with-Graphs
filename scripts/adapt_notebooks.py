import json
from pathlib import Path

notebooks_dir = Path(__file__).resolve().parent.parent / "notebooks"

replacements = {
    "01_eda_and_feature_selection.ipynb": [
        ("IEEE-CIS Fraud Detection (Kaggle, 2019)", "gen-fraud-graph (Santander, 2026)"),
        ("TransactionAmt", "amount"),
        ("TransactionDT", "timestamp"),
        ("TransactionID", "tx_id"),
        ("12,192,853", "80th percentile of timestamp"),
        ("TEMPORAL_SPLIT_BOUNDARY", "pd.to_datetime('2024-01-24 23:44:43')"),
        ("relative seconds", "timestamp"),
    ],
    "02_model_and_verification.ipynb": [
        ("IEEE-CIS", "gen-fraud-graph"),
        ("TransactionAmt", "amount"),
        ("TransactionDT", "timestamp"),
        ("TransactionID", "tx_id"),
        ("id_01", "src_id"),
        ("12_192_853", "pd.to_datetime('2024-01-24 23:44:43')"),
    ]
}

for nb_name, reps in replacements.items():
    nb_path = notebooks_dir / nb_name
    print(f"Adapting {nb_name}...")
    
    with open(nb_path) as f:
        nb = json.load(f)
        
    for cell in nb.get("cells", []):
        if "source" in cell:
            new_source = []
            for line in cell["source"]:
                for old, new in reps:
                    line = line.replace(old, new)
                new_source.append(line)
            cell["source"] = new_source
            
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)
        
print("Notebooks adapted successfully.")
