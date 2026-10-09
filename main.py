import argparse
from pathlib import Path
import pandas as pd

from src.features import (
    load_train,
    load_ieee_train,
    temporal_split,
    random_split,
    drop_high_null_cols,
    drop_near_zero_variance,
    get_categorical_cols,
    label_encode_categoricals,
    identity_split,
    get_feature_cols,
    compute_graph_features,
    compute_graph_features_ieee,
    apply_v_pca,
    compute_gnn_embeddings,
    train_pure_gnn,
)
from src.models import train, save
from src.models.baseline import predict_proba
from src.evaluation.metrics import verification_suite, save_metrics


def run_baseline(
    seed: int,
    model_type: str,
    features_type: str,
    hardness: str = "low",
    dataset: str = "synthetic",
    sample: float | None = None,
    split_type: str = "temporal",
):
    print(f"--- Running Model Pipeline (Seed={seed}, Dataset={dataset}, Split={split_type}, Model={model_type}, Features={features_type}) ---")

    # 1. Load data
    if dataset == "ieee":
        df = load_ieee_train(sample_fraction=sample, verbose=True)
        # Rename TransactionDT to timestamp so temporal_split works correctly
        df = df.rename(columns={"TransactionDT": "timestamp"})
    else:
        hardness_suffix = f"_{hardness}" if hardness != "low" else ""
        data_dir = Path("data") / f"raw{hardness_suffix}"
        df = load_train(data_dir=data_dir, verbose=True)

    # 2. Split
    if split_type == "random":
        train_df, val_df = random_split(df, seed=seed, verbose=True)
    else:
        train_df, val_df = temporal_split(df, verbose=True)

    # If augmented or augmented_gnn, compute graph features
    if features_type in ["augmented", "augmented_gnn"]:
        if dataset == "ieee":
            train_df, val_df = compute_graph_features_ieee(train_df, val_df, verbose=True)
        else:
            train_df, val_df = compute_graph_features(train_df, val_df, verbose=True)

    # Compute GNN embeddings if requested
    if features_type == "augmented_gnn":
        train_df, val_df = compute_gnn_embeddings(train_df, val_df, verbose=True)


    # 3. Preprocessing (Fit on train, apply to validation)
    # We remove high null or zero-variance columns, and label-encode categoricals.
    # Note: On synthetic data, these checks run but usually drop/encode nothing.
    print("\n--- Preprocessing Features ---")
    train_df, dropped_nulls = drop_high_null_cols(train_df, verbose=True)
    val_df = val_df.drop(columns=dropped_nulls, errors="ignore")

    train_df, dropped_vars = drop_near_zero_variance(train_df, verbose=True)
    val_df = val_df.drop(columns=dropped_vars, errors="ignore")

    # Apply PCA on Vesta V-features if augmented_gnn is active
    if features_type == "augmented_gnn":
        train_df, val_df = apply_v_pca(train_df, val_df, n_components=15, verbose=True)

    cat_cols = get_categorical_cols(train_df, exclude=["TransactionID"])
    train_df, encoders = label_encode_categoricals(train_df, cat_cols, verbose=True)
    val_df, _ = label_encode_categoricals(val_df, cat_cols, encoders=encoders, verbose=True)

    # 4. Identity split
    print("\n--- Identity Routing ---")
    if dataset == "ieee":
        # Bipartite entity split: Group A has no Device details, Group B has Device details
        is_null_dev = train_df["DeviceType"].isnull()
        train_group_a = train_df[is_null_dev].copy()
        train_group_b = train_df[~is_null_dev].copy()
        
        is_null_dev_val = val_df["DeviceType"].isnull()
        val_group_a = val_df[is_null_dev_val].copy()
        val_group_b = val_df[~is_null_dev_val].copy()
        
        print(f"[split] IEEE-CIS identity routing:")
        print(f"  Group A (No Device): {len(train_group_a):,} | fraud: {train_group_a['isFraud'].mean():.3%}")
        print(f"  Group B (Has Device): {len(train_group_b):,} | fraud: {train_group_b['isFraud'].mean():.3%}")
    else:
        train_group_a, train_group_b = identity_split(train_df, verbose=True)
        val_group_a, val_group_b = identity_split(val_df, verbose=True)

    # 5. Get correct features
    if dataset == "ieee":
        # Group A uses only Transaction features (excludes all Vesta identity columns)
        ieee_id_cols = ["DeviceType", "DeviceInfo"] + [f"id_{i:02d}" for i in range(1, 39)]
        # Exclude target-encoded features to prevent temporal overfitting
        exclude_cols = ["card_fraud_rate", "device_fraud_rate"]
        features_a = [c for c in train_group_a.columns if c not in ieee_id_cols and c not in exclude_cols and c not in ["TransactionID", "timestamp", "isFraud"]]
        features_b = [c for c in train_group_b.columns if c not in exclude_cols and c not in ["TransactionID", "timestamp", "isFraud"]]
    else:
        features_a = get_feature_cols(train_group_a, include_identity=False)
        features_b = get_feature_cols(train_group_b, include_identity=True)

    # If GNN, train and evaluate directly
    if model_type == "gnn":
        if dataset != "ieee":
            raise NotImplementedError("Pure GNN is currently only supported for 'ieee' dataset.")
            
        print("\n--- Running Pure GNN Pipeline ---")
        y_score = train_pure_gnn(train_df, val_df, epochs=30, verbose=True)
        y_true = val_df["isFraud"]
        
        print("\n--- Running Verification Suite ---")
        results = verification_suite(
            train_df=train_df,
            val_df=val_df,
            y_true=y_true,
            y_score=y_score,
            group_a_idx=val_group_a.index,
            group_b_idx=val_group_b.index,
            verbose=True,
        )
        
        print("\n--- Saving Artifacts ---")
        suffix = f"{dataset}" if dataset == "ieee" else f"{dataset}_{hardness}"
        if split_type != "temporal":
            suffix += f"_{split_type}"
        save_metrics(results, f"results/metrics_{features_type}_{model_type}_{suffix}.json")
        return

    # 6. Train Group A Model (No Identity)
    print("\n--- Training Model A (No Identity) ---")
    model_a = train(
        X_train=train_group_a[features_a],
        y_train=train_group_a["isFraud"],
        X_val=val_group_a[features_a],
        y_val=val_group_a["isFraud"],
        verbose=True,
        model_type=model_type,
    )

    # 7. Train Group B Model (With Identity)
    print("\n--- Training Model B (Has Identity) ---")
    model_b = train(
        X_train=train_group_b[features_b],
        y_train=train_group_b["isFraud"],
        X_val=val_group_b[features_b],
        y_val=val_group_b["isFraud"],
        verbose=True,
        model_type=model_type,
    )


    # 8. Prediction and Evaluation
    print("\n--- Generating Validation Predictions ---")
    y_score_a = predict_proba(model_a, val_group_a[features_a])
    y_score_b = predict_proba(model_b, val_group_b[features_b])

    y_score = pd.Series(index=val_df.index, dtype=float)
    y_score.loc[val_group_a.index] = y_score_a
    y_score.loc[val_group_b.index] = y_score_b
    y_true = val_df["isFraud"]

    print("\n--- Running Verification Suite ---")
    results = verification_suite(
        train_df=train_df,
        val_df=val_df,
        y_true=y_true,
        y_score=y_score,
        group_a_idx=val_group_a.index,
        group_b_idx=val_group_b.index,
        verbose=True,
    )

    # 9. Save models and metrics
    print("\n--- Saving Artifacts ---")
    suffix = f"{dataset}" if dataset == "ieee" else f"{dataset}_{hardness}"
    if split_type != "temporal":
        suffix += f"_{split_type}"
    ext = "pkl" if model_type == "lgbm" else "json"
    save(model_a, f"models/model_a_{features_type}_{model_type}_{suffix}.{ext}", verbose=True)
    save(model_b, f"models/model_b_{features_type}_{model_type}_{suffix}.{ext}", verbose=True)
    save_metrics(results, f"results/metrics_{features_type}_{model_type}_{suffix}.json")


def main():
    parser = argparse.ArgumentParser(description="TG-graph Fraud Detection Experiment Runner")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--model",
        type=str,
        default="xgb",
        choices=["xgb", "lgbm", "gnn"],
        help="Classifier algorithm to train",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="tabular",
        choices=["tabular", "augmented", "augmented_gnn"],
        help="Feature set to use",
    )
    parser.add_argument(
        "--hardness",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="Difficulty level of synthetic fraud data to load/evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "ieee"],
        help="Dataset to load and train on",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Fraction of data to sample (0.0 to 1.0) for rapid testing",
    )
    parser.add_argument(
        "--split-type",
        dest="split_type",
        type=str,
        default="temporal",
        choices=["temporal", "random"],
        help="Data split strategy: 'temporal' (time-based) or 'random' (stratified random)",
    )

    args = parser.parse_args()

    run_baseline(
        seed=args.seed,
        model_type=args.model,
        features_type=args.features,
        hardness=args.hardness,
        dataset=args.dataset,
        sample=args.sample,
        split_type=args.split_type,
    )


if __name__ == "__main__":
    main()
