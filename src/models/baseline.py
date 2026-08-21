from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


def get_xgb_params(scale_pos_weight: float = 1.0) -> dict:
    """Return the canonical baseline XGBoost parameter dict."""
    return {
        # Booster
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        # Regularisation
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,   # L1
        "reg_lambda": 1.0,  # L2
        # Task
        "objective": "binary:logistic",
        "eval_metric": "aucpr",          # PR-AUC — primary metric (ADR-0003)
        "scale_pos_weight": scale_pos_weight,
        # Performance
        "tree_method": "hist",           # Fast histogram-based; handles NaN natively
        "device": "cpu",
        "random_state": 42,
        "verbosity": 0,
    }


def get_lgbm_params(scale_pos_weight: float = 1.0) -> dict:
    """Return the canonical baseline LightGBM parameter dict."""
    return {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    verbose: bool = True,
    model_type: str = "xgb",
):
    """Train an XGBoost or LightGBM classifier."""
    if model_type == "xgb":
        if params is None:
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            scale_pos_weight = n_neg / max(n_pos, 1)
            params = get_xgb_params(scale_pos_weight=scale_pos_weight)

        model = xgb.XGBClassifier(**params)

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = 50 if verbose else False

        model.fit(X_train, y_train, **fit_kwargs)

        if verbose:
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            print(
                f"[model] trained XGBoost | "
                f"{len(X_train):,} rows × {X_train.shape[1]} features | "
                f"pos={n_pos:,} neg={n_neg:,} | "
                f"scale_pos_weight={params.get('scale_pos_weight', 1.0):.1f}"
            )
            
    elif model_type == "lgbm":
        import lightgbm as lgb
        if params is None:
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            scale_pos_weight = n_neg / max(n_pos, 1)
            params = get_lgbm_params(scale_pos_weight=scale_pos_weight)

        model = lgb.LGBMClassifier(**params)

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [lgb.log_evaluation(period=50 if verbose else 0)]

        model.fit(X_train, y_train, **fit_kwargs)

        if verbose:
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            print(
                f"[model] trained LightGBM | "
                f"{len(X_train):,} rows × {X_train.shape[1]} features | "
                f"pos={n_pos:,} neg={n_neg:,} | "
                f"scale_pos_weight={params.get('scale_pos_weight', 1.0):.1f}"
            )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save(model, path: str | Path, verbose: bool = True) -> None:
    """Save model to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(model, xgb.XGBClassifier):
        model.save_model(str(path))
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)
            
    if verbose:
        print(f"[model] saved → {path}")


def load(path: str | Path, verbose: bool = True):
    """Load model from path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[model] no model file at {path}")
        
    if "lgbm" in str(path).lower() or str(path).endswith(".pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
    else:
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        
    if verbose:
        print(f"[model] loaded ← {path}")
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return fraud probability scores (column 1 of predict_proba output)."""
    return model.predict_proba(X)[:, 1]
