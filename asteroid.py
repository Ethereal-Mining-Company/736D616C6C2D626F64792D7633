# asteroid.py
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Union, List
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Custom type alias for validation hints
AsteroidInput = Union[Dict[str, Union[float, int, str]], pd.DataFrame]

def train(dataset_path: str = None, config_path: str = "config.yaml") -> dict:
    """
    Evaluates asteroid feature subsets using 5-fold cross-validation and 
    trains final robust production model pipelines based on a YAML configuration.

    Args:
        dataset_path (str, optional): Custom file path to a dataset. If None, the default 
            path inside `config.yaml` will be resolved. Defaults to None.
        config_path (str, optional): The target YAML parameter file configuration. 
            Defaults to "config.yaml".

    Returns:
        dict: Lightweight execution run summary and evaluation metric metrics.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at: '{config_path}'")
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    final_data_path = dataset_path if dataset_path is not None else cfg["paths"]["dataset_cleaned"]
    
    if not os.path.exists(final_data_path):
        raise FileNotFoundError(f"Cleaned asteroid dataset not found at: '{final_data_path}'")

    model_dir = cfg["paths"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    
    x_full = cfg["features"]["x_full"]
    x_restricted = cfg["features"]["x_restricted"]
    target_col = cfg["features"]["target_col"]
    
    df_model = pd.read_csv(final_data_path)
    X = df_model[x_full]
    y = df_model[target_col].values.ravel()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"full": {"r2": [], "mae_km": []}, "restricted": {"r2": [], "mae_km": []}}

    params = cfg["model_params"]
    
    print("Running Cross-Validation Evaluation...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf_f = RandomForestRegressor(**params)
        rf_r = RandomForestRegressor(**params)
        
        rf_f.fit(X_train[x_full], y_train)
        preds_f = rf_f.predict(X_test[x_full])
        
        rf_r.fit(X_train[x_restricted], y_train)
        preds_r = rf_r.predict(X_test[x_restricted])
        
        metrics["full"]["r2"].append(r2_score(y_test, preds_f))
        metrics["restricted"]["r2"].append(r2_score(y_test, preds_r))
        
        y_test_km = 10 ** y_test
        metrics["full"]["mae_km"].append(mean_absolute_error(y_test_km, 10 ** preds_f))
        metrics["restricted"]["mae_km"].append(mean_absolute_error(y_test_km, 10 ** preds_r))

    mean_full_r2 = float(np.mean(metrics["full"]["r2"]))
    mean_restricted_r2 = float(np.mean(metrics["restricted"]["r2"]))
    mean_full_mae = float(np.mean(metrics["full"]["mae_km"]))
    mean_restricted_mae = float(np.mean(metrics["restricted"]["mae_km"]))

    full_composer = ColumnTransformer(transformers=[("keep", "passthrough", x_full)], remainder="drop")
    restricted_composer = ColumnTransformer(transformers=[("keep", "passthrough", x_restricted)], remainder="drop")

    production_pipeline_full = Pipeline([
        ("selector", full_composer),
        ("regressor", RandomForestRegressor(**params))
    ])
    production_pipeline_restricted = Pipeline([
        ("selector", restricted_composer),
        ("regressor", RandomForestRegressor(**params))
    ])

    production_pipeline_full.fit(X, y)
    production_pipeline_restricted.fit(X, y)

    def _generate_metadata_payload(fitted_pipeline, feature_names, description, cv_metrics):
        underlying_regressor = fitted_pipeline.named_steps["regressor"]
        return {
            "model_metadata": {
                "description": description,
                "target_variable": target_col,
                "prediction_space_transformation": "10 ** prediction",
                "hyperparameters": underlying_regressor.get_params(),
            },
            "features": {
                "input_features": feature_names,
                "feature_importance": dict(zip(feature_names, underlying_regressor.feature_importances_.tolist()))
            },
            "validation_metrics": {
                "mean_cv_r2": float(np.mean(cv_metrics["r2"])),
                "std_cv_r2": float(np.std(cv_metrics["r2"])),
                "mean_cv_mae_km": float(np.mean(cv_metrics["mae_km"])),
            }
        }

    full_payload = _generate_metadata_payload(production_pipeline_full, x_full, "Full model with H.", metrics["full"])
    restricted_payload = _generate_metadata_payload(production_pipeline_restricted, x_restricted, "Restricted model.", metrics["restricted"])

    joblib.dump({"model": production_pipeline_full, "metadata": full_payload}, cfg["paths"]["rf_full_model"])
    joblib.dump({"model": production_pipeline_restricted, "metadata": restricted_payload}, cfg["paths"]["rf_restricted_model"])
    print("🎉 Success! Production models and metadata bundles saved safely.")
    
    return {
        "status": "success",
        "model_paths": {
            "full_model": cfg["paths"]["rf_full_model"], 
            "restricted_model": cfg["paths"]["rf_restricted_model"]
        },
        "metrics": {
            "full_mean_cv_r2": mean_full_r2, "restricted_mean_cv_r2": mean_restricted_r2,
            "full_mean_cv_mae_km": mean_full_mae, "restricted_mean_cv_mae_km": mean_restricted_mae
        }
    }

def predict(
    input_data: AsteroidInput, 
    model_type: str = "full", 
    config_path: str = "config.yaml"
) -> Union[float, List[float], dict]:
    """
    Predicts the physical diameter of an asteroid in kilometers using the saved production pipeline.

    Args:
        input_data (dict | DataFrame): Raw asteroid feature measurements.
        model_type (str): Execution configuration route. Accepting 'full', 'restricted', or 'both'.
            Defaults to 'full'.
        config_path (str): Target filesystem path mapping to configuration values.
            Defaults to "config.yaml".

    Returns:
        float | list | dict: Evaluated diameter values scaled to kilometers. Returns a dictionary 
            if model_type="both" is requested.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at: '{config_path}'")
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    if model_type.lower() == "both":
        return {
            "full_model_prediction": predict(input_data, model_type="full", config_path=config_path),
            "restricted_model_prediction": predict(input_data, model_type="restricted", config_path=config_path)
        }

    if model_type.lower() == "full":
        model_key = "rf_full_model"
    elif model_type.lower() == "restricted":
        model_key = "rf_restricted_model"
    else:
        raise ValueError("model_type must be either 'full', 'restricted', or 'both'.")
        
    model_path = cfg["paths"][model_key]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing at '{model_path}'. Please run asteroid.train() first.")
        
    artifacts = joblib.load(model_path)
    pipeline = artifacts["model"]
    
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise TypeError("input_data must be a dictionary or a Pandas DataFrame.")
        
    log_prediction = pipeline.predict(df_input)
    diameter_km = 10 ** log_prediction
    
    if isinstance(input_data, dict):
        return float(diameter_km) if isinstance(diameter_km, np.ndarray) else float(diameter_km)
    return diameter_km.tolist()