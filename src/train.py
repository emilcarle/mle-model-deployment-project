# Functions to train, register and promote the best rf classifier in MLflow

from pathlib import Path
import mlflow
import pandas as pd
from mlflow.sklearn import log_model
from mlflow import MlflowClient
from dotenv import load_dotenv
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import optuna
import numpy as np
from src.config import REGISTERED_MODEL_NAME, MODEL_ALIAS, MLFLOW_EXPERIMENT_NAME

def objective(trial, x_train, y_train, fixed_param_dict):
    param_dict = {
        "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.00001, 0.001, log=True),
        "max_depth":        trial.suggest_int("max_depth", 10, 40),
        "max_features":     trial.suggest_int("max_features", 2, 5),
        "max_samples":      trial.suggest_float("max_samples", 0.05, 0.5, log=True),
    }
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(param_dict)        
        model = RandomForestRegressor(
            **fixed_param_dict,
            **param_dict, 
        )
        rmse = cross_val_score(
            model,
            x_train,
            y_train,
            cv=3,
            scoring="neg_root_mean_squared_error"
        ).mean()
        mlflow.log_metrics({
            "neg_rmse": rmse
        })
        return rmse


def train_model(input_data: Path, target_column: str):
    # initiate the experiement with a name
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    # Load data
    df = pd.read_parquet(input_data)
    y_df = df[target_column]
    x_df = df.drop(columns=[target_column])
    # Create train/test split
    x_train, x_test, y_train, y_test = train_test_split(
    x_df, y_df, test_size=0.3, random_state=42
    )
    # tune and predict
    x_tune = x_train.sample(frac=0.05, random_state=42)
    y_tune = y_train.loc[x_tune.index]
    fixed_param_dict = {
        "n_estimators": 100, 
        "random_state": 42, 
        "n_jobs": -1,
    }
    with mlflow.start_run(run_name="rf-optuna"):                      
        mlflow.set_tags({                                                     
            "model_type": "RandomForestRegressor",
            "tuning": "optuna"
        })
        mlflow.log_params({
            "n_train_rows": len(x_train),
            "n_tune_rows": len(x_tune),
            "n_trials": 20,
            "features": ",".join(x_train.columns),
            **fixed_param_dict,
        })
        optuna_study = optuna.create_study(direction="maximize")
        optuna_study.optimize(lambda trial: objective(trial, x_tune, y_tune, fixed_param_dict), n_trials=20)
        mlflow.log_params(optuna_study.best_params)
        mlflow.log_metric("cv_rmse", -optuna_study.best_value) 
        model = Pipeline(
            [
                ("model", RandomForestRegressor(
                        **fixed_param_dict,
                        **optuna_study.best_params
                    ))
            ]
        )
        model.fit(x_train, y_train)
        # preform a prediction and assess accuracy
        model_y_pred = model.predict(x_test)
        baseline__y_pred = np.full(len(y_test), y_train.mean())
        mlflow.log_metrics({
            "neg_rmse": -root_mean_squared_error(y_test, model_y_pred),
            "mae": mean_absolute_error(y_test, model_y_pred),
            "rmse_baseline": root_mean_squared_error(y_test, baseline__y_pred),
        })
        model_info = log_model(                              
            sk_model=model,
            name="model",
            input_example=x_train.head(5),
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print(model_info)
        return model_info

def set_production_model(client=None, model_name=REGISTERED_MODEL_NAME, model_alias=MODEL_ALIAS):
    client = client or MlflowClient()
    versions = [v for v in client.search_model_versions(f"name='{model_name}'") if v.run_id is not None]
    def neg_rmse(version):
        return client.get_run(version.run_id).data.metrics.get("neg_rmse", float("-inf"))
    best = max(versions, key=neg_rmse)
    best_neg_rmse = neg_rmse(best)
    client.set_registered_model_alias(name=model_name, alias=model_alias, version=best.version)
    print(f"Version {best.version} (neg_rmse {best_neg_rmse:.3f}) is now @{model_alias}")