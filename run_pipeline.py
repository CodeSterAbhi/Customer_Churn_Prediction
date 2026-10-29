import os
import logging
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from churn_pipeline.config import load_config
from churn_pipeline.data_validation import validate_raw_data
from churn_pipeline.data_preprocessing import get_preprocessing_pipeline
from churn_pipeline.train import train_baseline_models
from churn_pipeline.tune import tune_xgboost
from churn_pipeline.evaluate import evaluate_model, find_optimal_threshold
from churn_pipeline.tracking import setup_mlflow
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    config = load_config()
    logger.info("Starting MLOps training pipeline...")
    
    raw_path = config['data']['raw_path']
    if not os.path.exists(raw_path):
        logger.error(f"Dataset not found at {raw_path}. Please place Telco-Customer-Churn.csv in data/raw/")
        return
        
    df = pd.read_csv(raw_path)
    validate_raw_data(df)
    
    target_col = config['data']['target_col']
    df[target_col] = df[target_col].replace({'Yes': 1, 'No': 0})
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])
    
    test_size = config['data']['test_size']
    val_size = config['data']['val_size']
    random_state = config['random_state']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=random_state, stratify=y_train_val)
    
    preprocessor = get_preprocessing_pipeline()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    
    setup_mlflow(config['mlflow']['experiment_name'], config['mlflow']['tracking_uri'])
    
    with mlflow.start_run(run_name="full_pipeline_run"):
        baselines = train_baseline_models(X_train_trans, y_train, X_val_trans, y_val, random_state)
        
        logger.info("Tuning XGBoost with Optuna...")
        best_model = tune_xgboost(X_train_trans, y_train, X_val_trans, y_val, n_trials=config['optuna']['n_trials'], random_state=random_state)
        
        threshold = find_optimal_threshold(best_model, X_val_trans, y_val)
        
        test_metrics = evaluate_model(best_model, X_test_trans, y_test, threshold=threshold)
        
        mlflow.log_metric("test_roc_auc", test_metrics['roc_auc'])
        mlflow.log_metric("test_f1", test_metrics['f1_score'])
        mlflow.log_param("optimal_threshold", threshold)
        
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', best_model)
        ])
        
        os.makedirs(config['artifacts']['model_dir'], exist_ok=True)
        model_path = os.path.join(config['artifacts']['model_dir'], config['artifacts']['model_filename'])
        joblib.dump(full_pipeline, model_path)
        
        metadata = {
            "threshold": threshold,
            "test_metrics": test_metrics
        }
        meta_path = os.path.join(config['artifacts']['model_dir'], config['artifacts']['metadata_filename'])
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Pipeline executed successfully. Artifacts saved to {config['artifacts']['model_dir']}")

if __name__ == "__main__":
    main()