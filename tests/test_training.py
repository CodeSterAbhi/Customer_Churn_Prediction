import numpy as np
from churn_pipeline.train import train_baseline_models

def test_train_baseline_models():
    X_train = np.random.rand(50, 10)
    y_train = np.random.choice([0, 1], size=50)
    X_val = np.random.rand(20, 10)
    y_val = np.random.choice([0, 1], size=20)
    
    results = train_baseline_models(X_train, y_train, X_val, y_val)
    assert "logistic_regression" in results
    assert "random_forest" in results
    assert "xgboost" in results
    assert "roc_auc" in results["xgboost"]