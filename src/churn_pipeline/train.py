import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import logging

logger = logging.getLogger(__name__)

def train_baseline_models(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, random_state: int = 42) -> dict:
    models = {
        "logistic_regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "xgboost": XGBClassifier(random_state=random_state, eval_metric='logloss')
    }
    
    results = {}
    for name, model in models.items():
        logger.info(f"Training baseline model: {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else preds
        
        score_roc = roc_auc_score(y_val, probs)
        score_f1 = f1_score(y_val, preds)
        score_acc = accuracy_score(y_val, preds)
        
        results[name] = {
            "model": model,
            "roc_auc": score_roc,
            "f1": score_f1,
            "accuracy": score_acc
        }
        logger.info(f"{name} - ROC-AUC: {score_roc:.4f}, F1: {score_f1:.4f}, Accuracy: {score_acc:.4f}")
        
    return results