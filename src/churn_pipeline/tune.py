import optuna
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 20, random_state: int = 42) -> XGBClassifier:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': random_state,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds_proba)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best Optuna ROC-AUC score: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters: {study.best_params}")
    
    best_params = study.best_params
    best_params['random_state'] = random_state
    best_params['eval_metric'] = 'logloss'
    
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)
    return best_model