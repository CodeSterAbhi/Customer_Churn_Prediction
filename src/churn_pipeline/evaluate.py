import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)
    
    metrics = {
        "roc_auc": roc_auc,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "threshold": threshold
    }
    logger.info(f"Test Evaluation -> ROC-AUC: {roc_auc:.4f}, F1-Score: {f1:.4f}")
    return metrics

def find_optimal_threshold(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    logger.info(f"Optimal classification threshold found: {best_threshold:.4f}")
    return float(best_threshold)