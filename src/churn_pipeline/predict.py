import pandas as pd
import joblib
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ChurnPredictor:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.pipeline = None
        self.metadata = None
        self.load_artifacts()

    def load_artifacts(self):
        model_path = self.artifacts_dir / "best_churn_pipeline.joblib"
        meta_path = self.artifacts_dir / "model_metadata.json"
        
        if model_path.exists():
            self.pipeline = joblib.load(model_path)
            logger.info("Loaded model pipeline successfully.")
        else:
            logger.warning(f"Model artifact not found at {model_path}")
            
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

    def predict(self, input_data: pd.DataFrame) -> dict:
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded. Train or place artifacts first.")
            
        threshold = self.metadata.get("threshold", 0.5) if self.metadata else 0.5
        
        probs = self.pipeline.predict_proba(input_data)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        return {
            "prediction": int(preds[0]),
            "probability": float(probs[0]),
            "threshold": float(threshold)
        }