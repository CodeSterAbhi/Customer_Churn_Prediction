import mlflow
import logging

logger = logging.getLogger(__name__)

def setup_mlflow(experiment_name: str, tracking_uri: str = "file:./mlruns"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")