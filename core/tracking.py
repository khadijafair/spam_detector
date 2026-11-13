"""
tracking.py — MLflow setup shared by the training, evaluation, retraining and
drift-monitoring scripts.

By default runs are stored locally in mlflow.db (SQLite, needed for the Model
Registry) and artifacts in mlruns/. Browse them with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
Set MLFLOW_TRACKING_URI to log to a remote MLflow server instead.
"""
import os

import mlflow

import core.config as config

EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "sentinel-spam")
REGISTERED_MODEL = "sentinel-spam-svm"


def setup() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    uri = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{config.PROJECT_ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(config.PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def log_scores(scores: dict, prefix: str = "") -> None:
    """Logs the numeric entries of a research.evaluate.scores() dict."""
    for k, v in scores.items():
        if k == "confusion_matrix":
            for ck, cv in v.items():
                mlflow.log_metric(f"{prefix}{ck}", cv)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            mlflow.log_metric(f"{prefix}{k}", float(v))
