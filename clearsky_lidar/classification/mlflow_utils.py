"""MLflow utilities for waste classification training."""

import mlflow
from pathlib import Path


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "yolov8-cls-waste"


def init_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> None:
    """Initialize MLflow tracking with sqlite backend and create/get experiment.

    Args:
        tracking_uri: MLflow tracking URI (default: sqlite:///mlflow.db)
        experiment_name: Name of the experiment (default: yolov8-cls-waste)
    """
    mlflow.set_tracking_uri(tracking_uri)

    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        print(f"Created MLflow experiment: {experiment_name}")
    else:
        print(f"Using existing MLflow experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)


def get_mlflow_client() -> mlflow.MlflowClient:
    """Get MLflow client for the current tracking URI.

    Returns:
        MLflow client instance
    """
    return mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_experiment_id(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> str:
    """Get experiment ID by name.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Experiment ID string

    Raises:
        ValueError: If experiment doesn't exist
    """
    client = get_mlflow_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    return experiment.experiment_id


def ensure_mlflow_db_dir() -> None:
    """Ensure the mlflow.db parent directory exists."""
    db_path = Path("mlflow.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)