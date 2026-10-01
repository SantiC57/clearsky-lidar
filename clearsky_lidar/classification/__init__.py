"""Classification package for ClearSky waste classification."""

from .model import create_model, load_model, freeze_backbone_fn, unfreeze_head, save_model
from .mlflow_utils import init_mlflow, get_mlflow_client, get_experiment_id, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from .train import train_model, create_data_yaml
from .export_engine import export_engine, verify_engine
from .inference import WasteClassifier

__all__ = [
    "create_model",
    "load_model",
    "freeze_backbone_fn",
    "unfreeze_head",
    "save_model",
    "init_mlflow",
    "get_mlflow_client",
    "get_experiment_id",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "train_model",
    "create_data_yaml",
    "export_engine",
    "verify_engine",
    "WasteClassifier",
]