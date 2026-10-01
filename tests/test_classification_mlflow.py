"""Tests for MLflow utilities module."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from clearsky_lidar.classification.mlflow_utils import (
    init_mlflow,
    get_mlflow_client,
    get_experiment_id,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)


class TestMLflowUtils:
    """Tests for MLflow utilities."""

    def test_constants(self):
        """Test MLflow constants have correct values."""
        assert MLFLOW_TRACKING_URI == "sqlite:///mlflow.db"
        assert MLFLOW_EXPERIMENT_NAME == "yolov8-cls-waste"

    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.set_tracking_uri")
    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.get_experiment_by_name")
    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.create_experiment")
    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.set_experiment")
    def test_init_mlflow_creates_experiment(self, mock_set_experiment, mock_create, mock_get_exp, mock_set_uri):
        """Test init_mlflow creates experiment when it doesn't exist."""
        mock_get_exp.return_value = None

        init_mlflow()

        mock_set_uri.assert_called_once_with("sqlite:///mlflow.db")
        mock_get_exp.assert_called_once_with("yolov8-cls-waste")
        mock_create.assert_called_once_with("yolov8-cls-waste")
        mock_set_experiment.assert_called_once_with("yolov8-cls-waste")

    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.set_tracking_uri")
    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.get_experiment_by_name")
    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.create_experiment")
    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.set_experiment")
    def test_init_mlflow_uses_existing_experiment(self, mock_set_experiment, mock_create, mock_get_exp, mock_set_uri):
        """Test init_mlflow uses existing experiment."""
        mock_exp = MagicMock()
        mock_get_exp.return_value = mock_exp

        init_mlflow()

        mock_set_uri.assert_called_once_with("sqlite:///mlflow.db")
        mock_get_exp.assert_called_once_with("yolov8-cls-waste")
        mock_create.assert_not_called()
        mock_set_experiment.assert_called_once_with("yolov8-cls-waste")

    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.set_tracking_uri")
    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.get_experiment_by_name")
    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.create_experiment")
    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.set_experiment")
    def test_init_mlflow_custom_params(self, mock_set_experiment, mock_create, mock_get_exp, mock_set_uri):
        """Test init_mlflow with custom tracking URI and experiment name."""
        mock_get_exp.return_value = None

        init_mlflow(tracking_uri="sqlite:///custom.db", experiment_name="custom-exp")

        mock_set_uri.assert_called_once_with("sqlite:///custom.db")
        mock_get_exp.assert_called_once_with("custom-exp")
        mock_create.assert_called_once_with("custom-exp")
        mock_set_experiment.assert_called_once_with("custom-exp")

    @patch("clearsky_lidar.classification.mlflow_utils.mlflow.MlflowClient")
    def test_get_mlflow_client(self, mock_client_class):
        """Test get_mlflow_client returns client instance."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client = get_mlflow_client()

        mock_client_class.assert_called_once_with(tracking_uri="sqlite:///mlflow.db")
        assert client == mock_client

    @patch("clearsky_lidar.classification.mlflow_utils.get_mlflow_client")
    def test_get_experiment_id_found(self, mock_get_client):
        """Test get_experiment_id returns ID when experiment exists."""
        mock_client = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "123"
        mock_client.get_experiment_by_name.return_value = mock_exp
        mock_get_client.return_value = mock_client

        exp_id = get_experiment_id("test-exp")

        assert exp_id == "123"
        mock_client.get_experiment_by_name.assert_called_once_with("test-exp")

    @patch("clearsky_lidar.classification.mlflow_utils.get_mlflow_client")
    def test_get_experiment_id_not_found(self, mock_get_client):
        """Test get_experiment_id raises ValueError when experiment doesn't exist."""
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        mock_get_client.return_value = mock_client

        with pytest.raises(ValueError, match="Experiment 'missing-exp' not found"):
            get_experiment_id("missing-exp")

    def test_ensure_mlflow_db_dir(self, tmp_path):
        """Test ensure_mlflow_db_dir creates parent directory."""
        from clearsky_lidar.classification.mlflow_utils import ensure_mlflow_db_dir

        # Change to temp directory
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            ensure_mlflow_db_dir()
            assert (tmp_path / "mlflow.db").parent.exists()
        finally:
            os.chdir(old_cwd)