"""Tests for classification model module."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from clearsky_lidar.classification.model import (
    create_model,
    load_model,
    freeze_backbone_fn,
    unfreeze_head,
    save_model,
)


class TestCreateModel:
    """Tests for create_model function."""

    @patch("clearsky_lidar.classification.model.YOLO")
    def test_create_model_default_params(self, mock_yolo):
        """Test create_model with default parameters."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        model = create_model()

        mock_yolo.assert_called_once_with("yolov8n-cls.pt")
        assert model == mock_model

    @patch("clearsky_lidar.classification.model.YOLO")
    def test_create_model_custom_num_classes(self, mock_yolo):
        """Test create_model with custom num_classes."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        model = create_model(num_classes=10)

        mock_yolo.assert_called_once_with("yolov8n-cls.pt")
        assert model == mock_model

    @patch("clearsky_lidar.classification.model.YOLO")
    def test_create_model_pretrained_false(self, mock_yolo):
        """Test create_model with pretrained=False."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        model = create_model(pretrained=False)

        mock_yolo.assert_called_once_with("yolov8n-cls.yaml")
        assert model == mock_model

    @patch("clearsky_lidar.classification.model.YOLO")
    @patch("clearsky_lidar.classification.model.freeze_backbone_fn")
    def test_create_model_freeze_backbone_true(self, mock_freeze, mock_yolo):
        """Test create_model freezes backbone when freeze_backbone=True."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        model = create_model(freeze_backbone=True)

        mock_freeze.assert_called_once_with(mock_model, freeze=True)
        assert model == mock_model

    @patch("clearsky_lidar.classification.model.YOLO")
    @patch("clearsky_lidar.classification.model.freeze_backbone_fn")
    def test_create_model_freeze_backbone_false(self, mock_freeze, mock_yolo):
        """Test create_model doesn't call freeze_backbone_fn when freeze_backbone=False."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        model = create_model(freeze_backbone=False)

        mock_freeze.assert_not_called()
        assert model == mock_model


class TestLoadModel:
    """Tests for load_model function."""

    @patch("clearsky_lidar.classification.model.YOLO")
    def test_load_model_pt_file(self, mock_yolo):
        """Test loading .pt file."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        model = load_model("models/best.pt")

        mock_yolo.assert_called_once_with("models/best.pt")
        assert model == mock_model

    @patch("clearsky_lidar.classification.model.YOLO")
    def test_load_model_engine_file(self, mock_yolo):
        """Test loading .engine file."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        model = load_model("models/best.engine")

        mock_yolo.assert_called_once_with("models/best.engine")
        assert model == mock_model

    @patch("clearsky_lidar.classification.model.YOLO")
    def test_load_model_pathlib_path(self, mock_yolo):
        """Test loading with Path object."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        model = load_model(Path("models/best.pt"))

        mock_yolo.assert_called_once_with("models/best.pt")
        assert model == mock_model

    def test_load_model_invalid_extension(self):
        """Test ValueError for unsupported extension."""
        with pytest.raises(ValueError, match="Unsupported model format"):
            load_model("model.onnx")


class TestFreezeBackbone:
    """Tests for freeze_backbone_fn function."""

    def test_freeze_backbone_true(self):
        """Test freezing backbone layers."""
        mock_model = MagicMock()
        # Create mock parameters for layers 0-7
        mock_params = [MagicMock() for _ in range(8)]
        mock_layers = [MagicMock() for _ in range(8)]
        for layer, param in zip(mock_layers, mock_params):
            layer.parameters.return_value = [param]
        mock_model.model.model = mock_layers + [MagicMock() for _ in range(4)]  # 12 total layers

        freeze_backbone_fn(mock_model, freeze=True)

        for param in mock_params:
            assert param.requires_grad is False

    def test_freeze_backbone_false(self):
        """Test unfreezing backbone layers."""
        mock_model = MagicMock()
        mock_params = [MagicMock() for _ in range(8)]
        mock_layers = [MagicMock() for _ in range(8)]
        for layer, param in zip(mock_layers, mock_params):
            layer.parameters.return_value = [param]
        mock_model.model.model = mock_layers + [MagicMock() for _ in range(4)]

        freeze_backbone_fn(mock_model, freeze=False)

        for param in mock_params:
            assert param.requires_grad is True


class TestUnfreezeHead:
    """Tests for unfreeze_head function."""

    def test_unfreeze_head(self):
        """Test unfreezing head layers (8+)."""
        mock_model = MagicMock()
        mock_params = [MagicMock() for _ in range(4)]  # layers 8, 9, 10, 11
        mock_head_layers = [MagicMock() for _ in range(4)]
        for layer, param in zip(mock_head_layers, mock_params):
            layer.parameters.return_value = [param]
        mock_model.model.model = [MagicMock() for _ in range(8)] + mock_head_layers

        unfreeze_head(mock_model)

        for param in mock_params:
            assert param.requires_grad is True


class TestSaveModel:
    """Tests for save_model function."""

    @patch("clearsky_lidar.classification.model.Path.mkdir")
    def test_save_model(self, mock_mkdir):
        """Test saving model."""
        mock_model = MagicMock()
        mock_model.save = MagicMock()

        path = save_model(mock_model, "models/best.pt")

        mock_model.save.assert_called_once_with("models/best.pt")
        assert path == Path("models/best.pt")
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)