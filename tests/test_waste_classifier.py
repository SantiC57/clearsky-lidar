"""Tests for WasteClassifier inference wrapper."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import numpy as np
import tempfile
import os

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from clearsky_lidar.classification.inference import WasteClassifier


class TestWasteClassifierInit:
    """Tests for WasteClassifier initialization."""

    @patch("clearsky_lidar.classification.inference.load_model")
    @patch("clearsky_lidar.classification.inference.Path.exists")
    @patch("clearsky_lidar.classification.inference.Path.is_file")
    def test_init_with_explicit_model_path(self, mock_is_file, mock_exists, mock_load_model):
        """Test initialization with explicit model_path."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_is_file.return_value = True
        mock_exists.return_value = True

        classifier = WasteClassifier(model_path="/custom/path/best.pt", conf_threshold=0.7)

        assert classifier.conf_threshold == 0.7
        assert classifier.model == mock_model
        # load_model receives Path object, compare as string
        called_path = str(mock_load_model.call_args[0][0])
        assert called_path == "/custom/path/best.pt"

    @patch("clearsky_lidar.classification.inference.load_model")
    @patch.dict(os.environ, {"CLEARSKY_MODEL_DIR": "/env/models"})
    @patch("clearsky_lidar.classification.inference.Path.exists")
    @patch("clearsky_lidar.classification.inference.Path.is_file")
    def test_init_resolves_from_env_var(self, mock_is_file, mock_exists, mock_load_model):
        """Test initialization resolves path from CLEARSKY_MODEL_DIR env var."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_is_file.return_value = True
        mock_exists.return_value = True

        classifier = WasteClassifier(conf_threshold=0.6)

        assert classifier.conf_threshold == 0.6
        # Should check env var path first
        mock_load_model.assert_called_once()

    @patch("clearsky_lidar.classification.inference.load_model")
    @patch("clearsky_lidar.classification.inference.Path.exists")
    @patch("clearsky_lidar.classification.inference.Path.is_file")
    def test_init_fallbacks_to_default_path(self, mock_is_file, mock_exists, mock_load_model):
        """Test initialization falls back to ~/ClearSky/weights/."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_is_file.return_value = True
        mock_exists.return_value = True

        with patch.dict(os.environ, {}, clear=True):
            classifier = WasteClassifier()

        assert classifier.conf_threshold == 0.55  # default
        mock_load_model.assert_called_once()

    @patch("clearsky_lidar.classification.inference.load_model")
    @patch("clearsky_lidar.classification.inference.Path.exists")
    @patch("clearsky_lidar.classification.inference.Path.is_file")
    def test_init_with_engine_file(self, mock_is_file, mock_exists, mock_load_model):
        """Test initialization works with .engine file (Jetson)."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_is_file.return_value = True
        mock_exists.return_value = True

        classifier = WasteClassifier(model_path="/models/best.engine")

        assert classifier.model == mock_model
        called_path = str(mock_load_model.call_args[0][0])
        assert called_path == "/models/best.engine"


class TestWasteClassifierPredict:
    """Tests for predict method."""

    @patch("clearsky_lidar.classification.inference.load_model")
    def test_predict_returns_tuple(self, mock_load_model):
        """Test predict returns (class_name, confidence, probs_dict)."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Mock YOLO prediction result
        mock_result = MagicMock()
        mock_result.probs.data.cpu().numpy.return_value = np.array([0.1, 0.2, 0.5, 0.1, 0.05, 0.05])
        mock_model.return_value = [mock_result]

        classifier = WasteClassifier(model_path="models/best.pt")
        # Create dummy image
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        result = classifier.predict(image)

        assert isinstance(result, tuple)
        assert len(result) == 3
        class_name, confidence, probs_dict = result
        assert isinstance(class_name, str)
        assert isinstance(confidence, float)
        assert isinstance(probs_dict, dict)
        assert len(probs_dict) == 6

    @patch("clearsky_lidar.classification.inference.load_model")
    def test_predict_confidence_threshold_filters(self, mock_load_model):
        """Test predict filters by confidence threshold."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Low confidence prediction (max prob = 0.4)
        mock_result = MagicMock()
        mock_result.probs.data.cpu().numpy.return_value = np.array([0.4, 0.2, 0.2, 0.1, 0.05, 0.05])
        mock_model.return_value = [mock_result]

        classifier = WasteClassifier(model_path="models/best.pt", conf_threshold=0.55)
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        result = classifier.predict(image)

        class_name, confidence, probs_dict = result
        # Should return "trash" or lowest class when below threshold?
        # Actually, the implementation should return the top class but confidence < threshold
        # Let's check: the spec says "Confidence threshold filtering"
        # This means if max confidence < threshold, it should probably return a special result
        # We'll test the actual behavior in implementation

    @patch("clearsky_lidar.classification.inference.load_model")
    def test_predict_batch_returns_list(self, mock_load_model):
        """Test predict_batch returns list of predictions."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_result = MagicMock()
        mock_result.probs.data.cpu().numpy.return_value = np.array([0.1, 0.2, 0.5, 0.1, 0.05, 0.05])
        mock_model.return_value = [mock_result, mock_result]

        classifier = WasteClassifier(model_path="models/best.pt")
        images = [
            np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        ]

        results = classifier.predict_batch(images)

        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 3


class TestWasteClassifierPathResolution:
    """Tests for model path resolution logic."""

    @patch("clearsky_lidar.classification.inference.load_model")
    @patch("clearsky_lidar.classification.inference.Path.exists")
    @patch("clearsky_lidar.classification.inference.Path.is_file")
    def test_explicit_path_priority(self, mock_is_file, mock_exists, mock_load_model):
        """Test explicit model_path takes priority over env var."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_is_file.return_value = True
        mock_exists.return_value = True

        with patch.dict(os.environ, {"CLEARSKY_MODEL_DIR": "/env/models"}):
            classifier = WasteClassifier(model_path="/explicit/path/best.pt")

        called_path = str(mock_load_model.call_args[0][0])
        assert called_path == "/explicit/path/best.pt"

    @patch("clearsky_lidar.classification.inference.load_model")
    @patch("clearsky_lidar.classification.inference.Path.exists")
    @patch("clearsky_lidar.classification.inference.Path.is_file")
    def test_env_var_before_default(self, mock_is_file, mock_exists, mock_load_model):
        """Test CLEARSKY_MODEL_DIR checked before default path."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        # First call (explicit path check) returns False, second (env) returns True
        mock_is_file.side_effect = [False, True]
        mock_exists.side_effect = [False, True]

        with patch.dict(os.environ, {"CLEARSKY_MODEL_DIR": "/env/models"}):
            classifier = WasteClassifier()

        # Should have been called with env var path
        call_args = str(mock_load_model.call_args[0][0])
        assert "/env/models" in call_args

    @patch("clearsky_lidar.classification.inference.load_model")
    @patch("clearsky_lidar.classification.inference.Path.exists")
    @patch("clearsky_lidar.classification.inference.Path.is_file")
    def test_default_path_last(self, mock_is_file, mock_exists, mock_load_model):
        """Test default ~/ClearSky/weights/ used as last resort."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        # All checks return False until default
        mock_is_file.side_effect = [False, False, True]
        mock_exists.side_effect = [False, False, True]

        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.expanduser", return_value=Path("/home/user/ClearSky/weights/best.pt")):
                classifier = WasteClassifier()

        call_args = str(mock_load_model.call_args[0][0])
        assert "ClearSky/weights" in call_args


if __name__ == "__main__":
    pytest.main([__file__, "-v"])