"""Tests for classification inference module."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from clearsky_lidar.classification.inference import WasteClassifier
from clearsky_lidar.classification.dataset import CLASS_NAMES


class TestWasteClassifierIntegration:
    """Integration-style tests for WasteClassifier with mocked model."""

    @patch("clearsky_lidar.classification.inference.load_model")
    def test_class_names_order_matches_dataset(self, mock_load_model):
        """Test classifier uses correct class names order."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_result = MagicMock()
        mock_result.probs.data.cpu().numpy.return_value = np.array([0.1, 0.2, 0.5, 0.1, 0.05, 0.05])
        mock_model.return_value = [mock_result]

        classifier = WasteClassifier(model_path="models/best.pt")
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        class_name, confidence, probs_dict = classifier.predict(image)

        # Verify class name is from CLASS_NAMES
        assert class_name in CLASS_NAMES
        # Verify probs dict has all class names
        assert set(probs_dict.keys()) == set(CLASS_NAMES)
        # Verify index 2 (metal) has highest prob (0.5)
        assert class_name == CLASS_NAMES[2]  # metal

    @patch("clearsky_lidar.classification.inference.load_model")
    def test_predict_batch_consistency(self, mock_load_model):
        """Test predict_batch gives same results as multiple predict calls."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        probs = np.array([0.1, 0.2, 0.5, 0.1, 0.05, 0.05])
        mock_result = MagicMock()
        mock_result.probs.data.cpu().numpy.return_value = probs
        mock_model.return_value = [mock_result, mock_result]

        classifier = WasteClassifier(model_path="models/best.pt")
        image1 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        batch_results = classifier.predict_batch([image1, image2])
        single_results = [classifier.predict(image1), classifier.predict(image2)]

        assert len(batch_results) == len(single_results) == 2
        for batch_res, single_res in zip(batch_results, single_results):
            assert batch_res[0] == single_res[0]  # class_name
            assert np.isclose(batch_res[1], single_res[1])  # confidence
            assert batch_res[2] == single_res[2]  # probs_dict

    @patch("clearsky_lidar.classification.inference.load_model")
    def test_conf_threshold_default(self, mock_load_model):
        """Test default confidence threshold is 0.55."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_result = MagicMock()
        mock_result.probs.data.cpu().numpy.return_value = np.array([0.1, 0.2, 0.5, 0.1, 0.05, 0.05])
        mock_model.return_value = [mock_result]

        classifier = WasteClassifier(model_path="models/best.pt")
        assert classifier.conf_threshold == 0.55

    @patch("clearsky_lidar.classification.inference.load_model")
    def test_conf_threshold_custom(self, mock_load_model):
        """Test custom confidence threshold."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_result = MagicMock()
        mock_result.probs.data.cpu().numpy.return_value = np.array([0.1, 0.2, 0.5, 0.1, 0.05, 0.05])
        mock_model.return_value = [mock_result]

        classifier = WasteClassifier(model_path="models/best.pt", conf_threshold=0.8)
        assert classifier.conf_threshold == 0.8

    @patch("clearsky_lidar.classification.inference.load_model")
    def test_probs_dict_values_sum_to_one(self, mock_load_model):
        """Test probs_dict values sum to approximately 1.0."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        probs = np.array([0.1, 0.2, 0.5, 0.1, 0.05, 0.05])
        mock_result = MagicMock()
        mock_result.probs.data.cpu().numpy.return_value = probs
        mock_model.return_value = [mock_result]

        classifier = WasteClassifier(model_path="models/best.pt")
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        _, _, probs_dict = classifier.predict(image)

        total = sum(probs_dict.values())
        assert np.isclose(total, 1.0, atol=1e-5)

    @patch("clearsky_lidar.classification.inference.load_model")
    def test_predict_handles_different_image_sizes(self, mock_load_model):
        """Test predict works with different input image sizes (YOLO handles resize)."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_result = MagicMock()
        mock_result.probs.data.cpu().numpy.return_value = np.array([0.1, 0.2, 0.5, 0.1, 0.05, 0.05])
        mock_model.return_value = [mock_result]

        classifier = WasteClassifier(model_path="models/best.pt")

        # Test various sizes
        for h, w in [(480, 640), (640, 480), (640, 640), (1280, 720)]:
            image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            result = classifier.predict(image)
            assert isinstance(result, tuple)
            assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])