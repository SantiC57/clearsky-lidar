"""Tests for classification export_engine module."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call
from pathlib import Path
import tempfile
import shutil
import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExportEngine:
    """Tests for export_engine module."""

    @pytest.mark.gpu
    @patch("clearsky_lidar.classification.export_engine.load_model")
    @patch("clearsky_lidar.classification.export_engine.torch.cuda.is_available")
    @patch("shutil.move")
    @patch("clearsky_lidar.classification.export_engine.Path.exists")
    def test_export_engine_success(
        self,
        mock_exists,
        mock_shutil_move,
        mock_cuda_available,
        mock_load_model,
    ):
        """Test successful TensorRT engine export."""
        mock_exists.return_value = True
        mock_cuda_available.return_value = True

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Mock export to engine
        mock_model.export.return_value = "models/best.engine"

        from clearsky_lidar.classification.export_engine import export_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            (models_dir / "best.pt").touch()
            (models_dir / "data.yaml").write_text("names: [cardboard, glass, metal, paper, plastic, trash]\nnc: 6\n")

            result = export_engine(
                model_path=models_dir / "best.pt",
                output_dir=models_dir,
                imgsz=640,
                half=True,
            )

        assert result["engine_path"] == str(models_dir / "best.engine")
        assert result["format"] == "engine"
        mock_load_model.assert_called_once_with(models_dir / "best.pt")
        mock_model.export.assert_called_once_with(
            format="engine",
            imgsz=640,
            half=True,
            device=0,
        )

    @pytest.mark.gpu
    @patch("clearsky_lidar.classification.export_engine.load_model")
    @patch("clearsky_lidar.classification.export_engine.torch.cuda.is_available")
    @patch("shutil.move")
    @patch("clearsky_lidar.classification.export_engine.Path.exists")
    def test_export_engine_fallback_to_onnx(
        self,
        mock_exists,
        mock_shutil_move,
        mock_cuda_available,
        mock_load_model,
    ):
        """Test fallback to ONNX when engine export fails."""
        mock_exists.return_value = True
        mock_cuda_available.return_value = True

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # First export (engine) fails, second (ONNX) succeeds
        mock_model.export.side_effect = [
            Exception("TensorRT not available"),
            "models/best.onnx",
        ]

        from clearsky_lidar.classification.export_engine import export_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            (models_dir / "best.pt").touch()
            (models_dir / "data.yaml").write_text("names: [cardboard, glass, metal, paper, plastic, trash]\nnc: 6\n")

            result = export_engine(
                model_path=models_dir / "best.pt",
                output_dir=models_dir,
                imgsz=640,
                half=True,
            )

        assert result["engine_path"] == str(models_dir / "best.onnx")
        assert result["format"] == "onnx"
        assert "trtexec" in result["fallback_command"]
        assert mock_model.export.call_count == 2

    @pytest.mark.gpu
    @patch("clearsky_lidar.classification.export_engine.load_model")
    @patch("clearsky_lidar.classification.export_engine.torch.cuda.is_available")
    @patch("clearsky_lidar.classification.export_engine.Path.exists")
    @patch("clearsky_lidar.classification.export_engine.get_dataloaders")
    def test_verify_engine_accuracy(
        self,
        mock_get_dataloaders,
        mock_exists,
        mock_cuda_available,
        mock_load_model,
    ):
        """Test PyTorch vs TensorRT output verification."""
        mock_exists.return_value = True
        mock_cuda_available.return_value = True

        # Expected probabilities (matching between PT and engine)
        expected_probs = torch.tensor([[0.1, 0.2, 0.3, 0.1, 0.2, 0.1]])
        # Logits that produce these probabilities after softmax
        # Using log(probs) as approximation for logits
        logits = torch.log(expected_probs + 1e-10)

        # Mock PyTorch model - return logits that give expected probs after softmax
        mock_pt_model = MagicMock()
        mock_pt_model.model = MagicMock()
        mock_pt_model.model.return_value = logits
        mock_pt_model.model.eval = MagicMock()
        mock_pt_model.device = "cuda"

        # Mock TensorRT model - return same probabilities directly
        mock_engine_model = MagicMock()
        mock_engine_model.return_value = MagicMock()
        mock_engine_model.return_value.probs = MagicMock()
        mock_engine_model.return_value.probs.data = expected_probs.clone()
        mock_engine_model.model = MagicMock()
        mock_engine_model.model.eval = MagicMock()
        mock_engine_model.device = "cuda"

        mock_load_model.side_effect = [mock_pt_model, mock_engine_model]

        from clearsky_lidar.classification.export_engine import verify_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            (models_dir / "best.pt").touch()
            (models_dir / "best.engine").touch()
            (models_dir / "data.yaml").write_text("names: [cardboard, glass, metal, paper, plastic, trash]\nc: 6\n")

            # Mock validation dataloader - return torch tensors
            mock_val_loader = MagicMock()
            mock_val_loader.__len__.return_value = 2
            mock_val_loader.__iter__.return_value = iter([
                (torch.rand(1, 3, 640, 640), torch.tensor([0])),
                (torch.rand(1, 3, 640, 640), torch.tensor([1])),
            ])
            mock_get_dataloaders.return_value = (None, mock_val_loader, None)

            result = verify_engine(
                pt_model_path=models_dir / "best.pt",
                engine_path=models_dir / "best.engine",
                data_root=Path(tmpdir) / "data",
                imgsz=640,
                num_samples=2,
            )

        assert result["max_diff"] < 1e-3
        assert result["verified"] == True

    @pytest.mark.gpu
    @patch("clearsky_lidar.classification.export_engine.load_model")
    @patch("clearsky_lidar.classification.export_engine.torch.cuda.is_available")
    @patch("clearsky_lidar.classification.export_engine.Path.exists")
    @patch("clearsky_lidar.classification.export_engine.get_dataloaders")
    def test_verify_engine_fails_on_large_diff(
        self,
        mock_get_dataloaders,
        mock_exists,
        mock_cuda_available,
        mock_load_model,
    ):
        """Test verification fails when diff exceeds threshold."""
        mock_exists.return_value = True
        mock_cuda_available.return_value = True

        # PT probabilities
        pt_probs = torch.tensor([[0.1, 0.2, 0.3, 0.1, 0.2, 0.1]])
        logits = torch.log(pt_probs + 1e-10)

        # Mock PyTorch model
        mock_pt_model = MagicMock()
        mock_pt_model.model = MagicMock()
        mock_pt_model.model.return_value = logits
        mock_pt_model.model.eval = MagicMock()
        mock_pt_model.device = "cuda"

        # Mock TensorRT model - large diff
        mock_engine_model = MagicMock()
        mock_engine_model.return_value = MagicMock()
        mock_engine_model.return_value.probs = MagicMock()
        mock_engine_model.return_value.probs.data = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])  # Large diff
        mock_engine_model.model = MagicMock()
        mock_engine_model.model.eval = MagicMock()
        mock_engine_model.device = "cuda"

        mock_load_model.side_effect = [mock_pt_model, mock_engine_model]

        from clearsky_lidar.classification.export_engine import verify_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            (models_dir / "best.pt").touch()
            (models_dir / "best.engine").touch()
            (models_dir / "data.yaml").write_text("names: [cardboard, glass, metal, paper, plastic, trash]\nc: 6\n")

            mock_val_loader = MagicMock()
            mock_val_loader.__len__.return_value = 1
            mock_val_loader.__iter__.return_value = iter([
                (torch.rand(1, 3, 640, 640), torch.tensor([0])),
            ])
            mock_get_dataloaders.return_value = (None, mock_val_loader, None)

            result = verify_engine(
                pt_model_path=models_dir / "best.pt",
                engine_path=models_dir / "best.engine",
                data_root=Path(tmpdir) / "data",
                imgsz=640,
                num_samples=1,
            )

        assert result["max_diff"] > 1e-3
        assert result["verified"] == False


class TestCopyDataYaml:
    """Tests for copy_data_yaml function."""

    def test_copy_data_yaml_from_source(self):
        """Test copying data.yaml from source."""
        from clearsky_lidar.classification.export_engine import copy_data_yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            source_root.mkdir()
            (source_root / "data.yaml").write_text("names: [a, b, c]\nnc: 3\n")

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            result = copy_data_yaml(source_root, output_dir)

            assert result.exists()
            content = result.read_text()
            assert "a" in content
            assert "nc: 3" in content

    def test_copy_data_yaml_creates_default(self):
        """Test creating default data.yaml when source doesn't exist."""
        from clearsky_lidar.classification.export_engine import copy_data_yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            source_root.mkdir()
            # No data.yaml in source

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            result = copy_data_yaml(source_root, output_dir)

            assert result.exists()
            content = result.read_text()
            assert "cardboard" in content
            assert "nc: 6" in content


class TestMain:
    """Tests for main function."""

    @patch("clearsky_lidar.classification.export_engine.export_engine")
    @patch("clearsky_lidar.classification.export_engine.verify_engine")
    @patch("clearsky_lidar.classification.export_engine.copy_data_yaml")
    @patch("clearsky_lidar.classification.export_engine.argparse.ArgumentParser.parse_args")
    def test_main_export_and_verify(
        self,
        mock_parse_args,
        mock_copy_data_yaml,
        mock_verify,
        mock_export,
    ):
        """Test main calls export and verify."""
        mock_args = MagicMock()
        mock_args.model_path = Path("models/best.pt")
        mock_args.output_dir = Path("models")
        mock_args.imgsz = 640
        mock_args.half = True
        mock_args.verify = True
        mock_args.num_samples = 100
        mock_args.data_root = Path("/tmp/data")
        mock_args.tolerance = 1e-3
        mock_parse_args.return_value = mock_args

        mock_export.return_value = {"engine_path": "models/best.engine", "format": "engine"}
        mock_verify.return_value = {"verified": True, "max_diff": 1e-4}
        mock_copy_data_yaml.return_value = Path("models/data.yaml")

        from clearsky_lidar.classification.export_engine import main

        main()

        mock_export.assert_called_once()
        mock_verify.assert_called_once()
        mock_copy_data_yaml.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])