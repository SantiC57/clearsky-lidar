"""Tests for classification train module."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call
from pathlib import Path
import tempfile
import shutil
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clearsky_lidar.classification.train import (
    create_data_yaml,
    train_model,
    main,
    create_freeze_unfreeze_callback,
)


class TestCreateDataYaml:
    """Tests for create_data_yaml function."""

    def setup_method(self):
        """Create temporary directory structure."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_root = self.temp_dir / "data"
        self.data_root.mkdir()

        # Create train/valid/test splits with _classes.csv
        for split in ["train", "valid", "test"]:
            split_dir = self.data_root / split
            split_dir.mkdir()
            csv_path = split_dir / "_classes.csv"
            with open(csv_path, "w") as f:
                f.write("filename,cardboard,glass,metal,paper,plastic,trash\n")
                f.write(f"img1.jpg,1,0,0,0,0,0\n")

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_create_data_yaml_basic(self):
        """Test creating data.yaml with default class names."""
        output_path = self.temp_dir / "data.yaml"
        create_data_yaml(self.data_root, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "train: " in content
        assert "val: " in content
        assert "names:" in content
        assert "cardboard" in content
        assert "glass" in content

    def test_create_data_yaml_custom_names(self):
        """Test creating data.yaml with custom class names."""
        output_path = self.temp_dir / "data.yaml"
        custom_names = ["class1", "class2"]
        create_data_yaml(self.data_root, output_path, class_names=custom_names)

        content = output_path.read_text()
        assert "class1" in content
        assert "class2" in content


class TestFreezeUnfreezeCallback:
    """Tests for create_freeze_unfreeze_callback function."""

    def test_callback_creation(self):
        """Test callback function is created correctly."""
        mock_model = MagicMock()
        callback_fn = create_freeze_unfreeze_callback(mock_model, freeze_epochs=10, total_epochs=100)

        assert callable(callback_fn)

    @patch("clearsky_lidar.classification.train.freeze_backbone_fn")
    @patch("clearsky_lidar.classification.train.unfreeze_head")
    def test_callback_freezes_at_epoch_0(self, mock_unfreeze_head, mock_freeze_backbone):
        """Test callback freezes backbone at epoch 0."""
        mock_model = MagicMock()
        callback_fn = create_freeze_unfreeze_callback(mock_model, freeze_epochs=10, total_epochs=100)

        mock_trainer = MagicMock()
        mock_trainer.epoch = 0  # 0-indexed

        callback_fn(mock_trainer)

        mock_freeze_backbone.assert_called_once_with(mock_model, freeze=True)
        mock_unfreeze_head.assert_called_once_with(mock_model)

    @patch("clearsky_lidar.classification.train.freeze_backbone_fn")
    @patch("clearsky_lidar.classification.train.unfreeze_head")
    def test_callback_unfreezes_after_freeze_epochs(self, mock_unfreeze_head, mock_freeze_backbone):
        """Test callback unfreezes backbone after freeze_epochs."""
        mock_model = MagicMock()
        callback_fn = create_freeze_unfreeze_callback(mock_model, freeze_epochs=10, total_epochs=100)

        mock_trainer = MagicMock()
        mock_trainer.epoch = 10  # After freeze_epochs (0-indexed)

        callback_fn(mock_trainer)

        # Should unfreeze (freeze=False)
        mock_freeze_backbone.assert_called_once_with(mock_model, freeze=False)

    @patch("clearsky_lidar.classification.train.freeze_backbone_fn")
    def test_callback_does_not_unfreeze_twice(self, mock_freeze_backbone):
        """Test callback doesn't unfreeze twice."""
        mock_model = MagicMock()
        callback_fn = create_freeze_unfreeze_callback(mock_model, freeze_epochs=10, total_epochs=100)

        mock_trainer = MagicMock()
        mock_trainer.epoch = 10
        callback_fn(mock_trainer)  # First call - unfreezes

        mock_trainer.epoch = 15
        callback_fn(mock_trainer)  # Second call - should not unfreeze again

        # Should only be called once (for the unfreeze at epoch 10)
        assert mock_freeze_backbone.call_count == 1


class TestTrainModel:
    """Tests for train_model function."""

    @pytest.mark.gpu
    @patch("clearsky_lidar.classification.model.YOLO")
    @patch("clearsky_lidar.classification.train.init_mlflow")
    @patch("clearsky_lidar.classification.train.mlflow.start_run")
    @patch("clearsky_lidar.classification.train.mlflow.log_params")
    @patch("clearsky_lidar.classification.train.mlflow.log_metrics")
    @patch("clearsky_lidar.classification.train.mlflow.log_artifact")
    @patch("clearsky_lidar.classification.train.get_dataloaders")
    @patch("clearsky_lidar.classification.train.create_data_yaml")
    @patch("clearsky_lidar.classification.train.convert_to_ultralytics_format")
    @patch("clearsky_lidar.classification.train.save_confusion_matrix_plot")
    @patch("clearsky_lidar.classification.train.save_classification_report_txt")
    @patch("clearsky_lidar.classification.train.save_model")
    @patch("clearsky_lidar.classification.train.evaluate_on_validation_set")
    def test_train_model_runs_one_epoch(
        self,
        mock_evaluate,
        mock_save_model,
        mock_save_report,
        mock_save_cm,
        mock_convert_dataset,
        mock_create_data_yaml,
        mock_get_dataloaders,
        mock_log_artifact,
        mock_log_metrics,
        mock_log_params,
        mock_start_run,
        mock_init_mlflow,
        mock_yolo_class,
    ):
        """Test train_model runs 1 epoch and logs to MLflow."""
        # Setup mocks
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock train results
        mock_results = MagicMock()
        mock_results.results_dict = {
            "train/loss": 0.5,
            "metrics/accuracy_top1": 0.8,
            "metrics/accuracy_top5": 0.95,
            "val/loss": 0.4,
            "metrics/f1_macro": 0.75,
        }
        mock_model.train.return_value = mock_results

        # Mock evaluation results
        mock_evaluate.return_value = {
            "confusion_matrix": MagicMock(),
            "classification_report": {},
            "f1_per_class": {name: 0.8 for name in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]},
            "f1_macro": 0.8,
        }

        # Mock dataloaders (not used since we mock evaluate)
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_get_dataloaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)

        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        # Call train_model with epochs=1 for quick test
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            data_root.mkdir()
            for split in ["train", "valid", "test"]:
                (data_root / split).mkdir(parents=True, exist_ok=True)
                (data_root / split / "_classes.csv").write_text("filename,cardboard,glass,metal,paper,plastic,trash\nimg1.jpg,1,0,0,0,0,0\n")

            results = train_model(
                data_root=data_root,
                epochs=1,
                batch=16,
                lr0=0.002,
                patience=20,
                imgsz=640,
                freeze_epochs=0,  # No freeze for quick test
                run_name="test-run",
            )

        # Verify MLflow was initialized
        mock_init_mlflow.assert_called_once()
        mock_start_run.assert_called_once()

        # Verify model.train was called with correct params
        mock_model.train.assert_called_once()
        train_kwargs = mock_model.train.call_args.kwargs
        assert train_kwargs["epochs"] == 1
        assert train_kwargs["batch"] == 16
        assert train_kwargs["lr0"] == 0.002
        assert train_kwargs["patience"] == 20
        assert train_kwargs["imgsz"] == 640

        # Verify params were logged
        mock_log_params.assert_called_once()
        logged_params = mock_log_params.call_args[0][0]
        assert logged_params["epochs"] == 1
        assert logged_params["batch"] == 16
        assert logged_params["lr0"] == 0.002

        # Verify evaluation was called
        mock_evaluate.assert_called_once()

        # Verify callback was added
        mock_model.add_callback.assert_called_once()

    @pytest.mark.gpu
    @patch("clearsky_lidar.classification.model.YOLO")
    @patch("clearsky_lidar.classification.train.init_mlflow")
    @patch("clearsky_lidar.classification.train.mlflow.start_run")
    @patch("clearsky_lidar.classification.train.mlflow.log_params")
    @patch("clearsky_lidar.classification.train.mlflow.log_metrics")
    @patch("clearsky_lidar.classification.train.mlflow.log_artifact")
    @patch("clearsky_lidar.classification.train.get_dataloaders")
    @patch("clearsky_lidar.classification.train.create_data_yaml")
    @patch("clearsky_lidar.classification.train.convert_to_ultralytics_format")
    @patch("clearsky_lidar.classification.train.save_confusion_matrix_plot")
    @patch("clearsky_lidar.classification.train.save_classification_report_txt")
    @patch("clearsky_lidar.classification.train.save_model")
    @patch("clearsky_lidar.classification.train.evaluate_on_validation_set")
    def test_train_model_freeze_unfreeze_schedule(
        self,
        mock_evaluate,
        mock_save_model,
        mock_save_report,
        mock_save_cm,
        mock_convert_dataset,
        mock_create_data_yaml,
        mock_get_dataloaders,
        mock_log_artifact,
        mock_log_metrics,
        mock_log_params,
        mock_start_run,
        mock_init_mlflow,
        mock_yolo_class,
    ):
        """Test that freeze/unfreeze schedule is applied correctly."""
        mock_model = MagicMock()
        mock_model.model.model = [MagicMock() for _ in range(12)]
        mock_yolo_class.return_value = mock_model

        mock_results = MagicMock()
        mock_results.results_dict = {
            "train/loss": 0.5,
            "metrics/accuracy_top1": 0.8,
            "metrics/accuracy_top5": 0.95,
            "val/loss": 0.4,
            "metrics/f1_macro": 0.75,
        }
        mock_model.train.return_value = mock_results

        mock_evaluate.return_value = {
            "confusion_matrix": MagicMock(),
            "classification_report": {},
            "f1_per_class": {name: 0.8 for name in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]},
            "f1_macro": 0.8,
        }

        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_get_dataloaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            data_root.mkdir()
            for split in ["train", "valid", "test"]:
                (data_root / split).mkdir(parents=True, exist_ok=True)
                (data_root / split / "_classes.csv").write_text("filename,cardboard,glass,metal,paper,plastic,trash\nimg1.jpg,1,0,0,0,0,0\n")

            train_model(
                data_root=data_root,
                epochs=15,
                batch=16,
                lr0=0.002,
                patience=20,
                imgsz=640,
                freeze_epochs=10,
                run_name="test-freeze",
            )

        # Verify train was called
        mock_model.train.assert_called_once()
        # Verify callback was added
        mock_model.add_callback.assert_called_once()


class TestMain:
    """Tests for main function."""

    @patch("clearsky_lidar.classification.train.train_model")
    @patch("clearsky_lidar.classification.train.argparse.ArgumentParser.parse_args")
    def test_main_calls_train_model(self, mock_parse_args, mock_train_model):
        """Test main parses args and calls train_model."""
        mock_args = MagicMock()
        mock_args.data_root = Path("/tmp/data")  # Use Path object
        mock_args.epochs = 1
        mock_args.batch = 16
        mock_args.lr0 = 0.002
        mock_args.patience = 20
        mock_args.imgsz = 640
        mock_args.freeze_epochs = 10
        mock_args.run_name = "test"
        mock_parse_args.return_value = mock_args

        main()

        mock_train_model.assert_called_once()
        call_kwargs = mock_train_model.call_args.kwargs
        assert call_kwargs["data_root"] == Path("/tmp/data")
        assert call_kwargs["epochs"] == 1
        assert call_kwargs["batch"] == 16
        assert call_kwargs["lr0"] == 0.002
        assert call_kwargs["patience"] == 20
        assert call_kwargs["imgsz"] == 640
        assert call_kwargs["freeze_epochs"] == 10
        assert call_kwargs["run_name"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])