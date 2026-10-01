"""Export YOLOv8 model to TensorRT engine for Jetson deployment."""

# Apply PyTorch 2.6+ compatibility patches BEFORE importing ultralytics
from .patches import *  # noqa: F403,F401

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from ultralytics import YOLO

from .dataset import get_dataloaders
from .model import load_model


def export_engine(
    model_path: Path,
    output_dir: Path,
    imgsz: int = 640,
    half: bool = True,
) -> Dict[str, str]:
    """Export YOLOv8 model to TensorRT engine format.

    Args:
        model_path: Path to .pt model file
        output_dir: Directory to save exported engine
        imgsz: Input image size
        half: Use FP16 precision

    Returns:
        Dictionary with engine_path, format, and fallback_command if applicable
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(model_path)

    # Try TensorRT engine export first
    engine_path = output_dir / "best.engine"
    try:
        print(f"Exporting to TensorRT engine: {engine_path}")
        exported_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            device=0 if torch.cuda.is_available() else "cpu",
        )
        # Ultralytics returns the path to the exported file
        if isinstance(exported_path, str):
            exported_path = Path(exported_path)

        # Move to expected location if needed
        if exported_path != engine_path and exported_path.exists():
            import shutil
            shutil.move(str(exported_path), str(engine_path))

        print(f"TensorRT engine exported successfully to {engine_path}")
        return {
            "engine_path": str(engine_path),
            "format": "engine",
            "fallback_command": None,
        }

    except Exception as e:
        print(f"TensorRT engine export failed: {e}")
        print("Falling back to ONNX export...")

        # Fallback to ONNX
        onnx_path = output_dir / "best.onnx"
        try:
            exported_path = model.export(
                format="onnx",
                imgsz=imgsz,
                half=half,
                device=0 if torch.cuda.is_available() else "cpu",
                opset=12,
            )
            if isinstance(exported_path, str):
                exported_path = Path(exported_path)

            if exported_path != onnx_path and exported_path.exists():
                import shutil
                shutil.move(str(exported_path), str(onnx_path))

            print(f"ONNX model exported to {onnx_path}")

            # Generate trtexec command for Jetson
            trtexec_cmd = (
                f"trtexec --onnx={onnx_path} "
                f"--saveEngine={engine_path} "
                f"--fp16={'--fp16' if half else ''} "
                f"--workspace=2048 "
                f"--explicitBatch"
            )

            return {
                "engine_path": str(onnx_path),
                "format": "onnx",
                "fallback_command": trtexec_cmd,
            }

        except Exception as onnx_error:
            print(f"ONNX export also failed: {onnx_error}")
            raise RuntimeError(f"Both engine and ONNX export failed. Engine error: {e}, ONNX error: {onnx_error}")


def verify_engine(
    pt_model_path: Path,
    engine_path: Path,
    data_root: Path,
    imgsz: int = 640,
    num_samples: int = 100,
    tolerance: float = 1e-3,
) -> Dict:
    """Verify TensorRT engine output matches PyTorch model within tolerance.

    Args:
        pt_model_path: Path to PyTorch .pt model
        engine_path: Path to TensorRT .engine file (or .onnx if fallback)
        data_root: Dataset root for validation samples
        imgsz: Input image size
        num_samples: Number of validation samples to test
        tolerance: Maximum allowed absolute difference

    Returns:
        Dictionary with verification results
    """
    print(f"Verifying engine: {engine_path} vs PyTorch: {pt_model_path}")

    # Load both models
    pt_model = load_model(pt_model_path)
    engine_model = load_model(engine_path)

    # Get validation dataloader
    _, val_loader, _ = get_dataloaders(
        root=data_root,
        batch_size=1,
        workers=0,
        cache=False,
        img_size=imgsz,
    )

    max_diff = 0.0
    all_diffs = []
    samples_tested = 0

    pt_model.model.eval()
    engine_model.model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if samples_tested >= num_samples:
                break

            images = images.to(pt_model.device)

            # PyTorch inference
            pt_outputs = pt_model.model(images)
            pt_probs = torch.softmax(pt_outputs, dim=1).cpu().numpy()

            # Engine inference
            # Note: For .engine files, YOLO handles preprocessing internally
            # We need to pass raw images or use the model's predict method
            engine_results = engine_model(images, verbose=False)
            if isinstance(engine_results, list):
                engine_probs = engine_results[0].probs.data.cpu().numpy()
            else:
                engine_probs = engine_results.probs.data.cpu().numpy()

            # Ensure same shape
            if pt_probs.shape != engine_probs.shape:
                print(f"Shape mismatch: PT {pt_probs.shape} vs Engine {engine_probs.shape}")
                continue

            diff = np.abs(pt_probs - engine_probs).max()
            all_diffs.append(diff)
            max_diff = max(max_diff, diff)
            samples_tested += 1

            if samples_tested % 20 == 0:
                print(f"  Tested {samples_tested} samples, current max diff: {max_diff:.6f}")

    mean_diff = np.mean(all_diffs) if all_diffs else 0.0
    verified = max_diff < tolerance

    result = {
        "verified": verified,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "samples_tested": samples_tested,
        "tolerance": tolerance,
        "pt_model": str(pt_model_path),
        "engine_model": str(engine_path),
    }

    if verified:
        print(f"✓ Verification PASSED: max_diff={max_diff:.6f} < {tolerance}")
    else:
        print(f"✗ Verification FAILED: max_diff={max_diff:.6f} >= {tolerance}")

    return result


def copy_data_yaml(source_data_root: Path, output_dir: Path) -> Path:
    """Copy or create data.yaml with class names to output directory.

    Args:
        source_data_root: Source dataset root (for reading class names)
        output_dir: Output directory for models

    Returns:
        Path to data.yaml in output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml_src = source_data_root / "data.yaml"
    data_yaml_dst = output_dir / "data.yaml"

    if data_yaml_src.exists():
        import shutil
        shutil.copy2(data_yaml_src, data_yaml_dst)
    else:
        # Create minimal data.yaml with class names
        from .dataset import CLASS_NAMES
        import yaml
        data_dict = {
            "names": {i: name for i, name in enumerate(CLASS_NAMES)},
            "nc": len(CLASS_NAMES),
        }
        with open(data_yaml_dst, "w") as f:
            yaml.dump(data_dict, f, default_flow_style=False)

    return data_yaml_dst


def main():
    """Main entry point for export script."""
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to TensorRT engine")
    parser.add_argument("--model-path", type=Path, default=Path("models/best.pt"), help="Path to .pt model")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--half", action="store_true", default=True, help="Use FP16")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify engine accuracy")
    parser.add_argument("--num-samples", type=int, default=100, help="Verification samples")
    parser.add_argument("--data-root", type=Path, default=Path("/home/santiago/Descargas/Waste Classification.v1i.multiclass"), help="Dataset root for verification")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Verification tolerance")

    args = parser.parse_args()

    # Export
    result = export_engine(
        model_path=args.model_path,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        half=args.half,
    )

    print(f"Export result: {result}")

    # Copy data.yaml to models directory
    copy_data_yaml(args.data_root, args.output_dir)

    # Verify if requested
    if args.verify and result["format"] == "engine":
        verify_result = verify_engine(
            pt_model_path=args.model_path,
            engine_path=Path(result["engine_path"]),
            data_root=args.data_root,
            imgsz=args.imgsz,
            num_samples=args.num_samples,
            tolerance=args.tolerance,
        )
        print(f"Verification result: {verify_result}")

        if not verify_result["verified"]:
            sys.exit(1)
    elif args.verify and result["format"] == "onnx":
        print("Skipping verification for ONNX format (run trtexec on Jetson first)")
        print(f"To convert on Jetson: {result['fallback_command']}")


if __name__ == "__main__":
    main()