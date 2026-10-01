# YOLOv8-cls Waste Classification Fine-tuning — Specification

## Purpose

Fine-tune YOLOv8n-cls on 6-class waste dataset (cardboard, glass, metal, paper, plastic, trash), export TensorRT FP16 engine for Jetson Nano 4GB deployment, and integrate into ClearSky inference pipeline via `WasteClassifier` wrapper.

---

## Functional Requirements (FR)

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-01 | System MUST download and prepare Roboflow Waste Classification v1i.multiclass dataset with 6 classes | MUST | Dataset splits (train/val/test) load correctly; class mapping matches data.yaml |
| FR-02 | System MUST fine-tune YOLOv8n-cls with frozen backbone, unfreeze last layers, cosine LR scheduler | MUST | Training completes 50-100 epochs; validation loss decreases; top-1 accuracy ≥ 0.90 on test split |
| FR-03 | System MUST export trained model to TensorRT FP16 engine (`best.engine`) on Jetson Nano | MUST | Export command succeeds; engine file created; FP16 precision verified |
| FR-04 | System MUST provide `WasteClassifier` wrapper with `predict(image) -> (class_name, confidence, all_probs)` interface | MUST | Wrapper loads .pt on PC, .engine on Jetson automatically; returns correct tuple format |
| FR-05 | System MUST support confidence threshold parameter and batch inference | SHOULD | `predict(batch, conf=0.5)` filters low-confidence predictions; batch size > 1 works |
| FR-06 | System MUST integrate `WasteClassifier` into `clearsky_lidar.detection` facade and `__init__.py` exports | MUST | `from clearsky_lidar import WasteClassifier` works; facade routes classification correctly |
| FR-07 | System MUST track experiments in MLflow (sqlite) with params, metrics, artifacts, tags | SHOULD | MLflow UI shows experiment `yolov8-cls-waste`; runs log top-1, top-5, per-class F1, confusion matrix |
| FR-08 | System MUST resolve model path via `CLEARSKY_MODEL_DIR` env var or default `~/ClearSky/weights/` | MUST | Wrapper finds models when env var set; falls back to default; errors clearly if neither exists |

---

## Non-Functional Requirements (NFR)

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-01 | Performance | Inference latency < 30ms on Jetson Nano 4GB (FP16 TensorRT, batch=1) |
| NFR-02 | Performance | Model size < 10 MB (YOLOv8n-cls) |
| NFR-03 | Compatibility | ultralytics==8.0.196 pinned for JetPack 4.6.1 / TensorRT 7.1.3 / CUDA 10.2 |
| NFR-04 | Compatibility | Python 3.8 (Jetson) and 3.10+ (PC) both supported |
| NFR-05 | Deployment | Artifacts: `best.pt` + `best.engine` + `data.yaml` → `ClearSky/weights/` |
| NFR-06 | Deployment | Dynamic batch support in TensorRT engine (opt profile min=1, opt=4, max=8) |
| NFR-07 | Verification | PyTorch vs TensorRT output diff < 1e-3 (max absolute difference on logits) |
| NFR-08 | Training | Batch size 16-32 on RTX 3050 4GB; gradient accumulation if OOM |
| NFR-09 | Training | Early stopping patience=10; save best model by validation top-1 accuracy |

---

## Scenarios (Given/When/Then)

### Training

**Scenario: Happy path — full training run on PC**
- GIVEN Roboflow dataset downloaded with train/val/test splits
- GIVEN YOLOv8n-cls pretrained weights available
- WHEN `train_cls.py` runs with epochs=100, batch=16, lr0=0.01, cosine LR, freeze_backbone=True
- THEN training completes without OOM
- AND validation top-1 accuracy ≥ 0.90 on test split
- AND best.pt saved to `models/best.pt`

**Scenario: Edge case — class imbalance detected**
- GIVEN dataset has imbalanced class distribution (> 3:1 ratio)
- WHEN training starts
- THEN class weights computed automatically and applied to loss function
- AND per-class F1 logged to MLflow

**Scenario: Edge case — early stopping triggers**
- GIVEN validation top-1 accuracy plateaus for 10 epochs
- WHEN early stopping callback fires
- THEN training stops; best.pt corresponds to best validation epoch

### Validation

**Scenario: Happy path — test split evaluation**
- GIVEN trained best.pt exists
- WHEN validation script runs on test split
- THEN top-1 accuracy ≥ 0.90 logged
- AND top-5 accuracy logged
- AND per-class precision/recall/F1 logged
- AND confusion matrix artifact saved

### Export

**Scenario: Happy path — TensorRT FP16 export on Jetson**
- GIVEN best.pt copied to Jetson Nano
- GIVEN JetPack 4.6.1, TensorRT 7.1.3, ultralytics 8.0.196 installed
- WHEN `yolo export model=best.pt format=engine imgsz=640 half=True` executes
- THEN best.engine created successfully
- AND engine loads and runs inference

**Scenario: Edge case — export compatibility failure**
- GIVEN ultralytics version mismatch or TensorRT version mismatch
- WHEN export command runs
- THEN clear error message with version requirements
- AND fallback: export to ONNX then convert via trtexec (documented)

**Scenario: Verification — PyTorch vs TensorRT parity**
- GIVEN best.pt and best.engine both exist
- WHEN same input batch passed through both
- THEN max absolute difference on logits < 1e-3
- AND top-1 predictions match for ≥ 99% of samples

### Inference

**Scenario: Happy path — single image inference on PC**
- GIVEN WasteClassifier instantiated with model_path="best.pt"
- GIVEN input image (numpy array or PIL Image, 640x640 RGB)
- WHEN `predict(image)` called
- THEN returns `(class_name: str, confidence: float, all_probs: np.ndarray)`
- AND class_name in 6 waste classes
- AND confidence in [0, 1]
- AND all_probs.shape == (6,) summing to 1.0

**Scenario: Happy path — single image inference on Jetson**
- GIVEN WasteClassifier instantiated with model_path="best.engine"
- WHEN `predict(image)` called
- THEN same interface and output format as PC path

**Scenario: Edge case — confidence threshold filtering**
- GIVEN WasteClassifier with conf=0.5
- WHEN prediction confidence < 0.5
- THEN class_name="unknown" or None; confidence=0.0; all_probs unchanged

**Scenario: Edge case — batch inference**
- GIVEN list of 4 images
- WHEN `predict(batch)` called
- THEN returns list of 4 tuples matching single-image format

**Scenario: Error — model file not found**
- GIVEN neither CLEARSKY_MODEL_DIR nor default path contains model
- WHEN WasteClassifier instantiated
- THEN FileNotFoundError with clear message listing searched paths

### Integration

**Scenario: Happy path — facade import**
- GIVEN clearsky_lidar installed
- WHEN `from clearsky_lidar import WasteClassifier`
- THEN import succeeds; class is re-exported from detection facade

**Scenario: Happy path — detection facade routes classification**
- GIVEN detection.py facade imports WasteClassifier
- WHEN external code calls `detect_waste(image)` (or similar)
- THEN WasteClassifier used internally; classification result returned

**Scenario: Backward compatibility — existing detection API unchanged**
- GIVEN existing tests for detection module
- WHEN tests run after classification integration
- THEN all existing detection tests pass without modification

---

## Data Specs

### Dataset Structure (Roboflow Classification Format)
```
dataset/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├── valid/
│   ├── cardboard/
│   └── ...
└── test/
    ├── cardboard/
    └── ...
```

### Class Names & Mapping (6 classes, fixed order)
| Index | Class Name |
|-------|------------|
| 0 | cardboard |
| 1 | glass |
| 2 | metal |
| 3 | paper |
| 4 | plastic |
| 5 | trash |

### data.yaml Format (ultralytics classification)
```yaml
path: /path/to/dataset
train: train
val: valid
test: test
names:
  0: cardboard
  1: glass
  2: metal
  3: paper
  4: plastic
  5: trash
```

### Train/Val/Test Split Ratios
- Train: 70%
- Val: 20%
- Test: 10%
(Provided by Roboflow export; verified in dataset.py)

---

## Model Specs

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv8n-cls (ultralytics classification) |
| Input | 640×640 RGB (3 channels) |
| Output | 6-class probabilities (softmax) |
| Backbone | YOLOv8n (EfficientNet-like, ~3.2M params) |
| Fine-tuning Strategy | Freeze backbone (layers 0-7); unfreeze head (layers 8+) |
| Optimizer | AdamW |
| LR Scheduler | Cosine annealing (lr0=0.01, lrf=0.01) |
| Epochs | 100 (early stopping patience=10) |
| Batch Size | 16-32 (RTX 3050 4GB) |
| Weight Decay | 0.0005 |
| Augmentation | Albumentations: RandomResizedCrop, HorizontalFlip, ColorJitter, Normalize |
| Loss | CrossEntropyLoss with class weights (if imbalance) |
| Mixed Precision | FP16 (amp=True) |

---

## Export Specs

### TensorRT Engine Export Command
```bash
yolo export model=best.pt format=engine imgsz=640 half=True
```

### Export Parameters
| Parameter | Value |
|-----------|-------|
| Format | engine (TensorRT) |
| Precision | FP16 (half=True) |
| Input Size | 640×640 |
| Dynamic Batch | Enabled (min=1, opt=4, max=8) |
| Device | 0 (GPU) |
| Opset | 12+ (ultralytics default) |

### Verification Procedure
1. Load best.pt with ultralytics YOLO
2. Load best.engine with ultralytics YOLO
3. Generate random batch (N=4, 3, 640, 640)
4. Run inference on both
5. Compute max absolute difference on raw logits
6. Assert diff < 1e-3
7. Assert top-1 class match rate ≥ 99%

---

## Inference Specs

### Wrapper Class: `WasteClassifier`
**Location**: `clearsky_lidar/classification/inference.py`

```python
class WasteClassifier:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        device: Optional[str] = None
    ):
        ...
    
    def predict(
        self,
        image: Union[np.ndarray, Image.Image, List],
        conf: Optional[float] = None
    ) -> Union[Tuple[str, float, np.ndarray], List[Tuple]]:
        ...
```

### Interface Contract
| Method | Input | Output |
|--------|-------|--------|
| `predict(image)` | Single image (H,W,3) or batch (N,H,W,3) | `(class_name, confidence, all_probs)` or list of tuples |
| `predict(image, conf=0.5)` | Same + threshold | Filtered: low-conf → `("unknown", 0.0, probs)` |

### Auto-Loading Logic
- If `model_path` ends with `.engine` → load via TensorRT (Jetson)
- If `model_path` ends with `.pt` → load via PyTorch (PC)
- If `model_path` is None → resolve via `CLEARSKY_MODEL_DIR` env var or `~/ClearSky/weights/`
- Search order: env var → default → error

### Confidence Threshold
- Default: 0.25 (ultralytics default)
- Parameter: `conf` in `predict()` overrides instance default
- Behavior: predictions below threshold return `class_name="unknown"`, `confidence=0.0`

### Batch Support
- Input: list of images or numpy array (N, H, W, C)
- Output: list of tuples matching single-image format
- Batch size limited by TensorRT opt profile (max=8)

---

## Integration Specs

### Facade Re-export (`clearsky_lidar/detection.py`)
```python
# Existing detection imports...
from .classification.inference import WasteClassifier

__all__ = [..., "WasteClassifier"]
```

### Package Export (`clearsky_lidar/__init__.py`)
```python
from .classification.inference import WasteClassifier
from .detection import WasteClassifier  # re-export from facade

__all__ = [..., "WasteClassifier"]
```

### Model Path Resolution
```python
def resolve_model_path(model_path: Optional[str] = None) -> Path:
    if model_path:
        return Path(model_path)
    env_dir = os.environ.get("CLEARSKY_MODEL_DIR")
    if env_dir:
        return Path(env_dir) / "best.pt"  # or .engine based on platform
    default = Path.home() / "ClearSky" / "weights"
    return default / ("best.engine" if is_jetson() else "best.pt")
```

### Backward Compatibility
- Existing `clearsky_lidar.detection` public API unchanged
- New `WasteClassifier` added to `__all__` only
- No breaking changes to function signatures
- Existing tests must pass without modification

---

## MLflow Specs

| Parameter | Value |
|-----------|-------|
| Tracking URI | `sqlite:///mlflow.db` (local file) |
| Experiment Name | `yolov8-cls-waste` |
| Run Name | `{model}-{dataset}-{timestamp}` (e.g., `yolov8n-cls-roboflow-20260922-1430`) |

### Logged Parameters
- `model`: "yolov8n-cls"
- `dataset`: "roboflow-waste-v1i-multiclass"
- `epochs`, `batch_size`, `lr0`, `optimizer`, `freeze_backbone`
- `imgsz`, `augmentation_config`
- `hardware`: "RTX 3050 4GB" or "Jetson Nano 4GB"

### Logged Metrics (per epoch + final)
- `train/loss`, `val/loss`
- `val/top1_acc`, `val/top5_acc`
- `test/top1_acc`, `test/top5_acc`
- `test/per_class_f1` (6 values)
- `test/per_class_precision`, `test/per_class_recall`

### Logged Artifacts
- `best.pt` (model weights)
- `confusion_matrix.png` (test split)
- `training_curves.png` (loss/accuracy vs epoch)
- `data.yaml` (class mapping)

### Tags
- `model`: "yolov8n-cls"
- `dataset`: "roboflow-waste-classification"
- `task`: "classification"
- `hardware_train`: "RTX 3050 4GB"
- `hardware_inference`: "Jetson Nano 4GB"

---

## Test Specs

### Unit Tests (no GPU required — synthetic data)

| Test File | Coverage |
|-----------|----------|
| `tests/test_dataset.py` | Roboflow dataset loading, class mapping, splits, data.yaml parsing |
| `tests/test_model.py` | YOLOv8n-cls creation, forward pass, parameter count, freeze/unfreeze logic |
| `tests/test_export.py` | Export command construction, ONNX intermediate verification (mock TensorRT) |
| `tests/test_inference.py` | WasteClassifier instantiation, predict() interface, confidence threshold, batch, path resolution |
| `tests/test_mlflow.py` | MLflow logging structure, parameter/metric/artifact schema |

### Integration Tests (require GPU — marked `@pytest.mark.gpu`)

| Test File | Coverage |
|-----------|----------|
| `tests/test_integration_train.py` | 1-epoch smoke train → validate → export → inference (PC) |
| `tests/test_integration_jetson.py` | Engine load + inference on Jetson (manual/ci-gpu) |

### Synthetic Data for CI
- Generate random tensors: `(batch, 3, 640, 640)` for images
- Generate random labels: `(batch,)` with values 0-5
- Use `torch.utils.data.TensorDataset` for DataLoader testing
- Mock ultralytics YOLO `train()` and `val()` for fast CI

### Test Configuration (pytest.ini)
```ini
[tool.pytest.ini_options]
markers = [
    "gpu: marks tests requiring GPU",
    "integration: marks integration tests",
    "slow: marks slow tests (>30s)"
]
testpaths = ["tests"]
```

---

## Summary

| Section | Requirements | Scenarios |
|---------|--------------|-----------|
| Functional (FR) | 8 | — |
| Non-Functional (NFR) | 9 | — |
| Training | — | 3 |
| Validation | — | 1 |
| Export | — | 3 |
| Inference | — | 5 |
| Integration | — | 3 |
| **Total** | **17** | **15** |

**Coverage**: All happy paths covered; edge cases for class imbalance, early stopping, export failure, confidence threshold, batch inference, missing model, backward compatibility covered.

**Next Step**: Ready for design (sdd-design).