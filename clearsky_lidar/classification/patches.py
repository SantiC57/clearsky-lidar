"""PyTorch 2.6+ compatibility patches for ultralytics.

This module must be imported BEFORE importing ultralytics to ensure
the patches are applied in time.
"""

import os
import sys
import torch

# Fix for PyTorch 2.6+ weights_only=True default
# Must be set before importing ultralytics
os.environ.setdefault("TORCH_WEIGHTS_ONLY", "0")

# Compatibility fix for older ultralytics checkpoints that reference 'ultralytics.yolo'
# Current ultralytics 8.x doesn't have this module, but old checkpoints do
import ultralytics
if not hasattr(ultralytics, 'yolo'):
    # Create alias for backward compatibility
    import ultralytics.nn.tasks as yolo_tasks
    sys.modules['ultralytics.yolo'] = yolo_tasks
    sys.modules['ultralytics.yolo.utils'] = ultralytics.utils
    sys.modules['ultralytics.yolo.utils.checks'] = ultralytics.utils.checks
    sys.modules['ultralytics.yolo.utils.torch_utils'] = ultralytics.utils.torch_utils
    # Add data module
    sys.modules['ultralytics.yolo.data'] = ultralytics.data

# Patch ultralytics torch_safe_load to use weights_only=False
import ultralytics.nn.tasks as ultralytics_tasks
_original_torch_safe_load = ultralytics_tasks.torch_safe_load

def _patched_torch_safe_load(weight):
    """Patched version that uses weights_only=False for trusted ultralytics weights."""
    from ultralytics.utils.downloads import attempt_download_asset
    from ultralytics.utils.checks import check_suffix

    check_suffix(file=weight, suffix='.pt')
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        # Use weights_only=False for trusted ultralytics weights
        ckpt = torch.load(file, map_location='cpu', weights_only=False)
        return ckpt, file
    except ModuleNotFoundError as e:
        # Handle missing modules (should not happen with our compatibility aliases)
        if e.name == 'models':
            raise TypeError(
                f'{weight} appears to be an Ultralytics YOLOv5 model. '
                f'Not compatible with YOLOv8.') from e
        import logging
        logging.warning(f'{weight} requires {e.name}, attempting to install...')
        from ultralytics.utils.checks import check_requirements
        check_requirements(e.name)
        ckpt = torch.load(file, map_location='cpu', weights_only=False)
        return ckpt, file

ultralytics_tasks.torch_safe_load = _patched_torch_safe_load

# Register safe globals for ultralytics model loading (backup)
try:
    from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, PoseModel
    from ultralytics.nn.modules import (
        Conv, Conv2, DWConv, DWConvTranspose2d, ConvTranspose,
        Bottleneck, BottleneckCSP, C1, C2, C2f, C3, C3Ghost, C3TR, C3x,
        SPP, SPPF, Focus, GhostBottleneck, GhostConv, HGBlock, HGStem,
        RepC3, RepConv, Concat, Detect, Segment, Pose, Classify, RTDETRDecoder,
        AIFI, TransformerBlock, TransformerLayer, MLPBlock, LayerNorm2d,
        DFL, Proto
    )
    torch.serialization.add_safe_globals([
        ClassificationModel,
        DetectionModel,
        SegmentationModel,
        PoseModel,
        Conv, Conv2, DWConv, DWConvTranspose2d, ConvTranspose,
        Bottleneck, BottleneckCSP, C1, C2, C2f, C3, C3Ghost, C3TR, C3x,
        SPP, SPPF, Focus, GhostBottleneck, GhostConv, HGBlock, HGStem,
        RepC3, RepConv, Concat, Detect, Segment, Pose, Classify, RTDETRDecoder,
        AIFI, TransformerBlock, TransformerLayer, MLPBlock, LayerNorm2d,
        DFL, Proto,
        torch.nn.modules.container.Sequential,
        torch.nn.modules.container.ModuleList,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.activation.SiLU,
        torch.nn.modules.pooling.MaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.normalization.GroupNorm,
        torch.nn.modules.dropout.Dropout,
        torch.nn.modules.activation.Hardswish,
        torch.nn.modules.activation.Mish,
    ])
except (ImportError, AttributeError):
    # If ultralytics structure changes or torch version < 2.6, ignore
    pass

# Patch strip_optimizer to use weights_only=False
import ultralytics.utils.torch_utils as torch_utils
_original_strip_optimizer = torch_utils.strip_optimizer

def _patched_strip_optimizer(f, s=''):  # noqa: C901
    """Patched strip_optimizer that uses weights_only=False."""
    from copy import deepcopy
    import torch
    x = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    if x.get('ema'):
        x['model'] = deepcopy(x['ema'].float())
    else:
        x['model'] = deepcopy(x['model'].float())
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in x['train_args'].items() if k != 'data'}
    # Remove optimizer
    for k in ['optimizer', 'best_fitness', 'ema', 'updates']:
        x.pop(k, None)
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    return x

torch_utils.strip_optimizer = _patched_strip_optimizer

# Also patch classify train module which imports strip_optimizer directly
import ultralytics.models.yolo.classify.train as classify_train
classify_train.strip_optimizer = _patched_strip_optimizer

# Also patch detection/segmentation/pose train modules
for module_name in [
    'ultralytics.models.yolo.detect.train',
    'ultralytics.models.yolo.segment.train',
    'ultralytics.models.yolo.pose.train',
]:
    try:
        mod = __import__(module_name, fromlist=['strip_optimizer'])
        mod.strip_optimizer = _patched_strip_optimizer
    except (ImportError, AttributeError):
        pass