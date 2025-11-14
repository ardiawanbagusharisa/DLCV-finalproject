# RT-DETR Rain-Robust Object Detection Project

## Project Overview

This project implements **rain-robust object detection** by combining de-raining models (SPDNet/DRSformer) with RT-DETR object detection. The goal is to improve detection accuracy in rainy conditions while optimizing inference speed through various integration strategies.

### Architecture Components

1. **De-raining Models** (preprocessing)
   - **SPDNet**: Spatial Pyramid Dilated Network (faster, ~120ms)
   - **DRSformer**: Transformer-based (higher quality, ~200ms)
   - Both located outside project: `E:\Python\DLCV\SPDNet` and `E:\Python\DLCV\DRSformer`
   
2. **Detection Model**
   - **RT-DETR**: Real-time DETR from HuggingFace (`PekingU/rtdetr_r18vd`)
   - COCO-pretrained, 80 object classes

3. **Two-Stage Pipeline** (current bottleneck)
   ```
   Rainy Image → De-raining Model → Clean Image → RT-DETR → Detections
   ```

### Dataset Structure

- **Clean COCO**: `E:\Python\DLCV\Project\dataset\coco` (standard COCO format)
- **Rainy COCO**: `E:\Python\DLCV\Project\dataset\coco_rain` (synthetic rain degradation)
- **Default mixing ratio**: 90% clean, 10% rainy for training robustness

## Code Organization

### Modular Architecture (Refactored from Jupyter Notebooks)

All utility code lives in `utils/`:

- **`data_utils.py`**: Dataset loading, augmentation, domain-aware sampling
  - `load_datasets()`: Combines COCO + COCO_rain with configurable ratios
  - `AugmentedDetectionDataset`: Domain-aware augmentations (lighter for rainy images)
  - `split_by_domain()`, `check_split_leakage()`: Domain analysis utilities
  
- **`model_utils.py`**: Model loading and configuration
  - `load_model_and_processor()`: Loads RT-DETR from HuggingFace
  
- **`training_utils.py`**: Custom trainer with domain balancing
  - `ObjectDetectionTrainer`: Overrides `get_train_dataloader()` for weighted sampling (~20% rainy per batch)
  - `FreezeBackboneCallback`: Freeze backbone for first N epochs, then unfreeze
  - `compute_metrics()`: COCO mAP computation during training
  
- **`eval_utils.py`**: Evaluation, visualization, COCO metrics
  - `run_inference()`: Single-image inference with optional de-raining
  - `generate_predictions()`: Batch predictions in COCO format
  - `evaluate_coco()`: Official COCO evaluation (AP, AP50, AP75, etc.)
  - `calculate_pr_curve_per_class()`, `plot_pr_curves()`: Per-class analysis
  
- **`spdnet_utils.py`**: SPDNet integration
  - **Critical**: SPDNet has hardcoded `.cuda()` calls - requires CUDA
  - `load_spdnet_model()`: Loads pretrained model from `model_spa.pt`
  - `derain_image()`: Preprocesses PIL images
  
- **`drsformer_utils.py`**: DRSformer integration
  - **Performance tip**: Use `tile=256` for 3-5x speedup on large images
  - `load_drsformer_model()`: Loads from `E:\Python\DLCV\DRSformer\pretrained_models\deraining.pth`
  - `derain_image()`: Supports tiled inference (default: 256px tiles, 32px overlap)

- **`integrated_model.py`**: ⭐ **NEW** - End-to-end integrated architecture
  - `RainRobustRTDETR`: Combines SPDNet + RT-DETR in single model
  - `load_integrated_model()`: Loads both pretrained models and combines them
  - Supports phased training (freeze/unfreeze individual components)
  - Single forward pass for inference (eliminates two-stage bottleneck)

### Main Scripts

- **`Training.py`**: End-to-end training pipeline for standalone RT-DETR
  - Configure at top: dataset ratios, epochs, batch size, learning rate
  - Uses domain-balanced sampling and gradient accumulation
  - Automatically plots training curves and saves best model

- **`Training_integrated.py`**: ⭐ **NEW** - Integrated SPDNet+RT-DETR training
  - End-to-end integration of de-raining and detection
  - 3-phase training strategy: detection head → SPDNet+head → full end-to-end
  - Uses pretrained weights from both SPDNet and RT-DETR
  - Saves integrated model for single-pass inference
  
- **`Eval_rain_compare.py`**: Comparative evaluation of 3 methods
  - Vanilla RT-DETR vs. SPDNet+RT-DETR vs. DRSformer+RT-DETR
  - Generates COCO metrics, PR curves, and visualizations

- **`Eval_integrated.py`**: ⭐ **NEW** - Evaluation for integrated model
  - Compares integrated model against vanilla RT-DETR baseline
  - Demonstrates end-to-end performance improvements

- **Notebooks**: Legacy interactive versions (HuggingFace_Training.ipynb, HuggingFace_Evaluation.ipynb)

## Critical Developer Knowledge

### Domain-Aware Training Strategy

This project uses **domain balancing** to prevent overfitting to clean images:

1. **Weighted sampling**: `ObjectDetectionTrainer` overrides dataloader to target ~20% rainy samples per batch
2. **Differential augmentation**: Rainy images get lighter augmentations to avoid compounding degradations
   ```python
   # Clean: full augmentation pipeline
   # Rainy: only HorizontalFlip + light brightness adjustment
   ```
3. **Validation splits**: Use `split_by_domain()` to evaluate separately on clean vs. rainy subsets

### Performance Optimization Patterns

1. **Mixed Precision (FP16)**
   - Enabled by default: `FP16 = True` in Training.py
   - 2x speedup with minimal accuracy loss on NVIDIA GPUs
   
2. **Gradient Accumulation**
   - Effective batch size = `BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS`
   - Default: 16 × 1 = 16 (adjust based on GPU memory)
   
3. **DataLoader Workers**
   - Default: 16 workers (matches available CPU cores)
   - Set via `DATALOADER_WORKERS` in Training.py
   
4. **DRSformer Tiling**
   - **Always use tiled inference**: `tile=256` in drsformer_utils
   - Full-image processing is 3-5× slower
   - Trade-off: slight quality loss at tile boundaries

### SPDNet CUDA Limitation

SPDNet source code has hardcoded `.cuda()` calls that prevent CPU-only execution:
```python
# Location: E:\Python\DLCV\SPDNet\src\model\spdnet.py (lines 147-148)
```
**Workaround**: Modify SPDNet source or use DRSformer (CUDA-optional) instead.

### COCO Evaluation Thresholds

Two thresholds serve different purposes:

- **Visualization threshold** (`CONFIDENCE_THRESHOLD = 0.3`): High threshold for clean visualizations
- **Evaluation threshold** (`INFERENCE_THRESHOLD = 0.01`): Low threshold for COCO eval to preserve score distribution

### Common Pitfalls

1. **Don't modify model architecture without retraining**: Pretrained weights are tightly coupled
2. **Check data leakage**: Use `check_split_leakage()` to verify no overlap between train/val
3. **Memory management**: Clear CUDA cache after evaluation (`torch.cuda.empty_cache()`)
4. **Bounding box format**: Always use Pascal VOC (x_min, y_min, x_max, y_max) internally

## Development Workflows

### Training the Integrated Model (Recommended)

```bash
# 1. Edit configuration in Training_integrated.py (top section)
COCO_RATIO = 0.9          # 90% clean images
RAIN_RATIO = 0.1          # 10% rainy images
NUM_EPOCHS = 12
BATCH_SIZE = 8            # Lower due to two models in memory
LEARNING_RATE = 1e-5

# Training phases:
PHASE1_EPOCHS = 2   # Train detection head only (SPDNet frozen)
PHASE2_EPOCHS = 8   # Train SPDNet + head (RT-DETR backbone frozen)
PHASE3_EPOCHS = 12  # Fine-tune everything end-to-end

# 2. Run integrated training
python Training_integrated.py

# 3. Outputs
# - Checkpoints: ./outputs_integrated/checkpoint-{step}/
# - Best model: ./outputs_integrated/best_integrated/
# - Final model: ./outputs_integrated/final_integrated/
# - Training curves: ./outputs_integrated/training_curves.png
```

### Training a Standalone Model

```bash
# 1. Edit configuration in Training.py (top section)
COCO_RATIO = 0.9          # 90% clean images
RAIN_RATIO = 0.1          # 10% rainy images
NUM_EPOCHS = 12
BATCH_SIZE = 16
LEARNING_RATE = 1e-5

# 2. Run training
python Training.py

# 3. Outputs
# - Checkpoints: ./outputs/checkpoint-{step}/
# - Best model: ./outputs/best_from_training/
# - Training curves: ./outputs/training_curves.png
# - TensorBoard logs: ./outputs/runs/
```

### Evaluating on Rainy Data

```bash
# Option 1: Evaluate integrated model (RECOMMENDED)
# 1. Edit Eval_integrated.py
INTEGRATED_MODEL_PATH = "./outputs_integrated/best_integrated"
DATASET_PATH = "E:\\Python\\DLCV\\Project\\dataset\\coco_rain"

# 2. Run evaluation
python Eval_integrated.py

# This compares:
# - Vanilla RT-DETR (baseline)
# - Integrated SPDNet+RT-DETR (single-pass inference)

# Option 2: Compare all de-raining methods
# 1. Edit Eval_rain_compare.py
MODEL_PATH = "./outputs/best_from_training"
DATASET_PATH = "E:\\Python\\DLCV\\Project\\dataset\\coco_rain"
DRSFORMER_TILE = 256  # Use tiled inference for speed

# 2. Run comparative evaluation
python Eval_rain_compare.py

# This compares:
# - Vanilla RT-DETR (baseline)
# - SPDNet + RT-DETR (two-stage)
# - DRSformer + RT-DETR (two-stage)
```

### Quick Inference Test

```python
# Option 1: Integrated model (single-pass, faster)
from utils.integrated_model import load_integrated_model
from utils.eval_utils import run_inference

# Load integrated model
model, processor = load_integrated_model(
    spdnet_path="./model_spa.pt",
    rtdetr_name="PekingU/rtdetr_r18vd"
)

# Run inference (automatic de-raining + detection)
results, size = run_inference(
    "rainy_image.jpg", 
    model, 
    processor, 
    device='cuda'
)

# Option 2: Two-stage pipeline (for comparison)
from utils.model_utils import load_model_and_processor
from utils.drsformer_utils import load_drsformer_model

# Load models separately
model, processor = load_model_and_processor("PekingU/rtdetr_r18vd")
derain_model = load_drsformer_model("path/to/deraining.pth")

# Run inference with de-raining
results, size = run_inference(
    "rainy_image.jpg", 
    model, 
    processor, 
    device='cuda',
    derain_model=derain_model  # Two-stage processing
)
```

## Integration Strategy (Future Work)

See `Project_goal.md` for detailed architecture proposals:

1. **Phase 1 (Quick wins)**: TensorRT + FP16 optimization (2-3x speedup)
2. **Phase 2 (Integration)**: Combine de-raining + detection into single model (end-to-end training)
3. **Phase 3 (Research)**: Feature-level de-raining or domain adaptation

**Key insight**: Current two-stage pipeline proves concept, but production requires integrated architecture to eliminate redundant computation.

## Dependencies & Environment

### Installation

```bash
pip install -r utils/requirements.txt
```

### External Model Dependencies

- SPDNet: Clone from source, place at `E:\Python\DLCV\SPDNet`
- DRSformer: Clone from source, place at `E:\Python\DLCV\DRSformer`

### GPU Requirements

- **Minimum**: 8GB VRAM (for inference)
- **Recommended**: 16GB+ VRAM (for training with batch_size=16)
- **SPDNet**: Requires CUDA (no CPU fallback without source modification)

## Project-Specific Conventions

1. **Absolute paths**: Use `E:\Python\DLCV\...` for dataset/model paths (Windows environment)
2. **Device selection**: Always check `torch.cuda.is_available()` before loading models
3. **Reproducibility**: Set `SEED = 42` for all random operations
4. **Progress tracking**: Use `tqdm` for long-running loops (dataset iteration, evaluation)
5. **Error handling**: Training/eval scripts use `max_retries=5` for dataset __getitem__ to handle corrupted images

## Debugging Tips

- **CUDA out of memory**: Reduce `BATCH_SIZE` or increase `GRADIENT_ACCUMULATION_STEPS`
- **Slow training**: Check `DATALOADER_WORKERS` (should be 4-16 depending on CPU cores)
- **Low mAP on rainy data**: Increase `RAIN_RATIO` in training or check de-raining model quality
- **Bbox validation errors**: Check `validate_bbox()` in data_utils.py for min_size thresholds
- **NaN losses**: Lower learning rate or check gradient clipping (`max_grad_norm`)
