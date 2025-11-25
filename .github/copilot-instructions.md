# RT-DETR Rain-Robust Object Detection Project

## Project Overview

This project implements **rain-robust object detection** by combining de-raining models (SPDNet) with RT-DETR object detection. The goal is to improve detection accuracy in rainy conditions while optimizing inference speed through three main approaches:

1. **Conditional Model**: Lightweight rain detector + selective de-raining (fastest)
2. **Integrated Model**: End-to-end SPDNet + RT-DETR pipeline (balanced)
3. **Comparison Baseline**: Vanilla RT-DETR vs. de-raining methods

### Architecture Components

1. **Rain Detector** (Binary Classifier)
   - **Model**: MobileNetV3-Small (lightweight, ~3-5ms inference)
   - **Training**: Pretrained in `Pretrain_rain_detector.py`
   - **Checkpoint**: `./rain_detector_pretrained/rain_detector_best.pt`
   - **Purpose**: Conditionally apply de-raining only on detected rainy images
   - **Performance**: 96.8% accuracy, 99.5% precision on validation set

2. **De-raining Model**
   - **SPDNet**: Spatial Pyramid Dilated Network (~120ms)
   - **Location**: `E:\Python\DLCV\SPDNet` (external, not in this project)
   - **Checkpoint**: `E:\Python\DLCV\Project DLCV\model_spa.pt`
   - **Input**: [0, 255] pixel range (standard 8-bit images)
   - **Output**: [0, 255] pixel range
   - **CRITICAL DETAIL**: Requires proper scaling when used with RT-DETR (see integration notes)

3. **Detection Model**
   - **RT-DETR**: Real-time DETR from HuggingFace (`PekingU/rtdetr_r18vd`)
   - **Input**: [0, 1] normalized images at 640×640
   - **COCO-pretrained**, 80 object classes

### Dataset Structure

- **Clean COCO**: `E:\Python\DLCV\dataset\coco` (standard COCO format)
- **Rainy COCO**: `E:\Python\DLCV\dataset\coco_rain` (synthetic rain)
- **Training ratio**: 90% clean, 10% rainy (configurable)
- **Validation ratio**: 90% clean, 10% rainy (same as training)

## Performance Targets

### Rain Detector
- ✅ Accuracy: ≥ 96%
- ✅ Precision: ≥ 99% (minimize false positives)
- ✅ Recall: ≥ 95% (catch most rainy images)

### Conditional Model on Rainy COCO
- 🎯 Target mAP: ≥ 0.31 (match or exceed vanilla RT-DETR)
- 📊 Expected: ~10-15% improvement due to selective de-raining

### Integrated Model on Rainy COCO
- 🎯 Target mAP: ≥ 0.31 (match or exceed vanilla RT-DETR)
- 📊 Expected: ~5-10% improvement (less than conditional due to all-image de-raining)

### Adapted SPDNet (Feature-Based Adaptation)
- ✅ mAP: 0.355 (+4.1% vs baseline)
- ✅ AP50: 0.507 (+5.0% vs baseline)
- ✅ Outperforms Original SPDNet (0.337 mAP)

### Baseline (Vanilla RT-DETR on Rainy COCO)
- Current: mAP ≈ 0.341
- AP @ IoU=0.50: ≈ 0.483
- AP @ IoU=0.75: ≈ 0.366

## Component Integration Patterns

### Critical: Input/Output Range Handling

Each component expects specific input ranges and produces specific output ranges:

| Component | Input Range | Output Range | Notes |
|-----------|-------------|--------------|-------|
| RT-DETR processor | Any image format | [0, 1] normalized | Standard PIL/Pillow format accepted |
| Rain Detector | 224×224 ImageNet normalized | Logits | Must preprocess: resize + normalize |
| SPDNet | [0, 255] pixel values | [0, 255] pixel values | Requires scaling when chained |
| RT-DETR model | [0, 1] normalized at 640×640 | Detection results | Expects proper normalization range |

### Pattern 1: Conditional Model (Rain → De-rain → Detect)

```python
# Input: pixel_values [0, 1] at 640×640 from RT-DETR processor
# Step 1: Preprocess for rain detector
rain_input = self._preprocess_for_rain_detector(pixel_values)  # →224×224 ImageNet norm
rain_logits = self.rain_detector(rain_input)  # Binary classification
rain_mask = rain_logits > 0.5  # True if rainy

# Step 2: Conditionally de-rain only rainy images
clean_images = pixel_values.clone()
if rain_mask.any():
    # CRITICAL: Scale to [0, 255] for SPDNet
    spdnet_input = pixel_values[rain_mask] * 255.0
    derain_output = self.derain_module(spdnet_input)
    # CRITICAL: Scale back to [0, 1] for RT-DETR
    clean_images[rain_mask] = torch.clamp(derain_output / 255.0, 0, 1)

# Step 3: Detect on potentially de-rained images
outputs = self.detector(clean_images)
```

### Pattern 2: Integrated Model (Always De-rain)

```python
# Input: pixel_values [0, 1] at 640×640 from RT-DETR processor
# CRITICAL: SPDNet scaling must be applied
spdnet_input = pixel_values * 255.0  # Scale to [0, 255]
derain_output = self.derain_module(spdnet_input)
clean_images = torch.clamp(derain_output / 255.0, 0, 1)  # Scale back to [0, 1], clamp safety

# Detect on de-rained images
outputs = self.detector(clean_images)
```

### Pattern 3: Two-Stage Pipeline (For Comparison)

```python
# External de-raining followed by detection
from utils.drsformer_utils import load_drsformer_model

derain_model = load_drsformer_model(path)
spdnet_input = pixel_values * 255.0
derain_output = derain_model(spdnet_input)
clean_images = torch.clamp(derain_output / 255.0, 0, 1)

# Then pass through detector
outputs = detector(clean_images)
```

## Common Pitfalls & Troubleshooting

### Issue: Rain Detector Accuracy Poor

**Symptom**: Rain detection accuracy ~50%, treating all images same

**Root Cause**: 
1. Path-based labeling using substring match (`if 'rain' in path`) instead of exact folder (`if 'coco_rain' in path`)
2. Input preprocessing mismatch (receiving [0,1] but expecting ImageNet-normalized)
3. RGB/BGR color channel mismatch from OpenCV/Supervision

**Fix**:
```python
# ✅ CORRECT: Check specific folder
if 'coco_rain' in path.lower():  # Domain label
    
# ❌ WRONG: Substring match conflicts with 'train'
if 'rain' in path.lower():  # Matches 'train2017' too!
```

**Prevention**: 
- Always preprocess rain detector input via `_preprocess_for_rain_detector()`
- Use `get_rain_detection_transforms()` for ImageNet normalization
- Verify channel order with supervision: `supervision.detections_from_detections()`

### Issue: SPDNet Integration Breaks Detection (-80% mAP)

**Symptom**: Integrated model mAP drops from 0.306 → 0.064, detection count wrong

**Root Cause**: SPDNet output range mismatch
- SPDNet outputs [-4.6, 233.3] when receiving [0, 233]
- RT-DETR expects [-1, 1] range for internal math
- Values outside this break normalization layers

**Fix** (CRITICAL):
```python
# BEFORE: ❌ Just clipping loses information
clean_images = torch.clamp(spdnet_output, 0, 1)

# AFTER: ✅ Proper scaling preserves distribution
spdnet_input = pixel_values * 255.0  # Scale UP for SPDNet
derain_output = self.derain_module(spdnet_input)
clean_images = torch.clamp(derain_output / 255.0, 0, 1)  # Scale DOWN for RT-DETR
```

**Prevention**: 
- Always check `torch.min()` and `torch.max()` of SPDNet output
- Use debug script: `test_scaling_fix.py` (in root directory)
- Verify output range before RT-DETR input: should be [0, 1]

### Issue: OOM (Out of Memory) During Training

**Symptom**: CUDA out of memory error, typically with batch_size=16

**Root Cause**: Two models in GPU memory simultaneously

**Fix**:
```python
# In Training_conditional.py configuration
BATCH_SIZE = 8  # Reduce from 16
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 8×2 = 16
```

### Issue: Rain Detector Always Predicts Positive

**Symptom**: 100% rainy predictions on all images despite high accuracy in training

**Root Cause**: 
- Model trained on imbalanced data (all positive)
- Threshold problem (default logit threshold 0.5 wrong)
- Preprocessing not applied during inference

**Fix**:
```python
# Apply preprocessing before inference
rain_input = _preprocess_for_rain_detector(pixel_values)
rain_logits = self.rain_detector(rain_input)
rain_prob = torch.sigmoid(rain_logits)  # Convert logits to probability
is_rainy = rain_prob > 0.5  # Use 0.5 probability threshold, not logit
```

### Issue: Dataset Evaluation Slow (20+ minutes)

**Symptom**: Evaluation running for >20 minutes on rainy COCO

**Fix**: Use dataset fraction for testing
```python
# In evaluation script
DATASET_FRACTION = 0.05  # Use 5% (250 images) for testing
# Change to 1.0 for full evaluation when ready
```

Expected times:
- 5% (250 images): 2-3 minutes
- 50% (2500 images): 10-15 minutes  
- 100% (5000 images): 20-30 minutes

## Component Input/Output Reference

### Rain Detector

**Input**: 224×224 ImageNet-normalized tensor
```python
# Example preprocessing
from utils.rain_detector import get_rain_detection_transforms
transforms = get_rain_detection_transforms()
rain_input = transforms(image)  # Returns [3, 224, 224] normalized
```

**Output**: Binary logits (before sigmoid)
```python
logits = rain_detector(rain_input)  # Shape: [batch_size, 2] or [batch_size]
probability = torch.sigmoid(logits)  # Probability of rain
is_rainy = probability > 0.5
```

### SPDNet

**Input**: [0, 255] pixel values (not normalized)
```python
# Proper input creation
spdnet_input = pixel_values * 255.0  # Scale from [0,1] to [0,255]
```

**Output**: [0, 255] de-rained pixel values
```python
derain_output = spdnet_model(spdnet_input)
# Output range is typically [-4, 240+] - MUST be scaled back
clean_images = torch.clamp(derain_output / 255.0, 0, 1)
```

### RT-DETR

**Input**: [0, 1] normalized 640×640 image
```python
# Via processor
processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")
processed = processor(image)  # Returns [0, 1] normalized pixel_values
```

**Output**: Detection results with boxes, scores, labels
```python
outputs = model(**processed)
# outputs.boxes: [N, 4] in format [x1, y1, x2, y2]
# outputs.scores: [N] confidence scores
# outputs.labels: [N] class indices
```

- **`rain_detector.py`**: Binary rain classification model
  - `RainDetector`: MobileNetV3-based architecture
  - `RainDetectionDataset`: Converts detection dataset to binary classification
  - `get_rain_detection_transforms()`: ImageNet normalization + augmentation
  - **CRITICAL**: Uses 224×224 ImageNet normalization, requires preprocessing from 640×640

- **`data_utils.py`**: Dataset loading and augmentation
  - `load_datasets()`: Combines COCO + COCO_rain with configurable ratios
  - `create_rain_detection_datasets()`: Converts to binary classification format
  - `AugmentedDetectionDataset`: Domain-aware augmentations
  - `split_by_domain()`: Separates clean vs. rainy images
  
- **`model_utils.py`**: RT-DETR loading
  - `load_model_and_processor()`: Loads RT-DETR from HuggingFace
  
- **`spdnet_utils.py`**: SPDNet integration
  - `load_spdnet_model()`: Loads pretrained model from `model_spa.pt`
  - `derain_image()`: Applies de-raining to PIL images
  - **CRITICAL**: Expects [0, 255] input, outputs [0, 255]

- **`conditional_model.py`**: ⭐ **Conditional Model** - Rain detector + selective de-raining
  - `ConditionalRainRobustRTDETR`: Main class combining three components
  - `forward()`: Detects rain → conditionally de-rains → detects objects
  - `_preprocess_for_rain_detector()`: Resizes 640×640 to 224×224 + ImageNet norm
  - **KEY FIX**: Rain detector receives properly preprocessed inputs
  - Rain detection accuracy: 96.8%

- **`integrated_model.py`**: ⭐ **Integrated Model** - End-to-end SPDNet + RT-DETR
  - `RainRobustRTDETR`: Always applies de-raining on all images
  - `forward()`: SPDNet(image) → RT-DETR(de-rained)
  - **KEY FIX**: Scales input to [0, 255], scales output back to [0, 1]
  - Single forward pass (eliminates two-stage bottleneck)

- **`Training_SPDNet_Lightweight.py`**: ⭐ **SPDNet Adaptation** - Feature-based fine-tuning
  - Trains SPDNet to preserve detection features
  - Uses lightweight feature extractor and 320x320 resolution
  - Output: `outputs_spdnet_feature_adaptation/spdnet_adapted_best.pt`

- **`Eval_Adapted_SPDNet.py`**: Evaluation for adapted model
  - Compares Vanilla, Original SPDNet, and Adapted SPDNet
  - Computes COCO mAP and simple metrics

- **`eval_utils.py`**: Evaluation and metrics
  - `generate_predictions()`: Batch inference + COCO format
  - `evaluate_coco()`: COCO mAP calculation
  
### Training Scripts

- **`Pretrain_rain_detector.py`**: Train rain detector
  - Binary classification on full dataset
  - Output: `./rain_detector_pretrained/rain_detector_best.pt`
  - Time: ~5-10 minutes
  
- **`Training_conditional.py`**: Train conditional model
  - Phase 1: Train detection head + rain detector
  - Phase 2: Fine-tune SPDNet
  - Phase 3: End-to-end fine-tuning
  - Output: `./outputs_conditional/best_conditional/`
  
- **`Training.py`**: Train vanilla RT-DETR
  - Baseline training without de-raining
  - Output: `./outputs/best_from_training/`

### Evaluation Scripts

- **`Eval_integrated.py`**: Compare integrated model vs vanilla RT-DETR
  - Evaluates on rainy COCO validation set
  - `DATASET_FRACTION = 0.05` (5% = 250 images for fast testing)
  - Set to `1.0` for full evaluation
  
- **`Eval_conditional.py`**: Compare conditional model vs vanilla RT-DETR
  - Includes rain detection accuracy evaluation
  - `DATASET_FRACTION = 0.05` for fast testing
  - Shows when de-raining helps vs. hurts

- **`Eval_rain_compare.py`**: Detailed comparison of all methods
  - Vanilla RT-DETR baseline
  - Conditional model (rain detection + SPDNet)
  - Integrated model (always de-rain)
  - Generates PR curves and visualizations

## Critical Bug Fixes & Lessons Learned

### 1. Rain Detector Data Labeling (FIXED)
**Problem**: All images labeled as positive (rainy) due to checking `'rain' in path` which matched `'train2017'`
**Solution**: Check specifically for `'coco_rain'` folder in path
**Result**: Accuracy improved from 56% to 99.6%

### 2. Rain Detector Input Preprocessing (FIXED)
**Problem**: Receiving 640×640 [0,1] normalized images, but trained on 224×224 ImageNet-normalized inputs
**Solution**: Added `_preprocess_for_rain_detector()` to resize + normalize before rain detector
**Result**: Rain detection now works correctly with conditional model

### 3. SPDNet Input/Output Scaling (FIXED - CRITICAL)
**Problem**: SPDNet expects [0, 255] but receiving [0, 1]; outputs [0, 255] but RT-DETR expects [0, 1]
- Without fix: Performance dropped 80% (mAP 0.3 → 0.064)
- Root cause: Out-of-range values (-2.7 to 1.9) broke RT-DETR's internal math

**Solution**: Scale inputs `× 255`, scale outputs `÷ 255` then clamp
```python
spdnet_input = pixel_values * 255.0
derain_output = self.derain_module(spdnet_input)
clean_images = torch.clamp(derain_output / 255.0, 0, 1)
```

**Result**: Proper de-raining integration (currently testing performance improvement)

### Key Lessons
1. ✅ Use specific folder names for path-based labels, not substrings
2. ✅ Match input/output ranges to model's training assumptions
3. ✅ Scale instead of clip when converting between ranges
4. ✅ Use `register_buffer()` for normalization constants (automatically moved to device)

See `docs/RAIN_DETECTOR_AND_INTEGRATION_FIXES.md` for detailed analysis.

## Terminal & Development Setup

### IMPORTANT: Virtual Environment Activation

**ALWAYS activate the virtual environment before running any Python scripts:**

```powershell
# Activate virtual environment
& E:\Python\DLCV\.venv\Scripts\Activate.ps1

# Verify activation (should show (.venv) prefix)
python --version
```

**When using terminal through any tool**, ensure you run:
```powershell
& E:\Python\DLCV\.venv\Scripts\Activate.ps1
```

**DO NOT** use bare `python` or `pip` commands - they will use the system Python instead of the project's virtual environment.

### Common Development Commands

```powershell
# Activate venv
& E:\Python\DLCV\.venv\Scripts\Activate.ps1

# Train rain detector (required first)
python Pretrain_rain_detector.py

# Train conditional model
python Training_conditional.py

# Quick evaluation (5% dataset)
python Eval_conditional.py

# Full evaluation (100% dataset)
# Edit: DATASET_FRACTION = 1.0 in script first
python Eval_integrated.py

# Test specific component
python -c "from utils.rain_detector import RainDetector; print('OK')"
```

## Development Workflows

### Pre-requisite: Train Rain Detector

```powershell
& E:\Python\DLCV\.venv\Scripts\Activate.ps1
python Pretrain_rain_detector.py
```

**Output**: `./rain_detector_pretrained/rain_detector_best.pt`
**Time**: ~5-10 minutes
**Validation accuracy**: Should be ≥ 99%

### Workflow 1: Quick Testing (5% Dataset)

```powershell
& E:\Python\DLCV\.venv\Scripts\Activate.ps1

# Test conditional model
python Eval_conditional.py

# Test integrated model  
python Eval_integrated.py
```

**Time**: ~2-3 minutes each
**Purpose**: Verify no crashes, check logging, inspect early results

### Workflow 2: Full Evaluation (100% Dataset)

```powershell
& E:\Python\DLCV\.venv\Scripts\Activate.ps1

# Edit script to set DATASET_FRACTION = 1.0
# Then run:
python Eval_integrated.py
```

**Time**: ~20-30 minutes
**Purpose**: Compute official COCO metrics

### Workflow 3: Training Conditional Model

```powershell
& E:\Python\DLCV\.venv\Scripts\Activate.ps1

# Edit configuration at top of script
# Set epochs, batch size, learning rate
python Training_conditional.py
```

**Output**: 
- `./outputs_conditional/best_conditional/` (best model)
- `./outputs_conditional/final_conditional/` (final model)
- Training curves + metrics

## Performance Targets

### Rain Detector
- ✅ Accuracy: ≥ 96%
- ✅ Precision: ≥ 99% (minimize false positives)
- ✅ Recall: ≥ 95% (catch most rainy images)

### Conditional Model on Rainy COCO
- 🎯 Target mAP: ≥ 0.31 (match or exceed vanilla RT-DETR)
- 📊 Expected: ~10-15% improvement due to selective de-raining

### Integrated Model on Rainy COCO
- 🎯 Target mAP: ≥ 0.31 (match or exceed vanilla RT-DETR)
- 📊 Expected: ~5-10% improvement (less than conditional due to all-image de-raining)

### Baseline (Vanilla RT-DETR on Rainy COCO)
- Current: mAP ≈ 0.306
- AP @ IoU=0.50: ≈ 0.437
- AP @ IoU=0.75: ≈ 0.327

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
