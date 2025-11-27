# Feature-Level De-raining Implementation Guide

## Overview

This document describes the **feature-level de-raining** approach for RT-DETR object detection - a novel alternative to traditional pixel-level de-raining (SPDNet).

## Why Feature-Level De-raining?

### Current Bottleneck (Pixel-Level)
```
Rainy Image → SPDNet (182ms) → Clean Image → RT-DETR (32ms) → Detections
              ↑ SLOW!
```

### New Approach (Feature-Level)
```
Rainy Image → RT-DETR Backbone (15ms) → Feature Enhancement (5ms) → Encoder → Decoder
                                         ↑ FAST!
```

## Performance Comparison

| Approach | De-rain Time | Total Time | FPS | mAP Impact |
|----------|--------------|------------|-----|------------|
| **No De-rain** | 0ms | 32ms | 31.3 | Baseline |
| **SPDNet (Pixel)** | 182ms | 214ms | 4.7 | -1.4% (original) / +4.1% (adapted) |
| **Feature-Level (Multiscale)** | ~8ms | ~40ms | 25 | TBD (needs training) |
| **Feature-Level (Lightweight)** | ~3ms | ~35ms | 28.5 | TBD (needs training) |

## Architecture

### RT-DETR Backbone Feature Maps

```
Input Image (640×640, 3ch)
    ↓
Stage 1: 64 channels, 160×160 (not used by encoder)
Stage 2: 128 channels, 80×80   ← encoder_input_proj[0]
Stage 3: 256 channels, 40×40   ← encoder_input_proj[1]
Stage 4: 512 channels, 20×20   ← encoder_input_proj[2]
    ↓
[FEATURE DE-RAINING INSERTED HERE]
    ↓
Hybrid Encoder (AIFI + CCFF)
    ↓
Transformer Decoder
    ↓
Detection Heads
```

### Feature De-raining Module Options

#### 1. Multi-Scale Feature De-rain (`MultiScaleFeatureDerain`)
- **Best for**: Maximum accuracy
- **Parameters**: ~1.5M
- **Latency**: ~8ms
- **Components**:
  - Per-scale rain mask estimation
  - CBAM attention (channel + spatial)
  - Residual refinement blocks

```python
from utils.feature_derain import create_feature_derain_rtdetr

model = create_feature_derain_rtdetr(
    derain_type="multiscale",
    num_residual_blocks=2
)
```

#### 2. Lightweight Feature De-rain (`LightweightFeatureDerain`)
- **Best for**: Maximum speed
- **Parameters**: ~50K
- **Latency**: ~3ms
- **Components**:
  - Spatial attention only
  - Learnable suppression factors
  - No heavy convolutions

```python
model = create_feature_derain_rtdetr(
    derain_type="lightweight"
)
```

## Key Technical Innovations

### 1. Rain Mask Estimation at Feature Level
Unlike pixel-level where rain streaks are visible, at feature level rain manifests as:
- High-frequency noise in certain channels
- Spatially correlated activation patterns
- Cross-channel inconsistencies

```python
class FeatureDerainBlock:
    # Estimate which spatial regions have rain-affected features
    rain_mask = self.rain_mask(features)  # (B, 1, H, W)
    
    # Suppress rain regions with learnable factor
    suppressed = features * (1 - rain_mask * self.blend)
    
    # Refine with residual blocks
    refined = self.refine(suppressed)
```

### 2. CBAM Attention for Rain Detection
- **Channel Attention**: Identifies which feature channels are affected by rain
- **Spatial Attention**: Identifies which spatial locations have rain artifacts

### 3. Multi-Scale Processing
Rain has different scales:
- Large rain streaks → affect low-res features (Stage 4: 512ch, 20×20)
- Small rain drops → affect high-res features (Stage 2: 128ch, 80×80)

Processing at all scales ensures comprehensive de-raining.

## Training Strategy

### Phase 1: De-rain Module Training (Epochs 1-5)
```python
# Freeze detector, train only de-raining
for param in model.rtdetr.parameters():
    param.requires_grad = False

optimizer = AdamW(model.derain_module.parameters(), lr=1e-4)
```

**Loss**: Detection loss (IoU + classification) from frozen detector
- Forces de-raining module to produce features that detector likes
- No need for paired clean/rainy images!

### Phase 2: Joint Training (Epochs 6-15)
```python
# Unfreeze detector with lower LR
optimizer = AdamW([
    {'params': model.derain_module.parameters(), 'lr': 1e-4},
    {'params': model.rtdetr.parameters(), 'lr': 1e-5}  # 10x lower
])
```

### Training Data
Use existing rainy COCO dataset:
- Clean COCO: `E:\Python\DLCV\dataset\coco`
- Rainy COCO: `E:\Python\DLCV\dataset\coco_rain`

Mix ratio: 70% rainy, 30% clean (to prevent overfitting to rain-only)

## Training

The training process is handled by `Training_FeatureDerain.py`.

### Key Features
1.  **Two-Phase Training**:
    *   **Phase 1**: Train *only* the de-raining module. Backbone and detector are frozen.
    *   **Phase 2**: Joint fine-tuning of de-raining module and detector (with lower LR for detector).
2.  **Loss Function**: Uses standard RT-DETR detection loss. No paired clean/rainy images required.
3.  **Dataset Mixing**: Uses a mix of clean (30%) and rainy (70%) images to prevent over-processing clean images.

### Usage

```powershell
# Activate environment
& E:\Python\DLCV\.venv\Scripts\Activate.ps1

# Run training
python Training_FeatureDerain.py
```

### Configuration (in script)
*   `DERAIN_TYPE`: "lightweight" (default) or "multiscale"
*   `PERCENT_DATASET`: Percentage of dataset to use (default 10% for speed)
*   `PHASE1_EPOCHS`: Epochs for de-rain module only (default 5)
*   `PHASE2_EPOCHS`: Epochs for joint fine-tuning (default 10)

## Usage Example

```python
import torch
from utils.feature_derain import create_feature_derain_rtdetr
from transformers import RTDetrImageProcessor

# Create model
model = create_feature_derain_rtdetr(
    model_name="PekingU/rtdetr_r18vd",
    derain_type="multiscale",  # or "lightweight"
    freeze_backbone=False,
    device="cuda"
)

# Load processor
processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")

# Inference
from PIL import Image
image = Image.open("rainy_image.jpg")
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to("cuda")

with torch.no_grad():
    outputs = model(pixel_values)
    
# Post-process detections
results = processor.post_process_object_detection(
    outputs, 
    target_sizes=[(image.height, image.width)],
    threshold=0.5
)
```

## Comparison with Existing Approaches

| Aspect | SPDNet (Pixel) | Feature-Level | Notes |
|--------|----------------|---------------|-------|
| **Speed** | 182ms | 3-8ms | 22-60x faster |
| **Memory** | High (full image) | Low (feature maps) | ~4x less |
| **Training** | Needs paired data | Works with detection loss | Simpler |
| **Interpretability** | Visual (can see de-rained image) | Hidden (no visual output) | Trade-off |
| **Artifacts** | May blur edges | Preserves features | Better for detection |
| **Domain Gap** | Trained for visual quality | Trained for detection | Task-aligned |

## Expected Results

Based on similar approaches in literature:
- **Multiscale**: Expected +2-4% mAP over vanilla on rainy images
- **Lightweight**: Expected +1-2% mAP with minimal speed impact
- Both should maintain performance on clean images (no degradation)

## Files

| File | Purpose |
|------|---------|
| `utils/feature_derain.py` | Core implementation |
| `Training_FeatureDerain.py` | Training script (TODO) |
| `Eval_FeatureDerain.py` | Evaluation script (TODO) |

## Next Steps

1. **Test building blocks**: Run `python utils/feature_derain.py`
2. **Create training script**: Implement `Training_FeatureDerain.py`
3. **Train on rainy COCO**: ~10 epochs with detection loss
4. **Evaluate on MixedRain**: Compare with SPDNet approaches
5. **Optimize for deployment**: TensorRT compilation if needed

## References

- CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)
- RT-DETR: Real-Time Detection Transformer (Zhao et al., 2023)
- Feature-level image restoration concepts from FFA-Net, GridDehazeNet
