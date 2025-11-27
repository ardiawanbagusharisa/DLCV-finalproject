# RT-DETR Rain-Robust Object Detection

## Quick Reference

| Component | Input | Output | Checkpoint |
|-----------|-------|--------|------------|
| RT-DETR | [0,1] 640×640 | Detections | `PekingU/rtdetr_r18vd` |
| SPDNet | [0,255] | [0,255] | `model_spa.pt` |
| Rain Detector | 224×224 ImageNet-norm | Logits | `rain_detector_pretrained/rain_detector_best.pt` |
| Adapted SPDNet | [0,255] | [0,255] | `outputs_spdnet_feature_adaptation/spdnet_adapted_best.pt` |

## ⚠️ Critical Integration Pattern

```python
# SPDNet + RT-DETR (MUST follow this pattern or 80% mAP drop!)
spdnet_input = pixel_values * 255.0      # Scale UP for SPDNet
derain_output = spdnet_model(spdnet_input)
clean_images = torch.clamp(derain_output / 255.0, 0, 1)  # Scale DOWN for RT-DETR
```

## Dataset Paths
- **Clean COCO**: `E:\Python\DLCV\dataset\coco`
- **Rainy COCO**: `E:\Python\DLCV\dataset\coco_rain`
- **SPDNet external**: `E:\Python\DLCV\SPDNet`

## Performance Benchmarks

| Model | mAP | AP50 | Notes |
|-------|-----|------|-------|
| Vanilla RT-DETR | 0.341 | 0.483 | Baseline |
| Original SPDNet | 0.337 | 0.483 | Slight degradation |
| **Adapted SPDNet** | **0.355** | **0.507** | +4.1% ✅ |

## Terminal Setup
```powershell
& E:\Python\DLCV\.venv\Scripts\Activate.ps1  # ALWAYS activate first
```

## ⚠️ Critical Scripting Rules
- **NEVER use `Start-Sleep`**: When waiting for a script or process, do NOT use `Start-Sleep` as it triggers keyboard interrupts in some environments. Instead, use a proper wait mechanism or just let the script finish naturally.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `Training_SPDNet_Lightweight.py` | Train adapted SPDNet |
| `Eval_Adapted_SPDNet.py` | Compare models |
| `Pretrain_rain_detector.py` | Train rain classifier |

## Common Bugs

### Rain Labeling
```python
# ❌ WRONG: if 'rain' in path  (matches 'train2017')
# ✅ CORRECT: if 'coco_rain' in path
```

### OOM Fix
```python
BATCH_SIZE = 8; GRADIENT_ACCUMULATION_STEPS = 2
```

## Utils Reference

| Module | Key Functions |
|--------|---------------|
| `data_utils.py` | `load_datasets()`, `split_by_domain()` |
| `model_utils.py` | `load_model_and_processor()` |
| `spdnet_utils.py` | `load_spdnet_model()` (CUDA required) |
| `eval_utils.py` | `generate_predictions()`, `evaluate_coco()` |
| `rain_detector.py` | `RainDetector`, `get_rain_detection_transforms()` |
| `conditional_model.py` | `ConditionalRainRobustRTDETR` |
| `integrated_model.py` | `RainRobustRTDETR` |

## Config Defaults
```python
BATCH_SIZE = 8; LEARNING_RATE = 1e-5; NUM_EPOCHS = 12
FP16 = True; DATALOADER_WORKERS = 16
COCO_RATIO = 0.9; RAIN_RATIO = 0.1
DATASET_FRACTION = 0.05  # Quick test (1.0 for full)
```

## Architecture Notes
1. **Conditional**: Rain detector → selective SPDNet → RT-DETR (fastest)
2. **Integrated**: Always SPDNet → RT-DETR (balanced)
3. **Adapted SPDNet**: Feature-loss preserves detection features (best)

## GPU: 8GB min (inference), 16GB+ (training), CUDA required for SPDNet

## Debugging
- **OOM**: Reduce BATCH_SIZE, increase GRADIENT_ACCUMULATION_STEPS
- **Low mAP**: Check SPDNet scaling with `torch.min()/max()`
- **Slow eval**: Use DATASET_FRACTION = 0.05
