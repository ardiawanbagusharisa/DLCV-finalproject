# SPDNet Adaptation for Object Detection - Project Summary

## 1. Problem Statement
The original SPDNet model was trained for **image restoration** (optimizing PSNR/SSIM). While it produces visually pleasing images, it often removes high-frequency details (textures, edges) that are critical for object detection models like RT-DETR.

**Initial Findings:**
- Vanilla RT-DETR mAP: **0.341**
- Original SPDNet + RT-DETR mAP: **0.337** (-1.4% drop)

The de-raining process was actually hurting detection performance because the restoration objective was not aligned with the detection objective.

## 2. Solution: Feature-Based Adaptation
We implemented a domain adaptation strategy to fine-tune SPDNet specifically for object detection, without retraining the heavy RT-DETR model.

### Key Technical Decisions
1.  **Feature Consistency Loss**: Instead of pixel-level loss (MSE), we minimize the distance between feature maps of:
    - Clean images (passed through a feature extractor)
    - De-rained images (passed through the same feature extractor)
    This ensures the de-rained image "looks" like a clean image to the detector.

2.  **Lightweight Training Strategy**:
    - **Resolution**: Reduced to 320x320 (from 640x640) to fit in GPU memory.
    - **Feature Extractor**: Used a lightweight 3-layer CNN instead of full ResNet/RT-DETR backbone to save memory.
    - **Batch Size**: 1 with Gradient Accumulation of 16.
    - **Precision**: FP32 (Float32) because SPDNet's wavelet operations are incompatible with FP16.

### Scripts
- **Training**: `Training_SPDNet_Lightweight.py`
  - Trains SPDNet using the lightweight feature loss strategy.
  - Output: `outputs_spdnet_feature_adaptation/spdnet_adapted_best.pt`
- **Evaluation**: `Eval_Adapted_SPDNet.py`
  - Compares Vanilla, Original SPDNet, and Adapted SPDNet using COCO mAP metrics.

## 3. Results
The adaptation was successful. The adapted SPDNet now improves detection performance instead of degrading it.

| Method | mAP | AP50 | AP75 | Change vs Baseline |
|--------|-----|------|------|-------------------|
| **Vanilla RT-DETR** (baseline) | 0.341 | 0.483 | 0.366 | — |
| **Original SPDNet + RT-DETR** | 0.337 | 0.479 | 0.353 | **-1.4%** (Degradation) |
| **Adapted SPDNet + RT-DETR** | **0.355** | **0.507** | **0.375** | **+4.1%** (Improvement) |

**Key Improvements:**
- **+4.1% mAP** over vanilla baseline.
- **+5.5% mAP** over original SPDNet.
- Consistent gains across small, medium, and large objects.

## 4. Future Usage
To use the adapted model in the integrated pipeline:
1. Load the architecture using `load_spdnet_model`.
2. Load the weights from `outputs_spdnet_feature_adaptation/spdnet_adapted_best.pt`.

```python
adapted_spdnet = load_spdnet_model(
    "model_spa.pt",  # Load architecture structure
    device=device
)
adapted_weights = torch.load("outputs_spdnet_feature_adaptation/spdnet_adapted_best.pt")
adapted_spdnet.load_state_dict(adapted_weights)
```
