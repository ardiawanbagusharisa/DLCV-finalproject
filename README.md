# Feature-Level De-raining for Robust Object Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)

A novel approach to rain-robust object detection that operates at the **feature level** rather than pixel level, achieving **10x faster inference** while maintaining detection accuracy.

## 📋 Overview

### Problem
Object detection models (like RT-DETR) suffer significant accuracy drops in rainy conditions. Traditional solutions apply pixel-level de-raining (SPDNet) before detection, but this adds **~180ms latency** per image.

### Our Solution
**Feature-Level De-raining**: Instead of reconstructing clean images, we suppress rain-related features directly in the detector's backbone output. This achieves:

- ⚡ **10x faster** than pixel-level de-raining (3-5ms vs 180ms)
- 🎯 **+0.78% mAP** over vanilla RT-DETR on mixed weather
- 📦 **12.76M parameters** for the de-rain module
- 🔗 **End-to-end training** with detection loss

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rainy Image (640×640×3)                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RT-DETR Backbone (ResNet18-vd)                  │
│    Stage 2 → (128, 80, 80)                                      │
│    Stage 3 → (256, 40, 40)                                      │
│    Stage 4 → (512, 20, 20)                                      │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Feature De-rain Module (Our Contribution)           │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│    │ DerainBlock │  │ DerainBlock │  │ DerainBlock │           │
│    │   128 ch    │  │   256 ch    │  │   512 ch    │           │
│    └─────────────┘  └─────────────┘  └─────────────┘           │
│    (Rain mask estimation + CBAM attention + Residual refinement)│
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RT-DETR Encoder + Decoder                     │
│                    100 detection queries                         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Rain Mask Estimation**: Learns to identify rain-affected spatial regions
2. **CBAM Attention**: Channel + Spatial attention for selective feature refinement  
3. **Residual Design**: Preserves clean features while suppressing rain artifacts
4. **Multi-Scale Processing**: Handles rain at different scales (droplets → streaks)

## 📊 Results

### COCO-Rain Evaluation

| Model | mAP | AP50 | Latency |
|-------|-----|------|---------|
| Vanilla RT-DETR | 34.1% | 48.3% | 25ms |
| SPDNet + RT-DETR | 33.7% | 48.3% | 205ms |
| **Feature De-rain (Ours)** | **34.9%** | **49.1%** | **28ms** |

### MixedRain Dataset

| Model | mAP | AP50 | Notes |
|-------|-----|------|-------|
| Vanilla RT-DETR | 25.69% | 44.56% | Baseline |
| SPDNet + RT-DETR | 24.03% | 42.94% | Pixel-level |
| **Feature De-rain (Ours)** | **26.47%** | **45.73%** | +3.0% over vanilla |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/machiningman/Project-DLCV.git
cd Project-DLCV

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM for inference, 16GB+ for training)
- PyTorch 2.0+

### Training

```bash
python Training_FeatureDerain.py
```

**Training Configuration** (in script):
```python
PERCENT_DATASET = 100   # Use 100% for full training
COCO_RATIO = 0.3        # 30% clean images
RAIN_RATIO = 0.7        # 70% rainy images
PHASE1_EPOCHS = 5       # De-rain module only
PHASE2_EPOCHS = 10      # Joint fine-tuning
BATCH_SIZE = 16         # Adjust for your GPU
```

**Two-Phase Training Strategy:**
1. **Phase 1**: Train de-rain module only (RT-DETR frozen)
2. **Phase 2**: Joint fine-tuning with differential learning rates

### Evaluation

```bash
# Evaluate on COCO-Rain
python Eval_FeatureDerain.py

# Evaluate on MixedRain (compares all models)
python Eval_FeatureDerain_MixedRain.py
```

### Inference

#### Command Line Interface

Use the inference script for quick testing on single images:

```bash
# Basic inference (displays result)
python inference.py --image path/to/test.jpg

# With custom confidence threshold
python inference.py --image path/to/test.jpg --threshold 0.3

# Save result without displaying
python inference.py --image path/to/test.jpg --output results/ --save --no-show

# Compare Feature De-rain vs Vanilla RT-DETR
python inference.py --image path/to/test.jpg --compare --output comparison/

# Use vanilla RT-DETR only (for comparison)
python inference.py --image path/to/test.jpg --vanilla
```

**Inference Script Options:**

| Argument | Description |
|----------|-------------|
| `--image, -i` | Path to input image (required) |
| `--threshold, -t` | Confidence threshold (default: 0.5) |
| `--output, -o` | Output directory for results |
| `--save, -s` | Save the output image |
| `--no-show` | Don't display the result image |
| `--compare, -c` | Compare with vanilla RT-DETR |
| `--checkpoint` | Path to model checkpoint |
| `--vanilla` | Use vanilla RT-DETR instead |

#### Python API

```python
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from utils.feature_derain import FeatureDerainRTDETR, create_feature_derain_rtdetr

# Load model
rtdetr = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd")
model = FeatureDerainRTDETR(rtdetr, derain_type="multiscale")

# Load trained weights
checkpoint = torch.load("outputs_feature_derain/feature_derain_best.pt")
model.derain_module.load_state_dict(checkpoint['derain_module'])
model.eval().cuda()

# Inference
processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r18vd")
image = Image.open("rainy_image.jpg")
inputs = processor(images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
    
results = processor.post_process_object_detection(
    outputs, target_sizes=[(image.height, image.width)], threshold=0.5
)
```

## 📁 Project Structure

```
.
├── Training_FeatureDerain.py       # Main training script
├── Eval_FeatureDerain.py           # Evaluation on COCO-Rain
├── Eval_FeatureDerain_MixedRain.py # Evaluation on MixedRain
├── inference.py                    # Single image inference script
├── model_spa.pt                    # SPDNet weights (for comparison)
├── requirements.txt                # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── data_utils.py               # Dataset loading & augmentation
│   ├── eval_utils.py               # COCO evaluation metrics
│   ├── feature_derain.py           # Feature de-rain module (core)
│   └── spdnet_utils.py             # SPDNet utilities (baseline)
├── outputs_feature_derain/         # Training outputs & checkpoints
└── docs/
    ├── FEATURE_DERAIN_ARCHITECTURE.md  # Detailed architecture docs
    └── Project_goal.md                  # Project motivation
```

## 🔧 Configuration

### Dataset Paths

Update paths in scripts:
```python
COCO_DIR = "path/to/coco"           # Clean COCO dataset
COCO_RAIN_DIR = "path/to/coco_rain" # Rainy COCO dataset
```

### SPDNet Setup (for comparison)

SPDNet requires external setup:
1. Clone SPDNet repository to `E:\Python\DLCV\SPDNet`
2. Update `SPDNET_SRC_PATH` in `utils/spdnet_utils.py`

## 📈 Training Tips

1. **OOM Issues**: Reduce `BATCH_SIZE` or enable `GRADIENT_ACCUMULATION_STEPS`
2. **Fast Testing**: Set `PERCENT_DATASET = 5` for quick validation
3. **Better Results**: Train with `PERCENT_DATASET = 100` for full performance
4. **Mixed Precision**: `USE_AMP = True` (enabled by default)

## 🔬 Technical Details

### Why Feature-Level Works

1. **Rain in Feature Space**: Rain activates specific CNN channels (edge/gradient detectors, vertical filters). CBAM attention learns to suppress these.

2. **Efficiency**: Feature tensors are 0.17x-0.67x the size of input images, making processing inherently faster.

3. **Task Alignment**: Training with detection loss ensures features are optimized for detection, not visual quality.

### Model Variants

| Variant | Parameters | Speed | Use Case |
|---------|------------|-------|----------|
| `lightweight` | ~50K | Fastest | Resource-constrained |
| `multiscale` | ~12.7M | Balanced | Best accuracy (default) |

## 📚 Documentation

- [Architecture Design](docs/FEATURE_DERAIN_ARCHITECTURE.md) - Detailed technical documentation
- [Project Goals](docs/Project_goal.md) - Problem statement and motivation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [RT-DETR](https://github.com/PaddlePaddle/PaddleDetection) - Real-Time DEtection TRansformer
- [SPDNet](https://github.com/panhongrui/SPDNet) - Structure-Preserving Deraining Network
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) - Model hosting and utilities

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@misc{feature-derain-rtdetr,
  title={Feature-Level De-raining for Robust Object Detection},
  author={Andi Alfian Kartika Aji},
  year={2025},
  url={https://github.com/machiningman/Project-DLCV}
}
```
