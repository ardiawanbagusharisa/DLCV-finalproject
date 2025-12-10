# Feature-Level De-raining for Robust Object Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)

A novel approach to rain-robust object detection that operates at the **feature level** rather than pixel level, achieving **10x faster inference** while maintaining detection accuracy.

## 📹 Demo

https://github.com/user-attachments/assets/f706c62e-ca4b-4d12-8d0a-3ffe0b355ce9

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

<img width="2376" height="1256" alt="Research Roadmap" src="https://github.com/user-attachments/assets/1806371b-a7da-4c8d-bfbf-9921e592be31" />


### Key Components

1. **Rain Mask Estimation**: Learns to identify rain-affected spatial regions
2. **CBAM Attention**: Channel + Spatial attention for selective feature refinement
3. **Residual Design**: Preserves clean features while suppressing rain artifacts
4. **Multi-Scale Processing**: Handles rain at different scales (droplets → streaks)

## 📊 Results

### Real Rain Dataset

| Model                            | mAP              | AP50             | AP75             | Latency          | Notes       |
| -------------------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ----------- |
| Vanilla RT-DETR                  | 40.50%           | 51.90%           | 45.84%           | 15.3ms           | Baseline    |
| SPDNet + RT-DETR                 | 42.21%           | 53.73%           | 47.18%           | 244.2ms          | Pixel-level |
| **Feature De-rain (Ours)** | **41.27%** | **52.20%** | **44.09%** | **21.0ms** | Balanced    |

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

#### Single Image Inference

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

| Argument            | Description                         |
| ------------------- | ----------------------------------- |
| `--image, -i`     | Path to input image (required)      |
| `--threshold, -t` | Confidence threshold (default: 0.5) |
| `--output, -o`    | Output directory for results        |
| `--save, -s`      | Save the output image               |
| `--no-show`       | Don't display the result image      |
| `--compare, -c`   | Compare with vanilla RT-DETR        |
| `--checkpoint`    | Path to model checkpoint            |
| `--vanilla`       | Use vanilla RT-DETR instead         |

#### Video Inference

Process videos with real-time FPS display and object detection:

```bash
# Basic video inference (Feature De-rain model, shows progress)
python video_inference.py --video path/to/video.mp4

# Use vanilla RT-DETR model
python video_inference.py --video path/to/video.mp4 --model vanilla

# Process specific time range (e.g., first 2 minutes)
python video_inference.py --video path/to/video.mp4 --start 0 --end 120

# Process from minute 5 to minute 7
python video_inference.py --video path/to/video.mp4 --start 300 --end 420

# Custom output path and threshold
python video_inference.py --video path/to/video.mp4 --output my_result.mp4 --threshold 0.6
```
