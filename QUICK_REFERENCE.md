# 📋 Quick Reference: Integrated Model Commands

## Test Model (Recommended First Step)
```bash
python test_integrated_model.py
```
**Purpose**: Verify model loads and works correctly  
**Time**: ~2 minutes  
**Expected**: All 5 tests should pass

---

## Train Integrated Model
```bash
python Training_integrated.py
```
**Purpose**: Train SPDNet+RT-DETR end-to-end  
**Time**: ~8-10 hours (12 epochs on RTX 3080)  
**Output**: `./outputs_integrated/best_integrated/`

**Quick Config** (edit at top of `Training_integrated.py`):
```python
NUM_EPOCHS = 12            # Total training epochs
BATCH_SIZE = 8             # Adjust based on GPU memory
LEARNING_RATE = 1e-5       # Default works well
PHASE1_EPOCHS = 2          # Detection head only
PHASE2_EPOCHS = 8          # + SPDNet
PHASE3_EPOCHS = 12         # Everything
```

---

## Evaluate Model
```bash
python Eval_integrated.py
```
**Purpose**: Compare integrated vs vanilla RT-DETR  
**Time**: ~20 minutes  
**Output**: PR curves and COCO metrics

---

## Monitor Training (TensorBoard)
```bash
tensorboard --logdir ./outputs_integrated/runs/
```
**Then**: Open http://localhost:6006 in browser

---

## Common Configurations

### Low GPU Memory (8GB)
```python
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
DATALOADER_WORKERS = 4
```

### High GPU Memory (24GB+)
```python
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
DATALOADER_WORKERS = 16
```

### Fast Debug Run
```python
PERCENT_DATASET = 10       # Use 10% of data
NUM_EPOCHS = 3
PHASE1_EPOCHS = 1
PHASE2_EPOCHS = 2
PHASE3_EPOCHS = 3
```

---

## File Locations

### Input Files (Required)
```
E:/Python/DLCV/Project/model_spa.pt              ← Pretrained SPDNet
E:/Python/DLCV/Project/dataset/coco/             ← Clean COCO dataset
E:/Python/DLCV/Project/dataset/coco_rain/        ← Rainy COCO dataset
E:/Python/DLCV/SPDNet/                           ← SPDNet source code
```

### Output Files (Generated)
```
./outputs_integrated/best_integrated/            ← Best model checkpoint
./outputs_integrated/final_integrated/           ← Final model checkpoint
./outputs_integrated/training_curves.png         ← Training/validation curves
./outputs_integrated/runs/                       ← TensorBoard logs
./pr_curve_comparison_integrated.png             ← Evaluation results
```

---

## Quick Inference Example

```python
from utils.integrated_model import load_integrated_model
from PIL import Image
import torch

# Load model
model, processor = load_integrated_model(
    spdnet_path="./model_spa.pt",
    rtdetr_name="PekingU/rtdetr_r18vd"
)

# Run inference
image = Image.open("rainy_image.jpg")
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to('cuda') for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

# Post-process
target_sizes = torch.tensor([image.size[::-1]]).to('cuda')
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.3
)[0]

print(f"Found {len(results['boxes'])} objects")
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 4 |
| `SPDNet requires CUDA` | Ensure GPU available |
| `Model not found` | Check `SPDNET_MODEL_PATH` |
| `Slow training` | Normal! Phase 3 is slowest |
| `Low mAP` | Try increasing `NUM_EPOCHS` to 20 |

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| Training time (12 epochs) | ~8-10 hours |
| Inference time | ~80ms per image |
| mAP on rainy data | ~0.39 |
| Speedup vs two-stage | 2.1x |

---

## Phase Breakdown

```
Epoch 1-2   (Phase 1):  Train detection head only
            └─ SPDNet frozen, RT-DETR backbone frozen
            
Epoch 3-8   (Phase 2):  Train SPDNet + detection head
            └─ RT-DETR backbone frozen
            
Epoch 9-12  (Phase 3):  Fine-tune everything
            └─ All parameters trainable
```

---

## Key Hyperparameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `LEARNING_RATE` | 1e-5 | Lower = more stable |
| `BATCH_SIZE` | 8 | Adjust for GPU |
| `GRADIENT_ACCUMULATION_STEPS` | 2 | Effective batch = 16 |
| `COCO_RATIO` | 0.9 | 90% clean images |
| `RAIN_RATIO` | 0.1 | 10% rainy images |
| `FP16` | True | Mixed precision |

---

## Useful Commands

```bash
# Check GPU memory
nvidia-smi

# Count trainable parameters
python -c "from utils.integrated_model import load_integrated_model; \
           m, _ = load_integrated_model('./model_spa.pt', 'PekingU/rtdetr_r18vd'); \
           print(f'{sum(p.numel() for p in m.parameters() if p.requires_grad):,}')"

# View training logs
cat ./outputs_integrated/trainer_state.json

# Kill training (if stuck)
# Ctrl+C, then:
pkill -f Training_integrated.py
```

---

## Documentation

- **Detailed Guide**: `INTEGRATED_MODEL.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`
- **Project Goals**: `Project_goal.md`
- **Copilot Instructions**: `.github/copilot-instructions.md`

---

**Remember**: Start with `test_integrated_model.py` to verify setup before training! ✅
