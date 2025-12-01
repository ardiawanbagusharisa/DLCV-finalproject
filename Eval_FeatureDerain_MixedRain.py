#!/usr/bin/env python3
"""
Feature De-rain Evaluation on MixedRain Dataset

Compares models using COCO mAP metrics:
1. Vanilla RT-DETR (baseline)
2. Feature De-rain RT-DETR (ours)
3. Adapted SPDNet → RT-DETR (pixel-level)

Usage:
    python Eval_FeatureDerain_MixedRain.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import random
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json
import tempfile
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import cv2
cv2.setNumThreads(0)
import gc
import time

# =============================================================================
# Reproducibility Seed
# =============================================================================

SEED = 42

def set_seed(seed=SEED):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import pycocotools for mAP
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCOTOOLS = True
except ImportError:
    print("[ERROR] pycocotools required. Install with: pip install pycocotools")
    HAS_PYCOCOTOOLS = False

from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from utils.feature_derain import FeatureDerainRTDETR, create_feature_derain_rtdetr

# =============================================================================
# Configuration
# =============================================================================

MIXEDRAIN_DIR = "E:/Python/DLCV/mixedrain"
LABELS_FILE = os.path.join(MIXEDRAIN_DIR, "labels.txt")

# Model paths
FEATURE_DERAIN_CHECKPOINT = "./outputs_feature_derain/coco-rain_0.3-full/feature_derain_best.pt"
ORIGINAL_SPDNET_PATH = "./model_spa.pt"  # Original SPDNet weights
RTDETR_MODEL_NAME = "PekingU/rtdetr_r18vd"
NUM_LABELS = 80

# Feature De-rain settings
DERAIN_TYPE = "multiscale"  # Must match training

# SPDNet Configuration
SPDNET_N_FEATS = 32
SPDNET_N_RESBLOCKS = 3

# Settings
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.01  # Low threshold for mAP calculation

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Utilities
# =============================================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Dataset
# =============================================================================

class MixedRainDataset(torch.utils.data.Dataset):
    """MixedRain dataset with YOLO format labels"""
    
    def __init__(self, data_dir, labels_file, image_size=640):
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Parse labels
        self.annotations = defaultdict(list)
        self.image_ids = []
        
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        img_id = parts[0]
                        class_id = int(parts[1])
                        x_center = float(parts[2])
                        y_center = float(parts[3])
                        width = float(parts[4])
                        height = float(parts[5])
                        
                        self.annotations[img_id].append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
                        
                        if img_id not in self.image_ids:
                            self.image_ids.append(img_id)
        
        # Find image files
        self.image_files = {}
        for img_id in self.image_ids:
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = os.path.join(data_dir, f"{img_id}{ext}")
                if os.path.exists(img_path):
                    self.image_files[img_id] = img_path
                    break
        
        self.image_ids = [img_id for img_id in self.image_ids if img_id in self.image_files]
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
        print(f"[Dataset] Loaded {len(self.image_ids)} images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.image_files[img_id]
        
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size  # (W, H)
        
        image_tensor = self.transform(image)
        
        # Convert YOLO to COCO format [x, y, w, h]
        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        areas = []
        
        orig_w, orig_h = orig_size
        
        for ann in anns:
            x_center = ann['x_center']
            y_center = ann['y_center']
            w = ann['width']
            h = ann['height']
            
            # Convert to absolute COCO format [x, y, width, height]
            x = (x_center - w / 2) * orig_w
            y = (y_center - h / 2) * orig_h
            box_w = w * orig_w
            box_h = h * orig_h
            
            if box_w > 0 and box_h > 0:
                boxes.append([x, y, box_w, box_h])
                labels.append(ann['class_id'])
                areas.append(box_w * box_h)
        
        return {
            'image': image_tensor,
            'boxes': boxes,
            'labels': labels,
            'areas': areas,
            'image_id': idx,
            'orig_size': orig_size,
            'str_id': img_id
        }


def collate_fn(batch):
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'boxes': [item['boxes'] for item in batch],
        'labels': [item['labels'] for item in batch],
        'areas': [item['areas'] for item in batch],
        'image_ids': [item['image_id'] for item in batch],
        'orig_sizes': [item['orig_size'] for item in batch],
        'str_ids': [item['str_id'] for item in batch]
    }


# =============================================================================
# Model Loading
# =============================================================================

def load_vanilla_rtdetr():
    """Load vanilla RT-DETR model"""
    model = RTDetrForObjectDetection.from_pretrained(RTDETR_MODEL_NAME)
    processor = RTDetrImageProcessor.from_pretrained(RTDETR_MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    return model, processor


def load_feature_derain_model():
    """Load Feature De-rain RT-DETR"""
    rtdetr = RTDetrForObjectDetection.from_pretrained(RTDETR_MODEL_NAME)
    model = FeatureDerainRTDETR(
        rtdetr_model=rtdetr,
        derain_type=DERAIN_TYPE,
        freeze_backbone=False
    )
    
    checkpoint = torch.load(FEATURE_DERAIN_CHECKPOINT, map_location=DEVICE)
    
    if 'derain_module' in checkpoint:
        model.derain_module.load_state_dict(checkpoint['derain_module'])
    if 'rtdetr' in checkpoint:
        model.rtdetr.load_state_dict(checkpoint['rtdetr'])
    
    model = model.to(DEVICE)
    model.eval()
    return model


def load_spdnet():
    """Load Original SPDNet + RT-DETR pipeline"""
    from utils.spdnet_utils import load_spdnet_model
    
    spdnet = load_spdnet_model(
        ORIGINAL_SPDNET_PATH,
        device=DEVICE,
        n_feats=SPDNET_N_FEATS,
        n_resblocks=SPDNET_N_RESBLOCKS
    )
    spdnet.eval()
    return spdnet


def apply_spdnet(spdnet, images):
    """Apply SPDNet de-raining"""
    spdnet_input = images * 255.0
    with torch.no_grad():
        output = spdnet(spdnet_input)
        if isinstance(output, tuple):
            output = output[0]
    return torch.clamp(output / 255.0, 0, 1)


# =============================================================================
# COCO mAP Evaluation
# =============================================================================

def create_coco_gt(dataset):
    """Create COCO ground truth object"""
    gt_dict = {
        'info': {'description': 'MixedRain Dataset'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [{'id': i, 'name': f'class_{i}'} for i in range(80)]
    }
    
    ann_id = 1
    for idx in range(len(dataset)):
        sample = dataset[idx]
        img_id = sample['image_id']
        orig_w, orig_h = sample['orig_size']
        
        gt_dict['images'].append({
            'id': img_id,
            'width': orig_w,
            'height': orig_h,
            'file_name': f'{sample["str_id"]}.jpg'
        })
        
        for box, label, area in zip(sample['boxes'], sample['labels'], sample['areas']):
            gt_dict['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': label,
                'bbox': [float(b) for b in box],  # [x, y, w, h]
                'area': float(area),
                'iscrowd': 0
            })
            ann_id += 1
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(gt_dict, f)
        temp_path = f.name
    
    coco_gt = COCO(temp_path)
    os.unlink(temp_path)
    
    return coco_gt


def evaluate_with_coco(predictions, coco_gt):
    """Evaluate predictions using COCO metrics"""
    if len(predictions) == 0:
        return {'mAP': 0.0, 'AP50': 0.0, 'AP75': 0.0, 'AP_small': 0.0, 'AP_medium': 0.0, 'AP_large': 0.0}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        temp_path = f.name
    
    coco_dt = coco_gt.loadRes(temp_path)
    os.unlink(temp_path)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return {
        'mAP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5]
    }


# =============================================================================
# Prediction Generation
# =============================================================================

def generate_vanilla_predictions(model, processor, dataloader, device, desc="Vanilla RT-DETR"):
    """Generate predictions with vanilla RT-DETR"""
    model.eval()
    predictions = []
    
    total_time = 0.0
    total_count = 0
    
    # Warmup
    warmup_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    for _ in range(3):
        with torch.no_grad():
            _ = model(pixel_values=warmup_img)
    torch.cuda.synchronize()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  {desc}"):
            images = batch['images'].to(device)
            image_ids = batch['image_ids']
            orig_sizes = batch['orig_sizes']
            
            batch_size = images.shape[0]
            total_count += batch_size
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = model(pixel_values=images)
            
            torch.cuda.synchronize()
            total_time += time.perf_counter() - start_time
            
            # Post-process
            target_sizes = torch.tensor([(h, w) for w, h in orig_sizes]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
            )
            
            for i, result in enumerate(results):
                boxes = result['boxes'].detach().cpu().numpy()
                scores = result['scores'].detach().cpu().numpy()
                labels = result['labels'].detach().cpu().numpy()
                img_id = image_ids[i]
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    predictions.append({
                        'image_id': int(img_id),
                        'category_id': int(label),
                        'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        'score': float(score)
                    })
    
    avg_time_ms = (total_time / total_count) * 1000
    fps = total_count / total_time
    
    print(f"  Speed: {avg_time_ms:.1f} ms/img ({fps:.1f} FPS)")
    
    return predictions, {'avg_time_ms': avg_time_ms, 'fps': fps, 'total_images': total_count}


def generate_feature_derain_predictions(model, processor, dataloader, device, desc="Feature De-rain"):
    """Generate predictions with Feature De-rain RT-DETR"""
    model.eval()
    predictions = []
    
    total_time = 0.0
    total_count = 0
    
    # Warmup
    warmup_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    for _ in range(3):
        with torch.no_grad():
            _ = model(warmup_img)
    torch.cuda.synchronize()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  {desc}"):
            images = batch['images'].to(device)
            image_ids = batch['image_ids']
            orig_sizes = batch['orig_sizes']
            
            batch_size = images.shape[0]
            total_count += batch_size
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = model(images)
            
            torch.cuda.synchronize()
            total_time += time.perf_counter() - start_time
            
            # Post-process
            target_sizes = torch.tensor([(h, w) for w, h in orig_sizes]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
            )
            
            for i, result in enumerate(results):
                boxes = result['boxes'].detach().cpu().numpy()
                scores = result['scores'].detach().cpu().numpy()
                labels = result['labels'].detach().cpu().numpy()
                img_id = image_ids[i]
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    predictions.append({
                        'image_id': int(img_id),
                        'category_id': int(label),
                        'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        'score': float(score)
                    })
    
    avg_time_ms = (total_time / total_count) * 1000
    fps = total_count / total_time
    
    print(f"  Speed: {avg_time_ms:.1f} ms/img ({fps:.1f} FPS)")
    
    return predictions, {'avg_time_ms': avg_time_ms, 'fps': fps, 'total_images': total_count}


def generate_spdnet_predictions(model, processor, spdnet, dataloader, device, desc="SPDNet + RT-DETR"):
    """Generate predictions with SPDNet + RT-DETR pipeline"""
    model.eval()
    spdnet.eval()
    predictions = []
    
    total_time = 0.0
    total_count = 0
    
    # Warmup
    warmup_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    for _ in range(3):
        with torch.no_grad():
            clean = apply_spdnet(spdnet, warmup_img)
            _ = model(pixel_values=clean)
    torch.cuda.synchronize()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  {desc}"):
            images = batch['images'].to(device)
            image_ids = batch['image_ids']
            orig_sizes = batch['orig_sizes']
            
            batch_size = images.shape[0]
            total_count += batch_size
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Apply SPDNet
            clean_images = apply_spdnet(spdnet, images)
            
            # Run detection
            outputs = model(pixel_values=clean_images)
            
            torch.cuda.synchronize()
            total_time += time.perf_counter() - start_time
            
            # Post-process
            target_sizes = torch.tensor([(h, w) for w, h in orig_sizes]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
            )
            
            for i, result in enumerate(results):
                boxes = result['boxes'].detach().cpu().numpy()
                scores = result['scores'].detach().cpu().numpy()
                labels = result['labels'].detach().cpu().numpy()
                img_id = image_ids[i]
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    predictions.append({
                        'image_id': int(img_id),
                        'category_id': int(label),
                        'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        'score': float(score)
                    })
    
    avg_time_ms = (total_time / total_count) * 1000
    fps = total_count / total_time
    
    print(f"  Speed: {avg_time_ms:.1f} ms/img ({fps:.1f} FPS)")
    
    return predictions, {'avg_time_ms': avg_time_ms, 'fps': fps, 'total_images': total_count}


# =============================================================================
# Main
# =============================================================================

def main():
    if not HAS_PYCOCOTOOLS:
        print("ERROR: pycocotools required for mAP evaluation")
        return
    
    print("\n" + "="*70)
    print("Feature De-rain Evaluation on MixedRain Dataset")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {MIXEDRAIN_DIR}")
    print(f"Feature De-rain checkpoint: {FEATURE_DERAIN_CHECKPOINT}")
    
    # Check paths
    if not os.path.exists(MIXEDRAIN_DIR):
        print(f"[ERROR] Dataset not found: {MIXEDRAIN_DIR}")
        return
    
    if not os.path.exists(FEATURE_DERAIN_CHECKPOINT):
        print(f"[ERROR] Feature De-rain checkpoint not found: {FEATURE_DERAIN_CHECKPOINT}")
        return
    
    # Load dataset
    print("\n[1/4] Loading Dataset...")
    dataset = MixedRainDataset(MIXEDRAIN_DIR, LABELS_FILE, IMAGE_SIZE)
    
    if len(dataset) == 0:
        print("[ERROR] No images found in dataset")
        return
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create COCO GT
    print("\n[2/4] Creating COCO Ground Truth...")
    coco_gt = create_coco_gt(dataset)
    print(f"  GT annotations: {len(coco_gt.anns)}")
    print(f"  GT images: {len(coco_gt.imgs)}")
    
    # Load processor (shared across models)
    processor = RTDetrImageProcessor.from_pretrained(RTDETR_MODEL_NAME)
    
    results = {}
    timing = {}
    
    # =========================================================================
    # Evaluate Vanilla RT-DETR
    # =========================================================================
    print("\n" + "="*70)
    print("[3/4] Evaluating Models...")
    print("="*70)
    
    print("\n--- 1. Vanilla RT-DETR (Baseline) ---")
    vanilla_model, _ = load_vanilla_rtdetr()
    preds_vanilla, timing['vanilla'] = generate_vanilla_predictions(
        vanilla_model, processor, dataloader, DEVICE
    )
    results['vanilla'] = evaluate_with_coco(preds_vanilla, coco_gt)
    
    del vanilla_model
    clear_memory()
    
    # =========================================================================
    # Evaluate Feature De-rain RT-DETR
    # =========================================================================
    print("\n--- 2. Feature De-rain RT-DETR (Ours) ---")
    feature_model = load_feature_derain_model()
    preds_feature, timing['feature_derain'] = generate_feature_derain_predictions(
        feature_model, processor, dataloader, DEVICE
    )
    results['feature_derain'] = evaluate_with_coco(preds_feature, coco_gt)
    
    del feature_model
    clear_memory()
    
    # =========================================================================
    # Evaluate Original SPDNet + RT-DETR
    # =========================================================================
    if os.path.exists(ORIGINAL_SPDNET_PATH):
        print("\n--- 3. Original SPDNet + RT-DETR ---")
        rtdetr_model, _ = load_vanilla_rtdetr()
        spdnet = load_spdnet()
        preds_spdnet, timing['spdnet'] = generate_spdnet_predictions(
            rtdetr_model, processor, spdnet, dataloader, DEVICE
        )
        results['spdnet'] = evaluate_with_coco(preds_spdnet, coco_gt)
        
        del rtdetr_model, spdnet
        clear_memory()
    
    # =========================================================================
    # Print Results
    # =========================================================================
    print("\n" + "="*70)
    print("[4/4] RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<35} {'mAP':>8} {'AP50':>8} {'AP75':>8} {'ms/img':>8} {'FPS':>8}")
    print("-"*75)
    
    model_names = [
        ("Vanilla RT-DETR", "vanilla"),
        ("Feature De-rain RT-DETR", "feature_derain"),
    ]
    
    if 'spdnet' in results:
        model_names.append(("Original SPDNet + RT-DETR", "spdnet"))
    
    for name, key in model_names:
        m = results[key]
        t = timing[key]
        print(f"{name:<35} {m['mAP']*100:>7.2f}% {m['AP50']*100:>7.2f}% {m['AP75']*100:>7.2f}% {t['avg_time_ms']:>7.1f} {t['fps']:>7.1f}")
    
    # Comparison
    print("\n" + "-"*75)
    vanilla_map = results['vanilla']['mAP']
    feature_map = results['feature_derain']['mAP']
    diff = (feature_map - vanilla_map) * 100
    sign = "+" if diff >= 0 else ""
    print(f"Feature De-rain vs Vanilla: {sign}{diff:.2f}% mAP")
    
    if 'spdnet' in results:
        spdnet_map = results['spdnet']['mAP']
        diff_spdnet = (feature_map - spdnet_map) * 100
        sign_spdnet = "+" if diff_spdnet >= 0 else ""
        print(f"Feature De-rain vs SPDNet: {sign_spdnet}{diff_spdnet:.2f}% mAP")
        
        # Speed comparison
        feature_fps = timing['feature_derain']['fps']
        spdnet_fps = timing['spdnet']['fps']
        print(f"Feature De-rain vs SPDNet: {feature_fps/spdnet_fps:.1f}x faster")
    
    # Save results
    output_path = os.path.join(MIXEDRAIN_DIR, "feature_derain_results.json")
    full_results = {
        'metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'timing': timing,
        'config': {
            'derain_type': DERAIN_TYPE,
            'checkpoint': FEATURE_DERAIN_CHECKPOINT,
            'dataset': MIXEDRAIN_DIR
        }
    }
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
