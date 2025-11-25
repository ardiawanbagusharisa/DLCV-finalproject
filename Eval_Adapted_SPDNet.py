#!/usr/bin/env python3
"""
Evaluation Script for Adapted SPDNet with COCO mAP

Compares three approaches on rainy COCO validation set:
1. Vanilla RT-DETR (baseline - no de-raining)
2. Original SPDNet + RT-DETR (pretrained de-raining)
3. Adapted SPDNet + RT-DETR (your trained model)

Computes:
- COCO mAP (AP, AP50, AP75, AP_small, AP_medium, AP_large)
- Simple metrics (Precision, Recall, F1)

Usage:
    python Eval_Adapted_SPDNet.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.spdnet_utils import load_spdnet_model
from utils.model_utils import load_model_and_processor
from torch.utils.data import DataLoader
import cv2
cv2.setNumThreads(0)
from PIL import Image
import torchvision.transforms as T

# Import pycocotools for mAP
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCOTOOLS = True
except ImportError:
    print("[WARNING] pycocotools not installed. Install with: pip install pycocotools")
    HAS_PYCOCOTOOLS = False

# =============================================================================
# Configuration
# =============================================================================

# Dataset paths
COCO_RAIN_DIR = "E:/Python/DLCV/dataset/coco_rain"
VAL_SPLIT = "val2017"
ANNOTATIONS_PATH = f"{COCO_RAIN_DIR}/annotations/instances_val2017.json"

# Model paths
ORIGINAL_SPDNET_PATH = "E:/Python/DLCV/Project DLCV/model_spa.pt"
ADAPTED_SPDNET_PATH = "./outputs_spdnet_feature_adaptation/spdnet_adapted_best.pt"
RTDETR_MODEL_NAME = "PekingU/rtdetr_r18vd"
NUM_LABELS = 80

# Evaluation settings
PERCENT_DATASET = 10   # Use 10% for faster evaluation (~500 images)
IMAGE_SIZE = 640       # Full resolution for evaluation (different from training!)
BATCH_SIZE = 1         # Single image for fair comparison
CONFIDENCE_THRESHOLD = 0.5  # For counting detections
IOU_THRESHOLD = 0.5    # For matching predictions to ground truth

# SPDNet Configuration
SPDNET_N_FEATS = 32
SPDNET_N_RESBLOCKS = 3

# COCO 80 classes mapping (contiguous 0-79 to COCO 1-90 with gaps)
COCO_CATEGORIES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def model_id_to_coco_id(model_id):
    """Convert model's contiguous ID (0-79) to COCO category ID (1-90 with gaps)"""
    if 0 <= model_id < len(COCO_CATEGORIES):
        return COCO_CATEGORIES[model_id]
    return model_id


# =============================================================================
# Dataset for Evaluation
# =============================================================================

class RainyCocoDataset(torch.utils.data.Dataset):
    """Dataset for loading rainy COCO images with annotations"""
    
    def __init__(self, rain_dir, split="val2017", annotations_path=None, 
                 image_size=640, percent_dataset=100):
        import json
        
        self.img_dir = os.path.join(rain_dir, split)
        self.image_size = image_size
        
        # Load annotations
        if annotations_path and os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                coco_data = json.load(f)
            
            # Create image_id to annotations mapping
            self.img_to_anns = defaultdict(list)
            for ann in coco_data['annotations']:
                self.img_to_anns[ann['image_id']].append(ann)
            
            # Get image info
            self.images = {img['id']: img for img in coco_data['images']}
            self.image_ids = list(self.images.keys())
        else:
            # No annotations, just list images
            image_files = [f for f in os.listdir(self.img_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.images = {i: {'file_name': f} for i, f in enumerate(image_files)}
            self.image_ids = list(self.images.keys())
            self.img_to_anns = {}
        
        # Subsample
        np.random.seed(42)
        num_samples = max(1, int(len(self.image_ids) * percent_dataset / 100))
        indices = np.random.choice(len(self.image_ids), num_samples, replace=False)
        self.image_ids = [self.image_ids[i] for i in sorted(indices)]
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
        print(f"Loaded {len(self.image_ids)} images for evaluation")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size  # (W, H)
        
        # Transform
        image_tensor = self.transform(image)
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        
        for ann in anns:
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                # Convert to [x1, y1, x2, y2] and scale to image_size
                scale_x = self.image_size / orig_size[0]
                scale_y = self.image_size / orig_size[1]
                boxes.append([
                    x * scale_x,
                    y * scale_y,
                    (x + w) * scale_x,
                    (y + h) * scale_y
                ])
                labels.append(ann['category_id'])
        
        return {
            'image': image_tensor,
            'boxes': torch.tensor(boxes) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels) if labels else torch.zeros((0,), dtype=torch.long),
            'image_id': img_id,
            'orig_size': orig_size
        }


def collate_fn(batch):
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'boxes': [item['boxes'] for item in batch],
        'labels': [item['labels'] for item in batch],
        'image_ids': [item['image_id'] for item in batch],
        'orig_sizes': [item['orig_size'] for item in batch]
    }


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area + 1e-6
    
    return inter_area / union_area


def generate_coco_predictions(model, processor, dataloader, device, spdnet=None, desc="Evaluating"):
    """
    Generate predictions in COCO format for mAP evaluation.
    
    Returns:
        predictions: List of prediction dicts in COCO format
        ground_truth_by_image: Dict mapping image_id to list of GT annotations
    """
    model.eval()
    if spdnet is not None:
        spdnet.eval()
    
    predictions = []
    ground_truth_by_image = {}
    pred_id = 1
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images = batch['images'].to(device)
            gt_boxes_list = batch['boxes']
            gt_labels_list = batch['labels']
            image_ids = batch['image_ids']
            orig_sizes = batch['orig_sizes']
            
            # Apply de-raining if SPDNet provided
            if spdnet is not None:
                spdnet_input = images * 255.0
                derain_output = spdnet(spdnet_input)
                if isinstance(derain_output, tuple):
                    images = derain_output[0]
                else:
                    images = derain_output
                images = torch.clamp(images / 255.0, 0, 1)
            
            # Run detection
            outputs = model(pixel_values=images)
            
            # Post-process
            target_sizes = torch.tensor([[IMAGE_SIZE, IMAGE_SIZE]] * images.shape[0]).to(device)
            results = processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes,
                threshold=0.001  # Very low threshold to capture all detections
            )
            
            for i, (result, gt_boxes, gt_labels, img_id, orig_size) in enumerate(
                zip(results, gt_boxes_list, gt_labels_list, image_ids, orig_sizes)
            ):
                pred_boxes = result['boxes'].cpu().numpy()
                pred_scores = result['scores'].cpu().numpy()
                pred_labels = result['labels'].cpu().numpy()
                
                # Scale factors to convert back to original size
                scale_x = orig_size[0] / IMAGE_SIZE
                scale_y = orig_size[1] / IMAGE_SIZE
                
                # Add predictions in COCO format
                for j in range(len(pred_boxes)):
                    x1, y1, x2, y2 = pred_boxes[j]
                    # Scale back and convert to xywh
                    x1_orig = x1 * scale_x
                    y1_orig = y1 * scale_y
                    w_orig = (x2 - x1) * scale_x
                    h_orig = (y2 - y1) * scale_y
                    
                    # Convert model ID (0-79) to COCO ID (1-90 with gaps)
                    coco_cat_id = model_id_to_coco_id(int(pred_labels[j]))
                    
                    predictions.append({
                        'id': pred_id,
                        'image_id': int(img_id),
                        'category_id': coco_cat_id,
                        'bbox': [float(x1_orig), float(y1_orig), float(w_orig), float(h_orig)],
                        'score': float(pred_scores[j]),
                        'area': float(w_orig * h_orig)
                    })
                    pred_id += 1
                
                # Store ground truth
                gt_anns = []
                for k in range(len(gt_boxes)):
                    x1, y1, x2, y2 = gt_boxes[k].numpy()
                    x1_orig = x1 * scale_x
                    y1_orig = y1 * scale_y
                    w_orig = (x2 - x1) * scale_x
                    h_orig = (y2 - y1) * scale_y
                    
                    gt_anns.append({
                        'bbox': [float(x1_orig), float(y1_orig), float(w_orig), float(h_orig)],
                        'category_id': int(gt_labels[k]),
                        'area': float(w_orig * h_orig)
                    })
                ground_truth_by_image[int(img_id)] = gt_anns
    
    return predictions, ground_truth_by_image


def compute_coco_map(predictions, ground_truth_by_image, annotations_path):
    """
    Compute COCO mAP using pycocotools.
    
    Returns:
        Dictionary with AP, AP50, AP75, AP_small, AP_medium, AP_large
    """
    if not HAS_PYCOCOTOOLS:
        print("[WARNING] pycocotools not available, skipping mAP calculation")
        return None
    
    if not predictions:
        return {
            'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
            'AP_small': 0.0, 'AP_medium': 0.0, 'AP_large': 0.0
        }
    
    # Load COCO GT
    coco_gt = COCO(annotations_path)
    
    # Filter GT to only include images we evaluated
    evaluated_img_ids = list(ground_truth_by_image.keys())
    
    # Create COCO-format results
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        pred_file = f.name
    
    try:
        # Load predictions
        coco_dt = coco_gt.loadRes(pred_file)
        
        # Run evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = evaluated_img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        results = {
            'AP': coco_eval.stats[0],      # AP @ IoU=0.50:0.95
            'AP50': coco_eval.stats[1],    # AP @ IoU=0.50
            'AP75': coco_eval.stats[2],    # AP @ IoU=0.75
            'AP_small': coco_eval.stats[3],
            'AP_medium': coco_eval.stats[4],
            'AP_large': coco_eval.stats[5],
            'AR_1': coco_eval.stats[6],    # AR given 1 detection per image
            'AR_10': coco_eval.stats[7],   # AR given 10 detections per image
            'AR_100': coco_eval.stats[8],  # AR given 100 detections per image
        }
        
        return results
        
    finally:
        os.unlink(pred_file)


def evaluate_model(model, processor, dataloader, device, spdnet=None, 
                   confidence_threshold=0.5, iou_threshold=0.5, desc="Evaluating"):
    """
    Evaluate a model on the dataset (simple metrics).
    
    Args:
        model: RT-DETR model
        processor: RT-DETR processor
        dataloader: DataLoader
        device: Device
        spdnet: Optional SPDNet for de-raining (None = vanilla RT-DETR)
        confidence_threshold: Threshold for counting detections
        iou_threshold: IoU threshold for matching
        desc: Progress bar description
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    if spdnet is not None:
        spdnet.eval()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_detections = 0
    total_gt = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images = batch['images'].to(device)
            gt_boxes_list = batch['boxes']
            gt_labels_list = batch['labels']
            
            # Apply de-raining if SPDNet provided
            if spdnet is not None:
                # Scale for SPDNet
                spdnet_input = images * 255.0
                derain_output = spdnet(spdnet_input)
                if isinstance(derain_output, tuple):
                    images = derain_output[0]
                else:
                    images = derain_output
                images = torch.clamp(images / 255.0, 0, 1)
            
            # Run detection
            outputs = model(pixel_values=images)
            
            # Post-process
            target_sizes = torch.tensor([[IMAGE_SIZE, IMAGE_SIZE]] * images.shape[0]).to(device)
            results = processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes,
                threshold=0.01  # Low threshold for evaluation
            )
            
            # Evaluate each image
            for i, (result, gt_boxes, gt_labels) in enumerate(zip(results, gt_boxes_list, gt_labels_list)):
                pred_boxes = result['boxes'].cpu()
                pred_scores = result['scores'].cpu()
                pred_labels = result['labels'].cpu()
                
                # Filter by confidence
                mask = pred_scores >= confidence_threshold
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]
                
                total_detections += len(pred_boxes)
                total_gt += len(gt_boxes)
                
                # Match predictions to ground truth
                matched_gt = set()
                for pb in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx, gb in enumerate(gt_boxes):
                        if gt_idx in matched_gt:
                            continue
                        iou = compute_iou(pb.numpy(), gb.numpy())
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_threshold and best_gt_idx >= 0:
                        total_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        total_fp += 1
                
                total_fn += len(gt_boxes) - len(matched_gt)
    
    # Compute metrics
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_detections': total_detections,
        'total_gt': total_gt,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    print("=" * 80)
    print("Adapted SPDNet Evaluation with COCO mAP")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}")
    print(f"pycocotools available: {HAS_PYCOCOTOOLS}")
    
    # Check if adapted model exists
    if not os.path.exists(ADAPTED_SPDNET_PATH):
        print(f"\n[ERROR] Adapted SPDNet not found at: {ADAPTED_SPDNET_PATH}")
        print("Please run training first or check the path.")
        return
    
    # ==========================================================================
    # Load dataset
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Loading dataset...")
    print("=" * 80)
    
    dataset = RainyCocoDataset(
        rain_dir=COCO_RAIN_DIR,
        split=VAL_SPLIT,
        annotations_path=ANNOTATIONS_PATH,
        image_size=IMAGE_SIZE,
        percent_dataset=PERCENT_DATASET
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # ==========================================================================
    # Load models
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Loading models...")
    print("=" * 80)
    
    # Load RT-DETR
    print("\n1. Loading RT-DETR...")
    rtdetr_model, processor = load_model_and_processor(
        model_name=RTDETR_MODEL_NAME,
        num_labels=NUM_LABELS
    )
    rtdetr_model = rtdetr_model.to(DEVICE)
    rtdetr_model.eval()
    
    # Load Original SPDNet
    print("\n2. Loading Original SPDNet...")
    original_spdnet = load_spdnet_model(
        ORIGINAL_SPDNET_PATH,
        device=DEVICE,
        n_feats=SPDNET_N_FEATS,
        n_resblocks=SPDNET_N_RESBLOCKS
    )
    original_spdnet.eval()
    
    # Load Adapted SPDNet
    print("\n3. Loading Adapted SPDNet...")
    adapted_spdnet = load_spdnet_model(
        ORIGINAL_SPDNET_PATH,  # Load architecture
        device=DEVICE,
        n_feats=SPDNET_N_FEATS,
        n_resblocks=SPDNET_N_RESBLOCKS
    )
    # Load adapted weights
    adapted_weights = torch.load(ADAPTED_SPDNET_PATH, map_location=DEVICE)
    adapted_spdnet.load_state_dict(adapted_weights)
    adapted_spdnet.eval()
    print(f"   Loaded weights from: {ADAPTED_SPDNET_PATH}")
    
    # ==========================================================================
    # Evaluate all three approaches with COCO mAP
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Running COCO mAP Evaluation...")
    print("=" * 80)
    
    results = {}
    mAP_results = {}
    
    # 1. Vanilla RT-DETR (no de-raining)
    print("\n[1/3] Evaluating Vanilla RT-DETR...")
    preds_vanilla, gt_vanilla = generate_coco_predictions(
        rtdetr_model, processor, dataloader, DEVICE,
        spdnet=None, desc="Vanilla RT-DETR"
    )
    mAP_results['vanilla'] = compute_coco_map(preds_vanilla, gt_vanilla, ANNOTATIONS_PATH)
    results['vanilla'] = evaluate_model(
        rtdetr_model, processor, dataloader, DEVICE,
        spdnet=None,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        desc="Vanilla RT-DETR (simple metrics)"
    )
    torch.cuda.empty_cache()
    
    # 2. Original SPDNet + RT-DETR
    print("\n[2/3] Evaluating Original SPDNet + RT-DETR...")
    preds_original, gt_original = generate_coco_predictions(
        rtdetr_model, processor, dataloader, DEVICE,
        spdnet=original_spdnet, desc="Original SPDNet"
    )
    mAP_results['original_spdnet'] = compute_coco_map(preds_original, gt_original, ANNOTATIONS_PATH)
    results['original_spdnet'] = evaluate_model(
        rtdetr_model, processor, dataloader, DEVICE,
        spdnet=original_spdnet,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        desc="Original SPDNet (simple metrics)"
    )
    torch.cuda.empty_cache()
    
    # 3. Adapted SPDNet + RT-DETR
    print("\n[3/3] Evaluating Adapted SPDNet + RT-DETR...")
    preds_adapted, gt_adapted = generate_coco_predictions(
        rtdetr_model, processor, dataloader, DEVICE,
        spdnet=adapted_spdnet, desc="Adapted SPDNet"
    )
    mAP_results['adapted_spdnet'] = compute_coco_map(preds_adapted, gt_adapted, ANNOTATIONS_PATH)
    results['adapted_spdnet'] = evaluate_model(
        rtdetr_model, processor, dataloader, DEVICE,
        spdnet=adapted_spdnet,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        desc="Adapted SPDNet (simple metrics)"
    )
    
    # ==========================================================================
    # Print results
    # ==========================================================================
    print("\n" + "=" * 80)
    print("COCO mAP RESULTS")
    print("=" * 80)
    
    print(f"\nDataset: Rainy COCO Validation ({PERCENT_DATASET}%)")
    
    if HAS_PYCOCOTOOLS and mAP_results.get('vanilla'):
        print("\n" + "-" * 100)
        print(f"{'Method':<30} {'mAP':>10} {'AP50':>10} {'AP75':>10} {'AP_S':>10} {'AP_M':>10} {'AP_L':>10}")
        print("-" * 100)
        
        for method in ['vanilla', 'original_spdnet', 'adapted_spdnet']:
            method_name = {
                'vanilla': 'Vanilla RT-DETR',
                'original_spdnet': 'Original SPDNet + RT-DETR',
                'adapted_spdnet': 'Adapted SPDNet + RT-DETR'
            }[method]
            
            if mAP_results[method]:
                m = mAP_results[method]
                print(f"{method_name:<30} "
                      f"{m['AP']:>10.4f} "
                      f"{m['AP50']:>10.4f} "
                      f"{m['AP75']:>10.4f} "
                      f"{m['AP_small']:>10.4f} "
                      f"{m['AP_medium']:>10.4f} "
                      f"{m['AP_large']:>10.4f}")
        
        print("-" * 100)
        
        # Calculate mAP improvements
        print("\n" + "-" * 80)
        print("mAP IMPROVEMENTS OVER BASELINE (Vanilla RT-DETR)")
        print("-" * 80)
        
        baseline_map = mAP_results['vanilla']['AP']
        baseline_ap50 = mAP_results['vanilla']['AP50']
        
        for method in ['original_spdnet', 'adapted_spdnet']:
            method_name = {
                'original_spdnet': 'Original SPDNet',
                'adapted_spdnet': 'Adapted SPDNet (yours)'
            }[method]
            
            if mAP_results[method]:
                map_diff = mAP_results[method]['AP'] - baseline_map
                ap50_diff = mAP_results[method]['AP50'] - baseline_ap50
                map_pct = map_diff / baseline_map * 100 if baseline_map > 0 else 0
                ap50_pct = ap50_diff / baseline_ap50 * 100 if baseline_ap50 > 0 else 0
                
                print(f"{method_name}:")
                print(f"  mAP:  {mAP_results[method]['AP']:.4f} ({map_diff:+.4f}, {map_pct:+.2f}%)")
                print(f"  AP50: {mAP_results[method]['AP50']:.4f} ({ap50_diff:+.4f}, {ap50_pct:+.2f}%)")
        
        # Compare adapted vs original
        print("\n" + "-" * 80)
        print("ADAPTED vs ORIGINAL SPDNet")
        print("-" * 80)
        
        if mAP_results['adapted_spdnet'] and mAP_results['original_spdnet']:
            adapted_map = mAP_results['adapted_spdnet']['AP']
            original_map = mAP_results['original_spdnet']['AP']
            diff = adapted_map - original_map
            pct = diff / original_map * 100 if original_map > 0 else 0
            
            if diff > 0:
                print(f"✅ Adapted SPDNet mAP: {adapted_map:.4f} (Original: {original_map:.4f})")
                print(f"   Improvement: +{diff:.4f} ({pct:+.2f}%)")
            elif diff < 0:
                print(f"⚠️ Adapted SPDNet mAP: {adapted_map:.4f} (Original: {original_map:.4f})")
                print(f"   Decrease: {diff:.4f} ({pct:.2f}%)")
            else:
                print(f"➖ Adapted SPDNet mAP: {adapted_map:.4f} (same as Original)")
    
    # Simple metrics table
    print("\n" + "=" * 80)
    print("SIMPLE METRICS (Precision/Recall/F1 @ IoU=0.5, Conf=0.5)")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print(f"{'Method':<30} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Detections':>12}")
    print("-" * 80)
    
    for method, metrics in results.items():
        method_name = {
            'vanilla': 'Vanilla RT-DETR',
            'original_spdnet': 'Original SPDNet + RT-DETR',
            'adapted_spdnet': 'Adapted SPDNet + RT-DETR'
        }[method]
        
        print(f"{method_name:<30} "
              f"{metrics['precision']:>12.4f} "
              f"{metrics['recall']:>12.4f} "
              f"{metrics['f1']:>12.4f} "
              f"{metrics['total_detections']:>12}")
    
    print("-" * 80)
    print("=" * 80)
    
    # Save results
    results_path = f"{os.path.dirname(ADAPTED_SPDNET_PATH)}/evaluation_results.json"
    save_results = {
        'config': {
            'dataset': COCO_RAIN_DIR,
            'percent_dataset': PERCENT_DATASET,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'iou_threshold': IOU_THRESHOLD
        },
        'simple_metrics': results,
        'coco_mAP': {k: v for k, v in mAP_results.items() if v is not None}
    }
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n[OK] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
