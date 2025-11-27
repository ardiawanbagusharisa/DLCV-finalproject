#!/usr/bin/env python3
"""
Evaluation Script for Feature-Level De-raining

Compares the trained feature de-rain model against:
1. Vanilla RT-DETR (baseline)
2. SPDNet + RT-DETR (pixel-level)
3. Feature De-rain + RT-DETR (ours)

Metrics:
- COCO mAP (AP, AP50, AP75, AP_small, AP_medium, AP_large)
- Inference speed (ms/image, FPS)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import gc

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from utils.feature_derain import (
    FeatureDerainRTDETR,
    LightweightFeatureDerain,
    create_feature_derain_rtdetr
)
from utils.eval_utils import evaluate_coco, generate_predictions


# =============================================================================
# Configuration
# =============================================================================

# Paths
COCO_RAIN_DIR = "E:/Python/DLCV/dataset/coco_rain"
OUTPUT_DIR = "./outputs_feature_derain"
CHECKPOINT_PATH = "./outputs_feature_derain/feature_derain_best.pt"

# Model
MODEL_NAME = "PekingU/rtdetr_r18vd"
DERAIN_TYPE = "lightweight"

# Evaluation
PERCENT_DATASET = 10  # Evaluate on 10% of val set
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
BATCH_SIZE = 8
NUM_WORKERS = 4

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Utilities
# =============================================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_feature_derain_model(checkpoint_path, device="cuda"):
    """Load trained feature de-rain model"""
    # Create model
    rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME)
    model = FeatureDerainRTDETR(
        rtdetr_model=rtdetr,
        derain_type=DERAIN_TYPE,
        freeze_backbone=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'derain_module' in checkpoint:
        model.derain_module.load_state_dict(checkpoint['derain_module'])
    if 'rtdetr' in checkpoint:
        model.rtdetr.load_state_dict(checkpoint['rtdetr'])
    
    model = model.to(device)
    model.eval()
    
    return model


def load_vanilla_model(device="cuda"):
    """Load vanilla RT-DETR model"""
    model = RTDetrForObjectDetection.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def benchmark_inference(model, images, device="cuda", num_warmup=10, num_runs=100):
    """Benchmark inference speed"""
    # Warmup
    for _ in range(num_warmup):
        _ = model(images[:1].to(device))
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for i in range(num_runs):
        _ = model(images[i % len(images):i % len(images) + 1].to(device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    total_time = time.time() - start
    ms_per_image = (total_time / num_runs) * 1000
    fps = num_runs / total_time
    
    return ms_per_image, fps


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    print("=" * 80)
    print("Feature-Level De-raining Evaluation")
    print("=" * 80)
    
    # Load processor
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    
    # Load validation dataset
    print("\n[1/4] Loading validation dataset...")
    import supervision as sv
    
    ds_valid = sv.DetectionDataset.from_coco(
        images_directory_path=f"{COCO_RAIN_DIR}/val2017",
        annotations_path=f"{COCO_RAIN_DIR}/annotations/instances_val2017.json",
    )
    
    # Subsample
    np.random.seed(42)
    all_paths = list(ds_valid.image_paths)
    n_samples = max(1, int(len(all_paths) * PERCENT_DATASET / 100))
    sample_indices = np.random.choice(len(all_paths), n_samples, replace=False)
    sample_paths = [all_paths[i] for i in sample_indices]
    
    print(f"  Using {n_samples} images ({PERCENT_DATASET}%)")
    
    # Prepare images for benchmark
    print("\n[2/4] Preparing benchmark images...")
    benchmark_images = []
    for path in sample_paths[:100]:  # Use 100 images for speed benchmark
        img = Image.open(path).convert('RGB')
        inputs = processor(images=img, return_tensors="pt")
        benchmark_images.append(inputs['pixel_values'])
    benchmark_tensor = torch.cat(benchmark_images, dim=0)
    
    # Results storage
    results = {}
    
    # =========================================================================
    # Evaluate Vanilla RT-DETR
    # =========================================================================
    print("\n[3/4] Evaluating models...")
    print("\n--- Vanilla RT-DETR ---")
    
    vanilla_model = load_vanilla_model(DEVICE)
    
    # Speed benchmark
    ms_vanilla, fps_vanilla = benchmark_inference(vanilla_model, benchmark_tensor, DEVICE)
    print(f"  Speed: {ms_vanilla:.1f} ms/img ({fps_vanilla:.1f} FPS)")
    
    # Generate predictions
    vanilla_preds = []
    for path in tqdm(sample_paths, desc="  Predicting"):
        img = Image.open(path).convert('RGB')
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(DEVICE)
        
        outputs = vanilla_model(pixel_values)
        
        # Post-process
        target_sizes = torch.tensor([[img.height, img.width]], device=DEVICE)
        results_batch = processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=0.0  # Get all predictions for mAP
        )[0]
        
        vanilla_preds.append({
            'path': path,
            'boxes': results_batch['boxes'].cpu().numpy(),
            'scores': results_batch['scores'].cpu().numpy(),
            'labels': results_batch['labels'].cpu().numpy()
        })
    
    results['vanilla'] = {
        'ms_per_img': ms_vanilla,
        'fps': fps_vanilla,
        'predictions': vanilla_preds
    }
    
    del vanilla_model
    clear_memory()
    
    # =========================================================================
    # Evaluate Feature De-rain Model
    # =========================================================================
    if os.path.exists(CHECKPOINT_PATH):
        print("\n--- Feature De-rain RT-DETR ---")
        
        feature_model = load_feature_derain_model(CHECKPOINT_PATH, DEVICE)
        
        # Speed benchmark
        ms_feature, fps_feature = benchmark_inference(feature_model, benchmark_tensor, DEVICE)
        print(f"  Speed: {ms_feature:.1f} ms/img ({fps_feature:.1f} FPS)")
        
        # Generate predictions
        feature_preds = []
        for path in tqdm(sample_paths, desc="  Predicting"):
            img = Image.open(path).convert('RGB')
            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(DEVICE)
            
            outputs = feature_model(pixel_values)
            
            # Post-process
            target_sizes = torch.tensor([[img.height, img.width]], device=DEVICE)
            results_batch = processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes,
                threshold=0.0
            )[0]
            
            feature_preds.append({
                'path': path,
                'boxes': results_batch['boxes'].cpu().numpy(),
                'scores': results_batch['scores'].cpu().numpy(),
                'labels': results_batch['labels'].cpu().numpy()
            })
        
        results['feature_derain'] = {
            'ms_per_img': ms_feature,
            'fps': fps_feature,
            'predictions': feature_preds
        }
        
        del feature_model
        clear_memory()
    else:
        print(f"\n[SKIP] Feature de-rain checkpoint not found: {CHECKPOINT_PATH}")
    
    # =========================================================================
    # Compute mAP
    # =========================================================================
    print("\n[4/4] Computing mAP metrics...")
    
    # Prepare ground truth
    gt_annotations = []
    for path in sample_paths:
        ann = ds_valid.annotations.get(path)
        if ann is not None:
            gt_annotations.append({
                'path': path,
                'boxes': ann.xyxy,
                'labels': ann.class_id
            })
        else:
            gt_annotations.append({
                'path': path,
                'boxes': np.array([]).reshape(0, 4),
                'labels': np.array([])
            })
    
    # Compute mAP for each model
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    def compute_coco_metrics(predictions, gt_annotations):
        """Compute COCO-style mAP metrics"""
        # Create COCO format annotations
        coco_gt = {
            'images': [],
            'annotations': [],
            'categories': [{'id': i, 'name': f'class_{i}'} for i in range(80)]
        }
        
        coco_dt = []
        ann_id = 0
        
        for img_id, (pred, gt) in enumerate(zip(predictions, gt_annotations)):
            coco_gt['images'].append({
                'id': img_id,
                'file_name': pred['path'],
                'height': 640,
                'width': 640
            })
            
            # Ground truth
            for box, label in zip(gt['boxes'], gt['labels']):
                x1, y1, x2, y2 = box
                coco_gt['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': int(label),
                    'bbox': [x1, y1, x2-x1, y2-y1],  # xywh
                    'area': (x2-x1) * (y2-y1),
                    'iscrowd': 0
                })
                ann_id += 1
            
            # Predictions
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                x1, y1, x2, y2 = box
                coco_dt.append({
                    'image_id': img_id,
                    'category_id': int(label),
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'score': float(score)
                })
        
        # Run COCO evaluation
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_gt, f)
            gt_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_dt, f)
            dt_file = f.name
        
        coco_gt_obj = COCO(gt_file)
        coco_dt_obj = coco_gt_obj.loadRes(dt_file)
        
        coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Cleanup
        os.unlink(gt_file)
        os.unlink(dt_file)
        
        return {
            'AP': coco_eval.stats[0],
            'AP50': coco_eval.stats[1],
            'AP75': coco_eval.stats[2],
            'AP_small': coco_eval.stats[3],
            'AP_medium': coco_eval.stats[4],
            'AP_large': coco_eval.stats[5]
        }
    
    # Compute metrics for each model
    final_results = {}
    
    for model_name, model_results in results.items():
        print(f"\n  {model_name}:")
        metrics = compute_coco_metrics(model_results['predictions'], gt_annotations)
        
        final_results[model_name] = {
            'speed': {
                'ms_per_img': model_results['ms_per_img'],
                'fps': model_results['fps']
            },
            'mAP': metrics
        }
        
        print(f"    AP: {metrics['AP']*100:.2f}%")
        print(f"    AP50: {metrics['AP50']*100:.2f}%")
        print(f"    AP75: {metrics['AP75']*100:.2f}%")
        print(f"    Speed: {model_results['ms_per_img']:.1f} ms ({model_results['fps']:.1f} FPS)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n| Model | AP | AP50 | AP75 | ms/img | FPS |")
    print("|-------|-----|------|------|--------|-----|")
    
    for model_name, res in final_results.items():
        ap = res['mAP']['AP'] * 100
        ap50 = res['mAP']['AP50'] * 100
        ap75 = res['mAP']['AP75'] * 100
        ms = res['speed']['ms_per_img']
        fps = res['speed']['fps']
        print(f"| {model_name} | {ap:.2f}% | {ap50:.2f}% | {ap75:.2f}% | {ms:.1f} | {fps:.1f} |")
    
    # Compare improvements
    if 'vanilla' in final_results and 'feature_derain' in final_results:
        vanilla_ap = final_results['vanilla']['mAP']['AP']
        feature_ap = final_results['feature_derain']['mAP']['AP']
        improvement = (feature_ap - vanilla_ap) / vanilla_ap * 100
        
        print(f"\nFeature De-rain vs Vanilla: {improvement:+.2f}% mAP")
    
    # Save results
    output_file = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
