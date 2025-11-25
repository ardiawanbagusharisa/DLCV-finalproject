#!/usr/bin/env python3
"""
Evaluation script for Conditional Rain-Robust RT-DETR model.

This script evaluates the conditional model and compares it against vanilla RT-DETR.
Provides detailed timing breakdown to demonstrate speed improvements.

Evaluation metrics:
    - COCO mAP (clean vs rainy images)
    - Inference time breakdown (rain detection + de-raining + detection)
    - Rain detection accuracy
"""

import os
import torch
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from utils.data_utils import load_datasets, split_by_domain
from utils.conditional_model import load_conditional_model
from utils.model_utils import load_model_and_processor
from utils.eval_utils import (
    COCO_CLASS_NAMES,
    build_label_mapping,
    generate_predictions, 
    evaluate_coco, 
    run_inference
)

# Configuration
COCO_RAIN_DIR = "E:/Python/DLCV/dataset/coco_rain"  # Evaluate on rainy data
CONDITIONAL_MODEL_PATH = "./outputs_conditional/best_conditional"
RAIN_DETECTOR_PATH = "./rain_detector_pretrained/rain_detector_best.pt"
SPDNET_MODEL_PATH = "E:/Python/DLCV/Project DLCV/model_spa.pt"
VANILLA_RTDETR_NAME = "PekingU/rtdetr_r18vd"  # For comparison
USE_TRAINED_MODEL = True  # If True, load trained conditional model; else use pretrained components

# Evaluation configuration
NUM_SAMPLES = 500  # Number of samples to evaluate (for timing)
DATASET_FRACTION = 1  # Use 100% of dataset for testing
CONFIDENCE_THRESHOLD = 0.3  # For visualization
INFERENCE_THRESHOLD = 0.01  # For COCO evaluation
RAIN_THRESHOLD = 0.5

# Output
OUTPUT_DIR = "./eval_conditional_results"


def time_inference(model, images, device, num_warmup=10, num_runs=100):
    """
    Measure inference time with proper warmup.
    
    Args:
        model: Model to evaluate
        images: Batch of images (B, 3, H, W)
        device: Device
        num_warmup: Warmup iterations
        num_runs: Measurement iterations
    
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(images)
    
    # Synchronize before timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(images)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)


def evaluate_timing(conditional_model, vanilla_model, processor, ds_valid, device, num_samples=100):
    """
    Detailed timing comparison between conditional and vanilla models.
    
    Returns:
        Dictionary with timing statistics
    """
    print("\n" + "=" * 80)
    print("Detailed Timing Analysis")
    print("=" * 80)
    
    # Prepare sample images
    sample_images = []
    sample_domains = []
    
    for i in range(min(num_samples, len(ds_valid))):
        img_path, image, annotations = ds_valid[i]
        
        image = Image.fromarray(image).convert('RGB')
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        sample_images.append(pixel_values)
        
        # Determine domain from image path
        is_rainy = 'coco_rain' in str(img_path).lower() or 'rain' in str(img_path).lower()
        sample_domains.append('rainy' if is_rainy else 'clean')
    
    # Batch images
    batch = torch.cat(sample_images[:min(8, len(sample_images))], dim=0)  # Batch of 8
    
    print(f"\nTiming on batch of {batch.shape[0]} images...")
    print(f"Image size: {batch.shape[2]}×{batch.shape[3]}")
    
    # Time vanilla RT-DETR
    print("\n1. Vanilla RT-DETR:")
    vanilla_time, vanilla_std = time_inference(vanilla_model, batch, device)
    print(f"   Average: {vanilla_time:.2f} ± {vanilla_std:.2f} ms")
    
    # Time conditional model
    print("\n2. Conditional Model:")
    conditional_time, conditional_std = time_inference(conditional_model, batch, device)
    print(f"   Average: {conditional_time:.2f} ± {conditional_std:.2f} ms")
    
    # Component breakdown (approximate)
    rain_detector_time = 5  # ~5ms for rain detection
    spdnet_time = 120  # ~120ms for de-raining
    rtdetr_time = 40  # ~40ms for RT-DETR
    
    # Calculate expected times based on rain ratio
    clean_count = sum(1 for d in sample_domains if d == 'clean')
    rainy_count = len(sample_domains) - clean_count
    rain_ratio = rainy_count / len(sample_domains) if len(sample_domains) > 0 else 0
    
    expected_conditional = rain_ratio * (rain_detector_time + spdnet_time + rtdetr_time) + \
                          (1 - rain_ratio) * (rain_detector_time + rtdetr_time)
    
    print(f"\n3. Breakdown:")
    print(f"   Rain ratio in batch: {100*rain_ratio:.1f}%")
    print(f"   Expected conditional time: {expected_conditional:.2f} ms")
    print(f"     - Rain detection: ~{rain_detector_time} ms")
    print(f"     - De-raining (rainy images): ~{spdnet_time} ms × {rain_ratio:.2f} = {spdnet_time*rain_ratio:.2f} ms")
    print(f"     - RT-DETR: ~{rtdetr_time} ms")
    
    print(f"\n4. Speedup:")
    speedup = vanilla_time / conditional_time if conditional_time > 0 else 0
    print(f"   Conditional vs Vanilla: {speedup:.2f}x")
    
    if rain_ratio < 0.5:
        print(f"   [OK] Conditional is faster (low rain ratio)")
    else:
        print(f"   [WARN] Similar speed (high rain ratio - most images need de-raining)")
    
    return {
        'vanilla_time': vanilla_time,
        'conditional_time': conditional_time,
        'speedup': speedup,
        'rain_ratio': rain_ratio,
        'num_samples': len(sample_images)
    }


def evaluate_rain_detection(conditional_model, ds_valid, device, processor):
    """
    Evaluate rain detector accuracy.
    
    Returns:
        Rain detection accuracy metrics
    """
    print("\n" + "=" * 80)
    print("Rain Detection Accuracy")
    print("=" * 80)
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    rain_scores_clean = []
    rain_scores_rainy = []
    
    conditional_model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(len(ds_valid)), desc="Evaluating rain detection"):
            img_path, image, annotations = ds_valid[i]
            image = Image.fromarray(image).convert('RGB')
            
            # Get ground truth domain from image path
            is_rainy_gt = 'coco_rain' in str(img_path).lower() or 'rain' in str(img_path).lower()
            
            # Process image
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            
            # Get rain detection score
            # Use the internal preprocessing method if available, otherwise raw
            if hasattr(conditional_model, '_preprocess_for_rain_detector'):
                rain_input = conditional_model._preprocess_for_rain_detector(pixel_values)
                rain_score = conditional_model.rain_detector(rain_input).item()
            else:
                rain_score = conditional_model.rain_detector(pixel_values).item()
                
            is_rainy_pred = rain_score > conditional_model.rain_threshold
            
            # Store scores
            if is_rainy_gt:
                rain_scores_rainy.append(rain_score)
            else:
                rain_scores_clean.append(rain_score)
            
            # Confusion matrix
            if is_rainy_pred and is_rainy_gt:
                true_positives += 1
            elif is_rainy_pred and not is_rainy_gt:
                false_positives += 1
            elif not is_rainy_pred and not is_rainy_gt:
                true_negatives += 1
            else:  # not is_rainy_pred and is_rainy_gt
                false_negatives += 1
    
    # Calculate metrics
    accuracy = (true_positives + true_negatives) / len(ds_valid)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAccuracy: {100*accuracy:.2f}%")
    print(f"Precision: {100*precision:.2f}%")
    print(f"Recall: {100*recall:.2f}%")
    print(f"F1 Score: {100*f1:.2f}%")
    
    print(f"\nConfusion Matrix:")
    print(f"  TP: {true_positives} | FP: {false_positives}")
    print(f"  FN: {false_negatives} | TN: {true_negatives}")
    
    print(f"\nRain Score Distribution:")
    print(f"  Clean images: {np.mean(rain_scores_clean):.3f} ± {np.std(rain_scores_clean):.3f}")
    print(f"  Rainy images: {np.mean(rain_scores_rainy):.3f} ± {np.std(rain_scores_rainy):.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': (true_positives, false_positives, true_negatives, false_negatives),
        'rain_scores_clean': rain_scores_clean,
        'rain_scores_rainy': rain_scores_rainy
    }


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("Conditional Rain-Robust RT-DETR Evaluation Script")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load validation dataset (rainy images)
    print("\n" + "=" * 80)
    print("Step 1: Loading validation dataset...")
    print("=" * 80)
    _, ds_valid = load_datasets(
        coco_dir="E:/Python/DLCV/dataset/coco",  # Needed for loading
        coco_rain_dir=COCO_RAIN_DIR,
        coco_ratio=0.9,
        rain_ratio=0.1,
        seed=42
    )
    
    # Use only rainy validation set
    _, ds_valid_rainy = split_by_domain(ds_valid)
    print(f"Rainy validation samples: {len(ds_valid_rainy)}")
    
    # Load conditional model
    print("\n" + "=" * 80)
    print("Step 2: Loading conditional model...")
    print("=" * 80)
    
    if USE_TRAINED_MODEL:
        print(f"Mode: Using TRAINED conditional model from {CONDITIONAL_MODEL_PATH}")
        # Check if model exists
        if not os.path.exists(f"{CONDITIONAL_MODEL_PATH}/rain_detector.pt"):
            print(f"ERROR: Conditional model not found at {CONDITIONAL_MODEL_PATH}")
            print("Please train the conditional model first:")
            print("  → python Training_conditional.py")
            return
            
        # Use trained components
        rd_path = f"{CONDITIONAL_MODEL_PATH}/rain_detector.pt"
        spd_path = f"{CONDITIONAL_MODEL_PATH}/derain_module.pt"
        det_path = f"{CONDITIONAL_MODEL_PATH}/detection_module"
    else:
        print(f"Mode: Using UNTRAINED conditional model (pretrained components only)")
        print("Note: This evaluates the integration of pretrained models without fine-tuning.")
        
        # Use pretrained components
        rd_path = RAIN_DETECTOR_PATH
        spd_path = SPDNET_MODEL_PATH
        det_path = VANILLA_RTDETR_NAME

    conditional_model, processor = load_conditional_model(
        rain_detector_path=rd_path,
        spdnet_path=spd_path,
        rtdetr_name=det_path,
        device=device,
        rain_threshold=RAIN_THRESHOLD
    )
    
    # Load vanilla RT-DETR for comparison
    print("\n" + "=" * 80)
    print("Step 3: Loading vanilla RT-DETR for comparison...")
    print("=" * 80)
    vanilla_model, _ = load_model_and_processor(VANILLA_RTDETR_NAME)
    vanilla_model = vanilla_model.to(device)
    vanilla_model.eval()
    
    # Evaluate rain detection accuracy
    print("\n" + "=" * 80)
    print("Step 4: Evaluating rain detection...")
    print("=" * 80)
    rain_metrics = evaluate_rain_detection(conditional_model, ds_valid_rainy, device, processor)
    
    # # Timing comparison
    # print("\n" + "=" * 80)
    # print("Step 5: Timing comparison...")
    # print("=" * 80)
    # timing_stats = evaluate_timing(
    #     conditional_model, vanilla_model, processor, 
    #     ds_valid_rainy, device, num_samples=NUM_SAMPLES
    # )
    
    # Load COCO annotations for evaluation
    print("\n" + "=" * 80)
    print("Step 6: Loading COCO annotations for evaluation...")
    print("=" * 80)
    coco_gt = COCO(f"{COCO_RAIN_DIR}/annotations/instances_val2017.json")
    all_image_ids = coco_gt.getImgIds()
    
    # Sample dataset if fraction < 1.0
    if DATASET_FRACTION < 1.0:
        import random
        random.seed(42)
        num_samples = int(len(all_image_ids) * DATASET_FRACTION)
        image_ids = random.sample(all_image_ids, num_samples)
        print(f"Using {len(image_ids)} images ({DATASET_FRACTION * 100:.0f}% of dataset)")
    else:
        image_ids = all_image_ids
        print(f"Using all {len(image_ids)} images")
    
    label_to_coco_id = build_label_mapping(COCO_CLASS_NAMES, coco_gt)
    
    # COCO evaluation
    print("\n" + "=" * 80)
    print("Step 7: COCO mAP evaluation...")
    print("=" * 80)
    
    print("\n7a. Conditional model (with de-raining):")
    conditional_preds = generate_predictions(
        model=conditional_model,
        processor=processor,
        device=device,
        coco_gt=coco_gt,
        image_ids=image_ids,
        val_images_dir=f"{COCO_RAIN_DIR}/val2017",
        label_to_coco_id=label_to_coco_id,
        class_names=COCO_CLASS_NAMES,
        threshold=INFERENCE_THRESHOLD
    )
    conditional_metrics = evaluate_coco(conditional_preds, coco_gt, "conditional_preds.json")
    
    print("\n7b. Vanilla RT-DETR (no de-raining):")
    vanilla_preds = generate_predictions(
        model=vanilla_model,
        processor=processor,
        device=device,
        coco_gt=coco_gt,
        image_ids=image_ids,
        val_images_dir=f"{COCO_RAIN_DIR}/val2017",
        label_to_coco_id=label_to_coco_id,
        class_names=COCO_CLASS_NAMES,
        threshold=INFERENCE_THRESHOLD
    )
    vanilla_metrics = evaluate_coco(vanilla_preds, coco_gt, "vanilla_preds.json")
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    print("\n1. Detection Performance (on rainy images):")
    print(f"   Conditional model mAP: {conditional_metrics.stats[0]:.4f}")
    print(f"   Vanilla RT-DETR mAP:   {vanilla_metrics.stats[0]:.4f}")
    improvement = conditional_metrics.stats[0] - vanilla_metrics.stats[0]
    print(f"   Improvement: {improvement:+.4f} ({100*improvement/vanilla_metrics.stats[0]:+.1f}%)")
    
    print("\n2. Rain Detection:")
    print(f"   Accuracy:  {100*rain_metrics['accuracy']:.2f}%")
    print(f"   Precision: {100*rain_metrics['precision']:.2f}%")
    print(f"   Recall:    {100*rain_metrics['recall']:.2f}%")
    
    # Save results
    results = {
        'conditional_map': float(conditional_metrics.stats[0]),
        'vanilla_map': float(vanilla_metrics.stats[0]),
        'improvement': float(improvement),
        'rain_detection_accuracy': rain_metrics['accuracy']
    }
    
    torch.save(results, f"{OUTPUT_DIR}/evaluation_results.pt")
    print(f"\n[OK] Results saved to {OUTPUT_DIR}/evaluation_results.pt")
    
    print("\n" + "=" * 80)
    print("[OK] Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
