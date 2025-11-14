#!/usr/bin/env python3
"""
Evaluation script for Integrated Rain-Robust RT-DETR model.

This script evaluates the integrated SPDNet+RT-DETR model on rainy datasets.
Compares against vanilla RT-DETR baseline.
"""

import os
import torch
import warnings
from pycocotools.coco import COCO

# Import utility modules
from utils.model_utils import load_model_and_processor
from utils.integrated_model import load_integrated_model, RainRobustRTDETR
from utils.eval_utils import (
    COCO_CLASS_NAMES,
    build_label_mapping,
    generate_predictions,
    evaluate_coco,
    calculate_pr_curve_per_class,
    plot_pr_curves,
    plot_overall_pr_curve
)

warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = "E:\\Python\\DLCV\\Project\\dataset\\coco_rain"
INTEGRATED_MODEL_PATH = "./outputs_integrated/best_integrated"  # Path to saved integrated model
SPDNET_MODEL_PATH = "E:\\Python\\DLCV\\Project\\model_spa.pt"  # For loading architecture

# SPDNet Configuration
SPDNET_N_FEATS = 32
SPDNET_N_RESBLOCKS = 3

# Dataset sampling
DATASET_FRACTION = 1.0  # Use 100% of dataset

CONFIDENCE_THRESHOLD = 0.3  # For visualization
INFERENCE_THRESHOLD = 0.01  # Low threshold for COCO eval

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("Integrated Rain-Robust RT-DETR Evaluation")
    print("=" * 80)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Integrated model path: {INTEGRATED_MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Load integrated model
    print("\n" + "=" * 80)
    print("Loading Integrated Model")
    print("=" * 80)
    
    if os.path.exists(INTEGRATED_MODEL_PATH):
        print(f"Loading from: {INTEGRATED_MODEL_PATH}")
        
        # Load SPDNet architecture (needed for loading weights)
        from utils.spdnet_utils import load_spdnet_model
        spdnet_model = load_spdnet_model(
            SPDNET_MODEL_PATH,
            device=DEVICE,
            n_feats=SPDNET_N_FEATS,
            n_resblocks=SPDNET_N_RESBLOCKS
        )
        
        # Load integrated model
        integrated_model = RainRobustRTDETR.from_pretrained(
            INTEGRATED_MODEL_PATH,
            spdnet_model=spdnet_model,
            device=DEVICE
        )
        integrated_model = integrated_model.to(DEVICE)
        integrated_model.eval()
        
        # Load processor
        from transformers import RTDetrImageProcessor
        processor = RTDetrImageProcessor.from_pretrained(
            os.path.join(INTEGRATED_MODEL_PATH, "processor")
        )
        
        print("✓ Integrated model loaded successfully")
    else:
        print(f"✗ Integrated model not found at: {INTEGRATED_MODEL_PATH}")
        print("Loading pretrained models separately for comparison...")
        
        integrated_model, processor = load_integrated_model(
            spdnet_path=SPDNET_MODEL_PATH,
            rtdetr_name="PekingU/rtdetr_r18vd",
            num_labels=80,
            device=DEVICE,
            freeze_derain=False,
            freeze_detection=False,
            spdnet_n_feats=SPDNET_N_FEATS,
            spdnet_n_resblocks=SPDNET_N_RESBLOCKS
        )
        integrated_model = integrated_model.to(DEVICE)
        integrated_model.eval()
    
    # Load vanilla RT-DETR for comparison
    print("\n" + "=" * 80)
    print("Loading Vanilla RT-DETR (Baseline)")
    print("=" * 80)
    vanilla_model, vanilla_processor = load_model_and_processor("PekingU/rtdetr_r18vd")
    vanilla_model = vanilla_model.to(DEVICE)
    vanilla_model.eval()
    print("✓ Vanilla RT-DETR loaded")
    
    # Load COCO dataset
    print("\n" + "=" * 80)
    print("Loading COCO Dataset")
    print("=" * 80)
    coco_gt = COCO(f"{DATASET_PATH}/annotations/instances_val2017.json")
    
    # Sample images if needed
    all_image_ids = sorted(coco_gt.getImgIds())
    if DATASET_FRACTION < 1.0:
        import random
        random.seed(42)
        num_samples = int(len(all_image_ids) * DATASET_FRACTION)
        image_ids = random.sample(all_image_ids, num_samples)
        print(f"Using {len(image_ids)} images ({DATASET_FRACTION * 100:.0f}% of dataset)")
    else:
        image_ids = all_image_ids
        print(f"Using all {len(image_ids)} images")
    
    # Build label mapping
    label_to_coco_id = build_label_mapping(COCO_CLASS_NAMES, coco_gt)
    
    # Evaluate vanilla RT-DETR (baseline)
    print("\n" + "=" * 80)
    print("Evaluating Vanilla RT-DETR (Baseline)")
    print("=" * 80)
    
    vanilla_predictions = generate_predictions(
        coco_gt=coco_gt,
        image_ids=image_ids,
        model=vanilla_model,
        processor=vanilla_processor,
        label_to_coco_id=label_to_coco_id,
        device=DEVICE,
        threshold=INFERENCE_THRESHOLD,
        images_dir=f"{DATASET_PATH}/val2017"
    )
    
    print("\nVanilla RT-DETR Results:")
    vanilla_metrics = evaluate_coco(coco_gt, vanilla_predictions)
    
    # Evaluate integrated model
    print("\n" + "=" * 80)
    print("Evaluating Integrated Rain-Robust RT-DETR")
    print("=" * 80)
    
    integrated_predictions = generate_predictions(
        coco_gt=coco_gt,
        image_ids=image_ids,
        model=integrated_model,
        processor=processor,
        label_to_coco_id=label_to_coco_id,
        device=DEVICE,
        threshold=INFERENCE_THRESHOLD,
        images_dir=f"{DATASET_PATH}/val2017"
    )
    
    print("\nIntegrated Model Results:")
    integrated_metrics = evaluate_coco(coco_gt, integrated_predictions)
    
    # Compare results
    print("\n" + "=" * 80)
    print("Comparison: Vanilla vs Integrated")
    print("=" * 80)
    
    metrics_to_compare = [
        ('Average Precision (AP) @ IoU=0.50:0.95', 0),
        ('AP @ IoU=0.50', 1),
        ('AP @ IoU=0.75', 2),
        ('AP (small)', 3),
        ('AP (medium)', 4),
        ('AP (large)', 5),
    ]
    
    print(f"\n{'Metric':<40} {'Vanilla':<12} {'Integrated':<12} {'Improvement':<12}")
    print("=" * 80)
    
    for metric_name, idx in metrics_to_compare:
        vanilla_val = vanilla_metrics[idx] if idx < len(vanilla_metrics) else 0.0
        integrated_val = integrated_metrics[idx] if idx < len(integrated_metrics) else 0.0
        improvement = integrated_val - vanilla_val
        improvement_pct = (improvement / vanilla_val * 100) if vanilla_val > 0 else 0.0
        
        print(f"{metric_name:<40} {vanilla_val:<12.3f} {integrated_val:<12.3f} "
              f"{improvement:+.3f} ({improvement_pct:+.1f}%)")
    
    # Calculate and plot PR curves
    print("\n" + "=" * 80)
    print("Calculating PR Curves")
    print("=" * 80)
    
    print("\nVanilla RT-DETR PR curves...")
    vanilla_pr_data = calculate_pr_curve_per_class(
        coco_gt, vanilla_predictions, COCO_CLASS_NAMES
    )
    
    print("\nIntegrated Model PR curves...")
    integrated_pr_data = calculate_pr_curve_per_class(
        coco_gt, integrated_predictions, COCO_CLASS_NAMES
    )
    
    # Plot overall PR curve comparison
    print("\nPlotting overall PR curve comparison...")
    plot_overall_pr_curve(
        [vanilla_pr_data, integrated_pr_data],
        ['Vanilla RT-DETR', 'Integrated (SPDNet+RT-DETR)'],
        save_path='./pr_curve_comparison_integrated.png'
    )
    
    # Plot per-class PR curves for top classes
    print("\nPlotting per-class PR curves...")
    
    # Find top 10 classes by number of instances
    class_counts = {}
    for img_id in image_ids:
        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        for ann in anns:
            cat_id = ann['category_id']
            class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
    
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_class_names = [coco_gt.loadCats(cat_id)[0]['name'] for cat_id, _ in top_classes]
    
    plot_pr_curves(
        vanilla_pr_data,
        top_class_names,
        save_path='./pr_curves_vanilla_top10.png'
    )
    
    plot_pr_curves(
        integrated_pr_data,
        top_class_names,
        save_path='./pr_curves_integrated_top10.png'
    )
    
    print("\n" + "=" * 80)
    print("✓ Evaluation completed successfully!")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - PR curve comparison: ./pr_curve_comparison_integrated.png")
    print(f"  - Vanilla top-10 PR curves: ./pr_curves_vanilla_top10.png")
    print(f"  - Integrated top-10 PR curves: ./pr_curves_integrated_top10.png")
    
    # Print key takeaway
    print("\n" + "=" * 80)
    print("KEY RESULTS")
    print("=" * 80)
    vanilla_map = vanilla_metrics[0]
    integrated_map = integrated_metrics[0]
    improvement = integrated_map - vanilla_map
    improvement_pct = (improvement / vanilla_map * 100) if vanilla_map > 0 else 0.0
    
    print(f"Vanilla RT-DETR mAP:       {vanilla_map:.3f}")
    print(f"Integrated Model mAP:      {integrated_map:.3f}")
    print(f"Improvement:               {improvement:+.3f} ({improvement_pct:+.1f}%)")
    
    if improvement > 0:
        print(f"\n✓ Integrated model OUTPERFORMS vanilla RT-DETR on rainy data!")
    else:
        print(f"\n⚠ Integrated model needs more training or tuning")
    
    # Clean up
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
