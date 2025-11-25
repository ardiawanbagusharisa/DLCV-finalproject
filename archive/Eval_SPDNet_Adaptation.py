#!/usr/bin/env python3
"""
Evaluation script for Adapted SPDNet with Conditional Model.

This script evaluates the SPDNet adaptation approach:
    1. Vanilla RT-DETR on clean images (baseline)
    2. Adapted SPDNet + Vanilla RT-DETR on rainy images
    3. Rain Detector decides which path to use

Key Insight:
    - Clean images bypass SPDNet entirely → RT-DETR performance is PRESERVED
    - Rainy images go through Adapted SPDNet → Detection-optimized de-raining

Comparison:
    - Vanilla RT-DETR (no de-raining)
    - Original SPDNet + RT-DETR (generic de-raining)
    - Adapted SPDNet + RT-DETR (detection-optimized de-raining)
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO

from utils.data_utils import load_datasets, split_by_domain
from utils.model_utils import load_model_and_processor
from utils.spdnet_utils import load_spdnet_model
from utils.eval_utils import (
    COCO_CLASS_NAMES,
    build_label_mapping,
    generate_predictions, 
    evaluate_coco
)

# =============================================================================
# Configuration
# =============================================================================

# Dataset paths
COCO_DIR = "E:/Python/DLCV/dataset/coco"
COCO_RAIN_DIR = "E:/Python/DLCV/dataset/coco_rain"

# Model paths
ADAPTED_SPDNET_PATH = "./outputs_spdnet_adaptation/spdnet_adapted.pt"
ORIGINAL_SPDNET_PATH = "E:/Python/DLCV/Project DLCV/model_spa.pt"
VANILLA_RTDETR_NAME = "PekingU/rtdetr_r18vd"

# SPDNet configuration
SPDNET_N_FEATS = 32
SPDNET_N_RESBLOCKS = 3

# Evaluation configuration
DATASET_FRACTION = 0.1  # Use 10% for quick testing, set to 1.0 for full eval
INFERENCE_THRESHOLD = 0.01  # For COCO evaluation

# Output
OUTPUT_DIR = "./eval_spdnet_adaptation_results"


class AdaptedSPDNetRTDETR(torch.nn.Module):
    """
    Simple wrapper that chains SPDNet (adapted) with RT-DETR.
    
    This is used for evaluating the adapted SPDNet on rainy images.
    Unlike the conditional model, this ALWAYS applies de-raining.
    """
    
    def __init__(self, spdnet_model, rtdetr_model):
        super().__init__()
        self.spdnet = spdnet_model
        self.rtdetr = rtdetr_model
    
    def forward(self, pixel_values, labels=None, **kwargs):
        """
        Forward pass: SPDNet → RT-DETR
        
        Args:
            pixel_values: Input images (B, 3, H, W) in [0, 1] range
            labels: Ground truth labels for training (optional)
        
        Returns:
            RT-DETR outputs
        """
        # Scale to [0, 255] for SPDNet
        spdnet_input = pixel_values * 255.0
        
        # De-rain
        with torch.no_grad():  # SPDNet is frozen during evaluation
            derain_output = self.spdnet(spdnet_input)
            if isinstance(derain_output, tuple):
                derain_output = derain_output[0]
        
        # Scale back to [0, 1] for RT-DETR
        clean_images = torch.clamp(derain_output / 255.0, 0, 1)
        
        # Detect
        outputs = self.rtdetr(pixel_values=clean_images, labels=labels, **kwargs)
        
        return outputs


def create_spdnet_architecture(device='cuda', n_feats=32, n_resblocks=3):
    """
    Create SPDNet architecture without loading weights.
    
    Args:
        device: Device to create model on
        n_feats: Number of feature maps
        n_resblocks: Number of residual blocks
    
    Returns:
        SPDNet model (uninitialized weights)
    """
    import sys
    SPDNET_SRC_PATH = r'E:\Python\DLCV\SPDNet\src'
    if SPDNET_SRC_PATH not in sys.path:
        sys.path.insert(0, SPDNET_SRC_PATH)
    
    from model import spdnet
    
    class Args:
        def __init__(self):
            self.n_feats = n_feats
            self.n_resblocks = n_resblocks
    
    args = Args()
    model = spdnet.make_model(args)
    model = model.to(device)
    
    return model


def load_adapted_model(adapted_spdnet_path, rtdetr_name, device='cuda'):
    """
    Load the adapted SPDNet with vanilla RT-DETR.
    
    Args:
        adapted_spdnet_path: Path to adapted SPDNet weights
        rtdetr_name: HuggingFace model name for RT-DETR
        device: Device to load on
    
    Returns:
        Tuple of (model, processor)
    """
    print(f"\nLoading Adapted SPDNet from: {adapted_spdnet_path}")
    
    # Create SPDNet architecture (without loading pretrained weights)
    spdnet = create_spdnet_architecture(
        device=device,
        n_feats=SPDNET_N_FEATS,
        n_resblocks=SPDNET_N_RESBLOCKS
    )
    
    # Load adapted weights
    state_dict = torch.load(adapted_spdnet_path, map_location=device)
    spdnet.load_state_dict(state_dict)
    spdnet.eval()
    print(f"[OK] Adapted SPDNet weights loaded")
    
    # Load RT-DETR
    print(f"Loading RT-DETR from: {rtdetr_name}")
    rtdetr, processor = load_model_and_processor(rtdetr_name)
    rtdetr = rtdetr.to(device)
    rtdetr.eval()
    print(f"[OK] RT-DETR loaded")
    
    # Combine
    model = AdaptedSPDNetRTDETR(spdnet, rtdetr)
    model = model.to(device)
    model.eval()
    
    return model, processor


def load_original_spdnet_model(original_spdnet_path, rtdetr_name, device='cuda'):
    """
    Load the original (non-adapted) SPDNet with RT-DETR for comparison.
    """
    print(f"\nLoading Original SPDNet from: {original_spdnet_path}")
    
    spdnet = load_spdnet_model(
        model_path=original_spdnet_path,
        device=device,
        n_feats=SPDNET_N_FEATS,
        n_resblocks=SPDNET_N_RESBLOCKS
    )
    spdnet.eval()
    
    rtdetr, processor = load_model_and_processor(rtdetr_name)
    rtdetr = rtdetr.to(device)
    rtdetr.eval()
    
    model = AdaptedSPDNetRTDETR(spdnet, rtdetr)
    model = model.to(device)
    model.eval()
    
    return model, processor


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("Adapted SPDNet Evaluation Script")
    print("=" * 80)
    print("\nThis script compares three approaches on RAINY images:")
    print("  1. Vanilla RT-DETR (no de-raining)")
    print("  2. Original SPDNet + RT-DETR (generic de-raining)")
    print("  3. Adapted SPDNet + RT-DETR (detection-optimized de-raining)")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Check if adapted model exists
    if not os.path.exists(ADAPTED_SPDNET_PATH):
        print(f"\n❌ ERROR: Adapted SPDNet not found at {ADAPTED_SPDNET_PATH}")
        print(f"   Please run Training_SPDNet_Adaptation.py first!")
        return
    
    # ==========================================================================
    # Step 1: Load COCO annotations
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Loading COCO annotations...")
    print("=" * 80)
    
    coco_gt = COCO(f"{COCO_RAIN_DIR}/annotations/instances_val2017.json")
    all_image_ids = coco_gt.getImgIds()
    
    # Sample if needed
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
    
    # ==========================================================================
    # Step 2: Evaluate Vanilla RT-DETR (baseline)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Evaluating Vanilla RT-DETR (no de-raining)...")
    print("=" * 80)
    
    vanilla_model, processor = load_model_and_processor(VANILLA_RTDETR_NAME)
    vanilla_model = vanilla_model.to(device)
    vanilla_model.eval()
    
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
    vanilla_metrics = evaluate_coco(vanilla_preds, coco_gt, f"{OUTPUT_DIR}/vanilla_preds.json")
    
    # Clean up
    del vanilla_model
    torch.cuda.empty_cache()
    
    # ==========================================================================
    # Step 3: Evaluate Original SPDNet + RT-DETR
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Evaluating Original SPDNet + RT-DETR...")
    print("=" * 80)
    
    original_model, _ = load_original_spdnet_model(ORIGINAL_SPDNET_PATH, VANILLA_RTDETR_NAME, device)
    
    original_preds = generate_predictions(
        model=original_model,
        processor=processor,
        device=device,
        coco_gt=coco_gt,
        image_ids=image_ids,
        val_images_dir=f"{COCO_RAIN_DIR}/val2017",
        label_to_coco_id=label_to_coco_id,
        class_names=COCO_CLASS_NAMES,
        threshold=INFERENCE_THRESHOLD
    )
    original_metrics = evaluate_coco(original_preds, coco_gt, f"{OUTPUT_DIR}/original_spdnet_preds.json")
    
    # Clean up
    del original_model
    torch.cuda.empty_cache()
    
    # ==========================================================================
    # Step 4: Evaluate Adapted SPDNet + RT-DETR
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Evaluating Adapted SPDNet + RT-DETR...")
    print("=" * 80)
    
    adapted_model, _ = load_adapted_model(ADAPTED_SPDNET_PATH, VANILLA_RTDETR_NAME, device)
    
    adapted_preds = generate_predictions(
        model=adapted_model,
        processor=processor,
        device=device,
        coco_gt=coco_gt,
        image_ids=image_ids,
        val_images_dir=f"{COCO_RAIN_DIR}/val2017",
        label_to_coco_id=label_to_coco_id,
        class_names=COCO_CLASS_NAMES,
        threshold=INFERENCE_THRESHOLD
    )
    adapted_metrics = evaluate_coco(adapted_preds, coco_gt, f"{OUTPUT_DIR}/adapted_spdnet_preds.json")
    
    # ==========================================================================
    # Step 5: Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (on Rainy Images)")
    print("=" * 80)
    
    vanilla_map = vanilla_metrics.stats[0]
    original_map = original_metrics.stats[0]
    adapted_map = adapted_metrics.stats[0]
    
    print(f"\n{'Method':<35} {'mAP':<10} {'vs Vanilla':<15}")
    print("-" * 60)
    print(f"{'Vanilla RT-DETR (no de-rain)':<35} {vanilla_map:.4f}     {'(baseline)':<15}")
    
    original_diff = original_map - vanilla_map
    original_pct = 100 * original_diff / vanilla_map if vanilla_map > 0 else 0
    print(f"{'Original SPDNet + RT-DETR':<35} {original_map:.4f}     {original_diff:+.4f} ({original_pct:+.1f}%)")
    
    adapted_diff = adapted_map - vanilla_map
    adapted_pct = 100 * adapted_diff / vanilla_map if vanilla_map > 0 else 0
    print(f"{'Adapted SPDNet + RT-DETR':<35} {adapted_map:.4f}     {adapted_diff:+.4f} ({adapted_pct:+.1f}%)")
    
    # Compare adapted vs original
    print("\n" + "-" * 60)
    adapt_vs_orig = adapted_map - original_map
    adapt_vs_orig_pct = 100 * adapt_vs_orig / original_map if original_map > 0 else 0
    print(f"Adapted vs Original SPDNet: {adapt_vs_orig:+.4f} ({adapt_vs_orig_pct:+.1f}%)")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if adapted_map > original_map:
        print("✅ Adaptation IMPROVED de-raining for detection!")
        print(f"   The adapted SPDNet produces images that are better for RT-DETR.")
    elif adapted_map < original_map:
        print("⚠️ Adaptation did NOT improve results.")
        print("   Consider: more training epochs, different learning rate, or data augmentation.")
    else:
        print("➡️ Adaptation has similar performance to original.")
    
    if adapted_map > vanilla_map:
        print(f"\n✅ De-raining HELPS detection on rainy images (+{adapted_pct:.1f}%)")
    else:
        print(f"\n⚠️ De-raining may not be helping. Check SPDNet output quality.")
    
    # Save results
    results = {
        'vanilla_map': float(vanilla_map),
        'original_spdnet_map': float(original_map),
        'adapted_spdnet_map': float(adapted_map),
        'improvement_vs_vanilla': float(adapted_diff),
        'improvement_vs_original': float(adapt_vs_orig),
        'num_images_evaluated': len(image_ids)
    }
    
    torch.save(results, f"{OUTPUT_DIR}/evaluation_results.pt")
    print(f"\n[OK] Results saved to: {OUTPUT_DIR}/evaluation_results.pt")
    
    print("\n" + "=" * 80)
    print("[OK] Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
