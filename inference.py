#!/usr/bin/env python3
"""
Inference Script for Feature-Level De-raining RT-DETR

This script performs object detection on a single test image using the
trained Feature De-rain RT-DETR model.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --image path/to/image.jpg --output results/
    python inference.py --image path/to/image.jpg --threshold 0.5 --save
    python inference.py --image path/to/image.jpg --compare  # 4-panel: GT, Vanilla, SPDNet+RTDETR, Ours
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from utils.feature_derain import FeatureDerainRTDETR
from utils.spdnet_utils import load_spdnet_model, derain_image


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "PekingU/rtdetr_r18vd"
DERAIN_TYPE = "multiscale"
CHECKPOINT_PATH = "./outputs_feature_derain/feature_derain_best.pt"
SPDNET_PATH = "./model_spa.pt"  # SPDNet checkpoint for baseline comparison
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MixedRain dataset paths
MIXEDRAIN_DIR = "E:/Python/DLCV/mixedrain"
MIXEDRAIN_LABELS = os.path.join(MIXEDRAIN_DIR, "labels.txt")

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Colors for visualization (one per class, cycling)
np.random.seed(42)
COLORS = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)) 
          for _ in range(len(COCO_CLASSES))]

# Fixed colors for GT/Predictions
GT_COLOR = (0, 255, 0)  # Green
VANILLA_COLOR = (255, 0, 0)  # Red  
SPDNET_COLOR = (255, 165, 0)  # Orange - for SPDNet+RTDETR
DERAIN_COLOR = (0, 100, 255)  # Blue - for Feature De-rain (Ours)


# =============================================================================
# Ground Truth Loading (MixedRain)
# =============================================================================

def load_mixedrain_labels(labels_file: str = MIXEDRAIN_LABELS):
    """Load YOLO format labels from MixedRain dataset."""
    annotations = defaultdict(list)
    
    if not os.path.exists(labels_file):
        return annotations
    
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                img_id = parts[0]
                annotations[img_id].append({
                    'class_id': int(parts[1]),
                    'x_center': float(parts[2]),
                    'y_center': float(parts[3]),
                    'width': float(parts[4]),
                    'height': float(parts[5])
                })
    
    return annotations


def get_image_id_from_path(image_path: str):
    """Extract image ID from path (filename without extension)."""
    return Path(image_path).stem


def yolo_to_xyxy(ann: dict, img_w: int, img_h: int):
    """Convert YOLO normalized format to absolute (x1, y1, x2, y2)."""
    x_center = ann['x_center'] * img_w
    y_center = ann['y_center'] * img_h
    w = ann['width'] * img_w
    h = ann['height'] * img_h
    
    return (x_center - w/2, y_center - h/2, x_center + w/2, y_center + h/2)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load the trained Feature De-rain RT-DETR model."""
    print(f"Loading model from: {checkpoint_path}")
    print(f"Device: {device}")
    
    # Create model
    rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME)
    model = FeatureDerainRTDETR(
        rtdetr_model=rtdetr,
        derain_type=DERAIN_TYPE,
        freeze_backbone=False
    )
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'derain_module' in checkpoint:
            model.derain_module.load_state_dict(checkpoint['derain_module'])
            print("✓ Loaded de-rain module weights")
        if 'rtdetr' in checkpoint:
            model.rtdetr.load_state_dict(checkpoint['rtdetr'])
            print("✓ Loaded RT-DETR weights")
    else:
        print(f"⚠ Checkpoint not found at {checkpoint_path}, using pretrained weights only")
    
    model = model.to(device)
    model.eval()
    
    return model


def load_vanilla_model(device: str = "cuda"):
    """Load vanilla RT-DETR model (for comparison)."""
    print(f"Loading vanilla RT-DETR...")
    model = RTDetrForObjectDetection.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    return model


def load_spdnet_for_comparison(device: str = "cuda"):
    """
    Load SPDNet model for image de-raining (baseline comparison).
    
    Note: SPDNet has hardcoded CUDA calls, so it requires a GPU.
    """
    if not torch.cuda.is_available():
        print("⚠ SPDNet requires CUDA. Skipping SPDNet comparison.")
        return None
    
    if not os.path.exists(SPDNET_PATH):
        print(f"⚠ SPDNet model not found at {SPDNET_PATH}. Skipping SPDNet comparison.")
        return None
    
    try:
        print(f"Loading SPDNet from: {SPDNET_PATH}")
        spdnet_model = load_spdnet_model(SPDNET_PATH, device=device)
        return spdnet_model
    except Exception as e:
        print(f"⚠ Failed to load SPDNet: {e}")
        return None


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def run_inference(
    model, 
    processor, 
    image_path: str, 
    device: str = "cuda",
    confidence_threshold: float = 0.5
):
    """
    Run inference on a single image.
    
    Args:
        model: The detection model
        processor: RT-DETR image processor
        image_path: Path to the input image
        device: Device to run inference on
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        dict with 'boxes', 'scores', 'labels', 'inference_time'
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Run inference with timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    outputs = model(pixel_values=pixel_values)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Post-process results
    target_sizes = torch.tensor([original_size[::-1]]).to(device)  # (height, width)
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes,
        threshold=confidence_threshold
    )[0]
    
    return {
        'boxes': results['boxes'].cpu().numpy(),
        'scores': results['scores'].cpu().numpy(),
        'labels': results['labels'].cpu().numpy(),
        'inference_time': inference_time,
        'original_size': original_size
    }


# =============================================================================
# Visualization
# =============================================================================

def visualize_detections(
    image_path: str,
    results: dict,
    output_path: str = None,
    show: bool = True
):
    """
    Visualize detection results on the image.
    
    Args:
        image_path: Path to the input image
        results: Detection results dict
        output_path: Path to save the output image (optional)
        show: Whether to display the image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    boxes = results['boxes']
    scores = results['scores']
    labels = results['labels']
    
    print(f"\nDetected {len(boxes)} objects:")
    print("-" * 50)
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Get class name and color
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        color = COLORS[label % len(COLORS)]
        
        # Draw bounding box
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        text = f"{class_name}: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                      fill=color)
        draw.text((x1, y1), text, fill='white', font=font)
        
        print(f"  [{i+1}] {class_name}: {score:.3f} | Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    print("-" * 50)
    print(f"Inference time: {results['inference_time']:.2f} ms")
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        image.save(output_path)
        print(f"✓ Saved result to: {output_path}")
    
    if show:
        image.show()
    
    return image


def draw_detections_on_image(image: Image.Image, results: dict, title: str = None):
    """
    Draw detections on a PIL image and return it.
    
    Args:
        image: PIL Image to draw on (will be copied)
        results: Detection results dict
        title: Optional title to draw on image
        
    Returns:
        PIL Image with detections drawn
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    # Try to load fonts
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    boxes = results['boxes']
    scores = results['scores']
    labels = results['labels']
    
    # Draw title if provided
    if title:
        # Draw title background
        title_bbox = draw.textbbox((10, 10), title, font=title_font)
        draw.rectangle([title_bbox[0]-5, title_bbox[1]-5, title_bbox[2]+5, title_bbox[3]+5], 
                      fill='black')
        draw.text((10, 10), title, fill='white', font=title_font)
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Get class name and color
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        color = COLORS[label % len(COLORS)]
        
        # Draw bounding box
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        text = f"{class_name}: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                      fill=color)
        draw.text((x1, y1), text, fill='white', font=font)
    
    return image


def create_side_by_side_comparison(
    image_path: str,
    vanilla_results: dict,
    derain_results: dict,
    spdnet_results: dict = None,
    spdnet_derained_image: Image.Image = None,
    gt_annotations: list = None,
    output_path: str = None,
    show: bool = True
):
    """
    Create a 4-panel comparison image: GT | Vanilla | SPDNet+RT-DETR (derained) | Feature De-rain (Ours).
    
    Args:
        image_path: Path to original image
        vanilla_results: Results from vanilla RT-DETR
        derain_results: Results from Feature De-rain RT-DETR
        spdnet_results: Results from SPDNet + RT-DETR (optional)
        spdnet_derained_image: The derained image from SPDNet (shown as background for panel 3)
        gt_annotations: Ground truth annotations (YOLO format dicts)
        output_path: Path to save comparison image
        show: Whether to display the image
        
    Returns:
        Combined PIL Image
    """
    # Load original image
    original = Image.open(image_path).convert('RGB')
    width, height = original.size
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        title_font = ImageFont.truetype("arial.ttf", 18)
        header_font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = font
        header_font = font
    
    # Panel 1: Ground Truth
    panel1 = original.copy()
    draw1 = ImageDraw.Draw(panel1)
    n_gt = len(gt_annotations) if gt_annotations else 0
    title1 = f"Ground Truth ({n_gt} objects)"
    draw1.rectangle([5, 5, 280, 35], fill='black')
    draw1.text((10, 10), title1, fill='white', font=title_font)
    
    if gt_annotations:
        for ann in gt_annotations:
            x1, y1, x2, y2 = yolo_to_xyxy(ann, width, height)
            draw1.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=3)
            class_id = ann['class_id']
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            text_bbox = draw1.textbbox((x1, y1-18), class_name, font=font)
            draw1.rectangle([text_bbox[0]-1, text_bbox[1]-1, text_bbox[2]+1, text_bbox[3]+1], fill=GT_COLOR)
            draw1.text((x1, y1-18), class_name, fill='black', font=font)
    else:
        draw1.text((width//2-100, height//2), "No GT available", fill='red', font=title_font)
    
    # Panel 2: Vanilla RT-DETR
    panel2 = original.copy()
    draw2 = ImageDraw.Draw(panel2)
    title2 = f"Vanilla RT-DETR ({len(vanilla_results['boxes'])} det)"
    draw2.rectangle([5, 5, 280, 35], fill='black')
    draw2.text((10, 10), title2, fill='white', font=title_font)
    
    for box, score, label in zip(vanilla_results['boxes'], vanilla_results['scores'], vanilla_results['labels']):
        x1, y1, x2, y2 = box
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        draw2.rectangle([x1, y1, x2, y2], outline=VANILLA_COLOR, width=3)
        text = f"{class_name}:{score:.2f}"
        text_bbox = draw2.textbbox((x1, y1-18), text, font=font)
        draw2.rectangle([text_bbox[0]-1, text_bbox[1]-1, text_bbox[2]+1, text_bbox[3]+1], fill=VANILLA_COLOR)
        draw2.text((x1, y1-18), text, fill='white', font=font)
    
    # Panel 3: SPDNet + RT-DETR (showing derained image as background)
    if spdnet_results is not None and spdnet_derained_image is not None:
        # Use the SPDNet derained image as background (resized to match original)
        panel3 = spdnet_derained_image.copy()
        if panel3.size != (width, height):
            panel3 = panel3.resize((width, height), Image.LANCZOS)
        draw3 = ImageDraw.Draw(panel3)
        title3 = f"SPDNet+RT-DETR ({len(spdnet_results['boxes'])} det)"
        draw3.rectangle([5, 5, 280, 35], fill='black')
        draw3.text((10, 10), title3, fill='white', font=title_font)
        
        for box, score, label in zip(spdnet_results['boxes'], spdnet_results['scores'], spdnet_results['labels']):
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
            draw3.rectangle([x1, y1, x2, y2], outline=SPDNET_COLOR, width=3)
            text = f"{class_name}:{score:.2f}"
            text_bbox = draw3.textbbox((x1, y1-18), text, font=font)
            draw3.rectangle([text_bbox[0]-1, text_bbox[1]-1, text_bbox[2]+1, text_bbox[3]+1], fill=SPDNET_COLOR)
            draw3.text((x1, y1-18), text, fill='black', font=font)
    else:
        # SPDNet not available - show placeholder
        panel3 = original.copy()
        draw3 = ImageDraw.Draw(panel3)
        title3 = "SPDNet+RT-DETR (N/A)"
        draw3.rectangle([5, 5, 280, 35], fill='black')
        draw3.text((10, 10), title3, fill='white', font=title_font)
        draw3.text((width//2-100, height//2), "SPDNet unavailable", fill='orange', font=title_font)
    
    # Panel 4: Feature De-rain RT-DETR (Ours)
    panel4 = original.copy()
    draw4 = ImageDraw.Draw(panel4)
    title4 = f"Feature De-rain (Ours) ({len(derain_results['boxes'])} det)"
    draw4.rectangle([5, 5, 330, 35], fill='black')
    draw4.text((10, 10), title4, fill='white', font=title_font)
    
    for box, score, label in zip(derain_results['boxes'], derain_results['scores'], derain_results['labels']):
        x1, y1, x2, y2 = box
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        draw4.rectangle([x1, y1, x2, y2], outline=DERAIN_COLOR, width=3)
        text = f"{class_name}:{score:.2f}"
        text_bbox = draw4.textbbox((x1, y1-18), text, font=font)
        draw4.rectangle([text_bbox[0]-1, text_bbox[1]-1, text_bbox[2]+1, text_bbox[3]+1], fill=DERAIN_COLOR)
        draw4.text((x1, y1-18), text, fill='white', font=font)
    
    # Create 4-panel image (2x2 layout)
    gap = 10
    combined_width = width * 2 + gap
    combined_height = height * 2 + gap + 80  # Extra space for header and legend
    combined = Image.new('RGB', (combined_width, combined_height), color='white')
    
    # Draw header
    draw = ImageDraw.Draw(combined)
    header_text = "Model Comparison: GT | Vanilla | SPDNet+RT-DETR | Feature De-rain (Ours)"
    header_bbox = draw.textbbox((0, 0), header_text, font=header_font)
    header_x = (combined_width - (header_bbox[2] - header_bbox[0])) // 2
    draw.text((header_x, 10), header_text, fill='black', font=header_font)
    
    # Paste panels in 2x2 grid
    y_offset = 50
    combined.paste(panel1, (0, y_offset))                          # Top-left: GT
    combined.paste(panel2, (width + gap, y_offset))                # Top-right: Vanilla
    combined.paste(panel3, (0, y_offset + height + gap))           # Bottom-left: SPDNet
    combined.paste(panel4, (width + gap, y_offset + height + gap)) # Bottom-right: Ours
    
    # Draw separator lines
    # Vertical line
    draw.line([(width + gap//2, y_offset), (width + gap//2, y_offset + height * 2 + gap)], fill='gray', width=2)
    # Horizontal line
    draw.line([(0, y_offset + height + gap//2), (combined_width, y_offset + height + gap//2)], fill='gray', width=2)
    
    # Draw legend at bottom
    legend_y = combined_height - 25
    legend_spacing = combined_width // 4
    
    draw.rectangle([10, legend_y, 30, legend_y + 15], fill=GT_COLOR)
    draw.text((35, legend_y), "Ground Truth", fill='black', font=font)
    
    draw.rectangle([legend_spacing, legend_y, legend_spacing + 20, legend_y + 15], fill=VANILLA_COLOR)
    draw.text((legend_spacing + 25, legend_y), "Vanilla RT-DETR", fill='black', font=font)
    
    draw.rectangle([legend_spacing * 2, legend_y, legend_spacing * 2 + 20, legend_y + 15], fill=SPDNET_COLOR)
    draw.text((legend_spacing * 2 + 25, legend_y), "SPDNet+RT-DETR", fill='black', font=font)
    
    draw.rectangle([legend_spacing * 3, legend_y, legend_spacing * 3 + 20, legend_y + 15], fill=DERAIN_COLOR)
    draw.text((legend_spacing * 3 + 25, legend_y), "Ours", fill='black', font=font)
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        combined.save(output_path, quality=95)
        print(f"\n✓ Saved comparison to: {output_path}")
    
    # Show if requested
    if show:
        combined.show()
    
    return combined


def compare_models(
    image_path: str, 
    confidence_threshold: float = 0.5, 
    output_dir: str = None,
    device: str = "cuda",
    show: bool = True
):
    """
    Compare Feature De-rain model with Vanilla RT-DETR and SPDNet+RT-DETR, including Ground Truth.
    
    Args:
        image_path: Path to the input image
        confidence_threshold: Minimum confidence for detections
        output_dir: Directory to save comparison results
        device: Device to use for inference
        show: Whether to display the comparison image
    """
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    
    print("\n" + "=" * 80)
    print("Model Comparison: GT | Vanilla | SPDNet+RT-DETR | Feature De-rain (Ours)")
    print("=" * 80)
    
    # Try to load ground truth from MixedRain labels
    img_id = get_image_id_from_path(image_path)
    gt_labels = load_mixedrain_labels()
    gt_annotations = gt_labels.get(img_id, [])
    
    if gt_annotations:
        print(f"\n✓ Found Ground Truth: {len(gt_annotations)} objects")
        for i, ann in enumerate(gt_annotations):
            class_id = ann['class_id']
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            print(f"  [GT-{i+1}] {class_name}")
    else:
        print(f"\n⚠ No Ground Truth found for image ID: {img_id}")
    
    # Load models
    print("\nLoading models...")
    vanilla_model = load_vanilla_model(device)
    derain_model = load_model(CHECKPOINT_PATH, device)
    spdnet_model = load_spdnet_for_comparison(device)
    
    # Run inference with Vanilla RT-DETR
    print("\n--- Vanilla RT-DETR ---")
    vanilla_results = run_inference(
        vanilla_model, processor, image_path, 
        device, confidence_threshold
    )
    print(f"Detections: {len(vanilla_results['boxes'])}")
    print(f"Inference time: {vanilla_results['inference_time']:.2f} ms")
    
    # Run inference with SPDNet + RT-DETR (if available)
    spdnet_results = None
    spdnet_derained_image = None
    if spdnet_model is not None:
        print("\n--- SPDNet + RT-DETR ---")
        # First, de-rain the image with SPDNet
        original_image = Image.open(image_path).convert('RGB')
        
        # SPDNet expects [0, 255] range
        start_time = time.time()
        spdnet_derained_image = derain_image(original_image, spdnet_model, device=device)
        derain_time = (time.time() - start_time) * 1000
        print(f"SPDNet de-raining time: {derain_time:.2f} ms")
        
        # Now run RT-DETR on the derained image
        # Save temporarily to file or process directly
        # Process derained image through RT-DETR
        inputs = processor(images=spdnet_derained_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        outputs = vanilla_model(pixel_values=pixel_values)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        rtdetr_time = (time.time() - start_time) * 1000
        
        # Post-process results
        original_size = original_image.size  # (width, height)
        target_sizes = torch.tensor([original_size[::-1]]).to(device)  # (height, width)
        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=confidence_threshold
        )[0]
        
        spdnet_results = {
            'boxes': results['boxes'].detach().cpu().numpy(),
            'scores': results['scores'].detach().cpu().numpy(),
            'labels': results['labels'].detach().cpu().numpy(),
            'inference_time': derain_time + rtdetr_time,
            'original_size': original_size
        }
        print(f"Detections: {len(spdnet_results['boxes'])}")
        print(f"Total time (SPDNet + RT-DETR): {spdnet_results['inference_time']:.2f} ms")
    else:
        print("\n--- SPDNet + RT-DETR ---")
        print("⚠ SPDNet not available (requires CUDA)")
    
    # Run inference with Feature De-rain RT-DETR (Ours)
    print("\n--- Feature De-rain RT-DETR (Ours) ---")
    derain_results = run_inference(
        derain_model, processor, image_path, 
        device, confidence_threshold
    )
    print(f"Detections: {len(derain_results['boxes'])}")
    print(f"Inference time: {derain_results['inference_time']:.2f} ms")
    
    # Print detailed results
    print("\n" + "-" * 40)
    print("Vanilla RT-DETR Detections:")
    for i, (box, score, label) in enumerate(zip(vanilla_results['boxes'], vanilla_results['scores'], vanilla_results['labels'])):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        print(f"  [{i+1}] {class_name}: {score:.3f}")
    
    if spdnet_results is not None:
        print("\nSPDNet + RT-DETR Detections:")
        for i, (box, score, label) in enumerate(zip(spdnet_results['boxes'], spdnet_results['scores'], spdnet_results['labels'])):
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
            print(f"  [{i+1}] {class_name}: {score:.3f}")
    
    print("\nFeature De-rain (Ours) Detections:")
    for i, (box, score, label) in enumerate(zip(derain_results['boxes'], derain_results['scores'], derain_results['labels'])):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        print(f"  [{i+1}] {class_name}: {score:.3f}")
    print("-" * 40)
    
    # Create 4-panel comparison visualization
    output_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_comparison.jpg")
    
    create_side_by_side_comparison(
        image_path, vanilla_results, derain_results,
        spdnet_results=spdnet_results,
        spdnet_derained_image=spdnet_derained_image,
        gt_annotations=gt_annotations,
        output_path=output_path, show=show
    )
    
    return vanilla_results, spdnet_results, derain_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Feature De-rain RT-DETR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --image test.jpg
  python inference.py --image test.jpg --threshold 0.3
  python inference.py --image test.jpg --output results/ --save
  python inference.py --image test.jpg --compare --output comparison/
  python inference.py --image test.jpg --device cpu  # For when GPU is busy
        """
    )
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save the output image')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the result image')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare with vanilla RT-DETR')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help=f'Path to model checkpoint (default: {CHECKPOINT_PATH})')
    parser.add_argument('--vanilla', action='store_true',
                        help='Use vanilla RT-DETR instead of Feature De-rain')
    parser.add_argument('--device', '-d', type=str, default=None,
                        help='Device to use: "cuda" or "cpu" (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Use provided checkpoint or default
    checkpoint_path = args.checkpoint if args.checkpoint else CHECKPOINT_PATH
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = DEVICE
    
    print("\n" + "=" * 80)
    print("Feature De-rain RT-DETR Inference")
    print("=" * 80)
    print(f"Input image: {args.image}")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Device: {device}")
    
    if args.compare:
        # Compare both models with side-by-side visualization
        compare_models(
            args.image, 
            args.threshold, 
            args.output, 
            device=device,
            show=not args.no_show
        )
    else:
        # Single model inference
        processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
        
        if args.vanilla:
            print("\nUsing: Vanilla RT-DETR")
            model = load_vanilla_model(device)
        else:
            print("\nUsing: Feature De-rain RT-DETR")
            model = load_model(checkpoint_path, device)
        
        # Run inference
        results = run_inference(
            model, processor, args.image, 
            device, args.threshold
        )
        
        # Determine output path
        output_path = None
        if args.save or args.output:
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                image_name = Path(args.image).stem
                model_type = "vanilla" if args.vanilla else "feature_derain"
                output_path = os.path.join(args.output, f"{image_name}_{model_type}_result.jpg")
            else:
                output_path = str(Path(args.image).with_suffix('')) + "_result.jpg"
        
        # Visualize
        visualize_detections(
            args.image, 
            results, 
            output_path=output_path,
            show=not args.no_show
        )
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
