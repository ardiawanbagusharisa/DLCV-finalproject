#!/usr/bin/env python3
"""
Training Script for Lightweight Feature-Level De-raining

This script trains the LightweightFeatureDerain module integrated with RT-DETR.

Key Features:
1. Uses RT-DETR's detection loss (no paired clean/rainy images needed)
2. Two-phase training: de-rain only → joint fine-tuning
3. Memory efficient with gradient accumulation
4. Compatible with existing data utilities

Architecture:
    Rainy Image → RT-DETR Backbone → Feature Derain (TRAIN) → Encoder → Decoder → Predictions
                  (FROZEN Phase 1)   (~50K params)            (FROZEN Phase 1)

Expected Training Time: ~90 minutes on RTX 3090 (10% dataset)
Expected Improvement: +1-2% mAP on rainy images
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import gc
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from utils.feature_derain import (
    LightweightFeatureDerain,
    MultiScaleFeatureDerain,
    create_feature_derain_rtdetr,
    FeatureDerainRTDETR
)
from utils.data_utils import (
    load_datasets,
    get_augmentation_transforms,
    AugmentedDetectionDataset,
    collate_fn
)

# =============================================================================
# Configuration
# =============================================================================

# Paths
COCO_DIR = "E:/Python/DLCV/dataset/coco"
COCO_RAIN_DIR = "E:/Python/DLCV/dataset/coco_rain"
OUTPUT_DIR = "./outputs_feature_derain"

# Model
MODEL_NAME = "PekingU/rtdetr_r18vd"
DERAIN_TYPE = "multiscale"  # "lightweight" or "multiscale" - multiscale has ~50K+ params

# Dataset
PERCENT_DATASET = 20   # Use 20% for faster training (~24K images)
COCO_RATIO = 0.3       # 30% clean images (prevent over-processing)
RAIN_RATIO = 0.7       # 70% rainy images

# Training - Phase 1 (De-rain module only)
PHASE1_EPOCHS = 5
PHASE1_LR = 5e-4  # Increased LR for better convergence

# Training - Phase 2 (Joint fine-tuning)
PHASE2_EPOCHS = 10
PHASE2_LR_DERAIN = 1e-4
PHASE2_LR_DETECTOR = 1e-5  # 10x lower for pretrained parts

# General training
BATCH_SIZE = 16  # Increased further for 16GB GPU
GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation needed with larger batch
EVAL_BATCH_SIZE = 24  # Larger eval batch
NUM_WORKERS = 4  # Enable parallel data loading
SEED = 42

# Evaluation
EVAL_EVERY_N_EPOCHS = 1
LOG_EVERY_N_STEPS = 50

# Early Stopping
EARLY_STOPPING_PATIENCE = 3  # Stop if val loss doesn't improve for N epochs

# Memory
USE_AMP = True  # Feature derain is compatible with AMP (unlike SPDNet)
GRADIENT_CHECKPOINTING = False

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def count_parameters(model, trainable_only=True):
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_time(seconds):
    """Format seconds to readable time"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"


# =============================================================================
# Custom collate function for RT-DETR training
# =============================================================================

def collate_fn_for_rtdetr(batch):
    """
    Custom collate function for RT-DETR training.
    Handles variable-length annotations properly.
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Format labels for RT-DETR
    labels = []
    for item in batch:
        label = {
            'class_labels': item['labels']['class_labels'],
            'boxes': item['labels']['boxes'],
        }
        labels.append(label)
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


# =============================================================================
# Dataset Wrapper for Feature Derain Training
# =============================================================================

class COCODerainDataset(torch.utils.data.Dataset):
    """
    Fast COCO-format dataset for feature de-raining training.
    Uses pycocotools for efficient loading.
    """
    
    def __init__(self, img_dir, ann_file, processor, percent_dataset=100, seed=42):
        """
        Args:
            img_dir: Path to images directory
            ann_file: Path to COCO annotations JSON file
            processor: RTDetrImageProcessor
            percent_dataset: Percentage of dataset to use
            seed: Random seed for subsampling
        """
        from pycocotools.coco import COCO
        
        self.img_dir = img_dir
        self.processor = processor
        
        # Load COCO annotations
        print(f"  Loading annotations from {ann_file}...")
        self.coco = COCO(ann_file)
        
        # Get all image IDs
        all_img_ids = self.coco.getImgIds()
        
        # Subsample if needed
        if percent_dataset < 100:
            np.random.seed(seed)
            n_samples = max(1, int(len(all_img_ids) * percent_dataset / 100))
            indices = np.random.choice(len(all_img_ids), n_samples, replace=False)
            self.img_ids = [all_img_ids[i] for i in indices]
        else:
            self.img_ids = all_img_ids
        
        print(f"  Using {len(self.img_ids)} images ({percent_dataset}%)")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        import cv2
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            if 'bbox' in ann and ann['area'] > 0:
                x, y, bw, bh = ann['bbox']  # COCO format: [x, y, width, height]
                # Convert to normalized cxcywh
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                bw_norm = bw / w
                bh_norm = bh / h
                
                # Clip to valid range
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                bw_norm = max(0, min(1, bw_norm))
                bh_norm = max(0, min(1, bh_norm))
                
                if bw_norm > 0 and bh_norm > 0:
                    boxes.append([cx, cy, bw_norm, bh_norm])
                    labels.append(ann['category_id'] - 1)  # Convert to 0-indexed
        
        # Process image with RT-DETR processor
        from PIL import Image
        pil_image = Image.fromarray(image)
        encoding = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze(0)
        
        # Create tensors
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        
        return {
            'pixel_values': pixel_values,
            'labels': {
                'boxes': boxes_tensor,
                'class_labels': labels_tensor
            }
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    phase,
    scaler=None,
    gradient_accumulation_steps=1,
    log_every_n_steps=50
):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Phase {phase} | Epoch {epoch}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels']
        
        # Move labels to device
        labels = [
            {
                'boxes': l['boxes'].to(device),
                'class_labels': l['class_labels'].to(device)
            }
            for l in labels
        ]
        
        # Forward pass with AMP
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Logging
        if (step + 1) % log_every_n_steps == 0:
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, scaler=None):
    """Evaluate model on validation set"""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels']
        
        # Move labels to device
        labels = [
            {
                'boxes': l['boxes'].to(device),
                'class_labels': l['class_labels'].to(device)
            }
            for l in labels
        ]
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values, labels=labels)
        else:
            outputs = model(pixel_values, labels=labels)
        
        total_loss += outputs.loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    print("=" * 80)
    print("Feature-Level De-raining Training")
    print("=" * 80)
    print(f"Config:")
    print(f"  - De-rain type: {DERAIN_TYPE}")
    print(f"  - Dataset: {PERCENT_DATASET}% ({RAIN_RATIO*100:.0f}% rainy, {COCO_RATIO*100:.0f}% clean)")
    print(f"  - Phase 1: {PHASE1_EPOCHS} epochs, LR={PHASE1_LR}")
    print(f"  - Phase 2: {PHASE2_EPOCHS} epochs, LR={PHASE2_LR_DERAIN}/{PHASE2_LR_DETECTOR}")
    print(f"  - Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Device: {DEVICE}")
    print(f"  - AMP: {USE_AMP}")
    print("=" * 80)
    
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n[1/5] Loading datasets...")
    
    # Load RT-DETR processor
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    
    # Load combined COCO + COCO_rain datasets using data_utils
    ds_train, ds_valid = load_datasets(
        coco_dir=COCO_DIR,
        coco_rain_dir=COCO_RAIN_DIR,
        coco_ratio=COCO_RATIO,
        rain_ratio=RAIN_RATIO,
        seed=SEED
    )
    
    print(f"  Total: {len(ds_train)} train, {len(ds_valid)} val images")
    
    # Subsample if needed
    if PERCENT_DATASET < 100:
        np.random.seed(SEED)
        train_paths = list(ds_train.image_paths)
        valid_paths = list(ds_valid.image_paths)
        
        n_train = max(1, int(len(train_paths) * PERCENT_DATASET / 100))
        n_valid = max(1, int(len(valid_paths) * PERCENT_DATASET / 100))
        
        train_indices = np.random.choice(len(train_paths), n_train, replace=False)
        valid_indices = np.random.choice(len(valid_paths), n_valid, replace=False)
        
        # Create subsampled image paths and annotations
        train_paths_subset = [train_paths[i] for i in train_indices]
        valid_paths_subset = [valid_paths[i] for i in valid_indices]
        
        # Create subsampled datasets
        import supervision as sv
        ds_train = sv.DetectionDataset(
            classes=ds_train.classes,
            images=train_paths_subset,
            annotations={p: ds_train.annotations[p] for p in train_paths_subset}
        )
        ds_valid = sv.DetectionDataset(
            classes=ds_valid.classes,
            images=valid_paths_subset,
            annotations={p: ds_valid.annotations[p] for p in valid_paths_subset}
        )
        
        print(f"  Subsampled to {n_train} train, {n_valid} val images ({PERCENT_DATASET}%)")
    
    # Create datasets
    train_transform = get_augmentation_transforms(is_train=True)
    val_transform = get_augmentation_transforms(is_train=False)
    
    train_dataset = AugmentedDetectionDataset(ds_train, processor, train_transform, is_train=True)
    val_dataset = AugmentedDetectionDataset(ds_valid, processor, val_transform, is_train=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_for_rtdetr,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_for_rtdetr,
        pin_memory=True
    )
    
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} images, {len(val_loader)} batches")
    
    # =========================================================================
    # Create Model
    # =========================================================================
    print("\n[2/5] Creating model...")
    
    # Load base RT-DETR
    rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME)
    
    # Create feature derain model
    model = FeatureDerainRTDETR(
        rtdetr_model=rtdetr,
        derain_type=DERAIN_TYPE,
        num_residual_blocks=2,
        freeze_backbone=True  # Start with frozen backbone (Phase 1)
    )
    model = model.to(DEVICE)
    
    print(f"  De-rain module params: {count_parameters(model.derain_module):,}")
    print(f"  Total trainable params: {count_parameters(model):,}")
    
    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
    # =========================================================================
    # Phase 1: Train De-rain Module Only
    # =========================================================================
    print("\n[3/5] Phase 1: Training de-rain module (detector frozen)...")
    
    # Freeze everything except de-rain module
    for param in model.rtdetr.parameters():
        param.requires_grad = False
    for param in model.derain_module.parameters():
        param.requires_grad = True
    
    print(f"  Trainable params: {count_parameters(model):,}")
    
    # Phase 1 optimizer
    optimizer1 = torch.optim.AdamW(
        model.derain_module.parameters(),
        lr=PHASE1_LR,
        weight_decay=0.01
    )
    
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, 
        T_max=PHASE1_EPOCHS
    )
    
    # Training log
    training_log = {
        'phase1': [],
        'phase2': [],
        'config': {
            'derain_type': DERAIN_TYPE,
            'percent_dataset': PERCENT_DATASET,
            'phase1_epochs': PHASE1_EPOCHS,
            'phase2_epochs': PHASE2_EPOCHS,
            'batch_size': BATCH_SIZE,
            'gradient_accumulation': GRADIENT_ACCUMULATION_STEPS
        }
    }
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, PHASE1_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer1, DEVICE, epoch, phase=1,
            scaler=scaler, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            log_every_n_steps=LOG_EVERY_N_STEPS
        )
        
        scheduler1.step()
        
        # Evaluate
        val_loss = float('inf')
        if epoch % EVAL_EVERY_N_EPOCHS == 0:
            val_loss = evaluate(model, val_loader, DEVICE, scaler)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler1.get_last_lr()[0],
            'time': epoch_time
        }
        training_log['phase1'].append(log_entry)
        
        print(f"  Epoch {epoch}/{PHASE1_EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {format_time(epoch_time)}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.derain_module.state_dict(),
                os.path.join(OUTPUT_DIR, 'derain_phase1_best.pt')
            )
        
        clear_memory()
    
    phase1_time = time.time() - start_time
    print(f"  Phase 1 completed in {format_time(phase1_time)}")
    
    # =========================================================================
    # Phase 2: Joint Fine-tuning
    # =========================================================================
    print("\n[4/5] Phase 2: Joint fine-tuning (all trainable)...")
    
    # Unfreeze detector with lower learning rate
    for param in model.rtdetr.parameters():
        param.requires_grad = True
    
    print(f"  Trainable params: {count_parameters(model):,}")
    
    # Phase 2 optimizer with different LRs
    optimizer2 = torch.optim.AdamW([
        {'params': model.derain_module.parameters(), 'lr': PHASE2_LR_DERAIN},
        {'params': model.rtdetr.parameters(), 'lr': PHASE2_LR_DETECTOR}
    ], weight_decay=0.01)
    
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, 
        T_max=PHASE2_EPOCHS
    )
    
    phase2_start = time.time()
    
    # Early stopping tracking
    early_stop_counter = 0
    phase2_best_val_loss = float('inf')
    
    for epoch in range(1, PHASE2_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer2, DEVICE, epoch, phase=2,
            scaler=scaler, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            log_every_n_steps=LOG_EVERY_N_STEPS
        )
        
        scheduler2.step()
        
        # Evaluate
        val_loss = float('inf')
        if epoch % EVAL_EVERY_N_EPOCHS == 0:
            val_loss = evaluate(model, val_loader, DEVICE, scaler)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr_derain': PHASE2_LR_DERAIN,
            'lr_detector': PHASE2_LR_DETECTOR,
            'time': epoch_time
        }
        training_log['phase2'].append(log_entry)
        
        print(f"  Epoch {epoch}/{PHASE2_EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {format_time(epoch_time)}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'derain_module': model.derain_module.state_dict(),
                'rtdetr': model.rtdetr.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, os.path.join(OUTPUT_DIR, 'feature_derain_best.pt'))
        
        # Early stopping check
        if val_loss < phase2_best_val_loss:
            phase2_best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping triggered! Val loss hasn't improved for {EARLY_STOPPING_PATIENCE} epochs.")
                print(f"  Best val loss: {phase2_best_val_loss:.4f}")
                break
        
        clear_memory()
    
    phase2_time = time.time() - phase2_start
    total_time = time.time() - start_time
    
    print(f"  Phase 2 completed in {format_time(phase2_time)}")
    
    # =========================================================================
    # Save Final Model and Logs
    # =========================================================================
    print("\n[5/5] Saving final model...")
    
    # Save final model
    torch.save({
        'derain_module': model.derain_module.state_dict(),
        'rtdetr': model.rtdetr.state_dict(),
        'config': {
            'derain_type': DERAIN_TYPE,
            'model_name': MODEL_NAME
        }
    }, os.path.join(OUTPUT_DIR, 'feature_derain_final.pt'))
    
    # Save training log
    training_log['total_time'] = total_time
    training_log['best_val_loss'] = best_val_loss
    
    with open(os.path.join(OUTPUT_DIR, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Best model: feature_derain_best.pt")
    print("=" * 80)
    
    return model


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
