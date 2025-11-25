#!/usr/bin/env python3
"""
Rain Detector Pretraining Script

This script pre-trains the rain detector (binary classifier) to distinguish
between clean and rainy images. This is Phase 0 before training the full
conditional model.

Training task: Binary classification (0 = clean, 1 = rainy)
"""

import os
import sys

import warnings
warnings.filterwarnings('ignore', message='Unable to import Axes3D')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Ensure local 'utils' package is found first, avoiding conflicts with other libraries
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utils modules at top level for Windows multiprocessing support
from utils.data_utils import load_datasets, create_rain_detection_datasets
from utils.rain_detector import RainDetector

# Configuration
COCO_DIR = "E:/Python/DLCV/dataset/coco"
COCO_RAIN_DIR = "E:/Python/DLCV/dataset/coco_rain"
OUTPUT_DIR = "./rain_detector_pretrained"
CHECKPOINT_PATH = f"{OUTPUT_DIR}/rain_detector_best.pt"

# Training configuration
COCO_RATIO = 0.5  # 90% clean images
RAIN_RATIO = 0.5  # 10% rainy images
NUM_EPOCHS = 5    # Rain detection is easy, doesn't need many epochs
BATCH_SIZE = 256  # Optimized for single-process data loading (small model trains fast anyway)
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42
USE_AMP = True    # Use automatic mixed precision for faster training

# DataLoader workers
# Windows multiprocessing cannot pickle custom Dataset classes from utils package
# The RainDetectionDataset class references utils.rain_detector module
# Workers fail when trying to unpickle the dataset objects
# For this simple pretraining (small model, fast training), single-process is acceptable
# The main RT-DETR training uses HuggingFace Trainer which handles this better
NUM_WORKERS = 10

# Early stopping
EARLY_STOPPING_PATIENCE = 3


def train_epoch(model, dataloader, criterion, optimizer, device, use_amp=False):
    """Train for one epoch"""
    import time
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Timing diagnostics
    data_time = 0
    compute_time = 0
    
    # Create GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    pbar = tqdm(dataloader, desc="Training")
    batch_start = time.time()
    
    for batch_idx, batch in enumerate(pbar):
        data_time += time.time() - batch_start
        
        compute_start = time.time()
        images = batch['images'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # Forward pass with automatic mixed precision
        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        # Apply sigmoid to logits before thresholding
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        compute_time += time.time() - compute_start
        
        # Update progress bar
        if batch_idx % 10 == 0:  # Update every 10 batches
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%',
                'data%': f'{100*data_time/(data_time+compute_time):.1f}'
            })
        
        batch_start = time.time()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # Print timing summary
    total_time = data_time + compute_time
    print(f"\n  Time breakdown: Data loading={data_time:.1f}s ({100*data_time/total_time:.1f}%), "
          f"Training={compute_time:.1f}s ({100*compute_time/total_time:.1f}%)")
    if data_time > compute_time:
        print(f"  [WARNING] Data loading is the bottleneck! Increase NUM_WORKERS.")
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, use_amp=False):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # For confusion matrix
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            images = batch['images'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass with AMP
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            # Apply sigmoid to logits before thresholding
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Confusion matrix
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            true_negatives += ((predictions == 0) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {
            'TP': true_positives,
            'FP': false_positives,
            'TN': true_negatives,
            'FN': false_negatives
        }
    }
    
    return metrics


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', marker='o')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[OK] Training curves saved to {save_path}")


def rain_detection_collate_fn(batch):
    """
    Collate function for rain detection dataloader.
    Must be defined at module level (not inside main) for multiprocessing to pickle it.
    
    Args:
        batch: List of {'image': tensor, 'label': tensor}
    
    Returns:
        Dict with stacked images and labels
    """
    images = torch.stack([x['image'] for x in batch])
    labels = torch.stack([x['label'] for x in batch])
    
    return {
        'images': images,
        'labels': labels
    }


def main():
    """Main training function"""
    
    print("=" * 80)
    print("Rain Detector Pretraining Script")
    print("Binary Classification: Clean vs Rainy Images")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print(f"\nTraining Configuration:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Mixed precision (FP16): {USE_AMP}")
    print(f"  - DataLoader workers: {NUM_WORKERS}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    if NUM_WORKERS == 0:
        print("\n[WARNING] NUM_WORKERS=0 will cause low GPU utilization!")
        print("GPU will wait idle while CPU loads data.")
        print("Enable workers for better performance.")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load datasets
    print("\n" + "=" * 80)
    print("Step 1: Loading datasets...")
    print("=" * 80)
    ds_train, ds_valid = load_datasets(
        coco_dir=COCO_DIR,
        coco_rain_dir=COCO_RAIN_DIR,
        coco_ratio=COCO_RATIO,
        rain_ratio=RAIN_RATIO,
        seed=SEED
    )
    
    # Create rain detection datasets
    print("\n" + "=" * 80)
    print("Step 2: Creating rain detection datasets...")
    print("=" * 80)
    rain_train_dataset, rain_val_dataset = create_rain_detection_datasets(
        ds_train, ds_valid
    )
    
    # Create dataloaders with optimized settings for GPU utilization
    train_loader = DataLoader(
        rain_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=rain_detection_collate_fn,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    val_loader = DataLoader(
        rain_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=rain_detection_collate_fn,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    # Create model
    print("\n" + "=" * 80)
    print("Step 3: Creating rain detector model...")
    print("=" * 80)
    model = RainDetector(pretrained=True, dropout=0.2)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits (safe for AMP)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("Step 4: Training rain detector...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=USE_AMP
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp=USE_AMP)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {100*val_metrics['accuracy']:.2f}%")
        print(f"  Precision: {100*val_metrics['precision']:.2f}% | Recall: {100*val_metrics['recall']:.2f}%")
        print(f"  F1 Score: {100*val_metrics['f1']:.2f}%")
        
        cm = val_metrics['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {cm['TP']} | FP: {cm['FP']}")
        print(f"    FN: {cm['FN']} | TN: {cm['TN']}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"\n  [OK] Best model saved! (Val Loss: {best_val_loss:.4f}, Val Acc: {100*best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"\n  Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n[OK] Early stopping triggered after {epoch} epochs")
            break
    
    # Plot training curves
    print("\n" + "=" * 80)
    print("Step 5: Plotting training curves...")
    print("=" * 80)
    plot_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        f"{OUTPUT_DIR}/training_curves.png"
    )
    
    # Final summary
    print("\n" + "=" * 80)
    print("[OK] Rain Detector Pretraining Completed!")
    print("=" * 80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {100*best_val_acc:.2f}%")
    print(f"\nModel saved to: {CHECKPOINT_PATH}")
    print(f"Training curves: {OUTPUT_DIR}/training_curves.png")
    print("\nNext step: Use this pretrained rain detector in conditional model")
    print(f"  → python Training_conditional.py")


if __name__ == '__main__':
    # Windows multiprocessing requires this guard
    main()
