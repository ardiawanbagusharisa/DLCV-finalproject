#!/usr/bin/env python3
"""
FIXED Feature-Based Loss Training for SPDNet Adaptation.

FIXES:
1. AMP DISABLED - SPDNet's pytorch_wavelets doesn't support FP16
2. LIGHTWEIGHT FEATURE EXTRACTION - Use ResNet backbone only, not full RT-DETR
3. MEMORY OPTIMIZED - Detach features, clear cache frequently
4. SIMPLER LOSS - Direct feature matching without complex structures

TRAINING FLOW:
    Clean Image ───────────→ ResNet Backbone (frozen) ───→ Clean Features
                                                                ↓
                                                           Feature Loss (L1)
                                                                ↑
    Rainy Image → SPDNet → De-rained → ResNet Backbone → Derained Features
                     ↑                                         
                (gradients flow back)                                    
"""

import warnings
warnings.filterwarnings('ignore', message='Unable to import Axes3D')
warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import gc

# Ensure local 'utils' package is found first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.spdnet_utils import load_spdnet_model
from torch.utils.data import Dataset, DataLoader
import cv2
cv2.setNumThreads(0)
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

# =============================================================================
# Configuration
# =============================================================================

# Dataset paths
COCO_DIR = "E:/Python/DLCV/dataset/coco"
COCO_RAIN_DIR = "E:/Python/DLCV/dataset/coco_rain"
OUTPUT_DIR = "./outputs_spdnet_feature_adaptation"

# Model paths  
SPDNET_MODEL_PATH = "E:/Python/DLCV/Project DLCV/model_spa.pt"

# Dataset configuration
PERCENT_DATASET = 5    # Use 5% of paired images (~6000 pairs) - faster iteration
TRAIN_SPLIT = "train2017"
VAL_SPLIT = "val2017"
IMAGE_SIZE = 640       # RT-DETR input size

# Training configuration - OPTIMIZED FOR SPEED AND MEMORY
NUM_EPOCHS = 10
BATCH_SIZE = 4         # Keep small - SPDNet is memory intensive
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 4 * 4 = 16
LEARNING_RATE = 1e-4
SEED = 42

# CRITICAL: AMP must be DISABLED for SPDNet (wavelets don't support FP16)
USE_AMP = False

DATALOADER_WORKERS = 4  # Reduced to save memory
DATALOADER_PIN_MEMORY = True

# SPDNet Configuration
SPDNET_N_FEATS = 32
SPDNET_N_RESBLOCKS = 3

# Loss configuration
PERCEPTUAL_WEIGHT = 1.0   # Weight for feature matching loss
CONTENT_WEIGHT = 0.1      # Weight for pixel-level loss

# Training frequency
LOG_EVERY_N_STEPS = 25
EVAL_EVERY_N_EPOCHS = 2
SAVE_EVERY_N_EPOCHS = 2

# Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# =============================================================================
# Lightweight Feature Extractor (ResNet18 only, NOT full RT-DETR)
# =============================================================================

class LightweightFeatureExtractor(nn.Module):
    """
    Lightweight feature extractor using ResNet18 backbone.
    
    Much faster and less memory than full RT-DETR!
    ResNet18 is pretrained on ImageNet, captures similar features.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Load pretrained ResNet18 (lightweight)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Extract layers up to layer3 (before avgpool)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Move to device FIRST
        self.to(device)
        
        # Register buffers AFTER moving to device (they auto-move with model)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        
        self.eval()
        
        print(f"[OK] Lightweight Feature Extractor loaded (ResNet18) on {device}")
    
    def normalize(self, x):
        """Apply ImageNet normalization - ensure same device"""
        # mean and std are registered buffers, should be on same device as model
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def forward(self, x):
        """
        Extract multi-scale features.
        
        Args:
            x: Input images [B, C, H, W] in [0, 1] range
            
        Returns:
            List of feature maps at different scales
        """
        # Ensure input is on same device as model
        x = x.to(self.device)
        
        # Normalize
        x = self.normalize(x)
        
        # Extract features
        x0 = self.layer0(x)   # [B, 64, H/4, W/4]
        x1 = self.layer1(x0)  # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)  # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)  # [B, 256, H/16, W/16]
        
        return [x1, x2, x3]


# =============================================================================
# Paired Clean/Rainy Dataset (Simplified)
# =============================================================================

class PairedRainDataset(Dataset):
    """
    Dataset that returns paired (clean, rainy) images.
    """
    
    def __init__(self, clean_dir, rain_dir, split="train2017", 
                 image_size=640, percent_dataset=100):
        """
        Args:
            clean_dir: Path to clean COCO directory
            rain_dir: Path to rainy COCO directory  
            split: "train2017" or "val2017"
            image_size: Target image size
            percent_dataset: Percentage of dataset to use
        """
        self.clean_img_dir = os.path.join(clean_dir, split)
        self.rain_img_dir = os.path.join(rain_dir, split)
        self.image_size = image_size
        
        # Find common images
        clean_images = set(os.listdir(self.clean_img_dir))
        rain_images = set(os.listdir(self.rain_img_dir))
        common_images = sorted(clean_images & rain_images)
        
        # Filter to only image files
        common_images = [f for f in common_images 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Subsample
        np.random.seed(SEED)
        num_samples = int(len(common_images) * percent_dataset / 100)
        indices = np.random.choice(len(common_images), num_samples, replace=False)
        self.image_files = [common_images[i] for i in sorted(indices)]
        
        # Simple transform
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # Converts to [0, 1]
        ])
        
        print(f"[{split}] Found {len(common_images)} paired images, using {len(self.image_files)} ({percent_dataset}%)")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        
        try:
            # Load images
            clean_path = os.path.join(self.clean_img_dir, image_name)
            rain_path = os.path.join(self.rain_img_dir, image_name)
            
            clean_img = Image.open(clean_path).convert('RGB')
            rain_img = Image.open(rain_path).convert('RGB')
            
            # Transform
            clean_tensor = self.transform(clean_img)
            rain_tensor = self.transform(rain_img)
            
            return {
                'clean': clean_tensor,
                'rain': rain_tensor,
                'name': image_name
            }
        except Exception as e:
            # Return placeholder on error
            placeholder = torch.zeros(3, self.image_size, self.image_size)
            return {
                'clean': placeholder,
                'rain': placeholder,
                'name': 'error'
            }


def collate_fn(batch):
    """Simple collate function"""
    clean = torch.stack([item['clean'] for item in batch])
    rain = torch.stack([item['rain'] for item in batch])
    names = [item['name'] for item in batch]
    return {'clean': clean, 'rain': rain, 'names': names}


# =============================================================================
# Feature-Based Loss (Simplified)
# =============================================================================

class FeatureLoss(nn.Module):
    """
    Simple feature matching loss.
    """
    
    def __init__(self, perceptual_weight=1.0, content_weight=0.1):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.content_weight = content_weight
    
    def forward(self, derained_features, clean_features, derained_img, clean_img):
        """
        Compute combined loss.
        """
        # Feature matching loss (perceptual)
        perceptual_loss = 0.0
        for df, cf in zip(derained_features, clean_features):
            perceptual_loss += F.l1_loss(df, cf.detach())
        perceptual_loss /= len(derained_features)
        
        # Content loss (pixel-level)
        content_loss = F.l1_loss(derained_img, clean_img)
        
        # Total
        total = self.perceptual_weight * perceptual_loss + self.content_weight * content_loss
        
        return total, {
            'perceptual': perceptual_loss.item(),
            'content': content_loss.item(),
            'total': total.item()
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(spdnet, feature_extractor, loss_fn, dataloader, optimizer, 
                scheduler, device, epoch, log_every=25):
    """
    Train one epoch.
    """
    spdnet.train()
    
    total_loss = 0.0
    total_perceptual = 0.0
    total_content = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        clean = batch['clean'].to(device)
        rain = batch['rain'].to(device)
        
        # Step 1: Get clean features (no gradients - frozen extractor)
        with torch.no_grad():
            clean_features = feature_extractor(clean)
        
        # Step 2: De-rain (WITH gradients)
        spdnet_input = rain * 255.0
        derain_output = spdnet(spdnet_input)
        if isinstance(derain_output, tuple):
            derained = derain_output[0]
        else:
            derained = derain_output
        derained = torch.clamp(derained / 255.0, 0, 1)
        
        # Step 3: Get derained features (gradients flow through derained)
        derained_features = feature_extractor(derained)
        
        # Step 4: Compute loss
        loss, loss_dict = loss_fn(derained_features, clean_features, derained, clean)
        
        # Scale loss for gradient accumulation
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(spdnet.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss_dict['total']
        total_perceptual += loss_dict['perceptual']
        total_content += loss_dict['content']
        num_batches += 1
        
        # Logging
        if batch_idx % log_every == 0:
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'perc': f'{total_perceptual/num_batches:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # Memory cleanup every 100 batches
        if batch_idx % 100 == 0:
            del clean_features, derained_features, derained, loss
            torch.cuda.empty_cache()
    
    return {
        'loss': total_loss / num_batches,
        'perceptual': total_perceptual / num_batches,
        'content': total_content / num_batches
    }


def evaluate(spdnet, feature_extractor, loss_fn, dataloader, device):
    """
    Evaluate on validation set.
    """
    spdnet.eval()
    
    total_loss = 0.0
    total_perceptual = 0.0
    total_content = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            clean = batch['clean'].to(device)
            rain = batch['rain'].to(device)
            
            # Get clean features
            clean_features = feature_extractor(clean)
            
            # De-rain
            spdnet_input = rain * 255.0
            derain_output = spdnet(spdnet_input)
            if isinstance(derain_output, tuple):
                derained = derain_output[0]
            else:
                derained = derain_output
            derained = torch.clamp(derained / 255.0, 0, 1)
            
            # Get derained features
            derained_features = feature_extractor(derained)
            
            # Compute loss
            _, loss_dict = loss_fn(derained_features, clean_features, derained, clean)
            
            total_loss += loss_dict['total']
            total_perceptual += loss_dict['perceptual']
            total_content += loss_dict['content']
            num_batches += 1
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'perceptual': total_perceptual / max(num_batches, 1),
        'content': total_content / max(num_batches, 1)
    }


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    print("=" * 80)
    print("FIXED Feature-Based SPDNet Adaptation Training")
    print("=" * 80)
    print("\nKey fixes applied:")
    print("  - AMP DISABLED (SPDNet wavelets don't support FP16)")
    print("  - Lightweight ResNet18 feature extractor (not full RT-DETR)")
    print("  - Memory optimized with frequent cache clearing")
    print("=" * 80)
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Memory: {mem_total:.2f} GB")
    
    # ==========================================================================
    # Create datasets
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Creating paired datasets...")
    print("=" * 80)
    
    train_dataset = PairedRainDataset(
        clean_dir=COCO_DIR,
        rain_dir=COCO_RAIN_DIR,
        split=TRAIN_SPLIT,
        image_size=IMAGE_SIZE,
        percent_dataset=PERCENT_DATASET
    )
    
    val_dataset = PairedRainDataset(
        clean_dir=COCO_DIR,
        rain_dir=COCO_RAIN_DIR,
        split=VAL_SPLIT,
        image_size=IMAGE_SIZE,
        percent_dataset=PERCENT_DATASET
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=DATALOADER_WORKERS,
        collate_fn=collate_fn,
        pin_memory=DATALOADER_PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        collate_fn=collate_fn,
        pin_memory=DATALOADER_PIN_MEMORY
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # ==========================================================================
    # Load models
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Loading models...")
    print("=" * 80)
    
    # Load SPDNet (trainable)
    spdnet = load_spdnet_model(
        SPDNET_MODEL_PATH,
        device=device,
        n_feats=SPDNET_N_FEATS,
        n_resblocks=SPDNET_N_RESBLOCKS
    )
    
    # Load lightweight feature extractor
    feature_extractor = LightweightFeatureExtractor(device=device)
    
    # Print parameter counts
    spdnet_params = sum(p.numel() for p in spdnet.parameters())
    spdnet_trainable = sum(p.numel() for p in spdnet.parameters() if p.requires_grad)
    fe_params = sum(p.numel() for p in feature_extractor.parameters())
    
    print(f"\nParameter Summary:")
    print(f"  SPDNet: {spdnet_trainable:,}/{spdnet_params:,} trainable")
    print(f"  Feature Extractor (ResNet18): 0/{fe_params:,} trainable (FROZEN)")
    
    # ==========================================================================
    # Setup loss and optimizer
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Setting up training...")
    print("=" * 80)
    
    loss_fn = FeatureLoss(
        perceptual_weight=PERCEPTUAL_WEIGHT,
        content_weight=CONTENT_WEIGHT
    )
    
    optimizer = torch.optim.AdamW(
        spdnet.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    print(f"\nConfiguration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Total steps: {total_steps}")
    print(f"  AMP: {USE_AMP} (disabled for SPDNet compatibility)")
    
    # ==========================================================================
    # Training loop
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # Clear memory before each epoch
        gc.collect()
        torch.cuda.empty_cache()
        
        # Train
        train_metrics = train_epoch(
            spdnet, feature_extractor, loss_fn, train_loader,
            optimizer, scheduler, device, epoch, LOG_EVERY_N_STEPS
        )
        train_history.append(train_metrics)
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Perceptual: {train_metrics['perceptual']:.4f}, "
              f"Content: {train_metrics['content']:.4f}")
        
        # Evaluate periodically
        if epoch % EVAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
            val_metrics = evaluate(spdnet, feature_extractor, loss_fn, val_loader, device)
            val_history.append({'epoch': epoch, **val_metrics})
            
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Perceptual: {val_metrics['perceptual']:.4f}, "
                  f"Content: {val_metrics['content']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_path = f"{OUTPUT_DIR}/spdnet_adapted_best.pt"
                torch.save(spdnet.state_dict(), save_path)
                print(f"[OK] New best model saved! Loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = f"{OUTPUT_DIR}/spdnet_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': spdnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"[OK] Checkpoint saved: {checkpoint_path}")
    
    # ==========================================================================
    # Save final model
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    
    final_path = f"{OUTPUT_DIR}/spdnet_adapted_final.pt"
    torch.save(spdnet.state_dict(), final_path)
    print(f"[OK] Final model saved: {final_path}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot([m['loss'] for m in train_history], label='Train')
    if val_history:
        val_epochs = [m['epoch'] for m in val_history]
        axes[0].plot([e-1 for e in val_epochs], [m['loss'] for m in val_history], 'o-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Perceptual loss
    axes[1].plot([m['perceptual'] for m in train_history], label='Train')
    if val_history:
        axes[1].plot([e-1 for e in val_epochs], [m['perceptual'] for m in val_history], 'o-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Perceptual Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # Content loss
    axes[2].plot([m['content'] for m in train_history], label='Train')
    if val_history:
        axes[2].plot([e-1 for e in val_epochs], [m['content'] for m in val_history], 'o-', label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Content Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150)
    plt.close()
    
    print(f"\n[OK] Training curves saved")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"\nOutput files:")
    print(f"  - Best model: {OUTPUT_DIR}/spdnet_adapted_best.pt")
    print(f"  - Final model: {OUTPUT_DIR}/spdnet_adapted_final.pt")
    print(f"  - Training curves: {OUTPUT_DIR}/training_curves.png")
    print(f"\nNext steps:")
    print(f"  1. Run Eval_conditional.py with the adapted SPDNet")
    print(f"  2. Compare detection mAP vs original SPDNet")
    print("=" * 80)


if __name__ == "__main__":
    main()
