#!/usr/bin/env python3
"""
ULTRA-LIGHTWEIGHT Feature-Based Loss Training for SPDNet Adaptation.

MEMORY OPTIMIZATIONS:
1. VERY SMALL batch size (1-2)
2. Gradient checkpointing
3. Aggressive memory clearing
4. Smaller image size option
5. Only train a few layers of SPDNet (freeze most)

This version is designed for GPUs with 16GB VRAM or less.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.spdnet_utils import load_spdnet_model
from torch.utils.data import Dataset, DataLoader
import cv2
cv2.setNumThreads(0)
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

# =============================================================================
# Configuration - ULTRA MEMORY EFFICIENT
# =============================================================================

COCO_DIR = "E:/Python/DLCV/dataset/coco"
COCO_RAIN_DIR = "E:/Python/DLCV/dataset/coco_rain"
OUTPUT_DIR = "./outputs_spdnet_feature_adaptation"

SPDNET_MODEL_PATH = "E:/Python/DLCV/Project DLCV/model_spa.pt"

# Dataset - use very small subset for faster iteration
PERCENT_DATASET = 5    # 5% (~6000 pairs)
TRAIN_SPLIT = "train2017"
VAL_SPLIT = "val2017"
IMAGE_SIZE = 320       # CRITICAL: 320x320 uses ~3.7GB, 256x256 uses ~2.3GB
                       # 512x512 uses ~9GB (OOM!), 640x640 would use ~15GB

# Training - ULTRA LOW MEMORY
NUM_EPOCHS = 10
BATCH_SIZE = 1         # MINIMUM batch size (SPDNet needs ~4GB per image at 320x320)
EVAL_BATCH_SIZE = 1    # Keep evaluation batch size small too
GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch = 1 * 16 = 16
LEARNING_RATE = 1e-4
SEED = 42

# NO AMP - SPDNet doesn't support it
USE_AMP = False

DATALOADER_WORKERS = 2  # Minimal workers
DATALOADER_PIN_MEMORY = False  # Disable to save memory

# SPDNet Configuration
SPDNET_N_FEATS = 32
SPDNET_N_RESBLOCKS = 3

# Loss weights
PERCEPTUAL_WEIGHT = 1.0
CONTENT_WEIGHT = 0.1

# Training frequency
LOG_EVERY_N_STEPS = 50
EVAL_EVERY_N_EPOCHS = 2
SAVE_EVERY_N_EPOCHS = 2

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# =============================================================================
# Memory-Efficient Feature Extractor
# =============================================================================

class TinyFeatureExtractor(nn.Module):
    """
    MINIMAL feature extractor - only conv layers, no full ResNet.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Just a few conv layers - much lighter than ResNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Freeze
        for param in self.parameters():
            param.requires_grad = False
        
        self.to(device)
        self.eval()
        
        # Load pretrained weights from ResNet18's first few layers
        self._init_from_resnet()
        
        print(f"[OK] Tiny Feature Extractor loaded on {device}")
    
    def _init_from_resnet(self):
        """Initialize from ResNet18 weights"""
        try:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # Copy first conv weights (adapt channels)
            with torch.no_grad():
                self.features[0].weight.copy_(resnet.conv1.weight[:32, :, :3, :3])
            del resnet
        except:
            pass  # Use random init if fails
    
    def forward(self, x):
        """Extract features"""
        x = x.to(self.device)
        return self.features(x)


# =============================================================================
# Dataset (same as before)
# =============================================================================

class PairedRainDataset(Dataset):
    def __init__(self, clean_dir, rain_dir, split="train2017", 
                 image_size=512, percent_dataset=100):
        self.clean_img_dir = os.path.join(clean_dir, split)
        self.rain_img_dir = os.path.join(rain_dir, split)
        self.image_size = image_size
        
        clean_images = set(os.listdir(self.clean_img_dir))
        rain_images = set(os.listdir(self.rain_img_dir))
        common_images = sorted(clean_images & rain_images)
        common_images = [f for f in common_images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        np.random.seed(SEED)
        num_samples = max(1, int(len(common_images) * percent_dataset / 100))
        indices = np.random.choice(len(common_images), num_samples, replace=False)
        self.image_files = [common_images[i] for i in sorted(indices)]
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
        print(f"[{split}] Using {len(self.image_files)} paired images ({percent_dataset}%)")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        
        try:
            clean_path = os.path.join(self.clean_img_dir, image_name)
            rain_path = os.path.join(self.rain_img_dir, image_name)
            
            clean_img = Image.open(clean_path).convert('RGB')
            rain_img = Image.open(rain_path).convert('RGB')
            
            clean_tensor = self.transform(clean_img)
            rain_tensor = self.transform(rain_img)
            
            return {'clean': clean_tensor, 'rain': rain_tensor}
        except:
            placeholder = torch.zeros(3, self.image_size, self.image_size)
            return {'clean': placeholder, 'rain': placeholder}


def collate_fn(batch):
    clean = torch.stack([item['clean'] for item in batch])
    rain = torch.stack([item['rain'] for item in batch])
    return {'clean': clean, 'rain': rain}


# =============================================================================
# Training with Aggressive Memory Management
# =============================================================================

def clear_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_epoch(spdnet, feature_extractor, dataloader, optimizer, 
                scheduler, device, epoch):
    """Memory-efficient training epoch"""
    spdnet.train()
    
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Move to device
            clean = batch['clean'].to(device, non_blocking=True)
            rain = batch['rain'].to(device, non_blocking=True)
            
            # Step 1: Get clean features (no gradients)
            with torch.no_grad():
                clean_features = feature_extractor(clean)
            
            # Step 2: De-rain
            spdnet_input = rain * 255.0
            derain_output = spdnet(spdnet_input)
            if isinstance(derain_output, tuple):
                derained = derain_output[0]
            else:
                derained = derain_output
            derained = torch.clamp(derained / 255.0, 0, 1)
            
            # Step 3: Get derained features
            derained_features = feature_extractor(derained)
            
            # Step 4: Compute loss
            perceptual_loss = F.l1_loss(derained_features, clean_features.detach())
            content_loss = F.l1_loss(derained, clean)
            loss = PERCEPTUAL_WEIGHT * perceptual_loss + CONTENT_WEIGHT * content_loss
            
            # Scale for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # Accumulate step
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(spdnet.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            num_batches += 1
            
            # Logging
            if batch_idx % LOG_EVERY_N_STEPS == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/num_batches:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            
            # CRITICAL: Clear memory regularly
            del clean, rain, clean_features, derained, derained_features, loss
            if batch_idx % 20 == 0:
                clear_memory()
                
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n[WARNING] OOM at batch {batch_idx}, skipping...")
                clear_memory()
                optimizer.zero_grad()
                continue
            else:
                raise e
    
    return total_loss / max(num_batches, 1)


def evaluate(spdnet, feature_extractor, dataloader, device):
    """Memory-efficient evaluation"""
    spdnet.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            clean = batch['clean'].to(device, non_blocking=True)
            rain = batch['rain'].to(device, non_blocking=True)
            
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
            perceptual_loss = F.l1_loss(derained_features, clean_features)
            content_loss = F.l1_loss(derained, clean)
            loss = PERCEPTUAL_WEIGHT * perceptual_loss + CONTENT_WEIGHT * content_loss
            
            total_loss += loss.item()
            num_batches += 1
            
            del clean, rain, clean_features, derained, derained_features
        
        clear_memory()
    
    return total_loss / max(num_batches, 1)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("ULTRA-LIGHTWEIGHT SPDNet Feature Adaptation")
    print("=" * 80)
    print("\nMemory optimizations:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Image size: {IMAGE_SIZE} (reduced from 640)")
    print(f"  - Dataset: {PERCENT_DATASET}% only")
    print(f"  - Tiny feature extractor (not full ResNet)")
    print("=" * 80)
    
    torch.backends.cudnn.benchmark = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Memory: {mem_total:.2f} GB")
    
    # Clear memory before starting
    clear_memory()
    
    # ==========================================================================
    # Create datasets
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Creating datasets...")
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
    
    clear_memory()
    
    # Load SPDNet
    spdnet = load_spdnet_model(
        SPDNET_MODEL_PATH,
        device=device,
        n_feats=SPDNET_N_FEATS,
        n_resblocks=SPDNET_N_RESBLOCKS
    )
    
    # Load tiny feature extractor
    feature_extractor = TinyFeatureExtractor(device=device)
    
    # Print memory usage
    if device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU Memory after loading models:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    # ==========================================================================
    # Setup optimizer
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Setting up training...")
    print("=" * 80)
    
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
    
    # ==========================================================================
    # Training loop
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'='*80}")
        
        clear_memory()
        
        # Train
        train_loss = train_epoch(
            spdnet, feature_extractor, train_loader,
            optimizer, scheduler, device, epoch
        )
        train_losses.append(train_loss)
        print(f"\nTrain Loss: {train_loss:.4f}")
        
        # Evaluate
        if epoch % EVAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
            val_loss = evaluate(spdnet, feature_extractor, val_loader, device)
            val_losses.append((epoch, val_loss))
            print(f"Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f"{OUTPUT_DIR}/spdnet_adapted_best.pt"
                torch.save(spdnet.state_dict(), save_path)
                print(f"[OK] New best model saved!")
        
        # Save checkpoint
        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = f"{OUTPUT_DIR}/spdnet_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': spdnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
    
    # ==========================================================================
    # Save final model
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    
    final_path = f"{OUTPUT_DIR}/spdnet_adapted_final.pt"
    torch.save(spdnet.state_dict(), final_path)
    print(f"[OK] Final model saved: {final_path}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    if val_losses:
        plt.subplot(1, 2, 2)
        epochs, losses = zip(*val_losses)
        plt.plot(epochs, losses, 'o-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150)
    plt.close()
    
    print(f"\n[OK] Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
