#!/usr/bin/env python3
"""
Feature-Based Loss Training for SPDNet Adaptation.

STRATEGY: Train SPDNet to produce images whose RT-DETR backbone features
          match those of clean (non-rainy) images.

WHY THIS WORKS:
- RT-DETR backbone extracts visual features (edges, textures, objects)
- If de-rained features ≈ clean features → RT-DETR will detect similarly
- Only backbone forward pass needed (no Hungarian matching, no decoder)
- ~5x faster than full detection loss

REQUIREMENTS:
- Paired clean/rainy images (COCO + COCO_rain with same filenames)
- Pretrained RT-DETR backbone (frozen)
- Pretrained SPDNet (trainable)

TRAINING FLOW:
    Clean Image ─────────────────────────┐
         ↓                               ↓
    RT-DETR Backbone (frozen)    [no gradients]
         ↓                               
    Clean Features ──────────────┐       
                                 │       
    Rainy Image                  │       
         ↓                       │       
    SPDNet (trainable)           │       
         ↓                       │       
    De-rained Image              │       
         ↓                       │       
    RT-DETR Backbone (frozen)    │       
         ↓                       │       
    De-rained Features ──────────┼──→ Feature Loss (L1/MSE)
                                        ↓
                                   Backprop → SPDNet
"""

import warnings
warnings.filterwarnings('ignore', message='Unable to import Axes3D')

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Ensure local 'utils' package is found first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.spdnet_utils import load_spdnet_model
from utils.model_utils import load_model_and_processor
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
cv2.setNumThreads(0)

# =============================================================================
# Configuration
# =============================================================================

# Dataset paths
COCO_DIR = "E:/Python/DLCV/dataset/coco"
COCO_RAIN_DIR = "E:/Python/DLCV/dataset/coco_rain"
OUTPUT_DIR = "./outputs_spdnet_feature_adaptation"

# Model paths  
SPDNET_MODEL_PATH = "E:/Python/DLCV/Project DLCV/model_spa.pt"
RTDETR_MODEL_NAME = "PekingU/rtdetr_r18vd"
NUM_LABELS = 80

# Dataset configuration
PERCENT_DATASET = 10   # Use 10% of paired images (~11,800 pairs)
TRAIN_SPLIT = "train2017"
VAL_SPLIT = "val2017"

# Training configuration - OPTIMIZED FOR SPEED
NUM_EPOCHS = 10
BATCH_SIZE = 4         # Can be larger now (no detection loss backward)
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 8 * 2 = 16
LEARNING_RATE = 5e-5   # Slightly lower for stability
SEED = 42

# FP16 for faster training (now possible since no full RT-DETR backward)
USE_AMP = True  # Automatic Mixed Precision

DATALOADER_WORKERS = 8
DATALOADER_PIN_MEMORY = True

# SPDNet Configuration
SPDNET_N_FEATS = 32
SPDNET_N_RESBLOCKS = 3

# Loss configuration
FEATURE_LOSS_TYPE = "l1"  # "l1", "mse", or "cosine"
PERCEPTUAL_WEIGHT = 1.0   # Weight for feature matching loss
CONTENT_WEIGHT = 0.1      # Weight for pixel-level loss (prevents color shift)

# Training frequency
LOG_EVERY_N_STEPS = 50
EVAL_EVERY_N_EPOCHS = 2
SAVE_EVERY_N_EPOCHS = 2

# Early stopping
EARLY_STOPPING_PATIENCE = 5

# Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# =============================================================================
# Paired Clean/Rainy Dataset
# =============================================================================

class PairedRainDataset(Dataset):
    """
    Dataset that returns paired (clean, rainy) images.
    
    Assumes COCO and COCO_rain have the same image filenames.
    """
    
    def __init__(self, clean_dir, rain_dir, split="train2017", 
                 processor=None, transform=None, percent_dataset=100):
        """
        Args:
            clean_dir: Path to clean COCO directory
            rain_dir: Path to rainy COCO directory  
            split: "train2017" or "val2017"
            processor: RT-DETR image processor
            transform: Albumentations transform (applied to both)
            percent_dataset: Percentage of dataset to use
        """
        self.clean_img_dir = os.path.join(clean_dir, split)
        self.rain_img_dir = os.path.join(rain_dir, split)
        self.processor = processor
        self.transform = transform
        
        # Find common images (exist in both directories)
        clean_images = set(os.listdir(self.clean_img_dir))
        rain_images = set(os.listdir(self.rain_img_dir))
        common_images = sorted(clean_images & rain_images)
        
        # Filter to only image files
        common_images = [f for f in common_images 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Subsample if requested
        np.random.seed(SEED)
        num_samples = int(len(common_images) * percent_dataset / 100)
        indices = np.random.choice(len(common_images), num_samples, replace=False)
        self.image_files = [common_images[i] for i in sorted(indices)]
        
        print(f"[{split}] Found {len(common_images)} paired images, using {len(self.image_files)} ({percent_dataset}%)")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
            - clean_pixel_values: Processed clean image [C, H, W]
            - rain_pixel_values: Processed rainy image [C, H, W]
            - image_name: Filename for debugging
        """
        image_name = self.image_files[idx]
        
        # Load images
        clean_path = os.path.join(self.clean_img_dir, image_name)
        rain_path = os.path.join(self.rain_img_dir, image_name)
        
        clean_img = cv2.imread(clean_path)
        rain_img = cv2.imread(rain_path)
        
        if clean_img is None or rain_img is None:
            # Return a placeholder on error
            return self._get_placeholder()
        
        # Convert BGR to RGB
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        rain_img = cv2.cvtColor(rain_img, cv2.COLOR_BGR2RGB)
        
        # Apply same transform to both (e.g., resize, flip)
        if self.transform:
            # Need to apply same random transform to both
            # Use replay feature of albumentations
            transformed_clean = self.transform(image=clean_img)
            clean_img = transformed_clean['image']
            
            # Apply same transform using replay
            if hasattr(transformed_clean, 'replay'):
                rain_img = A.ReplayCompose.replay(
                    transformed_clean['replay'], image=rain_img
                )['image']
            else:
                # Fallback: just apply transform (may differ slightly)
                rain_img = self.transform(image=rain_img)['image']
        
        # Process with RT-DETR processor
        clean_processed = self.processor(images=clean_img, return_tensors="pt")
        rain_processed = self.processor(images=rain_img, return_tensors="pt")
        
        return {
            'clean_pixel_values': clean_processed['pixel_values'].squeeze(0),
            'rain_pixel_values': rain_processed['pixel_values'].squeeze(0),
            'image_name': image_name
        }
    
    def _get_placeholder(self):
        """Return placeholder for failed loads"""
        placeholder = torch.zeros(3, 640, 640)
        return {
            'clean_pixel_values': placeholder,
            'rain_pixel_values': placeholder,
            'image_name': 'error'
        }


def collate_paired_fn(batch):
    """Collate function for paired dataset"""
    clean_pixels = torch.stack([item['clean_pixel_values'] for item in batch])
    rain_pixels = torch.stack([item['rain_pixel_values'] for item in batch])
    names = [item['image_name'] for item in batch]
    
    return {
        'clean_pixel_values': clean_pixels,
        'rain_pixel_values': rain_pixels,
        'image_names': names
    }


# =============================================================================
# Feature Extraction Module
# =============================================================================

class RTDETRFeatureExtractor(nn.Module):
    """
    Extract backbone features from RT-DETR.
    
    RT-DETR uses a ResNet backbone followed by transformer encoder.
    We extract features from the backbone (before transformer).
    """
    
    def __init__(self, rtdetr_model):
        super().__init__()
        self.model = rtdetr_model
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Hook to capture backbone features
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features"""
        
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Register hook on backbone output
        # RT-DETR architecture: model.model.backbone
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'backbone'):
            self.model.model.backbone.register_forward_hook(hook_fn('backbone'))
    
    def forward(self, pixel_values):
        """
        Extract features from RT-DETR backbone.
        
        Args:
            pixel_values: Input images [B, C, H, W] in [0, 1] range
            
        Returns:
            Dictionary of feature maps from different levels
        """
        self.features = {}  # Clear previous features
        
        with torch.no_grad():
            # Run forward pass (features captured by hooks)
            _ = self.model(pixel_values=pixel_values)
        
        return self.features
    
    def extract_multi_scale_features(self, pixel_values):
        """
        Extract multi-scale features for more comprehensive matching.
        
        Returns:
            List of feature tensors at different scales
        """
        self.features = {}
        
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        
        # Get encoder hidden states (multi-scale features)
        if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
            return list(outputs.encoder_hidden_states)
        
        # Fallback: use backbone features
        if 'backbone' in self.features:
            return [self.features['backbone']]
        
        # Last resort: use last hidden state
        if hasattr(outputs, 'last_hidden_state'):
            return [outputs.last_hidden_state]
        
        return []


# =============================================================================
# Feature-Based Loss Module
# =============================================================================

class FeatureBasedLoss(nn.Module):
    """
    Compute feature-based loss between de-rained and clean images.
    """
    
    def __init__(self, loss_type="l1", perceptual_weight=1.0, content_weight=0.1):
        super().__init__()
        self.loss_type = loss_type
        self.perceptual_weight = perceptual_weight
        self.content_weight = content_weight
    
    def forward(self, derained_features, clean_features, 
                derained_image=None, clean_image=None):
        """
        Compute combined loss.
        
        Args:
            derained_features: Features from de-rained image (list of tensors)
            clean_features: Features from clean image (list of tensors)
            derained_image: De-rained image tensor (optional, for content loss)
            clean_image: Clean image tensor (optional, for content loss)
            
        Returns:
            Total loss
        """
        # Feature matching loss
        perceptual_loss = self._compute_feature_loss(derained_features, clean_features)
        
        # Content loss (pixel-level)
        content_loss = 0.0
        if derained_image is not None and clean_image is not None:
            content_loss = F.l1_loss(derained_image, clean_image)
        
        total_loss = (self.perceptual_weight * perceptual_loss + 
                     self.content_weight * content_loss)
        
        return total_loss, {
            'perceptual_loss': perceptual_loss.item(),
            'content_loss': content_loss.item() if isinstance(content_loss, torch.Tensor) else content_loss,
            'total_loss': total_loss.item()
        }
    
    def _compute_feature_loss(self, pred_features, target_features):
        """Compute feature matching loss"""
        if not pred_features or not target_features:
            return torch.tensor(0.0, requires_grad=True)
        
        total_loss = 0.0
        num_features = min(len(pred_features), len(target_features))
        
        for i in range(num_features):
            pred = pred_features[i]
            target = target_features[i]
            
            # Handle different feature formats
            if isinstance(pred, (list, tuple)):
                pred = pred[0] if pred else torch.zeros(1)
            if isinstance(target, (list, tuple)):
                target = target[0] if target else torch.zeros(1)
            
            # Ensure same shape
            if pred.shape != target.shape:
                # Resize pred to match target
                pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
            
            # Compute loss
            if self.loss_type == "l1":
                total_loss += F.l1_loss(pred, target)
            elif self.loss_type == "mse":
                total_loss += F.mse_loss(pred, target)
            elif self.loss_type == "cosine":
                # Flatten and compute cosine similarity
                pred_flat = pred.view(pred.size(0), -1)
                target_flat = target.view(target.size(0), -1)
                cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1)
                total_loss += (1 - cosine_sim.mean())  # Convert similarity to loss
        
        return total_loss / max(num_features, 1)


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(spdnet, feature_extractor, loss_fn, dataloader, optimizer, 
                scheduler, device, epoch, scaler=None, log_every_n_steps=50):
    """
    Train one epoch with feature-based loss.
    """
    spdnet.train()
    feature_extractor.eval()
    
    total_loss = 0.0
    total_perceptual = 0.0
    total_content = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        clean_pixels = batch['clean_pixel_values'].to(device)
        rain_pixels = batch['rain_pixel_values'].to(device)
        
        # Use AMP if available
        use_amp = scaler is not None
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Step 1: Extract clean features (no gradients needed)
            clean_features = feature_extractor.extract_multi_scale_features(clean_pixels)
            
            # Step 2: De-rain the rainy images
            spdnet_input = rain_pixels * 255.0
            derain_outputs = spdnet(spdnet_input)
            if isinstance(derain_outputs, tuple):
                derained = derain_outputs[0]
            else:
                derained = derain_outputs
            derained = torch.clamp(derained / 255.0, 0, 1)
            
            # Step 3: Extract features from de-rained images (WITH gradients)
            # Need to run through backbone with gradients
            # But feature_extractor has torch.no_grad() - need workaround
            
            # Workaround: Run RT-DETR encoder manually with gradients through derained
            # Actually, we need the features to depend on derained for gradient flow
            # Solution: Use a trainable feature extractor that shares weights
            
            # For now, use a simpler approach: match at image level + feature level
            # Get features by running through the model (gradients flow through derained)
            
            # Actually, the correct approach is:
            # 1. Extract clean features (no_grad) - done above
            # 2. De-rain (with grad) - done above  
            # 3. Extract derained features (with grad through derained)
            
            # Since feature_extractor uses hooks, gradients should flow through derained
            # Let's verify by NOT using torch.no_grad in extract:
            
            derained_features = []
            outputs = feature_extractor.model(pixel_values=derained, output_hidden_states=True)
            if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
                derained_features = list(outputs.encoder_hidden_states)
            elif hasattr(outputs, 'last_hidden_state'):
                derained_features = [outputs.last_hidden_state]
            
            # Step 4: Compute loss
            loss, loss_dict = loss_fn(
                derained_features, clean_features,
                derained, clean_pixels
            )
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(spdnet.parameters(), max_norm=1.0)
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss_dict['total_loss']
        total_perceptual += loss_dict['perceptual_loss']
        total_content += loss_dict['content_loss']
        
        # Logging
        if batch_idx % log_every_n_steps == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'perc': f'{total_perceptual/(batch_idx+1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
    
    return {
        'loss': total_loss / num_batches,
        'perceptual_loss': total_perceptual / num_batches,
        'content_loss': total_content / num_batches
    }


def evaluate(spdnet, feature_extractor, loss_fn, dataloader, device):
    """
    Evaluate feature matching loss on validation set.
    """
    spdnet.eval()
    feature_extractor.eval()
    
    total_loss = 0.0
    total_perceptual = 0.0
    total_content = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            clean_pixels = batch['clean_pixel_values'].to(device)
            rain_pixels = batch['rain_pixel_values'].to(device)
            
            # Extract clean features
            clean_features = feature_extractor.extract_multi_scale_features(clean_pixels)
            
            # De-rain
            spdnet_input = rain_pixels * 255.0
            derain_outputs = spdnet(spdnet_input)
            if isinstance(derain_outputs, tuple):
                derained = derain_outputs[0]
            else:
                derained = derain_outputs
            derained = torch.clamp(derained / 255.0, 0, 1)
            
            # Extract derained features
            outputs = feature_extractor.model(pixel_values=derained, output_hidden_states=True)
            derained_features = []
            if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
                derained_features = list(outputs.encoder_hidden_states)
            elif hasattr(outputs, 'last_hidden_state'):
                derained_features = [outputs.last_hidden_state]
            
            # Compute loss
            _, loss_dict = loss_fn(
                derained_features, clean_features,
                derained, clean_pixels
            )
            
            total_loss += loss_dict['total_loss']
            total_perceptual += loss_dict['perceptual_loss']
            total_content += loss_dict['content_loss']
            num_batches += 1
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'perceptual_loss': total_perceptual / max(num_batches, 1),
        'content_loss': total_content / max(num_batches, 1)
    }


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    print("=" * 80)
    print("Feature-Based SPDNet Adaptation Training")
    print("=" * 80)
    print("\nStrategy: Match RT-DETR backbone features between de-rained and clean images")
    print("          This is ~5x faster than full detection loss!")
    print("=" * 80)
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision('high')
    except:
        pass
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ==========================================================================
    # Create datasets
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Creating paired clean/rainy datasets...")
    print("=" * 80)
    
    # Load processor first
    _, processor = load_model_and_processor(
        model_name=RTDETR_MODEL_NAME,
        num_labels=NUM_LABELS
    )
    
    # Simple transform (only geometric, applied same to both)
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
    ])
    
    train_dataset = PairedRainDataset(
        clean_dir=COCO_DIR,
        rain_dir=COCO_RAIN_DIR,
        split=TRAIN_SPLIT,
        processor=processor,
        transform=train_transform,
        percent_dataset=PERCENT_DATASET
    )
    
    val_dataset = PairedRainDataset(
        clean_dir=COCO_DIR,
        rain_dir=COCO_RAIN_DIR,
        split=VAL_SPLIT,
        processor=processor,
        transform=None,
        percent_dataset=PERCENT_DATASET
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=DATALOADER_WORKERS,
        collate_fn=collate_paired_fn,
        pin_memory=DATALOADER_PIN_MEMORY,
        persistent_workers=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        collate_fn=collate_paired_fn,
        pin_memory=DATALOADER_PIN_MEMORY
    )
    
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
    spdnet.train()
    
    # Load RT-DETR (frozen, for feature extraction)
    rtdetr_model, _ = load_model_and_processor(
        model_name=RTDETR_MODEL_NAME,
        num_labels=NUM_LABELS
    )
    rtdetr_model = rtdetr_model.to(device)
    
    feature_extractor = RTDETRFeatureExtractor(rtdetr_model)
    feature_extractor.eval()
    
    # Print parameter counts
    spdnet_params = sum(p.numel() for p in spdnet.parameters())
    spdnet_trainable = sum(p.numel() for p in spdnet.parameters() if p.requires_grad)
    rtdetr_params = sum(p.numel() for p in rtdetr_model.parameters())
    
    print(f"\nParameter Summary:")
    print(f"  SPDNet: {spdnet_trainable:,}/{spdnet_params:,} trainable")
    print(f"  RT-DETR: 0/{rtdetr_params:,} trainable (FROZEN)")
    
    # ==========================================================================
    # Setup loss and optimizer
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Setting up loss and optimizer...")
    print("=" * 80)
    
    loss_fn = FeatureBasedLoss(
        loss_type=FEATURE_LOSS_TYPE,
        perceptual_weight=PERCEPTUAL_WEIGHT,
        content_weight=CONTENT_WEIGHT
    )
    
    optimizer = torch.optim.AdamW(
        spdnet.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    total_steps = len(train_dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Total steps: {total_steps}")
    print(f"  AMP: {USE_AMP}")
    print(f"  Loss type: {FEATURE_LOSS_TYPE}")
    
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
    patience_counter = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(
            spdnet, feature_extractor, loss_fn, train_dataloader,
            optimizer, scheduler, device, epoch, scaler, LOG_EVERY_N_STEPS
        )
        train_history.append(train_metrics)
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Perceptual: {train_metrics['perceptual_loss']:.4f}, "
              f"Content: {train_metrics['content_loss']:.4f}")
        
        # Evaluate
        if epoch % EVAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
            val_metrics = evaluate(spdnet, feature_extractor, loss_fn, val_dataloader, device)
            val_history.append({'epoch': epoch, **val_metrics})
            
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Perceptual: {val_metrics['perceptual_loss']:.4f}, "
                  f"Content: {val_metrics['content_loss']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                save_path = f"{OUTPUT_DIR}/spdnet_adapted_best.pt"
                torch.save(spdnet.state_dict(), save_path)
                print(f"[OK] New best model saved! Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} eval(s)")
                
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered!")
                    break
        
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
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # ==========================================================================
    # Save final model and plot
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
        axes[0].plot(val_epochs, [m['loss'] for m in val_history], 'o-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Perceptual loss
    axes[1].plot([m['perceptual_loss'] for m in train_history], label='Train')
    if val_history:
        axes[1].plot(val_epochs, [m['perceptual_loss'] for m in val_history], 'o-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Perceptual (Feature) Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # Content loss
    axes[2].plot([m['content_loss'] for m in train_history], label='Train')
    if val_history:
        axes[2].plot(val_epochs, [m['content_loss'] for m in val_history], 'o-', label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Content Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150)
    plt.close()
    
    print(f"\n[OK] Training curves saved to {OUTPUT_DIR}/training_curves.png")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Load adapted SPDNet: {OUTPUT_DIR}/spdnet_adapted_best.pt")
    print(f"  2. Use with conditional model for inference")
    print(f"  3. Run Eval_conditional.py to measure detection mAP")
    print("=" * 80)


if __name__ == "__main__":
    main()
