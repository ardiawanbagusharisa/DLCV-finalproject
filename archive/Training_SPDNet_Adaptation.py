#!/usr/bin/env python3
"""
Training script for SPDNet Adaptation with Frozen RT-DETR.

This script trains ONLY SPDNet to produce detection-friendly de-rained images,
while keeping RT-DETR completely frozen to preserve its performance on clean images.

Key Design Decisions:
    1. RT-DETR is FROZEN throughout training → guarantees no degradation on clean images
    2. Only RAINY images are used for training → SPDNet learns to help RT-DETR detect in rain
    3. FP16 is DISABLED → SPDNet has numerical issues with mixed precision
    4. Detection loss backpropagates through SPDNet → learns detection-friendly de-raining

Training Flow:
    Rainy Image → SPDNet (trainable) → RT-DETR (frozen) → Detection Loss → Update SPDNet

Inference (after training):
    Image → Rain Detector → [If Rainy] → Adapted SPDNet → RT-DETR → Detections
                          → [If Clean] → RT-DETR → Detections
"""

# CRITICAL: Suppress matplotlib warnings before any imports
import warnings
warnings.filterwarnings('ignore', message='Unable to import Axes3D')

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# Ensure local 'utils' package is found first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utility modules
from utils.data_utils import (
    load_datasets, 
    create_detection_datasets, 
    collate_fn, 
    split_by_domain,
    check_split_leakage
)
from utils.integrated_model import load_integrated_model
from utils.training_utils import (
    compute_metrics, 
    get_training_arguments,
    get_early_stopping_callback
)
from transformers import Trainer, TrainerCallback

# =============================================================================
# Configuration
# =============================================================================

# Dataset paths
COCO_DIR = "E:/Python/DLCV/dataset/coco"
COCO_RAIN_DIR = "E:/Python/DLCV/dataset/coco_rain"
OUTPUT_DIR = "./outputs_spdnet_adaptation"

# Model paths
SPDNET_MODEL_PATH = "E:/Python/DLCV/Project DLCV/model_spa.pt"  # Pretrained SPDNet
RTDETR_MODEL_NAME = "PekingU/rtdetr_r18vd"  # Pretrained RT-DETR (will be frozen)
NUM_LABELS = 80  # COCO classes

# Dataset configuration - ONLY RAINY IMAGES
# We want SPDNet to learn how to help RT-DETR on rainy images
# Clean images are not useful here since RT-DETR is frozen
PERCENT_DATASET = 10   # Use 10% of rainy images (~11,800 images) for faster iteration
COCO_RATIO = 0.0       # 0% clean images (we don't need them)
RAIN_RATIO = 1.0       # 100% rainy images (this is what we're training for)

# Training configuration
NUM_EPOCHS = 10        # Fewer epochs with larger effective dataset coverage
BATCH_SIZE = 2         # Reduced from 8: SPDNet needs more memory
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # Increased: Effective batch size = 2 * 8 = 16
LEARNING_RATE = 1e-4   # Higher LR for faster convergence with frozen RT-DETR
SEED = 42

# CRITICAL: FP16 must be DISABLED for SPDNet
# SPDNet has numerical stability issues with half precision
FP16 = False

DATALOADER_WORKERS = 8  # More workers for faster data loading

# SPDNet Configuration
SPDNET_N_FEATS = 32
SPDNET_N_RESBLOCKS = 3

# Callback configuration
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for N epochs

# Set environment variable for CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# =============================================================================
# Callback for monitoring SPDNet training
# =============================================================================

class SPDNetAdaptationCallback(TrainerCallback):
    """Callback to monitor SPDNet adaptation training"""
    
    def __init__(self):
        self.best_map = 0.0
        self.epoch_maps = []
    
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of each epoch"""
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        
        print("\n" + "=" * 80)
        print(f"EPOCH {current_epoch + 1}/{args.num_train_epochs}")
        print("Training Mode: SPDNet Adaptation (RT-DETR Frozen)")
        print("=" * 80)
        
        # Verify freeze status
        self._verify_freeze_status(model)
    
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics and 'eval_map' in metrics:
            current_map = metrics['eval_map']
            self.epoch_maps.append(current_map)
            
            if current_map > self.best_map:
                self.best_map = current_map
                print(f"\n🎉 NEW BEST mAP: {current_map:.4f}")
            else:
                print(f"\nCurrent mAP: {current_map:.4f} (Best: {self.best_map:.4f})")
    
    def _verify_freeze_status(self, model):
        """Verify that RT-DETR is frozen and SPDNet is trainable"""
        # Count trainable parameters in each module
        spdnet_trainable = sum(p.numel() for p in model.derain_module.parameters() if p.requires_grad)
        spdnet_total = sum(p.numel() for p in model.derain_module.parameters())
        
        rtdetr_trainable = sum(p.numel() for p in model.detection_module.parameters() if p.requires_grad)
        rtdetr_total = sum(p.numel() for p in model.detection_module.parameters())
        
        print(f"\nModule Status:")
        print(f"  SPDNet: {spdnet_trainable:,}/{spdnet_total:,} trainable ({100*spdnet_trainable/spdnet_total:.1f}%)")
        print(f"  RT-DETR: {rtdetr_trainable:,}/{rtdetr_total:,} trainable ({100*rtdetr_trainable/rtdetr_total:.1f}%)")
        
        # Warn if RT-DETR is not fully frozen
        if rtdetr_trainable > 0:
            print(f"  ⚠️ WARNING: RT-DETR has {rtdetr_trainable:,} trainable parameters!")
            print(f"     This may cause performance degradation on clean images.")


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    """Main training function for SPDNet adaptation"""
    print("=" * 80)
    print("SPDNet Adaptation Training Script")
    print("=" * 80)
    print("\nStrategy: Train SPDNet to produce detection-friendly de-rained images")
    print("          while keeping RT-DETR frozen to preserve clean-image performance.")
    print("\nKey Settings:")
    print(f"  - RT-DETR: FROZEN (no gradient updates)")
    print(f"  - SPDNet: TRAINABLE")
    print(f"  - Dataset: RAINY IMAGES ONLY ({RAIN_RATIO*100:.0f}% rain, {COCO_RATIO*100:.0f}% clean)")
    print(f"  - FP16: DISABLED (SPDNet numerical stability)")
    print("=" * 80)
    
    # Performance optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ==========================================================================
    # Step 1: Load datasets (RAINY ONLY)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Loading RAINY-ONLY dataset...")
    print("=" * 80)
    
    ds_train, ds_valid = load_datasets(
        coco_dir=COCO_DIR,
        coco_rain_dir=COCO_RAIN_DIR,
        coco_ratio=COCO_RATIO,
        rain_ratio=RAIN_RATIO,
        seed=SEED
    )
    
    # Verify we have only rainy images
    _, train_rainy = split_by_domain(ds_train)
    _, valid_rainy = split_by_domain(ds_valid)
    
    print(f"\nDataset Composition:")
    print(f"  Training: {len(train_rainy)} rainy images")
    print(f"  Validation: {len(valid_rainy)} rainy images")
    
    if len(train_rainy) == 0:
        print("ERROR: No rainy images found! Check dataset configuration.")
        return
    
    check_split_leakage(ds_train, ds_valid)
    
    # ==========================================================================
    # Step 2: Load integrated model (SPDNet + RT-DETR)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Loading model with RT-DETR FROZEN...")
    print("=" * 80)
    print(f"SPDNet weights: {SPDNET_MODEL_PATH}")
    print(f"RT-DETR weights: {RTDETR_MODEL_NAME}")
    
    model, processor = load_integrated_model(
        spdnet_path=SPDNET_MODEL_PATH,
        rtdetr_name=RTDETR_MODEL_NAME,
        num_labels=NUM_LABELS,
        device=device,
        freeze_derain=False,      # SPDNet is TRAINABLE
        freeze_detection=True,    # RT-DETR is FROZEN
        spdnet_n_feats=SPDNET_N_FEATS,
        spdnet_n_resblocks=SPDNET_N_RESBLOCKS
    )
    
    # Double-check that RT-DETR is truly frozen
    rtdetr_trainable = sum(p.numel() for p in model.detection_module.parameters() if p.requires_grad)
    if rtdetr_trainable > 0:
        print(f"\n⚠️ Forcing RT-DETR to be fully frozen...")
        for param in model.detection_module.parameters():
            param.requires_grad = False
        print(f"   Done. RT-DETR now has 0 trainable parameters.")
    
    # ==========================================================================
    # Step 3: Create augmented datasets
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Creating augmented datasets...")
    print("=" * 80)
    
    train_dataset, valid_dataset = create_detection_datasets(
        ds_train=ds_train,
        ds_valid=ds_valid,
        processor=processor,
        percent_dataset=PERCENT_DATASET
    )
    
    # ==========================================================================
    # Step 4: Configure training arguments
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Configuring training arguments...")
    print("=" * 80)
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  FP16: {FP16}")
    
    args = get_training_arguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        fp16=FP16,  # CRITICAL: Must be False for SPDNet
        dataloader_num_workers=DATALOADER_WORKERS,
        lr_scheduler_type="cosine",
        logging_steps=50,
        warmup_steps=200,
    )
    
    # ==========================================================================
    # Step 5: Create trainer
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Creating trainer...")
    print("=" * 80)
    
    callbacks = [
        SPDNetAdaptationCallback(),
        get_early_stopping_callback(patience=EARLY_STOPPING_PATIENCE)
    ]
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=partial(compute_metrics, processor=processor, threshold=0.05),
        callbacks=callbacks,
    )
    
    # ==========================================================================
    # Step 6: Train!
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Starting SPDNet adaptation training...")
    print("=" * 80)
    
    trainer.train()
    
    # ==========================================================================
    # Step 7: Save the adapted SPDNet
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Saving adapted SPDNet...")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save only the SPDNet weights (RT-DETR is unchanged)
    spdnet_save_path = f"{OUTPUT_DIR}/spdnet_adapted.pt"
    torch.save(model.derain_module.state_dict(), spdnet_save_path)
    print(f"[OK] Adapted SPDNet saved to: {spdnet_save_path}")
    
    # Also save the full integrated model for convenience
    full_model_path = f"{OUTPUT_DIR}/full_model"
    os.makedirs(full_model_path, exist_ok=True)
    
    # Save SPDNet
    torch.save(model.derain_module.state_dict(), f"{full_model_path}/derain_module.pt")
    
    # Save processor
    processor.save_pretrained(full_model_path)
    
    # Save config
    config = {
        'spdnet_n_feats': SPDNET_N_FEATS,
        'spdnet_n_resblocks': SPDNET_N_RESBLOCKS,
        'rtdetr_name': RTDETR_MODEL_NAME,
        'training_epochs': NUM_EPOCHS,
        'training_lr': LEARNING_RATE,
    }
    torch.save(config, f"{full_model_path}/config.pt")
    
    print(f"[OK] Full model saved to: {full_model_path}")
    
    # ==========================================================================
    # Step 8: Plot training curves
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 8: Plotting training curves...")
    print("=" * 80)
    
    plot_training_curves(trainer, OUTPUT_DIR, NUM_EPOCHS)
    
    # ==========================================================================
    # Step 9: Final evaluation summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 9: Final Evaluation Summary")
    print("=" * 80)
    
    # Get best metrics from training history
    best_map = 0.0
    for entry in trainer.state.log_history:
        if 'eval_map' in entry:
            best_map = max(best_map, entry['eval_map'])
    
    print(f"\nTraining Results:")
    print(f"  Best mAP on rainy validation: {best_map:.4f}")
    print(f"\nNext Steps:")
    print(f"  1. Use the adapted SPDNet with your conditional model:")
    print(f"     - Load adapted SPDNet from: {spdnet_save_path}")
    print(f"     - Use vanilla RT-DETR: {RTDETR_MODEL_NAME}")
    print(f"     - Use existing Rain Detector")
    print(f"  2. Run evaluation on both clean and rainy images to verify:")
    print(f"     - Clean image performance is preserved (using vanilla RT-DETR path)")
    print(f"     - Rainy image performance is improved (using SPDNet path)")
    
    # Clean up
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("[OK] SPDNet Adaptation Training Completed!")
    print("=" * 80)


def plot_training_curves(trainer, output_dir, num_epochs):
    """Plot training and validation curves"""
    # Initialize arrays
    training_loss = np.full(num_epochs, np.nan)
    val_loss = np.full(num_epochs, np.nan)
    val_map = np.full(num_epochs, np.nan)
    
    log_history = trainer.state.log_history
    
    # Extract epoch-level metrics
    for entry in log_history:
        if 'epoch' not in entry:
            continue
        
        try:
            epoch_val = float(entry['epoch'])
        except Exception:
            continue
        
        if not epoch_val.is_integer():
            continue
        
        epoch_idx = int(epoch_val) - 1
        if epoch_idx < 0 or epoch_idx >= num_epochs:
            continue
        
        if 'eval_map' in entry:
            val_map[epoch_idx] = float(entry.get('eval_map', np.nan))
        if 'eval_loss' in entry:
            val_loss[epoch_idx] = float(entry.get('eval_loss', np.nan))
        if 'loss' in entry:
            training_loss[epoch_idx] = float(entry.get('loss', np.nan))
    
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss subplot
    epochs = np.arange(1, num_epochs + 1)
    
    valid_train = ~np.isnan(training_loss)
    valid_val = ~np.isnan(val_loss)
    
    if valid_train.any():
        axs[0].plot(epochs[valid_train], training_loss[valid_train], 
                    label='Training Loss', marker='o', color='blue')
    if valid_val.any():
        axs[0].plot(epochs[valid_val], val_loss[valid_val], 
                    label='Validation Loss', marker='s', color='orange')
    
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title('Training and Validation Loss\n(SPDNet Adaptation)')
    axs[0].grid(True, alpha=0.3)
    
    # mAP subplot
    valid_map = ~np.isnan(val_map)
    if valid_map.any():
        axs[1].plot(epochs[valid_map], val_map[valid_map], 
                    label='Validation mAP', color='green', marker='D')
    
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('mAP')
    axs[1].legend()
    axs[1].set_title('Validation mAP on Rainy Images\n(SPDNet Adaptation)')
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = f'{output_dir}/training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Training curves saved to: {save_path}")


if __name__ == "__main__":
    main()
