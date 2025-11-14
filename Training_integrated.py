#!/usr/bin/env python3
"""
Training script for Integrated Rain-Robust RT-DETR model.

This script trains the end-to-end integrated model combining SPDNet and RT-DETR.

Training Strategy:
    Phase 1 (Epochs 1-2): Freeze SPDNet, train RT-DETR detection head only
    Phase 2 (Epochs 3-8): Unfreeze SPDNet, freeze RT-DETR backbone, train both
    Phase 3 (Epochs 9-12): Unfreeze everything, fine-tune end-to-end
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# Import utility modules
from utils.data_utils import load_datasets, create_detection_datasets, collate_fn, split_by_domain, check_split_leakage
from utils.integrated_model import load_integrated_model
from utils.training_utils import (
    compute_metrics, 
    get_training_arguments,
    get_early_stopping_callback
)
from transformers import Trainer, TrainerCallback

# Configuration
COCO_DIR = "E:/Python/DLCV/Project/dataset/coco"
COCO_RAIN_DIR = "E:/Python/DLCV/Project/dataset/coco_rain"
OUTPUT_DIR = "./outputs_integrated"
SPDNET_MODEL_PATH = "E:/Python/DLCV/Project/model_spa.pt"  # Pretrained SPDNet
RTDETR_MODEL_NAME = "PekingU/rtdetr_r18vd"  # Pretrained RT-DETR
NUM_LABELS = 80  # COCO classes

# Training configuration
PERCENT_DATASET = 100  # Use 100% of dataset
COCO_RATIO = 0.9  # 90% clean images
RAIN_RATIO = 0.1  # 10% rainy images
NUM_EPOCHS = 12
BATCH_SIZE = 8  # Reduced due to two models in sequence (adjust based on GPU memory)
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 8 * 2 = 16
LEARNING_RATE = 1e-5
SEED = 42
FP16 = True  # Enable mixed precision
DATALOADER_WORKERS = 8  # Reduced due to memory constraints

# SPDNet Configuration
SPDNET_N_FEATS = 32
SPDNET_N_RESBLOCKS = 3

# Training phase configuration
PHASE1_EPOCHS = 2   # Train detection head only (SPDNet frozen)
PHASE2_EPOCHS = 8   # Train SPDNet + detection head (RT-DETR backbone frozen)
PHASE3_EPOCHS = 12  # Fine-tune everything end-to-end

# Callback configuration
EARLY_STOPPING_PATIENCE = 4  # Stop if no improvement for N epochs

# Set environment variable for CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class IntegratedModelCallback(TrainerCallback):
    """Callback to manage freezing/unfreezing of integrated model components"""
    
    def __init__(self, phase1_epochs=2, phase2_epochs=8):
        """
        Args:
            phase1_epochs: Epochs for phase 1 (detection head only)
            phase2_epochs: Epochs for phase 2 (+ SPDNet, freeze backbone)
        """
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.current_phase = 0
    
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of each epoch"""
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        
        # Phase 1: Train detection head only (epochs 0-1)
        if current_epoch < self.phase1_epochs and self.current_phase == 0:
            print("\n" + "=" * 80)
            print(f"PHASE 1: Training Detection Head Only (Epoch {current_epoch + 1}/{self.phase1_epochs})")
            print("=" * 80)
            # SPDNet should already be frozen from initialization
            # Also freeze RT-DETR backbone
            frozen_count = 0
            for name, param in model.detection_module.named_parameters():
                if 'backbone' in name or 'encoder' in name:
                    param.requires_grad = False
                    frozen_count += 1
            print(f"✓ SPDNet: FROZEN")
            print(f"✓ RT-DETR backbone/encoder: FROZEN ({frozen_count} layers)")
            print(f"✓ RT-DETR decoder: TRAINABLE")
            self._print_stats(model)
        
        # Transition to Phase 2
        elif current_epoch == self.phase1_epochs and self.current_phase == 0:
            print("\n" + "=" * 80)
            print(f"PHASE 2: Training SPDNet + Detection Head (Epoch {current_epoch + 1})")
            print("  → Unfreezing SPDNet")
            print("  → Keeping RT-DETR backbone frozen for stability")
            print("=" * 80)
            # Unfreeze SPDNet
            model.unfreeze_derain()
            # Keep RT-DETR backbone frozen
            frozen_count = 0
            for name, param in model.detection_module.named_parameters():
                if 'backbone' in name or 'encoder' in name:
                    param.requires_grad = False
                    frozen_count += 1
            print(f"✓ RT-DETR backbone/encoder: FROZEN ({frozen_count} layers)")
            self._print_stats(model)
            self.current_phase = 1
        
        # Phase 2: Continue training SPDNet + detection head
        elif current_epoch < self.phase2_epochs and self.current_phase == 1:
            print(f"\nPHASE 2: Epoch {current_epoch + 1}/{self.phase2_epochs}")
        
        # Transition to Phase 3
        elif current_epoch == self.phase2_epochs and self.current_phase == 1:
            print("\n" + "=" * 80)
            print(f"PHASE 3: End-to-End Fine-Tuning (Epoch {current_epoch + 1})")
            print("  → Unfreezing ALL components")
            print("=" * 80)
            # Unfreeze everything
            for param in model.parameters():
                param.requires_grad = True
            print(f"✓ SPDNet: TRAINABLE")
            print(f"✓ RT-DETR (all layers): TRAINABLE")
            self._print_stats(model)
            self.current_phase = 2
        
        # Phase 3: Continue end-to-end training
        elif self.current_phase == 2:
            print(f"\nPHASE 3: Epoch {current_epoch + 1}/{args.num_train_epochs}")
    
    def _print_stats(self, model):
        """Print parameter statistics"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")


def main():
    """Main training function"""
    print("=" * 80)
    print("Integrated Rain-Robust RT-DETR Training Script")
    print("End-to-End Integration: SPDNet + RT-DETR")
    print("=" * 80)
    
    # Performance knobs
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
    check_split_leakage(ds_train, ds_valid)
    
    # Load integrated model with pretrained weights
    print("\n" + "=" * 80)
    print("Step 2: Loading integrated model with pretrained weights...")
    print("=" * 80)
    print(f"SPDNet weights: {SPDNET_MODEL_PATH}")
    print(f"RT-DETR weights: {RTDETR_MODEL_NAME}")
    
    model, processor = load_integrated_model(
        spdnet_path=SPDNET_MODEL_PATH,
        rtdetr_name=RTDETR_MODEL_NAME,
        num_labels=NUM_LABELS,
        device=device,
        freeze_derain=True,  # Start with SPDNet frozen (Phase 1)
        freeze_detection=False,
        spdnet_n_feats=SPDNET_N_FEATS,
        spdnet_n_resblocks=SPDNET_N_RESBLOCKS
    )
    
    # Create augmented datasets
    print("\n" + "=" * 80)
    print("Step 3: Creating augmented datasets...")
    print("=" * 80)
    train_dataset, valid_dataset = create_detection_datasets(
        ds_train=ds_train,
        ds_valid=ds_valid,
        processor=processor,
        percent_dataset=PERCENT_DATASET
    )
    
    # Get training arguments
    print("\n" + "=" * 80)
    print("Step 4: Configuring training arguments...")
    print("=" * 80)
    args = get_training_arguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        fp16=FP16,
        dataloader_num_workers=DATALOADER_WORKERS,
        lr_scheduler_type="cosine",  # Cosine annealing for better convergence
        logging_steps=100,
        warmup_steps=300  # More warmup for integrated model
    )
    
    # Create trainer with integrated model callback
    print("\n" + "=" * 80)
    print("Step 5: Creating trainer...")
    print("=" * 80)
    
    print("\nTraining Strategy:")
    print(f"  Phase 1 (Epochs 1-{PHASE1_EPOCHS}): Detection head only (SPDNet + backbone frozen)")
    print(f"  Phase 2 (Epochs {PHASE1_EPOCHS + 1}-{PHASE2_EPOCHS}): SPDNet + head (backbone frozen)")
    print(f"  Phase 3 (Epochs {PHASE2_EPOCHS + 1}-{NUM_EPOCHS}): End-to-end fine-tuning (all trainable)")
    
    callbacks = [
        IntegratedModelCallback(phase1_epochs=PHASE1_EPOCHS, phase2_epochs=PHASE2_EPOCHS),
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
    
    # Train model
    print("\n" + "=" * 80)
    print("Step 6: Starting training...")
    print("=" * 80)
    trainer.train()
    
    # Save best model
    print("\n" + "=" * 80)
    print("Step 7: Saving best model...")
    print("=" * 80)
    best = trainer.state.best_model_checkpoint
    if best:
        best_save_path = f"{OUTPUT_DIR}/best_integrated"
        model.save_pretrained(best_save_path)
        processor.save_pretrained(os.path.join(best_save_path, "processor"))
        print(f"✓ Best integrated model saved to {best_save_path}")
    
    # Also save final model
    final_save_path = f"{OUTPUT_DIR}/final_integrated"
    model.save_pretrained(final_save_path)
    processor.save_pretrained(os.path.join(final_save_path, "processor"))
    print(f"✓ Final integrated model saved to {final_save_path}")
    
    # Plot training curves
    print("\n" + "=" * 80)
    print("Step 8: Plotting training curves...")
    print("=" * 80)
    plot_training_curves(trainer, args)
    
    # Evaluate on rainy-only validation split
    print("\n" + "=" * 80)
    print("Step 9: Evaluating on rainy-only validation split...")
    print("=" * 80)
    valid_clean, valid_rain = split_by_domain(ds_valid)
    _, valid_rain_dataset = create_detection_datasets(valid_clean, valid_rain, processor, percent_dataset=100)
    rain_metrics = trainer.evaluate(eval_dataset=valid_rain_dataset)
    print(f"Rainy-only evaluation: {rain_metrics}")
    
    # Clean up
    print("\n" + "=" * 80)
    print("Step 10: Cleaning up...")
    print("=" * 80)
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("✓ Training completed successfully!")
    print("=" * 80)
    print(f"\nModel outputs:")
    print(f"  - Best model: {OUTPUT_DIR}/best_integrated/")
    print(f"  - Final model: {OUTPUT_DIR}/final_integrated/")
    print(f"  - Training curves: {OUTPUT_DIR}/training_curves.png")
    print(f"  - TensorBoard logs: {OUTPUT_DIR}/runs/")


def plot_training_curves(trainer, args):
    """Plot training and validation curves"""
    # Initialize arrays
    num_epochs = int(getattr(args, 'num_train_epochs', 0))
    training_loss = np.full(num_epochs, np.nan)
    val_loss = np.full(num_epochs, np.nan)
    val_map = np.full(num_epochs, np.nan)
    
    log_history = trainer.state.log_history
    
    # Extract epoch-level metrics
    for i, entry in enumerate(log_history):
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
            val_loss[epoch_idx] = float(entry.get('eval_loss', np.nan))
            
            # Find most recent training loss
            for j in range(i - 1, -1, -1):
                prev = log_history[j]
                if 'loss' in prev:
                    try:
                        training_loss[epoch_idx] = float(prev['loss'])
                        break
                    except Exception:
                        continue
    
    # Trim trailing NaNs
    valid_epochs = ~np.isnan(val_map) | ~np.isnan(training_loss) | ~np.isnan(val_loss)
    if valid_epochs.sum() > 0:
        training_loss = training_loss[valid_epochs]
        val_loss = val_loss[valid_epochs]
        val_map = val_map[valid_epochs]
    
    print(f'Epochs found: {len(val_map)}')
    
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss subplot
    if len(training_loss) > 0:
        axs[0].plot(training_loss, label='Training Loss', marker='o')
    if len(val_loss) > 0:
        axs[0].plot(val_loss, label='Validation Loss', marker='s')
    
    # Add phase markers
    if PHASE1_EPOCHS > 0:
        axs[0].axvline(x=PHASE1_EPOCHS - 0.5, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
    if PHASE2_EPOCHS > 0:
        axs[0].axvline(x=PHASE2_EPOCHS - 0.5, color='green', linestyle='--', alpha=0.5, label='Phase 2→3')
    
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title('Training and Validation Loss over Epochs')
    axs[0].grid(True, alpha=0.3)
    
    # mAP subplot
    if len(val_map) > 0:
        axs[1].plot(val_map, label='Validation mAP', color='purple', marker='D')
    
    # Add phase markers
    if PHASE1_EPOCHS > 0:
        axs[1].axvline(x=PHASE1_EPOCHS - 0.5, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
    if PHASE2_EPOCHS > 0:
        axs[1].axvline(x=PHASE2_EPOCHS - 0.5, color='green', linestyle='--', alpha=0.5, label='Phase 2→3')
    
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('mAP')
    axs[1].legend()
    axs[1].set_title('Validation mAP over Epochs (Integrated Model)')
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Training curves saved to {OUTPUT_DIR}/training_curves.png")


if __name__ == "__main__":
    main()
