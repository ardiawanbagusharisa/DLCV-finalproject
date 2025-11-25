#!/usr/bin/env python3
"""
Training script for Conditional Rain-Robust RT-DETR model.

This script trains the conditional model that selectively applies de-raining
based on rain detection. Requires a pretrained rain detector.

Training Strategy:
    Phase 1 (Epochs 1-3): Train RT-DETR detection head only (SPDNet frozen)
    Phase 2 (Epochs 4-8): Unfreeze SPDNet, train both SPDNet + RT-DETR
    Phase 3 (Epochs 9-12): Fine-tune everything end-to-end
"""

# CRITICAL: Suppress matplotlib warnings before any imports
# (supervision library imports matplotlib internally)
import warnings
warnings.filterwarnings('ignore', message='Unable to import Axes3D')

import os
import sys
import torch
import numpy as np
from functools import partial
from transformers import Trainer, TrainerCallback

# Ensure local 'utils' package is found first, avoiding conflicts with other libraries
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utils modules at top level for Windows multiprocessing support
from utils.data_utils import load_datasets, create_detection_datasets, collate_fn, split_by_domain, check_split_leakage
from utils.conditional_model import load_conditional_model
from utils.training_utils import compute_metrics, get_training_arguments, get_early_stopping_callback, ObjectDetectionTrainer

# Configuration
COCO_DIR = "E:/Python/DLCV/dataset/coco"
COCO_RAIN_DIR = "E:/Python/DLCV/dataset/coco_rain"
OUTPUT_DIR = "./outputs_conditional"
RAIN_DETECTOR_PATH = "./rain_detector_pretrained/rain_detector_best.pt"  # From Phase 0
SPDNET_MODEL_PATH = "E:/Python/DLCV/Project DLCV/model_spa.pt"
RTDETR_MODEL_NAME = "PekingU/rtdetr_r18vd"
NUM_LABELS = 80  # COCO classes

# Training configuration
PERCENT_DATASET = 20   # Use 20% of dataset (Speed optimization: ~5x faster)
COCO_RATIO = 0.5       # 50% clean images (Balanced sampling)
RAIN_RATIO = 0.5       # 50% rainy images (Focus on rainy domain adaptation)
NUM_EPOCHS = 12        # Full training (2 phases: 3 epochs head-only, 9 epochs full)
BATCH_SIZE = 8         # Increased from 4 (SPDNet frozen = more memory available)
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 8x2 = 16
LEARNING_RATE = 1e-5
SEED = 42
FP16 = True            # Enabled! Safe now that SPDNet gradients are disabled
DATALOADER_WORKERS = 4  # Increased for faster data loading

# Rain detection threshold
RAIN_THRESHOLD = 0.5  # Adjust based on rain detector performance

# SPDNet Configuration
SPDNET_N_FEATS = 32
SPDNET_N_RESBLOCKS = 3

# Training phase configuration (SPDNet remains FROZEN throughout)
PHASE1_EPOCHS = 3   # Train detection head only (RT-DETR decoder adaptation)
PHASE2_EPOCHS = 12  # Unfreeze RT-DETR backbone (Full fine-tuning, SPDNet still frozen)

# Callback configuration
EARLY_STOPPING_PATIENCE = 10  # Increased to prevent early stopping during debugging

# Set environment variable for CUDA
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Removed for performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Help with memory fragmentation


class ConditionalModelCallback(TrainerCallback):
    """Callback to manage freezing/unfreezing of conditional model components"""
    
    def __init__(self, phase1_epochs=3):
        """
        Args:
            phase1_epochs: Epochs for phase 1 (detection head only)
        Note: SPDNet remains FROZEN throughout all phases for speed/memory.
        """
        self.phase1_epochs = phase1_epochs
        self.current_phase = 0
    
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of each epoch"""
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        
        # Phase 1: Train detection head only
        if current_epoch < self.phase1_epochs and self.current_phase == 0:
            print("\n" + "=" * 80)
            print(f"PHASE 1: Training Detection Head Only (Epoch {current_epoch + 1}/{self.phase1_epochs})")
            print("=" * 80)
            # Rain detector and SPDNet should already be frozen
            # Also freeze RT-DETR backbone
            frozen_count = 0
            for name, param in model.detection_module.named_parameters():
                if 'backbone' in name or 'encoder' in name:
                    param.requires_grad = False
                    frozen_count += 1
            print(f"[OK] Rain detector: FROZEN")
            print(f"[OK] SPDNet: FROZEN")
            print(f"[OK] RT-DETR backbone/encoder: FROZEN ({frozen_count} layers)")
            print(f"[OK] RT-DETR decoder: TRAINABLE")
            self._print_stats(model)
        
        # Transition to Phase 2
        elif current_epoch == self.phase1_epochs and self.current_phase == 0:
            print("\n" + "=" * 80)
            print(f"PHASE 2: Fine-Tuning RT-DETR (Epoch {current_epoch + 1})")
            print("  → Unfreezing RT-DETR backbone + decoder")
            print("  → SPDNet remains FROZEN (pretrained, saves memory/time)")
            print("  → Rain detector remains FROZEN (pretrained is stable)")
            print("=" * 80)
            # Unfreeze RT-DETR (all layers)
            for param in model.detection_module.parameters():
                param.requires_grad = True
            print(f"[OK] SPDNet: FROZEN (always)")
            print(f"[OK] RT-DETR (all layers): TRAINABLE")
            print(f"[OK] Rain detector: FROZEN (pretrained)")
            self._print_stats(model)
            self.current_phase = 1
        
        # Phase 2: Continue full RT-DETR training
        elif self.current_phase == 1:
            print(f"\nPHASE 2: Epoch {current_epoch + 1}/{args.num_train_epochs}")
    
    def _print_stats(self, model):
        """Print parameter statistics"""
        stats = model.get_model_stats()
        print(f"Total parameters: {stats['total_params']:,}")
        print(f"Trainable parameters: {stats['trainable_params']:,} ({100 * stats['trainable_ratio']:.1f}%)")


def main():
    """Main training function"""
    
    print("=" * 80)
    print("Conditional Rain-Robust RT-DETR Training Script")
    print("Selective De-raining: Only process rainy images")
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
    
    # Check if rain detector exists
    if not os.path.exists(RAIN_DETECTOR_PATH):
        print("\n" + "=" * 80)
        print("ERROR: Rain detector not found!")
        print("=" * 80)
        print(f"Expected path: {RAIN_DETECTOR_PATH}")
        print("\nYou need to pre-train the rain detector first:")
        print("  → python Pretrain_rain_detector.py")
        print("\nThis will create the rain detector model used for conditional de-raining.")
        return
    
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
    
    # Load conditional model with pretrained components
    print("\n" + "=" * 80)
    print("Step 2: Loading conditional model with pretrained weights...")
    print("=" * 80)
    print(f"Rain detector: {RAIN_DETECTOR_PATH}")
    print(f"SPDNet: {SPDNET_MODEL_PATH}")
    print(f"RT-DETR: {RTDETR_MODEL_NAME}")
    
    model, processor = load_conditional_model(
        rain_detector_path=RAIN_DETECTOR_PATH,
        spdnet_path=SPDNET_MODEL_PATH,
        rtdetr_name=RTDETR_MODEL_NAME,
        num_labels=NUM_LABELS,
        device=device,
        rain_threshold=RAIN_THRESHOLD,
        freeze_rain_detector=True,  # Keep rain detector frozen
        freeze_derain=True,  # Start with SPDNet frozen (Phase 1)
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
        lr_scheduler_type="cosine",
        logging_steps=100,
        warmup_steps=200
    )
    
    # Disable safetensors for the Trainer to avoid shared memory errors
    args.save_safetensors = False
    
    # Create trainer with conditional model callback
    print("\n" + "=" * 80)
    print("Step 5: Creating trainer...")
    print("=" * 80)
    
    print("\nTraining Strategy (Optimized - SPDNet Frozen):")
    print(f"  Phase 1 (Epochs 1-{PHASE1_EPOCHS}): Detection head only")
    print(f"  Phase 2 (Epochs {PHASE1_EPOCHS + 1}-{NUM_EPOCHS}): Full RT-DETR fine-tuning")
    print(f"\nSPDNet: FROZEN throughout (pretrained de-rainer, saves memory/time)")
    print(f"Rain detection threshold: {RAIN_THRESHOLD:.2f}")
    print(f"Expected speedup: ~3x on clean images, 2-3x faster training with FP16")
    
    callbacks = [
        ConditionalModelCallback(phase1_epochs=PHASE1_EPOCHS),
        get_early_stopping_callback(patience=EARLY_STOPPING_PATIENCE)
    ]
    
    trainer = ObjectDetectionTrainer(
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
        best_save_path = f"{OUTPUT_DIR}/best_conditional"
        model.save_pretrained(best_save_path)
    print(f"[OK] Best conditional model saved to {best_save_path}")
    
    # Also save final model
    final_save_path = f"{OUTPUT_DIR}/final_conditional"
    model.save_pretrained(final_save_path)
    print(f"[OK] Final conditional model saved to {final_save_path}")
    
    # Save training metrics to CSV (matplotlib removed)
    print("\n" + "=" * 80)
    print("Step 8: Saving training metrics...")
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
    print("[OK] Training completed successfully!")
    print("=" * 80)
    print(f"\nModel outputs:")
    print(f"  - Best model: {OUTPUT_DIR}/best_conditional/")
    print(f"  - Final model: {OUTPUT_DIR}/final_conditional/")
    print(f"  - Training metrics (CSV): {OUTPUT_DIR}/training_metrics.csv")
    print(f"  - Full training log (JSON): {OUTPUT_DIR}/training_log.json")
    print(f"  - TensorBoard logs: {OUTPUT_DIR}/runs/")
    print(f"\nNext step: Evaluate conditional model")
    print(f"  → python Eval_conditional.py")


def plot_training_curves(trainer, args):
    """Save training metrics to CSV file (matplotlib removed to avoid worker warnings)"""
    import json
    
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
    
    # Save metrics to CSV
    metrics_file = f'{OUTPUT_DIR}/training_metrics.csv'
    with open(metrics_file, 'w') as f:
        f.write('epoch,training_loss,val_loss,val_map\n')
        for i in range(len(training_loss)):
            f.write(f'{i+1},{training_loss[i]:.6f},{val_loss[i]:.6f},{val_map[i]:.6f}\n')
    
    print(f'[OK] Training metrics saved to {metrics_file}')
    print(f'Epochs logged: {(~np.isnan(val_map)).sum()}')
    
    # Also save full log history as JSON for reference
    json_file = f'{OUTPUT_DIR}/training_log.json'
    with open(json_file, 'w') as f:
        json.dump(log_history, f, indent=2)
    print(f'[OK] Full training log saved to {json_file}')


if __name__ == "__main__":
    main()
