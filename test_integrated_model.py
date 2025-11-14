#!/usr/bin/env python3
"""
Quick test script to verify integrated model loads correctly.

This script:
1. Loads the integrated model with pretrained weights
2. Runs a test forward pass
3. Verifies output shapes and gradient flow
"""

import torch
import numpy as np
from PIL import Image

from utils.integrated_model import load_integrated_model

# Configuration
SPDNET_MODEL_PATH = "E:/Python/DLCV/Project/model_spa.pt"
RTDETR_MODEL_NAME = "PekingU/rtdetr_r18vd"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_model_loading():
    """Test that model loads successfully"""
    print("=" * 80)
    print("Test 1: Model Loading")
    print("=" * 80)
    
    try:
        model, processor = load_integrated_model(
            spdnet_path=SPDNET_MODEL_PATH,
            rtdetr_name=RTDETR_MODEL_NAME,
            device=DEVICE
        )
        print("\n✓ Model loaded successfully")
        return model, processor
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        raise


def test_forward_pass(model, processor):
    """Test forward pass with dummy input"""
    print("\n" + "=" * 80)
    print("Test 2: Forward Pass")
    print("=" * 80)
    
    try:
        # Create dummy rainy image (640x640x3)
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_image)
        
        # Preprocess
        inputs = processor(images=dummy_image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        print(f"Input shape: {inputs['pixel_values'].shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Forward pass successful")
        print(f"  - Logits shape: {outputs.logits.shape}")
        print(f"  - Pred boxes shape: {outputs.pred_boxes.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_gradient_flow(model, processor):
    """Test that gradients flow through both modules"""
    print("\n" + "=" * 80)
    print("Test 3: Gradient Flow")
    print("=" * 80)
    
    try:
        # Create dummy input and target
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_image)
        
        inputs = processor(images=dummy_image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Create dummy labels (one box)
        labels = [{
            'class_labels': torch.tensor([1], dtype=torch.long, device=DEVICE),
            'boxes': torch.tensor([[0.3, 0.3, 0.5, 0.5]], dtype=torch.float32, device=DEVICE)
        }]
        
        # Forward pass with labels
        model.train()
        outputs = model(pixel_values=inputs['pixel_values'], labels=labels)
        
        print(f"✓ Forward pass with labels successful")
        print(f"  - Loss: {outputs.loss.item():.4f}")
        
        # Backward pass
        outputs.loss.backward()
        
        # Check gradients in both modules
        spdnet_has_grad = False
        rtdetr_has_grad = False
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'derain_module' in name:
                    spdnet_has_grad = True
                if 'detection_module' in name:
                    rtdetr_has_grad = True
        
        print(f"✓ Backward pass successful")
        print(f"  - SPDNet gradients: {'✓ Present' if spdnet_has_grad else '✗ Missing'}")
        print(f"  - RT-DETR gradients: {'✓ Present' if rtdetr_has_grad else '✗ Missing'}")
        
        model.eval()
        return True
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        return False


def test_freeze_unfreeze(model):
    """Test freezing/unfreezing functionality"""
    print("\n" + "=" * 80)
    print("Test 4: Freeze/Unfreeze Functionality")
    print("=" * 80)
    
    try:
        # Count trainable params before
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Freeze de-raining module
        model.freeze_derain()
        trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Freeze de-raining successful")
        print(f"  - Trainable params: {trainable_before:,} → {trainable_after_freeze:,}")
        
        # Unfreeze de-raining module
        model.unfreeze_derain()
        trainable_after_unfreeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Unfreeze de-raining successful")
        print(f"  - Trainable params: {trainable_after_freeze:,} → {trainable_after_unfreeze:,}")
        
        assert trainable_before == trainable_after_unfreeze, "Param count mismatch after unfreeze"
        
        return True
    except Exception as e:
        print(f"✗ Freeze/unfreeze test failed: {e}")
        return False


def test_save_load(model, processor):
    """Test saving and loading functionality"""
    print("\n" + "=" * 80)
    print("Test 5: Save/Load Functionality")
    print("=" * 80)
    
    import os
    import tempfile
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            
            # Save model
            model.save_pretrained(save_path)
            processor.save_pretrained(os.path.join(save_path, "processor"))
            print(f"✓ Model saved to {save_path}")
            
            # Load model
            from utils.integrated_model import RainRobustRTDETR
            from utils.spdnet_utils import load_spdnet_model
            
            spdnet = load_spdnet_model(SPDNET_MODEL_PATH, device=DEVICE)
            loaded_model = RainRobustRTDETR.from_pretrained(
                save_path,
                spdnet_model=spdnet,
                device=DEVICE
            )
            print(f"✓ Model loaded successfully")
            
            # Verify parameters match
            orig_params = sum(p.numel() for p in model.parameters())
            loaded_params = sum(p.numel() for p in loaded_model.parameters())
            
            assert orig_params == loaded_params, f"Parameter count mismatch: {orig_params} vs {loaded_params}"
            print(f"✓ Parameter count matches: {orig_params:,}")
            
        return True
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("INTEGRATED MODEL TEST SUITE")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"SPDNet path: {SPDNET_MODEL_PATH}")
    print(f"RT-DETR model: {RTDETR_MODEL_NAME}")
    
    results = {}
    
    # Test 1: Model loading
    try:
        model, processor = test_model_loading()
        results['loading'] = True
    except Exception:
        results['loading'] = False
        print("\n✗ Cannot proceed with other tests (model failed to load)")
        return results
    
    # Test 2: Forward pass
    results['forward_pass'] = test_forward_pass(model, processor)
    
    # Test 3: Gradient flow
    results['gradient_flow'] = test_gradient_flow(model, processor)
    
    # Test 4: Freeze/unfreeze
    results['freeze_unfreeze'] = test_freeze_unfreeze(model)
    
    # Test 5: Save/load
    results['save_load'] = test_save_load(model, processor)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Integrated model is ready for training!")
    else:
        print("✗ SOME TESTS FAILED - Please fix issues before training")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
