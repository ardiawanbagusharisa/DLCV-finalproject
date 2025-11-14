"""
Integrated Rain-Robust RT-DETR Model

Combines SPDNet de-raining with RT-DETR object detection into a single end-to-end architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RTDetrForObjectDetection


class RainRobustRTDETR(nn.Module):
    """
    End-to-end rain-robust object detection model.
    
    Architecture:
        Rainy Image → SPDNet (de-raining) → RT-DETR (detection) → Detections
    
    Args:
        spdnet_model: Pretrained SPDNet model for de-raining
        rtdetr_model: Pretrained RT-DETR model for object detection
        freeze_derain: If True, freeze SPDNet weights (default: False)
        freeze_detection: If True, freeze RT-DETR weights (default: False)
    """
    
    def __init__(self, spdnet_model, rtdetr_model, freeze_derain=False, freeze_detection=False):
        super().__init__()
        
        # De-raining module (SPDNet)
        self.derain_module = spdnet_model
        
        # Detection module (RT-DETR)
        self.detection_module = rtdetr_model
        
        # Freeze modules if requested
        if freeze_derain:
            self._freeze_module(self.derain_module, "De-raining (SPDNet)")
        
        if freeze_detection:
            self._freeze_module(self.detection_module, "Detection (RT-DETR)")
        
        # Track module states
        self.derain_frozen = freeze_derain
        self.detection_frozen = freeze_detection
        
        print("=" * 80)
        print("Integrated Rain-Robust RT-DETR Model Initialized")
        print("=" * 80)
        self._print_status()
    
    def _freeze_module(self, module, name):
        """Freeze all parameters in a module"""
        for param in module.parameters():
            param.requires_grad = False
        print(f"✓ {name} module frozen")
    
    def _unfreeze_module(self, module, name):
        """Unfreeze all parameters in a module"""
        for param in module.parameters():
            param.requires_grad = True
        print(f"✓ {name} module unfrozen")
    
    def freeze_derain(self):
        """Freeze de-raining module"""
        self._freeze_module(self.derain_module, "De-raining (SPDNet)")
        self.derain_frozen = True
        self._print_status()
    
    def unfreeze_derain(self):
        """Unfreeze de-raining module"""
        self._unfreeze_module(self.derain_module, "De-raining (SPDNet)")
        self.derain_frozen = False
        self._print_status()
    
    def freeze_detection(self):
        """Freeze detection module"""
        self._freeze_module(self.detection_module, "Detection (RT-DETR)")
        self.detection_frozen = True
        self._print_status()
    
    def unfreeze_detection(self):
        """Unfreeze detection module"""
        self._unfreeze_module(self.detection_module, "Detection (RT-DETR)")
        self.detection_frozen = False
        self._print_status()
    
    def _print_status(self):
        """Print current module status"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Module Status:")
        print(f"  - De-raining (SPDNet): {'FROZEN' if self.derain_frozen else 'TRAINABLE'}")
        print(f"  - Detection (RT-DETR): {'FROZEN' if self.detection_frozen else 'TRAINABLE'}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
    
    def forward(self, pixel_values=None, labels=None, **kwargs):
        """
        Forward pass through the integrated model.
        
        Args:
            pixel_values: Input rainy images (B, C, H, W)
            labels: Ground truth labels for training (optional)
            **kwargs: Additional arguments for RT-DETR
        
        Returns:
            RT-DETR outputs (with loss if labels provided)
        """
        # Step 1: De-rain the input images
        # SPDNet expects input in range [0, 1] and returns output in same range
        clean_images = self.derain_module(pixel_values)
        
        # Step 2: Run object detection on de-rained images
        outputs = self.detection_module(pixel_values=clean_images, labels=labels, **kwargs)
        
        return outputs
    
    def forward_with_derain_loss(self, pixel_values, labels=None, clean_targets=None, derain_weight=0.1):
        """
        Forward pass with optional de-raining supervision.
        
        Args:
            pixel_values: Input rainy images (B, C, H, W)
            labels: Ground truth labels for detection
            clean_targets: Ground truth clean images for de-raining supervision (optional)
            derain_weight: Weight for de-raining loss (default: 0.1)
        
        Returns:
            Dictionary with 'loss', 'detection_loss', 'derain_loss' (if clean_targets provided)
        """
        # De-rain the images
        clean_pred = self.derain_module(pixel_values)
        
        # Detection on de-rained images
        detection_outputs = self.detection_module(pixel_values=clean_pred, labels=labels)
        
        # Compute total loss
        total_loss = detection_outputs.loss
        losses = {
            'detection_loss': detection_outputs.loss.item() if detection_outputs.loss is not None else 0.0
        }
        
        # Add de-raining loss if clean targets provided
        if clean_targets is not None:
            derain_loss = F.l1_loss(clean_pred, clean_targets)
            total_loss = total_loss + derain_weight * derain_loss
            losses['derain_loss'] = derain_loss.item()
        
        # Create modified output
        detection_outputs.loss = total_loss
        detection_outputs.losses = losses
        
        return detection_outputs
    
    def get_derained_image(self, pixel_values):
        """
        Get de-rained image without running detection.
        Useful for visualization and debugging.
        
        Args:
            pixel_values: Input rainy images (B, C, H, W)
        
        Returns:
            De-rained images (B, C, H, W)
        """
        with torch.no_grad():
            return self.derain_module(pixel_values)
    
    def save_pretrained(self, save_directory):
        """
        Save the integrated model.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save de-raining module
        torch.save(self.derain_module.state_dict(), 
                   os.path.join(save_directory, "spdnet_weights.pt"))
        
        # Save detection module (using HuggingFace method)
        self.detection_module.save_pretrained(
            os.path.join(save_directory, "rtdetr")
        )
        
        # Save configuration
        config = {
            'derain_frozen': self.derain_frozen,
            'detection_frozen': self.detection_frozen,
        }
        torch.save(config, os.path.join(save_directory, "config.pt"))
        
        print(f"✓ Integrated model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory, spdnet_model, device='cuda'):
        """
        Load the integrated model from a directory.
        
        Args:
            load_directory: Directory containing saved model
            spdnet_model: SPDNet model architecture (for loading weights)
            device: Device to load model on
        
        Returns:
            Loaded RainRobustRTDETR model
        """
        import os
        
        # Load de-raining module weights
        spdnet_weights = torch.load(
            os.path.join(load_directory, "spdnet_weights.pt"),
            map_location=device
        )
        spdnet_model.load_state_dict(spdnet_weights)
        
        # Load detection module
        rtdetr_model = RTDetrForObjectDetection.from_pretrained(
            os.path.join(load_directory, "rtdetr")
        )
        
        # Load configuration
        config = torch.load(os.path.join(load_directory, "config.pt"))
        
        # Create integrated model
        model = cls(
            spdnet_model=spdnet_model,
            rtdetr_model=rtdetr_model,
            freeze_derain=config.get('derain_frozen', False),
            freeze_detection=config.get('detection_frozen', False)
        )
        
        print(f"✓ Integrated model loaded from {load_directory}")
        return model


def load_integrated_model(spdnet_path, rtdetr_name="PekingU/rtdetr_r18vd", 
                         num_labels=80, device='cuda',
                         freeze_derain=False, freeze_detection=False,
                         spdnet_n_feats=32, spdnet_n_resblocks=3):
    """
    Load integrated model with pretrained weights.
    
    Args:
        spdnet_path: Path to pretrained SPDNet model (.pt file)
        rtdetr_name: HuggingFace model name or path for RT-DETR
        num_labels: Number of detection classes (default: 80 for COCO)
        device: Device to load models on
        freeze_derain: If True, freeze SPDNet weights
        freeze_detection: If True, freeze RT-DETR weights
        spdnet_n_feats: SPDNet feature maps (default: 32)
        spdnet_n_resblocks: SPDNet residual blocks (default: 3)
    
    Returns:
        Tuple of (integrated_model, processor)
    """
    from utils.spdnet_utils import load_spdnet_model
    from utils.model_utils import load_model_and_processor
    
    print("=" * 80)
    print("Loading Integrated Rain-Robust RT-DETR Model")
    print("=" * 80)
    
    # Load SPDNet with pretrained weights
    print("\n1. Loading SPDNet de-raining module...")
    spdnet_model = load_spdnet_model(
        spdnet_path, 
        device=device,
        n_feats=spdnet_n_feats,
        n_resblocks=spdnet_n_resblocks
    )
    
    # Load RT-DETR with pretrained weights
    print("\n2. Loading RT-DETR detection module...")
    rtdetr_model, processor = load_model_and_processor(
        model_name=rtdetr_name,
        num_labels=num_labels
    )
    rtdetr_model = rtdetr_model.to(device)
    
    # Create integrated model
    print("\n3. Creating integrated model...")
    integrated_model = RainRobustRTDETR(
        spdnet_model=spdnet_model,
        rtdetr_model=rtdetr_model,
        freeze_derain=freeze_derain,
        freeze_detection=freeze_detection
    )
    
    print("\n" + "=" * 80)
    print("✓ Integrated model ready for training/inference")
    print("=" * 80)
    
    return integrated_model, processor
