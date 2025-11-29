#!/usr/bin/env python3
"""
Visualize Rain Mask and CBAM Attention from Feature De-rain Module

This script extracts and visualizes:
1. Rain masks (where the model thinks rain is affecting features)
2. Channel attention weights (which channels are being adjusted)
3. Spatial attention maps (which spatial regions are being adjusted)

Usage:
    python visualize_attention.py --image path/to/rainy_image.jpg
    python visualize_attention.py --image path/to/rainy_image.jpg --output attention_viz/
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from utils.feature_derain import FeatureDerainRTDETR, MultiScaleFeatureDerain


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "PekingU/rtdetr_r18vd"
DERAIN_TYPE = "multiscale"
CHECKPOINT_PATH = "./outputs_feature_derain/feature_derain_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Attention Extraction Hooks
# =============================================================================

class AttentionExtractor:
    """
    Hook-based extractor for rain masks and CBAM attention maps.
    """
    
    def __init__(self, model: FeatureDerainRTDETR):
        self.model = model
        self.rain_masks = []
        self.channel_attentions = []
        self.spatial_attentions = []
        self.backbone_features = []
        self.enhanced_features = []
        self.hooks = []
        
    def _register_hooks(self):
        """Register hooks on the de-rain module components."""
        self.rain_masks.clear()
        self.channel_attentions.clear()
        self.spatial_attentions.clear()
        self.backbone_features.clear()
        self.enhanced_features.clear()
        
        derain_module = self.model.derain_module
        
        if derain_module is None:
            print("No de-rain module found!")
            return
        
        # Hook into each FeatureDerainBlock
        for i, block in enumerate(derain_module.derain_blocks):
            # Hook rain mask output
            def rain_mask_hook(module, input, output, idx=i):
                self.rain_masks.append((idx, output.detach().cpu()))
            
            block.rain_mask.register_forward_hook(rain_mask_hook)
            
            # Hook CBAM components in residual blocks
            for j, res_block in enumerate(block.refine):
                if hasattr(res_block, 'attention') and hasattr(res_block.attention, 'channel_attention'):
                    def channel_hook(module, input, output, scale_idx=i, block_idx=j):
                        self.channel_attentions.append((scale_idx, block_idx, output.detach().cpu()))
                    
                    def spatial_hook(module, input, output, scale_idx=i, block_idx=j):
                        self.spatial_attentions.append((scale_idx, block_idx, output.detach().cpu()))
                    
                    res_block.attention.channel_attention.register_forward_hook(channel_hook)
                    res_block.attention.spatial_attention.register_forward_hook(spatial_hook)
    
    def extract(self, image_path: str):
        """
        Run inference and extract attention maps.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with extracted attention maps
        """
        self._register_hooks()
        
        # Load and preprocess image
        processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(DEVICE)
        
        # Run forward pass
        with torch.no_grad():
            _ = self.model(pixel_values=pixel_values)
        
        return {
            'rain_masks': self.rain_masks,
            'channel_attentions': self.channel_attentions,
            'spatial_attentions': self.spatial_attentions,
            'original_image': image,
            'original_size': original_size
        }


def extract_attention_maps(model: FeatureDerainRTDETR, image_path: str, device: str = "cuda"):
    """
    Extract rain masks and attention maps from the model for a given image.
    
    Uses manual forward pass through each component to capture intermediate outputs.
    """
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Create pixel mask (all ones for single image)
    batch_size, _, h, w = pixel_values.shape
    pixel_mask = torch.ones((batch_size, h, w), dtype=torch.long, device=device)
    
    # Get backbone features directly
    with torch.no_grad():
        # Access the backbone directly
        backbone = model.rtdetr.model.backbone
        
        # Run backbone forward with pixel_mask
        backbone_outputs = backbone(pixel_values, pixel_mask)
        
        # Extract features (backbone_outputs is list of tuples: [(features, mask), ...])
        features = [out[0] for out in backbone_outputs]
        
        # Now manually run through the de-rain module to capture intermediate outputs
        derain_module = model.derain_module
        
        rain_masks = []
        channel_attentions = []
        spatial_attentions = []
        enhanced_features = []
        
        if isinstance(derain_module, MultiScaleFeatureDerain):
            for i, (block, feat) in enumerate(zip(derain_module.derain_blocks, features)):
                # 1. Extract rain mask
                rain_mask = block.rain_mask(feat)  # (B, 1, H, W)
                rain_masks.append(rain_mask.cpu())
                
                # 2. Apply suppression
                suppressed = feat * (1 - rain_mask * block.blend.clamp(0, 1))
                
                # 3. Track CBAM attentions through residual blocks
                scale_channel_attn = []
                scale_spatial_attn = []
                
                x = suppressed
                for res_block in block.refine:
                    # ResidualBlock forward with attention tracking
                    residual = x
                    out = res_block.relu(res_block.bn1(res_block.conv1(x)))
                    out = res_block.bn2(res_block.conv2(out))
                    
                    # CBAM attention
                    cbam = res_block.attention
                    if hasattr(cbam, 'channel_attention'):
                        ca = cbam.channel_attention(out)
                        scale_channel_attn.append(ca.cpu())
                        out = out * ca
                        
                        sa = cbam.spatial_attention(out)
                        scale_spatial_attn.append(sa.cpu())
                        out = out * sa
                    
                    out = out + residual
                    x = res_block.relu(out)
                
                channel_attentions.append(scale_channel_attn)
                spatial_attentions.append(scale_spatial_attn)
                enhanced_features.append(x.cpu())
    
    return {
        'rain_masks': rain_masks,  # List of (B, 1, H, W) tensors
        'channel_attentions': channel_attentions,  # List of lists of (B, C, 1, 1) tensors
        'spatial_attentions': spatial_attentions,  # List of lists of (B, 1, H, W) tensors
        'original_features': [f.cpu() for f in features],
        'enhanced_features': enhanced_features,
        'original_image': image,
        'original_size': original_size
    }


def visualize_attention(attention_data: dict, output_path: str = None, show: bool = True):
    """
    Create comprehensive visualization of rain masks and attention maps.
    
    Args:
        attention_data: Dictionary from extract_attention_maps()
        output_path: Path to save the visualization
        show: Whether to display the visualization
    """
    original_image = attention_data['original_image']
    rain_masks = attention_data['rain_masks']
    spatial_attentions = attention_data['spatial_attentions']
    channel_attentions = attention_data['channel_attentions']
    
    # Create figure with subplots
    n_scales = len(rain_masks)
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Original image + Rain masks at each scale
    # Row 2: Spatial attention maps
    # Row 3: Channel attention histograms
    
    # Original image
    ax_orig = fig.add_subplot(3, n_scales + 1, 1)
    ax_orig.imshow(original_image)
    ax_orig.set_title('Original Image', fontsize=12, fontweight='bold')
    ax_orig.axis('off')
    
    scale_names = ['Scale 1 (128ch, 1/8)', 'Scale 2 (256ch, 1/16)', 'Scale 3 (512ch, 1/32)']
    
    # Rain masks
    for i, rain_mask in enumerate(rain_masks):
        ax = fig.add_subplot(3, n_scales + 1, i + 2)
        
        # Rain mask shape: (B, 1, H, W)
        mask = rain_mask[0, 0].numpy()  # (H, W)
        
        # Upsample to original image size for overlay
        mask_resized = np.array(Image.fromarray(mask).resize(original_image.size, Image.BILINEAR))
        
        # Create overlay
        img_array = np.array(original_image) / 255.0
        overlay = img_array.copy()
        
        # Red overlay where rain is detected
        alpha = 0.6
        overlay[:, :, 0] = overlay[:, :, 0] * (1 - mask_resized * alpha) + mask_resized * alpha
        
        ax.imshow(overlay)
        ax.set_title(f'Rain Mask - {scale_names[i]}', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        im = ax.imshow(mask_resized, cmap='hot', alpha=0.5)
        # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Spatial attentions (use last residual block for each scale)
    ax_spacer = fig.add_subplot(3, n_scales + 1, n_scales + 2)
    ax_spacer.text(0.5, 0.5, 'CBAM\nSpatial\nAttention', ha='center', va='center', 
                   fontsize=14, fontweight='bold')
    ax_spacer.axis('off')
    
    for i, scale_spatial in enumerate(spatial_attentions):
        ax = fig.add_subplot(3, n_scales + 1, n_scales + 3 + i)
        
        if scale_spatial:
            # Use last residual block's spatial attention
            spatial_attn = scale_spatial[-1][0, 0].numpy()  # (H, W)
            spatial_resized = np.array(Image.fromarray(spatial_attn).resize(original_image.size, Image.BILINEAR))
            
            ax.imshow(original_image)
            ax.imshow(spatial_resized, cmap='viridis', alpha=0.6)
            ax.set_title(f'Spatial Attn - {scale_names[i]}', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.axis('off')
    
    # Channel attentions (histogram)
    ax_spacer2 = fig.add_subplot(3, n_scales + 1, 2 * (n_scales + 1) + 1)
    ax_spacer2.text(0.5, 0.5, 'CBAM\nChannel\nAttention', ha='center', va='center', 
                    fontsize=14, fontweight='bold')
    ax_spacer2.axis('off')
    
    for i, scale_channel in enumerate(channel_attentions):
        ax = fig.add_subplot(3, n_scales + 1, 2 * (n_scales + 1) + 2 + i)
        
        if scale_channel:
            # Use last residual block's channel attention
            channel_attn = scale_channel[-1][0, :, 0, 0].numpy()  # (C,)
            
            # Sort and plot
            sorted_attn = np.sort(channel_attn)[::-1]
            ax.bar(range(len(sorted_attn)), sorted_attn, color='steelblue', alpha=0.7)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Neutral (0.5)')
            ax.set_xlabel('Channel (sorted)', fontsize=8)
            ax.set_ylabel('Attention Weight', fontsize=8)
            ax.set_title(f'Channel Attn - {scale_names[i]}', fontsize=10)
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
    
    plt.suptitle('Feature De-rain Module: Rain Mask & CBAM Attention Visualization', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved attention visualization to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_feature_difference(attention_data: dict, output_path: str = None, show: bool = True):
    """
    Visualize the difference between original and enhanced features.
    """
    original_features = attention_data['original_features']
    enhanced_features = attention_data['enhanced_features']
    original_image = attention_data['original_image']
    
    n_scales = len(original_features)
    fig, axes = plt.subplots(3, n_scales, figsize=(15, 12))
    
    scale_names = ['Scale 1 (128ch)', 'Scale 2 (256ch)', 'Scale 3 (512ch)']
    
    for i in range(n_scales):
        orig_feat = original_features[i][0]  # (C, H, W)
        enhanced_feat = enhanced_features[i][0]  # (C, H, W)
        
        # Compute feature statistics across channels
        orig_mean = orig_feat.mean(dim=0).numpy()
        enhanced_mean = enhanced_feat.mean(dim=0).numpy()
        diff_mean = enhanced_mean - orig_mean
        
        # Row 1: Original features
        axes[0, i].imshow(orig_mean, cmap='viridis')
        axes[0, i].set_title(f'Original - {scale_names[i]}')
        axes[0, i].axis('off')
        
        # Row 2: Enhanced features
        axes[1, i].imshow(enhanced_mean, cmap='viridis')
        axes[1, i].set_title(f'Enhanced - {scale_names[i]}')
        axes[1, i].axis('off')
        
        # Row 3: Difference (what the de-rain module changed)
        vmax = max(abs(diff_mean.min()), abs(diff_mean.max()))
        im = axes[2, i].imshow(diff_mean, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2, i].set_title(f'Difference - {scale_names[i]}')
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Feature Enhancement: Original vs De-rained Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved feature difference to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load the trained Feature De-rain RT-DETR model."""
    print(f"Loading model from: {checkpoint_path}")
    
    rtdetr = RTDetrForObjectDetection.from_pretrained(MODEL_NAME)
    model = FeatureDerainRTDETR(
        rtdetr_model=rtdetr,
        derain_type=DERAIN_TYPE,
        freeze_backbone=False
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'derain_module' in checkpoint:
            model.derain_module.load_state_dict(checkpoint['derain_module'])
            print("✓ Loaded de-rain module weights")
        if 'rtdetr' in checkpoint:
            model.rtdetr.load_state_dict(checkpoint['rtdetr'])
            print("✓ Loaded RT-DETR weights")
    else:
        print(f"⚠ Checkpoint not found, using pretrained weights only")
    
    model = model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Rain Mask and CBAM Attention from Feature De-rain Module"
    )
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', '-o', type=str, default='attention_viz',
                        help='Output directory for visualizations')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display visualizations')
    parser.add_argument('--device', '-d', type=str, default=None,
                        help='Device to use: "cuda" or "cpu"')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    device = args.device if args.device else DEVICE
    checkpoint_path = args.checkpoint if args.checkpoint else CHECKPOINT_PATH
    
    print("\n" + "=" * 60)
    print("Feature De-rain Attention Visualization")
    print("=" * 60)
    print(f"Input image: {args.image}")
    print(f"Device: {device}")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Extract attention maps
    print("\nExtracting attention maps...")
    attention_data = extract_attention_maps(model, args.image, device)
    
    # Create output paths
    os.makedirs(args.output, exist_ok=True)
    image_name = Path(args.image).stem
    
    # Visualize rain masks and CBAM attention
    print("\nGenerating visualizations...")
    attention_path = os.path.join(args.output, f"{image_name}_attention.png")
    visualize_attention(attention_data, attention_path, show=not args.no_show)
    
    # Visualize feature differences
    feature_diff_path = os.path.join(args.output, f"{image_name}_feature_diff.png")
    visualize_feature_difference(attention_data, feature_diff_path, show=not args.no_show)
    
    # Print summary
    print("\n" + "-" * 60)
    print("Rain Mask Summary:")
    for i, rain_mask in enumerate(attention_data['rain_masks']):
        mask = rain_mask[0, 0].numpy()
        print(f"  Scale {i+1}: mean={mask.mean():.4f}, max={mask.max():.4f}, "
              f"rain_area={(mask > 0.5).sum() / mask.size * 100:.1f}%")
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
