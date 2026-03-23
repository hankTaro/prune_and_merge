import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pm-vit'))
import tmvit

# Try to create the model
try:
    print("Testing if PM-ViT imports successfully...")
    # Using the name for tmvit_tiny
    model = tmvit.tmvit_tiny_patch16_224()
    print("Model created successfully!")
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Forward pass successful! Output shape: {out.shape}")
except Exception as e:
    print(f"Error encountered: {type(e).__name__} - {e}")
