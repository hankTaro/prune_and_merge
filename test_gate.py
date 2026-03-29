import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pm-vit'))
import tmvit

print("Testing PM-ViT with AdaptiveThresholdGate...")
model = tmvit.tmvit_tiny_patch16_224(use_adaptive_gate=True, merge_list=list(range(12)))

with torch.no_grad():
    dummy = torch.randn(1, 197, 192)  # (B, N, dim) sequence
    for i, block in enumerate(model.blocks):
        if block.threshold_gate is not None:
            g = block.threshold_gate(dummy, i)
            print(f"Block {i} initial gate: {g.item():.4f}")
