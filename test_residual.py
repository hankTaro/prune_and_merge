import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pm-vit'))
from tmvit import RecoverMLP
import numpy as np

print("Testing Residual RecoverMLP Initialization...")

# Simulate fake pruned shapes
n_pruned = 107
n_full = 197
dim = 192
B = 2

# Create a random positive matrix to simulate pseudo-inverse
matrix = torch.randn(n_full, n_pruned)

# Model
mlp = RecoverMLP(n_pruned=n_pruned, n_full=n_full, hidden_ratio=2.0, recover_matrix=matrix)

# Fake input
x = torch.randn(B, n_pruned, dim)

# Forward pass
mlp_out = mlp(x)

# True linear prediction
true_linear = torch.tensordot(matrix, x, dims=([1], [1])).permute(1, 0, 2)

# Verify
diff = (mlp_out - true_linear).abs().max().item()
print(f"Max difference between MLP output and pure linear target: {diff:.8f}")

if diff < 1e-6:
    print("SUCCESS: Residual RecoverMLP acts exactly as the linear matrix at init!")
else:
    print("FAILURE: There is a discrepancy. Initial MLPs are not outputting zero.")
