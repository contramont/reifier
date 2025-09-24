import torch
import torch.nn as nn

# Minimal reproduction of a PyTorch bug.
# The bug is triggered by highly regular, non-random weights.
# It causes nn.Linear to produce wrong outputs if bias=False and batch size > 1.
# The bug was detected in a MacOS VM, running PyTorch 2.8.0. It did not appear in Google Colab.

# --- 1. Parameters to trigger the bug ---
HIDDEN_SIZE = 1024
BATCH_SIZE = 64
NUM_LAYERS_DEEP = 100
DTYPE = torch.float32


# --- 2. The key components of the bug ---
# A simple binary activation to keep inputs structured and integer-like.
def step_activation(x: torch.Tensor) -> torch.Tensor:
    """Activations become 0.0 or 1.0."""
    return (x > 0.5).to(x.dtype)


# --- 3. Crafting highly regular, non-random weights to trigger the bug ---
buggy_layer = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False, dtype=DTYPE)
with torch.no_grad():
    pattern = torch.tensor([1.0, -1.0], dtype=DTYPE)
    weight_matrix = pattern.repeat(HIDDEN_SIZE, HIDDEN_SIZE // 2).contiguous()
    buggy_layer.weight.copy_(weight_matrix)

# --- 4. Generate a single, simple, binary input vector ---
# We will process this vector both by itself (batch_size=1) and as part of a batch.
x_single = torch.randint(0, 2, (1, HIDDEN_SIZE), dtype=DTYPE)
x_batch = x_single.repeat(BATCH_SIZE, 1)

# --- 5. Simulate the deep network forward pass ---
# We apply the same layer and activation repeatedly to both inputs.
for i in range(NUM_LAYERS_DEEP):
    x_single = step_activation(buggy_layer(x_single))  # correct
    x_batch = step_activation(buggy_layer(x_batch))  # buggy

# --- 6. Check for divergence ---
# After the loop, the results *should* be identical, as the input was the same.
result_from_batch = x_batch[0]
result_from_single = x_single.squeeze(0)
max_difference = torch.abs(result_from_batch - result_from_single).max().item()

print(f"PyTorch Version: {torch.__version__}")
if max_difference > 0:
    print("\n🔴 BUG REPRODUCED: Outputs for the same input vector are different.")
    print(f"   Max absolute difference: {max_difference}")
    diff_indices = (result_from_batch != result_from_single).nonzero(as_tuple=True)[0]
    first_diff_idx = int(diff_indices[0].item())
    print(f"   Example mismatch at index {first_diff_idx}:")
    print(f"     - From batch calculation:  {result_from_batch[first_diff_idx].item()}")
    print(
        f"     - From single calculation: {result_from_single[first_diff_idx].item()}"
    )
else:
    print("\n🟢 Bug not reproduced. The outputs match correctly.")
