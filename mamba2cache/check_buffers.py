import torch
import json

# Load the saved model
model_path = "../refactor/models/216992/best_model/pytorch_model.bin"
# model_path = "../traineagle3/models/1epoch/best_model/pytorch_model.bin"
state_dict = torch.load(model_path, map_location="cpu")

# Check if buffers are present
print("Checking for buffers in saved state dict:")
print(f"  'd2t' present: {'d2t' in state_dict}")
print(f"  't2d' present: {'t2d' in state_dict}")

# List all keys to see what's saved
print("\nAll keys in state dict:")
for key in sorted(state_dict.keys()):
    if 'd2t' in key or 't2d' in key:
        print(f"  ✓ {key}: shape={state_dict[key].shape}, dtype={state_dict[key].dtype}")
    elif 'buffer' in key.lower():
        print(f"  → {key}: shape={state_dict[key].shape}")

# Check buffer values and count unique values
if 'd2t' in state_dict:
    d2t = state_dict['d2t']
    print(f"\nd2t buffer:")
    print(f"  Shape: {d2t.shape}")
    print(f"  Dtype: {d2t.dtype}")
    print(f"  First 100 values: {d2t[:100]}")

if 't2d' in state_dict:
    t2d = state_dict['t2d']
    print(f"\nt2d buffer:")
    print(f"  Shape: {t2d.shape}")
    print(f"  Dtype: {t2d.dtype}")
    print(f"  Sum (should be draft_vocab_size): {t2d.sum().item()}")
    print(f"  First 20 values: {t2d[:20]}")