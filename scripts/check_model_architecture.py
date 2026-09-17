import torch

ckpt_path = "artifacts\model-et3oiob9-v74\elg_seed0_16.1.ckpt"

# Load the checkpoint (use map_location='cpu' if you don't have a GPU)
checkpoint = torch.load(ckpt_path, map_location="cpu")

# 1. Print available keys in the checkpoint dictionary
print("Checkpoint Keys:", checkpoint.keys())

# 2. Print hyperparameters (architecture configuration)
if "hyper_parameters" in checkpoint:
  print("\n--- Model Hyperparameters ---")
  for k, v in checkpoint["hyper_parameters"].items():
    print(f"{k}: {v}")

# 3. Print the raw layer names and shapes from the state dictionary
print("\n--- State Dict Layers ---")
for layer_name, tensor in checkpoint["state_dict"].items():
  print(f"{layer_name}: {tensor.shape}")