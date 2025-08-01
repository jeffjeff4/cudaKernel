# mps_test.py

import torch

# Check if MPS (Metal Performance Shader) is available
if not torch.backends.mps.is_available():
    print("MPS device is not available. Ensure you're on Mac with M1/M2/M3 and PyTorch is properly installed.")
else:
    device = torch.device("mps")

    # Create two random tensors on the MPS (GPU) device
    x = torch.rand(3, 3, device=device)
    y = torch.rand(3, 3, device=device)

    # Perform a simple operation (addition)
    z = x + y

    # Move result to CPU and print
    print("x:\n", x.cpu())
    print("y:\n", y.cpu())
    print("z = x + y:\n", z.cpu())

