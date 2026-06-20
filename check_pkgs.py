import bitsandbytes as bnb
import torch
print(f"bitsandbytes: {bnb.__version__}")
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
