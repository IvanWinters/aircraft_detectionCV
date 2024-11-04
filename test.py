import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Please check your installation.")
