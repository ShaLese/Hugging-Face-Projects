import torch
print("PyTorch version:", torch.__version__)
print("Is CUDA available:", torch.cuda.is_available() if hasattr(torch, 'cuda') else "CPU only")
x = torch.rand(5, 3)
print("Random tensor:\n", x)
