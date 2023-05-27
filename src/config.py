from torch.cuda import is_available as is_cuda_available
from torch.backends.mps import is_available as is_mps_available


DEVICE = "cuda" if is_cuda_available() else "mps" if is_mps_available() else "cpu"

if __name__ == "__main__":
    print(f"DEVICE: {DEVICE}")
