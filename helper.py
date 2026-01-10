import torch
from directory_tree import DisplayTree

def print_data_folder_structure(root_dir, max_depth=1):
    """Print the folder structure of the dataset directory."""
    config_tree = {
        "dirPath": root_dir,
        "onlyDirs": False,
        "maxDepth": max_depth,
        "sortBy": 1,  # Sort by type (files first, then folders)
    }
    DisplayTree(**config_tree)

def get_device() -> torch.device:
    """
    Select the best available device:
    - MPS on Apple Silicon (if available)
    - CUDA GPU on NVIDIA systems (if available)
    - CPU fallback otherwise
    """
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")