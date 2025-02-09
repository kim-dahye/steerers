import os
import re
import torch
from torch.utils.data import DataLoader, Dataset

class CustomFeatureDataset(Dataset):
    def __init__(self, path_to_chunks, block_name):
        """
        Custom dataset that preloads activation tensors from .pt files.
        
        Args:
            path_to_chunks (str): Path to the directory containing chunk .pt files.
            block_name (str): Block name to filter relevant .pt files.
        """
        self.activations = []
        self.chunk_files = []

        # Traverse through all child directories and collect relevant .pt files
        for root, _, files in os.walk(path_to_chunks):
            for f in files:
                if f.startswith(block_name) and f.endswith('.pt'):
                    self.chunk_files.append(os.path.join(root, f))

        # Sort chunk files by indices extracted from filenames
        self.chunk_files = sorted(
            self.chunk_files,
            key=lambda x: tuple(map(int, re.search(r'_(\d+)_(\d+)\.pt', os.path.basename(x)).groups()))
            if re.search(r'_(\d+)_(\d+)\.pt', os.path.basename(x)) else (float('inf'), float('inf'))
        )
        
        # Preload all activation chunks into memory
        for chunk_file in self.chunk_files:
            chunk = torch.load(chunk_file, map_location='cpu')
            self.activations.append(chunk.reshape(-1, chunk.shape[-1]))  # Load on CPU to save GPU memory
        
        # Concatenate all activations along the first dimension
        self.activations = torch.cat(self.activations, dim=0)  # Shape: [total_samples, dim]

    def __len__(self):
        """Return the total number of samples."""
        return len(self.activations)

    def __getitem__(self, idx):
        """Retrieve the activation tensor at a specific index."""
        return self.activations[idx].clone().detach()  # Return a clone to avoid in-place modifications


class SDActivationsStore:
    """
    Class for streaming activations from preloaded chunks while training.
    """
    def __init__(self, path_to_chunks, block_name, batch_size):
        self.feature_dataset = CustomFeatureDataset(path_to_chunks, block_name)
        self.feature_loader = DataLoader(self.feature_dataset, batch_size=batch_size, shuffle=True)
        self.loader_iter = iter(self.feature_loader)

    def next_batch(self):
        """Retrieve the next batch of activations."""
        try:
            activations = next(self.loader_iter)
        except StopIteration:
            # Reinitialize the iterator if exhausted
            self.loader_iter = iter(self.feature_loader)
            activations = next(self.loader_iter)
        
        return activations
