import torch
import torch.nn as nn
import numpy as np

from sphereface_pytorch.net_sphere import sphere20a


class SphereFaceEmbedder:
    def __init__(self, weights_path: str, device: str = "cpu"):
        # Set the computation device (CPU or GPU)
        self.device = torch.device(device)
        # Initialize the SphereFace network architecture
        self.model = sphere20a(feature=True).to(self.device)
        # Load pre-trained model weights
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        # Set model to evaluation mode
        self.model.eval()

    @staticmethod
    def _preprocess(aligned_bgr_96x112: np.ndarray) -> torch.Tensor:
        # Input: aligned image HxWxC (112x96x3), BGR format
        x = torch.from_numpy(aligned_bgr_96x112).float()
        # Normalize pixel values to range [-1, 1]
        x = (x - 127.5) / 128.0
        # Rearrange dimensions to Channel-First (CHW) for PyTorch
        x = x.permute(2, 0, 1)  # CHW
        return x

    @torch.no_grad()
    def embed(self, aligned_bgr_96x112: np.ndarray) -> np.ndarray:
        # Preprocess and add batch dimension (1, C, H, W)
        x = self._preprocess(aligned_bgr_96x112).unsqueeze(0).to(self.device)
        # Forward pass to get features
        emb = self.model(x)  # Output shape: (1, 512)
        # Convert tensor back to numpy array
        emb = emb.squeeze(0).detach().cpu().numpy()
        # Apply L2 normalization (crucial for Cosine Similarity)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb
