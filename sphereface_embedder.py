import torch.nn as nn
import numpy as np

from sphereface_pytorch.net_sphere import sphere20a  # come nel notebook

class SphereFaceEmbedder:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = sphere20a(feature=True).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    @staticmethod
    def _preprocess(aligned_bgr_96x112: np.ndarray) -> torch.Tensor:
        # aligned: HxWxC (112x96x3), BGR
        x = torch.from_numpy(aligned_bgr_96x112).float()
        x = (x - 127.5) / 128.0
        x = x.permute(2, 0, 1)  # CHW
        return x

    @torch.no_grad()
    def embed(self, aligned_bgr_96x112: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_bgr_96x112).unsqueeze(0).to(self.device)
        emb = self.model(x)  # (1,512)
        emb = emb.squeeze(0).detach().cpu().numpy()
        # spesso conviene normalizzare L2 per cosine
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb
