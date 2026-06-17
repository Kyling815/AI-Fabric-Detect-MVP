from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from backend.app.ai.model_registry import get_model_spec


@dataclass(frozen=True)
class FeatureExtractorInfo:
    model_name: str
    requested_device: str
    device: str
    use_pretrained_weights: bool
    embedding_dim: int
    model_input_size: int
    l2_normalized: bool


class CNNFeatureExtractor:
    def __init__(
        self,
        model_name: str = "resnet18",
        device: str = "auto",
        use_pretrained_weights: bool = False,
        model_input_size: int = 224,
        l2_normalize: bool = True,
    ) -> None:
        self.spec = get_model_spec(model_name)
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.use_pretrained_weights = use_pretrained_weights
        self.model_input_size = model_input_size
        self.l2_normalize = l2_normalize
        self.model = self.spec.build(use_pretrained_weights).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((model_input_size, model_input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @property
    def info(self) -> FeatureExtractorInfo:
        return FeatureExtractorInfo(
            model_name=self.spec.name,
            requested_device=self.requested_device,
            device=str(self.device),
            use_pretrained_weights=self.use_pretrained_weights,
            embedding_dim=self.spec.embedding_dim,
            model_input_size=self.model_input_size,
            l2_normalized=self.l2_normalize,
        )

    def extract(self, patches: Iterable[Image.Image], batch_size: int = 16) -> np.ndarray:
        patch_list = list(patches)
        if not patch_list:
            return np.empty((0, self.spec.embedding_dim), dtype=np.float32)
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        embeddings: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(patch_list), batch_size):
                batch_images = patch_list[start : start + batch_size]
                batch_tensor = torch.stack([self._prepare(image) for image in batch_images]).to(self.device)
                batch_embeddings = self.model(batch_tensor)
                if batch_embeddings.ndim > 2:
                    batch_embeddings = torch.flatten(batch_embeddings, start_dim=1)
                if self.l2_normalize:
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy().astype(np.float32))

        return np.concatenate(embeddings, axis=0)

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB"))

    def _resolve_device(self, requested_device: str) -> torch.device:
        normalized = requested_device.lower()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if normalized == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(normalized)
