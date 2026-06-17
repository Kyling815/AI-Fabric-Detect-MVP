from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ProjectionHeadInfo:
    input_dim: int
    projection_dim: int
    hidden_dim: int
    dropout: float
    checkpoint_path: str


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.classifier = nn.Linear(projection_dim, 1)

    def forward_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.projector(embeddings)
        return F.normalize(projected, p=2, dim=1)

    def forward_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.forward_features(embeddings)
        return self.classifier(projected).squeeze(1)


def save_projection_checkpoint(
    model: ProjectionHead,
    path: Path,
    base_model_name: str,
    use_pretrained_weights: bool,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "projection_dim": model.projection_dim,
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout,
        "base_model_name": base_model_name,
        "use_pretrained_weights": use_pretrained_weights,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_projection_head(path: Path, device: str | torch.device = "cpu") -> tuple[ProjectionHead, ProjectionHeadInfo]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ProjectionHead(
        input_dim=int(checkpoint["input_dim"]),
        projection_dim=int(checkpoint["projection_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, ProjectionHeadInfo(
        input_dim=model.input_dim,
        projection_dim=model.projection_dim,
        hidden_dim=model.hidden_dim,
        dropout=model.dropout,
        checkpoint_path=str(path),
    )


def project_embeddings(
    embeddings: np.ndarray,
    checkpoint_path: Path,
    device: str | torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, ProjectionHeadInfo]:
    if embeddings.size == 0:
        model, info = load_projection_head(checkpoint_path, device=device)
        return np.empty((0, model.projection_dim), dtype=np.float32), info

    model, info = load_projection_head(checkpoint_path, device=device)
    matrix = np.asarray(embeddings, dtype=np.float32)
    projected_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, matrix.shape[0], batch_size):
            batch = torch.from_numpy(matrix[start : start + batch_size]).to(device)
            projected = model.forward_features(batch).cpu().numpy().astype(np.float32)
            projected_batches.append(projected)
    return np.concatenate(projected_batches, axis=0), info
