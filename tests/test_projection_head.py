from pathlib import Path

import numpy as np

from backend.app.ai.projection_head import ProjectionHead, load_projection_head, project_embeddings, save_projection_checkpoint


def test_projection_head_checkpoint_roundtrip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "projection.pt"
    model = ProjectionHead(input_dim=4, projection_dim=2, hidden_dim=8, dropout=0.0)

    save_projection_checkpoint(
        model=model,
        path=checkpoint_path,
        base_model_name="unit",
        use_pretrained_weights=True,
    )
    loaded, info = load_projection_head(checkpoint_path, device="cpu")

    assert checkpoint_path.exists()
    assert loaded.input_dim == 4
    assert info.projection_dim == 2
    assert info.checkpoint_path == str(checkpoint_path)


def test_project_embeddings_returns_l2_normalized_projection(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "projection.pt"
    model = ProjectionHead(input_dim=4, projection_dim=2, hidden_dim=8, dropout=0.0)
    save_projection_checkpoint(
        model=model,
        path=checkpoint_path,
        base_model_name="unit",
        use_pretrained_weights=True,
    )

    embeddings = np.ones((3, 4), dtype=np.float32)
    projected, info = project_embeddings(embeddings, checkpoint_path, device="cpu", batch_size=2)

    assert projected.shape == (3, 2)
    assert info.input_dim == 4
    assert np.allclose(np.linalg.norm(projected, axis=1), 1.0, atol=1e-5)
