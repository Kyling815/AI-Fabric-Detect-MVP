import numpy as np

from scripts.train_extended_projection_head import build_pairs


def test_defect_only_company_group_contributes_training_pairs() -> None:
    embeddings = np.eye(4, dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    groups = np.asarray(["external_normal", "external_normal", "company_iqc", "company_iqc"], dtype=object)

    left, right, pair_labels = build_pairs(embeddings, labels, groups, pairs_per_group=2, seed=42)

    assert left.shape == right.shape
    assert left.shape[0] > 0
    assert np.any(pair_labels == 0.0)
