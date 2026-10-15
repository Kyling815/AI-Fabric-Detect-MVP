from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.ai.pattern_pooling import PatternAwareDenseExtractor
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime


def test_off_wrapper_preserves_current_dense_embeddings_and_index_scores(phase1_runtime):
    extractor = phase1_runtime.extractor
    # Phase 1's injectable fake has no real module; add a stateless eval module for fingerprinting.
    import torch
    extractor.model = torch.nn.Identity().eval()
    extractor.adapter = None
    features = extractor.extract_path(Path("fixture.png"), image_max_size=24)
    wrapper = PatternAwareDenseExtractor(extractor)
    result = wrapper.extract_path(Path("fixture.png"), image_max_size=24)
    np.testing.assert_array_equal(result.features.embeddings, features.embeddings)
    store = FaissIndexStore("cosine")
    index = store.load(Path(phase1_runtime.reference["faiss_index_path"]))
    np.testing.assert_array_equal(store.search(index, result.features.embeddings).scores,
                                  store.search(index, features.embeddings).scores)
    assert result.geometry.feature_dim == 2 and result.geometry.algorithm == "dense_baseline_v1"


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_production_api_does_not_use_pooling_ablation_and_keeps_golden_response(phase1_runtime, case):
    phase1_runtime.use_case(case)
    with patch.object(PatternAwareDenseExtractor, "extract_path", side_effect=AssertionError("production must remain baseline")):
        response = phase1_runtime.upload()
    assert response.status_code == 201
    snapshot = baseline()
    assert normalize_response(response.json()) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    saved = phase1_runtime.client.get(f"/api/inspections/{response.json()['inspection_id']}")
    assert normalize_response(saved.json()) == {**snapshot["common"], **snapshot["cases"][case]["get_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == snapshot["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == snapshot["response_schema"]
