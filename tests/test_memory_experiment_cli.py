import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory
from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryIdentity
from tests.memory_experiment_fixtures import bank, identity


def cli_inputs(tmp_path, *, overrides=None, source_corrupt=False):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "synthetic_good.png"
    source.write_bytes(b"synthetic fixture only; no real dataset")
    provenance = identity().model_dump(mode="json")
    provenance["source_good_images"][0]["path"] = str(source)
    provenance["source_good_images"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_corrupt:
        source.write_bytes(b"changed source")
    options = {"enabled": False, "distance_metric": "current", "coreset_ratio": 1.0,
               "selector": "seeded_random", "random_seed": 42, **(overrides or {})}
    values = tmp_path / "good_embeddings.npy"
    np.save(values, bank(), allow_pickle=False)
    query = tmp_path / "query.npy"
    np.save(query, bank()[:3], allow_pickle=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(options), encoding="utf-8")
    identity_path = tmp_path / "provenance.json"
    identity_path.write_text(json.dumps(provenance), encoding="utf-8")
    output = tmp_path / "new_experiment"
    command = [sys.executable, str(root / "scripts" / "experiment_anomaly_memory.py"),
               "--embeddings", str(values), "--provenance", str(identity_path), "--config", str(config_path),
               "--output-dir", str(output), "--query-embeddings", str(query)]
    return command, output, MemoryIdentity.model_validate(provenance), options


@pytest.mark.parametrize("enabled_in_json", [False, True])
def test_cli_default_is_read_only_plan_even_if_json_enabled(tmp_path, enabled_in_json):
    command, output, _, _ = cli_inputs(tmp_path, overrides={"enabled": enabled_in_json})
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    assert result["status"] == "PLAN_ONLY" and not result["writes_performed"]
    assert not output.exists()
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}


@pytest.mark.parametrize("metric,ratio", [("current", 1.0), ("cosine", 0.25), ("l2", 0.05)])
def test_cli_explicit_opt_in_creates_only_new_research_memory_and_reload_matches(tmp_path, metric, ratio):
    command, output, provenance, options = cli_inputs(tmp_path, overrides={"distance_metric": metric, "coreset_ratio": ratio})
    completed = subprocess.run([*command, "--enable-experiment"], capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    assert result["status"] == "EXPERIMENT_ONLY"
    assert result["qc_performed"] is False and result["registry_modified"] is False
    assert result["metadata"]["memory_bank_count"] == int(100 * ratio)
    memory = ExperimentalAnomalyMemory.load(output, expected_identity=provenance,
                                            config=MemoryExperimentConfig.model_validate({**options, "enabled": True}))
    query = memory.search(bank()[:3])
    np.testing.assert_array_equal(query.anomaly_distances, result["query_result"]["distances"])
    assert set(p.name for p in output.iterdir()) == {"memory_bank.npy", "memory_bank.faiss", "selected_indices.npy", "reference.json"}


def test_cli_wrong_source_hash_fails_without_creating_output(tmp_path):
    command, output, _, _ = cli_inputs(tmp_path, source_corrupt=True)
    completed = subprocess.run([*command, "--enable-experiment"], capture_output=True, text=True)
    assert completed.returncode != 0 and "Source Good hash mismatch" in completed.stderr
    assert not output.exists()


def test_cli_existing_output_is_not_overwritten(tmp_path):
    command, output, _, _ = cli_inputs(tmp_path)
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_bytes(b"user file")
    completed = subprocess.run([*command, "--enable-experiment"], capture_output=True, text=True)
    assert completed.returncode != 0
    assert marker.read_bytes() == b"user file" and list(output.iterdir()) == [marker]
