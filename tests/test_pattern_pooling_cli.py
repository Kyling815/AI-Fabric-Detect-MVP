import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from scripts import inspect_pattern_pooling
from tests.pattern_pooling_fixtures import feature_result


@pytest.mark.parametrize("explicit_flag", [False, True])
def test_cli_requires_explicit_feature_flag_even_if_json_enabled(tmp_path, monkeypatch, capsys, explicit_flag):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    fake = Mock()
    fake.extract_path.return_value = feature_result(explicit_flag)
    factory = Mock(return_value=fake)
    monkeypatch.setattr(inspect_pattern_pooling, "cached_research_extractor", factory)
    args = ["inspect_pattern_pooling.py", "fixture.png", "--config", str(config_path)]
    if explicit_flag:
        args.append("--enable-pattern-aware")
    monkeypatch.setattr(sys, "argv", args)
    inspect_pattern_pooling.main()
    value = json.loads(capsys.readouterr().out)
    assert value["pattern_pooling_enabled"] is explicit_flag
    assert factory.call_args.args[0].enabled is explicit_flag
    assert value["qc_performed"] is False and value["writes_performed"] is False
    assert value["embedding_shape"] == [6, 2]
    assert list(tmp_path.iterdir()) == [config_path]


def test_cli_help_does_not_run_model_or_write_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/inspect_pattern_pooling.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "--enable-pattern-aware" in result.stdout
    assert list(tmp_path.iterdir()) == []
