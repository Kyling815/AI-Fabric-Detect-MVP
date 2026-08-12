from types import SimpleNamespace

from scripts.evaluate_extended_pipeline import choose_recall_first


def test_choose_recall_first_prefers_more_conservative_threshold_on_tie() -> None:
    common = {
        "recall": 1.0,
        "fn": 0,
        "f1": 0.98,
        "precision": 1.0,
        "false_positive_rate": 0.0,
    }
    metrics = [
        SimpleNamespace(**common, threshold=0.03),
        SimpleNamespace(**common, threshold=0.075),
    ]

    assert choose_recall_first(metrics).threshold == 0.075
