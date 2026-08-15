from training.evaluate import validate_metrics


def test_quality_gate_accepts_valid_metrics():
    metrics = {
        "f1": 0.80,
        "roc_auc": 0.78,
    }

    failures = validate_metrics(
        metrics=metrics,
        min_f1=0.75,
        min_roc_auc=0.75,
    )

    assert failures == []


def test_quality_gate_rejects_low_f1():
    metrics = {
        "f1": 0.70,
        "roc_auc": 0.80,
    }

    failures = validate_metrics(
        metrics=metrics,
        min_f1=0.75,
        min_roc_auc=0.75,
    )

    assert len(failures) == 1
    assert "F1" in failures[0]


def test_quality_gate_rejects_low_roc_auc():
    metrics = {
        "f1": 0.80,
        "roc_auc": 0.70,
    }

    failures = validate_metrics(
        metrics=metrics,
        min_f1=0.75,
        min_roc_auc=0.75,
    )

    assert len(failures) == 1
    assert "ROC-AUC" in failures[0]
