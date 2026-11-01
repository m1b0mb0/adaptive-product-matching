"""Validation isolation and frozen operating points on small control fixtures."""
import copy
import json
import sys

import numpy as np
import pytest

import run_experiment
import run_validation as runner
from audit_wdc import offers
from matching import FeatureExtractor


def make_pairs(prefix, n_clusters=12):
    def offer(cluster, side):
        return {"id": f"{prefix}-{cluster}-{side}", "cluster_id": cluster,
                "title": f"Camera MODEL{cluster:04d} {side}", "description": "camera indoor",
                "brand": "Maker" if cluster % 2 else None}

    def pair(left, right):
        return {**{f"{key}_left": value for key, value in left.items()},
                **{f"{key}_right": value for key, value in right.items()},
                "label": int(left["cluster_id"] == right["cluster_id"])}

    return ([pair(offer(i, 0), offer(i, 1)) for i in range(n_clusters)]
            + [pair(offer(i, 0), offer(j, 0)) for i in range(n_clusters) for j in range(i + 1, n_clusters)])


@pytest.fixture
def fast_pipeline(monkeypatch):
    monkeypatch.setattr(runner, "METHODS", runner.MAIN_METHODS + runner.ABLATIONS)
    monkeypatch.setattr(runner, "GAMMAS", (0.5, 1.0))
    monkeypatch.setattr(run_experiment, "THRESHOLDS", np.array([0.25, 0.5, 0.75]))


def test_partition_is_order_independent_cluster_and_offer_disjoint():
    rows = make_pairs("v")
    tune, report, info = runner.partition_validation(rows)
    reversed_tune, reversed_report, reversed_info = runner.partition_validation(list(reversed(rows)))
    assert offers(tune) == offers(reversed_tune)
    assert offers(report) == offers(reversed_report)
    assert not offers(tune) & offers(report)
    assert info["tune_cluster_ids"] == reversed_info["tune_cluster_ids"]
    assert not set(info["tune_cluster_ids"]) & set(info["report_cluster_ids"])
    assert len(tune) + len(report) + info["excluded_crossing_pairs"] == len(rows)
    assert info["excluded_labels"]["1"] == 0


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan")])
def test_partition_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="tune_fraction"):
        runner.partition_validation(make_pairs("v"), tune_fraction=fraction)


def test_partition_requires_cluster_metadata():
    rows = make_pairs("v")
    del rows[0]["cluster_id_left"]
    with pytest.raises(ValueError, match="Cluster IDs"):
        runner.partition_validation(rows)


def test_complete_validation_overlap_rejected_before_fit(monkeypatch):
    def forbidden(*args):
        pytest.fail("Leaking data must be rejected before fit")
    monkeypatch.setattr(FeatureExtractor, "fit", forbidden)
    with pytest.raises(ValueError, match="Offer ID overlap"):
        runner.evaluate_validation(make_pairs("same"), make_pairs("same"))


def test_complete_validation_integrity_is_checked_even_for_crossing_pair(monkeypatch):
    rows = make_pairs("v")
    tune, report, _ = runner.partition_validation(rows)
    retained = {id(row) for row in tune + report}
    crossing = next(row for row in rows if id(row) not in retained)
    rows.append(copy.deepcopy(crossing))
    with pytest.raises(ValueError, match="Complete-split integrity"):
        runner.evaluate_validation(make_pairs("t"), rows)


def test_unseen_mode_refuses_train_cluster_overlap():
    with pytest.raises(ValueError, match="Unseen validation"):
        runner.evaluate_validation(make_pairs("t"), make_pairs("v"), validation_mode="unseen")


def test_fit_and_informativeness_use_train_only(monkeypatch, fast_pipeline):
    train, validation = make_pairs("t"), make_pairs("v")
    fit = FeatureExtractor.fit
    estimate = runner.train_informativeness
    calls = []

    def checked_fit(self, rows):
        assert rows is train
        calls.append("fit")
        return fit(self, rows)

    def checked_info(features, y):
        np.testing.assert_array_equal(y, runner.labels(train))
        assert len(features) == len(train)
        calls.append("info")
        return estimate(features, y)

    monkeypatch.setattr(FeatureExtractor, "fit", checked_fit)
    monkeypatch.setattr(runner, "train_informativeness", checked_info)
    result = runner.evaluate_validation(train, validation)
    assert calls == ["fit", "info"]
    assert result["protocol"]["final_test_evaluated"] is False
    assert "test_metrics" not in result
    json.dumps(result, allow_nan=False)


def test_report_labels_cannot_change_selected_parameters(monkeypatch, fast_pipeline):
    train, validation = make_pairs("t"), make_pairs("v")
    first = runner.evaluate_validation(train, validation)
    _, report, _ = runner.partition_validation(validation)
    report_ids = {id(pair) for pair in report}
    original_labels = runner.labels

    def counterfactual_report_labels(rows):
        values = original_labels(rows)
        return 1 - values if {id(pair) for pair in rows} == report_ids else values

    monkeypatch.setattr(runner, "labels", counterfactual_report_labels)
    second = runner.evaluate_validation(train, validation)
    assert first["validation_selected"] == second["validation_selected"]
    assert first["train_discriminativeness"] == second["train_discriminativeness"]
    assert first["validation_metrics"] != second["validation_metrics"]


def test_missingness_uses_frozen_thresholds_and_zero_rate_equals_clean(fast_pipeline):
    result = runner.evaluate_validation(make_pairs("t"), make_pairs("v"))
    for row in result["missingness"]["rows"]:
        config = result["validation_selected"][row["method"]]
        assert row["threshold"] == config["threshold"]
        assert row["gamma"] == config["gamma"]
        assert row["penalty"] == config["penalty"]
        assert 0 <= row["effective_removed_fraction"] <= 1
        assert sum(row[key] for key in ("tn", "fp", "fn", "tp")) == row["n"]
        if row["missing_rate"] == 0:
            assert row["effective_removed_fraction"] == 0
            assert row["f1"] == result["validation_metrics"][row["method"]]["f1"]
        if row["method"] == "text":
            assert row["f1"] == result["validation_metrics"]["text"]["f1"]
    full = result["validation_selected"]["adaptive"]
    for method in runner.ABLATIONS:
        assert result["validation_selected"][method]["gamma"] == full["gamma"]
        assert result["validation_selected"][method]["penalty"] == full["penalty"]
        assert result["ablation_fixed_threshold_metrics"][method]["threshold"] == full["threshold"]


def test_cli_rejects_test_argument_before_any_data_access(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_validation.py", "--train", "not-opened", "--valid", "not-opened", "--test", "reserved"])
    monkeypatch.setattr(runner, "read_pairs", lambda *args: pytest.fail("No test data access"))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_paired_error_counts_have_explicit_directions():
    y = np.array([0, 0, 1, 1])
    adaptive = np.array([0, 1, 1, 0])
    comparator = np.array([1, 0, 0, 1])
    values = runner.paired_errors(y, adaptive, comparator, "baseline")
    assert values["rescued_fp"] == values["rescued_fn"] == 1
    assert values["introduced_fp"] == values["introduced_fn"] == 1
    assert values["net_correct_change"] == 0
