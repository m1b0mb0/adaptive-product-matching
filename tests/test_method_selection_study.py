"""Diagnostic tests; temporary fixtures are not research observations."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import study_method_selection as study
from matching import FeatureExtractor, evidence_coverage, evidence_scores

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mask", list(itertools.product([False, True], repeat=4)))
def test_grouped_fixed_coverage_renorm_identity(mask):
    rng = np.random.default_rng(123)
    x = rng.uniform(0, 1, (20, 4))
    x[:, ~np.array(mask)] = np.nan
    f, r = (evidence_scores(x, method) for method in study.METHODS)
    np.testing.assert_allclose(f, evidence_coverage(x) * r, rtol=0, atol=2e-16)


def test_same_coverage_ranking_but_different_coverage_can_reverse():
    x = np.array([[.4, np.nan, np.nan, np.nan], [np.nan, 1., np.nan, np.nan]])
    f, r = (evidence_scores(x, method) for method in study.METHODS)
    assert f[0] > f[1] and r[0] < r[1]
    y = np.array([[.4, np.nan, np.nan, np.nan], [.6, np.nan, np.nan, np.nan]])
    assert np.array_equal(np.argsort(evidence_scores(y, "fixed")),
                          np.argsort(evidence_scores(y, "renorm")))


def test_threshold_equivalence_and_no_evidence_exception():
    x = np.array([[.4, np.nan, np.nan, np.nan],
                  [np.nan, .8, np.nan, np.nan], [np.nan]*4])
    c = evidence_coverage(x)
    f, r = (evidence_scores(x, method) for method in study.METHODS)
    threshold = .13
    expected = np.zeros(3, dtype=bool)
    expected[c > 0] = r[c > 0] >= threshold/c[c > 0]
    assert np.array_equal(f >= threshold, expected)


def test_confusion_undefined_and_empty():
    assert study.confusion([1, 0, 1, 0], [1, 1, 0, 0])["f1"] == .5
    empty = study.confusion([], [])
    assert empty["n"] == 0 and empty["f1"] == 0 and not empty["f1_defined"]
    negatives = study.confusion([0, 0], [0, 1])
    assert not negatives["recall_defined"] and negatives["fpr"] == .5


@pytest.fixture
def cache():
    return study.checked_cache(ROOT)


@pytest.mark.parametrize("mutation", ["pair", "label", "threshold", "score", "duplicate"])
def test_corrupt_pair_alignment_fails(cache, mutation):
    frame, result, _ = cache
    frame = frame.copy()
    ix = frame.index[frame.method == "renorm"][0]
    if mutation == "pair":
        frame.loc[ix, "pair_id"] = "missing"
    elif mutation == "label":
        frame.loc[ix, "label"] = 1-int(frame.loc[ix, "label"])
    elif mutation == "threshold":
        frame.loc[ix, "threshold"] = .99
    elif mutation == "score":
        frame.loc[ix, "score"] = .9999
    else:
        frame = pd.concat([frame, frame.loc[[ix]]], ignore_index=True)
    with pytest.raises(ValueError):
        study.align_predictions(frame, result)


def test_full_cache_replay_without_fit_or_selection(cache, monkeypatch):
    import run_known_methods
    from logistic_baseline import SimilarityLogisticRegression
    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostics must not fit models or select thresholds")
    monkeypatch.setattr(FeatureExtractor, "fit", forbidden)
    monkeypatch.setattr(SimilarityLogisticRegression, "fit", forbidden)
    monkeypatch.setattr(run_known_methods, "select_from_oof", forbidden)
    monkeypatch.setattr(run_known_methods, "best_threshold", forbidden)
    frame, result, _ = cache
    analysis = study.analyze(frame, result)
    assert analysis["checks"]["pairs"] == 1135
    assert analysis["checks"]["mask_metric_replays"] == 60
    assert analysis["checks"]["maximum_identity_absolute_error"] < 1e-14
    overall = next(r for r in analysis["clean_paired"] if r["dimension"] == "overall")
    assert (overall["disagreements"], overall["fixed_only_correct"],
            overall["renorm_only_correct"]) == (16, 10, 6)
    assert len(analysis["disagreements"]) == 16
    # Exact partitions, not overlapping diagnostic tags.
    for dim in ("coverage", "availability_pattern", "group_pattern", "group_count", "label"):
        for method in study.METHODS:
            rows = [r for r in analysis["clean_metrics"]
                    if r["dimension"] == dim and r["method"] == method]
            assert sum(r["n"] for r in rows) == 1135
            for key in ("tp", "fp", "fn", "tn"):
                assert sum(r[key] for r in rows) == result["clean_metrics"][method][key]


def test_possible_test_artifact_is_not_opened(tmp_path):
    path = tmp_path / "docs/final_test/results.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not valid JSON; must never be parsed")
    status = study.test_status_metadata(tmp_path)
    assert status["status"] == "unassessed_possible_artifacts"
    assert status["test_results_read"] is False


def test_outside_study_output_rejected_before_read(monkeypatch):
    def forbidden(*args):
        pytest.fail("Must reject output before reading any cache")
    monkeypatch.setattr(study, "checked_cache", forbidden)
    with pytest.raises(ValueError, match="Outputs"):
        study.main(["--out-dir", "docs/known_methods"])


def test_historical_hash_guard(tmp_path):
    base = tmp_path / study.CACHE
    base.mkdir(parents=True)
    (base / "clean_predictions.csv").write_text("tampered")
    (base / "pipeline_validation.json").write_text(json.dumps(
        {"artifact_sha256": {(study.CACHE/"clean_predictions.csv").as_posix(): "bad"}}))
    with pytest.raises(ValueError, match="Changed development artifact"):
        study.checked_cache(tmp_path)
