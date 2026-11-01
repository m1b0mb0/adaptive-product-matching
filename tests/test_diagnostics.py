"""Diagnostic isolation, provenance and source-aware removals on test fixtures."""
import copy
import gzip
import json
import sys

import numpy as np
import pytest

import diagnose_method as diagnostic
import run_experiment
from matching import FeatureExtractor, evidence_scores, train_informativeness


def pair(prefix, label=1):
    return {"pair_id": f"{prefix}1#{prefix}2", "id_left": f"{prefix}1", "id_right": f"{prefix}2",
            "cluster_id_left": "a", "cluster_id_right": "a" if label else "b",
            "title_left": "Camera model1000", "title_right": "Camera model1000" if label else "Camera model2000",
            "description_left": "Indoor camera", "description_right": "Indoor camera",
            "brand_left": "Acme", "brand_right": "Acme", "label": label,
            "is_hard_negative": not bool(label)}


def historical(train):
    extractor = FeatureExtractor().fit(train)
    info = train_informativeness(extractor.transform(train).matrix, run_experiment.labels(train))
    return {"train_discriminativeness": dict(zip(diagnostic.FIELDS, info.tolist())),
            "validation_selected": {m: {"gamma": .5, "penalty": 0., "threshold": .15 if m == "adaptive" else .14}
                                    for m in diagnostic.METHODS},
            "protocol": {"prior_score_reference": .2,
                         "missingness": {"rates": [0, .5], "seeds": [11, 22, 33]}}}


def test_routes_non_tune_before_label_or_attribute_validation(tmp_path):
    retained = pair("t")
    ignored = pair("r")
    ignored.update(cluster_id_left="report", cluster_id_right="report", label="do not inspect",
                   title_left={"deliberately": "invalid title payload"})
    path = tmp_path / "valid.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(ignored) + "\n" + json.dumps(retained) + "\n")
    assert diagnostic.read_tune_only(path, {"a"}) == [retained]


def test_validates_retained_tune_payload(tmp_path):
    row = pair("bad")
    row["label"] = 3
    path = tmp_path / "valid.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid binary label"):
        diagnostic.read_tune_only(path, {"a"})


def test_source_aware_title_removal_and_identifier_only_removal():
    matrix = np.array([[.8, .4, 1., .5]])
    title_removed = diagnostic.remove_channels(matrix, "title_family")
    assert np.isnan(title_removed[:, [0, 3]]).all()
    np.testing.assert_array_equal(title_removed[:, [1, 2]], matrix[:, [1, 2]])
    id_removed = diagnostic.remove_channels(matrix, "identifier_only")
    assert np.isnan(id_removed[0, 3])
    assert id_removed[0, 0] == matrix[0, 0]
    np.testing.assert_array_equal(matrix, [[.8, .4, 1., .5]])


def test_analyze_fits_train_only_never_selects_thresholds(monkeypatch):
    train = [pair("train", 1), pair("negative", 0)]
    tune = [pair("tune", 1), pair("other", 0)]
    old = historical(train)
    frozen = copy.deepcopy(old)
    original_fit = FeatureExtractor.fit
    calls = []

    def checked_fit(self, rows):
        assert rows is train
        calls.append("train_only")
        return original_fit(self, rows)

    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostics must not choose thresholds")

    monkeypatch.setattr(FeatureExtractor, "fit", checked_fit)
    monkeypatch.setattr(run_experiment, "best_threshold", forbidden)
    result, cases, removals = diagnostic.analyze(train, tune, old)
    assert calls == ["train_only"]
    assert old == frozen
    assert set(result["splits"]) == {"TRAIN", "TUNE"}
    assert all(row["split"] in {"TRAIN", "TUNE"} for row in cases + removals)
    json.dumps(result, allow_nan=False)


def test_analyze_rejects_train_tune_overlap_before_fit(monkeypatch):
    train = [pair("same")]
    monkeypatch.setattr(FeatureExtractor, "fit", lambda *args: pytest.fail("Must reject before fit"))
    with pytest.raises(ValueError, match="overlap"):
        diagnostic.analyze(train, train, {})


def test_changed_v1_source_is_rejected(monkeypatch):
    monkeypatch.setattr(diagnostic, "sha256", lambda path: "changed")
    with pytest.raises(ValueError, match="source hash differs"):
        diagnostic.check_frozen_sources({"metadata": {"source_sha256": {"matching.py": "historical"}}})


def test_changed_train_informativeness_is_rejected():
    train, tune = [pair("train")], [pair("tune")]
    old = historical(train)
    old["train_discriminativeness"]["title"] += .01
    with pytest.raises(ValueError, match="informativeness differs"):
        diagnostic.analyze(train, tune, old)


def test_reserved_path_in_artifact_is_rejected_before_dataset_access(tmp_path, monkeypatch):
    path = tmp_path / "historical.json"
    path.write_text(json.dumps({"metadata": {"train_path": "official_train_small.json.gz",
                                             "validation_path": "official_100un_gs.json.gz"}}), encoding="utf-8")
    monkeypatch.setattr(diagnostic, "check_frozen_sources", lambda artifact: {})
    monkeypatch.setattr(diagnostic, "sha256", lambda *args: pytest.fail("Dataset must not be opened"))
    monkeypatch.setattr(sys, "argv", ["diagnose_method.py", "--historical", str(path)])
    with pytest.raises(ValueError, match="official train and validation"):
        diagnostic.main()


def test_no_test_argument_accepted(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["diagnose_method.py", "--test", "reserved.json.gz"])
    monkeypatch.setattr(diagnostic, "read_pairs", lambda *args: pytest.fail("No data access permitted"))
    with pytest.raises(SystemExit) as exc:
        diagnostic.main()
    assert exc.value.code == 2


def test_low_coverage_can_lift_below_threshold_mean_with_prior_above_threshold():
    # title-only group has C=.7 and a small positive text similarity.
    matrix = np.array([[.142, np.nan, np.nan, np.nan]])
    mean = evidence_scores(matrix, "adaptive", use_coverage=False)[0]
    value = evidence_scores(matrix, "adaptive", gamma=.5, prior=.2)[0]
    assert mean < .15 <= value


def test_missingness_is_not_same_as_observed_identifier_disagreement():
    matrix = np.array([[.7, np.nan, np.nan, 0.], [.7, np.nan, np.nan, np.nan]])
    observed, missing = evidence_scores(matrix, "renorm")
    assert observed < missing


def test_affine_reconstruction_is_exact_but_renorm_not_identical():
    train = [pair("train", 1), pair("negative", 0)]
    old = historical(train)
    matrix = np.array([[.1, .5, 1., .3], [.4, np.nan, np.nan, np.nan], [.9, .1, 0., 0.]])
    info = np.array([1.1, .8, 1.5, 1.4])
    adaptive = diagnostic.score(matrix, "adaptive", info, old)
    renorm = diagnostic.score(matrix, "renorm", info, old)
    mean = evidence_scores(matrix, "adaptive", info, gamma=.5, use_coverage=False)
    result = diagnostic.relationships(matrix, adaptive, renorm, mean, old)
    assert result["reconstruction_max_abs_error"] < 1e-12
    assert result["score_max_abs_difference"] > 0
