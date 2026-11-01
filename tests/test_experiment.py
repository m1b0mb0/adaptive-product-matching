"""Small fixtures verify evaluation data flow, not scientific performance."""
import copy

import numpy as np
import pytest

import run_experiment
from matching import FeatureExtractor


def pair(left, right, label, title_left="Camera ABC123", title_right="Camera ABC123"):
    return {"id_left": left, "id_right": right, "title_left": title_left,
            "title_right": title_right, "label": label}


def test_evaluation_refuses_cross_split_offer_overlap_before_fitting(monkeypatch):
    def forbidden_fit(*args):
        pytest.fail("Leaking splits must be rejected before fitting")

    monkeypatch.setattr(FeatureExtractor, "fit", forbidden_fit)
    with pytest.raises(ValueError, match="[Oo]ffer ID overlap"):
        run_experiment.evaluate([pair("a", "b", 1)], [pair("a", "c", 0)], [pair("d", "e", 0)])


def test_fitting_and_validation_selection_do_not_use_test_labels(monkeypatch):
    train = [pair("a", "b", 1), pair("c", "d", 0, title_right="Monitor XYZ987")]
    valid = [pair("e", "f", 1), pair("g", "h", 0, title_right="Monitor XYZ987")]
    test = [pair("i", "j", 1), pair("k", "l", 0, title_right="Monitor XYZ987")]
    original_fit = FeatureExtractor.fit
    original_info = run_experiment.train_informativeness
    fit_calls, info_calls = [], []

    def checked_fit(self, pairs):
        assert pairs is train
        fit_calls.append(pairs)
        return original_fit(self, pairs)

    def checked_info(features, y):
        np.testing.assert_array_equal(y, run_experiment.labels(train))
        info_calls.append(y)
        return original_info(features, y)

    monkeypatch.setattr(FeatureExtractor, "fit", checked_fit)
    monkeypatch.setattr(run_experiment, "train_informativeness", checked_info)
    monkeypatch.setattr(run_experiment, "METHODS", ("text", "adaptive"))
    monkeypatch.setattr(run_experiment, "THRESHOLDS", np.array([.25, .5, .75]))
    first = run_experiment.evaluate(train, valid, test)
    flipped = copy.deepcopy(test)
    for row in flipped:
        row["label"] = 1 - row["label"]
    second = run_experiment.evaluate(train, valid, flipped)
    assert first["validation_selected"] == second["validation_selected"]
    assert first["train_discriminativeness"] == second["train_discriminativeness"]
    assert first["test_metrics"] != second["test_metrics"]
    assert len(fit_calls) == len(info_calls) == 2
