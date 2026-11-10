"""Mathematical and integration properties; temporary fixtures are not research data."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

import matching
from adaptive_corrected import (METHOD_SPECIFICATION, FrozenCorrectedScorer,
                                corrected_scores)
from matching import FeatureExtractor, evidence_coverage, evidence_scores


HISTORICAL_MATCHING_LF_SHA256 = "038f58693212c6eb04ed8a55e050ba43ea34270c0277f849fac953b92d6e72ab"


def test_all_missing_is_exact_zero_and_input_unchanged():
    matrix = np.full((3, 4), np.nan)
    original = matrix.copy()
    result = corrected_scores(matrix, np.ones(4))
    np.testing.assert_array_equal(result, np.zeros(3))
    np.testing.assert_array_equal(matrix, original)
    assert corrected_scores(np.empty((0, 4)), np.ones(4)).shape == (0,)


def test_title_and_derived_identifier_use_only_one_group_budget():
    matrix = np.array([[1., np.nan, np.nan, np.nan],
                       [np.nan, np.nan, np.nan, 1.],
                       [1., np.nan, np.nan, 1.]])
    np.testing.assert_allclose(corrected_scores(matrix, np.ones(4)), [.7, .7, .7])
    np.testing.assert_allclose(evidence_coverage(matrix), [.7, .7, .7])


def test_observed_zero_is_not_missing_and_removing_conflict_can_increase_score():
    # Valid q_title=1, q_description=1.5, with the derived identifier absent.
    matrix = np.array([[1., 0., np.nan, np.nan],
                       [1., np.nan, np.nan, np.nan]])
    result = corrected_scores(matrix, np.array([1., 1.5, 1., 1.]))
    np.testing.assert_allclose(evidence_coverage(matrix), [.85, .7])
    np.testing.assert_allclose(result, [.85 * .7 / (.7 + .15 * 1.5), .7])
    assert result[1] > result[0]  # No unsupported global monotonicity claim.
    assert result[0] != pytest.approx(evidence_scores(matrix, "fixed")[0])


def test_no_positive_prior_for_observed_disagreement():
    matrix = np.array([[np.nan, np.nan, 0., np.nan]])
    assert corrected_scores(matrix, np.ones(4))[0] == 0.
    historical = evidence_scores(matrix, "adaptive", np.ones(4), gamma=1., prior=.2, penalty=0.)
    assert historical[0] == pytest.approx(.17)
    assert historical[0] >= .15 > corrected_scores(matrix, np.ones(4))[0]


def test_uniform_effective_quality_exactly_recovers_grouped_fixed_baseline():
    rng = np.random.default_rng(20261110)
    matrix = rng.random((150, 4))
    matrix[rng.random(matrix.shape) < .4] = np.nan
    # I * inherited manual reliability is constant across every channel.
    information = 1. / np.array([1., 1., 1., .7])
    np.testing.assert_allclose(corrected_scores(matrix, information),
                               evidence_scores(matrix, "fixed"), rtol=1e-14, atol=1e-14)


def test_score_is_bounded_by_available_group_budget_for_valid_quality():
    rng = np.random.default_rng(20261110)
    matrix = rng.random((1000, 4))
    matrix[rng.random(matrix.shape) < .45] = np.nan
    matrix[0] = np.nan
    result = corrected_scores(matrix, np.array([.25, .5, 1.5, 1.1]))
    coverage = evidence_coverage(matrix)
    assert np.isfinite(result).all()
    assert (result >= 0).all()
    assert (result <= coverage + 1e-14).all()
    assert (coverage <= 1).all()


def test_repeated_scoring_is_reproducible_and_does_not_mutate_information():
    matrix = np.array([[.4, .8, 0., np.nan], [.7, np.nan, 1., .2]])
    information = np.array([1.2, .4, .5, 1.1])
    original = information.copy()
    np.testing.assert_array_equal(corrected_scores(matrix, information),
                                  corrected_scores(matrix, information))
    np.testing.assert_array_equal(information, original)


@pytest.mark.parametrize("information", [None, [], [1, 1, 1], [1, 1, 1, 0],
                                         [1, -1, 1, 1], [1, np.nan, 1, 1],
                                         [np.inf, 1, 1, 1]])
def test_train_information_must_be_explicit_finite_positive_four_vector(information):
    with pytest.raises(ValueError):
        corrected_scores(np.ones((1, 4)), information)


@pytest.mark.parametrize("matrix", [np.ones(4), np.ones((2, 3)),
                                    [[1, 1, 1, np.inf]], [[1, -.01, 1, 1]],
                                    [[1, 1.01, 1, 1]]])
def test_invalid_similarity_inputs_are_rejected(matrix):
    with pytest.raises(ValueError):
        corrected_scores(matrix, np.ones(4))


@pytest.mark.parametrize("knob", ["gamma", "prior", "penalty", "reliability"])
def test_fixed_formula_exposes_no_tunable_parameters(knob):
    with pytest.raises(TypeError):
        corrected_scores(np.ones((1, 4)), np.ones(4), **{knob: 0})


def test_method_specification_freezes_the_one_formula():
    assert METHOD_SPECIFICATION["formula"] == "S_corr = C * M_q; no available group -> 0"
    assert METHOD_SPECIFICATION["gamma"] == 1.
    assert METHOD_SPECIFICATION["prior"] == METHOD_SPECIFICATION["identifier_penalty"] == 0.
    assert METHOD_SPECIFICATION["manual_reliability"] == [1., 1., 1., .7]
    assert METHOD_SPECIFICATION["group_weights"] == [.7, .15, .15]
    assert METHOD_SPECIFICATION["global_masking_monotonicity_claimed"] is False
    assert METHOD_SPECIFICATION["calibration_established"] is False


def test_historical_implementation_is_unchanged_ignoring_checkout_line_endings():
    source = Path(matching.__file__).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(source).hexdigest() == HISTORICAL_MATCHING_LF_SHA256


@pytest.fixture
def fitted_extractor():
    rows = [{"id_left": "train-a", "id_right": "train-b",
             "title_left": "Camera AX12345 black", "title_right": "AX12345 Camera",
             "description_left": "Photo camera", "description_right": "Photo camera",
             "brand_left": "Example", "brand_right": "Example"}]
    return FeatureExtractor().fit(rows)


def test_frozen_scorer_rejects_unfitted_representation():
    with pytest.raises(ValueError, match="Fit"):
        FrozenCorrectedScorer(FeatureExtractor(), np.ones(4))


def test_frozen_scorer_copies_information_and_does_not_fit_on_queries(fitted_extractor, monkeypatch):
    information = np.array([1., 1.2, .5, 1.1])
    scorer = FrozenCorrectedScorer(fitted_extractor, information)
    frozen = scorer.informativeness.copy()
    information[:] = .25
    np.testing.assert_array_equal(scorer.informativeness, frozen)
    with pytest.raises(ValueError):
        scorer.informativeness[0] = 0.

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Query-time fitting is forbidden")

    monkeypatch.setattr(FeatureExtractor, "fit", forbidden_fit)
    monkeypatch.setattr(matching, "train_informativeness", forbidden_fit)
    vocabulary = fitted_extractor.vectorizers["title"].vocabulary_.copy()
    pairs = [{"id_left": "q", "id_right": "r", "title_left": "Camera AX12345",
              "title_right": "AX12345 Camera", "brand_left": "Example", "brand_right": "Other"}]
    result = scorer.score_pairs(pairs)
    np.testing.assert_array_equal(result, scorer.score_pairs(pairs))
    assert fitted_extractor.vectorizers["title"].vocabulary_ == vocabulary
    np.testing.assert_array_equal(scorer.informativeness, frozen)
    np.testing.assert_array_equal(scorer.score_pairs([]), np.empty(0))
    # Pair labels, if present in a caller's dictionary, are never used by scoring.
    np.testing.assert_array_equal(result, scorer.score_pairs([{**pairs[0], "label": "ignored"}]))
