"""Analytical checks for the predeclared evidence-group score.

These examples verify the mathematics; they are not benchmark observations.
"""
import numpy as np
import pytest

from matching import FeatureExtractor, evidence_coverage, evidence_scores, scores


METHODS = ("text", "fixed", "renorm", "adaptive", "adaptive_no_info",
           "adaptive_no_reliability", "adaptive_no_coverage", "adaptive_no_penalty")


def test_tfidf_cosine_roundoff_never_exceeds_mathematical_bounds():
    # Sparse summation of this identical long vector produced 1+4e-16 in the
    # audited environment. The feature boundary must remain valid either way.
    title = "a long technical product description 123 ABC with many ngrams"
    pairs = [{"title_left": title, "title_right": title}]
    values = FeatureExtractor().fit(pairs).transform(pairs).matrix
    assert 0 <= values[0, 0] <= 1
    assert values[0, 0] == pytest.approx(1)
    assert np.isfinite(evidence_scores(values, "adaptive")).all()


def test_title_and_derived_identifier_share_one_availability_budget():
    values = np.array([[.8, np.nan, np.nan, np.nan],
                       [.8, np.nan, np.nan, .8],
                       [np.nan, np.nan, np.nan, .8]])
    np.testing.assert_allclose(evidence_coverage(values), [.7, .7, .7])
    for method in ("fixed", "renorm", "adaptive"):
        actual = evidence_scores(values, method, reliability=np.ones(4))
        np.testing.assert_allclose(actual, actual[0])
    assert evidence_scores(values[:1], "fixed")[0] == pytest.approx(.7 * .8)
    # Legacy .4 budget for an available title is intentionally not the new API.
    assert scores(values[:1], "fixed")[0] == pytest.approx(.4 * .8)


def test_fixed_and_renorm_use_same_hierarchical_channels():
    values = np.array([[.8, .4, 1., .2], [.8, np.nan, 1., np.nan]])
    # First row: title family's .4/.3 split is the historical allocation.
    # Second row: title family retains its .7 budget when identifier is absent.
    expected_fixed = np.array([.4 * .8 + .3 * .2 + .15 * .4 + .15,
                               .7 * .8 + .15])
    np.testing.assert_allclose(evidence_scores(values, "fixed"), expected_fixed)
    np.testing.assert_allclose(evidence_scores(values, "renorm"),
                               expected_fixed / np.array([1., .85]))


def test_adaptive_matches_hand_computed_group_formula():
    values = np.array([[.8, .4, 1., .2], [.8, np.nan, np.nan, .2]])
    info = np.array([1.5, .5, 1.2, .8])
    family_numerator = .4 * 1.5 * .8 + .3 * .7 * .8 * .2
    family_weight = .4 * 1.5 + .3 * .7 * .8
    expected_complete = (family_numerator + .15 * .5 * .4 + .15 * 1.2) / (
        family_weight + .15 * .5 + .15 * 1.2)
    expected_family_only = .7 * family_numerator / family_weight + .3 * .2
    np.testing.assert_allclose(evidence_scores(values, "adaptive", info),
                               [expected_complete, expected_family_only])


def test_missing_identifier_quality_does_not_reduce_title_group_quality():
    values = np.array([[.8, .4, np.nan, np.nan]])
    info = np.array([1.5, .5, 1., 100.])
    expected_mean = (.7 * 1.5 * .8 + .15 * .5 * .4) / (.7 * 1.5 + .15 * .5)
    # A huge identifier quality factor has no effect when that channel is absent.
    actual = evidence_scores(values, "adaptive", info, use_coverage=False)
    assert actual[0] == pytest.approx(expected_mean)
    info[3] = .01
    np.testing.assert_allclose(actual, evidence_scores(
        values, "adaptive", info, use_coverage=False))


def test_disabling_quality_and_coverage_recovers_renormalization():
    values = np.array([[.8, .4, np.nan, .2], [.6, np.nan, 1., np.nan]])
    result = evidence_scores(values, "adaptive", np.arange(1, 5),
                             use_info=False, use_reliability=False, use_coverage=False)
    np.testing.assert_allclose(result, evidence_scores(values, "renorm"))


@pytest.mark.parametrize("flag,method", [("use_info", "adaptive_no_info"),
                                        ("use_reliability", "adaptive_no_reliability"),
                                        ("use_coverage", "adaptive_no_coverage")])
def test_named_ablation_matches_flag(flag, method):
    values = np.array([[.8, .4, np.nan, .2], [.6, np.nan, 1., np.nan]])
    info = np.array([1.4, .9, 1.1, 1.3])
    np.testing.assert_allclose(evidence_scores(values, method, info),
                               evidence_scores(values, "adaptive", info, **{flag: False}))


def test_reliability_ablation_is_exactly_unit_manual_factors():
    values = np.array([[.8, .4, np.nan, .2]])
    info = np.array([1.4, .9, 1.1, 1.3])
    np.testing.assert_allclose(evidence_scores(values, "adaptive_no_reliability", info),
                               evidence_scores(values, "adaptive", info,
                                               reliability=np.ones(4)))


def test_information_and_reliability_never_change_coverage():
    values = np.array([[.8, .4, np.nan, .2], [.6, np.nan, 1., np.nan]])
    expected = evidence_coverage(values)
    np.testing.assert_allclose(expected, [.85, .85])
    # Scores may differ through evidence quality, but C is solely raw availability.
    for info in (np.ones(4), np.array([.2, 4., .4, 2.])):
        mean = evidence_scores(values, "adaptive", info, use_coverage=False)
        full = evidence_scores(values, "adaptive", info)
        np.testing.assert_allclose(full, expected * mean + (1 - expected) * .2)


@pytest.mark.parametrize("method", METHODS)
def test_grouped_scores_empty_and_no_evidence(method):
    assert evidence_scores(np.full((1, 4), np.nan), method)[0] == 0
    assert evidence_scores(np.empty((0, 4)), method).shape == (0,)


def test_observed_zero_identifier_is_distinct_from_missing_and_penalty_optional():
    values = np.array([[.8, np.nan, np.nan, 0.],
                       [.8, np.nan, np.nan, np.nan]])
    plain = evidence_scores(values, "adaptive")
    penalized = evidence_scores(values, "adaptive", penalty=.1)
    np.testing.assert_allclose(penalized, plain - [.1, 0.])
    assert plain[0] < plain[1]
    np.testing.assert_allclose(evidence_scores(values, "adaptive_no_penalty", penalty=.1), plain)


def test_coverage_ablation_preserves_order_within_one_availability_pattern():
    values = np.array([[.9, np.nan, np.nan, .7],
                       [.8, np.nan, np.nan, .2],
                       [.1, np.nan, np.nan, .1]])
    full = evidence_scores(values, "adaptive", gamma=1.5)
    no_coverage = evidence_scores(values, "adaptive_no_coverage", gamma=1.5)
    np.testing.assert_array_equal(np.argsort(full), np.argsort(no_coverage))
    np.testing.assert_allclose(full, .7 ** 1.5 * no_coverage + (1 - .7 ** 1.5) * .2)


@pytest.mark.parametrize("values", [np.ones(4), np.zeros((2, 3)),
                                    np.array([[np.inf, 0., 0., 0.]]),
                                    np.array([[1.1, 0., 0., 0.]]),
                                    np.array([[-.1, 0., 0., 0.]])])
def test_grouped_scores_reject_invalid_similarity_arrays(values):
    with pytest.raises(ValueError):
        evidence_scores(values, "adaptive")
    with pytest.raises(ValueError):
        evidence_coverage(values)


@pytest.mark.parametrize("options", [{"gamma": 0}, {"gamma": np.nan},
                                     {"penalty": -.1}, {"penalty": np.inf},
                                     {"prior": -.1}, {"prior": np.nan},
                                     {"informativeness": [1, 2]},
                                     {"informativeness": [1, 1, 1, 0]},
                                     {"reliability": [1, 1, 1, 1.1]}])
def test_grouped_scores_reject_invalid_parameters(options):
    with pytest.raises(ValueError):
        evidence_scores(np.ones((1, 4)), "adaptive", **options)


def test_grouped_scores_reject_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        evidence_scores(np.ones((1, 4)), "not_a_method")


def test_grouped_scores_do_not_mutate_features_or_quality():
    values = np.array([[.8, np.nan, .2, .3]])
    original = values.copy()
    info = np.array([1.2, .8, .4, 1.4])
    old_info = info.copy()
    evidence_scores(values, "adaptive", info)
    np.testing.assert_equal(values, original)
    np.testing.assert_equal(info, old_info)
