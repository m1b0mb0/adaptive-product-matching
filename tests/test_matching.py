import numpy as np
import pandas as pd
import pytest

from audit_wdc import audit
from matching import (FeatureExtractor, Features, identifiers, norm, scores, masked,
                      train_informativeness, train_texts)
from run_experiment import metrics, best_threshold


def test_model_identifier_extraction_preserves_model_distinctions():
    assert "ds7208hq h i" not in identifiers("DVR Hikvision DS-7208HQHI-K1")
    assert "ds7208hqhik1" in identifiers("DVR Hikvision DS-7208HQHI-K1")
    assert "ds7216huhik2" in identifiers("DVR Hikvision DS-7216HUHI-K2")
    assert "16gb" not in identifiers("Storage 16GB")


def test_missing_versus_conflict():
    fully_matching = np.array([[.8, np.nan, 1.0, 1.0]])
    same_but_conflicting_identifier = np.array([[.8, np.nan, 1.0, 0.0]])
    missing_identifier = np.array([[.8, np.nan, 1.0, np.nan]])
    a = scores(fully_matching, "adaptive", np.ones(4), penalty=.1)[0]
    b = scores(same_but_conflicting_identifier, "adaptive", np.ones(4), penalty=.1)[0]
    c = scores(missing_identifier, "adaptive", np.ones(4), penalty=.1)[0]
    assert a > b
    assert c != b  # missing attribute must not be mistaken for a zero similarity


def test_coverage_is_not_only_renormalization():
    row = np.array([[0.95, np.nan, np.nan, np.nan]])
    renorm = scores(row, "renorm")[0]
    proposed = scores(row, "adaptive", np.ones(4), gamma=1, prior=.2)[0]
    assert renorm > proposed > 0


def test_training_informativeness_uses_labels():
    x = np.array([[.95, .9, 1., 1.], [.88, .8, 1., 1.], [.90, .7, 1., 1.],
                  [.40, .2, 0., 0.], [.30, .3, 0., 0.], [.25, .4, 0., 0.]])
    info = train_informativeness(x, np.array([1, 1, 1, 0, 0, 0]))
    assert np.isfinite(info).all()
    assert info.shape == (4,)


def test_consistent_offer_level_masking():
    f = Features(np.ones((2, 4)), ["offer-a", "offer-a"], ["offer-b", "offer-c"])
    m = masked(f, 1.0, 3)
    assert np.isnan(m[:, 1:3]).all()
    assert np.isfinite(m[:, [0, 3]]).all()


def test_metrics_precision_recall_f1():
    output = metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert output["tp"] == 1 and output["fp"] == 1
    assert output["fn"] == 1 and output["tn"] == 1
    assert output["f1"] == .5


def test_threshold_selection_does_not_require_test():
    t, stat = best_threshold(np.array([1, 0, 1, 0]), np.array([.9, .15, .8, .1]))
    assert .15 < t <= .8
    assert stat["f1"] == 1


def test_audit_split_overlap_detection():
    p = lambda a, b, y: {"id_left": a, "id_right": b, "title_left": "x", "title_right": "x", "label": y}
    result = audit([p(1, 2, 1)], [p(3, 4, 0)], [p(5, 6, 0)])
    assert result["offer_id_overlaps"] == {"train_valid": 0, "train_test": 0, "valid_test": 0}


@pytest.mark.parametrize("method", ["text", "fixed", "renorm", "adaptive", "adaptive_no_info",
                                    "adaptive_no_coverage", "adaptive_no_penalty"])
def test_no_observed_evidence_is_not_a_match(method):
    assert scores(np.full((1, 4), np.nan), method, np.ones(4))[0] == 0


def test_search_demo_excludes_query_offer():
    from search_cli import search
    products = {
        "1": {"id": "1", "title": "Brand ABC-123 Camera", "description": "Indoor camera", "brand": "Brand"},
        "2": {"id": "2", "title": "ABC-123 Camera by Brand", "description": "Indoor camera", "brand": "Brand"},
        "3": {"id": "3", "title": "Brand XYZ-777 Camera", "description": "Outdoor camera", "brand": "Brand"},
    }
    matches = search(products, "1", 2)
    assert len(matches) == 2
    assert all(offer_id != "1" for offer_id, _, _ in matches)


@pytest.mark.parametrize("value", [None, np.nan, np.float32("nan"), pd.NA, pd.NaT, "", " \t "])
def test_null_values_normalize_to_absent_text(value):
    assert norm(value) == ""
    assert identifiers(value) == set()


def test_normalization_handles_unicode_case_and_spacing():
    assert norm("  Straße\u3000ＴＶ\t①  ") == "strasse tv 1"
    assert norm(0) == "0"  # numeric zero is a value, not a missing sentinel
    assert norm("nan") == "nan"  # only actual nulls are treated as absent


def test_feature_extraction_handles_empty_and_unrepresentable_text():
    train = [{"id_left": "a", "id_right": "b", "title_left": "abcd", "title_right": "abcd"}]
    extractor = FeatureExtractor().fit(train)
    assert extractor.vectorizers["description"] is None
    assert extractor.transform([]).matrix.shape == (0, 4)
    # Both absent text and text outside the frozen training vocabulary carry
    # no measurable cosine evidence; they must not count as disagreements.
    unseen = [{"title_left": "ΩΩΩ", "title_right": "ΩΩΩ", "description_left": np.nan,
               "description_right": pd.NA, "brand_left": np.nan, "brand_right": np.nan}]
    features = extractor.transform(unseen).matrix
    assert np.isnan(features).all()
    assert scores(features, "adaptive")[0] == 0
    assert "ωω" not in extractor.vectorizers["title"].vocabulary_


def test_one_character_training_fields_do_not_create_fake_vocabulary():
    extractor = FeatureExtractor().fit([{"title_left": "a", "title_right": "b"}])
    assert extractor.vectorizers["title"] is None
    assert np.isnan(extractor.transform([{"title_left": "ab", "title_right": "ab"}]).matrix).all()


def test_missing_offer_ids_do_not_merge_distinct_training_texts():
    texts = train_texts([{"id_left": None, "id_right": np.nan,
                         "title_left": "alpha", "title_right": "beta"}], "title")
    assert texts == ["alpha", "beta"]


def test_feature_extractor_requires_fitted_nonempty_training_data():
    with pytest.raises(ValueError, match="empty training"):
        FeatureExtractor().fit([])
    with pytest.raises(ValueError, match="Fit the feature extractor"):
        FeatureExtractor().transform([])


def test_threshold_selection_uses_unrounded_f1(monkeypatch):
    import run_experiment
    monkeypatch.setattr(run_experiment, "THRESHOLDS", np.array([.49, .5]))
    y = np.ones(100001, dtype=int)
    values = np.full(len(y), .5)
    values[-1] = .49
    threshold, stat = best_threshold(y, values)
    assert threshold == .49
    assert stat["f1"] == 1
    assert metrics(y, (values >= .5).astype(int))["f1"] < 1


def test_threshold_ties_prefer_precision_then_higher_threshold(monkeypatch):
    import run_experiment
    monkeypatch.setattr(run_experiment, "THRESHOLDS", np.array([.4, .7, .8, .9]))
    threshold, stat = best_threshold(np.array([1, 0, 0, 1]), np.array([.9, .6, .5, .4]))
    assert threshold == .9
    assert stat["precision"] == 1
    assert stat["f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("y,values", [(np.array([]), np.array([])),
                                     (np.array([0, 1]), np.array([.5])),
                                     (np.array([0, 1]), np.array([.5, np.nan]))])
def test_threshold_selection_rejects_invalid_validation_scores(y, values):
    with pytest.raises(ValueError):
        best_threshold(y, values)


def test_metrics_support_single_class_split():
    stat = metrics(np.array([0, 0]), np.array([0, 1]))
    assert stat == {"precision": 0., "recall": 0., "f1": 0., "tp": 0, "fp": 1, "fn": 0, "tn": 1}


def test_search_handles_one_offer_and_null_repeated_fields():
    from search_cli import collect_offers, search
    products = {"1": {"id": "1", "title": "ABCD"}}
    assert search(products, "1", 5) == []
    pairs = [{"id_left": "1", "id_right": "2", "title_left": pd.NA, "title_right": "other"},
             {"id_left": "1", "id_right": "3", "title_left": "ABCD", "title_right": "another"}]
    assert collect_offers(pairs)["1"]["title"] == "ABCD"
    with pytest.raises(ValueError, match="positive integer"):
        search(products, "1", -1)


def test_partial_masking_is_stable_for_shared_and_reversed_offers():
    f = Features(np.ones((4, 4)), ["a", "b", "a", "b"], ["a", "b", "b", "a"])
    for seed in (11, 22, 33):
        result = masked(f, .35, seed)
        np.testing.assert_equal(result, masked(f, .35, seed))
        for j in (1, 2):
            assert np.isnan(result[2, j]) == (np.isnan(result[0, j]) or np.isnan(result[1, j]))
            assert np.isnan(result[2, j]) == np.isnan(result[3, j])
    np.testing.assert_array_equal(f.matrix, np.ones((4, 4)))
