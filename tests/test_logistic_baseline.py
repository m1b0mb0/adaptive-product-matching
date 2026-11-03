import json

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from logistic_baseline import (DESIGN_FEATURE_NAMES, SimilarityLogisticRegression,
                               similarity_design_matrix)
from matching import FeatureExtractor


def training_data():
    matrix = np.array([[.95, .8, 1., 1.], [.85, np.nan, 1., .8],
                       [.2, .1, 0., 0.], [.1, np.nan, 0., .1]])
    return matrix, np.array([1, 1, 0, 0])


def test_encoding_distinguishes_missing_from_observed_zero_without_mutation():
    matrix = np.array([[np.nan, 0., 1., .3], [0., np.nan, np.nan, .3]])
    original = matrix.copy()
    encoded = similarity_design_matrix(matrix)
    np.testing.assert_array_equal(encoded, [[0., 0., 1., .3, 0., 1., 1., 1.],
                                           [0., 0., 0., .3, 1., 0., 0., 1.]])
    np.testing.assert_array_equal(matrix, original)
    assert encoded.shape == (2, 8)
    assert DESIGN_FEATURE_NAMES == (
        "similarity_title", "similarity_description", "similarity_brand", "similarity_identifier",
        "available_title", "available_description", "available_brand", "available_identifier",
    )


@pytest.mark.parametrize("matrix", [np.ones(4), np.ones((3, 3)), np.ones((1, 5)),
                                   np.ones((1, 1, 4))])
def test_encoding_rejects_wrong_shape(matrix):
    with pytest.raises(ValueError, match="4 columns"):
        similarity_design_matrix(matrix)


@pytest.mark.parametrize("bad", [-.01, 1.01, np.inf, -np.inf])
def test_encoding_rejects_invalid_observed_similarity(bad):
    with pytest.raises(ValueError, match="finite and in"):
        similarity_design_matrix(np.array([[bad, 0., 0., np.nan]]))


@pytest.mark.parametrize("C", [0, -1, np.inf, np.nan, "invalid", None, True])
def test_invalid_regularization_rejected(C):
    with pytest.raises(ValueError, match="finite and positive"):
        SimilarityLogisticRegression(C=C)


def test_missingness_signal_is_learnable_even_when_all_similarity_values_are_zero():
    matrix = np.array([[np.nan, 0., 0., 0.], [0., 0., 0., 0.]] * 20)
    labels = np.array([1, 0] * 20)
    model = SimilarityLogisticRegression().fit(matrix, labels)
    missing, observed_zero = model.predict_score(matrix[:2])
    assert missing > .5 > observed_zero
    assert model.metadata()["coefficients"][4] < 0


@pytest.mark.parametrize("labels", [np.array([[0], [1], [0], [1]]), np.array([0, 1]),
                                  np.array([0, 1, 2, 0]), np.array([0., 1., .3, 0.]),
                                  np.array([0., 1., np.nan, 0.]),
                                  np.array(["0", "1", "0", "1"])])
def test_invalid_training_labels_rejected(labels):
    matrix, _ = training_data()
    with pytest.raises(ValueError, match="labels"):
        SimilarityLogisticRegression().fit(matrix, labels)


@pytest.mark.parametrize("labels", [np.zeros(4), np.ones(4)])
def test_single_class_training_rejected(labels):
    matrix, _ = training_data()
    with pytest.raises(ValueError, match="both label classes"):
        SimilarityLogisticRegression().fit(matrix, labels)


def test_empty_training_rejected():
    with pytest.raises(ValueError, match="both label classes"):
        SimilarityLogisticRegression().fit(np.empty((0, 4)), np.empty(0))


def test_prediction_and_metadata_require_fit():
    model = SimilarityLogisticRegression()
    matrix, _ = training_data()
    with pytest.raises(NotFittedError):
        model.predict_score(matrix)
    with pytest.raises(NotFittedError):
        model.metadata()


@pytest.mark.parametrize("C", [.1, 1., 10.])
def test_classifier_matches_direct_eight_column_sklearn_fit_and_is_deterministic(C):
    matrix, labels = training_data()
    model = SimilarityLogisticRegression(C=C).fit(matrix, labels)
    direct = LogisticRegression(C=C, class_weight=None, solver="lbfgs", max_iter=2000,
                                random_state=20261101)
    direct.fit(similarity_design_matrix(matrix), labels)
    expected = direct.predict_proba(similarity_design_matrix(matrix))[:, 1]
    np.testing.assert_allclose(model.predict_score(matrix), expected)
    second = SimilarityLogisticRegression(C=C).fit(matrix, labels)
    np.testing.assert_array_equal(model.predict_score(matrix), second.predict_score(matrix))


def test_prediction_does_not_fit_or_change_training_state():
    matrix, labels = training_data()
    model = SimilarityLogisticRegression().fit(matrix, labels)
    before = model.metadata()
    scores = model.predict_score(np.array([[np.nan] * 4, [0.] * 4, [1.] * 4]))
    assert scores.shape == (3,)
    assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
    assert model.metadata() == before
    assert model.predict_score(np.empty((0, 4))).shape == (0,)
    json.dumps(before, allow_nan=False)
    assert before["feature_order"] == list(DESIGN_FEATURE_NAMES)
    assert len(before["coefficients"]) == 8
    assert before["calibration_established"] is False


def test_compatibility_with_existing_training_only_feature_extractor():
    pairs = [
        {"id_left": "a", "id_right": "b", "title_left": "Acme ABC100 camera",
         "title_right": "ACME ABC100", "brand_left": "Acme", "brand_right": "Acme"},
        {"id_left": "a", "id_right": "c", "title_left": "Acme ABC100 camera",
         "title_right": "Other XYZ200 printer", "brand_left": "Acme", "brand_right": "Other"},
    ]
    extractor = FeatureExtractor().fit(pairs)
    features = extractor.transform(pairs).matrix
    assert np.isnan(features[:, 1]).all()
    model = SimilarityLogisticRegression().fit(features, np.array([1, 0]))
    scores = model.predict_score(features)
    assert scores[0] > scores[1]
