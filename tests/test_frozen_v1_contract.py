"""Numerical reference values captured in the historical 8dcf5db worktree."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from frozen_method_artifact import load_corrected_scorer
from matching import FIELDS, FeatureExtractor
from product_catalog import ProductOffer, offer_to_pair
from search_system import SearchEngine


CONFIG = Path(__file__).resolve().parents[1] / "docs/final_method/frozen_configuration.json"
CONFIG_SHA256 = "2976e18752a2463a7ef784714d5468a150db0a9e0142812d471e3fc13b0c655a"

CAMERA_PAIR = {
    "title_left": "Canon EOS AX12345 camera",
    "title_right": "Canon EOS AX12345 digital camera",
    "description_left": "18 MP digital camera with lens",
    "description_right": "Digital camera with 18 MP sensor",
    "brand_right": "Canon",
}
CASES = {
    "missing_fields": {},
    "empty_text": {
        "title_left": "", "title_right": "Canon EOS AX12345 camera",
        "description_left": "digital camera", "description_right": " ",
        "brand_left": "", "brand_right": "Canon",
    },
    "oov_text": {
        "title_left": "ЖЖЖЖ", "title_right": "ЖЖЖЖ",
        "description_left": "ЖЖЖЖ", "description_right": "ЖЖЖЖ",
    },
    "brand_conflict_only": {"brand_left": "Canon", "brand_right": "Nikon"},
    "missing_brand": CAMERA_PAIR,
    "full_match": CAMERA_PAIR | {"brand_left": "Canon", "brand_right": "CANON"},
    "full_conflict": CAMERA_PAIR | {
        "title_right": "Nikon ZZ98765 digital camera",
        "description_right": "Digital camera with 14 MP sensor",
        "brand_left": "Canon", "brand_right": "Nikon",
    },
}

# None records an unavailable feature. Observed zero remains an available value.
REFERENCE = {
    "missing_fields": ([None, None, None, None], 0.0),
    "empty_text": ([None, None, None, None], 0.0),
    "oov_text": ([None, None, None, None], 0.0),
    "brand_conflict_only": ([None, None, 0.0, None], 0.0),
    "missing_brand": ([0.7978448336710007, 0.7142841920763874, None, 1.0], 0.7283780947360525),
    "full_match": ([0.7978448336710007, 0.7142841920763874, 1.0, 1.0], 0.8871423567232782),
    "full_conflict": ([0.20955353033460564, 0.6036352383737326, 0.0, 0.0], 0.15210084574752702),
}


def forbid_fit(*args, **kwargs):
    raise AssertionError("Frozen pairwise scoring called fit")


def guard_text_fit(monkeypatch):
    monkeypatch.setattr(TfidfVectorizer, "fit", forbid_fit)
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", forbid_fit)


@pytest.fixture
def frozen_scorer(monkeypatch):
    raw_config = CONFIG.read_bytes()
    assert hashlib.sha256(raw_config).hexdigest() == CONFIG_SHA256
    with monkeypatch.context() as guard:
        guard.setattr(FeatureExtractor, "fit", forbid_fit)
        guard_text_fit(guard)
        return load_corrected_scorer(CONFIG)


@pytest.mark.parametrize("case", CASES)
def test_frozen_scorer_matches_historical_features_and_scores(frozen_scorer, monkeypatch, case):
    monkeypatch.setattr(FeatureExtractor, "fit", forbid_fit)
    guard_text_fit(monkeypatch)
    pairs = [CASES[case]]
    expected_features, expected_score = REFERENCE[case]
    assert FIELDS == ("title", "description", "brand", "identifier")
    matrix = frozen_scorer.extractor.transform(pairs).matrix
    assert matrix.shape == (1, 4)
    np.testing.assert_array_equal(~np.isnan(matrix[0]), [value is not None for value in expected_features])
    np.testing.assert_allclose(matrix[0], np.array(expected_features, dtype=float), rtol=0, atol=1e-14)
    np.testing.assert_allclose(frozen_scorer.score_pairs(pairs), [expected_score], rtol=0, atol=1e-14)


def test_frozen_vectorizers_keep_train_vocabulary_and_idf(frozen_scorer):
    config = json.loads(CONFIG.read_bytes())
    expected = config["full_train_fit"]["representation"]["vectorizers"]
    assert frozen_scorer.extractor.vectorizers.keys() == expected.keys()
    for field, vectorizer in frozen_scorer.extractor.vectorizers.items():
        indices = {term: int(index) for term, index in vectorizer.vocabulary_.items()}
        vocabulary = json.dumps(indices, sort_keys=True).encode()
        assert hashlib.sha256(vocabulary).hexdigest() == expected[field]["vocabulary_sha256"]
        assert hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest() == expected[field]["idf_sha256"]


def test_search_projects_only_pair_fields_and_keeps_frozen_scores(frozen_scorer, monkeypatch):
    pair = CASES["full_match"]
    query = ProductOffer(
        "query", pair["title_left"], pair["description_left"], pair["brand_left"],
        source="Shop A", url="https://shop-a.invalid/offer",
        attributes={"review_notes": "Demo only", "price": "1"},
        technical_attributes={"seller_sku": "ZZ98765"},
    )
    match = ProductOffer("match", pair["title_right"], pair["description_right"], pair["brand_right"])
    conflict = CASES["full_conflict"]
    other = ProductOffer("other", conflict["title_right"], conflict["description_right"], conflict["brand_right"])
    assert offer_to_pair(query, match) == {"id_left": "query", "id_right": "match", **pair}

    # The title retriever can build its catalog index. Pairwise features must remain frozen.
    monkeypatch.setattr(FeatureExtractor, "fit", forbid_fit)
    catalog = {offer.offer_id: offer for offer in (query, match, other)}
    engine = SearchEngine(catalog, method="adaptive-corrected", pair_scorer=frozen_scorer)
    guard_text_fit(monkeypatch)
    assert engine.extractor is None
    results = engine.search("query", candidate_k=2, top_k=2)
    assert [result.offer.offer_id for result in results] == ["match", "other"]
    np.testing.assert_allclose(
        [result.similarity_score for result in results],
        [REFERENCE["full_match"][1], REFERENCE["full_conflict"][1]], rtol=0, atol=1e-14,
    )
