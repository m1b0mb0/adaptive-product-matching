"""Search behavior checks with small fixtures, never scientific observations."""
from __future__ import annotations

import numpy as np
import pytest

from logistic_baseline import SimilarityLogisticRegression
from matching import FeatureExtractor, evidence_scores
from product_catalog import ProductOffer, offer_to_pair
from search_system import (
    FrozenLogisticPairScorer, SearchEngine, TfidfCandidateRetriever,
)


def offer(offer_id, title="Motor alpha 2207 1900kv", **kwargs):
    return ProductOffer(offer_id=offer_id, title=title, **kwargs)


def catalog(*offers):
    return {item.offer_id: item for item in offers}


@pytest.mark.parametrize("component", [TfidfCandidateRetriever, SearchEngine])
def test_empty_catalog_and_invalid_mapping_are_clear_errors(component):
    with pytest.raises(ValueError, match="empty catalog"):
        component({})
    with pytest.raises(ValueError, match="Catalog keys"):
        component({"different": offer("one")})


def test_retrieval_excludes_self_and_breaks_identical_title_ties_by_id():
    data = catalog(offer("query"), offer("z"), offer("a"), offer("m"))
    first = TfidfCandidateRetriever(data).retrieve("query", candidate_k=2)
    second = TfidfCandidateRetriever(dict(reversed(list(data.items())))).retrieve("query", 2)
    assert [row.offer.offer_id for row in first] == ["a", "m"]
    assert first == second
    assert all(0 < row.retrieval_score <= 1 for row in first)


def test_source_filter_precedes_truncation_and_does_not_guess_unknown_sources():
    data = catalog(
        offer("query", source="Shop A"), offer("a", source=" shop a "),
        offer("b", source=""), offer("z", source="Shop B"),
    )
    retriever = TfidfCandidateRetriever(data)
    assert [row.offer.offer_id for row in retriever.retrieve(
        "query", candidate_k=1, other_sources_only=True,
    )] == ["z"]
    with pytest.raises(ValueError, match="source is unknown"):
        retriever.retrieve("b", other_sources_only=True)
    assert [row.offer.offer_id for row in retriever.retrieve("query", 1)] == ["a"]


@pytest.mark.parametrize("query_title", ["", " ", "x"])
def test_empty_or_unrepresentable_query_does_not_fallback_to_brand(query_title):
    engine = SearchEngine(catalog(
        offer("query", query_title, brand="same"), offer("known", brand="same"),
    ))
    assert engine.retrieve("query") == []
    assert engine.search("query") == []


def test_catalog_without_any_title_vocabulary_and_zero_overlap_are_safe():
    assert SearchEngine(catalog(offer("a", "x"), offer("b", ""))).search("a") == []
    data = catalog(offer("a", "aaaa"), offer("b", "zzzz"), offer("empty", ""))
    assert TfidfCandidateRetriever(data).retrieve("a") == []


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "2", None])
def test_top_k_arguments_require_positive_integers(invalid):
    engine = SearchEngine(catalog(offer("a"), offer("b")))
    with pytest.raises(ValueError, match="candidate_k"):
        engine.retrieve("a", invalid)
    with pytest.raises(ValueError, match="top_k"):
        engine.search("a", top_k=invalid)


def test_invalid_query_and_source_filter_types_are_reported():
    engine = SearchEngine(catalog(offer("a")))
    for query_id in ("missing", "", None, []):
        with pytest.raises(ValueError, match="Unknown offer_id"):
            engine.search(query_id)
    with pytest.raises(ValueError, match="other_sources_only"):
        engine.search("a", other_sources_only="yes")


def test_reranking_is_distinct_from_title_candidate_retrieval():
    engine = SearchEngine(catalog(
        offer("query", brand="Target"),
        offer("exact_title", brand="Other"),
        offer("compatible_brand", "Motor alpha 2207 1900kv red", brand="Target"),
    ))
    assert engine.retrieve("query")[0].offer.offer_id == "exact_title"
    results = engine.search("query", candidate_k=2, top_k=2)
    assert results[0].offer.offer_id == "compatible_brand"
    assert results[0].retrieval_score < results[1].retrieval_score
    assert results[0].similarity_score > results[1].similarity_score
    # Candidate truncation is visible: reranking cannot recover an omitted row.
    assert engine.search("query", candidate_k=1, top_k=5)[0].offer.offer_id == "exact_title"


@pytest.mark.parametrize("method", ["fixed", "renorm"])
def test_builtin_scores_use_existing_grouped_formula_and_keep_missing_distinct(method):
    engine = SearchEngine(catalog(offer("a"), offer("b", brand="Known")), method=method)
    results = engine.search("a")
    assert len(results) == 1
    result = results[0]
    assert result.feature_similarities["brand"] is None
    assert result.feature_similarities["description"] is None
    pair = offer_to_pair(engine.catalog["a"], engine.catalog["b"])
    features = engine.extractor.transform([pair]).matrix
    assert result.similarity_score == pytest.approx(evidence_scores(features, method)[0])
    assert result.offer.brand == "Known"
    assert result.offer.title == engine.catalog["b"].title


def test_observed_brand_conflict_is_zero_not_missing():
    engine = SearchEngine(catalog(offer("a", brand="One"), offer("b", brand="Two")))
    assert engine.search("a")[0].feature_similarities["brand"] == 0.0


def test_technical_metadata_does_not_become_search_evidence():
    data = catalog(
        offer("query", technical_attributes={"kv_rpm_per_volt": "1900"}),
        offer("a", technical_attributes={"kv_rpm_per_volt": "500"}),
        offer("b", technical_attributes={"kv_rpm_per_volt": "1900"}),
    )
    results = SearchEngine(data).search("query")
    assert [item.offer.offer_id for item in results] == ["a", "b"]
    assert results[0].similarity_score == results[1].similarity_score
    assert results[0].offer.technical_attributes["kv_rpm_per_volt"] == "500"


def test_search_does_not_refit_any_representation(monkeypatch):
    engine = SearchEngine(catalog(offer("query"), offer("a"), offer("b")))

    def forbidden(*args, **kwargs):
        raise AssertionError("Search attempted to fit a representation")

    monkeypatch.setattr(FeatureExtractor, "fit", forbidden)
    monkeypatch.setattr(type(engine.retriever.vectorizer), "fit", forbidden)
    monkeypatch.setattr(type(engine.retriever.vectorizer), "fit_transform", forbidden)
    assert engine.search("query") == engine.search("query")


def test_unsupported_method_or_invalid_extension_is_rejected():
    data = catalog(offer("a"))
    with pytest.raises(ValueError, match="Search method"):
        SearchEngine(data, method="adaptive")
    with pytest.raises(ValueError, match="already-fitted"):
        SearchEngine(data, method="logistic")
    with pytest.raises(ValueError, match="only supported"):
        SearchEngine(data, pair_scorer=object())
    with pytest.raises(ValueError, match="implement score_pairs"):
        SearchEngine(data, method="logistic", pair_scorer=object())


@pytest.mark.parametrize("invalid_scores", [[np.nan], [np.inf], [-0.1], [1.01], [], [[0.5]]])
def test_external_scorer_cannot_return_malformed_or_nonfinite_values(invalid_scores):
    class InvalidScorer:
        def score_pairs(self, pairs):
            assert all("label" not in pair for pair in pairs)
            return invalid_scores

    engine = SearchEngine(catalog(offer("a"), offer("b")), method="logistic",
                          pair_scorer=InvalidScorer())
    with pytest.raises(ValueError, match="one finite score"):
        engine.search("a")


def test_frozen_logistic_extension_never_refits_or_uses_catalog_fit_features(monkeypatch):
    training_offers = [offer("train_a", "motor alpha 2207"),
                       offer("train_b", "motor beta 2207"),
                       offer("train_c", "different product")]
    pairs = [offer_to_pair(training_offers[0], training_offers[1]),
             offer_to_pair(training_offers[0], training_offers[2])]
    extractor = FeatureExtractor().fit(pairs)
    model = SimilarityLogisticRegression().fit(extractor.transform(pairs).matrix,
                                              np.array([1, 0]))
    scorer = FrozenLogisticPairScorer(extractor, model)
    before = model.metadata()
    data = catalog(offer("query"), offer("other"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Search attempted to retrain")

    monkeypatch.setattr(FeatureExtractor, "fit", forbidden)
    monkeypatch.setattr(SimilarityLogisticRegression, "fit", forbidden)
    engine = SearchEngine(data, method="logistic", pair_scorer=scorer)
    result = engine.search("query")[0]
    expected = scorer.score_pairs([offer_to_pair(data["query"], data["other"])])[0]
    assert result.similarity_score == pytest.approx(expected)
    assert result.feature_similarities is None
    assert model.metadata() == before
    assert engine.extractor is None


def test_frozen_logistic_extension_requires_fitted_components():
    with pytest.raises(ValueError, match="Fit the feature extractor"):
        FrozenLogisticPairScorer(FeatureExtractor(), SimilarityLogisticRegression())
