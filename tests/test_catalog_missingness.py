from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from catalog_missingness import mask_catalog, summarize_masked_validation
from final_evaluation import no_fit
from frozen_v1_runtime import load_corrected_scorer
from product_catalog import ProductOffer, offer_to_pair
from search_evaluation import evaluate_search
from search_system import SearchEngine, TfidfCandidateRetriever


FIELDS = {"title", "brand", "description"}


@pytest.fixture
def catalog():
    return {
        key: ProductOffer(
            key, title=f"Camera AB12345 {key}", brand="Shop Brand" if key != "f" else "\t",
            description="Camera body" if key in "abcd" else ("\u00a0" if key == "e" else ""),
            source="Store", url=f"https://example.invalid/{key}",
            attributes={"original_title": f"Camera AB12345 {key}"},
            technical_attributes={"seller_sku": "AB12345"},
        )
        for key in "abcdef"
    }


@pytest.mark.parametrize(("seed", "title", "brand", "description"), [
    (11, "abe", "ad", "ac"), (22, "acf", "bd", "ad"), (33, "cde", "ac", "ac"),
])
def test_exact_masks_for_each_seed_and_independent_field(catalog, seed, title, brand, description):
    original = deepcopy(catalog)
    masked = mask_catalog(catalog, FIELDS, seed)
    assert masked == mask_catalog(dict(reversed(catalog.items())), FIELDS, seed)
    assert masked.keys() == catalog.keys()
    for field, expected in (("title", title), ("brand", brand), ("description", description)):
        single = mask_catalog(catalog, {field}, seed)
        for key, clean in catalog.items():
            value = "" if key in expected else getattr(clean, field)
            assert getattr(masked[key], field) == getattr(single[key], field) == value
            for untouched in FIELDS - {field}:
                assert getattr(single[key], untouched) == getattr(clean, untouched)
    for key, offer in masked.items():
        assert offer is not catalog[key]
        assert (offer.offer_id, offer.source, offer.url) == (key, catalog[key].source, catalog[key].url)
        assert offer.attributes == catalog[key].attributes
        assert offer.technical_attributes == catalog[key].technical_attributes
    masked["a"].attributes.clear()
    masked["a"].technical_attributes.clear()
    assert catalog == original


def test_rejects_annotation_fields_and_inconsistent_ids(catalog):
    for field in ("cluster_id", "qrels", "identifier"):
        with pytest.raises(ValueError):
            mask_catalog(catalog, {field}, 11)
    with pytest.raises(ValueError):
        mask_catalog({"other": catalog["a"]}, {"brand"}, 11)


def pools(catalog):
    retriever = TfidfCandidateRetriever(catalog)
    return {key: [(c.offer.offer_id, c.retrieval_score) for c in retriever.retrieve(key)]
            for key in catalog}


@pytest.mark.parametrize("fields", [{"brand"}, {"description"}, {"brand", "description"}])
def test_auxiliary_fields_preserve_order_and_scores(catalog, fields):
    assert pools(mask_catalog(catalog, fields, 11)) == pools(catalog)


def test_masked_title_and_identifier_in_both_pair_roles(catalog):
    masked = mask_catalog(catalog, FIELDS, 11)
    fixed = SearchEngine(masked, method="fixed")
    with no_fit():
        frozen = load_corrected_scorer()
        for left, right, side in (("a", "d", "left"), ("d", "a", "right")):
            pair = offer_to_pair(masked[left], masked[right])
            assert all(pair[f"{field}_{side}"] == "" for field in FIELDS)
            clean_pair = offer_to_pair(catalog[left], catalog[right])
            assert frozen.extractor.transform([clean_pair]).matrix[0, 3] == 1
            for extractor in (fixed.extractor, frozen.extractor):
                assert np.isnan(extractor.transform([pair]).matrix).all()
            assert np.isfinite(frozen.score_pairs([pair])).all()


def test_shared_masked_pool_denominators_and_diagnostics(catalog):
    masked = mask_catalog(catalog, FIELDS, 11)
    expected = pools(mask_catalog(catalog, {"title"}, 11))
    assert pools(masked) == expected
    qrels = {key: {other} for key, other in zip("abcdef", "badcfe", strict=True)}
    with no_fit():
        scorer = load_corrected_scorer()
    engines = {method: SearchEngine(masked, method) for method in ("fixed", "renorm")}
    engines["adaptive-corrected"] = SearchEngine(masked, "adaptive-corrected", pair_scorer=scorer)
    shared = {}
    rank = SearchEngine.rank_candidates

    def check_pool(engine, query, pool, top_k):
        assert query is masked[query.offer_id]
        assert shared.setdefault(query.offer_id, pool) is pool
        assert all(c.offer is masked[c.offer.offer_id] for c in pool)
        return rank(engine, query, pool, top_k)

    with no_fit(), patch.object(SearchEngine, "rank_candidates", check_pool):
        result = evaluate_search(masked, list(masked), qrels, engines["fixed"].retriever, engines)
    assert set(shared) == set(catalog)
    assert result["eligible_query_count"] == 6
    assert result["excluded_no_qrels_count"] == 0
    for key, row in result["per_query"].items():
        assert row["ranking"]["tfidf"]["offer_ids"] == tuple(i for i, _ in expected[key])
        if key in "abe":
            assert row["candidate_ids"] == ()
            assert all(m["mrr_at_k"] == 0 for m in row["ranking"].values())

    stats = summarize_masked_validation(catalog, masked, qrels, result)
    assert stats["naturally_missing"] == {"title": 0, "brand": 1, "description": 2}
    assert stats["artificially_removed"] == {"title": 3, "brand": 2, "description": 2}
    assert stats["empty_title_queries"] == stats["empty_pool_count"] == 3
    assert stats["clusters_with_both_titles"] == 1
    assert stats["retrieval_miss_fraction"] == 4 / 6
    for method, conditional in stats["conditional_mrr_at_5"].items():
        assert conditional == sum(result["per_query"][key]["ranking"][method]["mrr_at_k"]
                                  for key in ("c", "d")) / 2


def test_undefined_conditional_mrr_when_no_positive_is_retrieved(catalog):
    clean = {key: catalog[key] for key in "ab"}
    masked = mask_catalog(clean, {"title"}, 11)
    qrels = {"a": {"b"}, "b": {"a"}}
    report = evaluate_search(masked, list(masked), qrels, TfidfCandidateRetriever(masked), {})
    diagnostics = summarize_masked_validation(clean, masked, qrels, report)
    assert diagnostics["empty_title_queries"] == 1
    assert diagnostics["empty_pool_count"] == 2
    assert diagnostics["retrieval_miss_fraction"] == 1
    assert diagnostics["conditional_mrr_at_5"] == {"tfidf": None}
