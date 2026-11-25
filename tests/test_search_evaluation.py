import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from matching import FeatureExtractor
from product_catalog import ProductOffer
from search_evaluation import evaluate_search
from search_system import Candidate, SearchEngine, TfidfCandidateRetriever


def make_catalog(*ids):
    return {offer_id: ProductOffer(offer_id, title="Identical product") for offer_id in ids}


def test_shared_pool_and_hand_calculated_macro_metrics(monkeypatch):
    class FixtureScorer:
        def score_pairs(self, pairs):
            assert all(set(pair) == {f"{field}_{side}" for field in ("id", "title", "description", "brand")
                                     for side in ("left", "right")} for pair in pairs)
            return np.array([0.9 if pair["id_right"] == "a" else 0.2 for pair in pairs])

    orders = {
        "first": ("a", "b", "c"),
        "third": ("b", "c", "a"),
        "sixth": ("b", "c", "d", "e", "f", "a"),
        "miss": ("b", "c", "d"),
        "empty": (),
        "unjudged": ("a",),
    }
    catalog = make_catalog(*orders, "a", "b", "c", "d", "e", "f", "s")
    qrels = {
        "first": frozenset({"a", "s"}), "third": {"a"}, "sixth": {"a"},
        "miss": {"a", "s"}, "empty": {"a"},
    }
    retriever = TfidfCandidateRetriever(catalog)
    engines = {
        "fixed": SearchEngine(catalog, method="fixed"),
        "renorm": SearchEngine(catalog),
        "fixture": SearchEngine(catalog, method="adaptive-corrected", pair_scorer=FixtureScorer()),
    }
    pools = {
        query: [Candidate(catalog[offer_id], 1.0 if query == "first" else 1 / rank)
                for rank, offer_id in enumerate(order, 1)]
        for query, order in orders.items()
    }
    before = {query: tuple((id(row), row.retrieval_score) for row in pool) for query, pool in pools.items()}
    calls = []
    shared = {}

    def retrieve(query_id, candidate_k, *, other_sources_only):
        assert candidate_k == 20
        assert other_sources_only is False
        calls.append(query_id)
        return pools[query_id]

    def track_ranker(method, rank_candidates):
        def rank(query, pool, *, top_k):
            assert isinstance(pool, tuple)
            assert shared.setdefault(query.offer_id, pool) is pool
            assert tuple((id(row), row.retrieval_score) for row in pool) == before[query.offer_id]
            results = rank_candidates(query, pool, top_k=top_k)
            retrieval_scores = {row.offer.offer_id: row.retrieval_score for row in pool}
            assert all(row.retrieval_score == retrieval_scores[row.offer.offer_id] for row in results)
            if query.offer_id == "first" and method in {"fixed", "renorm"}:
                assert len({row.similarity_score for row in results}) == 1
            return results
        return rank

    def forbidden(*args, **kwargs):
        pytest.fail("Evaluator tried to retrieve per ranker or fit")

    monkeypatch.setattr(retriever, "retrieve", retrieve)
    monkeypatch.setattr(FeatureExtractor, "fit", forbidden)
    monkeypatch.setattr(TfidfVectorizer, "fit", forbidden)
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", forbidden)
    for method, engine in engines.items():
        monkeypatch.setattr(engine, "retrieve", forbidden)
        monkeypatch.setattr(engine, "rank_candidates", track_ranker(method, engine.rank_candidates))

    report = evaluate_search(catalog, tuple(orders), qrels, retriever, engines, include_retrieval_scores=True)

    assert calls == list(orders)
    assert shared.keys() == qrels.keys()
    assert {query: tuple((id(row), row.retrieval_score) for row in pool) for query, pool in pools.items()} == before
    assert report["eligible_query_count"] == 5
    assert report["excluded_no_qrels_count"] == 1
    assert report["retrieval"] == {"candidate_recall_at_k": 0.5, "empty_pool_count": 1}
    assert report["ranking"]["tfidf"] == {"hit_at_k": {1: 0.2, 5: 0.4}, "mrr_at_k": pytest.approx(4 / 15)}
    assert report["ranking"]["fixed"] == report["ranking"]["renorm"] == report["ranking"]["tfidf"]
    assert report["ranking"]["fixture"] == {"hit_at_k": {1: 0.6, 5: 0.6}, "mrr_at_k": 0.6}
    for query_id, record in report["per_query"].items():
        assert record["candidate_ids"] == orders[query_id]
        assert record["retrieval_scores"] == tuple(c.retrieval_score for c in pools[query_id])
        assert set(record["ranking"]) == (set(report["ranking"]) if query_id in qrels else set())
    assert [report["per_query"][q]["ranking"]["tfidf"]["first_relevant_rank"] for q in qrels] == [
        1, 3, 6, None, None,
    ]
    assert report["per_query"]["sixth"]["ranking"]["tfidf"]["mrr_at_k"] == 0.0
    assert report["per_query"]["first"]["ranking"]["fixed"]["offer_ids"] == ("a", "b", "c")
    assert report["per_query"]["unjudged"]["candidate_recall_at_k"] is None


def test_all_queries_without_known_matches_have_undefined_aggregates():
    catalog = make_catalog("a", "b")
    report = evaluate_search(catalog, ["a", "b"], {"a": frozenset()},
                             TfidfCandidateRetriever(catalog), {})
    assert report["eligible_query_count"] == 0
    assert report["excluded_no_qrels_count"] == 2
    assert report["retrieval"]["candidate_recall_at_k"] is None
    assert report["ranking"]["tfidf"] == {"hit_at_k": {1: None, 5: None}, "mrr_at_k": None}


@pytest.mark.parametrize(("ids", "message"), [
    (("a", "a"), "Duplicate offer IDs"),
    (("unknown",), "Invalid candidate pool"),
    (("q",), "Invalid candidate pool"),
])
def test_invalid_candidate_pool_is_rejected(monkeypatch, ids, message):
    catalog = make_catalog("q", "a")
    retriever = TfidfCandidateRetriever(catalog)
    monkeypatch.setattr(retriever, "retrieve", lambda *args, **kwargs: [
        Candidate(ProductOffer(offer_id, title="Identical product"), 0.5) for offer_id in ids
    ])
    with pytest.raises(ValueError, match=message):
        evaluate_search(catalog, ["q"], {"q": {"a"}}, retriever, {})


def test_ranker_cannot_drop_candidates(monkeypatch):
    catalog = make_catalog("a", "b")
    engine = SearchEngine(catalog)
    monkeypatch.setattr(engine, "rank_candidates", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="changed the candidate pool"):
        evaluate_search(catalog, ["a"], {"a": {"b"}}, engine, {"renorm": engine})


@pytest.mark.parametrize(("query_ids", "qrels", "message"), [
    (["a", "a"], {}, "Duplicate offer IDs"),
    (["unknown"], {}, "Unknown query"),
    (["a"], {"a": {"unknown"}}, "Qrels must reference catalog IDs"),
    (["a"], {"a": {"a"}}, "Self-match in qrels"),
])
def test_bad_evaluation_inputs_are_rejected(query_ids, qrels, message):
    catalog = make_catalog("a", "b")
    with pytest.raises(ValueError, match=message):
        evaluate_search(catalog, query_ids, qrels, TfidfCandidateRetriever(catalog), {})
