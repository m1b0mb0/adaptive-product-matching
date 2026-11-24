"""Evaluate known matches using one immutable candidate pool per query."""
from collections.abc import Mapping, Sequence, Set
from statistics import mean

from product_catalog import ProductOffer
from search_metrics import (
    _positive_k,
    _validate_offer_ids,
    candidate_recall_at_k,
    first_relevant_rank,
    hit_at_k,
    mrr_at_k,
)
from search_system import Candidate, SearchEngine, TfidfCandidateRetriever


TFIDF_BASELINE = "tfidf"


def _ranking_metrics(
    offer_ids: tuple[str, ...], relevant_ids: Set[str], hit_ks: tuple[int, ...], mrr_k: int,
) -> dict:
    return {
        "offer_ids": offer_ids,
        "first_relevant_rank": first_relevant_rank(offer_ids, relevant_ids),
        "hit_at_k": {k: hit_at_k(offer_ids, relevant_ids, k) for k in hit_ks},
        "mrr_at_k": mrr_at_k(offer_ids, relevant_ids, mrr_k),
    }


def evaluate_search(
    catalog: Mapping[str, ProductOffer],
    query_ids: Sequence[str],
    qrels: Mapping[str, Set[str]],
    retriever: TfidfCandidateRetriever | SearchEngine,
    rankers: Mapping[str, SearchEngine],
    *,
    candidate_k: int = 20,
    hit_ks: Sequence[int] = (1, 5),
    mrr_k: int = 5,
    other_sources_only: bool = False,
) -> dict:
    """Compare initialized rankers and the retrieval order on a common cohort.

    Queries without known positives are recorded with no metrics. All other
    queries contribute, including those whose candidate pool is empty.
    """
    candidate_k = _positive_k(candidate_k, "candidate_k")
    hit_ks = tuple(_positive_k(k, "hit cutoff") for k in hit_ks)
    mrr_k = _positive_k(mrr_k, "mrr_k")
    if not isinstance(other_sources_only, bool):
        raise ValueError("other_sources_only must be boolean")
    if TFIDF_BASELINE in rankers:
        raise ValueError(f"Ranker name {TFIDF_BASELINE!r} is reserved for the retrieval baseline")
    if retriever.catalog != catalog or any(engine.catalog != catalog for engine in rankers.values()):
        raise ValueError("Retriever and rankers must use the evaluation catalog")

    _validate_offer_ids(query_ids)
    for query_id in query_ids:
        if query_id not in catalog:
            raise ValueError(f"Unknown query offer ID: {query_id!r}")
    for query_id, relevant_ids in qrels.items():
        if query_id not in catalog or not isinstance(relevant_ids, Set) or not relevant_ids <= catalog.keys():
            raise ValueError(f"Qrels must reference catalog IDs: {query_id!r}")
        if query_id in relevant_ids:
            raise ValueError(f"Self-match in qrels for {query_id!r}")

    per_query: dict[str, dict] = {}
    for query_id in query_ids:
        pool = tuple(retriever.retrieve(query_id, candidate_k, other_sources_only=other_sources_only))
        if any(not isinstance(candidate, Candidate) or not isinstance(candidate.offer, ProductOffer)
               for candidate in pool):
            raise ValueError(f"Invalid candidate for {query_id!r}")
        candidate_ids = tuple(candidate.offer.offer_id for candidate in pool)
        _validate_offer_ids(candidate_ids)
        if (len(pool) > candidate_k or query_id in candidate_ids
                or any(catalog.get(candidate.offer.offer_id) != candidate.offer for candidate in pool)):
            raise ValueError(f"Invalid candidate pool for {query_id!r}")

        relevant_ids = qrels.get(query_id, frozenset())
        record = {
            "eligible": bool(relevant_ids),
            "candidate_ids": candidate_ids,
            "candidate_recall_at_k": None,
            "ranking": {},
        }
        per_query[query_id] = record
        if not relevant_ids:
            continue

        record["candidate_recall_at_k"] = candidate_recall_at_k(candidate_ids, relevant_ids, candidate_k)
        ranking = {TFIDF_BASELINE: _ranking_metrics(candidate_ids, relevant_ids, hit_ks, mrr_k)}
        candidate_set = set(candidate_ids)
        for method, engine in rankers.items():
            results = engine.rank_candidates(catalog[query_id], pool, top_k=candidate_k)
            ranked_ids = tuple(result.offer.offer_id for result in results)
            if len(ranked_ids) != len(candidate_ids) or set(ranked_ids) != candidate_set:
                raise ValueError(f"Ranker {method!r} changed the candidate pool for {query_id!r}")
            ranking[method] = _ranking_metrics(ranked_ids, relevant_ids, hit_ks, mrr_k)
        record["ranking"] = ranking

    eligible = [record for record in per_query.values() if record["eligible"]]
    means_by_ranker = {}
    for method in (TFIDF_BASELINE, *rankers):
        rows = [record["ranking"][method] for record in eligible]
        means_by_ranker[method] = {
            "hit_at_k": {k: mean(row["hit_at_k"][k] for row in rows) if rows else None for k in hit_ks},
            "mrr_at_k": mean(row["mrr_at_k"] for row in rows) if rows else None,
        }
    return {
        "candidate_k": candidate_k,
        "hit_ks": hit_ks,
        "mrr_k": mrr_k,
        "other_sources_only": other_sources_only,
        "eligible_query_count": len(eligible),
        "excluded_no_qrels_count": len(per_query) - len(eligible),
        "retrieval": {
            "candidate_recall_at_k": mean(record["candidate_recall_at_k"] for record in eligible) if eligible else None,
            "empty_pool_count": sum(not record["candidate_ids"] for record in eligible),
        },
        "ranking": means_by_ranker,
        "per_query": per_query,
    }
