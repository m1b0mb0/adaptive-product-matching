"""Metrics over ordered offer IDs and explicit positive judgments."""
from collections.abc import Sequence, Set
from numbers import Integral


def _positive_k(k: int, name: str = "k") -> int:
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(k)


def _validate_offer_ids(offer_ids: Sequence[str]) -> None:
    if not isinstance(offer_ids, Sequence) or isinstance(offer_ids, (str, bytes)):
        raise ValueError("Expected a sequence of offer IDs")
    if any(not isinstance(offer_id, str) or not offer_id or offer_id != offer_id.strip()
           for offer_id in offer_ids):
        raise ValueError("Offer IDs must be nonempty, unpadded strings")
    if len(set(offer_ids)) != len(offer_ids):
        raise ValueError("Duplicate offer IDs")


def _validate_judgments(offer_ids: Sequence[str], relevant_ids: Set[str]) -> None:
    _validate_offer_ids(offer_ids)
    if not isinstance(relevant_ids, Set):
        raise ValueError("Relevant IDs must be provided as a set")
    if not relevant_ids:
        raise ValueError("The query must have at least one known relevant offer")


def first_relevant_rank(offer_ids: Sequence[str], relevant_ids: Set[str]) -> int | None:
    """Return the full list's first known match at a one-based rank."""
    _validate_judgments(offer_ids, relevant_ids)
    return next((rank for rank, offer_id in enumerate(offer_ids, 1)
                 if offer_id in relevant_ids), None)


def candidate_recall_at_k(offer_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of the query's known positives present in its first k candidates."""
    k = _positive_k(k)
    _validate_judgments(offer_ids, relevant_ids)
    return sum(offer_id in relevant_ids for offer_id in offer_ids[:k]) / len(relevant_ids)


def hit_at_k(offer_ids: Sequence[str], relevant_ids: Set[str], k: int) -> int:
    """Return one if the first k positions contain a known match."""
    k = _positive_k(k)
    rank = first_relevant_rank(offer_ids, relevant_ids)
    return int(rank is not None and rank <= k)


def mrr_at_k(offer_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Return the query's contribution to mean reciprocal rank at k."""
    k = _positive_k(k)
    rank = first_relevant_rank(offer_ids, relevant_ids)
    return 1.0 / rank if rank is not None and rank <= k else 0.0
