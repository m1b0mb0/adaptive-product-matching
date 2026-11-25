"""Offer-level masking before search indexing and diagnostics after evaluation."""
from collections.abc import Mapping, Set
from dataclasses import replace
from hashlib import sha256
from statistics import mean

from matching import norm
from product_catalog import ProductOffer


TEXT_FIELDS = ("title", "brand", "description")


def mask_catalog(catalog: Mapping[str, ProductOffer], fields: Set[str], seed: int) -> dict[str, ProductOffer]:
    """Copy all offers and blank half (rounded down) of each selected field's observed values."""
    if not isinstance(fields, Set) or fields - set(TEXT_FIELDS):
        raise ValueError("Mask fields must be a set drawn from title, brand and description")
    if type(seed) is not int:
        raise ValueError("Mask seed must be an integer")
    if any(key != offer.offer_id for key, offer in catalog.items()):
        raise ValueError("Catalog keys must equal ProductOffer.offer_id")

    changes: dict[str, dict[str, str]] = {key: {} for key in catalog}
    for field in sorted(fields):
        eligible = sorted(
            (sha256(f"{seed}:{field}:{key}".encode("utf-8")).digest(), key)
            for key, offer in catalog.items() if norm(getattr(offer, field))
        )
        for _, key in eligible[:len(eligible) // 2]:
            changes[key][field] = ""
    return {key: replace(catalog[key], **changes[key]) for key in sorted(catalog)}


def summarize_masked_validation(
    clean: Mapping[str, ProductOffer], masked: Mapping[str, ProductOffer],
    qrels: Mapping[str, Set[str]], evaluation: dict,
) -> dict:
    """Describe a completed evaluation of the WDC cohort of disjoint, verified pairs."""
    rows = evaluation["per_query"]
    if not (clean.keys() == masked.keys() == qrels.keys() == rows.keys()) or not rows:
        raise ValueError("Diagnostics require the full paired validation cohort")
    if evaluation["candidate_k"] != 20 or evaluation["mrr_k"] != 5:
        raise ValueError("Diagnostics require candidate_k=20 and mrr_k=5")
    found = []
    both_titles = 0
    for key, matches in qrels.items():
        if len(matches) != 1:
            raise ValueError("Expected one verified partner per query")
        partner, = matches
        if key == partner or qrels.get(partner) != {key}:
            raise ValueError("Expected symmetric, non-self pairs")
        if key < partner and norm(masked[key].title) and norm(masked[partner].title):
            both_titles += 1
        if partner in rows[key]["candidate_ids"]:
            found.append(rows[key])

    natural = {field: sum(not norm(getattr(offer, field)) for offer in clean.values())
               for field in TEXT_FIELDS}
    missing = {field: sum(not norm(getattr(offer, field)) for offer in masked.values())
               for field in TEXT_FIELDS}
    return {
        "naturally_missing": natural,
        "artificially_removed": {field: missing[field] - natural[field] for field in TEXT_FIELDS},
        "empty_title_queries": missing["title"],
        "clusters_with_both_titles": both_titles,
        "empty_pool_count": evaluation["retrieval"]["empty_pool_count"],
        "retrieval_miss_fraction": (len(rows) - len(found)) / len(rows),
        "conditional_mrr_at_5": {
            method: mean(row["ranking"][method]["mrr_at_k"] for row in found) if found else None
            for method in evaluation["ranking"]
        },
    }
