"""Build explicit positive WDC judgments for a validated evaluation catalog."""
from collections import defaultdict
from collections.abc import Mapping

from audit_wdc import SIDES, summary
from product_catalog import ProductOffer, _wdc_offer


def build_wdc_qrels(
    pairs: list[dict],
    catalog: Mapping[str, ProductOffer],
) -> dict[str, frozenset[str]]:
    """Return symmetric, positive-only judgments keyed by sorted catalog IDs.

    Every offer must have a cluster assignment in the supplied annotations.
    Reject inconsistent data and clusters with missing explicit positive pairs.
    Observed singleton clusters have empty sets.
    """
    audit = summary(pairs)
    for check, message in (
        ("conflicting_label_pairs", "Conflicting labels for unordered pairs"),
        ("self_pairs", "Self-pairs in WDC annotations"),
        ("duplicate_canonical_pairs", "Duplicate unordered pairs"),
        ("inconsistent_offer_payloads", "Inconsistent attributes or cluster_id for offer IDs"),
        ("label_cluster_mismatches", "Label and cluster_id disagree"),
    ):
        if audit[check]:
            raise ValueError(f"{message} ({audit[check]})")
    if audit["cluster_labeled_pairs"] != len(pairs):
        raise ValueError("Missing cluster_id on a WDC pair endpoint")

    clusters: dict[str, set[str]] = defaultdict(set)
    relevant: dict[str, set[str]] = {offer_id: set() for offer_id in sorted(catalog)}
    observed: set[str] = set()
    for row_number, row in enumerate(pairs, 1):
        for side in SIDES:
            offer = _wdc_offer(row, side, f"WDC record {row_number}")
            offer_id = offer.offer_id
            if offer_id not in catalog:
                raise ValueError(f"Annotated offer_id {offer_id!r} missing from the catalog")
            if offer != catalog[offer_id]:
                raise ValueError(f"Catalog attributes differ for offer_id {offer_id!r}")
            observed.add(offer_id)
            clusters[str(row[f"cluster_id_{side}"])].add(offer_id)
        if int(row["label"]) == 1:
            left, right = (str(row[f"id_{side}"]) for side in SIDES)
            relevant[left].add(right)
            relevant[right].add(left)

    unannotated = relevant.keys() - observed
    if unannotated:
        raise ValueError(f"Catalog offer_id {min(unannotated)!r} has no pairwise annotations")

    for cluster_id, members in sorted(clusters.items()):
        for offer_id in sorted(members):
            if len(relevant[offer_id]) != len(members) - 1:
                missing = min(members - relevant[offer_id] - {offer_id})
                raise ValueError(f"Missing explicit positive pair in cluster {cluster_id!r}: {offer_id!r}, {missing!r}")

    return {offer_id: frozenset(matches) for offer_id, matches in relevant.items()}
