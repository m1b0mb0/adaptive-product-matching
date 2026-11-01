"""Optional search demonstration over a set of offers extracted from a WDC pair file.

This is NOT a benchmark evaluation: the text representation is fitted on the
search collection, and candidate recall is not evaluated. Use run_experiment.py
for scientific metrics.
"""
import argparse
import numpy as np

from audit_wdc import read_pairs
from matching import FeatureExtractor, norm, scores


def collect_offers(pairs: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in pairs:
        for side in ("left", "right"):
            oid = str(row[f"id_{side}"])
            offer = {"id": oid, **{f: row.get(f"{f}_{side}") for f in ("title", "description", "brand")}}
            if oid not in result:
                result[oid] = offer
            else:
                # Prefer a non-empty value if repeated rows differ in completeness.
                for field in ("title", "description", "brand"):
                    if not norm(result[oid][field]) and norm(offer[field]):
                        result[oid][field] = offer[field]
    return result


def as_pair(left: dict, right: dict) -> dict:
    return {"id_left": left["id"], "id_right": right["id"],
            **{f"{field}_{side}": record.get(field)
               for side, record in (("left", left), ("right", right))
               for field in ("title", "description", "brand")}}


def search(offers: dict[str, dict], query_id: str, top: int) -> list[tuple[str, float, str]]:
    if top < 1:
        raise ValueError("top must be a positive integer")
    if query_id not in offers:
        raise ValueError(f"Offer ID {query_id!r} not in this collection")
    query = offers[query_id]
    corpus = list(offers.values())
    # This fits a standalone retrieval demo ONLY, not any thesis test metric.
    extractor = FeatureExtractor().fit([as_pair(r, r) for r in corpus])
    candidates = [r for r in corpus if r["id"] != query_id]
    features = extractor.transform([as_pair(query, x) for x in candidates])
    values = scores(features.matrix, "adaptive", np.ones(4), gamma=1, penalty=.10)
    order = np.argsort(-values)[:top]
    result = []
    for i in order:
        offer = candidates[int(i)]
        title = offer.get("title")
        result.append((offer["id"], float(values[i]), str(title) if norm(title) else ""))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="One official WDC pairwise .json.gz split")
    parser.add_argument("--query-id", required=True, help="ID of an offer in that split")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    collection = collect_offers(read_pairs(args.pairs))
    print(f"Query: {args.query_id}: {collection.get(args.query_id, {}).get('title')}")
    for oid, value, title in search(collection, args.query_id, args.top):
        print(f"{value:.3f} | {oid} | {title[:140]}")
    print("Demo similarity scores only: NOT evaluated end-to-end search accuracy.")


if __name__ == "__main__":
    main()
