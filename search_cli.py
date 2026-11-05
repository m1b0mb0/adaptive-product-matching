"""Catalogue search: title retrieval followed by fixed/renormalized ranking.

This is an unlabelled catalogue demonstration, not an independent experiment.
Only the audited WDC TRAIN artifact or an FPV offer CSV is accepted by the CLI.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from matching import norm
from product_catalog import ProductOffer, load_fpv_catalog, load_wdc_catalog
from search_system import SearchEngine

WDC_TRAIN_NAME = "wdcproducts80cc20rnd000un_train_small.json.gz"
PROJECT_DIRECTORY = Path(__file__).resolve().parent


def collect_offers(pairs: list[dict]) -> dict[str, dict]:
    """Legacy helper for callers of the original CLI, not the new import path.

    Its repeated-ID completion behavior is retained for API compatibility.
    New catalogue importers instead reject conflicting repeated observations.
    """
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
    """Compatibility wrapper using the new retrieval/renormalization engine."""
    if isinstance(top, bool) or not isinstance(top, int) or top < 1:
        raise ValueError("top must be a positive integer")
    if query_id not in offers:
        raise ValueError(f"Offer ID {query_id!r} not in this collection")
    def text(value):
        return str(value) if norm(value) else ""
    catalog = {str(oid): ProductOffer(
        offer_id=str(oid), title=text(record.get("title")),
        description=text(record.get("description")), brand=text(record.get("brand")),
        source=text(record.get("source")), url=text(record.get("url")))
        for oid, record in offers.items()}
    results = SearchEngine(catalog).search(str(query_id), candidate_k=top, top_k=top)
    return [(row.offer.offer_id, row.similarity_score, row.offer.title) for row in results]


def verify_wdc_train(path: str | Path) -> None:
    """Reject other named splits before opening them; pin TRAIN by checksum."""
    path = Path(path)
    if path.name != WDC_TRAIN_NAME:
        raise ValueError(f"WDC search accepts only audited TRAIN: {WDC_TRAIN_NAME}")
    manifest = json.loads((PROJECT_DIRECTORY / "docs/archive_manifest.json").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["members"] if entry["name"] == WDC_TRAIN_NAME]
    if len(entries) != 1:
        raise ValueError("Audited TRAIN checksum is missing or ambiguous")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != entries[0]["sha256"]:
        raise ValueError("WDC TRAIN checksum mismatch; search input was not the audited artifact")


def search_payload(catalog: dict[str, ProductOffer], query_id: str, *,
                   method: str = "renorm", candidate_k: int = 20, top_k: int = 5,
                   other_stores_only: bool = False) -> dict:
    """Perform two explicit search stages with no match labels or thresholds."""
    engine = SearchEngine(catalog, method=method)
    candidates = engine.retrieve(query_id, candidate_k=candidate_k,
                                 other_sources_only=other_stores_only)
    results = engine.search(query_id, candidate_k=candidate_k, top_k=top_k,
                            other_sources_only=other_stores_only)
    return {
        "catalog_size": len(catalog),
        "known_sources": sorted({offer.source for offer in catalog.values() if norm(offer.source)}),
        "unknown_source_offers": sum(not norm(offer.source) for offer in catalog.values()),
        "query": asdict(catalog[query_id]),
        "retrieval": {"method": "title_tfidf_cosine", "candidate_k": candidate_k,
                      "candidate_count": len(candidates),
                      "other_stores_only": other_stores_only,
                      "candidates": [{"offer_id": row.offer.offer_id,
                                      "retrieval_score": row.retrieval_score} for row in candidates]},
        "ranking": {"method": method, "top_k": top_k,
                    "score_type": "algorithmic similarity, not calibrated match probability",
                    "results": [{"offer": asdict(row.offer),
                                 "retrieval_score": row.retrieval_score,
                                 "similarity_score": row.similarity_score,
                                 "feature_similarities": row.feature_similarities} for row in results]},
        "verification": {"matches_confirmed": False, "search_metrics_computed": False,
                         "gold_labels_used": False},
    }


def render_console(payload: dict) -> str:
    query, retrieval, ranking = payload["query"], payload["retrieval"], payload["ranking"]
    lines = [f"Каталог: {payload['catalog_size']} пропозицій.",
             f"Запит: {query['offer_id']} | {query['title'] or '(назва відсутня)'}",
             f"Джерело запиту: {query['source'] or 'невідоме'}",
             f"Відібрано TF-IDF кандидатів: {retrieval['candidate_count']} (ліміт {retrieval['candidate_k']}).",
             f"Ранжування: {ranking['method']}; показано {len(ranking['results'])} результатів."]
    if payload["unknown_source_offers"]:
        lines.append(f"Магазин невідомий для {payload['unknown_source_offers']} пропозицій каталогу.")
    for index, row in enumerate(ranking["results"], 1):
        offer = row["offer"]
        lines.extend(["", f"{index}. {offer['title'] or '(назва відсутня)'} [{offer['offer_id']}]",
                      f"   Джерело: {offer['source'] or 'невідоме'}",
                      f"   URL: {offer['url'] or 'відсутній'}",
                      f"   Подібність: {row['similarity_score']:.6f}; TF-IDF retrieval: {row['retrieval_score']:.6f}"])
        attributes = {"brand": offer["brand"], **offer["technical_attributes"]}
        for key in ("price", "priceCurrency", "price_uah", "record_status"):
            if key in offer["attributes"]:
                attributes[key] = offer["attributes"][key]
        present = [f"{key}={value}" for key, value in attributes.items()
                   if value is not None and norm(value)]
        if present:
            lines.append("   Атрибути: " + "; ".join(present))
    if not ranking["results"]:
        lines.append("Немає кандидатів із ненульовою текстовою подібністю за заданими умовами.")
    lines.extend(["", "Оцінки алгоритму не є каліброваними ймовірностями тотожності.",
                  "Тотожність результатів не підтверджено; метрики якості пошуку не обчислювалися."])
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--catalog", type=Path, help="Шлях до WDC TRAIN JSONL/gzip або FPV offers CSV")
    sources.add_argument("--pairs", type=Path, help="Сумісний старий аргумент для WDC TRAIN")
    parser.add_argument("--catalog-format", "--format", choices=("wdc", "fpv"), default="wdc")
    parser.add_argument("--query-id", required=True, help="offer_id пропозиції з каталогу")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top", "--top-k", type=int, default=5)
    parser.add_argument("--method", choices=("fixed", "renorm"), default="renorm")
    parser.add_argument("--other-stores-only", "--other-sources-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    if args.pairs is not None and args.catalog_format != "wdc":
        parser.error("--pairs is supported only for WDC TRAIN; use --catalog for FPV")
    path = args.catalog if args.catalog is not None else args.pairs
    try:
        if args.catalog_format == "wdc":
            verify_wdc_train(path)
            catalog = load_wdc_catalog(path)
        else:
            catalog = load_fpv_catalog(path)
        payload = search_payload(catalog, args.query_id, method=args.method,
                                 candidate_k=args.candidate_k, top_k=args.top,
                                 other_stores_only=args.other_stores_only)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Помилка пошуку: {error}\n")
    payload["catalog_format"] = args.catalog_format
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(render_console(payload))


if __name__ == "__main__":
    main()
