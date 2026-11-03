"""Audit local FPV offer candidates and prepare an unconfirmed manual review queue.

No source records are changed; proposed labels never become ground truth here.
All grouping keys are provisional lexical keys, not resolved product entities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse

from fpv_dataset import CANDIDATE_COLUMNS, OFFER_COLUMNS, read_csv_records

DEFAULT_DIRECTORY = Path("data/external/fpv_motors")
VARIANT_FIELDS = ("brand", "family_model", "stator_code", "kv_rpm_per_volt")
MANUAL_COLUMNS = (
    "confirmed_label_1_match_0_nonmatch", "review_status", "reviewer", "reviewed_on",
    "evidence_url", "review_notes", "exclusion_reason", "variant_resolution",
)
HASH_COLUMNS = ("offers_sha256", "candidates_sha256")


def _key(value: str) -> str:
    """Case and whitespace only; do not erase model revisions, pack sizes or zeros."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _missing_summary(rows: list[dict[str, str]], columns: list[str]) -> dict:
    return {column: {
        "present": sum(bool(row[column].strip()) for row in rows),
        "missing": sum(not row[column].strip() for row in rows),
        "missing_fraction": sum(not row[column].strip() for row in rows) / len(rows)
        if rows else None,
    } for column in columns}


def _duplicates(rows: list[dict[str, str]], fields: tuple[str, ...],
                identifier: str, *, normalize: bool = False,
                skip_incomplete: bool = True) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        values = tuple(_key(row[field]) if normalize else row[field] for field in fields)
        if not skip_incomplete or all(values):
            groups[values].append(row[identifier])
    return [{"values": list(key), "ids": ids} for key, ids in sorted(groups.items())
            if len(ids) > 1]


def _numeric_problem(value: str, *, integer: bool = False,
                     allow_zero: bool = False) -> bool:
    if not value.strip():
        return False
    try:
        number = Decimal(value)
    except InvalidOperation:
        return True
    return (not number.is_finite() or number < 0 or (number == 0 and not allow_zero)
            or (integer and number != number.to_integral_value()))


def _offer_issues(row: dict[str, str]) -> list[str]:
    issues = []
    for field in ("offer_id", "store", "raw_title", "source_url"):
        if not row[field].strip():
            issues.append(f"missing_{field}")
    try:
        parsed = urlparse(row["source_url"])
        bad_url = (parsed.scheme not in {"http", "https"} or not parsed.netloc
                   or bool(re.search(r"\s", row["source_url"])))
    except ValueError:
        bad_url = True
    if row["source_url"] and bad_url:
        issues.append("malformed_source_url")
    if row["collected_on"]:
        try:
            if date.fromisoformat(row["collected_on"]).isoformat() != row["collected_on"]:
                raise ValueError("Expected YYYY-MM-DD")
        except ValueError:
            issues.append("malformed_collected_on")
    for field in ("kv_rpm_per_volt", "units_in_listing", "price_uah"):
        if _numeric_problem(row[field], integer=field == "units_in_listing",
                            allow_zero=field == "price_uah"):
            issues.append(f"invalid_{field}")
    if row["voltage_s"]:
        voltage = re.sub(r"\s+", "", row["voltage_s"].upper())
        if not re.fullmatch(r"[1-9]\d*S?(?:[-–—][1-9]\d*S)?", voltage) or not voltage.endswith("S"):
            issues.append("nonstandard_voltage_s")
    if row["stator_code"] and not re.fullmatch(r"\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?",
                                              row["stator_code"]):
        issues.append("nonstandard_stator_code")
    if row["record_status"] in {"needs_variant_split", "needs_variant_check"}:
        issues.append(row["record_status"])
    if not row["kv_rpm_per_volt"].strip():
        issues.append("kv_variant_unspecified")
    if row["units_in_listing"].strip() and not _numeric_problem(row["units_in_listing"]):
        if Decimal(row["units_in_listing"]) > 1:
            issues.append("multi_unit_listing")
    if "/" in row["family_model"]:
        issues.append("family_slash_name_needs_semantic_review")
    title = row["raw_title"].casefold()
    if "пропелер" in title or re.search(r"\b(?:cw|ccw)\b", title):
        issues.append("bundle_or_rotation_needs_review")
    if row["record_status"] == "listing_verified":
        issues.append("catalog_only_source_check_needed")
    return issues


def _group_models(offers: list[dict[str, str]]) -> dict:
    family = defaultdict(list)
    variants = defaultdict(list)
    complete_variants = defaultdict(list)
    incomplete = []
    for offer in offers:
        family_key = tuple(_key(offer[field]) for field in ("brand", "family_model"))
        variant_key = tuple(_key(offer[field]) for field in VARIANT_FIELDS)
        family[family_key].append(offer)
        variants[variant_key].append(offer)
        if all(variant_key):
            complete_variants[variant_key].append(offer)
        else:
            incomplete.append(offer["offer_id"])
    cross_store = []
    for key, group in sorted(complete_variants.items()):
        stores = sorted({row["store"] for row in group})
        if len(stores) < 2:
            continue
        pairs = [[a["offer_id"], b["offer_id"]] for a, b in combinations(group, 2)
                 if a["store"] != b["store"]]
        cross_store.append({
            "provisional_key": dict(zip(VARIANT_FIELDS, key)),
            "offer_ids": [row["offer_id"] for row in group], "stores": stores,
            "potential_cross_store_pairs": pairs,
            "units_in_listing_values": sorted({row["units_in_listing"] for row in group}),
            "status": "requires_manual_confirmation_of_revision_bundle_and_identity",
        })
    return {
        "key_normalization": "NFKC + casefold + collapsed whitespace; no brand aliases or inferred attributes",
        "identity_warning": "Provisional keys are not gold entities; equal stator and KV alone never establish identity.",
        "family_key_fields": ["brand", "family_model"],
        "provisional_family_groups": len(family),
        "variant_key_fields": list(VARIANT_FIELDS),
        "provisional_variant_groups_including_incomplete": len(variants),
        "complete_provisional_variant_groups": len(complete_variants),
        "offers_with_complete_variant_key": sum(map(len, complete_variants.values())),
        "incomplete_variant_offer_ids": incomplete,
        "cross_store_group_count": len(cross_store),
        "potential_cross_store_pair_count": sum(len(group["potential_cross_store_pairs"])
                                                for group in cross_store),
        "cross_store_groups": cross_store,
        "family_groups": [{"provisional_key": dict(zip(("brand", "family_model"), key)),
                           "offer_ids": [row["offer_id"] for row in rows]}
                          for key, rows in sorted(family.items())],
    }


def audit_data(offers_path: str | Path, candidates_path: str | Path) -> tuple[dict, list[dict]]:
    """Return a machine audit and all queue rows, never assigning confirmed labels."""
    offers = read_csv_records(offers_path, required_columns=OFFER_COLUMNS)
    candidates = read_csv_records(candidates_path, required_columns=CANDIDATE_COLUMNS)
    if not offers or not candidates:
        raise ValueError("Offers and candidate CSV files must contain records")
    offer_columns, pair_columns = list(offers[0]), list(candidates[0])
    reserved_columns = (set(MANUAL_COLUMNS) | set(HASH_COLUMNS)
                        | {"issue_flags", "review_priority"}
                        | {f"{field}_{side}" for field in offer_columns for side in ("a", "b")})
    collisions = sorted(set(pair_columns) & reserved_columns)
    if collisions:
        raise ValueError(f"Candidate columns collide with reserved review queue fields: {collisions}")
    hashes = {"offers_sha256": _hash(offers_path), "candidates_sha256": _hash(candidates_path)}
    offer_counts = Counter(row["offer_id"] for row in offers)
    pair_counts = Counter(row["pair_id"] for row in candidates)
    # Duplicate offer identifiers are reported; ambiguous references are never resolved as gold.
    by_id = {row["offer_id"]: row for row in offers if offer_counts[row["offer_id"]] == 1}
    offer_issues = {row["offer_id"]: _offer_issues(row) for row in offers}
    offer_issues = {key: value for key, value in offer_issues.items() if value}
    unknown_references, self_pairs, invalid_labels = [], [], []
    canonical_pairs = defaultdict(list)
    queue = []
    for pair in candidates:
        pair_id = pair["pair_id"]
        flags = []
        if not pair_id.strip():
            flags.append("missing_pair_id")
        if pair_counts[pair_id] > 1:
            flags.append("duplicate_pair_id")
        if pair["offer_a"] == pair["offer_b"]:
            self_pairs.append(pair_id)
            flags.append("self_pair")
        canonical_pairs[tuple(sorted((pair["offer_a"], pair["offer_b"])))].append(pair_id)
        if pair["proposed_label_1_match_0_nonmatch"] not in {"0", "1"}:
            invalid_labels.append(pair_id)
            flags.append("invalid_proposed_label")
        row = dict(pair)
        for side in ("a", "b"):
            offer_id = pair[f"offer_{side}"]
            if offer_counts[offer_id] != 1:
                unknown_references.append({"pair_id": pair_id, "side": side,
                                           "offer_id": offer_id,
                                           "reason": "ambiguous" if offer_counts[offer_id] else "missing"})
                flags.append(f"unresolved_reference_{side}")
            offer = by_id.get(offer_id, {})
            row.update({f"{field}_{side}": offer.get(field, "") for field in offer_columns})
            flags.extend(f"{side}:{issue}" for issue in offer_issues.get(offer_id, []))
        left, right = by_id.get(pair["offer_a"]), by_id.get(pair["offer_b"])
        if left and right:
            if left["store"] == right["store"]:
                flags.append("same_store_pair")
            for field in ("brand", "family_model", "stator_code", "kv_rpm_per_volt",
                          "units_in_listing"):
                if left[field].strip() and right[field].strip() and _key(left[field]) != _key(right[field]):
                    flags.append(f"known_{field}_difference")
        row["issue_flags"] = ";".join(flags)
        row["review_priority"] = "high" if any(
            "variant" in flag or "reference" in flag or "bundle" in flag
            or "duplicate" in flag or "invalid" in flag or "self_pair" == flag
            for flag in flags) else "normal"
        row.update(hashes)
        row.update({column: "" for column in MANUAL_COLUMNS})
        row["review_status"] = "pending"
        queue.append(row)
    repeated_pairs = [{"offer_ids": list(key), "pair_ids": ids}
                      for key, ids in sorted(canonical_pairs.items()) if len(ids) > 1]
    pair_ids = {row["pair_id"] for row in candidates}
    referenced_offer_ids = {row[f"offer_{side}"] for row in candidates for side in ("a", "b")}
    offers_without_pairs = sorted(set(offer_counts) - referenced_offer_ids)
    duplicate_pair_ids = sorted(key for key, value in pair_counts.items() if value > 1)
    malformed_offers = {key: [flag for flag in values if flag.startswith(("missing_", "invalid_", "malformed_", "nonstandard_"))]
                        for key, values in offer_issues.items()}
    malformed_offers = {key: value for key, value in malformed_offers.items() if value}
    report = {
        "schema_version": 1,
        "scope": "Local syntactic/data audit only; source URLs not opened; no matching metrics or experiments.",
        "sources": {"offers": str(Path(offers_path)), "candidates": str(Path(candidates_path)), **hashes},
        "counts": {"offers": len(offers), "unique_offer_ids": len(offer_counts),
                   "stores": len({row["store"] for row in offers}), "candidate_pairs": len(candidates),
                   "unique_pair_ids": len(pair_ids),
                   "unique_existing_offers_in_candidate_pairs": len(set(offer_counts) & referenced_offer_ids),
                   "offers_without_candidate_pairs": len(offers_without_pairs),
                   "proposed_positive": sum(row["proposed_label_1_match_0_nonmatch"] == "1" for row in candidates),
                   "proposed_negative": sum(row["proposed_label_1_match_0_nonmatch"] == "0" for row in candidates),
                   "confirmed_labels_in_sources": 0,
                   "source_entity_ids_present": sum(bool(row["entity_id"].strip()) for row in offers)},
        "stores": dict(sorted(Counter(row["store"] for row in offers).items())),
        "offer_columns": offer_columns, "candidate_columns": pair_columns,
        "offer_missingness": _missing_summary(offers, offer_columns),
        "candidate_missingness": _missing_summary(candidates, pair_columns),
        "record_status_counts": dict(sorted(Counter(row["record_status"] for row in offers).items())),
        "source_verification_status_counts": dict(sorted(Counter(row["verification_status"] for row in candidates).items())),
        "label_policy": "All source labels are proposals; every generated queue row starts pending with blank confirmed label, regardless of source status.",
        "integrity": {
            "duplicate_offer_ids": sorted(key for key, value in offer_counts.items() if value > 1),
            "duplicate_pair_ids": duplicate_pair_ids,
            "unresolved_references": unknown_references, "self_pairs": self_pairs,
            "invalid_proposed_label_pair_ids": invalid_labels,
            "duplicate_canonical_pairs": repeated_pairs,
            "duplicate_exact_offers_excluding_id": _duplicates(offers, tuple(field for field in offer_columns if field != "offer_id"), "offer_id", skip_incomplete=False),
            "duplicate_source_urls": _duplicates(offers, ("source_url",), "offer_id"),
            "duplicate_normalized_titles": _duplicates(offers, ("raw_title",), "offer_id", normalize=True),
            "duplicate_exact_candidate_rows": _duplicates(candidates, tuple(pair_columns), "pair_id", skip_incomplete=False),
            "malformed_offer_values": malformed_offers,
            "source_urls_syntactically_checked": len(offers), "live_source_validation_performed": False,
        },
        "offer_review_flags": offer_issues,
        "offer_ids_without_candidate_pairs": offers_without_pairs,
        "queue_flags_are_labels": False,
        "models": _group_models(offers),
        "queue": {"rows": len(queue), "initial_status": "pending", "initial_confirmed_labels": 0,
                  "manual_columns": list(MANUAL_COLUMNS),
                  "source_hash_columns": list(HASH_COLUMNS),
                  "preservation_policy": "Reruns preserve all manual and additional columns; any source/hash/generated-field drift or pair-set change refuses overwrite."},
    }
    return report, queue


def write_review_queue(path: str | Path, rows: list[dict[str, str]]) -> dict:
    """Write UTF-8 BOM CSV while preserving manual work and rejecting source drift."""
    if not rows:
        raise ValueError("Cannot write an empty review queue")
    path = Path(path)
    columns = list(rows[0])
    generated = [column for column in columns if column not in MANUAL_COLUMNS]
    expected = {row["pair_id"]: row for row in rows}
    if len(expected) != len(rows) or "" in expected:
        raise ValueError("Cannot safely write review queue: candidate pair IDs must be unique and nonblank")
    preserved = 0
    if path.exists():
        existing = read_csv_records(path, required_columns=columns)
        previous = {row["pair_id"]: row for row in existing}
        if len(previous) != len(existing) or previous.keys() != expected.keys():
            raise ValueError("Review queue pair set changed; refusing overwrite to protect manual annotations")
        extras = [column for column in existing[0] if column not in columns]
        columns.extend(extras)
        for pair_id, row in expected.items():
            prior = previous[pair_id]
            if any(prior[column] != row[column] for column in generated):
                raise ValueError(f"Source or generated-field drift for {pair_id}; refusing overwrite to protect manual annotations")
            if prior["confirmed_label_1_match_0_nonmatch"] not in {"", "0", "1"}:
                raise ValueError(f"Invalid confirmed label for {pair_id}; refusing overwrite")
            for column in (*MANUAL_COLUMNS, *extras):
                row[column] = prior[column]
            preserved += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids colliding with a source named e.g. queue.csv.tmp.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "rows": len(rows), "preserved_existing_rows": preserved,
            "pending_rows": sum(row["review_status"] == "pending" for row in rows),
            "confirmed_label_cells_present": sum(bool(row["confirmed_label_1_match_0_nonmatch"]) for row in rows),
            "encoding": "utf-8-sig", "delimiter": ","}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offers", type=Path, default=DEFAULT_DIRECTORY / "fpv_motors_offers_pilot.csv")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_DIRECTORY / "fpv_motors_pair_candidates.csv")
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_DIRECTORY / "review_queue.csv")
    parser.add_argument("--out", type=Path, default=Path("docs/fpv_dataset_audit.json"))
    args = parser.parse_args(argv)
    protected = {args.offers.resolve(), args.pairs.resolve()}
    if args.review_queue.resolve() in protected or args.out.resolve() in protected or args.review_queue.resolve() == args.out.resolve():
        parser.error("Output paths must be distinct and cannot overwrite source CSV files")
    try:
        report, rows = audit_data(args.offers, args.pairs)
        report["queue"].update(write_review_queue(args.review_queue, rows))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"FPV audit failed: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "models": {
        key: value for key, value in report["models"].items() if key.endswith("count")},
        "review_queue": str(args.review_queue), "audit": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
