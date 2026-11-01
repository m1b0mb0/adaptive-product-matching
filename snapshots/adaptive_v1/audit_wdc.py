"""Validate official WDC Products JSONL splits without fitting matching models."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

FIELDS = ("title", "description", "brand", "price", "priceCurrency")
SIDES = ("left", "right")
REQUIRED = {"id_left", "id_right", "title_left", "title_right", "label"}
SELECTED_MEMBERS = {
    "train": "wdcproducts80cc20rnd000un_train_small.json.gz",
    "valid": "wdcproducts80cc20rnd000un_valid_small.json.gz",
    "test": "wdcproducts80cc20rnd100un_gs.json.gz",
}


def is_missing(value: object) -> bool:
    """Absent/null/blank values are missing; numeric zero is present."""
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, float) and not math.isfinite(value))


def _identifier(value: object, name: str, location: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or is_missing(value):
        raise ValueError(f"{location}: invalid {name}: {value!r}")
    return str(value).strip()


def _label(value: object, location: str) -> int:
    # int(0.7), int(True), and int('01') silently accept malformed labels.
    if type(value) is int and value in (0, 1):
        return value
    if type(value) is str and value in ("0", "1"):
        return int(value)
    raise ValueError(f"{location}: invalid binary label: {value!r}")


def _validate_record(record: object, location: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{location}: expected a JSON object")
    absent = REQUIRED - record.keys()
    if absent:
        raise ValueError(f"{location}: missing required fields: {sorted(absent)}")
    _label(record["label"], location)
    for side in SIDES:
        _identifier(record[f"id_{side}"], f"id_{side}", location)
        cluster = f"cluster_id_{side}"
        if cluster in record and not is_missing(record[cluster]):
            _identifier(record[cluster], cluster, location)
        for field in FIELDS:
            name, value = f"{field}_{side}", record.get(f"{field}_{side}")
            permitted = (str, int, float) if field == "price" else (str,)
            if value is not None and (isinstance(value, bool) or not isinstance(value, permitted)):
                raise ValueError(f"{location}: invalid {name} type: {type(value).__name__}")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{location}: nonfinite {name}")
    if "pair_id" in record:
        _identifier(record["pair_id"], "pair_id", location)
    if "is_hard_negative" in record and type(record["is_hard_negative"]) is not bool:
        raise ValueError(f"{location}: is_hard_negative must be a JSON boolean")


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def _read_lines(lines, source: str) -> list[dict]:
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        location = f"{source}:{line_number}"
        try:
            record = json.loads(line, parse_constant=_reject_nonfinite)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{location}: invalid JSON: {exc}") from exc
        _validate_record(record, location)
        records.append(record)
    if not records:
        raise ValueError(f"Empty split: {source}")
    return records


def read_pairs(path: str | Path) -> list[dict]:
    """Read plain or gzip JSONL, validate every row, preserve original values."""
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return _read_lines(stream, str(path))


def offers(pairs: list[dict]) -> set[str]:
    return {str(pair[f"id_{side}"]).strip() for pair in pairs for side in SIDES}


def _canonical_pair(pair: dict) -> tuple[str, str]:
    return tuple(sorted(str(pair[f"id_{side}"]).strip() for side in SIDES))


def _normalized(value: object) -> str:
    return "" if is_missing(value) else " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _signature(pair: dict, side: str, fields: tuple[str, ...]) -> str | None:
    values = [_normalized(pair.get(f"{field}_{side}")) for field in fields]
    if not any(values):
        return None
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()


def _offer_versions(pairs: list[dict]) -> dict[str, set[str]]:
    versions = defaultdict(set)
    for pair in pairs:
        for side in SIDES:
            payload = [pair.get(f"{field}_{side}") for field in (*FIELDS, "cluster_id")]
            versions[str(pair[f"id_{side}"]).strip()].add(json.dumps(payload, ensure_ascii=False))
    return versions


def summary(pairs: list[dict]) -> dict:
    if not pairs:
        raise ValueError("Cannot audit an empty split")
    for row, pair in enumerate(pairs, 1):
        _validate_record(pair, f"record {row}")
    labels = Counter(str(_label(pair["label"], "record")) for pair in pairs)
    missing, counts = {}, {}
    for field in FIELDS:
        missing[field], counts[field] = {}, {}
        for side in SIDES:
            name = f"{field}_{side}"
            absent = sum(name not in pair for pair in pairs)
            nulls = sum(name in pair and pair[name] is None for pair in pairs)
            blanks = sum(isinstance(pair.get(name), str) and not pair[name].strip() for pair in pairs)
            total = absent + nulls + blanks
            missing[field][side] = round(total / len(pairs), 6)
            counts[field][side] = {"absent_key": absent, "null": nulls, "blank": blanks, "missing": total}
    canonical, orientations = defaultdict(set), defaultdict(set)
    for pair in pairs:
        key = _canonical_pair(pair)
        canonical[key].add(_label(pair["label"], "record"))
        orientations[key].add(tuple(str(pair[f"id_{side}"]).strip() for side in SIDES))
    direct = [tuple(str(pair[f"id_{side}"]).strip() for side in SIDES) for pair in pairs]
    explicit_pair_ids = [str(pair["pair_id"]) for pair in pairs if "pair_id" in pair]
    hard = [pair for pair in pairs if "is_hard_negative" in pair and _label(pair["label"], "record") == 0]
    offer_versions = _offer_versions(pairs)
    unique_payloads = {}
    for pair in pairs:
        for side in SIDES:
            unique_payloads.setdefault(str(pair[f"id_{side}"]).strip(),
                                       {field: pair.get(f"{field}_{side}") for field in FIELDS})
    clusters = [pair for pair in pairs if all(not is_missing(pair.get(f"cluster_id_{side}")) for side in SIDES)]
    return {
        "pairs": len(pairs), "unique_offers": len(offer_versions), "labels": {"0": labels["0"], "1": labels["1"]},
        "schema_keys": sorted(set().union(*(pair.keys() for pair in pairs))),
        "schema_keysets": len({tuple(sorted(pair)) for pair in pairs}),
        "observed_negative_hard_rate": sum(pair["is_hard_negative"] for pair in hard) / len(hard) if hard else None,
        "negative_hard_flag_present": len(hard),
        "hard_negative_counts": dict(Counter(str(pair["is_hard_negative"]).lower() for pair in hard)),
        "positive_hard_flag_true": sum(_label(pair["label"], "record") == 1 and pair.get("is_hard_negative") is True for pair in pairs),
        "missing_rates_by_pair_side": missing, "missing_counts_by_pair_side": counts,
        "missing_rates_by_unique_offer": {field: round(sum(is_missing(offer[field]) for offer in unique_payloads.values()) / len(unique_payloads), 6) for field in FIELDS},
        "both_present_pairs_by_field": {field: sum(all(not is_missing(pair.get(f"{field}_{side}")) for side in SIDES) for pair in pairs) for field in FIELDS},
        "duplicate_rows": len(pairs) - len({json.dumps(pair, sort_keys=True, ensure_ascii=False) for pair in pairs}),
        "duplicate_pair_ids": len(explicit_pair_ids) - len(set(explicit_pair_ids)),
        "explicit_pair_ids": len(explicit_pair_ids),
        "pair_id_mismatches": sum("pair_id" in pair and str(pair["pair_id"]) != "#".join(str(pair[f"id_{side}"]).strip() for side in SIDES) for pair in pairs),
        "duplicate_direct_pairs": len(pairs) - len(set(direct)),
        "duplicate_canonical_pairs": len(pairs) - len(canonical),
        "reversed_pair_groups": sum(len(value) > 1 for value in orientations.values()),
        "conflicting_label_pairs": sum(len(value) > 1 for value in canonical.values()),
        "self_pairs": sum(pair[0] == pair[1] for pair in direct),
        "cluster_labeled_pairs": len(clusters),
        "unique_clusters": len({str(pair[f"cluster_id_{side}"]) for pair in pairs for side in SIDES if not is_missing(pair.get(f"cluster_id_{side}"))}),
        "label_cluster_mismatches": sum(int(str(pair["cluster_id_left"]) == str(pair["cluster_id_right"])) != _label(pair["label"], "record") for pair in clusters),
        "inconsistent_offer_payloads": sum(len(versions) > 1 for versions in offer_versions.values()),
    }


def audit(train: list[dict], valid: list[dict], test: list[dict]) -> dict:
    splits = {"train": train, "valid": valid, "test": test}
    result = {"splits": {name: summary(pairs) for name, pairs in splits.items()}}
    extractors = {
        "offer_id": offers,
        "canonical_pair": lambda rows: {_canonical_pair(pair) for pair in rows},
        "pair_id": lambda rows: {str(pair["pair_id"]) for pair in rows if "pair_id" in pair},
        "cluster_id": lambda rows: {str(pair[f"cluster_id_{side}"]) for pair in rows for side in SIDES if not is_missing(pair.get(f"cluster_id_{side}"))},
        "text_signature": lambda rows: {value for pair in rows for side in SIDES if (value := _signature(pair, side, ("title", "description", "brand")))},
        "title_signature": lambda rows: {value for pair in rows for side in SIDES if (value := _signature(pair, side, ("title",)))},
    }
    for kind, extractor in extractors.items():
        sets = {name: extractor(pairs) for name, pairs in splits.items()}
        result[f"{kind}_overlaps"] = {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in combinations(splits, 2)}
    all_labels, pair_splits = defaultdict(set), defaultdict(set)
    for split_name, pairs in splits.items():
        for pair in pairs:
            all_labels[_canonical_pair(pair)].add(_label(pair["label"], "record"))
            pair_splits[_canonical_pair(pair)].add(split_name)
    result["cross_split_conflicting_label_pairs"] = sum(len(value) > 1 and len(pair_splits[key]) > 1 for key, value in all_labels.items())
    result["inconsistent_offer_payloads_across_splits"] = sum(len(value) > 1 for value in _offer_versions(train + valid + test).values())
    integrity_fields = ("duplicate_rows", "duplicate_pair_ids", "pair_id_mismatches", "duplicate_canonical_pairs", "conflicting_label_pairs", "self_pairs", "label_cluster_mismatches", "inconsistent_offer_payloads", "positive_hard_flag_true")
    errors = [f"{name}: {field}={stats[field]}" for name, stats in result["splits"].items() for field in integrity_fields if stats[field]]
    if result["cross_split_conflicting_label_pairs"]:
        errors.append("Conflicting labels for canonical pairs across splits")
    if result["inconsistent_offer_payloads_across_splits"]:
        errors.append("Inconsistent payloads for repeated offer IDs across splits")
    result["integrity_errors"] = errors
    warnings = []
    if any(result["offer_id_overlaps"].values()):
        warnings.append("Offer IDs overlap between splits; inspect the official protocol before modeling.")
    if any(result["canonical_pair_overlaps"].values()):
        warnings.append("The same unordered offer pair occurs in different splits.")
    if any(result["text_signature_overlaps"].values()):
        warnings.append("Normalized title/description/brand signatures overlap; identical text alone is not proof of leakage.")
    if any(result["cluster_id_overlaps"].values()):
        warnings.append("Product clusters overlap; this is expected for the official seen-model train/validation protocol.")
    result["warnings"] = warnings
    result["warning"] = " ".join(warnings) if warnings else "No offer ID, pair, cluster, or normalized text-signature overlap detected."
    result["signature_definition"] = "SHA256 of JSON [title, description, brand] after Unicode NFKC, casefold, whitespace collapse; all-missing excluded. Titles also checked separately."
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_archive(archive: str | Path, output_dir: str | Path = "docs", extract_dir: str | Path | None = None) -> dict:
    """Validate all members, audit small/unseen100, optionally extract three files.

    Only fixed basenames are extracted, never with extractall. Different existing
    files are never overwritten. The original ZIP remains untouched.
    """
    archive, output_dir = Path(archive), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"archive": archive.as_posix(), "size_bytes": archive.stat().st_size,
                "sha256": _sha256(archive), "members": []}
    records = {}
    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate member names")
        for info in zipped.infolist():
            member = {"name": info.filename, "size_bytes": info.file_size,
                      "compressed_size_bytes": info.compress_size, "crc32": f"{info.CRC:08x}"}
            if not info.is_dir() and info.filename.endswith(".json.gz"):
                with zipped.open(info) as compressed:
                    digest = hashlib.file_digest(compressed, "sha256").hexdigest()
                with zipped.open(info) as compressed, gzip.open(compressed, "rt", encoding="utf-8") as lines:
                    rows = _read_lines(lines, info.filename)
                member.update({"sha256": digest, "summary": summary(rows)})
                if info.filename in SELECTED_MEMBERS.values() or info.filename.endswith("un_gs.json.gz"):
                    records[info.filename] = rows
            manifest["members"].append(member)
        absent = set(SELECTED_MEMBERS.values()) - records.keys()
        if absent:
            raise ValueError(f"Selected official archive members missing: {sorted(absent)}")
        if extract_dir is not None:
            target_dir = Path(extract_dir).resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            for name in SELECTED_MEMBERS.values():
                destination = target_dir / name
                if destination.resolve().parent != target_dir or destination.is_symlink():
                    raise ValueError(f"Unsafe extraction target: {destination}")
                expected = next(member["sha256"] for member in manifest["members"] if member["name"] == name)
                if destination.exists():
                    if _sha256(destination) != expected:
                        raise ValueError(f"Refusing to overwrite different existing data: {destination}")
                    continue
                with zipped.open(name) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
                if _sha256(destination) != expected:
                    raise ValueError(f"Extracted file checksum mismatch: {destination}")
    result = audit(*(records[SELECTED_MEMBERS[name]] for name in ("train", "valid", "test")))
    variants = {}
    for name, rows in records.items():
        if not name.endswith("un_gs.json.gz"):
            continue
        alternative = audit(records[SELECTED_MEMBERS["train"]], records[SELECTED_MEMBERS["valid"]], rows)
        variants[name] = {key: value for key, value in alternative.items() if key.endswith("_overlaps") or key == "integrity_errors"}
        variants[name]["overlapping_offer_ids"] = {
            f"{split}_test": sorted(offers(records[SELECTED_MEMBERS[split]]) & offers(rows))
            for split in ("train", "valid")}
    manifest["alternative_gold_standard_overlaps_with_small_splits"] = variants
    config = {"dataset": "WDC Products", "archive": manifest["archive"], "archive_sha256": manifest["sha256"],
              "corner_case_cluster_percentage": 80, "random_cluster_percentage": 20,
              "training_size": "small", "train_validation_unseen_percentage": 0,
              "test_unseen_percentage": 100, "members": SELECTED_MEMBERS,
              "selection_reason": "Official paired small train/validation and 100%-unseen gold standard test transfer to clusters absent from train/validation. The 000un training names are the only training protocol in this archive.",
              "split_policy": "Use official splits verbatim; fit representations/weights on train, choose parameters on valid, evaluate once on test.",
              "validation_limitation": "Validation is seen-model; the separate optional unseen-validation archive is not included in 80pair.zip. Account for the validation/test distribution shift.",
              "source": "https://webdatacommons.org/largescaleproductcorpus/wdc-products/",
              "source_checked_on": "2026-11-01",
              "checksum_note": "Checksums identify the supplied local artifacts; upstream authenticity was not independently certified by a published checksum.",
              "scope": "Data/schema audit only; no matching scores or model-selection results generated."}
    result["configuration"] = config
    result["archive_members_validated"] = sum("summary" in member for member in manifest["members"])
    for name, payload in (("archive_manifest.json", manifest), ("wdc_config.json", config), ("data_audit.json", result)):
        (output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="Audit archive; save manifest/config/audit in --output-dir")
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument("--extract-dir", help="Extract only the three selected gzip files")
    for split in ("train", "valid", "test"):
        parser.add_argument(f"--{split}")
    parser.add_argument("--out", default="data_audit.json")
    args = parser.parse_args()
    if args.archive:
        if any((args.train, args.valid, args.test)):
            parser.error("--archive cannot be combined with explicit splits")
        result = audit_archive(args.archive, args.output_dir, args.extract_dir)
    else:
        if not all((args.train, args.valid, args.test)):
            parser.error("Provide --archive or all of --train, --valid, --test")
        result = audit(read_pairs(args.train), read_pairs(args.valid), read_pairs(args.test))
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": {name: stats["pairs"] for name, stats in result["splits"].items()},
                      "integrity_errors": result["integrity_errors"],
                      "offer_id_overlaps": result["offer_id_overlaps"],
                      "cluster_id_overlaps": result["cluster_id_overlaps"]}, indent=2))


if __name__ == "__main__":
    main()
