"""Prepare and audit official unseen WDC validation without inspecting test labels.

The test file is streamed through a metadata projection: only offer IDs, cluster
IDs and text fingerprints are retained. No test labels, class counts or matching
scores are produced. Training and validation use the strict audited WDC reader.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Iterable, Mapping
import unicodedata
from urllib.request import urlopen
from zipfile import ZipFile

from audit_wdc import SIDES, _identifier, _read_lines, is_missing, read_pairs, summary

SOURCE_PAGE = "https://webdatacommons.org/largescaleproductcorpus/wdc-products/"
SOURCE_URL = "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/val_pair.zip"
VALIDATION_BASENAME = "wdcproducts80cc20rnd100un_valid_small.json.gz"
NESTED_ARCHIVE = "80pair_addvalid.zip"
METADATA_KEYS = frozenset(f"{name}_{side}" for name in ("id", "cluster_id", "title", "description", "brand") for side in SIDES)


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_archive(destination: str | Path, url: str = SOURCE_URL) -> None:
    """Download only when absent; never replace existing archives or partial files."""
    destination = Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with partial.open("xb") as target, urlopen(url, timeout=60) as response:
        shutil.copyfileobj(response, target)
    # rename on Windows refuses an existing destination; explicit check is portable.
    if destination.exists():
        raise FileExistsError(f"Archive appeared during download: {destination}")
    partial.rename(destination)


def _normalized(value: object) -> str:
    return "" if is_missing(value) else " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _fingerprint(record: Mapping, side: str, fields: tuple[str, ...]) -> str | None:
    values = [_normalized(record.get(f"{name}_{side}")) for name in fields]
    if not any(values):
        return None
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass
class MetadataIndex:
    """An ID/fingerprint inventory with no labels and no original attribute values."""

    rows: int = 0
    offer_ids: set[str] = field(default_factory=set)
    cluster_ids: set[str] = field(default_factory=set)
    canonical_pairs: set[tuple[str, str]] = field(default_factory=set)
    text_signatures: set[str] = field(default_factory=set)
    title_signatures: set[str] = field(default_factory=set)

    def counts(self) -> dict:
        return {"pairs": self.rows, "unique_offers": len(self.offer_ids),
                "unique_clusters": len(self.cluster_ids),
                "duplicate_canonical_pairs": self.rows - len(self.canonical_pairs)}


def metadata_index(records: Iterable[Mapping]) -> MetadataIndex:
    """Use metadata keys only, also safe for records whose label cannot be read."""
    result = MetadataIndex()
    for row_number, record in enumerate(records, 1):
        ids = []
        for side in SIDES:
            location = f"metadata row {row_number}"
            identifier = _identifier(record.get(f"id_{side}"), f"id_{side}", location)
            ids.append(identifier)
            result.offer_ids.add(identifier)
            cluster = record.get(f"cluster_id_{side}")
            if not is_missing(cluster):
                result.cluster_ids.add(_identifier(cluster, f"cluster_id_{side}", location))
            for fields, target in ((("title", "description", "brand"), result.text_signatures),
                                   (("title",), result.title_signatures)):
                signature = _fingerprint(record, side, fields)
                if signature:
                    target.add(signature)
        result.canonical_pairs.add(tuple(sorted(ids)))
        result.rows += 1
    if not result.rows:
        raise ValueError("Empty metadata split")
    return result


def read_metadata(path: str | Path) -> MetadataIndex:
    """Stream label-blind metadata; decoder discards all non-whitelisted fields.

    JSON syntax is necessarily decoded, but label values are never returned to
    application code, validated, counted, retained, compared or used for scoring.
    """
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open

    def project(items):
        return {key: value for key, value in items if key in METADATA_KEYS}

    def records(stream):
        for row_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line, object_pairs_hook=project)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{row_number}: expected a JSON object")
            yield record

    with opener(path, "rt", encoding="utf-8") as stream:
        return metadata_index(records(stream))


def overlap(left: MetadataIndex, right: MetadataIndex) -> dict:
    return {name: len(getattr(left, name) & getattr(right, name)) for name in
            ("offer_ids", "cluster_ids", "canonical_pairs", "text_signatures", "title_signatures")}


def prepare_validation(archive: str | Path, train: str | Path, seen_validation: str | Path,
                       test_metadata: str | Path, output_dir: str | Path = "docs",
                       extract_dir: str | Path = "data/external") -> dict:
    """Audit the selected member, compare metadata, extract without overwriting."""
    archive, output_dir, extract_dir = Path(archive), Path(output_dir), Path(extract_dir)
    manifest = {"path": archive.as_posix(), "size_bytes": archive.stat().st_size,
                "sha256": sha256(archive), "source_url": SOURCE_URL, "members": []}
    with ExitStack() as stack:
        outer = stack.enter_context(ZipFile(archive))
        outer_names = outer.namelist()
        if len(outer_names) != len(set(outer_names)):
            raise ValueError("Archive contains duplicate member names")
        if NESTED_ARCHIVE in outer_names:
            manifest["nested_archive"] = NESTED_ARCHIVE
            for info in outer.infolist():
                entry = {"name": info.filename, "size_bytes": info.file_size,
                         "compressed_size_bytes": info.compress_size, "crc32": f"{info.CRC:08x}"}
                if not info.is_dir():
                    with outer.open(info) as stream:
                        entry["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                manifest["members"].append(entry)
            zipped = stack.enter_context(ZipFile(BytesIO(outer.read(NESTED_ARCHIVE))))
            nested_manifest = []
            manifest["selected_nested_members"] = nested_manifest
        else:
            zipped = outer
            nested_manifest = manifest["members"]
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate member names")
        candidates = [name for name in names if PurePosixPath(name).name == VALIDATION_BASENAME]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one {VALIDATION_BASENAME}, found {candidates}")
        selected = candidates[0]
        for info in zipped.infolist():
            entry = {"name": info.filename, "size_bytes": info.file_size,
                     "compressed_size_bytes": info.compress_size, "crc32": f"{info.CRC:08x}"}
            if not info.is_dir():
                with zipped.open(info) as stream:
                    entry["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            nested_manifest.append(entry)
        expected = next(item["sha256"] for item in nested_manifest if item["name"] == selected)
        with zipped.open(selected) as compressed, gzip.open(compressed, "rt", encoding="utf-8") as stream:
            validation_rows = _read_lines(stream, selected)
        small_candidates = {}
        for name in names:
            if name.endswith("un_valid_small.json.gz"):
                with zipped.open(name) as compressed:
                    uncompressed = gzip.decompress(compressed.read())
                small_candidates[name] = {
                    "uncompressed_sha256": hashlib.sha256(uncompressed).hexdigest(),
                    "rows": _read_lines(uncompressed.decode("utf-8").splitlines(), name)}
        train_rows, seen_rows = read_pairs(train), read_pairs(seen_validation)
        indices = {"train": metadata_index(train_rows), "seen_validation": metadata_index(seen_rows),
                   "unseen_validation": metadata_index(validation_rows), "test_metadata_only": read_metadata(test_metadata)}
        overlap_pairs = (("train", "unseen_validation"), ("seen_validation", "unseen_validation"),
                         ("unseen_validation", "test_metadata_only"), ("train", "test_metadata_only"),
                         ("train", "seen_validation"), ("seen_validation", "test_metadata_only"))
        overlaps = {f"{left}__{right}": overlap(indices[left], indices[right]) for left, right in overlap_pairs}
        failures = [name for name in ("train__unseen_validation", "unseen_validation__test_metadata_only")
                    if any(overlaps[name][field] for field in ("offer_ids", "cluster_ids", "canonical_pairs"))]
        if failures and any(overlaps["train__seen_validation"][name] for name in ("offer_ids", "canonical_pairs")):
            raise ValueError("Candidate rejected and fallback seen validation overlaps training offers/pairs")
        if failures and any(overlaps["seen_validation__test_metadata_only"][name] for name in ("offer_ids", "cluster_ids", "canonical_pairs")):
            raise ValueError("Candidate rejected and fallback validation overlaps reserved test metadata")
        candidate_audit = {}
        for name, candidate in small_candidates.items():
            candidate_index = metadata_index(candidate["rows"])
            candidate_overlaps = {split: overlap(index, candidate_index) for split, index in indices.items()
                                  if split != "unseen_validation"}
            candidate_audit[name] = {"summary": summary(candidate["rows"]),
                                     "uncompressed_sha256": candidate["uncompressed_sha256"],
                                     "overlaps": candidate_overlaps,
                                     "empirical_unseen_cluster_fraction_vs_train": len(candidate_index.cluster_ids - indices["train"].cluster_ids) / len(candidate_index.cluster_ids),
                                     "eligible_for_100_percent_unseen_validation": all(
                                         not candidate_overlaps[split][field]
                                         for split in ("train", "test_metadata_only")
                                         for field in ("offer_ids", "cluster_ids", "canonical_pairs"))}
        extract_dir = extract_dir.resolve()
        extract_dir.mkdir(parents=True, exist_ok=True)
        destination = extract_dir / VALIDATION_BASENAME
        if destination.is_symlink() or destination.resolve().parent != extract_dir:
            raise ValueError(f"Unsafe extraction target: {destination}")
        if destination.exists():
            if sha256(destination) != expected:
                raise ValueError(f"Refusing to overwrite different existing data: {destination}")
        else:
            with zipped.open(selected) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
        if sha256(destination) != expected:
            raise ValueError("Extracted validation checksum mismatch")
    # Keep paths relative when possible for portable experiment configuration.
    try:
        destination_display = destination.relative_to(Path.cwd()).as_posix()
    except ValueError:
        destination_display = destination.as_posix()
    train_candidate = overlaps["train__unseen_validation"]
    unseen_fraction = len(indices["unseen_validation"].cluster_ids - indices["train"].cluster_ids) / len(indices["unseen_validation"].cluster_ids)
    fallback_reason = (f"Official additional 80cc100un/small validation rejected: {train_candidate['offer_ids']} training offer IDs, "
                       f"{train_candidate['canonical_pairs']} training pairs, and {train_candidate['cluster_ids']} training clusters overlap; "
                       f"empirical unseen-cluster fraction is {unseen_fraction:.6f}; failed checks: {failures}. "
                       "Official original seen validation retained verbatim; no rows filtered.")
    selected_path = Path(seen_validation).as_posix() if failures else destination_display
    config = {"dataset": "WDC Products", "source_page": SOURCE_PAGE, "source_url": SOURCE_URL,
              "source_checked_on": "2026-11-01", "corner_case_cluster_percentage": 80,
              "training_size": "small", "validation_unseen_percentage": 0 if failures else 100,
              "mode": "fallback_seen" if failures else "official_unseen",
              "reason": fallback_reason if failures else "Official unseen validation is ID/cluster/pair disjoint from training and reserved test.",
              "archive": archive.as_posix(), "archive_sha256": manifest["sha256"],
              "nested_archive": manifest.get("nested_archive"),
              "member": Path(seen_validation).name if failures else selected, "candidate_member": selected,
              "candidate_validation_path": destination_display, "candidate_validation_sha256": expected,
              "candidate_rejected": bool(failures), "candidate_rejection_checks": failures,
              "validation_path": selected_path, "validation_sha256": sha256(selected_path),
              "train_path": Path(train).as_posix(), "seen_validation_path": Path(seen_validation).as_posix(),
              "reserved_test_metadata_path": Path(test_metadata).as_posix(),
              "policy": "Fit representations/informativeness on train only; choose thresholds/hyperparameters on validation. No test-label access or final evaluation.",
              "selection_reason": fallback_reason if failures else "Official additional small/80cc/100un validation matches the reserved 100%-unseen test regime; ID and product-cluster disjointness verified.",
              "validation_limitation": "Seen-model validation and reserved unseen-model test differ in product-cluster regime; validation results cannot be treated as estimates of unseen-test performance." if failures else None,
              "checksum_note": "Local SHA256 identifies the downloaded artifact; no independently published upstream checksum was found."}
    result = {"configuration": config, "archive_manifest": manifest,
              "splits": {"train": summary(train_rows), "seen_validation": summary(seen_rows),
                         "unseen_validation": summary(validation_rows)},
              "test_metadata_only": {**indices["test_metadata_only"].counts(),
                                     "path": Path(test_metadata).as_posix(), "sha256": sha256(test_metadata)},
              "overlaps": overlaps,
              "additional_small_candidates": candidate_audit,
              "additional_small_candidate_content_identical": len({candidate["uncompressed_sha256"] for candidate in candidate_audit.values()}) == 1 if len(candidate_audit) > 1 else None,
              "test_label_policy": "Test JSONL streamed using an allowlist of id/cluster_id/title/description/brand; labels discarded by decoder and never read by application logic. No labels, hard-negative flags, label-derived statistics, predictions or scores retained.",
              "signature_policy": "NFKC + casefold + whitespace; SHA256 of JSON [title,description,brand], plus separate title fingerprint; all-missing excluded.",
              "scope": "Top-level and nested 80pair members manifested and hashed; both additional 80cc small validation candidates schema/labels audited. Original 000un train/seen-validation audited; reserved test metadata only. No other test set inspected.",
              "fallback": {"used": bool(failures), "reason": config["reason"],
                           "no_filtering_or_relabeling": True,
                           "original_000un_validation_path": Path(seen_validation).as_posix()}}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("validation_data_audit.json", result), ("validation_config.json", config)):
        (output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/external/valpair.zip")
    parser.add_argument("--download", action="store_true", help="Download official archive only if absent")
    parser.add_argument("--train", default="data/external/wdcproducts80cc20rnd000un_train_small.json.gz")
    parser.add_argument("--seen-validation", default="data/external/wdcproducts80cc20rnd000un_valid_small.json.gz")
    parser.add_argument("--test-metadata", default="data/external/wdcproducts80cc20rnd100un_gs.json.gz")
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument("--extract-dir", default="data/external")
    args = parser.parse_args()
    if args.download:
        download_archive(args.archive)
    result = prepare_validation(args.archive, args.train, args.seen_validation, args.test_metadata,
                                args.output_dir, args.extract_dir)
    print(json.dumps({"validation_path": result["configuration"]["validation_path"],
                      "validation_pairs": result["splits"]["unseen_validation"]["pairs"],
                      "overlaps": result["overlaps"]}, indent=2))


if __name__ == "__main__":
    main()
