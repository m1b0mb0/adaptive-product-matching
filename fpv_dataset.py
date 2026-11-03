"""Strict, lossless FPV CSV import without altering the WDC matching pipeline.

Candidate labels are suggestions. ``matching_pairs`` deliberately has no
``label`` key. Only the separate, fail-closed manual-review importer can supply
evaluation labels; it never fits models or computes performance metrics.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit


OFFER_COLUMNS = (
    "offer_id", "store", "raw_title", "brand", "family_model", "stator_code",
    "kv_rpm_per_volt", "voltage_s", "units_in_listing", "seller_sku", "price_uah",
    "raw_description", "source_url", "collected_on", "record_status",
    "review_notes", "entity_id",
)
CANDIDATE_COLUMNS = (
    "pair_id", "offer_a", "offer_b", "proposed_label_1_match_0_nonmatch",
    "verification_status", "rationale",
)
REVIEW_COLUMNS = (
    "review_status", "confirmed_label_1_match_0_nonmatch", "reviewer",
    "reviewed_on", "evidence_url", "review_notes", "offers_sha256",
    "candidates_sha256",
)
# These describe an individual variant, not its independently verified identity.
# Unknown or ambiguous source statuses cannot be silently promoted to this set.
EVALUABLE_RECORD_STATUSES = frozenset({"single_variant", "listing_verified"})


def _read_csv(path: str | Path, required_columns=()) -> tuple[list[dict[str, str]], str]:
    path = Path(path)
    payload = path.read_bytes()
    try:
        source = io.StringIO(payload.decode("utf-8-sig"), newline="")
        reader = csv.reader(source, strict=True)
        headers = next(reader, None)
        if not headers or any(not name or name != name.strip() for name in headers):
            raise ValueError(f"{path}: missing, blank, or padded CSV header")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path}: duplicate CSV column names")
        missing = set(required_columns) - set(headers)
        if missing:
            raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
        rows = []
        for values in reader:
            if len(values) != len(headers):
                raise ValueError(
                    f"{path}:{reader.line_num}: expected {len(headers)} values, "
                    f"found {len(values)}"
                )
            rows.append(dict(zip(headers, values)))
    except (UnicodeError, csv.Error) as error:
        raise ValueError(f"{path}: invalid UTF-8 CSV: {error}") from error
    return rows, hashlib.sha256(payload).hexdigest()


def read_csv_records(path: str | Path, required_columns=()) -> list[dict[str, str]]:
    """Read UTF-8/BOM CSV as original strings; reject malformed shapes/headers.

    Empty strings, leading-zero codes, quotes, Unicode, and unknown columns are
    preserved. Missing values are never converted to floating-point NaN.
    """
    return _read_csv(path, required_columns)[0]


def _identifier(value: str, field: str, location: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{location}: blank or padded {field}")
    return value


def _binary(value: str, location: str) -> int:
    if value not in ("0", "1"):
        raise ValueError(f"{location}: binary label must be exactly '0' or '1'")
    return int(value)


def _positive_integer(value: str) -> int | None:
    stripped = value.strip()
    if not re.fullmatch(r"[0-9]+", stripped):
        return None
    result = int(stripped)
    return result if result > 0 else None


def technical_view(offer: dict[str, str]) -> dict:
    """Typed views from this record's explicit fields, with raw data untouched.

    In particular, KV is not extracted from titles or copied from another store.
    Stator codes (``0802``, ``2207.5``, ``22.6-6.6``) remain strings; a numerical
    reinterpretation of these different notations would require domain review.
    """
    issues = []
    result = {}
    for field in ("kv_rpm_per_volt", "units_in_listing"):
        raw = offer.get(field, "")
        result[field] = _positive_integer(raw) if raw.strip() else None
        if raw.strip() and result[field] is None:
            issues.append(f"unresolved_{field}")
    for field in ("stator_code", "voltage_s"):
        result[field] = offer.get(field, "") or None
    status = offer.get("record_status", "")
    if status not in EVALUABLE_RECORD_STATUSES:
        issues.append("unresolved_record_status")
    result["issues"] = issues
    return result


@dataclass(frozen=True)
class FpvDataset:
    offers: dict[str, dict[str, str]]
    candidates: list[dict[str, str]]
    source_hashes: dict[str, str]

    def _matching_pair(self, candidate: dict[str, str]) -> dict:
        result = {
            "pair_id": candidate["pair_id"],
            "proposed_label_1_match_0_nonmatch": candidate[
                "proposed_label_1_match_0_nonmatch"
            ],
            "verification_status": candidate["verification_status"],
            "review_status": "pending",
            "fpv_candidate": candidate.copy(),
            "fpv_technical_views": {},
            **self.source_hashes,
        }
        for source_side, side in (("a", "left"), ("b", "right")):
            offer = self.offers[candidate[f"offer_{source_side}"]]
            result[f"id_{side}"] = offer["offer_id"]
            result[f"title_{side}"] = offer["raw_title"]
            result[f"description_{side}"] = offer["raw_description"]
            result[f"brand_{side}"] = offer["brand"]
            result.update({f"fpv_{field}_{side}": value for field, value in offer.items()})
            # This container has no side suffix, so arbitrary raw CSV fields
            # such as an extra ``technical`` column cannot overwrite it.
            result["fpv_technical_views"][side] = technical_view(offer)
        return result

    def matching_pairs(self) -> list[dict]:
        """Unlabelled candidate pairs compatible with FeatureExtractor.

        No ``cluster_id`` is inferred from family/model, seller SKU, or entity_id.
        Technical fields are kept separately, not concatenated into text inputs.
        """
        return [self._matching_pair(candidate) for candidate in self.candidates]

    def confirmed_pairs(self, review_path: str | Path) -> list[dict]:
        """Import a confirmed-only export of review_queue.csv for future use.

        A mixed queue fails closed: export only completed rows before calling.
        Evidence metadata is a human attestation, not automatic URL verification.
        Ambiguous source records must first be resolved against their own source,
        then the queue regenerated and reviewed with its new content hashes.
        """
        required = (*CANDIDATE_COLUMNS, *REVIEW_COLUMNS,
                    *(f"{field}_{side}" for side in ("a", "b") for field in OFFER_COLUMNS))
        reviewed = read_csv_records(review_path, required)
        if not reviewed:
            raise ValueError("No confirmed review rows supplied")
        candidates = {row["pair_id"]: row for row in self.candidates}
        seen = set()
        result = []
        for row in reviewed:
            pair_id = row["pair_id"]
            location = f"Review pair {pair_id!r}"
            if pair_id in seen or pair_id not in candidates:
                raise ValueError(f"{location}: duplicate or unknown pair_id")
            seen.add(pair_id)
            candidate = candidates[pair_id]
            for field, value in self.source_hashes.items():
                if row[field] != value:
                    raise ValueError(f"{location}: stale or mismatched {field}")
            for field in CANDIDATE_COLUMNS:
                if row[field] != candidate[field]:
                    raise ValueError(f"{location}: original candidate {field} was changed")
            if row["review_status"] != "confirmed":
                raise ValueError(f"{location}: review_status must be exactly 'confirmed'")
            label = _binary(row["confirmed_label_1_match_0_nonmatch"], location)
            if not row["reviewer"].strip():
                raise ValueError(f"{location}: missing reviewer")
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", row["reviewed_on"]):
                raise ValueError(f"{location}: reviewed_on must be an ISO date")
            try:
                date.fromisoformat(row["reviewed_on"])
            except ValueError as error:
                raise ValueError(f"{location}: invalid reviewed_on date") from error
            evidence = row["evidence_url"]
            parsed = urlsplit(evidence)
            if parsed.scheme not in ("https", "http") or not parsed.netloc or any(
                char.isspace() for char in evidence
            ):
                raise ValueError(f"{location}: evidence_url must be a nonempty HTTP(S) URL")
            for side in ("a", "b"):
                offer = self.offers[candidate[f"offer_{side}"]]
                for field in OFFER_COLUMNS:
                    if row[f"{field}_{side}"] != offer[field]:
                        raise ValueError(f"{location}: original offer field {field}_{side} changed")
                if technical_view(offer)["issues"]:
                    raise ValueError(f"{location}: unresolved variant/technical fields for {offer['offer_id']}")
            pair = self._matching_pair(candidate)
            pair.update({field: row[field] for field in REVIEW_COLUMNS})
            pair["label"] = label
            result.append(pair)
        return result


def load_fpv_dataset(offers_path: str | Path, pairs_path: str | Path) -> FpvDataset:
    """Load originals with strict IDs/references; no labels are confirmed here."""
    offer_rows, offers_hash = _read_csv(offers_path, OFFER_COLUMNS)
    candidates, candidates_hash = _read_csv(pairs_path, CANDIDATE_COLUMNS)
    if not offer_rows or not candidates:
        raise ValueError("FPV dataset requires nonempty offers and candidate CSV files")
    offers = {}
    for row in offer_rows:
        offer_id = _identifier(row["offer_id"], "offer_id", str(offers_path))
        if offer_id in offers:
            raise ValueError(f"Duplicate offer_id: {offer_id!r}")
        offers[offer_id] = row
    seen_ids, seen_pairs = set(), set()
    for row in candidates:
        pair_id = _identifier(row["pair_id"], "pair_id", str(pairs_path))
        if pair_id in seen_ids:
            raise ValueError(f"Duplicate pair_id: {pair_id!r}")
        seen_ids.add(pair_id)
        sides = [_identifier(row[field], field, pair_id) for field in ("offer_a", "offer_b")]
        if any(offer_id not in offers for offer_id in sides):
            raise ValueError(f"{pair_id}: unknown offer reference")
        if sides[0] == sides[1]:
            raise ValueError(f"{pair_id}: self-pair is not permitted")
        key = tuple(sorted(sides))
        if key in seen_pairs:
            raise ValueError(f"{pair_id}: duplicate unordered candidate pair")
        seen_pairs.add(key)
        _binary(row["proposed_label_1_match_0_nonmatch"], pair_id)
    return FpvDataset(offers, candidates,
                      {"offers_sha256": offers_hash, "candidates_sha256": candidates_hash})
