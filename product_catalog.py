"""Label-free product catalog imports for the local search prototype.

The WDC pair file is only a source of observed offers: pair annotations are
discarded before constructing an offer. FPV fields remain their original
strings, including missing values and leading-zero technical codes. Neither
importer fills gaps from another occurrence or another shop.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from fpv_dataset import OFFER_COLUMNS, read_csv_records


FPV_TECHNICAL_FIELDS = (
    "family_model", "stator_code", "kv_rpm_per_volt", "voltage_s",
    "units_in_listing", "seller_sku",
)
_SCALAR_FIELDS = ("offer_id", "title", "description", "brand", "source", "url")


def _annotation_field(name: str) -> bool:
    """Exclude annotations, including extra source columns, from offer data.

    Original ``review_notes`` are retained for provenance/display only; the
    pair projection below never uses them as matching evidence.
    """
    lowered = name.casefold()
    if lowered == "review_notes":
        return False
    return (
        "label" in lowered
        or lowered.startswith(("entity_id", "cluster_id", "pair_id", "review_",
                               "reviewer", "reviewed_", "verification_"))
        or lowered in {"is_hard_negative", "gold", "gold_id", "evidence_url"}
    )


@dataclass(frozen=True)
class ProductOffer:
    """One observed offer; scores and confirmed model identities are separate.

    ``source`` is the actual store/source identifier when explicitly supplied.
    An empty source means unknown, rather than a store inferred from an ID.
    Attribute mappings hold display metadata; only ``offer_to_pair`` defines
    the input projection into the existing four-feature matching pipeline.
    """

    offer_id: str
    title: str = ""
    description: str = ""
    brand: str = ""
    source: str = ""
    url: str = ""
    attributes: dict[str, str] = field(default_factory=dict)
    technical_attributes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _SCALAR_FIELDS:
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"ProductOffer.{name} must be a string")
        if not self.offer_id or self.offer_id != self.offer_id.strip():
            raise ValueError("ProductOffer.offer_id must be nonempty and unpadded")
        for name in ("attributes", "technical_attributes"):
            values = getattr(self, name)
            if not isinstance(values, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in values.items()
            ):
                raise ValueError(f"ProductOffer.{name} must map strings to strings")
            if any(_annotation_field(key) for key in values):
                raise ValueError(f"ProductOffer.{name} cannot contain matching annotations")
            # An independent copy avoids changes via the importer's temporary row.
            object.__setattr__(self, name, values.copy())


def _text(value: object, name: str, location: str, *, numeric: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if numeric and type(value) in (int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{location}: nonfinite {name}")
        return str(value)
    raise ValueError(f"{location}: {name} must be a string or null")


def _offer_id(value: object, name: str, location: str) -> str:
    if type(value) not in (str, int):
        raise ValueError(f"{location}: invalid {name}")
    result = str(value)
    if not result or result != result.strip():
        raise ValueError(f"{location}: blank or padded {name}")
    return result


def _add_offer(catalog: dict[str, ProductOffer], offer: ProductOffer, location: str) -> None:
    previous = catalog.get(offer.offer_id)
    if previous is not None and previous != offer:
        raise ValueError(
            f"{location}: conflicting repeated offer_id {offer.offer_id!r}; "
            "source values must be resolved explicitly"
        )
    catalog[offer.offer_id] = offer


def _explicit_field(record: dict, side: str, names: tuple[str, ...], location: str) -> str:
    """Choose an explicitly provided source value without guessing a host/store."""
    values = [_text(record.get(f"{name}_{side}"), f"{name}_{side}", location)
              for name in names]
    nonempty = [value for value in values if value.strip()]
    # Multiple synonyms may be present, but conflicting identities cannot be
    # silently reconciled into an apparently reliable cross-store filter.
    if len(set(nonempty)) > 1:
        raise ValueError(f"{location}: conflicting explicit {names} for {side}")
    return nonempty[0] if nonempty else ""


def _wdc_offer(record: dict, side: str, location: str) -> ProductOffer:
    if f"id_{side}" not in record or f"title_{side}" not in record:
        raise ValueError(f"{location}: missing id_{side} or title_{side}")
    attributes = {}
    for name in ("price", "priceCurrency"):
        if f"{name}_{side}" in record:
            attributes[name] = _text(record[f"{name}_{side}"], f"{name}_{side}",
                                     location, numeric=name == "price")
    return ProductOffer(
        offer_id=_offer_id(record[f"id_{side}"], f"id_{side}", location),
        title=_text(record[f"title_{side}"], f"title_{side}", location),
        description=_text(record.get(f"description_{side}"),
                          f"description_{side}", location),
        brand=_text(record.get(f"brand_{side}"), f"brand_{side}", location),
        source=_explicit_field(record, side, ("store", "source", "host"), location),
        url=_explicit_field(record, side, ("url", "source_url"), location),
        attributes=attributes,
    )


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def load_wdc_catalog(path: str | Path) -> dict[str, ProductOffer]:
    """Project offers from plain/gzip WDC JSONL without using pair annotations.

    The CLI restricts the real-data demonstration to the audited TRAIN file.
    This generic reader also accepts small import-test fixtures. It deliberately
    does not require, validate, or expose ``label``, cluster IDs, or hard-negative
    flags: changing annotations cannot change retrieval inputs.
    """
    path = Path(path)
    opener = gzip.open if path.suffix.casefold() == ".gz" else open
    catalog: dict[str, ProductOffer] = {}
    try:
        with opener(path, "rt", encoding="utf-8-sig") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                location = f"{path}:{number}"
                try:
                    record = json.loads(line, parse_constant=_reject_nonfinite)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{location}: invalid JSON: {error}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{location}: expected a JSON object")
                for side in ("left", "right"):
                    _add_offer(catalog, _wdc_offer(record, side, location), location)
    except (UnicodeError, EOFError) as error:
        raise ValueError(f"{path}: invalid UTF-8/gzip JSONL: {error}") from error
    if not catalog:
        raise ValueError(f"{path}: empty WDC offer catalog")
    return catalog


def load_fpv_catalog(path: str | Path) -> dict[str, ProductOffer]:
    """Import only the FPV offers CSV; candidate/review CSVs are not inputs."""
    path = Path(path)
    rows = read_csv_records(path, OFFER_COLUMNS)
    if not rows:
        raise ValueError(f"{path}: empty FPV offer catalog")
    catalog: dict[str, ProductOffer] = {}
    mapped = {"offer_id", "raw_title", "raw_description", "brand", "store", "source_url"}
    for number, row in enumerate(rows, 2):
        location = f"{path}:{number}"
        offer_id = _offer_id(row["offer_id"], "offer_id", location)
        # FPV source IDs must be unique even if two rows happen to be identical.
        if offer_id in catalog:
            raise ValueError(f"{location}: duplicate FPV offer_id {offer_id!r}")
        attributes = {name: value for name, value in row.items()
                      if name not in mapped and name not in FPV_TECHNICAL_FIELDS
                      and not _annotation_field(name)}
        catalog[offer_id] = ProductOffer(
            offer_id=offer_id, title=row["raw_title"],
            description=row["raw_description"], brand=row["brand"],
            source=row["store"], url=row["source_url"], attributes=attributes,
            technical_attributes={name: row[name] for name in FPV_TECHNICAL_FIELDS},
        )
    return catalog


def offer_to_pair(left: ProductOffer, right: ProductOffer) -> dict[str, str]:
    """Exact input projection for existing methods; technical fields are display-only.

    No labels, model groups, URLs, review notes, or seller SKUs are concatenated
    into title/description. All four existing similarities keep their meaning.
    """
    result = {}
    for side, offer in (("left", left), ("right", right)):
        result[f"id_{side}"] = offer.offer_id
        for name in ("title", "description", "brand"):
            result[f"{name}_{side}"] = getattr(offer, name)
    return result
