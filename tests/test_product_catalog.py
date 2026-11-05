"""Catalog import fixtures verify safety, not scientific matching quality."""
import csv
import gzip
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fpv_dataset import OFFER_COLUMNS
from matching import FeatureExtractor
from product_catalog import (FPV_TECHNICAL_FIELDS, ProductOffer, load_fpv_catalog,
                             load_wdc_catalog, offer_to_pair)


def wdc_pair(**changes):
    result = {
        "id_left": 1, "title_left": "Motor Example1003 14800KV",
        "description_left": None, "brand_left": "Example", "price_left": "000.10",
        "priceCurrency_left": "UAH", "id_right": 2,
        "title_right": "Example1003 motor", "description_right": "Original description",
        "brand_right": "Example", "price_right": 12.0, "priceCurrency_right": None,
        "cluster_id_left": 55, "cluster_id_right": 55, "label": 1,
        "is_hard_negative": False, "pair_id": "P1",
    }
    result.update(changes)
    return result


def write_wdc(path, rows):
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as source:
            source.write(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def fpv_offer(offer_id="A", **changes):
    result = dict.fromkeys(OFFER_COLUMNS, "")
    result.update(offer_id=offer_id, store="Магазин А", raw_title="Мотор Example 0802",
                  brand="Example", family_model="ROBO 0802", stator_code="0802",
                  kv_rpm_per_volt="23000", voltage_s="1–2S", units_in_listing="4",
                  seller_sku="00042", source_url="https://example.invalid/motor",
                  collected_on="2026-11-03", record_status="single_variant",
                  review_notes="Комплект із чотирьох; потрібна перевірка", entity_id="unconfirmed")
    result.update(changes)
    return result


def write_fpv(path, rows):
    columns = list(rows[0]) if rows else list(OFFER_COLUMNS)
    with path.open("w", encoding="utf-8-sig", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_offer_scalar_fields_are_frozen_and_mapping_inputs_copied():
    attributes = {"price": "12.00"}
    technical = {"stator_code": "0802"}
    offer = ProductOffer("A", attributes=attributes, technical_attributes=technical)
    attributes["price"] = "99"
    technical["stator_code"] = "2207"
    assert offer.attributes == {"price": "12.00"}
    assert offer.technical_attributes == {"stator_code": "0802"}
    with pytest.raises(FrozenInstanceError):
        offer.title = "Modified after indexing"


@pytest.mark.parametrize("changes", [
    {"offer_id": ""}, {"offer_id": " A"}, {"offer_id": "A "}, {"offer_id": 1},
    {"title": None}, {"description": []}, {"brand": 4}, {"source": False}, {"url": 5},
    {"attributes": []}, {"attributes": {"price": 1}},
    {"technical_attributes": {1: "0802"}}, {"attributes": {"label": "1"}},
    {"technical_attributes": {"entity_id": "gold"}},
])
def test_offer_rejects_invalid_types_ids_and_annotation_metadata(changes):
    arguments = {"offer_id": "A", **changes}
    with pytest.raises(ValueError):
        ProductOffer(**arguments)


@pytest.mark.parametrize("suffix", [".jsonl", ".json.gz"])
def test_wdc_projects_original_offers_without_annotations_or_invented_sources(tmp_path, suffix):
    catalog = load_wdc_catalog(write_wdc(tmp_path / ("train" + suffix), [wdc_pair()]))
    assert list(catalog) == ["1", "2"]
    left = catalog["1"]
    assert left.title == "Motor Example1003 14800KV"
    assert left.description == ""
    assert left.source == left.url == ""
    assert left.attributes == {"price": "000.10", "priceCurrency": "UAH"}
    assert left.technical_attributes == {}
    assert catalog["2"].attributes == {"price": "12.0", "priceCurrency": ""}


def test_wdc_annotations_are_not_required_or_consulted(tmp_path):
    original = wdc_pair()
    changed = {key: value for key, value in original.items()
               if key not in {"label", "pair_id", "cluster_id_left", "cluster_id_right",
                              "is_hard_negative"}}
    changed.update(label={"unverified": "do not interpret"}, cluster_id_left=["ignored"],
                   is_hard_negative="not a search feature", proposed_label="1")
    first = load_wdc_catalog(write_wdc(tmp_path / "first.jsonl", [original]))
    second = load_wdc_catalog(write_wdc(tmp_path / "second.jsonl", [changed]))
    no_annotations = {key: value for key, value in original.items()
                      if not key.startswith(("cluster", "is_hard", "label", "pair_id"))}
    third = load_wdc_catalog(write_wdc(tmp_path / "third.jsonl", [no_annotations]))
    assert first == second == third


def test_wdc_duplicate_offer_rows_deduplicate_and_do_not_depend_on_pair_order(tmp_path):
    first = wdc_pair()
    second = wdc_pair(id_right=3, label=0, cluster_id_right=99)
    original = load_wdc_catalog(write_wdc(tmp_path / "a.jsonl", [first, second, first]))
    reordered = load_wdc_catalog(write_wdc(tmp_path / "b.jsonl", [second, first]))
    assert len(original) == 3
    assert original == reordered


@pytest.mark.parametrize("change", [
    {"title_left": "Conflicting title"}, {"description_left": "Do not fill this gap"},
    {"brand_left": None}, {"price_left": "99.00"}, {"source_left": "Other shop"},
])
def test_wdc_conflicting_same_id_fails_without_backfilling(tmp_path, change):
    with pytest.raises(ValueError, match="conflicting repeated offer_id"):
        load_wdc_catalog(write_wdc(tmp_path / "bad.jsonl", [wdc_pair(), wdc_pair(**change)]))


def test_wdc_sources_urls_are_used_only_when_explicit_and_kept_raw(tmp_path):
    pair = wdc_pair(store_left=" Store A ", url_left="https://a.invalid/original",
                    host_right="b.invalid", source_url_right="https://b.invalid/path")
    catalog = load_wdc_catalog(write_wdc(tmp_path / "train.jsonl", [pair]))
    assert catalog["1"].source == " Store A "
    assert catalog["1"].url == "https://a.invalid/original"
    assert catalog["2"].source == "b.invalid"
    assert catalog["2"].url == "https://b.invalid/path"


@pytest.mark.parametrize("changes", [
    {"id_left": None}, {"id_left": True}, {"id_left": 1.5}, {"id_left": " 1"},
    {"title_left": ["not text"]}, {"description_right": False},
    {"brand_right": 2}, {"price_left": True}, {"priceCurrency_left": []},
    {"source_left": "A", "store_left": "B"},
    {"url_left": "https://a.invalid/", "source_url_left": "https://b.invalid/"},
])
def test_wdc_invalid_scalar_fields_and_ambiguous_explicit_sources_fail(tmp_path, changes):
    with pytest.raises(ValueError):
        load_wdc_catalog(write_wdc(tmp_path / "bad.jsonl", [wdc_pair(**changes)]))


@pytest.mark.parametrize("payload", ["", "\n  \n", "{broken", "[]", "1", "null",
                                       '{"id_left": 1, "title_left": "A"}',
                                       '{"id_left": NaN}'])
def test_wdc_empty_malformed_or_nonobject_file_fails_cleanly(tmp_path, payload):
    path = tmp_path / "bad.jsonl"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_wdc_catalog(path)


def test_wdc_null_empty_and_one_character_titles_are_preserved(tmp_path):
    catalog = load_wdc_catalog(write_wdc(tmp_path / "train.jsonl", [
        wdc_pair(title_left=None, title_right="x")]))
    assert catalog["1"].title == ""
    assert catalog["2"].title == "x"


def test_fpv_requires_only_offers_file_and_preserves_technical_originals(tmp_path):
    row = fpv_offer(raw_description='Опис, "цитата"\nДругий рядок', custom="  N/A  ")
    catalog = load_fpv_catalog(write_fpv(tmp_path / "offers.csv", [row]))
    offer = catalog["A"]
    assert offer.title == row["raw_title"]
    assert offer.description == row["raw_description"]
    assert offer.source == row["store"]
    assert offer.url == row["source_url"]
    assert offer.technical_attributes == {key: row[key] for key in FPV_TECHNICAL_FIELDS}
    assert offer.technical_attributes["stator_code"] == "0802"
    assert offer.technical_attributes["seller_sku"] == "00042"
    assert offer.attributes["custom"] == "  N/A  "
    assert offer.attributes["review_notes"] == row["review_notes"]
    assert offer.attributes["record_status"] == "single_variant"
    assert "entity_id" not in offer.attributes


def test_fpv_missing_values_remain_missing_and_no_other_shop_imputation(tmp_path):
    first = fpv_offer(raw_title="", raw_description="", kv_rpm_per_volt="",
                      voltage_s="", seller_sku="", record_status="needs_variant_split")
    second = fpv_offer("B", store="Shop B", kv_rpm_per_volt="23000")
    catalog = load_fpv_catalog(write_fpv(tmp_path / "offers.csv", [first, second]))
    assert catalog["A"].title == catalog["A"].description == ""
    assert catalog["A"].technical_attributes["kv_rpm_per_volt"] == ""
    assert catalog["A"].technical_attributes["seller_sku"] == ""
    assert catalog["B"].technical_attributes["kv_rpm_per_volt"] == "23000"


def test_fpv_label_and_annotation_extra_columns_are_discarded(tmp_path):
    row = fpv_offer(proposed_label_1_match_0_nonmatch="1", confirmed_label="0",
                    verification_status="pending", review_status="confirmed",
                    cluster_id="invented", gold="1", evidence_url="https://e.invalid/")
    offer = load_fpv_catalog(write_fpv(tmp_path / "offers.csv", [row]))["A"]
    assert set(offer.attributes) == {"price_uah", "collected_on", "record_status", "review_notes"}
    assert not any("label" in key for key in offer_to_pair(offer, offer))


@pytest.mark.parametrize("rows", [[fpv_offer(), fpv_offer()],
                                  [fpv_offer(), fpv_offer(raw_title="Different")],
                                  [fpv_offer(" ")]])
def test_fpv_duplicate_or_invalid_offer_ids_fail(tmp_path, rows):
    with pytest.raises(ValueError):
        load_fpv_catalog(write_fpv(tmp_path / "offers.csv", rows))


@pytest.mark.parametrize("payload", ["", "offer_id,raw_title\nA,Title\n",
                                       "offer_id,offer_id\nA,A\n"])
def test_fpv_empty_or_incomplete_csv_fails(tmp_path, payload):
    path = tmp_path / "offers.csv"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_fpv_catalog(path)


def test_fpv_header_only_catalog_fails(tmp_path):
    with pytest.raises(ValueError, match="empty FPV"):
        load_fpv_catalog(write_fpv(tmp_path / "offers.csv", []))


def test_pair_projection_uses_only_existing_evidence_channels_and_is_compatible():
    left = ProductOffer("A", title="Example1003 motor", brand="Example",
                        attributes={"review_notes": "Unverified"},
                        technical_attributes={"stator_code": "0802", "seller_sku": "ABC99"})
    right = ProductOffer("B", title="Example1003 motor", brand="Other",
                         source="Shop B", url="https://b.invalid/")
    pair = offer_to_pair(left, right)
    assert pair == {"id_left": "A", "title_left": "Example1003 motor",
                    "description_left": "", "brand_left": "Example",
                    "id_right": "B", "title_right": "Example1003 motor",
                    "description_right": "", "brand_right": "Other"}
    features = FeatureExtractor().fit([pair]).transform([pair]).matrix
    assert features.shape == (1, 4)
    assert np.isclose(features[0, 0], 1)
    assert np.isnan(features[0, 1])
    assert features[0, 2] == 0
    assert features[0, 3] == 1
