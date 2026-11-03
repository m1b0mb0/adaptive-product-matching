"""Import safety tests use temporary fixtures, never fabricated research results."""
import csv

import numpy as np
import pytest

from fpv_dataset import (CANDIDATE_COLUMNS, OFFER_COLUMNS, REVIEW_COLUMNS,
                         load_fpv_dataset, read_csv_records, technical_view)
from matching import FeatureExtractor


def write_csv(path, rows, columns=None, encoding="utf-8-sig"):
    columns = columns or list(rows[0])
    with path.open("w", encoding=encoding, newline="") as source:
        writer = csv.DictWriter(source, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def offer(offer_id, **changes):
    row = dict.fromkeys(OFFER_COLUMNS, "")
    row.update(offer_id=offer_id, store="Магазин", raw_title="Мотор ROBO 0802 1300KV",
               brand="Example", family_model="ROBO", stator_code="0802",
               kv_rpm_per_volt="1300", units_in_listing="1", record_status="single_variant",
               seller_sku="00023", source_url=f"https://example.com/{offer_id}")
    row.update(changes)
    return row


def candidate(pair_id="P1", a="A", b="B", **changes):
    row = dict(pair_id=pair_id, offer_a=a, offer_b=b,
               proposed_label_1_match_0_nonmatch="1",
               verification_status="needs_manual_confirmation", rationale="Перевірити")
    row.update(changes)
    return row


def dataset(tmp_path, offers=None, pairs=None):
    offers = offers if offers is not None else [offer("A"), offer("B", store="Інший магазин")]
    pairs = pairs if pairs is not None else [candidate()]
    return load_fpv_dataset(write_csv(tmp_path / "offers.csv", offers),
                            write_csv(tmp_path / "pairs.csv", pairs))


def reviewed_row(data, pair=None, **changes):
    pair = pair if pair is not None else data.candidates[0]
    row = {**pair, **dict.fromkeys(REVIEW_COLUMNS, ""), **data.source_hashes}
    row.update(review_status="confirmed", confirmed_label_1_match_0_nonmatch="0",
               reviewer="Рецензент", reviewed_on="2026-11-03",
               evidence_url="https://manufacturer.example/variants")
    for side in ("a", "b"):
        row.update({f"{field}_{side}": value
                    for field, value in data.offers[pair[f"offer_{side}"]].items()})
    row.update(changes)
    return row


def test_raw_strings_bom_extra_fields_and_multiline_descriptions_preserved(tmp_path):
    original = offer("A", raw_description='Рядок, "цитата"\nдругий рядок', custom="  N/A  ")
    data = dataset(tmp_path, [original, offer("B", custom="")])
    assert data.offers["A"] == original
    assert data.offers["A"]["stator_code"] == "0802"
    assert data.offers["A"]["seller_sku"] == "00023"
    assert data.offers["B"]["raw_description"] == ""
    assert set(data.source_hashes) == {"offers_sha256", "candidates_sha256"}
    assert all(len(value) == 64 for value in data.source_hashes.values())


@pytest.mark.parametrize("payload", [
    "a,b\n1,2,3\n", "a,b\n1\n", "a,b\n\n", "a,a\n1,2\n",
    "a,\n1,2\n", " a,b\n1,2\n", "a,b\n\"unterminated,2\n", "",
])
def test_strict_csv_rejects_bad_shape_header_and_quoting(tmp_path, payload):
    path = tmp_path / "bad.csv"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        read_csv_records(path)


def test_missing_columns_invalid_utf8_and_empty_data_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_bytes(b"a\n\xff\n")
    with pytest.raises(ValueError, match="UTF-8"):
        read_csv_records(path)
    write_csv(path, [{"a": "1"}])
    with pytest.raises(ValueError, match="missing required"):
        read_csv_records(path, OFFER_COLUMNS)
    path = write_csv(tmp_path / "empty.csv", [], OFFER_COLUMNS)
    with pytest.raises(ValueError, match="nonempty"):
        load_fpv_dataset(path, write_csv(tmp_path / "pairs.csv", [candidate()]))


@pytest.mark.parametrize("offer_rows,pair_rows,message", [
    ([offer("A"), offer("A")], [candidate()], "Duplicate offer_id"),
    ([offer("")], [candidate()], "blank or padded"),
    ([offer(" A"), offer("B")], [candidate()], "blank or padded"),
    ([offer("A"), offer("B")], [candidate(a="")], "blank or padded"),
    ([offer("A"), offer("B")], [candidate(pair_id="")], "blank or padded"),
    ([offer("A"), offer("B")], [candidate(b="missing")], "unknown offer"),
    ([offer("A"), offer("B")], [candidate(b="A")], "self-pair"),
    ([offer("A"), offer("B")], [candidate(), candidate()], "Duplicate pair_id"),
    ([offer("A"), offer("B")], [candidate(), candidate("P2", "B", "A")], "duplicate unordered"),
])
def test_offer_and_pair_referential_integrity(tmp_path, offer_rows, pair_rows, message):
    with pytest.raises(ValueError, match=message):
        dataset(tmp_path, offer_rows, pair_rows)


@pytest.mark.parametrize("label", ["", " 1", "01", "True", "1.0", "2"])
def test_invalid_proposal_does_not_get_silently_coerced(tmp_path, label):
    with pytest.raises(ValueError, match="binary label"):
        dataset(tmp_path, pairs=[candidate(proposed_label_1_match_0_nonmatch=label)])


def test_mapping_keeps_technical_fields_separate_and_never_exposes_proposed_gold(tmp_path):
    data = dataset(tmp_path, [offer("A", entity_id="same", raw_description=""),
                              offer("B", entity_id="same", raw_description="Опис")],
                   [candidate(verification_status="already_verified_by_someone", label="1")])
    pair = data.matching_pairs()[0]
    assert "label" not in pair
    assert "cluster_id_left" not in pair
    assert pair["verification_status"] == "already_verified_by_someone"
    assert pair["review_status"] == "pending"
    assert pair["title_left"] == data.offers["A"]["raw_title"]
    assert pair["description_left"] == ""
    assert pair["fpv_seller_sku_left"] == "00023"
    assert pair["fpv_entity_id_left"] == "same"
    assert pair["fpv_candidate"] == data.candidates[0]
    features = FeatureExtractor().fit(data.matching_pairs()).transform(data.matching_pairs())
    assert features.ids_left == ["A"]
    assert features.ids_right == ["B"]
    assert np.isnan(features.matrix[0, 1])


def test_typed_view_does_not_extract_from_title_or_copy_other_offer(tmp_path):
    data = dataset(tmp_path, [offer("A", kv_rpm_per_volt="", voltage_s=""),
                              offer("B", kv_rpm_per_volt="1300", voltage_s="6S")])
    pair = data.matching_pairs()[0]
    assert pair["fpv_technical_views"]["left"]["kv_rpm_per_volt"] is None
    assert pair["fpv_technical_views"]["left"]["voltage_s"] is None
    assert pair["fpv_kv_rpm_per_volt_left"] == ""
    assert pair["fpv_kv_rpm_per_volt_right"] == "1300"


def test_additional_raw_columns_cannot_be_overwritten_by_typed_views(tmp_path):
    data = dataset(tmp_path, [offer("A", technical="literal", technical_views="original"),
                              offer("B", technical="", technical_views="")])
    pair = data.matching_pairs()[0]
    assert pair["fpv_technical_left"] == "literal"
    assert pair["fpv_technical_views_left"] == "original"
    assert pair["fpv_technical_views"]["left"]["kv_rpm_per_volt"] == 1300


@pytest.mark.parametrize("raw", ["1300/1750", "1300.0", "1e3", "0", "-1300", "1300 KV"])
def test_unresolved_kv_stays_raw_and_not_a_single_variant_value(raw):
    original = offer("A", kv_rpm_per_volt=raw)
    view = technical_view(original)
    assert view["kv_rpm_per_volt"] is None
    assert "unresolved_kv_rpm_per_volt" in view["issues"]
    assert original["kv_rpm_per_volt"] == raw


@pytest.mark.parametrize("stator", ["0802", "2207.5", "22.6-6.6"])
def test_stator_notations_remain_original_strings(stator):
    assert technical_view(offer("A", stator_code=stator))["stator_code"] == stator


def test_confirmed_import_uses_manual_label_even_when_proposal_differs(tmp_path):
    data = dataset(tmp_path)
    reviewed = reviewed_row(data, review_notes="Пропозицію відхилено за ревізією")
    pair = data.confirmed_pairs(write_csv(tmp_path / "review.csv", [reviewed]))[0]
    assert pair["label"] == 0
    assert pair["proposed_label_1_match_0_nonmatch"] == "1"
    assert pair["review_status"] == "confirmed"
    assert pair["verification_status"] == "needs_manual_confirmation"
    assert "label" not in data.matching_pairs()[0]


@pytest.mark.parametrize("changes,message", [
    ({"review_status": "pending"}, "review_status"),
    ({"review_status": "uncertain"}, "review_status"),
    ({"review_status": "CONFIRMED"}, "review_status"),
    ({"confirmed_label_1_match_0_nonmatch": ""}, "binary label"),
    ({"confirmed_label_1_match_0_nonmatch": "01"}, "binary label"),
    ({"reviewer": "  "}, "reviewer"),
    ({"reviewed_on": "2026-02-30"}, "date"),
    ({"reviewed_on": "20261103"}, "ISO date"),
    ({"evidence_url": ""}, "evidence_url"),
    ({"evidence_url": "ftp://example.com"}, "evidence_url"),
    ({"evidence_url": "https://example.com/a b"}, "evidence_url"),
    ({"offers_sha256": "0" * 64}, "offers_sha256"),
    ({"candidates_sha256": "0" * 64}, "candidates_sha256"),
    ({"offer_a": "B", "offer_b": "A"}, "original candidate"),
    ({"proposed_label_1_match_0_nonmatch": "0"}, "original candidate"),
    ({"raw_title_a": "Another title"}, "original offer field"),
    ({"pair_id": "unknown"}, "unknown pair_id"),
])
def test_confirmed_import_rejects_unreviewed_forged_and_stale_rows(tmp_path, changes, message):
    data = dataset(tmp_path)
    with pytest.raises(ValueError, match=message):
        data.confirmed_pairs(write_csv(tmp_path / "review.csv", [reviewed_row(data, **changes)]))


@pytest.mark.parametrize("changes", [
    {"record_status": "needs_variant_split"}, {"record_status": "needs_variant_check"},
    {"record_status": "another_unknown_status"}, {"kv_rpm_per_volt": "1300/1750"},
])
def test_confirmed_label_cannot_resolve_an_ambiguous_source_listing(tmp_path, changes):
    data = dataset(tmp_path, [offer("A", **changes), offer("B")])
    row = reviewed_row(data, variant_resolution="I think so")
    with pytest.raises(ValueError, match="unresolved variant"):
        data.confirmed_pairs(write_csv(tmp_path / "review.csv", [row]))


def test_confirmed_import_rejects_duplicates_empty_and_mixed_queues(tmp_path):
    data = dataset(tmp_path, [offer("A"), offer("B"), offer("C")],
                   [candidate(), candidate("P2", "A", "C")])
    first = reviewed_row(data)
    with pytest.raises(ValueError, match="duplicate"):
        data.confirmed_pairs(write_csv(tmp_path / "review.csv", [first, first]))
    second = reviewed_row(data, data.candidates[1], review_status="pending")
    with pytest.raises(ValueError, match="review_status"):
        data.confirmed_pairs(write_csv(tmp_path / "review.csv", [first, second]))
    with pytest.raises(ValueError, match="No confirmed"):
        data.confirmed_pairs(write_csv(tmp_path / "review.csv", [], list(first)))


def test_source_bytes_change_invalidates_previous_review(tmp_path):
    data = dataset(tmp_path)
    row = reviewed_row(data)
    original = tmp_path / "offers.csv"
    original.write_bytes(original.read_bytes().replace(b"1300", b"1400"))
    changed = load_fpv_dataset(original, tmp_path / "pairs.csv")
    with pytest.raises(ValueError, match="offers_sha256"):
        changed.confirmed_pairs(write_csv(tmp_path / "review.csv", [row]))
