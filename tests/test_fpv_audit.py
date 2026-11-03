"""FPV data audit and manual-review preservation, using isolated unit fixtures."""
import csv
import json

import pytest

from audit_fpv import audit_data, main, write_review_queue
from fpv_dataset import CANDIDATE_COLUMNS, OFFER_COLUMNS, read_csv_records


def offer(offer_id, store="shop", **kwargs):
    row = {column: "" for column in OFFER_COLUMNS}
    row.update(offer_id=offer_id, store=store, raw_title=f"Motor {offer_id}", brand="Maker",
               family_model="Series V2", stator_code="0802", kv_rpm_per_volt="23000",
               units_in_listing="1", source_url=f"https://shop.example/{offer_id}",
               collected_on="2026-11-03", record_status="single_variant")
    row.update(kwargs)
    return row


def candidate(pair_id="P1", left="a", right="b", **kwargs):
    row = dict(pair_id=pair_id, offer_a=left, offer_b=right,
               proposed_label_1_match_0_nonmatch="1",
               verification_status="needs_manual_confirmation", rationale="check identity")
    row.update(kwargs)
    return row


def write_csv(path, rows, columns=None):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns or rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def sources(tmp_path, offers=None, candidates=None):
    return (write_csv(tmp_path / "offers.csv", offers or [offer("a", "one"), offer("b", "two")]),
            write_csv(tmp_path / "pairs.csv", candidates or [candidate()]))


def test_audit_never_promotes_proposals_or_source_status(tmp_path):
    a, b = sources(tmp_path, candidates=[candidate(verification_status="confirmed")])
    report, rows = audit_data(a, b)
    assert report["counts"]["proposed_positive"] == 1
    assert report["counts"]["confirmed_labels_in_sources"] == 0
    assert rows[0]["verification_status"] == "confirmed"
    assert rows[0]["review_status"] == "pending"
    assert rows[0]["confirmed_label_1_match_0_nonmatch"] == ""
    assert rows[0]["stator_code_a"] == "0802"
    assert rows[0]["raw_title_b"] == "Motor b"
    assert rows[0]["raw_description_a"] == ""


@pytest.mark.parametrize("reserved", [
    "review_status", "confirmed_label_1_match_0_nonmatch", "offers_sha256",
    "issue_flags", "review_priority", "raw_title_a", "raw_description_b",
])
def test_candidate_extension_columns_cannot_overwrite_queue_fields(tmp_path, reserved):
    a, b = sources(tmp_path, candidates=[candidate(**{reserved: "original source value"})])
    original_sources = (a.read_bytes(), b.read_bytes())
    with pytest.raises(ValueError, match="Candidate columns collide with reserved review queue fields"):
        audit_data(a, b)
    assert (a.read_bytes(), b.read_bytes()) == original_sources


def test_missingness_counts_zero_as_present_without_imputation(tmp_path):
    a, b = sources(tmp_path, offers=[offer("a", price_uah="0", voltage_s="6S"), offer("b")])
    report, rows = audit_data(a, b)
    assert report["offer_missingness"]["price_uah"]["missing_fraction"] == 0.5
    assert report["offer_missingness"]["raw_description"]["missing_fraction"] == 1
    assert report["offer_missingness"]["voltage_s"]["missing"] == 1
    assert not report["integrity"]["malformed_offer_values"]
    assert rows[0]["voltage_s_b"] == ""


def test_model_groups_keep_brand_series_revision_and_unknown_variants(tmp_path):
    offers = [offer("a", "one"), offer("b", "two"),
              offer("c", "two", brand="Other"),
              offer("d", "two", family_model="Series V3"),
              offer("e", "two", kv_rpm_per_volt="", record_status="needs_variant_split")]
    a, b = sources(tmp_path, offers=offers)
    report, _ = audit_data(a, b)
    models = report["models"]
    assert models["provisional_family_groups"] == 3
    assert models["complete_provisional_variant_groups"] == 3
    assert models["provisional_variant_groups_including_incomplete"] == 4
    assert models["incomplete_variant_offer_ids"] == ["e"]
    assert models["cross_store_group_count"] == 1
    assert models["potential_cross_store_pair_count"] == 1
    assert models["cross_store_groups"][0]["offer_ids"] == ["a", "b"]
    assert "needs_variant_split" in report["offer_review_flags"]["e"]


def test_grouping_does_not_alias_brands_or_erase_pack_difference(tmp_path):
    offers = [offer("a", "one", brand="RCINPOWER"),
              offer("b", "two", brand="AOS RC"),
              offer("c", "two", brand="RCINPOWER", units_in_listing="4")]
    a, b = sources(tmp_path, offers=offers, candidates=[candidate(right="c")])
    report, rows = audit_data(a, b)
    assert report["models"]["complete_provisional_variant_groups"] == 2
    assert report["models"]["cross_store_groups"][0]["units_in_listing_values"] == ["1", "4"]
    assert "known_units_in_listing_difference" in rows[0]["issue_flags"]
    assert "multi_unit_listing" in report["offer_review_flags"]["c"]


def test_audit_reports_missing_ambiguous_and_self_references(tmp_path):
    a, b = sources(tmp_path, offers=[offer("a"), offer("a"), offer("b")],
                   candidates=[candidate(left="missing"), candidate("P2", "a", "b"),
                               candidate("P3", "b", "b")])
    report, rows = audit_data(a, b)
    assert report["integrity"]["duplicate_offer_ids"] == ["a"]
    assert report["integrity"]["self_pairs"] == ["P3"]
    assert {entry["reason"] for entry in report["integrity"]["unresolved_references"]} == {"missing", "ambiguous"}
    assert rows[0]["raw_title_a"] == ""
    assert rows[1]["raw_title_a"] == ""
    assert rows[0]["review_priority"] == "high"


def test_reverse_pairs_and_conflicting_proposals_are_reported(tmp_path):
    a, b = sources(tmp_path, candidates=[candidate(), candidate("N2", "b", "a", proposed_label_1_match_0_nonmatch="0")])
    report, rows = audit_data(a, b)
    assert report["integrity"]["duplicate_canonical_pairs"] == [
        {"offer_ids": ["a", "b"], "pair_ids": ["P1", "N2"]}]
    assert report["counts"]["proposed_positive"] == report["counts"]["proposed_negative"] == 1
    assert all(row["confirmed_label_1_match_0_nonmatch"] == "" for row in rows)


def test_duplicate_urls_and_normalized_titles_are_reported(tmp_path):
    offers = [offer("a", raw_title=" Motor  A "),
              offer("b", raw_title="motor a", source_url="https://shop.example/a")]
    a, b = sources(tmp_path, offers=offers)
    report, _ = audit_data(a, b)
    assert report["integrity"]["duplicate_source_urls"][0]["ids"] == ["a", "b"]
    assert report["integrity"]["duplicate_normalized_titles"][0]["ids"] == ["a", "b"]


def test_exact_duplicate_detection_keeps_legitimate_missing_values(tmp_path):
    first = offer("a")
    second = dict(first, offer_id="b")
    pairs = [candidate(rationale=""), candidate(rationale="")]
    a, b = sources(tmp_path, offers=[first, second], candidates=pairs)
    report, _ = audit_data(a, b)
    assert report["integrity"]["duplicate_exact_offers_excluding_id"][0]["ids"] == ["a", "b"]
    assert report["integrity"]["duplicate_exact_candidate_rows"][0]["ids"] == ["P1", "P1"]
    assert report["integrity"]["duplicate_pair_ids"] == ["P1"]


def test_offers_without_pairs_are_counted_separately(tmp_path):
    a, b = sources(tmp_path, offers=[offer("a"), offer("b"), offer("unused")])
    report, _ = audit_data(a, b)
    assert report["counts"]["unique_existing_offers_in_candidate_pairs"] == 2
    assert report["counts"]["offers_without_candidate_pairs"] == 1
    assert report["offer_ids_without_candidate_pairs"] == ["unused"]


@pytest.mark.parametrize(("field", "value", "flag"), [
    ("source_url", "not-a-url", "malformed_source_url"),
    ("source_url", "http://[broken", "malformed_source_url"),
    ("collected_on", "not-a-date", "malformed_collected_on"),
    ("collected_on", "20261103", "malformed_collected_on"),
    ("kv_rpm_per_volt", "NaN", "invalid_kv_rpm_per_volt"),
    ("units_in_listing", "0", "invalid_units_in_listing"),
    ("units_in_listing", "1.5", "invalid_units_in_listing"),
    ("price_uah", "-1", "invalid_price_uah"),
    ("stator_code", "unknown", "nonstandard_stator_code"),
    ("voltage_s", "some", "nonstandard_voltage_s"),
])
def test_malformed_values_are_reported_not_coerced(tmp_path, field, value, flag):
    a, b = sources(tmp_path, offers=[offer("a", **{field: value}), offer("b")])
    report, rows = audit_data(a, b)
    assert flag in report["integrity"]["malformed_offer_values"]["a"]
    assert rows[0][f"{field}_a"] == value


def test_queue_preserves_review_and_custom_columns_on_rerun(tmp_path):
    a, b = sources(tmp_path)
    report, rows = audit_data(a, b)
    queue = tmp_path / "queue.csv"
    first = write_review_queue(queue, rows)
    assert first["pending_rows"] == 1
    assert queue.read_bytes().startswith(b"\xef\xbb\xbf")
    edited = read_csv_records(queue)
    edited[0].update(confirmed_label_1_match_0_nonmatch="0", review_status="confirmed",
                     reviewer="Reviewer", reviewed_on="2026-11-03", evidence_url="https://example.test/evidence",
                     review_notes="manual decision", custom_comment="retain me")
    write_csv(queue, edited)
    _, fresh = audit_data(a, b)
    result = write_review_queue(queue, fresh)
    actual = read_csv_records(queue)[0]
    assert result["preserved_existing_rows"] == 1
    assert result["confirmed_label_cells_present"] == 1
    for field in ("confirmed_label_1_match_0_nonmatch", "review_status", "reviewer", "evidence_url", "review_notes", "custom_comment"):
        assert actual[field] == edited[0][field]
    assert report["sources"]["offers_sha256"] == actual["offers_sha256"]


def test_queue_refuses_source_drift_with_same_ids_and_preserves_bytes(tmp_path):
    a, b = sources(tmp_path)
    _, rows = audit_data(a, b)
    queue = tmp_path / "queue.csv"
    write_review_queue(queue, rows)
    original = queue.read_bytes()
    altered = read_csv_records(a)
    altered[0]["raw_title"] = "Revised source title with same ID"
    write_csv(a, altered)
    _, fresh = audit_data(a, b)
    with pytest.raises(ValueError, match="drift.*refusing overwrite"):
        write_review_queue(queue, fresh)
    assert queue.read_bytes() == original


def test_queue_refuses_pair_set_change_and_duplicate_ids(tmp_path):
    a, b = sources(tmp_path)
    _, rows = audit_data(a, b)
    queue = tmp_path / "queue.csv"
    write_review_queue(queue, rows)
    changed = [dict(rows[0], pair_id="new")]
    with pytest.raises(ValueError, match="pair set changed"):
        write_review_queue(queue, changed)
    with pytest.raises(ValueError, match="unique and nonblank"):
        write_review_queue(tmp_path / "duplicates.csv", rows + rows)


def test_queue_refuses_generated_cell_tampering_and_invalid_manual_label(tmp_path):
    a, b = sources(tmp_path)
    _, rows = audit_data(a, b)
    queue = tmp_path / "queue.csv"
    write_review_queue(queue, rows)
    edited = read_csv_records(queue)
    edited[0]["raw_title_a"] = "changed copied source"
    write_csv(queue, edited)
    with pytest.raises(ValueError, match="drift"):
        write_review_queue(queue, rows)
    edited[0]["raw_title_a"] = rows[0]["raw_title_a"]
    edited[0]["confirmed_label_1_match_0_nonmatch"] = "maybe"
    write_csv(queue, edited)
    with pytest.raises(ValueError, match="Invalid confirmed label"):
        write_review_queue(queue, rows)


def test_cli_keeps_sources_unchanged_and_writes_machine_audit(tmp_path):
    a, b = sources(tmp_path)
    originals = (a.read_bytes(), b.read_bytes())
    queue, report = tmp_path / "queue.csv", tmp_path / "audit.json"
    main(["--offers", str(a), "--pairs", str(b), "--review-queue", str(queue), "--out", str(report)])
    assert originals == (a.read_bytes(), b.read_bytes())
    assert json.loads(report.read_text(encoding="utf-8"))["queue"]["rows"] == 1
    assert read_csv_records(queue)[0]["confirmed_label_1_match_0_nonmatch"] == ""


def test_cli_refuses_source_overwrite(tmp_path):
    a, b = sources(tmp_path)
    original = a.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["--offers", str(a), "--pairs", str(b), "--review-queue", str(a)])
    assert exc.value.code == 2
    assert a.read_bytes() == original


def test_atomic_queue_temp_cannot_collide_with_source_filename(tmp_path):
    a = write_csv(tmp_path / "queue.csv.tmp", [offer("a"), offer("b")])
    b = write_csv(tmp_path / "pairs.csv", [candidate()])
    original = a.read_bytes()
    queue, report = tmp_path / "queue.csv", tmp_path / "audit.json"
    main(["--offers", str(a), "--pairs", str(b), "--review-queue", str(queue), "--out", str(report)])
    assert a.read_bytes() == original
    assert read_csv_records(queue)[0]["offer_a"] == "a"
