"""Regression coverage for schema coercion, missingness, and split leakage."""
import gzip
import json
from zipfile import ZipFile

import pytest

from audit_wdc import SELECTED_MEMBERS, audit, audit_archive, read_pairs, summary


def pair(left="a", right="b", label=1, **kwargs):
    return {"id_left": left, "id_right": right, "title_left": f"Title {left}",
            "title_right": f"Title {right}", "label": label, **kwargs}


def write_jsonl(path, records):
    payload = "\n".join(json.dumps(record) for record in records) + "\n"
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_reader_accepts_path_and_validates_later_rows(tmp_path, suffix):
    invalid = pair("c", "d")
    del invalid["title_right"]
    path = write_jsonl(tmp_path / f"pairs{suffix}", [pair(), invalid])
    with pytest.raises(ValueError, match="2: missing required fields.*title_right"):
        read_pairs(path)


@pytest.mark.parametrize("value", [0.7, 1.2, 1.0, True, False, None, "01", " 1", 2])
def test_reader_rejects_coerced_binary_labels(tmp_path, value):
    path = write_jsonl(tmp_path / "pairs.json", [pair(label=value)])
    with pytest.raises(ValueError, match="invalid binary label"):
        read_pairs(path)


@pytest.mark.parametrize("value", [0, 1, "0", "1"])
def test_reader_accepts_exact_binary_labels(tmp_path, value):
    path = write_jsonl(tmp_path / "pairs.json", [pair(label=value)])
    assert read_pairs(path)[0]["label"] == value


@pytest.mark.parametrize("value", [None, "", " \t", True, 1.5, [], {}])
def test_reader_rejects_missing_or_invalid_offer_ids(tmp_path, value):
    path = write_jsonl(tmp_path / "pairs.json", [pair(left=value)])
    with pytest.raises(ValueError, match="invalid id_left"):
        read_pairs(path)


@pytest.mark.parametrize("record", [[], 1, "object"])
def test_reader_rejects_non_object_rows(tmp_path, record):
    path = write_jsonl(tmp_path / "pairs.json", [record])
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_pairs(path)


def test_reader_rejects_nonstandard_json_nan(tmp_path):
    path = write_jsonl(tmp_path / "pairs.json", [pair(price_left=float("nan"))])
    with pytest.raises(ValueError, match="Nonstandard JSON numeric constant"):
        read_pairs(path)


def test_reader_rejects_string_hard_negative_flag(tmp_path):
    path = write_jsonl(tmp_path / "pairs.json", [pair(is_hard_negative="false")])
    with pytest.raises(ValueError, match="must be a JSON boolean"):
        read_pairs(path)


def test_reader_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty split"):
        read_pairs(path)


def test_missing_counts_distinguish_null_blank_absent_and_zero():
    rows = [pair(brand_left=" \t", brand_right=None, price_left=0),
            pair("c", "d", brand_left="Brand", brand_right="Brand", price_left="0")]
    result = summary(rows)
    assert result["missing_rates_by_pair_side"]["brand"] == {"left": .5, "right": .5}
    assert result["missing_counts_by_pair_side"]["brand"]["left"]["blank"] == 1
    assert result["missing_counts_by_pair_side"]["brand"]["right"]["null"] == 1
    assert result["missing_counts_by_pair_side"]["description"]["left"]["absent_key"] == 2
    assert result["missing_rates_by_pair_side"]["price"]["left"] == 0


def test_reverse_duplicates_conflicting_labels_and_self_pairs():
    rows = [pair(), pair(), pair("b", "a", 0), pair("c", "c", 1)]
    result = summary(rows)
    assert result["duplicate_rows"] == 1
    assert result["duplicate_direct_pairs"] == 1
    assert result["duplicate_canonical_pairs"] == 2
    assert result["reversed_pair_groups"] == 1
    assert result["conflicting_label_pairs"] == 1
    assert result["self_pairs"] == 1


def test_audit_checks_pair_cluster_and_text_overlap_separately():
    train = [pair(cluster_id_left=1, cluster_id_right=1, pair_id="a#b")]
    valid = [pair("b", "a", 0, cluster_id_left=1, cluster_id_right=1, pair_id="b#a")]
    test = [pair("c", "d", 0, cluster_id_left=2, cluster_id_right=3,
                 title_left="  TITLE  A ", title_right="Different")]
    result = audit(train, valid, test)
    assert result["offer_id_overlaps"]["train_valid"] == 2
    assert result["pair_id_overlaps"]["train_valid"] == 0
    assert result["canonical_pair_overlaps"]["train_valid"] == 1
    assert result["cluster_id_overlaps"]["train_valid"] == 1
    assert result["text_signature_overlaps"]["train_test"] == 1
    assert result["cross_split_conflicting_label_pairs"] == 1
    assert result["splits"]["valid"]["label_cluster_mismatches"] == 1
    assert result["integrity_errors"]


def test_unique_offer_missingness_does_not_overweight_frequent_offers():
    rows = [pair(brand_left=None, brand_right="Brand"),
            pair("a", "c", brand_left=None, brand_right="Brand")]
    result = summary(rows)
    assert result["unique_offers"] == 3
    assert result["missing_rates_by_unique_offer"]["brand"] == .333333


def test_same_offer_id_with_changed_attributes_is_reported():
    result = summary([pair(), pair("a", "c", title_left="Different title")])
    assert result["inconsistent_offer_payloads"] == 1


def test_conflicts_within_one_split_are_not_called_cross_split_conflicts():
    result = audit([pair(), pair(label=0)], [pair("c", "d")], [pair("e", "f")])
    assert result["splits"]["train"]["conflicting_label_pairs"] == 1
    assert result["cross_split_conflicting_label_pairs"] == 0


def test_all_missing_text_does_not_create_signature_overlap():
    result = audit(*[[pair(f"{i}a", f"{i}b", title_left=None, title_right=" ")]
                     for i in range(3)])
    assert not any(result["text_signature_overlaps"].values())


def test_direct_audit_validates_records_and_empty_splits():
    with pytest.raises(ValueError, match="invalid binary label"):
        audit([pair(label=True)], [pair("c", "d")], [pair("e", "f")])
    with pytest.raises(ValueError, match="empty split"):
        audit([], [pair("c", "d")], [pair("e", "f")])


def make_archive(path):
    with ZipFile(path, "w") as zipped:
        for index, member in enumerate(SELECTED_MEMBERS.values()):
            row = pair(f"{index}a", f"{index}b")
            zipped.writestr(member, gzip.compress((json.dumps(row) + "\n").encode()))
        zipped.writestr("../unselected.txt", "never extract unselected paths")


def test_archive_extracts_only_fixed_files_and_is_idempotent(tmp_path):
    path = tmp_path / "80pair.zip"
    make_archive(path)
    original = path.read_bytes()
    output, destination = tmp_path / "docs", tmp_path / "data"
    result = audit_archive(path, output, destination)
    assert result["splits"]["train"]["pairs"] == 1
    assert sorted(file.name for file in destination.iterdir()) == sorted(SELECTED_MEMBERS.values())
    assert len(json.loads((output / "archive_manifest.json").read_text())["sha256"]) == 64
    audit_archive(path, output, destination)
    assert path.read_bytes() == original
    assert not (tmp_path / "unselected.txt").exists()


def test_archive_refuses_to_overwrite_different_existing_data(tmp_path):
    path = tmp_path / "80pair.zip"
    make_archive(path)
    destination = tmp_path / "data"
    destination.mkdir()
    target = destination / SELECTED_MEMBERS["train"]
    target.write_bytes(b"existing research data")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        audit_archive(path, tmp_path / "docs", destination)
    assert target.read_bytes() == b"existing research data"
