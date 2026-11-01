"""Protocol tests use tiny fixtures, never benchmark-derived scientific results."""
import gzip
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from prepare_validation import (
    NESTED_ARCHIVE, VALIDATION_BASENAME, download_archive, metadata_index, overlap,
    prepare_validation, read_metadata, sha256,
)


def pair(left, right, cluster, label=1):
    return {"id_left": left, "id_right": right, "cluster_id_left": cluster,
            "cluster_id_right": cluster, "title_left": f"Product {left}",
            "title_right": f"Product {right}", "label": label}


def compressed(rows):
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    return gzip.compress(text.encode("utf-8"), mtime=0)


def setup_data(tmp_path, candidate=None, nested=True):
    train = tmp_path / "train.json.gz"
    seen = tmp_path / "seen.json.gz"
    test = tmp_path / "test.json.gz"
    train.write_bytes(compressed([pair("a", "b", "train-cluster")]))
    seen.write_bytes(compressed([pair("c", "d", "train-cluster")]))
    # Invalid test labels deliberately demonstrate that label validation is absent.
    test.write_bytes(compressed([pair("x", "y", "test-cluster", label="DO NOT READ")]))
    rows = candidate if candidate is not None else [pair("e", "f", "unseen-cluster")]
    archive = tmp_path / "valpair.zip"
    nested_bytes = BytesIO()
    with ZipFile(nested_bytes, "w") as zipped:
        zipped.writestr(VALIDATION_BASENAME, compressed(rows))
        zipped.writestr(VALIDATION_BASENAME.replace("100un", "050un"), compressed(rows))
    if nested:
        with ZipFile(archive, "w") as zipped:
            zipped.writestr(NESTED_ARCHIVE, nested_bytes.getvalue())
            zipped.writestr("20pair_addvalid.zip", b"manifested but never inspected")
    else:
        archive.write_bytes(nested_bytes.getvalue())
    return archive, train, seen, test


def prepare(tmp_path, **kwargs):
    return prepare_validation(*setup_data(tmp_path, **kwargs),
                              output_dir=tmp_path / "docs", extract_dir=tmp_path / "extracted")


class GuardedLabel(dict):
    def get(self, key, default=None):
        assert key != "label", "Test labels must remain inaccessible"
        return super().get(key, default)

    def __getitem__(self, key):
        assert key != "label", "Test labels must remain inaccessible"
        return super().__getitem__(key)


def test_metadata_index_never_reads_labels_and_is_symmetric():
    left = metadata_index([GuardedLabel(pair("a", "b", "one"))])
    right = metadata_index([pair("b", "a", "one", label="invalid")])
    assert overlap(left, right) == {"offer_ids": 2, "cluster_ids": 1,
                                  "canonical_pairs": 1, "text_signatures": 2, "title_signatures": 2}
    assert not hasattr(left, "labels")


def test_metadata_reader_discards_labels_and_flags_at_decoder(tmp_path):
    path = tmp_path / "test.json.gz"
    record = pair("a", "b", "one", label={"secret": "unused"})
    record["is_hard_negative"] = "invalid flag intentionally ignored"
    path.write_bytes(compressed([record]))
    inventory = read_metadata(path)
    assert inventory.counts() == {"pairs": 1, "unique_offers": 2,
                                  "unique_clusters": 1, "duplicate_canonical_pairs": 0}


@pytest.mark.parametrize("nested", [True, False])
def test_valid_unseen_protocol_extracts_only_selected_and_preserves_archive(tmp_path, nested):
    archive, train, seen, test = setup_data(tmp_path, nested=nested)
    original_hash = sha256(archive)
    result = prepare_validation(archive, train, seen, test, tmp_path / "docs", tmp_path / "extracted")
    config = result["configuration"]
    assert config["mode"] == "official_unseen"
    assert not config["candidate_rejected"]
    assert config["validation_unseen_percentage"] == 100
    assert sha256(archive) == original_hash
    assert [item.name for item in (tmp_path / "extracted").iterdir()] == [VALIDATION_BASENAME]
    assert result["additional_small_candidate_content_identical"]
    assert (tmp_path / "docs" / "validation_config.json").exists()
    assert "labels" not in result["test_metadata_only"]
    assert result["overlaps"]["train__seen_validation"]["cluster_ids"] == 1


def test_leaky_additional_validation_falls_back_without_filtering(tmp_path):
    rows = [pair("a", "b", "train-cluster"), pair("e", "f", "unseen-cluster")]
    result = prepare(tmp_path, candidate=rows)
    config = result["configuration"]
    assert config["mode"] == "fallback_seen"
    assert config["candidate_rejected"]
    assert config["validation_unseen_percentage"] == 0
    assert Path(config["validation_path"]).name == "seen.json.gz"
    assert result["splits"]["unseen_validation"]["pairs"] == 2
    assert result["fallback"]["no_filtering_or_relabeling"]
    assert result["overlaps"]["train__unseen_validation"]["canonical_pairs"] == 1
    for candidate in result["additional_small_candidates"].values():
        assert not candidate["eligible_for_100_percent_unseen_validation"]
        assert candidate["empirical_unseen_cluster_fraction_vs_train"] == .5


def test_shared_cluster_alone_rejects_claim_of_fully_unseen_validation(tmp_path):
    result = prepare(tmp_path, candidate=[pair("e", "f", "train-cluster")])
    assert result["configuration"]["mode"] == "fallback_seen"
    assert result["overlaps"]["train__unseen_validation"]["offer_ids"] == 0
    assert result["overlaps"]["train__unseen_validation"]["cluster_ids"] == 1


def test_fallback_refuses_train_offer_leakage(tmp_path):
    archive, train, seen, test = setup_data(tmp_path, candidate=[pair("a", "b", "train-cluster")])
    seen.write_bytes(compressed([pair("a", "c", "train-cluster")]))
    with pytest.raises(ValueError, match="fallback seen validation overlaps training"):
        prepare_validation(archive, train, seen, test, tmp_path / "docs", tmp_path / "extracted")


def test_fallback_refuses_reserved_test_cluster_overlap(tmp_path):
    archive, train, seen, test = setup_data(tmp_path, candidate=[pair("a", "b", "train-cluster")])
    seen.write_bytes(compressed([pair("c", "d", "test-cluster")]))
    with pytest.raises(ValueError, match="fallback validation overlaps reserved test"):
        prepare_validation(archive, train, seen, test, tmp_path / "docs", tmp_path / "extracted")


def test_existing_different_extraction_is_not_overwritten(tmp_path):
    archive, train, seen, test = setup_data(tmp_path)
    destination_dir = tmp_path / "extracted"
    destination_dir.mkdir()
    destination = destination_dir / VALIDATION_BASENAME
    destination.write_bytes(b"existing data")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_validation(archive, train, seen, test, tmp_path / "docs", destination_dir)
    assert destination.read_bytes() == b"existing data"


def test_repeated_preparation_preserves_identical_extracted_file(tmp_path):
    args = setup_data(tmp_path)
    first = prepare_validation(*args, tmp_path / "docs", tmp_path / "extracted")
    second = prepare_validation(*args, tmp_path / "docs", tmp_path / "extracted")
    assert first == second


def test_download_existing_archive_never_opens_network(tmp_path, monkeypatch):
    existing = tmp_path / "valpair.zip"
    existing.write_bytes(b"preserve me")
    monkeypatch.setattr("prepare_validation.urlopen", lambda *args, **kwargs: pytest.fail("network access"))
    download_archive(existing)
    assert existing.read_bytes() == b"preserve me"
