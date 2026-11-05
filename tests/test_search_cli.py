"""CLI checks use temporary fixtures, never synthetic research observations."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

import search_cli
from fpv_dataset import OFFER_COLUMNS
from product_catalog import ProductOffer


def fpv_file(tmp_path, *, empty=False, unknown_store=False):
    rows = []
    for oid, title, store in (("A", "Motor ROBO 0802 23000KV", "Store One"),
                               ("B", "ROBO 0802 motor 23000KV", "Store Two"),
                               ("C", "Other 2812 motor 1100KV", "Store One"),
                               ("D", "", "Store Three")):
        row = dict.fromkeys(OFFER_COLUMNS, "")
        row.update(offer_id=oid, raw_title=title, brand="Example",
                   store="" if unknown_store else store,
                   source_url=f"https://example.com/{oid}", stator_code="0802",
                   family_model="ROBO", kv_rpm_per_volt="23000",
                   units_in_listing="1", seller_sku="0001", record_status="single_variant")
        rows.append(row)
    path = tmp_path / "offers.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OFFER_COLUMNS)
        writer.writeheader()
        if not empty:
            writer.writerows(rows)
    return path


def test_cli_fpv_json_runs_without_candidates_and_separates_search_stages(tmp_path, capsys):
    path = fpv_file(tmp_path)
    search_cli.main(["--catalog-format", "fpv", "--catalog", str(path), "--query-id", "A",
                     "--candidate-k", "2", "--top", "5", "--other-stores-only", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert not (tmp_path / "fpv_motors_pair_candidates.csv").exists()
    assert payload["catalog_size"] == 4
    assert payload["query"]["description"] == ""
    assert payload["retrieval"]["candidate_count"] == 1
    assert payload["ranking"]["method"] == "renorm"
    result = payload["ranking"]["results"][0]
    assert result["offer"]["offer_id"] == "B"
    assert result["offer"]["source"] == "Store Two"
    assert result["offer"]["url"] == "https://example.com/B"
    assert result["offer"]["technical_attributes"]["stator_code"] == "0802"
    assert result["offer"]["technical_attributes"]["seller_sku"] == "0001"
    assert result["feature_similarities"]["description"] is None
    assert set(("retrieval_score", "similarity_score")) <= result.keys()
    assert payload["verification"] == {"matches_confirmed": False,
                                       "search_metrics_computed": False, "gold_labels_used": False}


def test_cli_fixed_console_displays_metadata_and_score_meaning(tmp_path, capsys):
    path = fpv_file(tmp_path)
    search_cli.main(["--format", "fpv", "--catalog", str(path), "--query-id", "A",
                     "--method", "fixed", "--candidate-k", "2", "--top-k", "1"])
    output = capsys.readouterr().out
    assert "Store Two" in output and "https://example.com/B" in output
    assert "stator_code=0802" in output and "seller_sku=0001" in output
    assert "TF-IDF retrieval:" in output and "Подібність:" in output
    assert "не є каліброваними" in output and "Тотожність результатів не підтверджено" in output


@pytest.mark.parametrize("extra", [
    ["--query-id", "unknown"], ["--query-id", "A", "--top", "0"],
    ["--query-id", "A", "--candidate-k", "-1"],
])
def test_cli_bad_query_or_counts_give_clean_error(tmp_path, capsys, extra):
    path = fpv_file(tmp_path)
    with pytest.raises(SystemExit) as error:
        search_cli.main(["--format", "fpv", "--catalog", str(path), *extra])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "Помилка пошуку:" in stderr and "Traceback" not in stderr


@pytest.mark.parametrize("case", ["empty", "unknown_store", "missing_file"])
def test_cli_catalog_and_source_errors_are_explicit(tmp_path, capsys, case):
    path = fpv_file(tmp_path, empty=case == "empty", unknown_store=case == "unknown_store")
    if case == "missing_file":
        path = tmp_path / "absent.csv"
    with pytest.raises(SystemExit) as error:
        search_cli.main(["--format", "fpv", "--catalog", str(path), "--query-id", "A",
                         "--other-stores-only"])
    assert error.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_empty_title_returns_valid_empty_json_response(tmp_path, capsys):
    path = fpv_file(tmp_path)
    search_cli.main(["--format", "fpv", "--catalog", str(path), "--query-id", "D", "--json"])
    response = json.loads(capsys.readouterr().out)
    assert response["retrieval"]["candidate_count"] == 0
    assert response["ranking"]["results"] == []


@pytest.mark.parametrize("name", ["wdcproducts80cc20rnd100un_gs.json.gz", "valid.json.gz"])
def test_nontrain_split_is_rejected_before_any_file_open(monkeypatch, name):
    def forbidden_open(*args, **kwargs):
        raise AssertionError("Non-TRAIN input must not be opened")
    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(ValueError, match="only audited TRAIN"):
        search_cli.verify_wdc_train(Path("never_open") / name)


def test_wdc_train_alias_and_checksum_gate_with_label_free_fixture(tmp_path, monkeypatch, capsys):
    record = {"id_left": "A", "id_right": "B", "title_left": "Camera AX12345",
              "title_right": "AX12345 Camera", "label": "not-used-in-search",
              "cluster_id_left": "also-ignored", "cluster_id_right": "different"}
    path = tmp_path / search_cli.WDC_TRAIN_NAME
    path.write_bytes(gzip.compress((json.dumps(record) + "\n").encode()))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "archive_manifest.json").write_text(json.dumps({"members": [{
        "name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}), encoding="utf-8")
    monkeypatch.setattr(search_cli, "PROJECT_DIRECTORY", tmp_path)
    search_cli.main(["--pairs", str(path), "--query-id", "A", "--json"])
    response = json.loads(capsys.readouterr().out)
    assert response["catalog_format"] == "wdc" and response["catalog_size"] == 2
    assert response["unknown_source_offers"] == 2
    assert response["ranking"]["results"][0]["offer"]["source"] == ""
    assert "label" not in response["query"]
    assert "not-used-in-search" not in json.dumps(response)
    assert "cluster_id" not in response["ranking"]["results"][0]["offer"]
    path.write_bytes(b"different artifact")
    with pytest.raises(ValueError, match="checksum mismatch"):
        search_cli.verify_wdc_train(path)


@pytest.mark.parametrize("method", ["adaptive", "logistic", "unknown"])
def test_cli_does_not_implicitly_enable_adaptive_or_untrained_logistic(tmp_path, method):
    path = fpv_file(tmp_path)
    with pytest.raises(SystemExit) as error:
        search_cli.main(["--format", "fpv", "--catalog", str(path), "--query-id", "A", "--method", method])
    assert error.value.code == 2


def test_json_and_console_empty_catalog_guard_and_missing_feature_output():
    with pytest.raises(ValueError, match="empty catalog"):
        search_cli.search_payload({}, "A")
    response = search_cli.search_payload({"A": ProductOffer("A")}, "A")
    console = search_cli.render_console(response)
    assert "назва відсутня" in console
    assert "Немає кандидатів" in console
