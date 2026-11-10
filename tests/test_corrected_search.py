"""Frozen corrected search checks; fixtures are software tests, not research data."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from adaptive_corrected import METHOD_SPECIFICATION, corrected_scores
from fpv_dataset import OFFER_COLUMNS
from frozen_method_artifact import load_corrected_scorer
from matching import FeatureExtractor
from product_catalog import ProductOffer, offer_to_pair
from search_system import SearchEngine
import search_cli


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def frozen_fixture(tmp_path):
    catalog = {
        "query": ProductOffer("query", "Brand Alpha AX12345 camera", brand="Brand", source="one"),
        "other": ProductOffer("other", "Alpha AX12345 camera", brand="Brand", source="two"),
        "different": ProductOffer("different", "Camera Beta ZZ99999", brand="Different", source="two"),
    }
    pairs = [offer_to_pair(catalog["query"], catalog["other"]),
             offer_to_pair(catalog["query"], catalog["different"])]
    extractor = FeatureExtractor().fit(pairs)
    info = np.array([1.2, .8, 1., .9])
    selected = {method: {"threshold": .2} for method in
                ("text", "fixed", "renorm", "adaptive", "logistic", "adaptive_corrected")}
    bundle = {"schema_version": 1, "extractor": extractor, "informativeness": info,
              "train_sha256": "a" * 64, "method_specification": copy.deepcopy(METHOD_SPECIFICATION),
              "selected": selected}
    bundle_path = tmp_path / "data/processed/final_method/frozen_models.joblib"
    bundle_path.parent.mkdir(parents=True)
    joblib.dump(bundle, bundle_path)
    code_hashes = {}
    for name in ("matching.py", "adaptive_corrected.py"):
        raw = (SOURCE_ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        (tmp_path / name).write_bytes(raw)
        code_hashes[name] = hashlib.sha256(raw).hexdigest()
    config = {"schema_version": 1, "method_specification": copy.deepcopy(METHOD_SPECIFICATION),
              "final_models": copy.deepcopy(selected),
              "artifacts": {"model_bundle": {"path": bundle_path.relative_to(tmp_path).as_posix(),
                                                "sha256": sha(bundle_path)}},
              "provenance": {"train_sha256": "a" * 64, "code_sha256_lf": code_hashes}}
    config_path = tmp_path / "docs/final_method/frozen_configuration.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return tmp_path, config_path, bundle_path, config, bundle, catalog, pairs


def save_config(path, configuration):
    path.write_text(json.dumps(configuration), encoding="utf-8")


def test_loaded_corrected_search_uses_frozen_train_features_without_refitting(frozen_fixture, monkeypatch):
    root, path, _, _, bundle, catalog, pairs = frozen_fixture

    def forbidden(*args, **kwargs):
        raise AssertionError("Corrected search attempted a training fit")

    monkeypatch.setattr(FeatureExtractor, "fit", forbidden)
    scorer = load_corrected_scorer(path, project_root=root)
    engine = SearchEngine(catalog, method="adaptive-corrected", pair_scorer=scorer)
    assert engine.extractor is None
    first = engine.search("query", candidate_k=2, top_k=2)
    assert first == engine.search("query", candidate_k=2, top_k=2)
    expected = corrected_scores(bundle["extractor"].transform(pairs).matrix,
                                bundle["informativeness"])
    expected_by_id = {pair["id_right"]: score for pair, score in zip(pairs, expected)}
    assert {row.offer.offer_id for row in first} == set(expected_by_id)
    for row in first:
        assert row.similarity_score == pytest.approx(expected_by_id[row.offer.offer_id])
        assert row.feature_similarities is None


def test_default_renormalization_unchanged_and_corrected_requires_frozen_scorer(frozen_fixture):
    catalog = frozen_fixture[5]
    assert SearchEngine(catalog).method == "renorm"
    assert search_cli.search_payload(catalog, "query")["ranking"]["method"] == "renorm"
    with pytest.raises(ValueError, match="already-fitted"):
        SearchEngine(catalog, method="adaptive-corrected")


def test_missing_bundle_is_not_silently_retrained(frozen_fixture):
    root, path, bundle_path, _, _, _, _ = frozen_fixture
    bundle_path.unlink()
    with pytest.raises(OSError):
        load_corrected_scorer(path, project_root=root)


def test_bundle_checksum_fails_before_deserialization(frozen_fixture, monkeypatch):
    root, path, bundle_path, _, _, _, _ = frozen_fixture
    bundle_path.write_bytes(bundle_path.read_bytes() + b"changed")

    def forbidden(*args, **kwargs):
        raise AssertionError("Checksum-mismatched artifact was opened by joblib")

    monkeypatch.setattr(joblib, "load", forbidden)
    with pytest.raises(ValueError, match="bundle checksum mismatch"):
        load_corrected_scorer(path, project_root=root)


def test_modified_scoring_code_is_rejected(frozen_fixture):
    root, path, _, _, _, _, _ = frozen_fixture
    (root / "matching.py").write_text("changed scoring rules", encoding="utf-8")
    with pytest.raises(ValueError, match="code checksum mismatch"):
        load_corrected_scorer(path, project_root=root)


def test_crlf_migration_does_not_change_normalized_scoring_code_identity(frozen_fixture):
    root, path, _, _, _, _, _ = frozen_fixture
    for name in ("matching.py", "adaptive_corrected.py"):
        code = root / name
        code.write_bytes(code.read_bytes().replace(b"\n", b"\r\n"))
    assert load_corrected_scorer(path, project_root=root) is not None


@pytest.mark.parametrize("change,match", [
    ({"schema_version": 2}, "configuration schema"),
    ({"method_specification": {}}, "specification differs"),
    ({"provenance": {}}, "Incomplete"),
])
def test_incompatible_configuration_rejected(frozen_fixture, change, match):
    root, path, _, config, _, _, _ = frozen_fixture
    config.update(change)
    save_config(path, config)
    with pytest.raises(ValueError, match=match):
        load_corrected_scorer(path, project_root=root)


@pytest.mark.parametrize("target", ["train_sha256", "method_specification", "selected", "schema_version"])
def test_configuration_and_bundle_metadata_must_agree(frozen_fixture, target):
    root, path, bundle_path, config, bundle, _, _ = frozen_fixture
    bundle[target] = {"changed": True} if target not in {"train_sha256", "schema_version"} else (
        "b" * 64 if target == "train_sha256" else 2)
    joblib.dump(bundle, bundle_path)
    config["artifacts"]["model_bundle"]["sha256"] = sha(bundle_path)
    save_config(path, config)
    with pytest.raises(ValueError):
        load_corrected_scorer(path, project_root=root)


def test_artifact_and_configuration_cannot_escape_project_root(frozen_fixture):
    root, path, _, config, _, _, _ = frozen_fixture
    config["artifacts"]["model_bundle"]["path"] = "../outside.joblib"
    save_config(path, config)
    with pytest.raises(ValueError, match="inside the project root"):
        load_corrected_scorer(path, project_root=root)
    with pytest.raises(ValueError, match="inside the project root"):
        load_corrected_scorer(root.parent / "external.json", project_root=root)


def fpv_path(root, catalog):
    path = root / "offers.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OFFER_COLUMNS)
        writer.writeheader()
        for offer in catalog.values():
            row = dict.fromkeys(OFFER_COLUMNS, "")
            row.update(offer_id=offer.offer_id, raw_title=offer.title,
                       brand=offer.brand, store=offer.source,
                       source_url=f"https://example.com/{offer.offer_id}", record_status="single_variant")
            writer.writerow(row)
    return path


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_corrected_loads_frozen_configuration_but_never_fits_pair_features(frozen_fixture, monkeypatch, capsys, explicit):
    root, path, _, _, _, catalog, _ = frozen_fixture
    offers_path = fpv_path(root, catalog)
    monkeypatch.setattr(search_cli, "PROJECT_DIRECTORY", root)

    def forbidden(*args, **kwargs):
        raise AssertionError("Query attempted to fit pairwise features")

    monkeypatch.setattr(FeatureExtractor, "fit", forbidden)
    args = ["--format", "fpv", "--catalog", str(offers_path), "--query-id", "query",
            "--method", "adaptive-corrected", "--json"]
    if explicit:
        args += ["--method-config", str(path)]
    search_cli.main(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["ranking"]["method"] == "adaptive-corrected"
    assert payload["verification"]["gold_labels_used"] is False
    assert payload["verification"]["search_metrics_computed"] is False
    assert all(row["offer"]["offer_id"] != "query" for row in payload["ranking"]["results"])


def test_cli_rejects_method_config_for_unrelated_mode(frozen_fixture, capsys):
    root, path, _, _, _, catalog, _ = frozen_fixture
    offers_path = fpv_path(root, catalog)
    with pytest.raises(SystemExit) as error:
        search_cli.main(["--format", "fpv", "--catalog", str(offers_path), "--query-id", "query",
                         "--method", "renorm", "--method-config", str(path)])
    assert error.value.code == 2
    assert "only supported" in capsys.readouterr().err
