"""Protect the archived implementation and its connection to historical results."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "snapshots" / "adaptive_v1"


def test_v1_snapshot_bytes_match_manifest_and_original_result_provenance():
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    result_path = ROOT / manifest["source_results"]
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == manifest["source_results_sha256"]
    results = json.loads(result_path.read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == info["sha256"]
        if info["original_path"] in results["metadata"]["source_sha256"]:
            assert info["sha256"] == results["metadata"]["source_sha256"][info["original_path"]]


def test_v1_snapshot_has_frozen_grouped_score_and_prior_behavior(monkeypatch):
    name = "archived_adaptive_v1_matching"
    spec = importlib.util.spec_from_file_location(name, SNAPSHOT / "matching.py")
    archived = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, archived)
    spec.loader.exec_module(archived)
    data = np.array([[.8, np.nan, np.nan, np.nan],
                     [.1, np.nan, np.nan, np.nan]])
    # Pin historical behavior, including upward shrinkage below the .20 anchor.
    actual = archived.evidence_scores(data, "adaptive", gamma=.5)
    expected = np.sqrt(.7) * data[:, 0] + (1 - np.sqrt(.7)) * .2
    np.testing.assert_allclose(actual, expected)
    assert actual[1] > .1
    assert archived.evidence_coverage(data).tolist() == [.7, .7]
