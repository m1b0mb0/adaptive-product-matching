"""Replay saved JSON demonstrations with local WDC TRAIN and FPV offers."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/search_system"
RUNS = json.loads((DOCS / "demo_manifest.json").read_text(encoding="utf-8"))["runs"]


def compare_cli_json(args, expected_path):
    result = subprocess.run(
        [sys.executable, "-B", *args], cwd=ROOT, check=True, capture_output=True, timeout=60,
    )
    assert json.loads(result.stdout) == json.loads(expected_path.read_bytes())


@pytest.mark.parametrize("run", RUNS, ids=[run["name"] for run in RUNS])
def test_saved_catalog_demonstration(run):
    compare_cli_json(run["command"].split()[1:], DOCS / run["json"])


@pytest.mark.parametrize("config_args", [[], ["--method-config", "docs/final_method/frozen_configuration.json"]],
                         ids=["default-config", "explicit-config"])
def test_saved_corrected_demonstration(config_args):
    compare_cli_json([
        "-X", "utf8", "search_cli.py", "--catalog-format", "wdc", "--catalog",
        "data/external/wdcproducts80cc20rnd000un_train_small.json.gz", "--query-id", "1017836",
        "--candidate-k", "20", "--top", "5", "--method", "adaptive-corrected", "--json", *config_args,
    ], ROOT / "docs/final_method/search_demo.json")
