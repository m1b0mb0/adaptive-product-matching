"""Run the original evaluation tests against their frozen source revision."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_REVISION = "8dcf5dbe717a9e69f422a8ebfd1b8f25dbb557f3"
HISTORICAL_TESTS = (
    "tests/test_final_evaluation.py",
    "tests/test_final_method_pipeline.py",
    "tests/test_v1_snapshot.py",
)
HISTORICAL_INPUTS = (
    "data/external/wdcproducts80cc20rnd000un_train_small.json.gz",
    "data/processed/final_method/frozen_models.joblib",
)


def test_original_evaluation_tests_run_in_pinned_checkout(tmp_path):
    checkout = tmp_path / "historical"
    checkout.mkdir()
    git_env = os.environ | {"GIT_INDEX_FILE": str(tmp_path / "historical.index")}
    subprocess.run(["git", "read-tree", HISTORICAL_REVISION], cwd=ROOT, env=git_env, check=True)
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "checkout-index", "--all",
         "--prefix=" + checkout.as_posix() + "/"],
        cwd=ROOT, env=git_env, check=True,
    )
    for relative_path in HISTORICAL_INPUTS:
        destination = checkout / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative_path, destination)

    report = tmp_path / "historical-pytest.xml"
    child_env = os.environ | {"PYTEST_ADDOPTS": "", "MPLCONFIGDIR": str(tmp_path / "matplotlib")}
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"--junitxml={report}", *HISTORICAL_TESTS],
        cwd=checkout, env=child_env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    tests = ET.parse(report).findall(".//testcase")
    assert len(tests) == 22, result.stdout
    for tag in ("skipped", "failure", "error"):
        assert all(test.find(tag) is None for test in tests), result.stdout
