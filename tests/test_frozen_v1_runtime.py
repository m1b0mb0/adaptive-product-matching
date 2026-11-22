import hashlib
import json
from pathlib import Path
import shutil

import joblib
import pytest

import frozen_v1_runtime as runtime
import search_cli
from product_catalog import ProductOffer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("docs/final_method/frozen_configuration.json")
BUNDLE = Path("data/processed/final_method/frozen_models.joblib")


@pytest.fixture
def deployment(tmp_path):
    for path in (CONFIG, BUNDLE):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    return tmp_path


@pytest.fixture
def no_deserialization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("joblib.load called before artifact verification")

    monkeypatch.setattr(joblib, "load", forbidden)


@pytest.mark.parametrize("path", [None, CONFIG, Path("copy.json")])
def test_official_bundle_loads_without_train_or_application_sources(deployment, path):
    if path == Path("copy.json"):
        shutil.copyfile(deployment / CONFIG, deployment / path)
    # The runtime checks the imported scoring modules, so no application source
    # files (or training data) need to be present in this artifact directory.
    scorer = runtime.load_corrected_scorer(path, project_root=deployment)
    assert scorer.score_pairs([{}]).tolist() == [0.0]


@pytest.mark.parametrize("path", [None, CONFIG, Path("copy.json")])
def test_modified_official_configuration_does_not_fall_back(deployment, path, no_deserialization):
    raw = (deployment / CONFIG).read_bytes()
    (deployment / (path or CONFIG)).write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="Official frozen configuration checksum mismatch"):
        runtime.load_corrected_scorer(path, project_root=deployment)


def test_noncanonical_config_cannot_load_official_bundle_from_a_copy(deployment, no_deserialization):
    config = json.loads((deployment / CONFIG).read_bytes())
    config["artifacts"]["model_bundle"]["path"] = "copied.joblib"
    shutil.copyfile(deployment / BUNDLE, deployment / "copied.joblib")
    (deployment / "copy.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Official frozen configuration checksum mismatch"):
        runtime.load_corrected_scorer("copy.json", project_root=deployment)


def test_corrupt_bundle_rejected_before_deserialization(deployment, no_deserialization):
    (deployment / BUNDLE).write_bytes(b"damaged bundle")
    with pytest.raises(ValueError, match="bundle checksum mismatch"):
        runtime.load_corrected_scorer(project_root=deployment)


def test_cli_reports_official_configuration_failure(deployment, no_deserialization, monkeypatch, capsys):
    (deployment / CONFIG).write_bytes(b"invalid")
    monkeypatch.setattr(search_cli, "PROJECT_DIRECTORY", deployment)
    monkeypatch.setattr(search_cli, "load_fpv_catalog", lambda _: {"query": ProductOffer("query")})
    with pytest.raises(SystemExit) as error:
        search_cli.main(["--catalog-format", "fpv", "--catalog", "offers.csv", "--query-id", "query",
                         "--method", "adaptive-corrected", "--json"])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Помилка пошуку: Official frozen configuration checksum mismatch\n"


def test_loader_deserializes_snapshot_after_on_disk_bundle_replaced(deployment, monkeypatch):
    bundle = deployment / BUNDLE
    expected = hashlib.sha256(bundle.read_bytes()).hexdigest()
    original_load = joblib.load

    def replace_file_then_load(stream):
        bundle.write_bytes(b"replacement")
        assert hashlib.sha256(stream.getvalue()).hexdigest() == expected
        return original_load(stream)

    monkeypatch.setattr(joblib, "load", replace_file_then_load)
    scorer = runtime.load_corrected_scorer(project_root=deployment)
    assert scorer.score_pairs([{}]).tolist() == [0.0]


def test_python_version_mismatch_rejected_before_deserialization(deployment, no_deserialization, monkeypatch):
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.12.0")
    with pytest.raises(ValueError, match="Python version mismatch"):
        runtime.load_corrected_scorer(project_root=deployment)


@pytest.mark.parametrize("dependency", ["scikit-learn", "joblib"])
def test_dependency_mismatch_rejected_before_deserialization(
    deployment, no_deserialization, monkeypatch, dependency,
):
    installed_version = runtime.version
    monkeypatch.setattr(runtime, "version", lambda name: "0.0" if name == dependency else installed_version(name))
    with pytest.raises(ValueError, match=f"dependency version mismatch.*{dependency}"):
        runtime.load_corrected_scorer(project_root=deployment)


@pytest.mark.parametrize("module", [runtime.matching, runtime.adaptive_corrected, runtime.logistic_baseline])
def test_changed_imported_scoring_source_rejected_before_deserialization(
    deployment, no_deserialization, monkeypatch, module,
):
    changed_source = deployment / f"{module.__name__}.py"
    changed_source.write_bytes(Path(module.__file__).read_bytes() + b"\n# changed")
    monkeypatch.setattr(module, "__file__", str(changed_source))
    with pytest.raises(ValueError, match=f"code checksum mismatch: {module.__name__}"):
        runtime.load_corrected_scorer(project_root=deployment)


@pytest.mark.parametrize(("component", "message"), [
    ("schema_version", "bundle schema"),
    ("selected", "selected configurations"),
    ("informativeness", "TRAIN informativeness"),
    ("logistic", "logistic model"),
    ("extractor", "TF-IDF fitted state"),
])
def test_loaded_fitted_state_must_match_config(deployment, monkeypatch, component, message):
    original_load = joblib.load

    def load_with_invalid_state(stream):
        bundle = original_load(stream)
        if component == "extractor":
            bundle["extractor"].vectorizers["title"].vocabulary_.clear()
        else:
            bundle[component] = None
        return bundle

    monkeypatch.setattr(joblib, "load", load_with_invalid_state)
    with pytest.raises(ValueError, match=message):
        runtime.load_corrected_scorer(project_root=deployment)
