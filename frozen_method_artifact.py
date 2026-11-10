"""Load the locally frozen corrected-method bundle without training on queries.

Only project-local, checksum-pinned artifacts are supported. A joblib bundle
contains executable Python objects and is a trusted local build artifact, not
an upload/import format for arbitrary third-party models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from adaptive_corrected import METHOD_SPECIFICATION, FrozenCorrectedScorer


def _inside_root(root: Path, value: str | Path, role: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{role} must remain inside the project root")
    return candidate


def _sha256(path: Path, *, normalize_lf: bool = False) -> str:
    payload = path.read_bytes()
    if normalize_lf:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def load_corrected_scorer(configuration_path: str | Path,
                          project_root: str | Path | None = None) -> FrozenCorrectedScorer:
    """Verify the frozen configuration and return a transform/score-only object.

    This function never reads TRAIN, validation, TEST, or catalogue labels, and
    never calls fit. It checks the recorded TRAIN digest only as bundle metadata.
    """
    root = (Path(__file__).resolve().parent if project_root is None
            else Path(project_root).resolve())
    config_path = _inside_root(root, configuration_path, "Configuration path")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read frozen corrected configuration: {error}") from error
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported frozen configuration schema")
    if config.get("method_specification") != METHOD_SPECIFICATION:
        raise ValueError("Frozen corrected method specification differs from implementation")
    try:
        artifact = config["artifacts"]["model_bundle"]
        bundle_path = _inside_root(root, artifact["path"], "Model bundle path")
        expected_hash = artifact["sha256"]
        provenance = config["provenance"]
        expected_train = provenance["train_sha256"]
        code_hashes = provenance["code_sha256_lf"]
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete frozen corrected configuration") from error
    if not isinstance(expected_train, str) or len(expected_train) != 64:
        raise ValueError("Frozen TRAIN digest is missing or invalid")
    if not isinstance(code_hashes, dict) or not {"matching.py", "adaptive_corrected.py"} <= code_hashes.keys():
        raise ValueError("Frozen scoring code digests are missing")
    for filename, expected in code_hashes.items():
        code_path = _inside_root(root, filename, "Scoring code path")
        if _sha256(code_path, normalize_lf=True) != expected:
            raise ValueError(f"Frozen scoring code checksum mismatch: {filename}")
    if _sha256(bundle_path) != expected_hash:
        raise ValueError("Frozen model bundle checksum mismatch")
    try:
        bundle = joblib.load(bundle_path)
    except Exception as error:
        raise ValueError(f"Cannot deserialize frozen local model bundle: {error}") from error
    if not isinstance(bundle, dict) or type(bundle.get("schema_version")) is not int or bundle["schema_version"] != 1:
        raise ValueError("Unsupported frozen model bundle schema")
    if bundle.get("train_sha256") != expected_train:
        raise ValueError("Frozen TRAIN metadata differs between configuration and bundle")
    if bundle.get("method_specification") != METHOD_SPECIFICATION:
        raise ValueError("Frozen model bundle has a different corrected specification")
    if config.get("final_models") != bundle.get("selected"):
        raise ValueError("Frozen selected configurations differ between configuration and bundle")
    try:
        return FrozenCorrectedScorer(bundle["extractor"], bundle["informativeness"])
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"Invalid frozen corrected scorer components: {error}") from error
