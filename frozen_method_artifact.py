"""Load the locally frozen corrected-method bundle without training on queries.

Only project-local, checksum-pinned artifacts are supported. A joblib bundle
contains executable Python objects and is a trusted local build artifact, not
an upload/import format for arbitrary third-party models.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
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


def _read_configuration(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read frozen corrected configuration: {error}") from error


def _parse_configuration(payload: bytes) -> dict:
    try:
        config = json.loads(payload.decode("utf-8"))
    except ValueError as error:
        raise ValueError(f"Cannot read frozen corrected configuration: {error}") from error
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported frozen configuration schema")
    if config.get("method_specification") != METHOD_SPECIFICATION:
        raise ValueError("Frozen corrected method specification differs from implementation")
    return config


def _bundle_reference(config: dict, root: Path) -> tuple[Path, str]:
    try:
        artifact = config["artifacts"]["model_bundle"]
        return _inside_root(root, artifact["path"], "Model bundle path"), artifact["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete frozen corrected configuration") from error


def _train_digest(config: dict) -> str:
    try:
        digest = config["provenance"]["train_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete frozen corrected configuration") from error
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Frozen TRAIN digest is missing or invalid")
    return digest


def _load_verified_bundle(path: Path, expected_hash: str) -> object:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("Frozen model bundle checksum mismatch")
    try:
        return joblib.load(BytesIO(payload))
    except Exception as error:
        raise ValueError(f"Cannot deserialize frozen local model bundle: {error}") from error


def _scorer_from_bundle(config: dict, bundle: object, expected_train: str) -> FrozenCorrectedScorer:
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


def _load_strict_corrected_scorer(
    config: dict, root: Path, bundle_reference: tuple[Path, str],
) -> FrozenCorrectedScorer:
    expected_train = _train_digest(config)
    try:
        code_hashes = config["provenance"]["code_sha256_lf"]
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete frozen corrected configuration") from error
    if not isinstance(code_hashes, dict) or not {"matching.py", "adaptive_corrected.py"} <= code_hashes.keys():
        raise ValueError("Frozen scoring code digests are missing")
    for filename, expected in code_hashes.items():
        code_path = _inside_root(root, filename, "Scoring code path")
        if _sha256(code_path, normalize_lf=True) != expected:
            raise ValueError(f"Frozen scoring code checksum mismatch: {filename}")
    bundle = _load_verified_bundle(*bundle_reference)
    return _scorer_from_bundle(config, bundle, expected_train)


def load_corrected_scorer(configuration_path: str | Path,
                          project_root: str | Path | None = None) -> FrozenCorrectedScorer:
    """Load a trusted local configuration and check every declared code hash."""
    root = (Path(__file__).resolve().parent if project_root is None
            else Path(project_root).resolve())
    config_path = _inside_root(root, configuration_path, "Configuration path")
    config = _parse_configuration(_read_configuration(config_path))
    return _load_strict_corrected_scorer(config, root, _bundle_reference(config, root))
