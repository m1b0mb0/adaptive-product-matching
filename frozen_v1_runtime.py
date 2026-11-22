"""Load the original corrected scorer for the application runtime."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import adaptive_corrected
from frozen_method_artifact import (
    _bundle_reference,
    _inside_root,
    _load_strict_corrected_scorer,
    _load_verified_bundle,
    _parse_configuration,
    _read_configuration,
    _scorer_from_bundle,
    _sha256,
    _train_digest,
)
import logistic_baseline
import matching


_OFFICIAL_CONFIG = Path("docs/final_method/frozen_configuration.json")
_OFFICIAL_CONFIG_SHA256 = "2976e18752a2463a7ef784714d5468a150db0a9e0142812d471e3fc13b0c655a"
_OFFICIAL_BUNDLE = Path("data/processed/final_method/frozen_models.joblib")
_OFFICIAL_BUNDLE_SHA256 = "329fa0919f570172956b3fe60aa97dce68357f8189370ea317f3221487ff5caa"


def _verify_runtime(provenance: dict) -> None:
    if platform.python_version() != provenance["python"]:
        raise ValueError(f"Frozen model Python version mismatch (requires {provenance['python']})")
    packages = provenance["packages"]
    required = {name: packages[name] for name in ("numpy", "pandas", "scipy", "scikit-learn")}
    required["joblib"] = "1.6.0"  # Version used by requirements-lock.txt.
    for name, expected in required.items():
        try:
            installed = version(name)
        except PackageNotFoundError as error:
            raise ValueError(f"Frozen model requires {name}=={expected}") from error
        if installed != expected:
            raise ValueError(f"Frozen model dependency version mismatch ({name} requires {expected})")

    code_hashes = provenance["code_sha256_lf"]
    for module in (matching, adaptive_corrected, logistic_baseline):
        name = f"{module.__name__}.py"
        if _sha256(Path(module.__file__), normalize_lf=True) != code_hashes[name]:
            raise ValueError(f"Frozen scoring code checksum mismatch: {name}")


def _verify_fitted_components(config: dict, bundle: dict) -> None:
    expected = config["full_train_fit"]
    if not np.array_equal(bundle["informativeness"], expected["informativeness"]):
        raise ValueError("Frozen TRAIN informativeness mismatch")

    extractor = bundle["extractor"]
    logistic = bundle["logistic"]
    if not isinstance(extractor, matching.FeatureExtractor):
        raise ValueError("Invalid frozen feature extractor")
    if not isinstance(logistic, logistic_baseline.SimilarityLogisticRegression):
        raise ValueError("Invalid frozen logistic model")
    if logistic.metadata() != expected["logistic"]:
        raise ValueError("Frozen logistic metadata mismatch")

    vectorizers = expected["representation"]["vectorizers"]
    if extractor.vectorizers.keys() != vectorizers.keys():
        raise ValueError("Frozen TF-IDF fields mismatch")
    for field, reference in vectorizers.items():
        vectorizer = extractor.vectorizers[field]
        if not isinstance(vectorizer, TfidfVectorizer):
            raise ValueError(f"Invalid frozen TF-IDF vectorizer: {field}")
        indices = {term: int(index) for term, index in vectorizer.vocabulary_.items()}
        vocabulary = json.dumps(indices, sort_keys=True).encode()
        if (len(indices) != reference["terms"]
                or hashlib.sha256(vocabulary).hexdigest() != reference["vocabulary_sha256"]
                or hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest() != reference["idf_sha256"]):
            raise ValueError(f"Frozen TF-IDF fitted state mismatch: {field}")


def load_corrected_scorer(
    configuration_path: str | Path | None = None, *, project_root: str | Path | None = None,
) -> adaptive_corrected.FrozenCorrectedScorer:
    """Load the pinned v1 scorer or an explicitly selected trusted local configuration."""
    root = Path(__file__).resolve().parent if project_root is None else Path(project_root).resolve()
    official_path = root / _OFFICIAL_CONFIG
    requested = official_path if configuration_path is None else root / configuration_path
    path = _inside_root(root, requested, "Configuration path")
    payload = _read_configuration(path)
    pinned = hashlib.sha256(payload).hexdigest() == _OFFICIAL_CONFIG_SHA256
    if not pinned and (Path(os.path.abspath(requested)) == official_path or path == official_path.resolve()):
        raise ValueError("Official frozen configuration checksum mismatch")

    config = _parse_configuration(payload)
    bundle_path, bundle_hash = _bundle_reference(config, root)
    if not pinned:
        artifact_path = config["artifacts"]["model_bundle"]["path"]
        official_bundle = root / _OFFICIAL_BUNDLE
        if (bundle_hash == _OFFICIAL_BUNDLE_SHA256
                or Path(os.path.abspath(root / artifact_path)) == official_bundle
                or bundle_path == official_bundle.resolve()):
            raise ValueError("Official frozen configuration checksum mismatch")
        return _load_strict_corrected_scorer(config, root, (bundle_path, bundle_hash))

    _verify_runtime(config["provenance"])
    bundle = _load_verified_bundle(bundle_path, bundle_hash)
    scorer = _scorer_from_bundle(config, bundle, _train_digest(config))
    try:
        _verify_fitted_components(config, bundle)
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError(f"Invalid frozen fitted components: {error}") from error
    return scorer
