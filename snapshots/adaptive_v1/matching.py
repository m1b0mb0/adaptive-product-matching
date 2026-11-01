"""Initial research prototype. NOT a validated scientific result.

Pairwise entity matching on WDC Products. The proxy identifier is deliberately
heuristic; its errors must be audited before interpreting measured scores.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

FIELDS = ("title", "description", "brand", "identifier")
WEIGHTS = np.array([0.40, 0.15, 0.15, 0.30], dtype=float)
RELIABILITY = np.array([1.0, 1.0, 1.0, 0.70], dtype=float)
# The identifier is extracted from the title: these are two measurements of
# one evidence family, not two independent sources of available information.
EVIDENCE_GROUPS = ((0, 3), (1,), (2,))
GROUP_WEIGHTS = np.array([0.70, 0.15, 0.15], dtype=float)
GROUP_CHANNEL_WEIGHTS = (np.array([4 / 7, 3 / 7]), np.ones(1), np.ones(1))
MODEL_TOKEN = re.compile(r"(?<!\w)[a-z0-9][a-z0-9_.\-]{3,}(?!\w)")
UNIT_VALUE = re.compile(r"^\d+(?:\.\d+)?(?:gb|tb|mb|kb|kv|hz|mhz|ghz|mm|cm|mah|wh|v|w|fps|mp)$")
COMMON_TECH = {"ddr3", "ddr3l", "ddr4", "ddr5", "usb2", "usb3", "wifi5", "wifi6", "wifi6e"}


def norm(value: object) -> str:
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def identifiers(title: object) -> set[str]:
    """Extract likely model-like tokens; may mistake generic specs for model IDs."""
    result = set()
    for token in MODEL_TOKEN.findall(norm(title)):
        canon = re.sub(r"[._\-]", "", token)
        if (len(canon) >= 5 and re.search(r"[a-z]", canon)
                and re.search(r"\d", canon) and canon not in COMMON_TECH
                and not UNIT_VALUE.fullmatch(canon)):
            result.add(canon)
    return result


def pair_field(pair: dict, field: str, side: str) -> object:
    return pair.get(f"{field}_{side}")


def train_texts(pairs: list[dict], field: str) -> list[str]:
    """Use only training offers to fit TF-IDF; de-duplicate by offer ID."""
    seen: dict[tuple, str] = {}
    for i, pair in enumerate(pairs):
        for side in ("left", "right"):
            offer_id = pair.get(f"id_{side}")
            # Missing IDs cannot identify a shared offer. Keep those rows
            # separate; the official-data reader requires non-missing IDs.
            key = ("id", str(offer_id)) if norm(offer_id) else ("row", i, side)
            seen.setdefault(key, norm(pair_field(pair, field, side)))
    return [s for s in seen.values() if s]


@dataclass
class Features:
    matrix: np.ndarray
    ids_left: list[str]
    ids_right: list[str]


class FeatureExtractor:
    def __init__(self) -> None:
        self.vectorizers: dict[str, TfidfVectorizer | None] = {}

    def fit(self, train_pairs: list[dict]) -> "FeatureExtractor":
        if not train_pairs:
            raise ValueError("Cannot fit feature extractor on an empty training split")
        self.vectorizers = {}
        for field in ("title", "description"):
            texts = train_texts(train_pairs, field)
            if not any(len(text) >= 2 for text in texts):
                # Character 2-4-grams cannot encode empty or one-character
                # texts. Do not fit a synthetic fallback vocabulary.
                self.vectorizers[field] = None
                continue
            vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                  max_features=30000, sublinear_tf=True)
            vec.fit(texts)
            self.vectorizers[field] = vec
        return self

    def transform(self, pairs: list[dict]) -> Features:
        if not all(field in self.vectorizers for field in ("title", "description")):
            raise ValueError("Fit the feature extractor before transforming pairs")
        n = len(pairs)
        matrix = np.full((n, len(FIELDS)), np.nan, dtype=float)
        for j, field in enumerate(("title", "description")):
            vec = self.vectorizers[field]
            if not n or vec is None:
                continue
            left = [norm(pair_field(p, field, "left")) for p in pairs]
            right = [norm(pair_field(p, field, "right")) for p in pairs]
            # Vectorizer normalizes rows by default, so a row-wise product = cosine.
            lv, rv = vec.transform(left), vec.transform(right)
            similarities = np.asarray(lv.multiply(rv).sum(axis=1)).ravel()
            # The exact cosine of nonnegative L2-normalized TF-IDF lies in
            # [0, 1]; sparse floating-point sums can exceed 1 by a few ulps.
            similarities = np.clip(similarities, 0.0, 1.0)
            # A zero vector has no train-vocabulary evidence; its cosine is
            # undefined, rather than an observed disagreement of zero.
            comparable = (lv.getnnz(axis=1) > 0) & (rv.getnnz(axis=1) > 0)
            matrix[comparable, j] = similarities[comparable]

        for i, p in enumerate(pairs):
            brand_l, brand_r = norm(p.get("brand_left")), norm(p.get("brand_right"))
            if brand_l and brand_r:
                matrix[i, 2] = float(brand_l == brand_r)
            code_l, code_r = identifiers(p.get("title_left")), identifiers(p.get("title_right"))
            if code_l and code_r:
                matrix[i, 3] = len(code_l & code_r) / len(code_l | code_r)
        return Features(matrix, [str(p.get("id_left", f"left-{i}")) for i, p in enumerate(pairs)],
                        [str(p.get("id_right", f"right-{i}")) for i, p in enumerate(pairs)])


def train_informativeness(train_features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Empirical separation of positive vs negative pair similarity, TRAIN ONLY.

    This is field-level discriminativeness, not per-value information content.
    It is a provisional definition to be tested and refined.
    """
    answer = np.ones(len(FIELDS), dtype=float)
    for i in range(len(FIELDS)):
        positives = train_features[(labels == 1) & ~np.isnan(train_features[:, i]), i]
        negatives = train_features[(labels == 0) & ~np.isnan(train_features[:, i]), i]
        if len(positives) >= 3 and len(negatives) >= 3:
            separation = float(positives.mean() - negatives.mean())
            answer[i] = np.clip(0.5 + 2 * separation, 0.25, 1.5)
    return answer


def scores(features: np.ndarray, method: str, informativeness: np.ndarray | None = None,
           gamma: float = 1.0, penalty: float = 0.0, prior: float = 0.20) -> np.ndarray:
    """Compute pair scores, not calibrated match probabilities.

    Fixed: missing fields contribute zero (naive baseline).
    Renorm: observed-field weighted average (essential comparator).
    Adaptive: observed-field info/reliability weighting PLUS evidence coverage
    shrinkage toward a conservative prior, PLUS observed identifier conflicts.
    """
    if features.ndim != 2 or features.shape[1] != len(FIELDS):
        raise ValueError(f"Expected feature array with {len(FIELDS)} columns")
    present = ~np.isnan(features)
    filled = np.nan_to_num(features, nan=0.0)
    if method == "fixed":
        return np.clip((filled * WEIGHTS).sum(axis=1) / WEIGHTS.sum(), 0, 1)
    if method == "renorm":
        numerator = (filled * WEIGHTS).sum(axis=1)
        denominator = (present * WEIGHTS).sum(axis=1)
        return np.clip(np.divide(numerator, denominator, out=np.zeros_like(numerator),
                                 where=denominator > 0), 0, 1)
    if method in ("adaptive", "adaptive_no_coverage", "adaptive_no_info", "adaptive_no_penalty"):
        info = (np.ones(len(FIELDS)) if method == "adaptive_no_info"
                else np.asarray(informativeness if informativeness is not None
                                else np.ones(len(FIELDS)), dtype=float))
        effective = WEIGHTS * RELIABILITY * info
        observed_weights = (present * effective).sum(axis=1)
        numerator = (filled * effective).sum(axis=1)
        mean = np.divide(numerator, observed_weights, out=np.zeros_like(numerator),
                         where=observed_weights > 0)
        coverage = observed_weights / effective.sum()
        if method == "adaptive_no_coverage":
            result = mean
        else:
            alpha = np.power(coverage, gamma)
            result = alpha * mean + (1 - alpha) * prior
        # A model-token mismatch is not necessarily a true model contradiction;
        # therefore penalty is separately tuned on validation, including zero.
        contradiction = present[:, 3] & (filled[:, 3] == 0)
        applied_penalty = 0.0 if method == "adaptive_no_penalty" else penalty
        result = result - applied_penalty * contradiction
        # No common evidence must not be interpreted as a positive match.
        return np.where(observed_weights > 0, np.clip(result, 0, 1), 0.0)
    if method == "text":
        return filled[:, 0]
    raise ValueError(f"Unknown method: {method}")


def _evidence_array(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(FIELDS):
        raise ValueError(f"Expected feature array with {len(FIELDS)} columns")
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all() or np.any((observed < 0) | (observed > 1)):
        raise ValueError("Observed similarities must be finite and in [0, 1]")
    return values


def evidence_coverage(features: np.ndarray) -> np.ndarray:
    """Available raw-field group budget, independent of learned quality.

    The title family is available if title OR its derived identifier can be
    compared. Having both does not increase this family's availability budget.
    This is a coverage proxy, not a statistical confidence or independence claim.
    """
    values = _evidence_array(features)
    available = np.column_stack([
        (~np.isnan(values[:, group])).any(axis=1) for group in EVIDENCE_GROUPS
    ])
    return (available * GROUP_WEIGHTS).sum(axis=1)


def evidence_scores(features: np.ndarray, method: str,
                    informativeness: np.ndarray | None = None,
                    gamma: float = 1.0, penalty: float = 0.0,
                    prior: float = 0.20, *, use_info: bool = True,
                    use_reliability: bool = True, use_coverage: bool = True,
                    reliability: np.ndarray | None = None) -> np.ndarray:
    """Hierarchical scores used by the validation-only experiment protocol.

    All attribute methods share groups {title, identifier}, {description},
    {brand}, with budgets .70/.15/.15. Within a group, observed channels are
    renormalized. Fixed retains unavailable group budgets as zero; renorm uses
    only observed groups. Adaptive adds observed channel information/reliability
    factors and coverage shrinkage toward ``prior``. No evidence always scores 0.

    Informativeness must be estimated using TRAIN ONLY; reliability is a manual
    assumption, not estimated extraction accuracy. ``penalty`` is an optional
    extra identifier-conflict heuristic and may reuse the same evidence twice.
    Scores are not calibrated probabilities. The original ``scores`` function is
    preserved for historical reproducibility and is not silently redefined.
    """
    values = _evidence_array(features)
    valid_methods = {"text", "fixed", "renorm", "adaptive", "adaptive_no_info",
                     "adaptive_no_reliability", "adaptive_no_coverage",
                     "adaptive_no_penalty"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method: {method}")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and non-negative")
    if not np.isfinite(prior) or not 0 <= prior <= 1:
        raise ValueError("prior must be finite and in [0, 1]")

    present = ~np.isnan(values)
    filled = np.nan_to_num(values, nan=0.0)
    if method == "text":
        return filled[:, 0]
    adaptive = method.startswith("adaptive")
    use_info = use_info and method != "adaptive_no_info"
    use_reliability = use_reliability and method != "adaptive_no_reliability"
    use_coverage = use_coverage and method != "adaptive_no_coverage"
    quality = np.ones(len(FIELDS), dtype=float)
    for enabled, factors, name in (
        (adaptive and use_info, informativeness, "informativeness"),
        (adaptive and use_reliability,
         RELIABILITY if reliability is None else reliability, "reliability"),
    ):
        if enabled and factors is not None:
            factors = np.asarray(factors, dtype=float)
            if factors.shape != (len(FIELDS),) or not np.isfinite(factors).all() or np.any(factors <= 0):
                raise ValueError(f"{name} must contain {len(FIELDS)} finite positive factors")
            if name == "reliability" and np.any(factors > 1):
                raise ValueError("reliability factors must be at most 1")
            quality *= factors

    n = len(values)
    numerator = np.zeros(n)
    denominator = np.zeros(n)
    coverage = np.zeros(n)
    for group, budget, channel_weights in zip(
            EVIDENCE_GROUPS, GROUP_WEIGHTS, GROUP_CHANNEL_WEIGHTS):
        # z_g is the observed fraction of this group's channel weights. Divide
        # by it BEFORE pooling groups, so absence of the title-derived token
        # does not artificially remove part of the raw title-field budget.
        observed_channel_weights = present[:, group] * channel_weights
        z = observed_channel_weights.sum(axis=1)
        available = z > 0
        weighted = observed_channel_weights * quality[list(group)]
        # U_g * h_g = W_g * sum(b_i m_i q_i s_i) / z_g;
        # U_g       = W_g * sum(b_i m_i q_i) / z_g.
        group_numerator = (weighted * filled[:, group]).sum(axis=1)
        group_quality = weighted.sum(axis=1)
        numerator += budget * np.divide(
            group_numerator, z, out=np.zeros(n), where=available)
        denominator += budget * np.divide(
            group_quality, z, out=np.zeros(n), where=available)
        coverage += budget * available

    if method == "fixed":
        return np.clip(numerator / GROUP_WEIGHTS.sum(), 0, 1)
    mean = np.divide(numerator, denominator, out=np.zeros(n), where=denominator > 0)
    if method == "renorm":
        return np.clip(mean, 0, 1)
    result = mean
    if use_coverage:
        alpha = np.power(coverage / GROUP_WEIGHTS.sum(), gamma)
        result = alpha * mean + (1 - alpha) * prior
    if method != "adaptive_no_penalty":
        result = result - penalty * (present[:, 3] & (filled[:, 3] == 0))
    return np.where(denominator > 0, np.clip(result, 0, 1), 0.0)


def masked(features: Features, missing_rate: float, seed: int) -> np.ndarray:
    """Mask brand/description at OFFER level consistently across all pairs.

    Identifiers come from titles and cannot be separately masked without
    changing titles. Thus these tests are about brand/description missingness.
    """
    out = features.matrix.copy()
    if not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate must be in [0, 1]")
    for j in (1, 2):  # description, brand only
        name = FIELDS[j]
        for i, (left, right) in enumerate(zip(features.ids_left, features.ids_right)):
            def dropped(offer_id: str) -> bool:
                key = f"{seed}:{name}:{offer_id}".encode()
                # Deterministic uniform pseudo-random float for stable splits.
                x = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
                return x < missing_rate
            if dropped(left) or dropped(right):
                out[i, j] = np.nan
    return out
