"""Minimal fixed-form correction of grouped Adaptive Evidence-Aware Similarity v1.

S_corr = C * M_q, with gamma=1, prior=0 and identifier penalty=0.
TRAIN informativeness must be supplied explicitly. Historical matching.py is
unchanged. Scores are heuristic similarities, not calibrated probabilities.
"""
from __future__ import annotations

import numpy as np

from matching import (FIELDS, GROUP_CHANNEL_WEIGHTS, GROUP_WEIGHTS,
                      FeatureExtractor, evidence_scores)


_FIXED_RELIABILITY = np.array([1.0, 1.0, 1.0, 0.70], dtype=float)
_FIXED_RELIABILITY.setflags(write=False)
METHOD_SPECIFICATION = {
    "method_id": "adaptive_corrected",
    "name": "Adaptive v1 corrected",
    "version": "adaptive-v1-corrected-1",
    "formula": "S_corr = C * M_q; no available group -> 0",
    "gamma": 1.0,
    "prior": 0.0,
    "identifier_penalty": 0.0,
    "field_order": list(FIELDS),
    "groups": [["title", "identifier"], ["description"], ["brand"]],
    "group_weights": GROUP_WEIGHTS.tolist(),
    "group_channel_weights": [value.tolist() for value in GROUP_CHANNEL_WEIGHTS],
    "manual_reliability": _FIXED_RELIABILITY.tolist(),
    "quality": "q_i = I_i * r_i; r is inherited manual reliability, not empirically established accuracy",
    "coverage": "Sum of available raw-group budgets; title OR derived identifier uses one title-group budget",
    "informativeness": {
        "source": "matching.train_informativeness",
        "formula": "clip(0.5 + 2 * (positive_mean - negative_mean), 0.25, 1.5)",
        "minimum_observed_per_class": 3,
        "insufficient_observations_fallback": 1.0,
        "fit_policy": "TRAIN fit portion only; refit inside every development training fold",
    },
    "threshold_policy": "Threshold is external; select only on permitted inner development predictions",
    "no_evidence_score": 0.0,
    "bound": "0 <= score <= coverage <= 1",
    "global_masking_monotonicity_claimed": False,
    "calibration_established": False,
}


def _validated_informativeness(informativeness: np.ndarray) -> np.ndarray:
    if informativeness is None:
        raise ValueError("Explicit TRAIN informativeness is required")
    values = np.asarray(informativeness, dtype=float)
    if values.shape != (len(FIELDS),) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"Informativeness must contain {len(FIELDS)} finite positive TRAIN factors")
    return values.copy()


def corrected_scores(matrix: np.ndarray, informativeness: np.ndarray) -> np.ndarray:
    """Score existing similarities with the one predeclared corrected formula.

    There are deliberately no gamma, prior, penalty, or reliability arguments.
    Missing values are NaN; observed zeros remain available disagreement. The
    quality factors must already have been estimated on allowed TRAIN data.
    """
    info = _validated_informativeness(informativeness)
    return evidence_scores(
        matrix, "adaptive", info,
        gamma=1.0, prior=0.0, penalty=0.0,
        use_info=True, use_reliability=True, use_coverage=True,
        reliability=_FIXED_RELIABILITY,
    )


class FrozenCorrectedScorer:
    """Search extension using a fitted extractor and frozen TRAIN information.

    Construction validates fittedness; calls only transform and score thereafter.
    The caller must preserve the matching fitted vocabulary/IDF and provenance.
    This object never estimates informativeness or fits on a user query.
    """

    def __init__(self, extractor: FeatureExtractor, informativeness: np.ndarray) -> None:
        extractor.transform([])
        self.extractor = extractor
        self._informativeness = _validated_informativeness(informativeness)
        self._informativeness.setflags(write=False)

    @property
    def informativeness(self) -> np.ndarray:
        """Return an independent read-only view of the frozen factors."""
        values = self._informativeness.copy()
        values.setflags(write=False)
        return values

    def score_pairs(self, pairs: list[dict]) -> np.ndarray:
        return corrected_scores(self.extractor.transform(pairs).matrix, self._informativeness)
