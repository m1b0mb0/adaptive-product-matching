"""Supervised baseline on existing similarities and explicit availability.

No vocabulary, imputer statistics, scaler, or threshold is fitted here. Fit the
upstream FeatureExtractor on the same training fold before calling ``fit``.
``predict_score`` returns the fitted logistic model's positive-class output;
external calibration has not been established, especially under missingness
or domain shift. Missing every field still receives the learned intercept and
is not automatically rejected by this baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from matching import FIELDS, _evidence_array


DESIGN_FEATURE_NAMES = tuple(f"similarity_{field}" for field in FIELDS) + tuple(
    f"available_{field}" for field in FIELDS
)


def similarity_design_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return four zero-filled similarities followed by four availability bits.

    NaN is the sole missing-value marker. Observed similarities, including an
    observed zero, must be finite values in [0, 1]. For a missing similarity the
    corresponding availability bit is zero; for an observed zero it is one.
    This fixed encoding needs no training statistics and never mutates input.
    """
    values = _evidence_array(matrix)
    available = ~np.isnan(values)
    return np.column_stack((np.where(available, values, 0.0), available.astype(float)))


class SimilarityLogisticRegression:
    """L2-regularized logistic regression with an explicit missingness channel.

    Fold selection, validation thresholds, and all upstream statistical feature
    extraction belong to the experiment runner. ``C`` is inverse regularization
    strength; the development protocol selects it from 0.1, 1, and 10.
    """

    def __init__(self, C: float = 1.0) -> None:
        if isinstance(C, (bool, np.bool_)):
            raise ValueError("C must be finite and positive")
        try:
            self.C = float(C)
        except (TypeError, ValueError) as exc:
            raise ValueError("C must be finite and positive") from exc
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError("C must be finite and positive")
        self._model: LogisticRegression | None = None

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> "SimilarityLogisticRegression":
        design = similarity_design_matrix(matrix)
        target = np.asarray(labels)
        if target.ndim != 1 or len(target) != len(design):
            raise ValueError("labels must be a one-dimensional array matching the training rows")
        if target.dtype.kind not in "biuf" or not np.isin(target, (0, 1)).all():
            raise ValueError("labels must contain only numeric binary values 0 and 1")
        if len(target) == 0 or set(target.tolist()) != {0, 1}:
            raise ValueError("Training must contain both label classes 0 and 1")
        model = LogisticRegression(
            C=self.C,
            class_weight=None,
            solver="lbfgs",
            max_iter=2000,
            random_state=20261101,
        )
        model.fit(design, target.astype(int))
        self._model = model
        return self

    def predict_score(self, matrix: np.ndarray) -> np.ndarray:
        """Positive-class outputs without selecting or modifying a threshold."""
        model = self._fitted_model()
        design = similarity_design_matrix(matrix)
        if len(design) == 0:
            return np.empty(0, dtype=float)
        positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
        return model.predict_proba(design)[:, positive_column]

    def metadata(self) -> dict:
        """JSON-compatible fitted parameters for reproducible development runs."""
        model = self._fitted_model()
        return {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "feature_order": list(DESIGN_FEATURE_NAMES),
            "missing_encoding": "NaN -> similarity 0 and availability 0; observed zero -> availability 1",
            "C": self.C,
            "penalty": "l2",
            "class_weight": None,
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": 20261101,
            "scaling": None,
            "classes": model.classes_.astype(int).tolist(),
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "n_iter": model.n_iter_.astype(int).tolist(),
            "iteration_limit_reached": bool(np.any(model.n_iter_ >= model.max_iter)),
            "calibration_established": False,
        }

    def _fitted_model(self) -> LogisticRegression:
        if self._model is None:
            raise NotFittedError("Fit SimilarityLogisticRegression before predicting or reading fitted metadata")
        return self._model
