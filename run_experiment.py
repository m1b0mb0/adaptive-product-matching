"""WDC comparison, validation-only tuning, blind test evaluation.

WARNING: results will depend on the chosen official WDC variant.
The proposed score is preliminary and must not be presented as established novel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from audit_wdc import audit, read_pairs
from matching import FeatureExtractor, masked, scores, train_informativeness

METHODS = ("text", "fixed", "renorm", "adaptive", "adaptive_no_info", "adaptive_no_coverage", "adaptive_no_penalty")
THRESHOLDS = np.linspace(0.01, 0.99, 99)


def labels(pairs: list[dict]) -> np.ndarray:
    return np.array([int(x["label"]) for x in pairs], dtype=int)


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel().tolist()
    # Preserve precision for validation selection. Rounding here can turn a
    # strictly worse F1 into a tie and select a worse decision threshold.
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_threshold(y: np.ndarray, values: np.ndarray) -> tuple[float, dict]:
    if not len(y) or y.shape != values.shape or y.ndim != 1:
        raise ValueError("Threshold selection requires non-empty, aligned 1D arrays")
    if not np.isfinite(values).all():
        raise ValueError("Validation scores must be finite")
    options = [(metrics(y, (values >= t).astype(int)), float(t)) for t in THRESHOLDS]
    # Maximize F1; break ties in favor of precision, then higher threshold.
    metric, thresh = max(options, key=lambda item: (item[0]["f1"], item[0]["precision"], item[1]))
    return thresh, metric


def evaluate(train: list[dict], valid: list[dict], test: list[dict]) -> dict:
    schema = audit(train, valid, test)
    if schema.get("integrity_errors"):
        raise ValueError(f"Dataset integrity errors: {schema['integrity_errors']}")
    if any(schema["offer_id_overlaps"].values()):
        raise ValueError("Offer ID overlap between splits: use matching WDC splits")
    if any(schema.get("canonical_pair_overlaps", {}).values()):
        raise ValueError("Duplicate offer pairs across splits: use matching WDC splits")
    extractor = FeatureExtractor().fit(train)
    train_f = extractor.transform(train)
    valid_f = extractor.transform(valid)
    test_f = extractor.transform(test)
    info = train_informativeness(train_f.matrix, labels(train))
    y_valid, y_test = labels(valid), labels(test)
    chosen: dict[str, dict] = {}
    test_results = {}
    for method in METHODS:
        if method == "adaptive":
            candidates = [{"gamma": g, "penalty": p}
                          for g in (0.5, 1.0, 1.5) for p in (0.0, 0.10, 0.25)]
        elif method.startswith("adaptive_"):
            # Ablate exactly one component; inherit other choices from the
            # validation-selected full model. Re-tune only decision threshold.
            candidates = [{"gamma": chosen["adaptive"]["gamma"],
                           "penalty": chosen["adaptive"]["penalty"]}]
        else:
            candidates = [{"gamma": 1.0, "penalty": 0.0}]
        possibilities = []
        for params in candidates:
            val_score = scores(valid_f.matrix, method, info, **params)
            threshold, val_metrics = best_threshold(y_valid, val_score)
            possibilities.append((val_metrics, threshold, params))
        val_met, threshold, params = max(
            possibilities, key=lambda x: (x[0]["f1"], x[0]["precision"], x[1]))
        chosen[method] = {"threshold": threshold, **params, "validation": val_met}
        test_score = scores(test_f.matrix, method, info, **params)
        tm = metrics(y_test, (test_score >= threshold).astype(int))
        hard_index = [i for i, p in enumerate(test)
                      if int(p["label"]) == 0 and p.get("is_hard_negative") is True]
        tm["hard_negative_fpr"] = (round(float(np.mean(test_score[hard_index] >= threshold)), 5)
                                    if hard_index else None)
        test_results[method] = tm

    robustness = []
    for rate in (0, .10, .20, .30, .50):
        for seed in (11, 22, 33):
            corrupted = masked(test_f, rate, seed)
            for method in METHODS:
                config = chosen[method]
                values = scores(corrupted, method, info, gamma=config["gamma"],
                                penalty=config["penalty"])
                record = metrics(y_test, (values >= config["threshold"]).astype(int))
                robustness.append({"missing_rate": rate, "seed": seed,
                                   "method": method, **record})
    return {"note": "Test data not used to fit vectorizers, weights or thresholds.",
            "schema": schema, "fields": ["title", "description", "brand", "identifier_heuristic"],
            "train_discriminativeness": dict(zip(["title", "description", "brand", "identifier"],
                                                [float(x) for x in info])),
            "validation_selected": chosen, "test_metrics": test_results,
            "missingness": {"mechanism": "offer-level, independent masking of brand and description only",
                            "rates": [0, .10, .20, .30, .50], "seeds": [11, 22, 33],
                            "rows": robustness},
            "limitations": ["Model identifiers are extracted heuristically from titles.",
                            "These data do not provide explicit FPV motor technical attributes.",
                            "Labels in WDC Products may contain annotation noise.",
                            "Reported metrics apply to preassembled candidate pairs, not end-to-end search recall."]}


def main() -> None:
    p = argparse.ArgumentParser()
    for split in ("train", "valid", "test"):
        p.add_argument(f"--{split}", required=True)
    p.add_argument("--out", default="results.json")
    args = p.parse_args()
    result = evaluate(read_pairs(args.train), read_pairs(args.valid), read_pairs(args.test))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, stat in result["test_metrics"].items():
        print(f"{name:22s} P={stat['precision']:.3f} R={stat['recall']:.3f} F1={stat['f1']:.3f} ")
    print(f"Full, inspectable output saved to {args.out}")


if __name__ == "__main__":
    main()
