"""TRAIN-only nested entity-group development comparison; no TEST/FPV input.

An entity pair belongs to a partition only when BOTH endpoints belong there.
Crossing pairs are excluded, not assigned by only one of their two entities.
All learned representations and supervised quantities are fitted inside folds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from audit_wdc import offers, read_pairs, summary
from logistic_baseline import SimilarityLogisticRegression
from matching import (FIELDS, GROUP_CHANNEL_WEIGHTS, GROUP_WEIGHTS, RELIABILITY,
                      FeatureExtractor, Features, evidence_coverage,
                      evidence_scores, masked, norm, train_informativeness)
from run_experiment import THRESHOLDS, best_threshold, labels, metrics

METHODS = ("text", "fixed", "renorm", "adaptive", "logistic")
C_VALUES = (0.1, 1.0, 10.0)
GAMMAS = (0.5, 1.0, 1.5)
RATES = (0.0, 0.1, 0.2, 0.3, 0.5)
SEEDS = (11, 22, 33)
SCENARIOS = ("auxiliary", "all_groups")
PARTITION_SEED = 20261101
PRIOR = 0.20
TRAIN_PATH = Path("data/external/wdcproducts80cc20rnd000un_train_small.json.gz")
OUT_DIRECTORY = Path("docs/known_methods")
INTEGRITY_FIELDS = (
    "duplicate_rows", "duplicate_pair_ids", "pair_id_mismatches",
    "duplicate_canonical_pairs", "conflicting_label_pairs", "self_pairs",
    "label_cluster_mismatches", "inconsistent_offer_payloads", "positive_hard_flag_true",
)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cluster(pair: dict, side: str) -> str:
    value = pair.get(f"cluster_id_{side}")
    if isinstance(value, bool) or value is None or not str(value).strip():
        raise ValueError("Both product cluster IDs are required for group development")
    return str(value).strip()


def _clusters(pairs: list[dict]) -> set[str]:
    return {_cluster(pair, side) for pair in pairs for side in ("left", "right")}


def _counts(pairs: list[dict]) -> dict:
    y = labels(pairs)
    return {"n": len(pairs), "positives": int(y.sum()),
            "negatives": int((y == 0).sum()),
            "hard_negatives": sum(int(pair["label"]) == 0 and
                                  pair.get("is_hard_negative") is True for pair in pairs),
            "offers": len(offers(pairs)), "entities": len(_clusters(pairs))}


def _signature_set(pairs: list[dict]) -> set[tuple[str, ...]]:
    return {tuple(norm(pair.get(f"{field}_{side}")) for field in
                  ("title", "description", "brand"))
            for pair in pairs for side in ("left", "right")}


def entity_folds(pairs: list[dict], n_splits: int = 3,
                 seed: int = PARTITION_SEED, namespace: str = "outer") -> list[dict]:
    """Balanced hash-ranked entity folds, excluding all boundary-crossing pairs.

    ``namespace`` is recorded as a role name; assignment uses the same declared
    SHA256(seed:cluster_id) ordering on the applicable entity set at every level.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    entities = _clusters(pairs)
    if len(entities) < n_splits:
        raise ValueError("Not enough product entities for the declared folds")
    entity_by_offer = {}
    for pair in pairs:
        for side in ("left", "right"):
            offer_id = str(pair[f"id_{side}"]).strip()
            entity = _cluster(pair, side)
            if offer_id in entity_by_offer and entity_by_offer[offer_id] != entity:
                raise ValueError("One offer ID belongs to multiple product entities")
            entity_by_offer[offer_id] = entity
    ordered = sorted(entities, key=lambda key: (
        hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), key))
    assignment = {entity: index % n_splits for index, entity in enumerate(ordered)}
    output = []
    for fold in range(n_splits):
        fit_indices, eval_indices, excluded_indices = [], [], []
        for index, pair in enumerate(pairs):
            held = [assignment[_cluster(pair, side)] == fold for side in ("left", "right")]
            destination = eval_indices if all(held) else (fit_indices if not any(held)
                                                        else excluded_indices)
            destination.append(index)
        fit_rows = [pairs[index] for index in fit_indices]
        eval_rows = [pairs[index] for index in eval_indices]
        for role, rows in (("fit", fit_rows), ("evaluation", eval_rows)):
            if {int(pair["label"]) for pair in rows} != {0, 1}:
                raise ValueError(f"{namespace} fold {fold} {role} lacks a class; protocol infeasible")
        overlap = {"offer_ids": len(offers(fit_rows) & offers(eval_rows)),
                   "entities": len(_clusters(fit_rows) & _clusters(eval_rows)),
                   "normalized_text_signatures": len(_signature_set(fit_rows) &
                                                     _signature_set(eval_rows))}
        if overlap["offer_ids"] or overlap["entities"]:
            raise ValueError("Group fold leaks offer IDs or product entities")
        output.append({
            "fold": fold, "namespace": namespace, "seed": seed,
            "fit_indices": fit_indices, "eval_indices": eval_indices,
            "excluded_indices": excluded_indices,
            "fit_summary": _counts(fit_rows), "eval_summary": _counts(eval_rows),
            "excluded_summary": _counts([pairs[index] for index in excluded_indices]),
            "fit_cluster_ids": sorted(_clusters(fit_rows)),
            "eval_cluster_ids": sorted(_clusters(eval_rows)), "overlaps": overlap,
        })
    return output


def nested_plan(pairs: list[dict]) -> dict:
    """Make and check every split before fitting anything; never choose a seed."""
    stats = summary(pairs)
    errors = [key for key in INTEGRITY_FIELDS if stats[key]]
    if errors:
        raise ValueError(f"TRAIN integrity failure: {errors}")
    outer = entity_folds(pairs)
    for split in outer:
        fit_rows = [pairs[index] for index in split["fit_indices"]]
        split["inner_folds"] = entity_folds(fit_rows, namespace=f"outer_{split['fold']}_inner")
    seen = [index for split in outer for index in split["eval_indices"]]
    if len(seen) != len(set(seen)):
        raise ValueError("An input pair occurs in multiple outer evaluation parts")
    excluded = sorted(set(range(len(pairs))) - set(seen))
    return {"outer_folds": outer, "outer_evaluation_indices": seen,
            "excluded_from_all_outer_evaluations": {
                **_counts([pairs[index] for index in excluded]), "indices": excluded},
            "pooled_evaluation": _counts([pairs[index] for index in seen])}


def protocol_spec() -> dict:
    return {
        "version": "known-methods-train-nested-entity-development-v1",
        "scope": "TRAIN-only exploratory development; no validation, REPORT, TEST or FPV labels read",
        "outer_folds": 3, "inner_folds": 3, "partition_seed": PARTITION_SEED,
        "assignment": "Sort entities by SHA256(seed:cluster_id), then round-robin modulo 3; same ordering rule inside outer fit",
        "pair_rule": "Both endpoints in a part; all crossing pairs excluded and counted",
        "fit_rule": "Fresh TF-IDF, train_informativeness and LR inside every inner fit; refit on outer fit after selection",
        "methods": list(METHODS), "fields": list(FIELDS),
        "threshold_grid": THRESHOLDS.tolist(), "gamma_grid": list(GAMMAS),
        "C_grid": list(C_VALUES),
        "selection": "Pooled inner OOF F1, then precision, higher threshold, then grid order (smaller C first)",
        "adaptive_version": "Existing grouped Adaptive v1, unchanged; prior=0.20, penalty=0",
        "group_weights": GROUP_WEIGHTS.tolist(),
        "group_channel_weights": [value.tolist() for value in GROUP_CHANNEL_WEIGHTS],
        "manual_reliability": RELIABILITY.tolist(),
        "logistic": {"similarities": 4, "availability_indicators": 4,
                     "missing_encoding": "zero similarity + availability=0, observed zero + availability=1",
                     "penalty": "l2", "solver": "lbfgs", "class_weight": None,
                     "max_iter": 2000, "scaling": None, "calibration_established": False},
        "hard_negatives": "Official is_hard_negative=True among label=0; report FPR/counts, not negative-only F1",
        "missingness": {"scenarios": list(SCENARIOS), "rates": list(RATES), "seeds": list(SEEDS),
                        "auxiliary": ["description", "brand"],
                        "all_groups": [["title", "identifier"], ["description"], ["brand"]],
                        "unit": "One SHA256(seed:raw_field:offer_id) mask per offer/raw field, stable across pairs and methods",
                        "training": "Clean training only; no masked refits",
                        "thresholds": "Each selected outer threshold remains unchanged across all rates/seeds/scenarios"},
        "aggregation": "Primary pooled outer confusion matrix; fold metrics secondary. Three mask seeds are not independent datasets.",
        "final_test_evaluated": False, "fpv_labels_used": False,
    }


def apply_missingness(features: Features, rate: float, seed: int,
                      scenario: str = "auxiliary") -> np.ndarray:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown missingness scenario: {scenario}")
    output = masked(features, rate, seed)
    if scenario == "all_groups":
        def dropped(offer_id: str) -> bool:
            value = hashlib.sha256(f"{seed}:title:{offer_id}".encode()).digest()
            return int.from_bytes(value[:8], "big") / 2**64 < rate
        for index, (left, right) in enumerate(zip(features.ids_left, features.ids_right)):
            if dropped(left) or dropped(right):
                output[index, [0, 3]] = np.nan
    return output


def _stats(y: np.ndarray, prediction: np.ndarray) -> dict:
    result = metrics(y, prediction)
    result.update(n=int(len(y)), confusion_matrix=[[result["tn"], result["fp"]],
                                                  [result["fn"], result["tp"]]])
    return result


def select_from_oof(y: np.ndarray, options: list[tuple[dict, np.ndarray]]) -> dict:
    """Select using INNER predictions only; deterministic predeclared ties."""
    if not options:
        raise ValueError("At least one parameter candidate is required")
    candidates = []
    for parameters, values in options:
        threshold, stats = best_threshold(y, np.asarray(values))
        candidates.append({**parameters, "threshold": threshold, "inner_metrics": stats,
                           "inner_n": len(y)})
    chosen = max(candidates, key=lambda option: (option["inner_metrics"]["f1"],
                                                option["inner_metrics"]["precision"],
                                                option["threshold"]))
    return {**chosen, "candidate_metrics": candidates}


def _parameter_grid(method: str) -> list[dict]:
    if method == "logistic":
        return [{"C": value} for value in C_VALUES]
    if method == "adaptive":
        return [{"gamma": value} for value in GAMMAS]
    return [{}]


def _score(matrix: np.ndarray, method: str, info: np.ndarray,
           config: dict, logistic: SimilarityLogisticRegression | None = None) -> np.ndarray:
    if method == "logistic":
        if logistic is None:
            raise ValueError("Logistic model is required")
        return logistic.predict_score(matrix)
    return evidence_scores(matrix, method, info, gamma=config.get("gamma", 1.0),
                           penalty=0.0, prior=PRIOR)


def _representation_metadata(extractor: FeatureExtractor, train: list[dict]) -> dict:
    vocabularies = {}
    for field, vectorizer in extractor.vectorizers.items():
        if vectorizer is None:
            vocabularies[field] = {"terms": 0, "vocabulary_sha256": None, "idf_sha256": None}
        else:
            vocabulary = json.dumps({term: int(index) for term, index in
                                     vectorizer.vocabulary_.items()}, sort_keys=True).encode()
            vocabularies[field] = {
                "terms": len(vectorizer.vocabulary_),
                "vocabulary_sha256": hashlib.sha256(vocabulary).hexdigest(),
                "idf_sha256": hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest()}
    return {"fit_offer_ids": sorted(offers(train)), "fit_cluster_ids": sorted(_clusters(train)),
            "vectorizers": vocabularies}


def _negative_stats(rows: list[dict], y: np.ndarray, prediction: np.ndarray) -> dict:
    result = {}
    for name, hard in (("hard", True), ("easy", False)):
        indexes = np.array([int(row["label"]) == 0 and row.get("is_hard_negative") is hard
                            for row in rows], dtype=bool)
        n = int(indexes.sum())
        fp = int(prediction[indexes].sum())
        result[name] = {"n": n, "fp": fp, "tn": n - fp,
                        "fpr": fp / n if n else None,
                        "specificity": (n - fp) / n if n else None}
    result["unknown_hard_flag_negatives"] = sum(int(row["label"]) == 0 and
        type(row.get("is_hard_negative")) is not bool for row in rows)
    return result


def _prediction_record(pair: dict, matrix_row: np.ndarray, method: str,
                       fold: int, value: float, threshold: float) -> dict:
    y, prediction = int(pair["label"]), int(value >= threshold)
    tags = []
    if not np.isnan(matrix_row[3]) and matrix_row[3] == 0:
        tags.append("identifier_zero_overlap_proxy")
    if not np.isnan(matrix_row[2]) and matrix_row[2] == 0:
        tags.append("brand_unequal_strings")
    if not np.isnan(matrix_row[0]) and matrix_row[0] >= 0.8:
        tags.append("high_title_similarity")
    if np.isnan(matrix_row).any():
        tags.append("some_unavailable_channels")
    if not tags:
        tags.append("other_lexical_case")
    result = {"method": method, "outer_fold": fold,
              "pair_id": pair.get("pair_id", f"{pair['id_left']}#{pair['id_right']}"),
              "label": y, "is_hard_negative": pair.get("is_hard_negative"),
              "score": float(value), "threshold": threshold, "prediction": prediction,
              "coverage": float(evidence_coverage(matrix_row[None, :])[0]),
              "error_type": "FP" if prediction > y else ("FN" if prediction < y else "correct"),
              "diagnostic_tags": ";".join(tags)}
    for side in ("left", "right"):
        for field in ("id", "cluster_id", "title", "brand"):
            result[f"{field}_{side}"] = pair.get(f"{field}_{side}")
        result[f"description_excerpt_{side}"] = norm(pair.get(f"description_{side}"))[:200]
    for field, value in zip(FIELDS, matrix_row):
        result[f"sim_{field}"] = None if np.isnan(value) else float(value)
    return result


def _paired(y: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict:
    return {"n": len(y), "different_decisions": int((left != right).sum()),
            "both_correct": int(((left == y) & (right == y)).sum()),
            "both_wrong": int(((left != y) & (right != y)).sum()),
            "first_only_correct": int(((left == y) & (right != y)).sum()),
            "second_only_correct": int(((left != y) & (right == y)).sum())}


def evaluate_development(pairs: list[dict], plan: dict | None = None,
                         progress=None) -> dict:
    """Nested development OOF predictions on TRAIN; no file reads or writes."""
    plan = nested_plan(pairs) if plan is None else plan
    fold_results, prediction_rows, fold_missingness = [], [], []
    pooled_rows, pooled_y = [], []
    clean_predictions = {method: [] for method in METHODS}
    masked_predictions = {(scenario, rate, seed, method): []
                          for scenario in SCENARIOS for rate in RATES
                          for seed in SEEDS for method in METHODS}
    removed_counts = {(scenario, rate, seed): [0, 0] for scenario in SCENARIOS
                      for rate in RATES for seed in SEEDS}
    for outer in plan["outer_folds"]:
        fold = outer["fold"]
        fit_rows = [pairs[index] for index in outer["fit_indices"]]
        eval_rows = [pairs[index] for index in outer["eval_indices"]]
        if progress:
            progress(f"Outer {fold + 1}/3: fit={len(fit_rows)}, development={len(eval_rows)}")
        inner_targets, inner_fits = [], []
        inner_scores = {method: [[] for _ in _parameter_grid(method)] for method in METHODS}
        for inner in outer["inner_folds"]:
            inner_train = [fit_rows[index] for index in inner["fit_indices"]]
            inner_eval = [fit_rows[index] for index in inner["eval_indices"]]
            extractor = FeatureExtractor().fit(inner_train)
            train_matrix = extractor.transform(inner_train).matrix
            eval_matrix = extractor.transform(inner_eval).matrix
            info = train_informativeness(train_matrix, labels(inner_train))
            inner_targets.append(labels(inner_eval))
            for method in METHODS:
                for index, config in enumerate(_parameter_grid(method)):
                    logistic = (SimilarityLogisticRegression(C=config["C"]).fit(
                        train_matrix, labels(inner_train)) if method == "logistic" else None)
                    if logistic and logistic.metadata()["iteration_limit_reached"]:
                        raise RuntimeError("Inner logistic fit reached its iteration limit")
                    inner_scores[method][index].append(_score(eval_matrix, method, info, config, logistic))
            inner_fits.append({"inner_fold": inner["fold"],
                               "representation": _representation_metadata(extractor, inner_train),
                               "train_informativeness": info.tolist(),
                               "eval_offer_ids": sorted(offers(inner_eval))})
        inner_y = np.concatenate(inner_targets)
        selected = {method: select_from_oof(inner_y, [
            (parameters, np.concatenate(inner_scores[method][index]))
            for index, parameters in enumerate(_parameter_grid(method))]) for method in METHODS}
        # Selection is complete before the outer representation or predictions.
        extractor = FeatureExtractor().fit(fit_rows)
        train_features = extractor.transform(fit_rows)
        eval_features = extractor.transform(eval_rows)
        info = train_informativeness(train_features.matrix, labels(fit_rows))
        logistic = SimilarityLogisticRegression(C=selected["logistic"]["C"]).fit(
            train_features.matrix, labels(fit_rows))
        if logistic.metadata()["iteration_limit_reached"]:
            raise RuntimeError("Outer logistic fit reached its iteration limit")
        y = labels(eval_rows)
        fold_clean = {}
        for method in METHODS:
            config = selected[method]
            values = _score(eval_features.matrix, method, info, config, logistic)
            pred = (values >= config["threshold"]).astype(int)
            clean_predictions[method].append(pred)
            fold_clean[method] = _stats(y, pred)
            prediction_rows.extend(_prediction_record(pair, matrix_row, method, fold, value,
                                                       config["threshold"])
                                   for pair, matrix_row, value in zip(eval_rows, eval_features.matrix, values))
        originally_present = ~np.isnan(eval_features.matrix)
        for scenario in SCENARIOS:
            columns = [1, 2] if scenario == "auxiliary" else [0, 1, 2, 3]
            original_count = int(originally_present[:, columns].sum())
            for rate in RATES:
                for seed in SEEDS:
                    matrix = apply_missingness(eval_features, rate, seed, scenario)
                    removed = int((originally_present[:, columns] & np.isnan(matrix[:, columns])).sum())
                    count = removed_counts[scenario, rate, seed]
                    count[0] += removed
                    count[1] += original_count
                    no_evidence = np.isnan(matrix).all(axis=1)
                    for method in METHODS:
                        config = selected[method]
                        values = _score(matrix, method, info, config, logistic)
                        pred = (values >= config["threshold"]).astype(int)
                        masked_predictions[scenario, rate, seed, method].append(pred)
                        fold_missingness.append({
                            "outer_fold": fold, "scenario": scenario, "missing_rate": rate,
                            "seed": seed, "method": method, "threshold": config["threshold"],
                            "effective_removed_fraction": removed / original_count if original_count else 0.0,
                            "removed_comparable_pair_fields": removed,
                            "original_comparable_pair_fields": original_count,
                            "no_evidence_n": int(no_evidence.sum()),
                            "no_evidence_predicted_positive": int(pred[no_evidence].sum()),
                            **_stats(y, pred)})
        fold_results.append({"outer_fold": fold, "split": outer, "selected": selected,
                             "inner_fits": inner_fits,
                             "representation": _representation_metadata(extractor, fit_rows),
                             "train_informativeness": info.tolist(),
                             "train_feature_missing_fraction": np.isnan(train_features.matrix).mean(axis=0).tolist(),
                             "logistic_metadata": logistic.metadata(), "clean": fold_clean})
        pooled_rows.extend(eval_rows)
        pooled_y.append(y)
    y = np.concatenate(pooled_y)
    predictions = {method: np.concatenate(values) for method, values in clean_predictions.items()}
    robustness = []
    for (scenario, rate, seed, method), values in masked_predictions.items():
        prediction = np.concatenate(values)
        removed, original = removed_counts[scenario, rate, seed]
        matching_rows = [row for row in fold_missingness if (
            row["scenario"], row["missing_rate"], row["seed"], row["method"]) == (scenario, rate, seed, method)]
        robustness.append({"scenario": scenario, "missing_rate": rate, "seed": seed, "method": method,
                           "effective_removed_fraction": removed / original if original else 0.0,
                           "removed_comparable_pair_fields": removed,
                           "original_comparable_pair_fields": original,
                           "no_evidence_n": sum(row["no_evidence_n"] for row in matching_rows),
                           "no_evidence_predicted_positive": sum(row["no_evidence_predicted_positive"] for row in matching_rows),
                           "threshold_policy": "Frozen individually within each outer fold",
                           **_stats(y, prediction)})
    return {
        "protocol": protocol_spec(), "dataset": summary(pairs), "splits": plan,
        "folds": fold_results,
        "clean_metrics": {method: _stats(y, prediction) for method, prediction in predictions.items()},
        "hard_negatives": {method: _negative_stats(pooled_rows, y, prediction)
                           for method, prediction in predictions.items()},
        "paired_comparisons": {
            "renorm_vs_adaptive": _paired(y, predictions["renorm"], predictions["adaptive"]),
            "adaptive_vs_logistic": _paired(y, predictions["adaptive"], predictions["logistic"])},
        "missingness": robustness, "fold_missingness": fold_missingness,
        "clean_predictions": prediction_rows,
        "limitations": [
            "TRAIN has already informed prior development; nested CV remains exploratory, not independent confirmation.",
            "All cross-fold endpoint pairs are excluded; negative retention and class prevalence change.",
            "Metrics classify supplied candidate pairs, not end-to-end retrieval recall.",
            "Title-derived identifier is correlated with title; LR coefficients are not causal independent importance.",
            "Whole-title missingness may be unsupported by clean training; no all-missing rejection rule added to LR.",
            "Model outputs have not been independently calibrated; v1 reliability is manually assumed.",
            "Hard-negative flag is used for reporting, not a model input.",
            "REPORT, final TEST and unconfirmed FPV labels were not read or evaluated.",
        ],
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError("Empty CSV export requires explicit fieldnames")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIRECTORY)
    args = parser.parse_args(argv)
    # CLI deliberately supports only the audited TRAIN artifact, even if renamed.
    # Its reference digest is in the existing archive manifest; no TEST file opens.
    manifest = json.loads(Path("docs/archive_manifest.json").read_text(encoding="utf-8"))
    members = [item for item in manifest["members"] if item["name"] == TRAIN_PATH.name]
    if len(members) != 1 or sha256(args.train) != members[0]["sha256"]:
        parser.error("--train must be the audited WDC small TRAIN artifact (SHA-256 mismatch)")
    pairs = read_pairs(args.train)
    plan = nested_plan(pairs)
    code_files = ["run_known_methods.py", "logistic_baseline.py", "matching.py",
                  "audit_wdc.py", "run_experiment.py"]
    provenance = {
        "train_path": args.train.as_posix(), "train_sha256": sha256(args.train),
        "code_sha256": {name: sha256(name) for name in code_files},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "pytest")},
        "input_roles": ["official TRAIN only"], "test_opened": False,
        "report_opened": False, "fpv_inputs_opened": False,
    }
    frozen_protocol = {"protocol": protocol_spec(), "manifest": plan, "provenance": provenance}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out_dir / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old != frozen_protocol:
            parser.error("Existing protocol differs; choose a new output directory to preserve results")
    protocol_path.write_text(json.dumps(frozen_protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = evaluate_development(pairs, plan, progress=lambda message: print(message, flush=True))
    prediction_rows = result.pop("clean_predictions")
    _write_csv(args.out_dir / "clean_predictions.csv", prediction_rows)
    _write_csv(args.out_dir / "errors.csv", [row for row in prediction_rows if row["error_type"] != "correct"],
               fieldnames=list(prediction_rows[0]))
    result["provenance"] = provenance
    result["paths"] = {"predictions": "clean_predictions.csv", "errors": "errors.csv"}
    (args.out_dir / "development_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["clean_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
