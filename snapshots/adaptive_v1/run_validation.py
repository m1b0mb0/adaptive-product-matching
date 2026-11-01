"""Train-only representations, cluster-disjoint validation tuning/reporting.

This entry point accepts no test split and performs no final test evaluation.
The protocol is fixed before results are computed. Crossing validation pairs
are excluded from both partitions, rather than leaking offers between them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from audit_wdc import audit, offers, read_pairs, summary
from matching import (FIELDS, GROUP_CHANNEL_WEIGHTS, GROUP_WEIGHTS, RELIABILITY,
                      FeatureExtractor, evidence_coverage, evidence_scores, masked,
                      scores, train_informativeness)
from run_experiment import best_threshold, labels, metrics

MAIN_METHODS = ("text", "fixed", "renorm", "adaptive")
ABLATIONS = ("adaptive_no_coverage", "adaptive_no_info", "adaptive_no_reliability")
LEGACY_METHODS = ("legacy_fixed", "legacy_renorm", "legacy_adaptive")
METHODS = MAIN_METHODS + ABLATIONS + LEGACY_METHODS
RATES = (0.0, 0.1, 0.2, 0.3, 0.5)
SEEDS = (11, 22, 33)
GAMMAS = (0.5, 1.0, 1.5)
PRIOR = 0.20
PARTITION_SEED = 20261101


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def protocol_spec(validation_mode: str, partition_seed: int = PARTITION_SEED,
                  tune_fraction: float = 0.5) -> dict:
    return {
        "version": "validation-grouped-evidence-v1",
        "validation_mode": validation_mode,
        "representation_fit": "train only, deduplicated offer IDs",
        "informativeness_fit": "train similarities and train labels only",
        "tuning": "clean TUNE partition of validation only",
        "reporting": "disjoint REPORT partition of validation; no retuning",
        "partition": {"unit": "product cluster_id", "seed": partition_seed,
                      "tune_fraction": tune_fraction,
                      "assignment": "Sort clusters by SHA256(seed:cluster_id); first floor(fraction*n) -> TUNE",
                      "crossing_pairs": "Exclude pairs whose endpoints fall in different partitions; preserve official files."},
        "threshold_grid": [float(x) for x in np.linspace(0.01, 0.99, 99)],
        "selection": "max F1, then precision, then higher threshold; parameter ties retain grid order",
        "gamma_grid": list(GAMMAS), "main_penalty": 0.0, "prior_score_reference": PRIOR,
        "group_weights": GROUP_WEIGHTS.tolist(),
        "groups": [["title", "identifier"], ["description"], ["brand"]],
        "group_channel_weights": [x.tolist() for x in GROUP_CHANNEL_WEIGHTS],
        "manual_reliability": RELIABILITY.tolist(),
        "ablation_policy": "Inherit full-model gamma and penalty; reselect only threshold on clean TUNE. Also report removal with the full-model threshold frozen.",
        "legacy_diagnostics": "Original raw-channel scores; legacy adaptive gamma=.5/1/1.5 and penalty=0/.10/.25 grid. Unequal tuning budget; descriptive diagnostics only.",
        "missingness": {"rates": list(RATES), "seeds": list(SEEDS),
                        "attributes": ["description", "brand"], "unit": "offer ID and attribute",
                        "mechanism": "SHA256(seed:field:offer_id), independent uniform masking; a pair loses the feature if either offer is masked",
                        "thresholds": "Frozen at clean-TUNE values for all REPORT masks"},
        "final_test_evaluated": False,
    }


def partition_validation(pairs: list[dict], seed: int = PARTITION_SEED,
                         tune_fraction: float = 0.5) -> tuple[list[dict], list[dict], dict]:
    if not 0 < tune_fraction < 1:
        raise ValueError("tune_fraction must be strictly between zero and one")
    clusters = set()
    for pair in pairs:
        for side in ("left", "right"):
            value = pair.get(f"cluster_id_{side}")
            if value is None or not str(value).strip():
                raise ValueError("Cluster IDs are required for disjoint validation partitioning")
            clusters.add(str(value).strip())
    if len(clusters) < 4:
        raise ValueError("At least four product clusters are required")
    ordered = sorted(clusters, key=lambda key: (hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), key))
    n_tune = int(len(ordered) * tune_fraction)
    tune_clusters = set(ordered[:n_tune])
    tune, report, dropped = [], [], []
    for pair in pairs:
        left, right = [str(pair[f"cluster_id_{side}"]).strip() in tune_clusters for side in ("left", "right")]
        if left and right:
            tune.append(pair)
        elif not left and not right:
            report.append(pair)
        else:
            dropped.append(pair)
    for name, rows in (("TUNE", tune), ("REPORT", report)):
        if {int(pair["label"]) for pair in rows} != {0, 1}:
            raise ValueError(f"{name} must contain both classes; choose a protocol before experiments")
    metadata = {"seed": seed, "original_pairs": len(pairs), "tune_clusters": len(tune_clusters),
                "report_clusters": len(clusters) - len(tune_clusters),
                "tune_cluster_ids": sorted(tune_clusters), "report_cluster_ids": sorted(clusters - tune_clusters),
                "tune": summary(tune), "report": summary(report),
                "excluded_crossing_pairs": len(dropped),
                "excluded_labels": {str(y): sum(int(p["label"]) == y for p in dropped) for y in (0, 1)},
                "offer_overlap_tune_report": len(offers(tune) & offers(report))}
    return tune, report, metadata


def method_scores(matrix: np.ndarray, method: str, info: np.ndarray, config: dict) -> np.ndarray:
    params = {key: config[key] for key in ("gamma", "penalty")}
    if method.startswith("legacy_"):
        return scores(matrix, method.removeprefix("legacy_"), info, prior=PRIOR, **params)
    return evidence_scores(matrix, method, info, prior=PRIOR, **params)


def classification_metrics(y: np.ndarray, values: np.ndarray, threshold: float) -> dict:
    result = metrics(y, (values >= threshold).astype(int))
    result["confusion_matrix"] = [[result["tn"], result["fp"]], [result["fn"], result["tp"]]]
    result["n"] = int(len(y))
    return result


def select_config(y: np.ndarray, matrix: np.ndarray, method: str, info: np.ndarray,
                  chosen: dict[str, dict]) -> dict:
    if method == "adaptive":
        candidates = [{"gamma": g, "penalty": 0.0} for g in GAMMAS]
    elif method in ABLATIONS:
        candidates = [{key: chosen["adaptive"][key] for key in ("gamma", "penalty")}]
    elif method == "legacy_adaptive":
        candidates = [{"gamma": g, "penalty": p} for g in GAMMAS for p in (0.0, 0.10, 0.25)]
    else:
        candidates = [{"gamma": 1.0, "penalty": 0.0}]
    options = []
    for params in candidates:
        values = method_scores(matrix, method, info, params)
        threshold, stats = best_threshold(y, values)
        options.append((stats, threshold, params))
    stats, threshold, params = max(options, key=lambda item: (item[0]["f1"], item[0]["precision"], item[1]))
    return {"threshold": threshold, **params, "validation": stats, "n_parameter_candidates": len(candidates)}


def paired_errors(y: np.ndarray, adaptive: np.ndarray, comparator: np.ndarray, name: str) -> dict:
    a_correct, b_correct = adaptive == y, comparator == y
    return {"comparator": name, "n": len(y),
            "both_correct": int((a_correct & b_correct).sum()),
            "both_wrong": int((~a_correct & ~b_correct).sum()),
            "adaptive_only_correct": int((a_correct & ~b_correct).sum()),
            "comparator_only_correct": int((~a_correct & b_correct).sum()),
            "rescued_fp": int(((y == 0) & a_correct & ~b_correct).sum()),
            "rescued_fn": int(((y == 1) & a_correct & ~b_correct).sum()),
            "introduced_fp": int(((y == 0) & ~a_correct & b_correct).sum()),
            "introduced_fn": int(((y == 1) & ~a_correct & b_correct).sum()),
            "net_correct_change": int(a_correct.sum() - b_correct.sum())}


def evaluate_validation(train: list[dict], validation: list[dict], *,
                        validation_mode: str = "fallback_seen", partition_seed: int = PARTITION_SEED,
                        tune_fraction: float = 0.5) -> dict:
    # Check the complete validation, including pairs later excluded by partitioning.
    original = {"train": summary(train), "validation": summary(validation)}
    checked = ("duplicate_rows", "duplicate_pair_ids", "pair_id_mismatches",
               "duplicate_canonical_pairs", "conflicting_label_pairs", "self_pairs",
               "label_cluster_mismatches", "inconsistent_offer_payloads", "positive_hard_flag_true")
    errors = [f"{name}: {key}={stats[key]}" for name, stats in original.items()
              for key in checked if stats[key]]
    if errors:
        raise ValueError(f"Complete-split integrity failure: {errors}")
    if offers(train) & offers(validation):
        raise ValueError("Offer ID overlap between train and complete validation")
    tune, report, partition = partition_validation(validation, partition_seed, tune_fraction)
    schema = audit(train, tune, report)
    if schema["integrity_errors"] or any(schema["offer_id_overlaps"].values()):
        raise ValueError(f"Split integrity failure: {schema['integrity_errors']}, {schema['offer_id_overlaps']}")
    if schema["cluster_id_overlaps"]["valid_test"]:
        raise ValueError("TUNE and REPORT clusters overlap")
    if validation_mode == "unseen" and (schema["cluster_id_overlaps"]["train_valid"] or schema["cluster_id_overlaps"]["train_test"]):
        raise ValueError("Unseen validation contains train clusters")
    # audit's legacy names valid/test mean TUNE/REPORT HERE, never the reserved benchmark test.
    integrity = {"split_role_names": {"train": "train", "valid": "TUNE", "test": "REPORT (validation subset)"},
                 **{key: value for key, value in schema.items() if key != "splits"}}
    extractor = FeatureExtractor().fit(train)
    train_f = extractor.transform(train)
    tune_f = extractor.transform(tune)
    report_f = extractor.transform(report)
    info = train_informativeness(train_f.matrix, labels(train))
    y_tune = labels(tune)
    chosen, report_metrics, report_predictions = {}, {}, {}
    for method in METHODS:
        chosen[method] = select_config(y_tune, tune_f.matrix, method, info, chosen)
    # Freeze every method before computing any REPORT classification metrics.
    y_report = labels(report)
    for method in METHODS:
        values = method_scores(report_f.matrix, method, info, chosen[method])
        report_metrics[method] = classification_metrics(y_report, values, chosen[method]["threshold"])
        report_predictions[method] = (values >= chosen[method]["threshold"]).astype(int)
    frozen_ablations = {}
    for method in ABLATIONS:
        values = method_scores(report_f.matrix, method, info, chosen[method])
        threshold = chosen["adaptive"]["threshold"]
        frozen_ablations[method] = {"threshold": threshold, **classification_metrics(y_report, values, threshold)}
    missingness = []
    originally_comparable = ~np.isnan(report_f.matrix[:, 1:3])
    original_count = int(originally_comparable.sum())
    for rate in RATES:
        for seed in SEEDS:
            corrupted = masked(report_f, rate, seed)
            removed = originally_comparable & np.isnan(corrupted[:, 1:3])
            effective = int(removed.sum()) / original_count if original_count else 0.0
            for method in METHODS:
                config = chosen[method]
                values = method_scores(corrupted, method, info, config)
                missingness.append({"missing_rate": rate, "seed": seed, "scope": "report", "method": method,
                                    "threshold": config["threshold"], "gamma": config["gamma"], "penalty": config["penalty"],
                                    "effective_removed_fraction": effective,
                                    "removed_comparable_pair_fields": int(removed.sum()),
                                    "original_comparable_pair_fields": original_count,
                                    **classification_metrics(y_report, values, config["threshold"])})
    diagnostics = []
    coverage = evidence_coverage(report_f.matrix)
    strata = {f"coverage_{value:.2f}": np.isclose(coverage, value) for value in sorted(set(coverage))}
    strata["identifier_conflict"] = ~np.isnan(report_f.matrix[:, 3]) & (report_f.matrix[:, 3] == 0)
    strata["identifier_missing"] = np.isnan(report_f.matrix[:, 3])
    for name, indexes in strata.items():
        if indexes.any():
            for method in MAIN_METHODS:
                pred = report_predictions[method][indexes]
                diagnostics.append({"stratum": name, "method": method, "n": int(indexes.sum()),
                                    "positives": int(y_report[indexes].sum()),
                                    **metrics(y_report[indexes], pred)})
    train_stats = {}
    for j, field in enumerate(FIELDS):
        row = {"factor": float(info[j])}
        for y, name in ((0, "negative"), (1, "positive")):
            values = train_f.matrix[(labels(train) == y) & ~np.isnan(train_f.matrix[:, j]), j]
            row[f"{name}_n"] = len(values)
            row[f"{name}_mean"] = float(values.mean()) if len(values) else None
        train_stats[field] = row
    return {
        "scope": "Preliminary validation only; reserved test labels and scores not accessed",
        "protocol": {**protocol_spec(validation_mode, partition_seed, tune_fraction),
                     "partition": {**protocol_spec(validation_mode, partition_seed, tune_fraction)["partition"], **partition}},
        "data_summary": original, "integrity": integrity,
        "train_discriminativeness": dict(zip(FIELDS, info.tolist())), "train_field_statistics": train_stats,
        "feature_comparability": {name: dict(zip(FIELDS, (~np.isnan(value.matrix)).sum(axis=0).tolist()))
                                  for name, value in (("train", train_f), ("tune", tune_f), ("report", report_f))},
        "validation_selected": chosen, "validation_metrics": report_metrics,
        "ablation_fixed_threshold_metrics": frozen_ablations,
        "missingness": {"mechanism": "Offer-level masking of description and brand only", "rows": missingness},
        "paired_error_analysis": [paired_errors(y_report, report_predictions["adaptive"], report_predictions[method], method)
                                  for method in ("text", "fixed", "renorm", "legacy_adaptive") if method in report_predictions],
        "by_evidence": diagnostics,
        "limitations": [
            "Seen-model fallback validation cannot establish generalization to reserved unseen-model test.",
            "Cluster-disjoint TUNE/REPORT discard crossing negative pairs, change class prevalence and difficulty; not directly comparable to official full-split benchmark scores.",
            "Single deterministic partition; no generalization confidence interval or proof of superiority.",
            "Reliability coefficients are manual sensitivity assumptions, not empirically validated source reliability.",
            "Informativeness is global field discriminativeness, not value-specific information content.",
            "Title and identifier remain correlated views of the same source even inside a shared budget.",
            "Only description and brand are masked; seed standard deviations describe masks, not sampling uncertainty.",
            "Metrics classify supplied pairs and do not measure end-to-end retrieval recall.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--data-config", default="docs/validation_config.json")
    parser.add_argument("--out", default="docs/validation/results.json")
    parser.add_argument("--partition-seed", type=int, default=PARTITION_SEED)
    parser.add_argument("--tune-fraction", type=float, default=0.5)
    args = parser.parse_args()
    data_config = json.loads(Path(args.data_config).read_text(encoding="utf-8"))
    train_path, valid_path = Path(args.train), Path(args.valid)
    for name, path in (("train_path", train_path), ("validation_path", valid_path)):
        if name in data_config and path.resolve() != Path(data_config[name]).resolve():
            parser.error(f"{name} differs from audited data configuration")
    if data_config.get("validation_sha256") and sha256(valid_path) != data_config["validation_sha256"]:
        parser.error("Validation checksum differs from audited configuration")
    mode = data_config.get("mode", "fallback_seen")
    protocol = protocol_spec(mode, args.partition_seed, args.tune_fraction)
    # Save before fitting/scoring, so the architecture/grid are reviewable independently of F1.
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    (destination.parent / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = evaluate_validation(read_pairs(train_path), read_pairs(valid_path), validation_mode=mode,
                                 partition_seed=args.partition_seed, tune_fraction=args.tune_fraction)
    source_paths = ["matching.py", "run_experiment.py", "run_validation.py", "audit_wdc.py"]
    result["metadata"] = {
        "date": "2026-11-01", "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib")},
        "train_path": train_path.as_posix(), "train_sha256": sha256(train_path),
        "validation_path": valid_path.as_posix(), "validation_sha256": sha256(valid_path),
        "validation_mode": mode, "validation_selection_reason": data_config.get("reason"),
        "source_sha256": {name: sha256(name) for name in source_paths},
        "data_config_sha256": sha256(args.data_config),
        "test_access": "This runner accepts no test argument and opens only train and selected validation. Separate preparation checks reserved-test identity metadata only.",
    }
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for method in MAIN_METHODS:
        values = result["validation_metrics"][method]
        print(f"{method:12s} P={values['precision']:.5f} R={values['recall']:.5f} F1={values['f1']:.5f}")
    print(f"Validation artifacts: {destination}")


if __name__ == "__main__":
    main()
