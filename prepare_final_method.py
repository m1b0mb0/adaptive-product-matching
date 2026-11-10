"""Freeze the single corrected candidate on TRAIN; never open final TEST.

Cached historical baseline results are verified and reused. New fitting is
limited to corrected nested development and required INNER-only calibration
of one full-TRAIN bundle. No old report is regenerated or overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from statistics import mean, stdev

import joblib
import numpy as np

from adaptive_corrected import METHOD_SPECIFICATION, corrected_scores
from audit_wdc import read_pairs, summary
from logistic_baseline import SimilarityLogisticRegression
from matching import FeatureExtractor, train_informativeness
from run_experiment import labels
from run_known_methods import (METHODS, C_VALUES, GAMMAS, PARTITION_SEED, PRIOR,
    RATES, SCENARIOS, SEEDS, TRAIN_PATH, THRESHOLDS, _negative_stats,
    _parameter_grid, _representation_metadata, _score, _stats,
    apply_missingness, nested_plan, select_from_oof)

ROOT = Path(__file__).resolve().parent
CORRECTED = "adaptive_corrected"
ALL_METHODS = (*METHODS, CORRECTED)
CODE_FILES = ("matching.py", "adaptive_corrected.py", "logistic_baseline.py",
              "run_known_methods.py", "run_experiment.py", "audit_wdc.py",
              "prepare_final_method.py", "frozen_method_artifact.py",
              "search_system.py", "search_cli.py", "product_catalog.py")


def digest(path, *, lf=False):
    data = Path(path).read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def json_digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def protocol_spec():
    return {
        "version": "adaptive-v1-corrected-finalization-1",
        "method_specification": METHOD_SPECIFICATION,
        "formula_fixed_before_results": True,
        "development": "Only corrected nested 3x3 on identical historical TRAIN entity folds; baseline scores reused after verification",
        "seed": PARTITION_SEED,
        "corrected_parameter_grid": [{"gamma": 1.0, "prior": 0.0, "penalty": 0.0}],
        "threshold_grid": THRESHOLDS.tolist(),
        "selection": "max F1, then precision, higher threshold, grid order",
        "final_calibration": {
            "source": "Nine INNER held-out prediction sets only, never outer predictions",
            "duplicates": "Mean raw inner held-out score per original pair_id; one equally weighted row per pair",
            "fold_thresholds": "Never average selected fold thresholds",
            "gamma_grid_v1": list(GAMMAS), "C_grid_logistic": list(C_VALUES),
            "corrected": "Fixed formula; select threshold only",
            "role": "Final calibration, not independent performance estimation",
        },
        "full_fit": "One full-TRAIN extractor/informativeness fit and one selected LR fit after final calibration",
        "missingness": {"scenarios": list(SCENARIOS), "rates": list(RATES),
                        "seeds": list(SEEDS), "thresholds": "Clean inner-selected outer thresholds frozen"},
        "historical_results_overwritten": False,
        "test_opened": False, "fpv_labels_used": False,
    }


def verify_historical_cache(pairs, plan, cache_dir, train_path):
    cache_dir = Path(cache_dir)
    cached = json.loads((cache_dir / "development_results.json").read_text(encoding="utf-8"))
    if digest(train_path) != cached["provenance"]["train_sha256"]:
        raise ValueError("Restored TRAIN does not match the historical experiment")
    code_checks = []
    for name, expected in cached["provenance"]["code_sha256"].items():
        raw, canonical = digest(name), digest(name, lf=True)
        if expected not in (raw, canonical):
            raise ValueError(f"Historical source differs beyond CRLF/LF: {name}")
        code_checks.append({"path": name, "historical_sha256": expected,
                            "current_sha256": raw, "current_lf_sha256": canonical})
    for name, expected in cached["provenance"]["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Historical package version differs: {name}")
    if cached["splits"] != plan:
        raise ValueError("Current entity split manifest differs from historical development")
    validated = json.loads((cache_dir / "pipeline_validation.json").read_text(encoding="utf-8"))
    artifact_checks = []
    for name in ("development_results.json", "protocol.json", "clean_predictions.csv"):
        path = cache_dir / name
        key = path.as_posix()
        expected = validated["artifact_sha256"].get(key)
        if expected is None or expected not in (digest(path), digest(path, lf=True)):
            raise ValueError(f"Historical artifact checksum mismatch: {name}")
        artifact_checks.append({"path": key, "historical_sha256": expected,
                                "current_sha256": digest(path), "current_lf_sha256": digest(path, lf=True)})
    with (cache_dir / "clean_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    expected_ids = {str(pairs[index]["pair_id"]) for index in plan["outer_evaluation_indices"]}
    for method in METHODS:
        rows = [row for row in predictions if row["method"] == method]
        if len(rows) != len(expected_ids) or {row["pair_id"] for row in rows} != expected_ids:
            raise ValueError("Historical baseline prediction IDs differ")
        y = np.array([int(row["label"]) for row in rows])
        pred = np.array([int(row["prediction"]) for row in rows])
        stats = _stats(y, pred)
        if stats != cached["clean_metrics"][method]:
            raise ValueError(f"Cached baseline metrics do not match decisions: {method}")
    return cached, {"code_checks": code_checks, "artifact_checks": artifact_checks,
                    "data_sha256_matches": True, "package_versions_match": True,
                    "split_manifest_matches": True, "baseline_metrics_recomputed_from_csv": True,
                    "baseline_experiments_rerun": False}


def _configuration_options(method):
    return [{}] if method == CORRECTED else _parameter_grid(method)


def _values(matrix, method, info, config, logistic=None):
    if method == CORRECTED:
        return corrected_scores(matrix, info)
    return _score(matrix, method, info, config, logistic)


def aggregate_inner_calibration(predictions):
    """One row per original pair: average only its held-out inner raw scores."""
    grouped = defaultdict(list)
    targets = {}
    for index, y, value in predictions:
        if index in targets and targets[index] != int(y):
            raise ValueError("One calibration pair has conflicting labels")
        targets[index] = int(y)
        grouped[int(index)].append(float(value))
    indices = sorted(grouped)
    return (indices, np.array([targets[i] for i in indices], dtype=int),
            np.array([mean(grouped[i]) for i in indices], dtype=float),
            [len(grouped[i]) for i in indices])


def run_corrected_development(pairs, plan, *, progress=None):
    calibration = {method: [[] for _ in _configuration_options(method)] for method in ALL_METHODS}
    folds, outer_predictions, robustness, pooled_y, pooled_rows = [], [], [], [], []
    pooled_predictions = []
    masked_predictions = {(scenario, rate, seed): [] for scenario in SCENARIOS for rate in RATES for seed in SEEDS}
    removals = {(scenario, rate, seed): [0, 0] for scenario in SCENARIOS for rate in RATES for seed in SEEDS}
    for split in plan["outer_folds"]:
        fold = split["fold"]
        fit_rows = [pairs[i] for i in split["fit_indices"]]
        eval_rows = [pairs[i] for i in split["eval_indices"]]
        if progress:
            progress(f"Corrected outer {fold+1}/3: fit={len(fit_rows)}, development={len(eval_rows)}")
        inner_y, inner_values, fit_records = [], [], []
        for inner in split["inner_folds"]:
            train_rows = [fit_rows[i] for i in inner["fit_indices"]]
            held_rows = [fit_rows[i] for i in inner["eval_indices"]]
            extractor = FeatureExtractor().fit(train_rows)
            train_matrix = extractor.transform(train_rows).matrix
            held_matrix = extractor.transform(held_rows).matrix
            y_train, y_held = labels(train_rows), labels(held_rows)
            info = train_informativeness(train_matrix, y_train)
            corrected = corrected_scores(held_matrix, info)
            inner_y.append(y_held)
            inner_values.append(corrected)
            global_indices = [split["fit_indices"][i] for i in inner["eval_indices"]]
            for method in ALL_METHODS:
                for option_index, parameters in enumerate(_configuration_options(method)):
                    logistic = (SimilarityLogisticRegression(C=parameters["C"]).fit(train_matrix, y_train)
                                if method == "logistic" else None)
                    if logistic is not None and logistic.metadata()["iteration_limit_reached"]:
                        raise RuntimeError("Inner final-calibration logistic fit did not converge")
                    values = corrected if method == CORRECTED else _values(held_matrix, method, info, parameters, logistic)
                    calibration[method][option_index].extend(
                        (int(i), int(y), float(value)) for i, y, value in zip(global_indices, y_held, values))
            fit_records.append({"inner_fold": inner["fold"],
                "representation": _representation_metadata(extractor, train_rows),
                "eval_offer_ids": sorted({str(p[f"id_{s}"]) for p in held_rows for s in ("left", "right")}),
                "global_eval_indices": global_indices, "train_informativeness": info.tolist()})
        selected = select_from_oof(np.concatenate(inner_y), [({}, np.concatenate(inner_values))])
        extractor = FeatureExtractor().fit(fit_rows)
        train_matrix = extractor.transform(fit_rows).matrix
        info = train_informativeness(train_matrix, labels(fit_rows))
        held_features = extractor.transform(eval_rows)
        y = labels(eval_rows)
        values = corrected_scores(held_features.matrix, info)
        prediction = (values >= selected["threshold"]).astype(int)
        for index, pair, value, pred in zip(split["eval_indices"], eval_rows, values, prediction):
            outer_predictions.append({"outer_fold": fold, "input_index": index,
                "pair_id": pair["pair_id"], "label": int(pair["label"]),
                "score": float(value), "threshold": selected["threshold"], "prediction": int(pred),
                "is_hard_negative": pair.get("is_hard_negative"),
                "id_left": str(pair["id_left"]), "id_right": str(pair["id_right"])})
        for scenario in SCENARIOS:
            columns = [1, 2] if scenario == "auxiliary" else [0, 1, 2, 3]
            present = ~np.isnan(held_features.matrix[:, columns])
            for rate in RATES:
                for seed in SEEDS:
                    matrix = apply_missingness(held_features, rate, seed, scenario)
                    pred = (corrected_scores(matrix, info) >= selected["threshold"]).astype(int)
                    masked_predictions[scenario, rate, seed].append(pred)
                    count = removals[scenario, rate, seed]
                    count[0] += int((present & np.isnan(matrix[:, columns])).sum())
                    count[1] += int(present.sum())
        folds.append({"outer_fold": fold, "selected": selected,
            "representation": _representation_metadata(extractor, fit_rows),
            "train_informativeness": info.tolist(), "inner_fits": fit_records,
            "clean_metrics": _stats(y, prediction)})
        pooled_y.append(y)
        pooled_rows.extend(eval_rows)
        pooled_predictions.append(prediction)
    y = np.concatenate(pooled_y)
    predictions = np.concatenate(pooled_predictions)
    for key, parts in masked_predictions.items():
        scenario, rate, seed = key
        removed, original = removals[key]
        robustness.append({"method": CORRECTED, "scenario": scenario, "missing_rate": rate,
            "seed": seed, "effective_removed_fraction": removed / original if original else 0.0,
            "threshold_policy": "Frozen separately within each outer fold",
            **_stats(y, np.concatenate(parts))})
    final_selected, calibration_rows, cohort = {}, [], None
    for method in ALL_METHODS:
        options = []
        for option_index, parameters in enumerate(_configuration_options(method)):
            indices, y_cal, values, multiplicities = aggregate_inner_calibration(calibration[method][option_index])
            signature = (indices, y_cal.tolist(), multiplicities)
            if cohort is None:
                cohort = signature
            if signature != cohort:
                raise ValueError("Final calibration cohorts differ between methods/configurations")
            options.append((parameters, values))
            for index, target, value, multiplicity in zip(indices, y_cal, values, multiplicities):
                calibration_rows.append({"method": method, "configuration_index": option_index,
                    "configuration": json.dumps(parameters, sort_keys=True), "input_index": index,
                    "pair_id": pairs[index]["pair_id"], "label": int(target),
                    "mean_inner_heldout_score": float(value), "multiplicity": multiplicity})
        final_selected[method] = select_from_oof(y_cal, options)
    indices, target, multiplicities = cohort
    return ({"folds": folds, "clean_metrics": _stats(y, predictions),
             "hard_negatives": _negative_stats(pooled_rows, y, predictions),
             "missingness": robustness, "outer_predictions": outer_predictions,
             "final_calibration": {"policy": protocol_spec()["final_calibration"],
                 "n_unique_pairs": len(indices), "positives": int(sum(target)),
                 "negative_pairs": len(target) - int(sum(target)),
                 "multiplicity_counts": {str(k): v for k, v in Counter(multiplicities).items()},
                 "included_input_indices": indices, "selected": final_selected,
                 "selection_metrics_are_independent_quality_estimates": False}}, calibration_rows)


def _csv(path, rows, columns=None):
    fieldnames = columns if columns is not None else list(rows[0])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/final_method"))
    args = parser.parse_args(argv)
    if args.train.name != TRAIN_PATH.name:
        parser.error("Only the audited TRAIN artifact is allowed; TEST is not an input")
    if not args.out_dir.resolve().is_relative_to((ROOT / "docs/final_method").resolve()):
        parser.error("Outputs must remain under docs/final_method")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_suffix = args.out_dir.resolve().relative_to((ROOT / "docs/final_method").resolve())
    artifact = Path("data/processed/final_method") / run_suffix / "frozen_models.joblib"
    if artifact.exists():
        parser.error("A model bundle already exists for this run; refusing overwrite")
    if (args.out_dir / "frozen_configuration.json").exists() or (args.out_dir / "corrected_development_results.json").exists():
        parser.error("Finalization already completed here; preserve results and use a new subdirectory")
    baseline_path = Path("docs/known_methods")
    cached_reference = json.loads((baseline_path / "development_results.json").read_text(encoding="utf-8"))
    if digest(args.train) != cached_reference["provenance"]["train_sha256"]:
        parser.error("Restored TRAIN SHA-256 differs from the historical artifact")
    pairs = read_pairs(args.train)
    plan = nested_plan(pairs)
    cached, compatibility = verify_historical_cache(pairs, plan, baseline_path, args.train)
    prospective = {"protocol": protocol_spec(), "split_manifest_sha256": json_digest(plan),
        "train_sha256": digest(args.train), "code_sha256_lf": {name: digest(name, lf=True) for name in CODE_FILES}}
    protocol_path = args.out_dir / "predeclared_protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != prospective:
            parser.error("Predeclared formula, TRAIN, code hashes or split manifest changed; refusing to run")
    _json(protocol_path, prospective)
    result, calibration_rows = run_corrected_development(pairs, plan, progress=lambda message: print(message, flush=True))
    selected = {}
    for method, record in result["final_calibration"]["selected"].items():
        selected[method] = {key: record[key] for key in ("threshold", "gamma", "C") if key in record}
    selected["adaptive"] |= {"prior": PRIOR, "penalty": 0.0}
    selected[CORRECTED] |= {"gamma": 1.0, "prior": 0.0, "penalty": 0.0}
    # ONE full-TRAIN fit, after final INNER-only calibration is complete.
    print("Full TRAIN fit for frozen bundle; no TEST access", flush=True)
    extractor = FeatureExtractor().fit(pairs)
    matrix = extractor.transform(pairs).matrix
    info = train_informativeness(matrix, labels(pairs))
    logistic = SimilarityLogisticRegression(C=selected["logistic"]["C"]).fit(matrix, labels(pairs))
    if logistic.metadata()["iteration_limit_reached"]:
        raise RuntimeError("Final full-TRAIN LR did not converge")
    bundle = {"schema_version": 1, "extractor": extractor, "informativeness": info,
        "logistic": logistic, "selected": selected, "train_sha256": digest(args.train),
        "method_specification": METHOD_SPECIFICATION}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        raise ValueError("A frozen model bundle already exists; refusing overwrite")
    joblib.dump(bundle, artifact, compress=3)
    provenance = {"workspace": str(ROOT), "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in cached["provenance"]["packages"]},
        "train_path": args.train.as_posix(), "train_sha256": digest(args.train),
        "code_sha256_lf": {name: digest(name, lf=True) for name in CODE_FILES},
        "code_sha256": {name: digest(name) for name in CODE_FILES},
        "code_hash_policy": "LF-normalized SHA-256 for cross-device CRLF compatibility; raw hashes also recorded",
        "test_opened": False, "fpv_labels_used": False, "historical_baseline_experiments_rerun": False}
    config = {"schema_version": 1, "method_specification": METHOD_SPECIFICATION,
        "final_models": selected, "parameter_sha256": json_digest(selected),
        "method_specification_sha256": json_digest(METHOD_SPECIFICATION),
        "artifacts": {"model_bundle": {"path": artifact.as_posix(), "sha256": digest(artifact)}},
        "provenance": provenance, "final_calibration": result["final_calibration"],
        "full_train_fit": {"representation": _representation_metadata(extractor, pairs),
            "informativeness": info.tolist(), "logistic": logistic.metadata()},
        "test_evaluation": {"status": "NOT RUN; separate user authorization required",
            "path": "data/external/wdcproducts80cc20rnd100un_gs.json.gz", "opened": False}}
    result["comparison"] = {**cached["clean_metrics"], CORRECTED: result["clean_metrics"]}
    result["hard_negative_comparison"] = {**cached["hard_negatives"], CORRECTED: result["hard_negatives"]}
    result["cache_compatibility"] = compatibility
    result["provenance"] = provenance
    result["protocol"] = protocol_spec()
    result["split_manifest_sha256"] = json_digest(plan)
    _csv(args.out_dir / "corrected_outer_predictions.csv", result.pop("outer_predictions"))
    _csv(args.out_dir / "inner_final_calibration.csv", calibration_rows)
    _csv(args.out_dir / "comparison_metrics.csv", [{"method": method, **stats}
         for method, stats in result["comparison"].items()])
    _csv(args.out_dir / "missingness_metrics.csv", [*cached["missingness"], *result["missingness"]],
         columns=["method", "scenario", "missing_rate", "seed", "threshold_policy", "effective_removed_fraction",
                  "removed_comparable_pair_fields", "original_comparable_pair_fields", "no_evidence_n",
                  "no_evidence_predicted_positive", "precision", "recall", "f1", "tn", "fp", "fn", "tp", "n", "confusion_matrix"])
    _json(args.out_dir / "corrected_development_results.json", result)
    _json(args.out_dir / "frozen_configuration.json", config)
    lines = ["# Development-порівняння остаточного кандидата", "",
        "Розвідувальна перевірка лише TRAIN. П'ять історичних baseline-результатів перевірено й повторно використано без перезапуску або перезапису.", "",
        "1135 pooled outer-пар: 500 позитивних і 635 негативних; 1365 міжгрупових негативів виключено. Це не незалежне підтвердження та не приріст відносно історичного REPORT.", "",
        "| Метод | Precision | Recall | F1 | TN | FP | FN | TP |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, stats in result["comparison"].items():
        lines.append(f"| {method} | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1']:.4f} | {stats['tn']} | {stats['fp']} | {stats['fn']} | {stats['tp']} |")
    lines += ["", "Confusion matrix: [[TN, FP], [FN, TP]]. Формулу corrected не змінено після перегляду результатів; кожен outer-поріг обрано тільки на внутрішніх OOF.", "",
        "## Офіційні hard negatives", "", "| Метод | N | FP | FPR |", "| --- | ---: | ---: | ---: |"]
    for method, strata in result["hard_negative_comparison"].items():
        stats = strata["hard"]
        lines.append(f"| {method} | {stats['n']} | {stats['fp']} | {stats['fpr']:.4f} |")
    lines += ["", "## Фіксація однієї фінальної конфігурації", "",
        "Final calibration використовує лише дев'ять INNER held-out наборів: raw scores повторної pair_id усереднюються один раз, кожна пара має однакову вагу. Fold-пороги не усереднюються. Outer scores не використано для нового добору параметрів. Selection-метрики не є незалежною якістю.", "",
        f"Калібровка охопила {result['final_calibration']['n_unique_pairs']} унікальних пар. Після неї виконано один full-TRAIN fit; параметри й пороги збережено у frozen_configuration.json.", "",
        "Маскування auxiliary/all_groups: 0/10/20/30/50%, seeds 11/22/33, без зміни outer-порогів. Повні P/R/F1 та confusion counts збережено в missingness_metrics.csv.", "",
        "Обмеження: TRAIN раніше аналізувався; виключення негативів змінює prevalence; після full-TRAIN refit можуть змінитися IDF/інформативність/шкала LR. Перенесення operating point не виправляється за TEST. Score не є каліброваною ймовірністю. Search Recall@K/Precision@K не вимірювалися."]
    (args.out_dir / "development_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2))
    print("Frozen parameter SHA-256:", config["parameter_sha256"])


if __name__ == "__main__":
    main()
