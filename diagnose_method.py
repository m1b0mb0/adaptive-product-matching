"""Frozen-v1 diagnostics on TRAIN and the original TUNE only; never retunes.

The runner accepts neither split paths nor a test argument. Paths, byte hashes,
TUNE membership and operating points come from the historical result artifact.
Non-TUNE validation rows are routed by cluster IDs before their labels, titles
or other attributes are validated or used. JSON decoding itself is unavoidable.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_wdc import _validate_record, read_pairs
from matching import (FIELDS, FeatureExtractor, evidence_coverage, evidence_scores,
                      identifiers, masked, norm, train_informativeness)
from run_experiment import labels, metrics

SELECTION_SEED = 20261101
REMOVALS = {"description": (1,), "brand": (2,), "identifier_only": (3,),
            "title_family": (0, 3), "all": (0, 1, 2, 3)}
METHODS = ("text", "fixed", "renorm", "adaptive")


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_frozen_sources(historical: dict) -> dict:
    checked = {}
    for name, expected in historical["metadata"]["source_sha256"].items():
        actual = sha256(name)
        if actual != expected:
            raise ValueError(f"Historical v1 source hash differs: {name}")
        checked[name] = actual
    return checked


def read_tune_only(path: str | Path, tune_clusters: set[str]) -> list[dict]:
    """Route by frozen metadata first; ignore REPORT and crossing payloads."""
    rows = []
    opener = gzip.open if Path(path).suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected JSON object for routing")
            if not all(str(row.get(f"cluster_id_{side}", "")).strip() in tune_clusters
                       for side in ("left", "right")):
                continue
            _validate_record(row, f"TUNE line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("No rows in frozen TUNE partition")
    return rows


def score(matrix: np.ndarray, method: str, info: np.ndarray, old: dict) -> np.ndarray:
    config = old["validation_selected"][method]
    return evidence_scores(matrix, method, info, gamma=config["gamma"],
                           penalty=config["penalty"],
                           prior=old["protocol"]["prior_score_reference"])


def stats(y: np.ndarray, values: np.ndarray, threshold: float) -> dict:
    result = metrics(y, (values >= threshold).astype(int))
    return {"n": len(y), **result,
            "confusion_matrix": [[result["tn"], result["fp"]], [result["fn"], result["tp"]]]}


def remove_channels(matrix: np.ndarray, name: str) -> np.ndarray:
    out = matrix.copy()
    out[:, REMOVALS[name]] = np.nan
    return out


def error_flags(pair: dict, vector: np.ndarray) -> list[str]:
    """Overlapping descriptive proxies; deliberately not a causal taxonomy."""
    flags = []
    if norm(pair["title_left"]) == norm(pair["title_right"]):
        flags.append("identical_normalized_titles")
    if not np.isnan(vector[0]) and vector[0] >= 0.5:
        flags.append("high_title_similarity_ge_0.5")
    if not np.isnan(vector[0]) and vector[0] < 0.2:
        flags.append("low_title_similarity_lt_0.2")
    for j, field in enumerate(FIELDS):
        if np.isnan(vector[j]):
            flags.append(f"{field}_unavailable")
    if vector[2] == 0:
        flags.append("observed_brand_inequality")
    if vector[2] == 1:
        flags.append("observed_brand_equality")
    if vector[3] == 0:
        flags.append("disjoint_extracted_token_sets")
    if vector[3] > 0:
        flags.append("shared_extracted_token")
    return flags


def stratum(pair: dict) -> str:
    if int(pair["label"]) == 1:
        return "positive"
    return "hard_negative" if pair.get("is_hard_negative") else "easy_negative"


def correlations(matrix: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for label, selected in (("all", np.ones(len(y), dtype=bool)),
                            ("positive", y == 1), ("negative", y == 0)):
        for i in range(len(FIELDS)):
            for j in range(i + 1, len(FIELDS)):
                keep = selected & ~np.isnan(matrix[:, i]) & ~np.isnan(matrix[:, j])
                a, b = matrix[keep, i], matrix[keep, j]
                nonconstant = len(a) >= 3 and np.ptp(a) > 0 and np.ptp(b) > 0
                rows.append({"population": label, "left": FIELDS[i], "right": FIELDS[j],
                             "both_observed_n": int(keep.sum()),
                             "pearson": float(np.corrcoef(a, b)[0, 1]) if nonconstant else None,
                             "spearman": float(pd.Series(a).rank().corr(pd.Series(b).rank())) if nonconstant else None})
    return rows


def relationships(matrix: np.ndarray, adaptive: np.ndarray, renorm: np.ndarray,
                  weighted_mean: np.ndarray, old: dict) -> dict:
    coverage = evidence_coverage(matrix)
    config = old["validation_selected"]["adaptive"]
    threshold = config["threshold"]
    renorm_threshold = old["validation_selected"]["renorm"]["threshold"]
    prior = old["protocol"]["prior_score_reference"]
    alpha = coverage ** config["gamma"]
    reconstructed = alpha * weighted_mean + (1 - alpha) * prior
    reconstruction_error = np.where(coverage > 0, abs(reconstructed - adaptive), abs(adaptive))
    disagreement = (adaptive >= threshold) != (renorm >= renorm_threshold)
    ranks = pd.DataFrame({"adaptive": adaptive, "renorm": renorm}).rank()
    # Count strict pairwise ranking inversions, excluding ties, once per unordered pair.
    da = adaptive[:, None] - adaptive[None, :]
    dr = renorm[:, None] - renorm[None, :]
    upper = np.triu(np.ones((len(matrix), len(matrix)), dtype=bool), 1)
    inversions = upper & (da * dr < 0)
    rows = []
    for c in sorted(set(coverage)):
        keep = np.isclose(coverage, c)
        a = c ** config["gamma"]
        x, yy = renorm[keep], adaptive[keep]
        slope, intercept = (np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), yy, rcond=None)[0]
                            if len(x) >= 2 and np.ptp(x) > 0 else [None, None])
        residual = yy - (slope * x + intercept) if slope is not None else np.array([])
        rows.append({"coverage": float(c), "n": int(keep.sum()), "alpha": float(a),
                     "effective_weighted_mean_threshold": float((threshold - (1 - a) * prior) / a) if a else None,
                     "adaptive_positive": int((yy >= threshold).sum()),
                     "renorm_positive": int((x >= renorm_threshold).sum()),
                     "decisions_different": int(disagreement[keep].sum()),
                     "adaptive_min_abs_margin": float(abs(yy - threshold).min()),
                     "renorm_min_abs_margin": float(abs(x - renorm_threshold).min()),
                     "ols_adaptive_from_renorm_slope": float(slope) if slope is not None else None,
                     "ols_intercept": float(intercept) if intercept is not None else None,
                     "ols_max_abs_residual": float(abs(residual).max()) if len(residual) else None,
                     "max_abs_mean_minus_renorm": float(abs(weighted_mean[keep] - x).max())})
    return {"n": len(matrix), "disagreement_n": int(disagreement.sum()),
            "score_mae": float(abs(adaptive - renorm).mean()),
            "score_max_abs_difference": float(abs(adaptive - renorm).max()),
            "spearman": float(ranks["adaptive"].corr(ranks["renorm"])),
            "strict_rank_inversion_pairs": int(inversions.sum()),
            "all_unordered_pairs": int(upper.sum()),
            "reconstruction_max_abs_error": float(reconstruction_error.max()),
            "by_coverage": rows}


def error_summary(rows: list[dict], matrix: np.ndarray, values: np.ndarray,
                  threshold: float) -> tuple[list[dict], list[dict]]:
    summaries, cases = [], []
    y = labels(rows)
    pred = values >= threshold
    for name in ("positive", "hard_negative", "easy_negative"):
        keep = np.array([stratum(p) == name for p in rows])
        if not keep.any():
            continue
        summaries.append({"stratum": name, **stats(y[keep], values[keep], threshold),
                          "error_fraction": float((pred[keep] != y[keep]).mean())})
        bad = np.where(keep & (pred != y))[0].tolist()
        flags = [error_flags(rows[i], matrix[i]) for i in bad]
        counts = {flag: sum(flag in f for f in flags) for flag in sorted({flag for f in flags for flag in f})}
        summaries[-1]["overlapping_proxy_counts_among_errors"] = counts
        summaries[-1]["error_proxy_denominator"] = len(bad)
        # High score and seeded hash samples complement each other. This is not a
        # prevalence sample; rates above always use the complete error population.
        chosen = {}
        for i in sorted(bad, key=lambda k: (-values[k], str(rows[k]["pair_id"])))[:2]:
            chosen[i] = "highest_score"
        for i in sorted(bad, key=lambda k: hashlib.sha256(
                f"{SELECTION_SEED}:{rows[k]['pair_id']}".encode()).hexdigest())[:2]:
            chosen[i] = chosen.get(i, "") + (";" if i in chosen else "") + "seeded_hash"
        for i, selection in chosen.items():
            pair = rows[i]
            cases.append({"selection": selection, "stratum": name, "error": "FN" if y[i] else "FP",
                          "pair_id": str(pair["pair_id"]), "id_left": str(pair["id_left"]),
                          "id_right": str(pair["id_right"]), "cluster_id_left": str(pair["cluster_id_left"]),
                          "cluster_id_right": str(pair["cluster_id_right"]),
                          "label": int(y[i]), "is_hard_negative": pair.get("is_hard_negative"),
                          "title_left": pair["title_left"], "title_right": pair["title_right"],
                          "brand_left": pair.get("brand_left"), "brand_right": pair.get("brand_right"),
                          "tokens_left": ";".join(sorted(identifiers(pair["title_left"]))),
                          "tokens_right": ";".join(sorted(identifiers(pair["title_right"]))),
                          **{f"sim_{field}": None if np.isnan(matrix[i, j]) else float(matrix[i, j]) for j, field in enumerate(FIELDS)},
                          "adaptive_score": float(values[i]), "threshold": threshold,
                          "proxy_flags": ";".join(error_flags(pair, matrix[i]))})
    return summaries, cases


def analyze(train: list[dict], tune: list[dict], old: dict) -> tuple[dict, list[dict], list[dict]]:
    """Fit on TRAIN only; all operating points inherited, no search routine."""
    train_offers = {str(p[f"id_{side}"]) for p in train for side in ("left", "right")}
    tune_offers = {str(p[f"id_{side}"]) for p in tune for side in ("left", "right")}
    if train_offers & tune_offers:
        raise ValueError("TRAIN/TUNE offer overlap")
    extractor = FeatureExtractor().fit(train)
    feature_sets = {"TRAIN": extractor.transform(train), "TUNE": extractor.transform(tune)}
    info = train_informativeness(feature_sets["TRAIN"].matrix, labels(train))
    expected_info = np.array([old["train_discriminativeness"][f] for f in FIELDS])
    if not np.allclose(info, expected_info, rtol=0, atol=1e-12):
        raise ValueError("Recomputed TRAIN informativeness differs from historical artifact")
    output = {"scope": "TRAIN and exact historical TUNE; no REPORT row analysis; no TEST access; no retuning",
              "frozen_parameters": old["validation_selected"],
              "prior": old["protocol"]["prior_score_reference"],
              "informativeness": dict(zip(FIELDS, info.tolist())), "splits": {}}
    all_cases, removals = [], []
    for split, rows in (("TRAIN", train), ("TUNE", tune)):
        features = feature_sets[split]
        matrix, y = features.matrix, labels(rows)
        values = {method: score(matrix, method, info, old) for method in METHODS}
        threshold = old["validation_selected"]["adaptive"]["threshold"]
        mean = evidence_scores(matrix, "adaptive", info, gamma=old["validation_selected"]["adaptive"]["gamma"],
                               penalty=0.0, use_coverage=False)
        error_stats, cases = error_summary(rows, matrix, values["adaptive"], threshold)
        for case in cases:
            case["split"] = split
            position = next(i for i, p in enumerate(rows) if str(p["pair_id"]) == case["pair_id"])
            case["renorm_score"] = float(values["renorm"][position])
        all_cases.extend(cases)
        field_stats = {}
        unique = {str(p[f"id_{side}"]): {f: p.get(f"{f}_{side}") for f in ("title", "description", "brand")}
                  for p in rows for side in ("left", "right")}
        for j, field in enumerate(FIELDS):
            keep = ~np.isnan(matrix[:, j])
            field_stats[field] = {"comparable_pairs_n": int(keep.sum()), "all_pairs_n": len(rows),
                                  "comparable_fraction": float(keep.mean()),
                                  "positive_mean": float(matrix[(y == 1) & keep, j].mean()) if ((y == 1) & keep).any() else None,
                                  "negative_mean": float(matrix[(y == 0) & keep, j].mean()) if ((y == 0) & keep).any() else None}
        source_stats = {field: {"present_offers_n": sum(bool(norm(p[field])) for p in unique.values()),
                                 "unique_offers_n": len(unique)} for field in ("title", "description", "brand")}
        source_stats["identifier"] = {"present_offers_n": sum(bool(identifiers(p["title"])) for p in unique.values()),
                                      "unique_offers_n": len(unique)}
        disagreements = []
        ap = values["adaptive"] >= threshold
        rp = values["renorm"] >= old["validation_selected"]["renorm"]["threshold"]
        for i in np.where(ap != rp)[0]:
            disagreements.append({"pair_id": str(rows[i]["pair_id"]), "label": int(y[i]),
                                  "adaptive_score": float(values["adaptive"][i]),
                                  "renorm_score": float(values["renorm"][i]),
                                  "coverage": float(evidence_coverage(matrix[i:i + 1])[0])})
        output["splits"][split] = {
            "n": len(rows), "positives": int(y.sum()), "unique_offers": len(unique),
            "clean_metrics": {m: stats(y, values[m], old["validation_selected"][m]["threshold"]) for m in METHODS},
            "field_statistics": field_stats, "unique_offer_availability": source_stats,
            "conditional_feature_correlations": correlations(matrix, y),
            "adaptive_renorm": relationships(matrix, values["adaptive"], values["renorm"], mean, old),
            "disagreement_pairs": disagreements, "adaptive_errors": error_stats}
        component_ablations = []
        for name, disabled in (("no_coverage", "use_coverage"), ("no_info", "use_info"),
                               ("no_manual_reliability", "use_reliability")):
            config = old["validation_selected"]["adaptive"]
            changed = evidence_scores(matrix, "adaptive", info, gamma=config["gamma"],
                                      penalty=config["penalty"],
                                      prior=old["protocol"]["prior_score_reference"],
                                      **{disabled: False})
            component_ablations.append({"component_removed": name, "threshold": threshold,
                                        "decisions_changed_n": int(((changed >= threshold) != ap).sum()),
                                        "score_mae_from_full": float(abs(changed - values["adaptive"]).mean()),
                                        **stats(y, changed, threshold)})
        output["splits"][split]["frozen_threshold_component_ablations"] = component_ablations
        for name in REMOVALS:
            damaged = remove_channels(matrix, name)
            for method in ("renorm", "adaptive"):
                new_values = score(damaged, method, info, old)
                threshold_m = old["validation_selected"][method]["threshold"]
                old_pred, new_pred = values[method] >= threshold_m, new_values >= threshold_m
                removals.append({"split": split, "removal": name, "method": method,
                                 "threshold": threshold_m,
                                 "scores_increased_n": int((new_values > values[method] + 1e-12).sum()),
                                 "negative_to_positive_n": int((~old_pred & new_pred).sum()),
                                 "positive_to_negative_n": int((old_pred & ~new_pred).sum()),
                                 **stats(y, new_values, threshold_m)})
    # Additional masks are TUNE diagnostics only, using the previously frozen
    # rates and seeds, not new choices selected using any metric.
    masked_rows = []
    features, y = feature_sets["TUNE"], labels(tune)
    for rate in old["protocol"]["missingness"]["rates"]:
        for seed in old["protocol"]["missingness"]["seeds"]:
            damaged = masked(features, rate, seed)
            av = score(damaged, "adaptive", info, old)
            rv = score(damaged, "renorm", info, old)
            masked_rows.append({"rate": rate, "seed": seed,
                                "disagreement_n": int(((av >= old["validation_selected"]["adaptive"]["threshold"]) !=
                                                       (rv >= old["validation_selected"]["renorm"]["threshold"])).sum())})
    output["tune_masks_frozen_seeds"] = masked_rows
    output["frozen_threshold_removals"] = removals
    return output, all_cases, removals


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", default="docs/validation/results.json")
    parser.add_argument("--out-dir", default="docs/diagnostics")
    args = parser.parse_args()
    historical = json.loads(Path(args.historical).read_text(encoding="utf-8"))
    source_hashes = check_frozen_sources(historical)
    metadata = historical["metadata"]
    train_path, valid_path = Path(metadata["train_path"]), Path(metadata["validation_path"])
    # Even a mistaken historical artifact cannot redirect this runner to the
    # benchmark gold standard test file.
    if "_train_" not in train_path.name or "_valid_" not in valid_path.name:
        raise ValueError("Historical paths must be official train and validation files")
    for name, path in (("train", train_path), ("validation", valid_path)):
        if sha256(path) != metadata[f"{name}_sha256"]:
            raise ValueError(f"Historical {name} hash differs")
    train = read_pairs(train_path)
    tune_ids = set(historical["protocol"]["partition"]["tune_cluster_ids"])
    tune = read_tune_only(valid_path, tune_ids)
    expected_n = historical["protocol"]["partition"]["tune"]["pairs"]
    if len(tune) != expected_n:
        raise ValueError("Frozen TUNE size differs")
    result, cases, removals = analyze(train, tune, historical)
    result["metadata"] = {"historical_results_sha256": sha256(args.historical),
                          "verified_v1_source_sha256": source_hashes,
                          "diagnostic_source_sha256": sha256(__file__),
                          "train_sha256": metadata["train_sha256"],
                          "validation_sha256": metadata["validation_sha256"],
                          "example_selection_seed": SELECTION_SEED,
                          "non_tune_routing": "JSON lines decoded; only cluster IDs read before routing; all other non-TUNE values ignored"}
    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(destination / "error_cases.csv", cases)
    write_csv(destination / "frozen_threshold_removals.csv", removals)
    print(f"Frozen v1 diagnosed on TRAIN={len(train)}, TUNE={len(tune)}; no parameter selection.")


if __name__ == "__main__":
    main()
