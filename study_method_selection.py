"""Read-only diagnostics of cached TRAIN development scores; no model selection.

Run from the project root. Does not read raw data, fit models, load the frozen
bundle, select thresholds, or open TEST results. Outputs only the new study.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from matching import FIELDS, EVIDENCE_GROUPS, Features, evidence_coverage, evidence_scores
from run_known_methods import apply_missingness

METHODS = ("fixed", "renorm")
SIM = [f"sim_{field}" for field in FIELDS]
CACHE = Path("docs/known_methods")
OUT = Path("docs/method_selection_study")


def sha(path: Path, lf: bool = False) -> str:
    value = path.read_bytes()
    return hashlib.sha256(value.replace(b"\r\n", b"\n") if lf else value).hexdigest()


def checked_cache(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Use only fixed allowlisted development paths; verify historical hashes."""
    validation = json.loads((root / CACHE / "pipeline_validation.json").read_text("utf-8"))
    provenance = {}
    for name in ("clean_predictions.csv", "development_results.json", "protocol.json"):
        rel = (CACHE / name).as_posix()
        actual = sha(root / rel)
        if actual != validation["artifact_sha256"][rel]:
            raise ValueError(f"Changed development artifact: {rel}")
        provenance[rel] = actual
    result = json.loads((root / CACHE / "development_results.json").read_text("utf-8"))
    for name, expected in result["provenance"]["code_sha256"].items():
        if sha(root / name, lf=True) != expected:
            raise ValueError(f"Changed historical source: {name}")
    protocol = json.loads((root / CACHE / "protocol.json").read_text("utf-8"))
    if protocol["protocol"] != result["protocol"]:
        raise ValueError("Historical protocols disagree")
    if result["provenance"]["test_opened"] or result["provenance"]["report_opened"]:
        raise ValueError("Unexpected source role in development cache")
    frame = pd.read_csv(root / CACHE / "clean_predictions.csv", keep_default_na=False,
                        float_precision="round_trip")
    return frame, result, provenance


def align_predictions(frame: pd.DataFrame, result: dict) -> pd.DataFrame:
    subsets = {}
    for method in METHODS:
        subset = frame.loc[frame.method == method].copy()
        if subset.empty or subset.pair_id.duplicated().any():
            raise ValueError("Missing method or repeated pair ID")
        subsets[method] = subset.set_index("pair_id").sort_index()
    fixed, renorm = (subsets[m] for m in METHODS)
    if not fixed.index.equals(renorm.index):
        raise ValueError("Methods must evaluate exactly the same pairs")
    for field in ["label", "outer_fold", "is_hard_negative", "id_left", "id_right",
                  "cluster_id_left", "cluster_id_right", "diagnostic_tags"] + SIM:
        if not fixed[field].equals(renorm[field]):
            raise ValueError(f"Misaligned pair metadata: {field}")
    out = fixed.reset_index()
    for field in SIM:
        out[field] = out[field].map(lambda value: np.nan if value == "" else float(value))
    matrix = out[SIM].to_numpy(float)
    coverage = evidence_coverage(matrix)
    if not np.allclose(coverage, out.coverage.astype(float), atol=1e-14, rtol=0):
        raise ValueError("Cached coverage disagrees with similarities")
    for method, subset in subsets.items():
        scores = evidence_scores(matrix, method)
        if not np.allclose(scores, subset.score.to_numpy(float), atol=1e-14, rtol=0):
            raise ValueError(f"Cached scores disagree: {method}")
        thresholds = np.array([
            result["folds"][int(fold)]["selected"][method]["threshold"]
            for fold in out.outer_fold
        ])
        if not np.array_equal(thresholds, subset.threshold.to_numpy(float)):
            raise ValueError(f"Threshold is not the historical inner-selected value: {method}")
        prediction = (scores >= thresholds).astype(int)
        if not np.array_equal(prediction, subset.prediction.to_numpy(int)):
            raise ValueError(f"Cached decisions disagree: {method}")
        out[f"{method}_score"] = scores
        out[f"{method}_threshold"] = thresholds
        out[f"{method}_prediction"] = prediction
    return out


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    if y.shape != pred.shape or not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("Binary aligned labels and decisions required")
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return dict(n=len(y), positives=tp+fn, negatives=fp+tn, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else 0.0,
                recall=tp/(tp+fn) if tp+fn else 0.0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                fpr=fp/(fp+tn) if fp+tn else None,
                precision_defined=bool(tp+fp), recall_defined=bool(tp+fn),
                f1_defined=bool(2*tp+fp+fn))


def comparisons(frame: pd.DataFrame, matrix: np.ndarray, f: np.ndarray, r: np.ndarray,
                *, detailed: bool = True) -> tuple[list[dict], list[dict]]:
    """Report all requested strata, including empty ones. Never select a rule."""
    y = frame.label.to_numpy(int)
    availability = ~np.isnan(matrix)
    groups = np.column_stack([availability[:, g].any(axis=1) for g in EVIDENCE_GROUPS])
    coverage = evidence_coverage(matrix)
    patterns = np.array(["".join(map(str, row.astype(int))) for row in availability])
    group_patterns = np.array(["".join(map(str, row.astype(int))) for row in groups])
    strata = [("overall", "all", np.ones(len(y), dtype=bool))]
    strata += [("coverage", f"{c:.2f}", np.isclose(coverage, c))
               for c in (0, .15, .30, .70, .85, 1)]
    strata += [("group_count", str(n), groups.sum(axis=1) == n) for n in range(4)]
    strata += [("group_pattern", "".join(p), group_patterns == "".join(p))
               for p in itertools.product("01", repeat=3)]
    if detailed:
        strata += [("availability_pattern", "".join(p), patterns == "".join(p))
                   for p in itertools.product("01", repeat=4)]
        for j, field in enumerate(FIELDS):
            strata += [(f"available_{field}", str(int(value)), availability[:, j] == value)
                       for value in (False, True)]
        strata += [("label", str(label), y == label) for label in (0, 1)]
        hard = frame.is_hard_negative.astype(str).str.lower().eq("true").to_numpy()
        strata += [("negative_type", "hard", (y == 0) & hard),
                   ("negative_type", "other", (y == 0) & ~hard)]
        tags = sorted({t for ts in frame.diagnostic_tags for t in str(ts).split(";") if t})
        for tag in tags:
            strata.append(("diagnostic_tag", tag,
                           np.array([tag in str(ts).split(";") for ts in frame.diagnostic_tags])))
        for c in (.70, .85, 1):
            for label in (0, 1):
                strata.append(("coverage_by_label", f"{c:.2f}:y={label}",
                               np.isclose(coverage, c) & (y == label)))
        for fold in sorted(frame.outer_fold.unique()):
            mask = frame.outer_fold.to_numpy() == fold
            strata.append(("fold", str(fold), mask))
            for c in (.70, .85, 1):
                strata.append(("fold_by_coverage", f"{fold}:{c:.2f}",
                               mask & np.isclose(coverage, c)))
    stats, paired = [], []
    for dimension, value, mask in strata:
        yy, ff, rr = y[mask], f[mask], r[mask]
        a, b = confusion(yy, ff), confusion(yy, rr)
        # These are descriptive support flags, not significance/power tests.
        support = ("empty" if not len(yy) else "one_class" if len(set(yy)) < 2
                   else "few_decision_disagreements" if np.sum(ff != rr) < 20
                   else "descriptive_only")
        for method, counts in zip(METHODS, (a, b)):
            stats.append(dict(dimension=dimension, stratum=value, method=method,
                              support=support, **counts))
        paired.append(dict(
            dimension=dimension, stratum=value, n=len(yy),
            positives=int(sum(yy == 1)), negatives=int(sum(yy == 0)), support=support,
            disagreements=int(sum(ff != rr)),
            fixed_only_correct=int(sum((ff == yy) & (rr != yy))),
            renorm_only_correct=int(sum((rr == yy) & (ff != yy))),
            both_correct=int(sum((ff == yy) & (rr == yy))),
            both_wrong=int(sum((ff != yy) & (rr != yy))),
            fixed_positive_renorm_negative=int(sum((ff == 1) & (rr == 0))),
            fixed_negative_renorm_positive=int(sum((ff == 0) & (rr == 1))),
            delta_f1_fixed_minus_renorm=a["f1"] - b["f1"],
        ))
    return stats, paired


def analyze(frame: pd.DataFrame, result: dict) -> dict:
    data = align_predictions(frame, result)
    matrix = data[SIM].to_numpy(float)
    f, r = (data[f"{m}_prediction"].to_numpy(int) for m in METHODS)
    clean_stats, clean_paired = comparisons(data, matrix, f, r)
    for method in METHODS:
        expected = result["clean_metrics"][method]
        actual = next(x for x in clean_stats if x["dimension"] == "overall" and x["method"] == method)
        for key in ("n", "tp", "fp", "fn", "tn", "precision", "recall", "f1"):
            if not np.isclose(actual[key], expected[key], rtol=0, atol=1e-14):
                raise ValueError(f"Clean metrics mismatch: {method}/{key}")
    stress_stats, stress_paired, identity_errors = [], [], []
    identity_errors.append(float(np.max(np.abs(
        data.fixed_score - evidence_coverage(matrix) * data.renorm_score))))
    features = Features(matrix, data.id_left.astype(str).tolist(), data.id_right.astype(str).tolist())
    historical = {(x["scenario"], x["missing_rate"], x["seed"], x["method"]): x
                  for x in result["missingness"]}
    protocol = result["protocol"]["missingness"]
    replay_checks = 0
    for scenario, rate, seed in itertools.product(
            protocol["scenarios"], protocol["rates"], protocol["seeds"]):
        masked = apply_missingness(features, rate, seed, scenario)
        values = {m: evidence_scores(masked, m) for m in METHODS}
        preds = {m: (values[m] >= data[f"{m}_threshold"].to_numpy(float)).astype(int)
                 for m in METHODS}
        identity_errors.append(float(np.max(np.abs(
            values["fixed"] - evidence_coverage(masked) * values["renorm"]))))
        for method in METHODS:
            actual = confusion(data.label.to_numpy(int), preds[method])
            expected = historical[scenario, rate, seed, method]
            if any(actual[key] != expected[key] for key in ("n", "tp", "fp", "fn", "tn")):
                raise ValueError("Replay differs from saved missingness experiment")
            replay_checks += 1
        stats, paired = comparisons(data, masked, preds["fixed"], preds["renorm"], detailed=False)
        meta = dict(scenario=scenario, missing_rate=rate, seed=seed)
        stress_stats.extend([{**meta, **row} for row in stats])
        stress_paired.extend([{**meta, **row} for row in paired])
    disagreements = data.loc[f != r].copy()
    disagreements["correct_method"] = np.where(
        disagreements.fixed_prediction == disagreements.label, "fixed", "renorm")
    errors = data.loc[(f != data.label) | (r != data.label)].copy()
    for method in METHODS:
        errors[f"{method}_error"] = np.where(
            errors[f"{method}_prediction"] == errors.label, "correct",
            np.where(errors[f"{method}_prediction"] == 1, "FP", "FN"))
    return dict(clean_metrics=clean_stats, clean_paired=clean_paired,
                stress_metrics=stress_stats, stress_paired=stress_paired,
                disagreements=disagreements, errors=errors,
                checks=dict(pairs=len(data), mask_metric_replays=replay_checks,
                            maximum_identity_absolute_error=max(identity_errors),
                            model_fits=0, threshold_selections=0, test_reads=0),
                thresholds={fold["outer_fold"]: {m: fold["selected"][m]["threshold"]
                                                for m in METHODS} for fold in result["folds"]})


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.4f}"
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    lines += ["| " + " | ".join(fmt(row[c]) for c in columns) + " |"
              for row in frame.to_dict("records")]
    return "\n".join(lines)


def export(out: Path, analysis: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("clean_metrics", "clean_paired", "stress_metrics", "stress_paired",
                 "disagreements", "errors"):
        pd.DataFrame(analysis[name]).to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    stats = pd.DataFrame(analysis["clean_metrics"])
    paired = pd.DataFrame(analysis["clean_paired"])
    columns = ["stratum", "method", "n", "positives", "negatives", "tp", "fp", "fn", "tn",
               "precision", "recall", "f1"]
    sections = ["# Числові таблиці development-діагностики",
                "Автоматично створено тільки зі збережених TRAIN outer-прогнозів. "
                "Нуль при невизначеній метриці — технічна конвенція; CSV містить *_defined. "
                "Порожні та однокласові групи не є доказом якості. "
                "У diagnostic_tag групи перетинаються; їх не можна підсумовувати."]
    for dimension in stats.dimension.unique():
        sections += [f"## {dimension}", table(stats[stats.dimension == dimension], columns),
                     table(paired[paired.dimension == dimension],
                           ["stratum", "n", "disagreements", "fixed_only_correct",
                            "renorm_only_correct", "both_wrong", "delta_f1_fixed_minus_renorm"])]
    stress = pd.DataFrame(analysis["stress_metrics"])
    whole = stress[stress.dimension == "overall"]
    summary = whole.groupby(["scenario", "missing_rate", "method"]).f1.agg(["mean", "std"]).reset_index()
    summary.to_csv(out / "stress_summary.csv", index=False)
    sections += ["## Повторення історичних масок",
                 "Середнє та sample SD трьох seeds; це ті самі пари, не незалежні вибірки.",
                 table(summary, ["scenario", "missing_rate", "method", "mean", "std"])]
    (out / "numerical_tables.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def test_status_metadata(root: Path) -> dict:
    """Inspect filenames and allowlisted readiness only, never TEST results."""
    candidates = []
    for directory in ("docs", "results", "artifacts", "data/processed"):
        base = root / directory
        if base.exists():
            for path in base.rglob("*"):
                name = path.as_posix().lower()
                if path.is_file() and any(token in name for token in (
                        "final_test", "test_result", "test_evaluation", "final_evaluation")):
                    if path.name != "final_evaluation_protocol.md":
                        candidates.append(path.relative_to(root).as_posix())
    path = root / "docs/final_method/pipeline_validation.json"
    last = json.loads(path.read_text("utf-8")) if path.exists() else {}
    opened = last.get("checks", {}).get("final_test_opened_or_hashed")
    status = ("evaluated_or_accessed_according_to_readiness" if opened is True else
              "unassessed_possible_artifacts" if candidates else
              "no_local_evaluation_evidence_last_readiness_not_opened" if opened is False
              else "unknown")
    return dict(status=status, possible_artifact_names_only=sorted(candidates),
                last_readiness=last.get("readiness"),
                last_test_opened_or_hashed=opened,
                limitation="Cannot rule out evaluation outside this workspace; "
                           "if TEST was evaluated/viewed it is not independent for new decisions.",
                test_results_read=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    out = (root / args.out_dir).resolve()
    if not out.is_relative_to((root / OUT).resolve()):
        raise ValueError("Outputs must stay inside docs/method_selection_study")
    frame, result, provenance = checked_cache(root)
    analysis = analyze(frame, result)
    export(out, analysis)
    report = dict(scope="Cached TRAIN development diagnostics only; no rule fitted or evaluated",
                  checks=analysis["checks"], thresholds=analysis["thresholds"],
                  provenance=provenance, code_sha256_lf=sha(Path(__file__), lf=True),
                  original_training_provenance=result["provenance"],
                  test_status=test_status_metadata(root),
                  decision="B: do not implement switching; investigate hard-negative contradictions",
                  support_note="Disagreement count below 20 is a descriptive flag only, "
                               "not a significance or power threshold.")
    (out / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                                        encoding="utf-8")
    print(json.dumps(analysis["checks"], indent=2))


if __name__ == "__main__":
    main()
