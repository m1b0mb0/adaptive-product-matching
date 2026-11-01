"""Small fixtures verify artifact logic; fixture metrics are not research results."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

import report_validation

from report_validation import (
    ABLATIONS,
    MAIN_METHODS,
    build_markdown,
    clean_rows,
    fixed_ablation_rows,
    generate_report,
    summarize_missingness,
    tuning_rows,
)


def metric(f1=.5):
    return {"precision": .5, "recall": .5, "f1": f1,
            "tn": 7, "fp": 1, "fn": 1, "tp": 1,
            "confusion_matrix": [[7, 1], [1, 1]]}


def fixture_result():
    methods = (*MAIN_METHODS, *ABLATIONS)
    return {
        "protocol": {"tuning": "cluster-disjoint tuning", "reporting": "cluster-disjoint report",
                     "validation_mode": "fallback_seen", "partition": {"seed": 20261101}},
        "validation_selected": {name: {"threshold": .61, "gamma": .5, "penalty": 0,
                                        "validation": metric(.7)} for name in methods},
        "validation_metrics": {name: metric(.5) for name in methods},
        "ablation_fixed_threshold_metrics": {
            name: {"threshold": .61, **metric(.4)} for name in ABLATIONS},
        "missingness": {"rows": [
            {"method": name, "missing_rate": rate, "seed": seed, "scope": "report", **metric(.5 - rate / 2)}
            for name in methods for rate in (0, .1, .2, .3, .5) for seed in (11, 22, 33)]},
        "paired_error_analysis": [{"comparator": "renorm", "both_correct": 6, "both_wrong": 0,
                                   "adaptive_only_correct": 2, "comparator_only_correct": 2,
                                   "rescued_fp": 1, "rescued_fn": 1,
                                   "introduced_fp": 1, "introduced_fn": 1}],
        "limitations": ["Fixture only; not a scientific result."],
    }


def test_summary_uses_sample_sd_instead_of_population_sd():
    rows = [{"method": "adaptive", "missing_rate": .1, "seed": seed, **metric(value)}
            for seed, value in ((11, .2), (22, .4), (33, .6))]
    summary, = summarize_missingness(rows)
    assert summary["n_seeds"] == 3
    assert summary["f1_mean"] == pytest.approx(.4)
    assert summary["f1_sd"] == pytest.approx(.2)
    assert summary["tn_mean"] == 7
    assert summary["tn_sd"] == 0


def test_single_seed_has_undefined_sd():
    summary, = summarize_missingness([{"method": "text", "missing_rate": 0, "seed": 11, **metric()}])
    assert summary["f1_sd"] is None


def test_duplicate_seed_cannot_inflate_replicate_count():
    row = {"method": "adaptive", "missing_rate": .2, "seed": 11, **metric()}
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_missingness([row, row.copy()])


@pytest.mark.parametrize("key,value", [("f1", math.nan), ("precision", 1.1), ("tn", -.1),
                                      ("fp", .5), ("missing_rate", 1.1), ("f1", True)])
def test_summary_refuses_invalid_values(key, value):
    row = {"method": "adaptive", "missing_rate": .2, "seed": 11, **metric()}
    row[key] = value
    with pytest.raises(ValueError):
        summarize_missingness([row])


def test_clean_report_uses_report_metrics_but_tuning_threshold():
    result = fixture_result()
    clean = clean_rows(result)
    tuned = tuning_rows(result)
    assert {row["f1"] for row in clean} == {.5}
    assert {row["f1"] for row in tuned} == {.7}
    assert {row["threshold"] for row in clean} == {.61}
    fixed = fixed_ablation_rows(result)
    assert {row["f1"] for row in fixed} == {.4}
    assert {row["threshold"] for row in fixed} == {.61}


def test_markdown_distinguishes_cluster_disjoint_report_from_tuning():
    result = fixture_result()
    markdown = build_markdown(result, clean_rows(result), summarize_missingness(result["missingness"]["rows"]),
                              source_path="docs/validation/results.json")
    assert "окремій reporting-частині" in markdown
    assert "fallback_seen" in markdown
    assert "оптимістичне зміщення добору" not in markdown
    assert "не довірчий інтервал" in markdown
    assert "ddof=1" in markdown
    assert "Пороги і параметри clean tuning заморожено" in markdown
    assert "rescued_fp" in markdown
    assert "0.5000" in markdown
    assert "0.7000" not in markdown


def test_markdown_reports_worse_adaptive_error_movement_without_retuning():
    result = fixture_result()
    result["validation_metrics"]["renorm"]["f1"] = .6
    markdown = build_markdown(result, clean_rows(result), summarize_missingness(result["missingness"]["rows"]),
                              source_path="docs/validation/results.json")
    assert "Проти `renorm` adaptive має нижчий F1" in markdown
    assert "додано FP=1 і FN=1" in markdown
    assert "параметри в цьому запуску не змінюються" in markdown
    assert "seen-validation" in markdown


def test_generate_report_writes_csv_figures_and_preserves_source(tmp_path):
    source = tmp_path / "results.json"
    original = json.dumps(fixture_result(), ensure_ascii=False)
    source.write_text(original, encoding="utf-8")
    report = tmp_path / "validation_results.md"
    paths = generate_report(source, tmp_path / "artifacts", report)
    names = {path.name for path in paths}
    assert names >= {"metrics.csv", "tuning_metrics.csv", "missingness.csv", "missingness_summary.csv",
                     "ablations.csv", "ablation_fixed_threshold_metrics.csv", "paired_errors.csv",
                     "f1_missingness.png", "f1_missingness.svg", "f1_missingness_ablations.png",
                     "f1_missingness_ablations.svg", "validation_results.md"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    with (tmp_path / "artifacts" / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(MAIN_METHODS) + len(ABLATIONS)
    assert {float(row["f1"]) for row in rows} == {.5}
    assert (tmp_path / "artifacts" / "f1_missingness.png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (tmp_path / "artifacts" / "f1_missingness.svg").read_text(encoding="utf-8")
    assert "artifacts/f1_missingness.png" in report.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == original


def test_markdown_uses_actual_class_counts_and_evidence_statistics():
    result = fixture_result()
    statistics = {"pairs": 10, "labels": {"0": 8, "1": 2}, "unique_offers": 14, "unique_clusters": 4}
    result["data_summary"] = {"train": statistics, "validation": statistics}
    result["protocol"]["partition"].update({"tune": statistics, "report": statistics,
                                          "tune_cluster_ids": ["a", "b"], "report_cluster_ids": ["c", "d"]})
    result["by_evidence"] = [{"stratum": "identifier_missing", "method": "renorm", "n": 10,
                              "positives": 2, **metric()}]
    result["train_field_statistics"] = {"title": {"factor": .9, "positive_n": 2, "positive_mean": .8,
                                                 "negative_n": 8, "negative_mean": .6}}
    for row in result["missingness"]["rows"]:
        row["effective_removed_fraction"] = .19 if row["missing_rate"] == .1 else row["missing_rate"]
    markdown = build_markdown(result, clean_rows(result), summarize_missingness(result["missingness"]["rows"]),
                              source_path="docs/validation/results.json")
    assert "| Train | 10 | 2 | 8 | 20.00% | 14 | 4 |" in markdown
    assert "| 10% | 19.00% | 0.00% |" in markdown
    assert "identifier_missing" in markdown
    assert "0.9000" in markdown
    assert '"full_ids": "See results.json"' in markdown


@pytest.mark.parametrize("mutation,error", [("test", "test_metrics"), ("scope", "scope='report'")])
def test_generate_report_refuses_test_results_and_nonreport_scope(tmp_path, mutation, error):
    result = fixture_result()
    if mutation == "test":
        result["test_metrics"] = {"text": metric()}
    else:
        result["missingness"]["rows"][0]["scope"] = "test"
    source = tmp_path / "results.json"
    source.write_text(json.dumps(result), encoding="utf-8")
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match=error):
        generate_report(source, output_dir, tmp_path / "report.md")
    assert not output_dir.exists()


def test_verification_sidecar_requires_exact_source_hash(tmp_path):
    source = tmp_path / "results.json"
    source.write_text(json.dumps(fixture_result()), encoding="utf-8")
    (tmp_path / "independent_verification.json").write_text(json.dumps({"results_sha256": "stale"}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 differs"):
        generate_report(source, tmp_path / "output", tmp_path / "report.md")
    assert not (tmp_path / "output").exists()


def test_verified_mask_decisions_are_distinguished_from_equal_scores(tmp_path, monkeypatch):
    source = tmp_path / "results.json"
    original = json.dumps(fixture_result())
    source.write_text(original, encoding="utf-8")
    verification = {
        "results_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "clean_metric_checks": [{"method": "adaptive", "manual_confusion_and_metrics_match": True}],
        "adaptive_renorm_mask_agreement": [
            {"missing_rate": rate, "seed": seed, "different_decisions": 0, "max_abs_score_difference": .12}
            for rate in (0, .1, .2, .3, .5) for seed in (11, 22, 33)],
    }
    (tmp_path / "independent_verification.json").write_text(json.dumps(verification), encoding="utf-8")
    monkeypatch.setattr(report_validation, "plot_missingness", lambda *args, **kwargs: [])
    report = tmp_path / "report.md"
    generate_report(source, tmp_path / "output", report)
    markdown = report.read_text(encoding="utf-8")
    assert "індивідуальних рішень" in markdown
    assert "15 умовах" in markdown
    assert "|Δscore|=0.12000" in markdown
    assert "1/1 clean-методів" in markdown
    assert "independent_verification.json" in markdown
    assert source.read_text(encoding="utf-8") == original
