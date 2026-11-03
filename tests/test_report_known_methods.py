"""Reporting fixtures are tiny unit examples, never research observations."""
from __future__ import annotations

import copy
import csv

import pytest

from report_known_methods import (METHODS, clean_rows, error_summary, fold_rows,
                                  plot_missingness, render, summarize_missingness,
                                  validate_prediction_metrics)


def stats(f1=.5):
    return {"precision": .5, "recall": .5, "f1": f1, "tn": 3,
            "fp": 1, "fn": 1, "tp": 1, "n": 6,
            "confusion_matrix": [[3, 1], [1, 1]]}


def result_fixture():
    metrics = {method: stats() for method in METHODS}
    selected = {method: {"threshold": .3, "C": 1 if method == "logistic" else None,
                         "gamma": 1 if method == "adaptive" else None} for method in METHODS}
    return {"clean_metrics": metrics,
            "folds": [{"outer_fold": 0, "clean": copy.deepcopy(metrics), "selected": selected}],
            "hard_negatives": {method: {"hard": {"n": 3, "fp": 1, "tn": 2, "fpr": 1/3},
                                        "easy": {"n": 1, "fp": 0, "tn": 1, "fpr": 0.},
                                        "unknown_hard_flag_negatives": 0}
                               for method in METHODS},
            "missingness": [{"scenario": scenario, "missing_rate": rate, "seed": seed,
                             "method": method, "effective_removed_fraction": rate, **stats()}
                            for scenario in ("auxiliary", "all_groups")
                            for rate in (0., .1) for seed in (11, 22, 33) for method in METHODS],
            "paired_comparisons": {"renorm_vs_adaptive": {"different": 0}}}


def mask_row(**kwargs):
    return {"scenario": "auxiliary", "method": "text", "missing_rate": .1,
            "seed": 11, **stats(), **kwargs}


def test_seed_summary_uses_sample_sd_and_separates_scenarios():
    rows = [mask_row(f1=.2, seed=22), mask_row(f1=.4),
            mask_row(scenario="all_groups", f1=.8)]
    summary = summarize_missingness(rows)
    assert len(summary) == 2
    auxiliary, all_groups = summary
    assert auxiliary["f1_mean"] == pytest.approx(.3)
    assert auxiliary["f1_sd"] == pytest.approx(.2 / 2**.5)
    assert auxiliary["n_seeds"] == 2
    assert auxiliary["seeds"] == "11;22"
    assert all_groups["f1_sd"] is None
    assert all_groups["f1_mean"] == .8


def test_rejects_duplicate_pooled_seed_result():
    with pytest.raises(ValueError, match="Duplicate pooled"):
        summarize_missingness([mask_row(), mask_row()])


@pytest.mark.parametrize("changes", [{"f1": float("nan")}, {"precision": 1.1},
                                  {"tn": -1}, {"fp": .5}, {"seed": True},
                                  {"missing_rate": 10}, {"method": "test"},
                                  {"scenario": "unknown"}, {"n": 7},
                                  {"confusion_matrix": [[1, 1], [1, 3]]},
                                  {"effective_removed_fraction": -1}])
def test_rejects_invalid_results(changes):
    with pytest.raises(ValueError):
        summarize_missingness([mask_row(**changes)])


def test_clean_tables_do_not_invent_global_threshold():
    result = result_fixture()
    rows = clean_rows(result)
    assert [r["method"] for r in rows] == list(METHODS)
    assert all("threshold" not in row for row in rows)
    folds = fold_rows(result)
    assert len(folds) == 5
    assert next(r for r in folds if r["method"] == "logistic")["C"] == 1
    assert next(r for r in folds if r["method"] == "adaptive")["gamma"] == 1


def test_rejects_duplicate_outer_fold():
    result = result_fixture()
    result["folds"].append(copy.deepcopy(result["folds"][0]))
    with pytest.raises(ValueError, match="Duplicate outer fold"):
        fold_rows(result)


def test_error_tags_overlap_without_double_counting_same_tag():
    rows = [{"method": "text", "label": "0", "prediction": "1", "diagnostic_tags": "brand_conflict;brand_conflict;low_coverage"},
            {"method": "text", "label": "1", "prediction": "0", "diagnostic_tags": "low_coverage"},
            {"method": "text", "label": "1", "prediction": "1", "diagnostic_tags": "brand_conflict"}]
    summary = error_summary(rows)
    lookup = {(r["error_type"], r["diagnostic_tag"]): r["count"] for r in summary}
    assert lookup == {("FP", "ALL"): 1, ("FP", "brand_conflict"): 1,
                      ("FP", "low_coverage"): 1, ("FN", "ALL"): 1,
                      ("FN", "low_coverage"): 1}


def test_render_creates_tables_and_ukrainian_docs_without_mutating_input(tmp_path):
    result = result_fixture()
    before = copy.deepcopy(result)
    paths = render(result, tmp_path, plots=False)
    assert len(paths) == 6
    assert result == before
    with (tmp_path / "missingness_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert all(row["n_seeds"] == "3" for row in rows)
    body = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "Logistic Regression" in body and "development-оцінками" in body
    assert "не довірчий інтервал" in body


def test_prediction_reader_rejects_duplicate_pair_in_same_method(tmp_path):
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("method,pair_id,label,prediction\ntext,one,0,1\ntext,one,0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Repeated out-of-fold"):
        render(result_fixture(), tmp_path / "out", predictions, plots=False)


def test_plot_writes_both_formats_for_both_scenarios(tmp_path):
    summary = summarize_missingness(result_fixture()["missingness"])
    paths = plot_missingness(summary, tmp_path)
    assert {path.name for path in paths} == {
        f"missingness_f1_{scenario}.{extension}"
        for scenario in ("auxiliary", "all_groups") for extension in ("png", "svg")}
    assert all(path.stat().st_size > 100 for path in paths)


def prediction_fixture():
    decisions = [(0, 0), (0, 0), (0, 0), (0, 1), (1, 0), (1, 1)]
    return [{"method": method, "pair_id": str(i), "outer_fold": "0", "label": str(label),
             "prediction": str(prediction), "score": str(.8 if prediction else .1), "threshold": ".3"}
            for method in METHODS for i, (label, prediction) in enumerate(decisions)]


def test_saved_decisions_crosscheck_confusion_counts():
    rows = prediction_fixture()
    validate_prediction_metrics(rows, result_fixture())
    rows[0].update(prediction="1", score=".8")
    with pytest.raises(ValueError, match="disagree with clean metrics"):
        validate_prediction_metrics(rows, result_fixture())


def test_saved_decisions_require_same_pairs_across_methods():
    rows = prediction_fixture()
    rows[-1]["pair_id"] = "different_pair"
    with pytest.raises(ValueError, match="same pooled pairs"):
        validate_prediction_metrics(rows, result_fixture())


def test_saved_decisions_reject_score_threshold_disagreement():
    rows = prediction_fixture()
    rows[0]["score"] = ".9"
    with pytest.raises(ValueError, match="score >= threshold"):
        validate_prediction_metrics(rows, result_fixture())
