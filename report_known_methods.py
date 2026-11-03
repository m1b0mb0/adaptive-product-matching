"""Render TRAIN-only development results without fitting or selecting anything.

The input is produced by ``run_known_methods.py``. Seed standard deviations
describe artificial masking variability; they are not confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

METHODS = ("text", "fixed", "renorm", "adaptive", "logistic")
LABELS = {"text": "TF-IDF + cosine", "fixed": "Fixed weighted",
          "renorm": "Renormalization", "adaptive": "Adaptive v1",
          "logistic": "Logistic Regression"}
SCENARIOS = ("auxiliary", "all_groups")
METRICS = ("precision", "recall", "f1", "tn", "fp", "fn", "tp")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _stats(row: dict) -> None:
    for key in METRICS:
        value = _finite(row[key], key)
        if key in ("precision", "recall", "f1"):
            if not 0 <= value <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        elif value < 0 or not value.is_integer():
            raise ValueError(f"{key} must be a nonnegative integer")
    total = sum(row[k] for k in ("tn", "fp", "fn", "tp"))
    if "n" in row and row["n"] != total:
        raise ValueError("n disagrees with confusion counts")
    if "confusion_matrix" in row and row["confusion_matrix"] != [
            [row["tn"], row["fp"]], [row["fn"], row["tp"]]]:
        raise ValueError("confusion_matrix must use [[tn, fp], [fn, tp]]")


def summarize_missingness(rows: list[dict]) -> list[dict]:
    """Pool fold predictions first upstream, then summarize distinct mask seeds.

    Never average fold F1 here: each input row must be the pooled prediction
    metric for one (scenario, rate, method, seed). Repeating a seed is an error.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    for row in rows:
        scenario, method, seed = row["scenario"], row["method"], row["seed"]
        if scenario not in SCENARIOS or method not in METHODS:
            raise ValueError("Unknown scenario or method")
        rate = _finite(row["missing_rate"], "missing_rate")
        if not 0 <= rate <= 1:
            raise ValueError("missing_rate must be in [0, 1]")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        key = (scenario, method, rate, seed)
        if key in seen:
            raise ValueError(f"Duplicate pooled mask result: {key}")
        seen.add(key)
        _stats(row)
        if "effective_removed_fraction" in row:
            removed = _finite(row["effective_removed_fraction"], "effective_removed_fraction")
            if not 0 <= removed <= 1:
                raise ValueError("effective_removed_fraction must be in [0, 1]")
        grouped[(scenario, method, rate)].append(row)
    output = []
    for (scenario, method, rate), items in sorted(
            grouped.items(), key=lambda item: (SCENARIOS.index(item[0][0]),
                                              METHODS.index(item[0][1]), item[0][2])):
        entry = {"scenario": scenario, "method": method, "missing_rate": rate,
                 "n_seeds": len(items), "seeds": ";".join(str(r["seed"]) for r in sorted(items, key=lambda r: r["seed"]))}
        keys = list(METRICS)
        for optional in ("effective_removed_fraction", "no_evidence_n", "no_evidence_predicted_positive"):
            if all(optional in r for r in items):
                keys.append(optional)
        for key in keys:
            values = [float(r[key]) for r in items]
            entry[f"{key}_mean"] = mean(values)
            entry[f"{key}_sd"] = stdev(values) if len(values) > 1 else None
        output.append(entry)
    return output


def clean_rows(result: dict) -> list[dict]:
    rows = []
    for method in METHODS:
        stats = result["clean_metrics"][method]
        _stats(stats)
        rows.append({"method": method, **{key: stats[key] for key in METRICS},
                     "n": sum(stats[key] for key in ("tn", "fp", "fn", "tp"))})
    return rows


def fold_rows(result: dict) -> list[dict]:
    rows = []
    seen = set()
    for fold in result["folds"]:
        outer = fold["outer_fold"]
        if outer in seen:
            raise ValueError(f"Duplicate outer fold {outer}")
        seen.add(outer)
        for method in METHODS:
            metrics, selected = fold["clean"][method], fold["selected"][method]
            _stats(metrics)
            threshold = _finite(selected["threshold"], "threshold")
            if not 0 <= threshold <= 1:
                raise ValueError("threshold must be in [0, 1]")
            rows.append({"outer_fold": outer, "method": method, "threshold": threshold,
                         "C": selected.get("C"), "gamma": selected.get("gamma"),
                         **{key: metrics[key] for key in METRICS}})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _table(columns: tuple[str, ...], rows: list[list]) -> str:
    def escape(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return "\n".join(["| " + " | ".join(columns) + " |",
                       "| " + " | ".join("---" for _ in columns) + " |",
                       *("| " + " | ".join(escape(v) for v in row) + " |" for row in rows)])


def plot_missingness(summary: list[dict], out_dir: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "tmp" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    styles = {"text": (":", "^"), "fixed": ("-.", "s"), "renorm": ("-", "o"),
              "adaptive": ("--", "x"), "logistic": ("-", "D")}
    for scenario in SCENARIOS:
        data = [r for r in summary if r["scenario"] == scenario]
        if not data:
            continue
        figure, axis = plt.subplots(figsize=(9, 5))
        for method in METHODS:
            records = sorted((r for r in data if r["method"] == method), key=lambda r: r["missing_rate"])
            if not records:
                continue
            x = [r["missing_rate"] * 100 for r in records]
            y = [r["f1_mean"] for r in records]
            sd = [r["f1_sd"] or 0 for r in records]
            style, marker = styles[method]
            axis.plot(x, y, linestyle=style, marker=marker, linewidth=1.6, label=LABELS[method],
                      markerfacecolor="none" if method == "renorm" else None)
            axis.fill_between(x, [max(0, v - s) for v, s in zip(y, sd)],
                              [min(1, v + s) for v, s in zip(y, sd)], alpha=.1)
        title = "brand / description" if scenario == "auxiliary" else "title + derived identifier / description / brand"
        axis.set(xlabel="Additional offer-level masking, %", ylabel="Pooled development F1 (frozen fold thresholds)",
                 title=f"{title}: mean ± 1 seed SD", ylim=(0, 1))
        axis.set_xticks(sorted({r["missing_rate"] * 100 for r in data}))
        axis.grid(alpha=.25)
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "svg"):
            path = out_dir / f"missingness_f1_{scenario}.{extension}"
            figure.savefig(path, dpi=180, metadata={"Creator": "report_known_methods.py", "Date": None}
                           if extension == "svg" else None)
            paths.append(path)
        plt.close(figure)
    return paths


def _read_predictions(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for row in rows:
        if row.get("method") not in METHODS:
            raise ValueError("Unknown method in predictions")
        key = (row["method"], row["pair_id"])
        if key in seen:
            raise ValueError(f"Repeated out-of-fold prediction: {key}")
        seen.add(key)
        if row["label"] not in ("0", "1") or row["prediction"] not in ("0", "1"):
            raise ValueError("Prediction labels must be binary")
    return rows


def validate_prediction_metrics(rows: list[dict], result: dict) -> None:
    """Cross-check reported counts and identical pooled pairs from saved decisions."""
    grouped = {method: [] for method in METHODS}
    for row in rows:
        grouped[row["method"]].append(row)
        score = float(row["score"])
        threshold = float(row["threshold"])
        if not math.isfinite(score) or not math.isfinite(threshold) or not (0 <= score <= 1 and 0 <= threshold <= 1):
            raise ValueError("Saved scores and thresholds must be finite in [0, 1]")
        if int(row["prediction"]) != int(score >= threshold):
            raise ValueError("Saved prediction disagrees with score >= threshold")
    reference = None
    for method, items in grouped.items():
        identities = {(r["pair_id"], r["label"], r.get("outer_fold", "")) for r in items}
        if reference is None:
            reference = identities
        elif identities != reference:
            raise ValueError("Methods do not evaluate the same pooled pairs and labels")
        counts = Counter((int(r["label"]), int(r["prediction"])) for r in items)
        actual = {"tn": counts[(0, 0)], "fp": counts[(0, 1)],
                  "fn": counts[(1, 0)], "tp": counts[(1, 1)]}
        expected = result["clean_metrics"][method]
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError(f"Saved predictions disagree with clean metrics for {method}")


def error_summary(rows: list[dict]) -> list[dict]:
    """Describe existing error tags; tags are overlapping diagnostics, not causes."""
    counts: Counter = Counter()
    for row in rows:
        label, prediction = int(row["label"]), int(row["prediction"])
        kind = "FP" if prediction == 1 and label == 0 else "FN" if prediction == 0 and label == 1 else None
        if kind is None:
            continue
        counts[(row["method"], kind, "ALL")] += 1
        for tag in set(filter(None, row.get("diagnostic_tags", "").split(";"))):
            counts[(row["method"], kind, tag)] += 1
    return [{"method": method, "error_type": kind, "diagnostic_tag": tag, "count": value}
            for (method, kind, tag), value in sorted(counts.items(), key=lambda item: (
                METHODS.index(item[0][0]), item[0][1], item[0][2]))]


def _results_markdown(result: dict, clean: list[dict], folds: list[dict], summary: list[dict]) -> str:
    clean_table = _table(("Метод", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP"), [
        [LABELS[r["method"]], *[_fmt(r[key]) for key in ("precision", "recall", "f1")],
         *[r[key] for key in ("tn", "fp", "fn", "tp")]] for r in clean])
    fold_table = _table(("Outer fold", "Метод", "Поріг", "C", "gamma", "F1"), [
        [r["outer_fold"], LABELS[r["method"]], _fmt(r["threshold"], 2), r["C"] or "—",
         r["gamma"] or "—", _fmt(r["f1"])] for r in folds])
    hard = []
    for method in METHODS:
        for kind in ("hard", "easy"):
            stats = result.get("hard_negatives", {}).get(method, {}).get(kind)
            if stats is None:
                continue
            hard.append([LABELS[method], kind, stats["n"], stats["fp"], stats["tn"], _fmt(stats["fpr"])])
    prior_rows = []
    if result.get("folds") and all("split" in f and "inner_folds" in f["split"] for f in result["folds"]):
        groups = {"Inner fit": [], "Outer fit": [], "Inner pooled evaluation": [], "Outer evaluation": []}
        for fold in result["folds"]:
            split = fold["split"]
            for name, key in (("Outer fit", "fit_summary"), ("Outer evaluation", "eval_summary")):
                groups[name].append(split[key]["positives"] / split[key]["n"])
            inner = split["inner_folds"]
            groups["Inner fit"].extend(f["fit_summary"]["positives"] / f["fit_summary"]["n"] for f in inner)
            groups["Inner pooled evaluation"].append(
                sum(f["eval_summary"]["positives"] for f in inner) / sum(f["eval_summary"]["n"] for f in inner))
        prior_rows = [[name, f"{100*min(values):.1f}%", f"{100*max(values):.1f}%"] for name, values in groups.items()]
    prior_lines = (["Відкидання негативів між fold змінює class prevalence на кожному рівні:", "",
                    _table(("Частина", "Мінімум позитивів", "Максимум позитивів"), prior_rows), "",
                    "Refit може змінювати intercept/score scale LR і перенесення внутрішнього порога. "
                    "Це правдоподібне обмеження, не доведена єдина причина результату; post-hoc корекція не проводилася.", ""]
                   if prior_rows else [])
    lines = ["# Development-порівняння відомих методів", "",
             "Джерело: `development_results.json`. Це вкладена групова перевірка всередині офіційного TRAIN; "
             "результати є development-оцінками. Фінальний TEST, попередній REPORT і непідтверджені FPV-мітки не використано.", "",
             "Метрики обчислено з об’єднаних outer-held-out рішень, а не усередненням F1 окремих fold. "
             "Кожна включена пара оцінюється один раз; навчений класифікатор і поріг можуть відрізнятися між fold.", "",
             *prior_lines,
             clean_table, "", "Confusion matrix має порядок `[[TN, FP], [FN, TP]]`. "
             "Таблиця не визначає єдиного порога або фінального навченого артефакту.", "",
             "## Параметри, обрані лише у внутрішній development-перевірці", "", fold_table, "",
             "## Складні негативні пари", "",
             "Використано офіційний прапорець складності негативної пари; нижче FPR = FP / (FP + TN). "
             "F1 на самій негативній підмножині не є показовою метрикою.", "",
             _table(("Метод", "Негативи", "N", "FP", "TN", "FPR"), hard), "",
             "## Стійкість до додаткових пропусків", "",
             "Моделі, статистичні перетворення та пороги зафіксовано за чистими навчальними частинами кожного outer fold. "
             "На замаскованих даних нічого не перенавчається й не підбирається. Маски однакові для всіх методів "
             "і залежать від offer ID; title та похідний identifier приховуються разом.", "",
             "Значення — середнє ± вибіркове SD за трьома seed; це опис чутливості до штучних масок, "
             "не довірчий інтервал і не міждоменна невизначеність. Номінальна частка маскування "
             "стосується пропозицій/груп; фактична втрата доступних парних similarities може бути іншою. "
             "`effective_removed_fraction` рахує similarity-канали, а не незалежні групи; приховування "
             "title може одночасно прибрати два корельовані канали title/identifier.", ""]
    for scenario in SCENARIOS:
        data = [r for r in summary if r["scenario"] == scenario]
        rates = sorted({r["missing_rate"] for r in data})
        lines.extend([f"### Сценарій `{scenario}`", "", _table(
            ("Метод", *(f"{rate:.0%}" for rate in rates)), [
                [LABELS[method], *[next((_fmt(r["f1_mean"]) + " ± " + _fmt(r["f1_sd"]) for r in data
                  if r["method"] == method and r["missing_rate"] == rate), "—") for rate in rates]]
                for method in METHODS]), "", f"![F1: {scenario}](missingness_f1_{scenario}.png)", ""])
    no_evidence = [r for r in summary if r["scenario"] == "all_groups" and
                   r["missing_rate"] == .5 and "no_evidence_n_mean" in r]
    if no_evidence:
        lines.extend(["### Повна відсутність свідчень за 50% маскування всіх груп", "",
                      "Логістичний baseline за нульового вектора використовує навчений intercept; "
                      "евристики повертають нуль. Таблиця показує середню за seed кількість таких пар "
                      "і позитивних рішень серед них; це діагностика поведінки, а не нове правило відсікання.", "",
                      _table(("Метод", "Без усіх свідчень, mean", "Позитивних серед них, mean"), [
                          [LABELS[r["method"]], _fmt(r["no_evidence_n_mean"], 2),
                           _fmt(r["no_evidence_predicted_positive_mean"], 2)] for r in no_evidence]), ""])
    lines.extend(["## Парні рішення", "", "```json", json.dumps(result.get("paired_comparisons", {}), ensure_ascii=False, indent=2),
                  "```", "", "Повні нерounded значення, split manifests, exclusions, параметри й контрольні суми збережено у JSON/CSV. "
                  "Порівняння зі старим REPORT не використовується як підтвердження переваги.", ""])
    return "\n".join(lines).replace("нерounded", "неокруглені")


def _error_markdown(result: dict, rows: list[dict], summary: list[dict]) -> str:
    tag_table = _table(("Метод", "Помилка", "Діагностичний тег", "Кількість"), [
        [LABELS[r["method"]], r["error_type"], r["diagnostic_tag"], r["count"]] for r in summary])
    lines = ["# Аналіз помилок на TRAIN development", "",
             "Нижче аналізуються outer-held-out пари з TRAIN. Мітки FP/FN є помилками відносно WDC gold; "
             "зміст товарних сторінок повторно не перевірявся. Автоматичні теги описують ознаки, "
             "можуть перетинатися і не доводять причину помилки.", "", tag_table, "",
             "Повний перелік — `errors.csv`; усі рішення, включно з правильними, — `clean_predictions.csv`. "
             "Оцінювання hard negatives наведено в `results.md`; офіційний hard-прапорець не визначається за score моделі.", "",
             "## Приклади FP та FN", "",
             "Наведено детерміновано перші дві помилки кожного типу для кожного методу за pair ID, "
             "без відбору найпереконливіших прикладів. Назви — оригінальні значення; довгі назви скорочено для читання.", ""]
    for method in METHODS:
        examples = []
        for kind in ("FP", "FN"):
            choices = sorted((r for r in rows if r["method"] == method and
                              ("FP" if r["label"] == "0" and r["prediction"] == "1" else
                               "FN" if r["label"] == "1" and r["prediction"] == "0" else "") == kind),
                             key=lambda r: r["pair_id"])[:2]
            for row in choices:
                examples.append([kind, row["pair_id"], row.get("is_hard_negative", ""),
                                 row.get("title_left", "")[:180], row.get("title_right", "")[:180],
                                 _fmt(float(row["score"])), _fmt(float(row["threshold"]), 2),
                                 row.get("diagnostic_tags", "")])
        lines.extend([f"### {LABELS[method]}", "", _table(
            ("Тип", "Pair ID", "Hard", "Ліва назва", "Права назва", "Score", "Поріг", "Теги"), examples), ""])
    if rows:
        by_method = {method: {r["pair_id"]: r for r in rows if r["method"] == method}
                     for method in METHODS}
        changes = Counter()
        adaptive, logistic = by_method["adaptive"], by_method["logistic"]
        for pair_id in adaptive.keys() & logistic.keys():
            left, right = adaptive[pair_id], logistic[pair_id]
            if left["prediction"] == "1" and right["prediction"] == "0":
                changes["corrected_fp" if right["label"] == "0" else "introduced_fn"] += 1
            elif left["prediction"] == "0" and right["prediction"] == "1":
                changes["introduced_fp" if right["label"] == "0" else "corrected_fn"] += 1
        lines.extend(["## Що змінила Logistic Regression відносно Adaptive v1", "",
                      _table(("Перехід", "Кількість"), [[name, changes[name]] for name in
                              ("corrected_fp", "introduced_fn", "introduced_fp", "corrected_fn")]), "",
                      "Менша кількість загальних помилок не гарантує вищого F1: виправлення FP та "
                      "поява FN змінюють баланс Precision/Recall. Це зафіксовані парні рішення, "
                      "а не результат нового підбору порога.", ""])
        lr_fp = [r for r in logistic.values() if r["label"] == "0" and r["prediction"] == "1"]
        lr_fn = [r for r in logistic.values() if r["label"] == "1" and r["prediction"] == "0"]
        hard_fp = sum(r.get("is_hard_negative", "").casefold() == "true" for r in lr_fp)
        missing_fn = sum("some_unavailable_channels" in r.get("diagnostic_tags", "").split(";") for r in lr_fn)
        identifiers = [r for r in lr_fn if r.get("sim_identifier", "") != ""]
        zero_ids = sum(float(r["sim_identifier"]) == 0 for r in identifiers)
        brands = sum(r.get("sim_brand", "") != "" for r in lr_fn)
        lines.extend([f"Серед {len(lr_fp)} FP Logistic Regression офіційний hard-прапорець мають {hard_fp}. "
                      f"Серед {len(lr_fn)} FN {missing_fn} мають хоча б один недоступний канал; "
                      f"identifier доступний у {len(identifiers)}, із них {zero_ids} мають нульовий Jaccard; "
                      f"brand порівнюється лише у {brands}. Ці ознаки можуть співіснувати; "
                      "їхня наявність не доводить причини FN.", ""])
        focused = (("12615558#57337098", "Logitech H570e / H390: різні model-like позначення; identifier недоступний. "
                    "Чотирисимвольний H390 не проходить незмінений мінімум довжини 5 після канонізації."),
                   ("47146148#46246978", "Instax Square / Instax mini Rainbow: спільні слова не кодують усі відмінності формату товару."),
                   ("56034511#80086981", "TP-Link T2U / T9UH: помилка залишається навіть за спостережуваного identifier Jaccard = 0."))
        focused_rows = []
        for pair_id, interpretation in focused:
            row = logistic.get(pair_id)
            if row is not None and row["label"] == "0" and row["prediction"] == "1":
                focused_rows.append([pair_id, row.get("title_left", "")[:180], row.get("title_right", "")[:180],
                                     row.get("sim_identifier", "") or "недоступний", interpretation])
        if focused_rows:
            lines.extend(["### Ілюстрації обмежень лексичних ознак", "",
                          "Ці приклади обрано для пояснення різних механізмів, а не оцінювання їхньої частоти. "
                          "Вони є FP відносно незмінених WDC labels; товарні сторінки не перевірялися.", "",
                          _table(("Pair ID", "Ліва назва", "Права назва", "Identifier", "Спостереження"), focused_rows), ""])
        suspect = by_method["text"].get("12432797#32885383")
        if suspect is not None and suspect["label"] == "1":
            lines.extend(["### Пара для перевірки якості gold labels", "",
                          "Gold-positive `12432797#32885383` містить назви `Canon 24-70mm f/2.8L II USM EF Lens` "
                          "та `Nikon AF-S NIKKOR FX 50mm F1.8G Lens`. Видима суперечність бренду й моделі "
                          "потребує ручного дослідження джерел та кластерної анотації. Тут мітка збережена, "
                          "пара не видаляється з метрик і не перерозмічається автоматично. "
                          "Цей приклад застерігає від інтерпретації кожного gold-relative FN як очевидної помилки алгоритму.", ""])
    lines.extend(["## Межі інтерпретації", "",
                  "Однакові brand або title/model-like токени можуть належати різним модифікаціям; "
                  "низька текстова подібність може приховувати відповідність за різного опису. Ці загальні механізми "
                  "потрібно відрізняти від підтвердженої ручною перевіркою причини кожної конкретної помилки. "
                  "Окремі технічні характеристики в цьому експерименті не вилучаються.", "",
                  "Після перегляду цих development-помилок зміни методу потребуватимуть нової незалежної "
                  "перевірки. Наявні результати не є дозволом оптимізувати TEST або оголосити наукову новизну.", ""])
    return "\n".join(lines)


def render(result: dict, output_dir: Path, predictions_path: Path | None = None,
           *, plots: bool = True) -> list[Path]:
    """Create reports from immutable results, with no estimator imports or fits."""
    clean, folds = clean_rows(result), fold_rows(result)
    summary = summarize_missingness(result["missingness"])
    predictions = _read_predictions(predictions_path)
    if predictions_path is not None:
        validate_prediction_metrics(predictions, result)
    errors = error_summary(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"clean_metrics.csv": clean, "fold_metrics.csv": folds,
              "missingness_summary.csv": summary, "error_summary.csv": errors}
    paths = []
    for name, rows in tables.items():
        path = output_dir / name
        write_csv(path, rows)
        paths.append(path)
    for name, body in (("results.md", _results_markdown(result, clean, folds, summary)),
                       ("error_analysis.md", _error_markdown(result, predictions, errors))):
        path = output_dir / name
        path.write_text(body, encoding="utf-8")
        paths.append(path)
    if plots:
        paths.extend(plot_missingness(summary, output_dir))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/known_methods/development_results.json"))
    parser.add_argument("--predictions", type=Path, default=Path("docs/known_methods/clean_predictions.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/known_methods"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    paths = render(result, args.out_dir, args.predictions, plots=not args.no_plots)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
