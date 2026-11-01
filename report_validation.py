"""Render validation-only experiment artifacts; never fit models or select scores.

The source JSON is produced by ``run_validation.py``. These reporting functions
also accept tiny fixtures for regression tests, which are not research results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

MAIN_METHODS = ("text", "fixed", "renorm", "adaptive")
ABLATIONS = ("adaptive_no_coverage", "adaptive_no_info", "adaptive_no_reliability")
METRICS = ("precision", "recall", "f1", "tn", "fp", "fn", "tp")
METHOD_LABELS = {
    "text": "TF-IDF + cosine (title)",
    "fixed": "Fixed weighted",
    "renorm": "Available-attribute renormalization",
    "adaptive": "Adaptive evidence-aware",
    "adaptive_no_coverage": "Adaptive без coverage",
    "adaptive_no_info": "Adaptive без informativeness",
    "adaptive_no_reliability": "Adaptive без reliability",
    "adaptive_no_penalty": "Adaptive без conflict penalty",
}


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def summarize_missingness(rows: list[dict]) -> list[dict]:
    """Summarize independent mask seeds using descriptive sample SD (ddof=1).

    Each (method, rate, seed) is one observation. Duplicate seeds must not make
    the sample appear larger. A single seed has undefined SD, represented by None.
    """
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    for row in rows:
        method, seed = row["method"], row["seed"]
        rate = _number(row["missing_rate"], name="missing_rate")
        if not 0 <= rate <= 1:
            raise ValueError("missing_rate must be in [0, 1]")
        key = (method, rate, seed)
        if key in seen:
            raise ValueError(f"Duplicate method/rate/seed: {key}")
        seen.add(key)
        for metric in METRICS:
            value = _number(row[metric], name=metric)
            if metric in ("precision", "recall", "f1") and not 0 <= value <= 1:
                raise ValueError(f"{metric} must be in [0, 1]")
            if metric in ("tn", "fp", "fn", "tp") and (value < 0 or not value.is_integer()):
                raise ValueError(f"{metric} must be a nonnegative integer")
        grouped[(method, rate)].append(row)
    order = {name: i for i, name in enumerate((*MAIN_METHODS, *ABLATIONS))}
    output = []
    for (method, rate), records in sorted(grouped.items(), key=lambda x: (order.get(x[0][0], 999), x[0])):
        summary = {"method": method, "missing_rate": rate, "n_seeds": len(records)}
        for metric in METRICS:
            values = [float(row[metric]) for row in records]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_sd"] = stdev(values) if len(values) > 1 else None
        output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict], columns: list[str] | tuple[str, ...] | None = None) -> None:
    """Write stable UTF-8 CSV without rounding the machine-readable numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_rows(result: dict) -> list[dict]:
    """Flatten reporting metrics and operating points selected on clean tuning."""
    selected = result["validation_selected"]
    explicit = result.get("validation_metrics", {})
    output = []
    for method, config in selected.items():
        values = explicit.get(method, config.get("validation"))
        if values is None:
            raise ValueError(f"Missing validation metrics for {method}")
        row = {"method": method, "threshold": config["threshold"],
               "gamma": config.get("gamma"), "penalty": config.get("penalty"),
               **{metric: values[metric] for metric in METRICS}}
        output.append(row)
    return output


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def tuning_rows(result: dict) -> list[dict]:
    output = []
    for method, config in result["validation_selected"].items():
        if "validation" in config:
            output.append({"method": method, "threshold": config["threshold"],
                           "gamma": config.get("gamma"), "penalty": config.get("penalty"),
                           **{metric: config["validation"][metric] for metric in METRICS}})
    return output


def fixed_ablation_rows(result: dict) -> list[dict]:
    full_threshold = result["validation_selected"].get("adaptive", {}).get("threshold")
    output = []
    for method, values in result.get("ablation_fixed_threshold_metrics", {}).items():
        actual = values.get("metrics", values)
        output.append({"method": method, "threshold": values.get("threshold", full_threshold),
                       **{metric: actual[metric] for metric in METRICS}})
    return output


def _markdown_table(columns: tuple[str, ...], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |" for row in rows)
    return "\n".join(lines)


def _compact_protocol(protocol: dict) -> dict:
    """Keep the Markdown protocol readable; the JSON retains the complete grid."""
    compact = dict(protocol)
    for key, value in compact.items():
        if "threshold" in key and isinstance(value, list) and len(value) > 10:
            compact[key] = {"count": len(value), "min": min(value), "max": max(value),
                            "full_grid": "See results.json"}
        elif key == "partition" and isinstance(value, dict):
            partition = dict(value)
            for name, item in partition.items():
                if name.endswith("cluster_ids") and isinstance(item, list):
                    partition[name] = {"count": len(item), "full_ids": "See results.json"}
                elif name in ("tune", "report") and isinstance(item, dict):
                    partition[name] = {k: item[k] for k in
                                       ("pairs", "labels", "unique_offers", "unique_clusters", "observed_negative_hard_rate")
                                       if k in item}
            compact[key] = partition
    return compact


def plot_missingness(summary: list[dict], output_dir: Path, *, methods: tuple[str, ...], stem: str) -> list[Path]:
    """Plot means and ±1 sample SD; bands are not confidence intervals."""
    # Set before importing pyplot: report generation works without a display and
    # never needs to write a font cache outside the workspace.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "tmp" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    visible_bounds = []
    styles = {"text": (":", "^"), "fixed": ("-.", "s"), "renorm": ("-", "o"),
              "adaptive": ("--", "x")}
    for method in methods:
        records = sorted((r for r in summary if r["method"] == method), key=lambda x: x["missing_rate"])
        if not records:
            continue
        x = [r["missing_rate"] * 100 for r in records]
        y = [r["f1_mean"] for r in records]
        sd = [r["f1_sd"] or 0 for r in records]
        style, marker = styles.get(method, ("-", "o"))
        axis.plot(x, y, marker=marker, linestyle=style, linewidth=1.7,
                  markerfacecolor="none" if method == "renorm" else None,
                  label=METHOD_LABELS.get(method, method))
        lower, upper = [max(0, v - s) for v, s in zip(y, sd)], [min(1, v + s) for v, s in zip(y, sd)]
        axis.fill_between(x, lower, upper, alpha=.1)
        visible_bounds.extend(lower + upper)
    y_min = max(0, min(visible_bounds) - .01) if visible_bounds else 0
    y_max = min(1, max(visible_bounds) + .01) if visible_bounds else 1
    axis.set(xlabel="Додаткове маскування brand / description на рівні пропозицій, %",
             ylabel="F1 на validation report (пороги зафіксовані)", ylim=(y_min, y_max),
             title="Стійкість до пропусків: середнє ± 1 SD масок")
    axis.set_xticks(sorted({r["missing_rate"] * 100 for r in summary}))
    axis.grid(alpha=.25)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=180,
                       metadata={"Creator": "report_validation.py", "Date": None} if suffix == "svg" else None)
        paths.append(path)
    plt.close(figure)
    return paths


def _clean_table(rows: list[dict], methods: tuple[str, ...]) -> str:
    by_method = {r["method"]: r for r in rows}
    table = []
    for name in methods:
        if name not in by_method:
            continue
        row = by_method[name]
        table.append([name, *[_fmt(row[key]) for key in ("threshold", "precision", "recall", "f1")],
                      *[row[key] for key in ("tn", "fp", "fn", "tp")]])
    return _markdown_table(("Метод", "Поріг", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP"), table)


def build_markdown(result: dict, clean: list[dict], summary: list[dict], *, source_path: str,
                   artifacts_link: str = "validation") -> str:
    """Generate a Ukrainian report using measured values only."""
    rates = sorted({r["missing_rate"] for r in summary})
    seeds = sorted({r["seed"] for r in result["missingness"]["rows"]})
    protocol = result.get("protocol", {})
    held_out = bool(protocol.get("tuning") and protocol.get("reporting"))
    selection_note = (
        "Пороги й гіперпараметри обрано лише на tuning-частині validation; clean-метрики "
        "нижче та всі маски обчислено на окремій reporting-частині. Product clusters між цими "
        "частинами не перетинаються. Це попередня валідаційна перевірка за зафіксованим протоколом, "
        "яка не замінює майбутнього фінального тестування."
        if held_out else
        "Пороги й гіперпараметри обрано на clean validation, тому метрики тієї самої "
        "вибірки мають оптимістичне зміщення добору. Це не незалежна оцінка узагальнення."
    )
    sections = [
        "# Попередні валідаційні експерименти",
        "Цей звіт сформовано `report_validation.py` із фактичного JSON запуску "
        f"`{source_path}`. Test-метрики тут не обчислюються. " + selection_note,
        "Докладні формули й протокол наведено в [методиці](experiment_methodology.md), "
        "походження validation й перевірки витоку — у [валідаційному аудиті](validation_data_audit.json).",
        "## Основне порівняння",
        "TF-IDF та розрізнювальність полів навчаються лише на train. Усі атрибутні методи "
        "отримують однакові нормалізовані ознаки; `text` є явно заявленим baseline лише за title. "
        "`title` та похідний `identifier` входять до спільної групи свідчень із сумарною базовою "
        "вагою 0.70; `description` і `brand` мають по 0.15. Внутрішня нормалізація доступних "
        "title/identifier є спільною для всіх атрибутних методів; fixed означає фіксовані "
        "ваги між групами. Точні неокруглені числа й параметри збережено в JSON/CSV.",
        _clean_table(clean, MAIN_METHODS),
    ]
    partition = protocol.get("partition", {})
    data_summaries = [("Train", result.get("data_summary", {}).get("train")),
                      ("Validation до розділення", result.get("data_summary", {}).get("validation")),
                      ("TUNE", partition.get("tune")), ("REPORT", partition.get("report"))]
    data_table = []
    for name, statistics in data_summaries:
        if statistics is None:
            continue
        count = statistics["pairs"]
        positives = statistics["labels"]["1"]
        data_table.append([name, count, positives, statistics["labels"]["0"],
                           f"{positives / count:.2%}", statistics.get("unique_offers", "—"),
                           statistics.get("unique_clusters", "—")])
    if data_table:
        sections[3:3] = ["## Дані",
                         _markdown_table(("Частина", "Пари", "Позитивні", "Негативні", "% позитивних",
                                          "Пропозиції", "Кластери"), data_table)]
    reason = result.get("metadata", {}).get("validation_selection_reason")
    if reason:
        sections.append("Зафіксована причина вибору validation з аудиту (оригінальний запис):\n\n> " + str(reason))
    by_method = {r["method"]: r for r in clean}
    if "adaptive" in by_method:
        adaptive = by_method["adaptive"]
        comparisons = [f"`{name}`: ΔF1={adaptive['f1'] - by_method[name]['f1']:+.4f}"
                       for name in MAIN_METHODS if name != "adaptive" and name in by_method]
        sections.append("Різниця F1 adaptive мінус comparator на clean reporting: " + "; ".join(comparisons) + ". "
                        "Ці різниці не є перевіркою статистичної значущості.")
        if "renorm" in by_method and all(adaptive[key] == by_method["renorm"][key] for key in METRICS):
            sections.append("**Adaptive не перевершив renorm:** усі наведені clean-метрики й confusion "
                            "matrix збігаються. Це не підтверджує додаткову корисність адаптивних "
                            "множників у цьому протоколі; рівність не є доказом еквівалентності формул.")
    if protocol:
        sections.extend(["## Розділення tuning / reporting",
                         "Вибрану validation small/80cc розділено детерміновано за товарними "
                         "cluster IDs. Пара з кінцями в різних частинах вилучається, тому кількість "
                         "негативних пар і баланс класів змінюються. Ці результати не слід "
                         "безпосередньо зіставляти з опублікованими оцінками на повній офіційній "
                         "validation. Маски застосовуються тільки до reporting. Ця незалежність "
                         "tuning від reporting не означає unseen щодо train: режим джерела "
                         "validation зафіксовано нижче й в окремому аудиті.",
                         "Параметри зафіксованого протоколу:\n\n```json\n" +
                         json.dumps(_compact_protocol(protocol), ensure_ascii=False, indent=2) + "\n```"])
        if "seen" in str(protocol.get("validation_mode", "")) and "unseen" not in str(protocol.get("validation_mode", "")):
            sections.append("Використано **seen-validation** як зафіксований fallback. Product clusters "
                            "validation можуть бути присутні в train, хоча offer IDs між train і "
                            "validation не повторюються. Отже, навіть окремий reporting не оцінює "
                            "перенесення на unseen товари й не усуває розрив до майбутнього unseen-test.")
    sections.extend([
        "## Маскування та фіксовані пороги",
        "Маскуються лише `brand` і `description`; `title` та похідний із нього `identifier` "
        "збережено. Та сама пропозиція має однакову маску в усіх парах, включно з перестановкою сторін. "
        "Маски вкладені за часткою пропусків. Наявні природні пропуски зберігаються. "
        "Задана частка є ймовірністю додаткового приховування поля пропозиції; частка втрати "
        "порівнюваних полів пари буде іншою. Пороги і параметри clean tuning заморожено "
        "для всіх часток; повторного добору за маскованими даними немає.",
        f"Перевірені seeds: `{', '.join(map(str, seeds))}`. У таблиці подано середнє F1 ± вибіркове SD "
        "за масками (`ddof=1`); це описова варіація маскування, не довірчий інтервал.",
    ])
    lookup = {(r["method"], r["missing_rate"]): r for r in summary}
    rows = []
    for method in MAIN_METHODS:
        row = [method]
        for rate in rates:
            record = lookup.get((method, rate))
            row.append("—" if record is None else f"{record['f1_mean']:.4f} ± {_fmt(record['f1_sd'])}")
        rows.append(row)
    sections.append(_markdown_table(("Метод", *(f"{r:.0%}" for r in rates)), rows))
    raw_masks = result["missingness"]["rows"]
    adaptive_masks = {(row["missing_rate"], row["seed"]): row for row in raw_masks if row["method"] == "adaptive"}
    renorm_masks = {(row["missing_rate"], row["seed"]): row for row in raw_masks if row["method"] == "renorm"}
    if adaptive_masks and adaptive_masks.keys() == renorm_masks.keys() and all(
            all(row[metric] == renorm_masks[key][metric] for metric in METRICS)
            for key, row in adaptive_masks.items()):
        sections.append(f"У всіх {len(adaptive_masks)} комбінаціях частка×seed adaptive і renorm "
                        "мають однакові Precision/Recall/F1 та confusion matrix. Отже цей "
                        "стрес-тест не показує переваги адаптивного механізму над простим "
                        "перенормуванням. Рівність агрегованих метрик окремо не доводить рівності "
                        "індивідуальних передбачень або формул.")
    verification = result.get("independent_verification", {})
    agreement = verification.get("adaptive_renorm_mask_agreement", [])
    agreement_keys = {(row["missing_rate"], row["seed"]) for row in agreement}
    if (agreement and len(agreement_keys) == len(agreement) and agreement_keys == adaptive_masks.keys()
            and all(row["different_decisions"] == 0 for row in agreement)):
        largest_difference = max(row["max_abs_score_difference"] for row in agreement)
        sections.append(f"Окрема [програмна перевірка]({artifacts_link}/independent_verification.json) "
                        f"підтвердила рівність **індивідуальних рішень** adaptive і renorm у всіх "
                        f"{len(agreement)} умовах на тих самих REPORT-парах із замороженими параметрами. "
                        f"Оцінки score при цьому різні (максимальна |Δscore|={largest_difference:.5f}). "
                        "Отже збіг F1 не спричинений випадково однаковими confusion counts, проте "
                        "за межами цих пар і порогів еквівалентність методів не доведено.")
    effective_rows = []
    for rate in rates:
        unique = {row["seed"]: row["effective_removed_fraction"] for row in result["missingness"]["rows"]
                  if row["missing_rate"] == rate and "effective_removed_fraction" in row}
        if unique:
            values = list(unique.values())
            effective_rows.append([f"{rate:.0%}", f"{mean(values):.2%}",
                                   f"{stdev(values):.2%}" if len(values) > 1 else "—"])
    if effective_rows:
        sections.extend(["Фактична частка видалених порівнюваних полів пари серед тих brand/description, "
                         "які були порівнювані до маскування. Один запис на seed, без повторення за методами:",
                         _markdown_table(("Задане маскування пропозицій", "Видалено полів пари: mean", "SD"), effective_rows)])
    sections.extend([
        f"![F1 залежно від частки додаткових пропусків]({artifacts_link}/f1_missingness.png)",
        "Вертикальну вісь графіків звужено до виміряного діапазону для читабельності малих "
        "змін. Різні типи ліній і маркерів показують криві, які накладаються; смуги ±SD не є CI.",
        "0% означає відсутність штучного маскування, а не повні атрибути. Три seeds при 0% "
        "дають ту саму clean reporting-вибірку й не є трьома незалежними експериментами.",
        "## Абляції",
        _clean_table(clean, ABLATIONS),
        "Абляції вимикають один механізм адаптивного методу. Інші гіперпараметри "
        "успадковуються від повної моделі. Якщо для абляції окремо обрано clean-поріг, її "
        "оцінка характеризує вилучення компонента з повторним добором робочої точки. "
        "Порівняння з порогом повної моделі слід читати окремо як зміну поведінки у фіксованій точці.",
        "![Абляції F1 під маскуванням]" + f"({artifacts_link}/f1_missingness_ablations.png)",
        "`no_reliability` перевіряє чутливість до вручну заданих множників, а не доводить "
        "реальну надійність джерел. У WDC немає gold confidence/позначок достовірності полів. "
        "Reliability та informativeness входять до ваг лише через добуток q_i=r_i·d_i. "
        "Окремі програмні перемикачі дозволяють перевірити вплив заданих множників, але не "
        "ідентифікують справжню надійність незалежно від розрізнювальності без зовнішньої розмітки. "
        "У головному методі penalty=0: identifier-конфлікт уже впливає на similarity відповідної "
        "групи. Окремий `no_penalty` дублював би головний метод, тому не додається.",
    ])
    fixed = fixed_ablation_rows(result)
    if fixed:
        sections.extend(["Абляції на reporting із тим самим порогом повної adaptive-моделі "
                         "(параметри успадковано, threshold не переналаштовано):",
                         _clean_table(fixed, ABLATIONS)])
        if "adaptive" in by_method:
            improved = [row for row in fixed if row["f1"] > by_method["adaptive"]["f1"]]
            if improved:
                sections.append("За незмінного порога вилучення компонента дає вищий reporting F1 для "
                                + ", ".join(f"`{row['method']}` (ΔF1={row['f1'] - by_method['adaptive']['f1']:+.4f})"
                                            for row in improved)
                                + ". Отже висновок про корисність компонентів залежить від політики "
                                "переналаштування порога; табличні різниці з окремим tuning-порогом "
                                "не доводять незалежного внеску надійності чи coverage. Ці параметри "
                                "не змінюються за результатами reporting.")
    legacy_methods = tuple(name for name in result["validation_selected"] if name.startswith("legacy_"))
    if legacy_methods:
        sections.extend(["## Допоміжна діагностика початкового методу",
                         _clean_table(clean, legacy_methods),
                         "Це збережені початкові raw-channel формули без нового спільного групування. "
                         "У legacy adaptive сітка включає і gamma, і penalty, отже бюджет добору "
                         "параметрів більший. Ця таблиця потрібна для прозорості: різниця з головними "
                         "методами не ізолює один компонент і не є доказом переваги групування."])
    if result.get("train_field_statistics"):
        sections.extend(["## Розрізнювальність, оцінена лише на train",
                         _markdown_table(("Канал", "d", "n позитивних", "mean similarity +", "n негативних", "mean similarity −"),
                                         [[field, _fmt(value["factor"]), value["positive_n"], _fmt(value["positive_mean"]),
                                           value["negative_n"], _fmt(value["negative_mean"])]
                                          for field, value in result["train_field_statistics"].items()]),
                         "Статистики рахуються лише серед доступних порівнянь відповідного класу. "
                         "Різна заповненість за класами й повторення пропозицій обмежують трактування d."])
        brand = result["train_field_statistics"].get("brand")
        if brand:
            sections.append(f"Фактор brand={brand['factor']:.4f} оцінено лише за "
                            f"{brand['positive_n']} доступними позитивними й {brand['negative_n']} "
                            "негативними навчальними порівняннями. Така вибірка серед доступних "
                            "значень може бути нерепрезентативною; високий d не означає високу "
                            "достовірність бренду або надійність вебджерела.")
    paired = result.get("paired_error_analysis", [])
    if isinstance(paired, dict):
        paired = [dict(value, comparator=key) for key, value in paired.items()]
    if paired:
        sections.append("## Парні зміни помилок на clean reporting")
        columns = ("comparator", "both_correct", "both_wrong", "adaptive_only_correct", "comparator_only_correct",
                   "rescued_fp", "rescued_fn", "introduced_fp", "introduced_fn")
        present = tuple(c for c in columns if any(c in row for row in paired))
        sections.append(_markdown_table(present, [[row.get(c, "—") for c in present] for row in paired]))
        sections.append("`rescued_fp/fn` — помилки comparator, виправлені adaptive; `introduced_fp/fn` — "
                        "помилки adaptive, яких не було в comparator. Це парне порівняння однакових рядків "
                        "за обраними на tuning для методів clean-порогами; воно описує рух рішень, а не причинний "
                        "внесок окремого множника. Повторення пропозицій у парах порушує незалежність спостережень.")
        movements = []
        for row in paired:
            comparator = row.get("comparator")
            if comparator not in by_method or "adaptive" not in by_method:
                continue
            if by_method["adaptive"]["f1"] >= by_method[comparator]["f1"]:
                continue
            needed = ("rescued_fp", "rescued_fn", "introduced_fp", "introduced_fn")
            if all(key in row for key in needed):
                movements.append(
                    f"Проти `{comparator}` adaptive має нижчий F1: виправлено FP={row['rescued_fp']} "
                    f"і FN={row['rescued_fn']}, натомість додано FP={row['introduced_fp']} "
                    f"і FN={row['introduced_fn']}."
                )
        if movements:
            sections.append(" ".join(movements) + " Розподіл цих переходів за наявністю identifier, "
                            "coverage та природними пропусками потрібно розглядати до будь-якої "
                            "зміни формули. Самі агреговані числа не встановлюють причину; "
                            "на підставі reporting-метрик параметри в цьому запуску не змінюються.")
    by_evidence = result.get("by_evidence", [])
    if by_evidence:
        sections.extend(["## Діагностика за свідченнями",
                         _markdown_table(("Прошарок", "Метод", "n", "Позитивні", "F1", "FP", "FN"),
                                         [[row["stratum"], row["method"], row["n"], row["positives"],
                                           _fmt(row["f1"]), row["fp"], row["fn"]] for row in by_evidence]),
                         "Coverage-прошарки утворюють розбиття за доступністю груп; identifier-conflict "
                         "і identifier-missing можуть перетинатися з ними. Числа таких прошарків не "
                         "можна складати. Усередині постійного coverage shrinkage є афінним "
                         "перетворенням score: рух помилок за спільним порогом не доводить покращення ранжування."])
        config = result["validation_selected"].get("adaptive")
        prior = protocol.get("prior_score_reference")
        if config and prior is not None:
            cutoff_table = []
            for row in by_evidence:
                if row["method"] == "adaptive" and row["stratum"].startswith("coverage_"):
                    coverage = float(row["stratum"].removeprefix("coverage_"))
                    if coverage > 0:
                        alpha = coverage ** config["gamma"]
                        cutoff = (config["threshold"] - (1 - alpha) * prior) / alpha
                        cutoff_table.append([_fmt(coverage, 2), row["n"], _fmt(cutoff, 5)])
            if cutoff_table:
                sections.extend([
                    f"Для обраних γ={config['gamma']:.2f}, τ={config['threshold']:.4f}, "
                    f"π={prior:.4f} правило A≥τ еквівалентне H≥(τ−(1−C^γ)π)/C^γ (C>0, penalty=0). "
                    "Це алгебраїчне пояснення робочої точки, а не новий добір параметрів:",
                    _markdown_table(("Coverage C", "Пар", "Ефективний поріг H"), cutoff_table),
                ])
                if prior > config["threshold"]:
                    sections.append("Оскільки π>τ, нижча coverage **полегшує прийняття позитивного "
                                    "рішення** відносно незмінного H-порога τ. Цю опорну точку не слід "
                                    "називати «консервативним prior». До фінального оцінювання потрібно "
                                    "погодити зміст shrinkage та калібрування шкали/порога. Якщо очікується "
                                    "лише зниження впевненості за неповноти, можливу зміну формули або "
                                    "допустимої опорної точки слід обґрунтувати, зафіксувати до нового "
                                    "незалежного валідаційного досліду й не підбирати за поточним REPORT чи test.")
        conflict = next((row for row in by_evidence if row["method"] == "adaptive" and row["stratum"] == "identifier_conflict"), None)
        if conflict:
            sections.append(f"Серед {conflict['n']} пар із нульовим identifier Jaccard є "
                            f"{conflict['positives']} позитивних за WDC labels; у цьому прошарку adaptive "
                            f"має FN={conflict['fn']}. Нульовий евристичний Jaccard не є достовірною "
                            "суперечністю товарної моделі. Це обґрунтовує ручний аудит extractor на "
                            "train/validation перед посиленням identifier-конфліктів, без висновку про "
                            "помилковість конкретних WDC labels.")
    sections.extend([
        "## Обмеження та рішення до фінального оцінювання",
        "1. Це результати на офіційних готових парах; вони не вимірюють повноту генерації кандидатів "
        "або end-to-end пошуку. Результатів на окремих технічних FPV-атрибутах цей WDC не дає.\n"
        "2. `identifier` евристично походить із title, тому ці два канали корельовані. Навіть спільна "
        "група свідчень не робить ознаки незалежними. Coverage є часткою доступних базових групових бюджетів, "
        "а не каліброваною ймовірністю достатності доказів.\n"
        "3. Інформативність тут означає train-різницю середніх similarity для класів, а не інформацію "
        "конкретного значення. Вручну задані reliability не можна називати емпірично доведеними.\n"
        "4. Незалежне маскування двох полів — контрольований MCAR стрес-тест, який не відтворює всі "
        "природні, залежні від товару чи вебджерела пропуски. Загальний score не є ймовірністю збігу.\n"
        "5. Потрібно погодити формулу, параметри й протокол перед окремим фінальним тестуванням. "
        "Нижчий validation F1 сам по собі не є підставою довільно змінювати метод; виправлення мають "
        "спиратися на аналіз кореляції, coverage і руху FP/FN та вимагатимуть нової незалежної перевірки. "
        "Наукову новизну й перевагу не доведено.",
        "## Артефакти та відтворення",
        f"Джерело: [{Path(source_path).name}]({artifacts_link}/{Path(source_path).name}); "
        f"[reporting CSV]({artifacts_link}/metrics.csv), "
        f"[tuning CSV]({artifacts_link}/tuning_metrics.csv), "
        f"[усі маски CSV]({artifacts_link}/missingness.csv), "
        f"[агрегація CSV]({artifacts_link}/missingness_summary.csv). "
        "PNG і SVG містять ті самі криві. Для відтворення звіту з готового JSON:",
        "```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 report_validation.py "
        "--results docs/validation/results.json --out-dir docs/validation "
        "--report docs/validation_results.md\n```",
    ])
    if verification:
        checks = verification.get("clean_metric_checks", [])
        passed = sum(item["manual_confusion_and_metrics_match"] is True for item in checks)
        sections.insert(-3, f"Повторна перевірка з навчанням представлень тільки на train й "
                        f"незмінними конфігураціями відтворила ручний підрахунок TN/FP/FN/TP та "
                        f"Precision/Recall/F1 для {passed}/{len(checks)} clean-методів. "
                        "Це перевірка експорту й обчислень, не нова статистично незалежна "
                        "вибірка. Sidecar зв'язаний із точними байтами results.json через SHA256.")
    if result.get("limitations"):
        sections.insert(-3, "Обмеження, зафіксовані під час запуску:\n\n" +
                        "\n".join(f"- {item}" for item in result["limitations"]))
    return "\n\n".join(sections) + "\n"


def generate_report(result_path: Path, output_dir: Path, report_path: Path) -> list[Path]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if "test_metrics" in result:
        raise ValueError("Expected validation-only result, found test_metrics")
    if not result.get("validation_selected") or not result.get("missingness", {}).get("rows"):
        raise ValueError("Result must contain validation_selected and missingness.rows")
    sidecar = result_path.with_name("independent_verification.json")
    if sidecar.exists():
        verification = json.loads(sidecar.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
        if verification.get("results_sha256") != actual_hash:
            raise ValueError("Independent verification SHA256 differs from results JSON")
        result["independent_verification"] = verification
    clean = clean_rows(result)
    missing_rows = result["missingness"]["rows"]
    summary = summarize_missingness(missing_rows)
    if any(row.get("scope", "report") != "report" for row in missing_rows):
        raise ValueError("Missingness reporting accepts scope='report' only")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in ("metrics.csv", "missingness.csv", "missingness_summary.csv")]
    write_csv(paths[0], clean)
    write_csv(paths[1], missing_rows)
    write_csv(paths[2], summary)
    extra = {
        "tuning_metrics.csv": tuning_rows(result),
        "ablations.csv": [row for row in clean if row["method"] in ABLATIONS],
        "ablation_fixed_threshold_metrics.csv": fixed_ablation_rows(result),
        "by_evidence.csv": result.get("by_evidence", []),
    }
    paired = result.get("paired_error_analysis", [])
    if isinstance(paired, dict):
        paired = [dict(value, comparator=key) for key, value in paired.items()]
    extra["paired_errors.csv"] = paired
    for name, records in extra.items():
        if records:
            path = output_dir / name
            write_csv(path, records)
            paths.append(path)
    paths += plot_missingness(summary, output_dir, methods=MAIN_METHODS, stem="f1_missingness")
    paths += plot_missingness(summary, output_dir, methods=("adaptive", *ABLATIONS), stem="f1_missingness_ablations")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    link = Path(os.path.relpath(output_dir, report_path.parent)).as_posix()
    report_path.write_text(build_markdown(result, clean, summary, source_path=result_path.as_posix(), artifacts_link=link),
                           encoding="utf-8")
    paths.append(report_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("docs/validation/results.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/validation"))
    parser.add_argument("--report", type=Path, default=Path("docs/validation_results.md"))
    args = parser.parse_args()
    paths = generate_report(args.results, args.out_dir, args.report)
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
