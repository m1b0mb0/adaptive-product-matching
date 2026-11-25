"""Run the pre-specified offer-level stress study on previously used WDC validation."""
import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from statistics import mean

from catalog_missingness import TEXT_FIELDS, mask_catalog, summarize_masked_validation
from final_evaluation import no_fit
from frozen_v1_runtime import load_corrected_scorer
from matching import norm
from run_search_validation import CANDIDATE_K, CONFIG, TOP_K, canonical_json, load_inputs, summarize
from search_evaluation import evaluate_search
from search_system import SearchEngine


REFERENCE = Path("docs/search_system/wdc_validation_search_summary.json")
SCENARIOS = {
    "clean": set(), "title": {"title"}, "brand": {"brand"}, "description": {"description"},
    "brand_description": {"brand", "description"}, "all_fields": set(TEXT_FIELDS),
}
SEEDS = (11, 22, 33)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_reference(actual, expected, name):
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(f"Reference mismatch: {name}")


def measure(clean, catalog, qrels, scorer, directory):
    engines = {method: SearchEngine(catalog, method) for method in ("fixed", "renorm")}
    engines["adaptive-corrected"] = SearchEngine(catalog, "adaptive-corrected", pair_scorer=scorer)
    with no_fit():
        evaluation = evaluate_search(
            catalog, sorted(clean), qrels, engines["fixed"].retriever, engines,
            candidate_k=CANDIDATE_K, hit_ks=(1, TOP_K), mrr_k=TOP_K,
            other_sources_only=False, include_retrieval_scores=True,
        )
    baseline_format = summarize(evaluation, qrels)
    diag = summarize_masked_validation(clean, catalog, qrels, evaluation)
    removed = {f: sorted(q for q in clean if getattr(clean[q], f) != getattr(catalog[q], f))
               for f in TEXT_FIELDS}
    mask_bytes = canonical_json(removed)
    directory.mkdir()
    (directory / "mask.json").write_bytes(mask_bytes)
    diag["total_missing"] = {f: n + len(removed[f]) for f, n in diag["naturally_missing"].items()}
    diag["all_text_missing"] = sum(not any(norm(getattr(o, f)) for f in TEXT_FIELDS)
                                   for o in catalog.values())
    groups = {"missing_title": 0, "both_titles_missed": 0, "retrieved": 0}
    pool_sha = hashlib.sha256()
    legacy_sha = hashlib.sha256()
    details_sha = hashlib.sha256()
    with (directory / "per_query.jsonl").open("xb") as details:
        for q, row in evaluation["per_query"].items():
            partner, = qrels[q]
            found = partner in row["candidate_ids"]
            both = bool(norm(catalog[q].title) and norm(catalog[partner].title))
            if found and not both:
                raise ValueError("Retrieved a known partner with a missing title")
            group = "retrieved" if found else ("both_titles_missed" if both else "missing_title")
            groups[group] += 1
            record = {"query_id": q, "relevant_ids": [partner], **row}
            pool_sha.update(canonical_json([q, row["candidate_ids"], row["retrieval_scores"]]) + b"\n")
            legacy = {k: v for k, v in record.items() if k != "retrieval_scores"}
            legacy_sha.update(canonical_json(legacy) + b"\n")
            line = canonical_json(record | {"group": group}) + b"\n"
            details.write(line)
            details_sha.update(line)
    both_queries = 2 * diag["clusters_with_both_titles"]
    if (sum(groups.values()) != len(clean) or groups["retrieved"] != baseline_format["retrieved_matches"]
            or groups["missing_title"] != len(clean) - both_queries):
        raise ValueError("Query groups disagree with verified cluster or retrieval counts")
    diag.update(groups=groups, recall_ceiling=both_queries / len(clean),
                outside_top20=baseline_format["missed_at_20"],
                retrieved_outside_top5={m: v["retrieved_outside_top5"]
                                        for m, v in baseline_format["methods"].items()})
    values = {
        m: {"recall20": baseline_format["candidate_recall_at_k"], "hit1": v["hit_at_k"][1],
            "hit5": v["hit_at_k"][5], "mrr5": v["mrr_at_k"], "conditional_mrr5": diag["conditional_mrr_at_5"][m]}
        for m, v in baseline_format["methods"].items()
    }
    result = {"metrics": values, "diagnostics": diag, "mask_sha256": digest(mask_bytes),
              "pools_sha256": pool_sha.hexdigest(), "per_query_sha256": details_sha.hexdigest()}
    return result, baseline_format, legacy_sha.hexdigest()


def aggregate(runs):
    table = {}
    for scenario in SCENARIOS:
        group = [r["metrics"] for r in runs if r["scenario"] == scenario]
        table[scenario] = {}
        for method, metrics in group[0].items():
            stats = {}
            for metric in metrics:
                values = [g[method][metric] for g in group]
                if None in values:
                    stats[metric] = dict.fromkeys(("mean", "min", "max", "range", "delta_clean"))
                    continue
                stats[metric] = {"mean": mean(values), "min": min(values), "max": max(values),
                                 "range": max(values) - min(values),
                                 "delta_clean": mean(values) - runs[0]["metrics"][method][metric]}
            table[scenario][method] = stats
    return table


def run(out_dir: Path):
    if out_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {out_dir}")
    if subprocess.check_output(["git", "status", "--porcelain"]).strip():
        raise ValueError("Record the implementation in a clean checkout before running")
    reference = json.loads(REFERENCE.read_bytes())
    clean, qrels, inputs = load_inputs()
    config = CONFIG.read_bytes()
    provenance = {
        **inputs, "python": platform.python_version(),
        "packages": {p: version(p) for p in reference["packages"]},
        "frozen_config_sha256": digest(config),
        "verified_bundle_sha256": json.loads(config)["artifacts"]["model_bundle"]["sha256"],
        "candidate_k": CANDIDATE_K, "top_k": TOP_K, "other_sources_only": False,
    }
    for key, value in provenance.items():
        check_reference(value, reference[key], key)
    with no_fit():
        scorer = load_corrected_scorer(CONFIG)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    out_dir.mkdir(parents=True)
    runs = []
    pool_hashes = {}
    for scenario, fields in SCENARIOS.items():
        for seed in SEEDS if fields else (None,):
            label = scenario if seed is None else f"{scenario}_{seed}"
            catalog = mask_catalog(clean, fields, seed) if fields else clean
            result, baseline, legacy_sha = measure(clean, catalog, qrels, scorer, out_dir / label)
            if scenario == "clean":
                check_reference(legacy_sha, reference["per_query_sha256"], "Task 6.3 per-query SHA")
                for key, value in baseline.items():
                    check_reference(value, reference[key], key)
            elif scenario != "title":
                control = f"title_{seed}" if "title" in fields else "clean"
                check_reference(result["pools_sha256"], pool_hashes[control], f"{label} vs {control} pools")
            pool_hashes[label] = result["pools_sha256"]
            values = result["metrics"]
            clean_metrics = runs[0]["metrics"] if runs else values
            delta = {m: {k: v - clean_metrics[m][k] if v is not None else None for k, v in metrics.items()}
                     for m, metrics in values.items()}
            runs.append({"scenario": scenario, "seed": seed, "directory": label, **result,
                         "delta_clean": delta})
            print(f"Verified {label}", flush=True)
    summary = {"implementation_git_sha": revision, **provenance, "seeds": SEEDS,
               "fields": {k: sorted(v) for k, v in SCENARIOS.items()},
               "quota": "floor(normalized_nonempty_count / 2) per selected field",
               "clean_task63_per_query_sha256": reference["per_query_sha256"],
               "runs": runs, "aggregate": aggregate(runs)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    run(parser.parse_args().out_dir)
