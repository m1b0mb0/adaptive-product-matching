"""Run the fixed WDC seen-validation search protocol from the repository root."""
import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

from audit_wdc import read_pairs
from frozen_v1_runtime import load_corrected_scorer
from product_catalog import load_wdc_catalog
from search_evaluation import TFIDF_BASELINE, evaluate_search
from search_system import SearchEngine
from wdc_qrels import build_wdc_qrels


VALIDATION = Path("data/external/wdcproducts80cc20rnd000un_valid_small.json.gz")
MANIFEST = Path("docs/validation_config.json")
CONFIG = Path("docs/final_method/frozen_configuration.json")
CANDIDATE_K = 20
TOP_K = 5


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def load_inputs():
    manifest = json.loads(MANIFEST.read_bytes())
    digest = hashlib.sha256(VALIDATION.read_bytes()).hexdigest()
    if Path(manifest["validation_path"]) != VALIDATION or digest != manifest["validation_sha256"]:
        raise ValueError("Validation file differs from the selected validation manifest")
    pairs = read_pairs(VALIDATION)
    catalog = load_wdc_catalog(VALIDATION)
    qrels = build_wdc_qrels(pairs, catalog)
    positives = sum(int(pair["label"]) for pair in pairs)
    if (len(catalog) != 1000 or positives != 500 or qrels.keys() != catalog.keys()
            or any(len(matches) != 1 for matches in qrels.values())):
        raise ValueError("Expected 1000 catalog queries with one qrel each and 500 positive pairs")
    return catalog, qrels, {
        "validation_path": VALIDATION.as_posix(), "validation_sha256": digest,
        "qrels_sha256": hashlib.sha256(canonical_json({q: sorted(v) for q, v in qrels.items()})).hexdigest(),
        "catalog_size": len(catalog), "pair_count": len(pairs), "positive_pair_count": positives,
        "directed_qrel_count": sum(map(len, qrels.values())),
    }


def summarize(evaluation: dict, qrels: dict) -> dict:
    rows = evaluation["per_query"]
    if (rows.keys() != qrels.keys() or len(rows) != evaluation["eligible_query_count"]
            or evaluation["excluded_no_qrels_count"]):
        raise ValueError("Evaluation cohort differs from the validated queries")
    # With one explicit match per query, pool membership is exactly retrieval Hit@20.
    found = {q for q, row in rows.items() if qrels[q].intersection(row["candidate_ids"])}
    recall = evaluation["retrieval"]["candidate_recall_at_k"]
    if recall != len(found) / len(rows):
        raise ValueError("Candidate recall differs from retrieval Hit@20")
    diagnostics = {}
    for method, metrics in evaluation["ranking"].items():
        hit1, hit5 = (metrics["hit_at_k"][k] for k in (1, TOP_K))
        if not (0 <= hit1 <= hit5 <= recall <= 1 and 0 <= metrics["mrr_at_k"] <= hit5):
            raise ValueError(f"Inconsistent aggregate metrics for {method}")
        moves = {"improved": 0, "unchanged": 0, "worsened": 0}
        outside = 0
        for q in found:
            ranking = rows[q]["ranking"]
            before = ranking[TFIDF_BASELINE]["first_relevant_rank"]
            after = ranking[method]["first_relevant_rank"]
            if before is None or after is None:
                raise ValueError(f"Retrieved match missing from ranking for {q}")
            outside += after > TOP_K
            if after < before:
                moves["improved"] += 1
            elif after > before:
                moves["worsened"] += 1
            else:
                moves["unchanged"] += 1
        if hit5 != (len(found) - outside) / len(rows):
            raise ValueError(f"Hit@5 differs from the counted positions for {method}")
        diagnostics[method] = metrics | {"retrieved_outside_top5": outside, "rank_changes_vs_tfidf": moves}
    return {
        "eligible_queries": len(rows), "excluded_queries": evaluation["excluded_no_qrels_count"],
        **evaluation["retrieval"], "retrieval_hit_at_20": len(found) / len(rows),
        "retrieved_matches": len(found), "missed_at_20": len(rows) - len(found), "methods": diagnostics,
    }


def run(out_dir: Path) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {out_dir}")
    if subprocess.check_output(["git", "status", "--porcelain"]).strip():
        raise ValueError("Use a clean checkout to record the experiment revision")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    catalog, qrels, inputs = load_inputs()
    scorer = load_corrected_scorer()
    engines = {
        "fixed": SearchEngine(catalog, method="fixed"),
        "renorm": SearchEngine(catalog, method="renorm"),
        "adaptive-corrected": SearchEngine(catalog, method="adaptive-corrected", pair_scorer=scorer),
    }
    evaluation = evaluate_search(catalog, sorted(catalog), qrels, engines["fixed"].retriever, engines,
                                 candidate_k=CANDIDATE_K, hit_ks=(1, TOP_K), mrr_k=TOP_K,
                                 other_sources_only=False)
    results = summarize(evaluation, qrels)
    details = b"".join(canonical_json({"query_id": q, "relevant_ids": sorted(qrels[q]), **row}) + b"\n"
                       for q, row in evaluation["per_query"].items())
    config = CONFIG.read_bytes()
    summary = {
        "experiment": "exploratory WDC seen-validation small", "git_sha": revision, **inputs,
        "frozen_config_sha256": hashlib.sha256(config).hexdigest(),
        "verified_bundle_sha256": json.loads(config)["artifacts"]["model_bundle"]["sha256"],
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "pandas", "scikit-learn", "joblib")},
        "candidate_k": CANDIDATE_K, "top_k": TOP_K, "other_sources_only": False,
        "qrels_policy": "Explicit, symmetric WDC positives after cluster and catalog consistency checks.",
        "denominator": "Macro over all queries with known positives, including retrieval misses.",
        "per_query_file": "per_query.jsonl", "per_query_sha256": hashlib.sha256(details).hexdigest(),
        **results,
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "per_query.jsonl").write_bytes(details)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                    allow_nan=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/wdc_search_validation"))
    run(parser.parse_args().out_dir)
