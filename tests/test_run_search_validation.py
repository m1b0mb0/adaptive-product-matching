import pytest

import run_search_validation as runner


def test_position_changes_and_misses_have_distinct_denominators():
    rows = {}
    qrels = {}
    for q, before, after in [("a", 6, 3), ("b", 2, 6), ("c", 2, 2), ("d", None, None)]:
        qrels[q] = {"match"}
        candidates = ["u1", "u2", "u3", "u4", "u5"]
        if before:
            candidates.insert(before - 1, "match")
        rows[q] = {"candidate_ids": tuple(candidates) if before else (),
                   "ranking": {"tfidf": {"first_relevant_rank": before},
                               "fixed": {"first_relevant_rank": after}}}
    evaluation = {
        "per_query": rows, "eligible_query_count": 4, "excluded_no_qrels_count": 0,
        "retrieval": {"candidate_recall_at_k": 0.75, "empty_pool_count": 1},
        "ranking": {m: {"hit_at_k": {1: 0, 5: 0.5}, "mrr_at_k": rr}
                    for m, rr in [("tfidf", 0.25), ("fixed", (1 / 2 + 1 / 3) / 4)]},
    }
    summary = runner.summarize(evaluation, qrels)
    assert summary["missed_at_20"] == 1
    assert summary["methods"]["fixed"]["retrieved_outside_top5"] == 1
    assert summary["methods"]["fixed"]["rank_changes_vs_tfidf"] == {
        "improved": 1, "unchanged": 1, "worsened": 1,
    }
    evaluation["retrieval"]["candidate_recall_at_k"] = 1
    with pytest.raises(ValueError, match="retrieval Hit@20"):
        runner.summarize(evaluation, qrels)


def test_rejects_wrong_validation_hash_before_import(tmp_path, monkeypatch):
    validation = tmp_path / "valid.json.gz"
    validation.write_bytes(b"wrong file")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(runner.canonical_json({"validation_path": str(validation), "validation_sha256": "wrong"}))
    monkeypatch.setattr(runner, "VALIDATION", validation)
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    with pytest.raises(ValueError, match="manifest"):
        runner.load_inputs()


def test_run_preserves_existing_results(tmp_path):
    with pytest.raises(FileExistsError, match="new output directory"):
        runner.run(tmp_path)
