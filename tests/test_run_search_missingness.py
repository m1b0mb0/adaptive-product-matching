import hashlib
import json

import pytest

from catalog_missingness import mask_catalog
from frozen_v1_runtime import load_corrected_scorer
from product_catalog import ProductOffer
from run_search_missingness import check_reference, measure


def test_three_groups_and_canonical_artifacts(tmp_path):
    # One pair retrievable, one with a masked title, one with disjoint titles.
    clean = {key: ProductOffer(key, title) for key, title in zip(
        "abcdef", ("xx", "xx", "identical", "identical", "unique", "zz"), strict=True)}
    qrels = dict(zip("abcdef", ({"b"}, {"a"}, {"d"}, {"c"}, {"f"}, {"e"}), strict=True))
    scorer = load_corrected_scorer()
    catalog = mask_catalog(clean, {"title"}, 11)
    directory = tmp_path / "masked"
    result, _, legacy = measure(clean, catalog, qrels, scorer, directory)
    for filename, key in (("mask.json", "mask_sha256"), ("per_query.jsonl", "per_query_sha256")):
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == result[key]
    assert result["diagnostics"]["groups"] == {"retrieved": 2, "missing_title": 4, "both_titles_missed": 0}
    assert result["diagnostics"]["recall_ceiling"] == 2 / 6
    rows = [json.loads(line) for line in (directory / "per_query.jsonl").read_bytes().splitlines()]
    assert all(len(row["candidate_ids"]) == len(row["retrieval_scores"]) for row in rows)
    clean_result, _, _ = measure(clean, clean, qrels, scorer, tmp_path / "clean")
    assert clean_result["diagnostics"]["groups"]["both_titles_missed"] == 2
    with pytest.raises(ValueError, match="Reference mismatch"):
        check_reference(legacy, result["per_query_sha256"], "clean baseline")
