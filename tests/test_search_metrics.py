import pytest

from search_metrics import candidate_recall_at_k, hit_at_k, mrr_at_k


@pytest.mark.parametrize(("offers", "relevant", "recall", "hit", "mrr"), [
    (["a"], {"a"}, 1.0, 1, 1.0),
    (["x", "y", "a"], {"a"}, 1.0, 1, 1 / 3),
    (["u", "v", "w", "x", "y", "a"], {"a"}, 0.0, 0, 0.0),
    (["x", "y"], {"a"}, 0.0, 0, 0.0),
    ([], {"a"}, 0.0, 0, 0.0),
    (["x", "a", "y", "b", "z", "c"], {"a", "b", "c"}, 2 / 3, 1, 0.5),
])
def test_metrics_match_hand_calculated_results(offers, relevant, recall, hit, mrr):
    assert candidate_recall_at_k(offers, relevant, 5) == pytest.approx(recall)
    assert hit_at_k(offers, relevant, 5) == hit
    assert mrr_at_k(offers, relevant, 5) == pytest.approx(mrr)


@pytest.mark.parametrize("metric", [candidate_recall_at_k, hit_at_k, mrr_at_k])
@pytest.mark.parametrize("bad_k", [0, -1, True, 5.0, "5"])
def test_metrics_require_a_positive_integer_cutoff(metric, bad_k):
    with pytest.raises(ValueError, match="positive integer"):
        metric(["a"], {"a"}, bad_k)


@pytest.mark.parametrize("metric", [candidate_recall_at_k, hit_at_k, mrr_at_k])
def test_metrics_reject_duplicate_or_invalid_ids_and_require_known_positives(metric):
    with pytest.raises(ValueError, match="Duplicate offer IDs"):
        metric(["a", "x", "x"], {"a"}, 1)
    with pytest.raises(ValueError, match="unpadded strings"):
        metric([" a"], {"a"}, 5)
    with pytest.raises(ValueError, match="provided as a set"):
        metric(["a"], ["a", "a"], 5)
    with pytest.raises(ValueError, match="at least one known relevant offer"):
        metric([], set(), 5)
