"""Protocol checks use small fixtures, never scientific benchmark outcomes."""
from collections import Counter
from itertools import combinations, product

import numpy as np
import pytest

from matching import FeatureExtractor, Features
import run_known_methods as development


@pytest.fixture
def entity_pairs():
    """18 independent entities, two offers each, all first-offer negatives."""
    offers = {}
    for entity in range(18):
        for seller in range(2):
            offer_id = f"entity-{entity:02d}-shop-{seller}"
            marker = chr(0x4E00 + entity) * 5
            offers[offer_id] = {
                "id": offer_id,
                "cluster_id": f"cluster-{entity:02d}",
                "title": f"Common camera Model AX{entity:03d} {marker} seller{seller}",
                "description": f"Photo camera item {entity:02d}",
                "brand": f"Brand{entity % 3}",
            }

    def pair(left, right, label):
        result = {"label": label, "is_hard_negative": bool(
            label == 0 and (int(left[7:9]) + int(right[7:9])) % 2 == 0)}
        for field in ("id", "cluster_id", "title", "description", "brand"):
            result[f"{field}_left"] = offers[left][field]
            result[f"{field}_right"] = offers[right][field]
        return result

    rows = [pair(f"entity-{entity:02d}-shop-0", f"entity-{entity:02d}-shop-1", 1)
            for entity in range(18)]
    rows += [pair(f"entity-{left:02d}-shop-0", f"entity-{right:02d}-shop-0", 0)
             for left, right in combinations(range(18), 2)]
    return rows


def offer_ids(pairs):
    return {str(pair[f"id_{side}"]) for pair in pairs for side in ("left", "right")}


def cluster_ids(pairs):
    return {str(pair[f"cluster_id_{side}"]) for pair in pairs for side in ("left", "right")}


def assert_disjoint_partition(pairs, fold):
    fit_indices, eval_indices, excluded = map(
        set, (fold["fit_indices"], fold["eval_indices"], fold["excluded_indices"]))
    assert not fit_indices & eval_indices
    assert not fit_indices & excluded
    assert not eval_indices & excluded
    assert fit_indices | eval_indices | excluded == set(range(len(pairs)))
    fit = [pairs[i] for i in fit_indices]
    evaluated = [pairs[i] for i in eval_indices]
    assert not offer_ids(fit) & offer_ids(evaluated)
    assert not cluster_ids(fit) & cluster_ids(evaluated)
    assert cluster_ids(fit) <= set(fold["fit_cluster_ids"])
    assert cluster_ids(evaluated) <= set(fold["eval_cluster_ids"])
    assert {pair["label"] for pair in fit} == {0, 1}
    assert {pair["label"] for pair in evaluated} == {0, 1}
    heldout = set(fold["eval_cluster_ids"])
    for index, row in enumerate(pairs):
        left = row["cluster_id_left"] in heldout
        right = row["cluster_id_right"] in heldout
        if left != right:
            assert index in excluded
        elif left:
            assert index in eval_indices
        else:
            assert index in fit_indices


def test_entity_split_accounts_for_both_negative_endpoints_and_excludes_crossing_pairs(entity_pairs):
    folds = development.entity_folds(entity_pairs)
    assert len(folds) == 3
    for fold in folds:
        assert_disjoint_partition(entity_pairs, fold)
    evaluations = Counter(index for fold in folds for index in fold["eval_indices"])
    assert all(evaluations[i] == 1 for i, row in enumerate(entity_pairs) if row["label"] == 1)
    assert max(evaluations.values()) == 1
    assert any(row["label"] == 0 and evaluations[index] == 0
               for index, row in enumerate(entity_pairs))


def test_entity_folds_are_deterministic_and_nested_indices_are_local(entity_pairs):
    assert development.entity_folds(entity_pairs) == development.entity_folds(entity_pairs)
    plan = development.nested_plan(entity_pairs)
    assert len(plan["outer_folds"]) == 3
    for outer in plan["outer_folds"]:
        assert_disjoint_partition(entity_pairs, outer)
        outer_fit = [entity_pairs[i] for i in outer["fit_indices"]]
        assert len(outer["inner_folds"]) == 3
        for inner in outer["inner_folds"]:
            assert_disjoint_partition(outer_fit, inner)
            inner_rows = [outer_fit[i] for i in inner["fit_indices"]]
            outer_eval = [entity_pairs[i] for i in outer["eval_indices"]]
            assert not cluster_ids(inner_rows) & cluster_ids(outer_eval)


def test_split_requires_ground_truth_entity_ids(entity_pairs):
    rows = [dict(row) for row in entity_pairs]
    del rows[0]["cluster_id_right"]
    with pytest.raises(ValueError, match="[Cc]luster"):
        development.entity_folds(rows)


@pytest.mark.parametrize("scenario", ["auxiliary", "all_groups"])
def test_masking_is_offer_consistent_symmetric_deterministic_and_non_mutating(scenario):
    ids = [f"offer-{i}" for i in range(12)]
    ordered = list(product(ids, ids))
    features = Features(np.ones((len(ordered), 4)),
                        [a for a, _ in ordered], [b for _, b in ordered])
    original = features.matrix.copy()
    corrupted = development.apply_missingness(features, .5, 11, scenario=scenario)
    missing = {pair: np.isnan(corrupted[i]) for i, pair in enumerate(ordered)}
    for a, b in ordered:
        np.testing.assert_array_equal(missing[a, b], missing[a, a] | missing[b, b])
        np.testing.assert_array_equal(missing[a, b], missing[b, a])
    np.testing.assert_array_equal(features.matrix, original)
    np.testing.assert_array_equal(
        corrupted, development.apply_missingness(features, .5, 11, scenario=scenario))
    reversed_features = Features(features.matrix[::-1], features.ids_left[::-1], features.ids_right[::-1])
    np.testing.assert_array_equal(
        corrupted[::-1], development.apply_missingness(reversed_features, .5, 11, scenario=scenario))
    if scenario == "all_groups":
        np.testing.assert_array_equal(np.isnan(corrupted[:, 0]), np.isnan(corrupted[:, 3]))
        assert np.isnan(corrupted[:, 0]).any()
    else:
        assert not np.isnan(corrupted[:, [0, 3]]).any()
    assert np.isnan(corrupted[:, 1:3]).any()


@pytest.mark.parametrize("scenario", ["auxiliary", "all_groups"])
def test_masks_are_nested_with_rate_and_keep_natural_missing_values(scenario):
    matrix = np.ones((20, 4))
    matrix[0, 1] = np.nan
    features = Features(matrix, [f"a{i}" for i in range(20)], [f"b{i}" for i in range(20)])
    zero = development.apply_missingness(features, 0., 22, scenario=scenario)
    np.testing.assert_array_equal(zero, matrix)
    prior = np.isnan(zero)
    for rate in (.1, .2, .3, .5, 1.):
        current = np.isnan(development.apply_missingness(features, rate, 22, scenario=scenario))
        assert (prior <= current).all()
        prior = current
    assert prior.all() if scenario == "all_groups" else prior[:, 1:3].all()


def test_auxiliary_masking_agrees_with_existing_offer_attribute_mask_scheme():
    from matching import masked
    features = Features(np.ones((20, 4)), [f"a{i}" for i in range(20)],
                        [f"b{i}" for i in range(20)])
    expected = masked(features, .3, 33)
    np.testing.assert_array_equal(
        development.apply_missingness(features, .3, 33, scenario="auxiliary"), expected)


def test_parameter_ties_keep_grid_order_and_threshold_is_selected_without_rounding():
    y = np.array([1, 1, 0, 0])
    scores = np.array([.9, .8, .2, .1])
    selected = development.select_from_oof(y, [({"C": .1}, scores), ({"C": 10.}, scores)])
    assert selected["C"] == .1
    assert .2 < selected["threshold"] <= .8
    assert selected["inner_metrics"]["f1"] == 1.


def test_empty_error_export_preserves_known_csv_header(tmp_path):
    import csv
    path = tmp_path / "errors.csv"
    development._write_csv(path, [], fieldnames=["pair_id", "error_type"])
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["pair_id", "error_type"]
        assert list(reader) == []


def test_pipeline_fits_every_representation_inside_its_fold_and_never_retunes_masks(entity_pairs, monkeypatch):
    plan = development.nested_plan(entity_pairs)
    expected_fit_ids = []
    for outer in plan["outer_folds"]:
        outer_fit = [entity_pairs[i] for i in outer["fit_indices"]]
        expected_fit_ids.append(frozenset(offer_ids(outer_fit)))
        for inner in outer["inner_folds"]:
            rows = [outer_fit[i] for i in inner["fit_indices"]]
            expected_fit_ids.append(frozenset(offer_ids(rows)))

    actual_fit_ids = []
    actual_fit_rows = []
    original_fit = FeatureExtractor.fit

    def fit_spy(self, rows):
        ids = offer_ids(rows)
        actual_fit_ids.append(frozenset(ids))
        actual_fit_rows.append(rows)
        fitted = original_fit(self, rows)
        vocabulary = self.vectorizers["title"].vocabulary_
        for entity in range(18):
            if f"entity-{entity:02d}-shop-0" not in ids:
                # Each entity has an exclusive Unicode bigram. A held-out
                # entity must not influence the fitted vocabulary.
                assert chr(0x4E00 + entity) * 2 not in vocabulary
        return fitted

    selections = []
    original_select = development.select_from_oof

    def select_spy(y, options):
        selected = original_select(y, options)
        selections.append(selected)
        return selected

    monkeypatch.setattr(FeatureExtractor, "fit", fit_spy)
    monkeypatch.setattr(development, "select_from_oof", select_spy)

    original_information = development.train_informativeness
    informativeness_calls = []

    def information_spy(matrix, labels):
        rows = actual_fit_rows[-1]
        assert len(matrix) == len(rows)
        np.testing.assert_array_equal(labels, [row["label"] for row in rows])
        informativeness_calls.append(len(rows))
        return original_information(matrix, labels)

    monkeypatch.setattr(development, "train_informativeness", information_spy)
    results = development.evaluate_development(entity_pairs, plan=plan)
    assert Counter(actual_fit_ids) == Counter(expected_fit_ids)
    assert len(actual_fit_ids) == 12
    assert len(informativeness_calls) == 12
    # Five methods are selected exactly once per clean outer fit. Corrupted
    # hold-outs must inherit these thresholds, without another selection.
    assert len(selections) == 3 * 5
    assert set(results["clean_metrics"]) == set(development.METHODS)

    # Every corruption shares a method's selected clean threshold. Rate zero
    # also checks pooled predictions retain exactly the clean evaluation rows.
    selected = {(fold["outer_fold"], method): config["threshold"]
                for fold in results["folds"] for method, config in fold["selected"].items()}
    assert len(results["missingness"]) == 2 * 5 * 3 * 5
    assert len(results["fold_missingness"]) == 3 * 2 * 5 * 3 * 5
    removal_fractions = {}
    for row in results["fold_missingness"]:
        assert row["threshold"] == selected[row["outer_fold"], row["method"]]
        mask_key = (row["outer_fold"], row["scenario"], row["missing_rate"], row["seed"])
        removal_fractions.setdefault(mask_key, set()).add(row["effective_removed_fraction"])
    assert all(len(values) == 1 for values in removal_fractions.values())
    for row in results["missingness"]:
        if row["missing_rate"] == 0:
            clean = results["clean_metrics"][row["method"]]
            for metric in ("precision", "recall", "f1", "tp", "fp", "tn", "fn", "n"):
                assert row[metric] == clean[metric]

    n = plan["pooled_evaluation"]["n"]
    assert len(results["clean_predictions"]) == n * len(development.METHODS)
    pooled_rows = [entity_pairs[i] for i in plan["outer_evaluation_indices"]]
    hard_n = sum(row["is_hard_negative"] for row in pooled_rows)
    assert hard_n > 0
    for method in development.METHODS:
        predictions = [row for row in results["clean_predictions"] if row["method"] == method]
        assert len({row["pair_id"] for row in predictions}) == n
        hard = [row for row in predictions if row["is_hard_negative"]]
        hard_stats = results["hard_negatives"][method]["hard"]
        assert hard_stats["n"] == hard_n
        assert hard_stats["fp"] == sum(row["prediction"] for row in hard)
        assert hard_stats["fpr"] == hard_stats["fp"] / hard_n
        stats = results["clean_metrics"][method]
        assert stats["confusion_matrix"] == [[stats["tn"], stats["fp"]], [stats["fn"], stats["tp"]]]
        assert sum(sum(row) for row in stats["confusion_matrix"]) == n
