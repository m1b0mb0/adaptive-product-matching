import json

import pytest

from audit_wdc import read_pairs
from product_catalog import ProductOffer, load_wdc_catalog
from wdc_qrels import build_wdc_qrels


def pair(left, right, left_cluster, right_cluster, label=1):
    return {
        "id_left": left, "id_right": right,
        "title_left": f"Product {left}", "title_right": f"Product {right}",
        "cluster_id_left": left_cluster, "cluster_id_right": right_cluster,
        "label": label,
    }


def catalog(*offer_ids):
    return {offer_id: ProductOffer(offer_id, title=f"Product {offer_id}") for offer_id in offer_ids}


def test_one_positive_pair_yields_two_directed_judgments(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps(pair("b", "a", 1, 1, label="1")) + "\n", encoding="utf-8")
    offers = load_wdc_catalog(path)
    before = offers.copy()

    qrels = build_wdc_qrels(read_pairs(path), offers)

    assert qrels == {"a": frozenset({"b"}), "b": frozenset({"a"})}
    assert offers == before


def test_larger_and_independent_clusters_are_symmetric_and_order_independent():
    rows = [
        pair("a", "b", 1, 1), pair("c", "a", 1, 1), pair("b", "c", 1, 1),
        pair("e", "d", 2, 2), pair("c", "d", 1, 2, label=0),
        pair("f", "a", 3, 1, label=0),
    ]
    qrels = build_wdc_qrels(rows, catalog("f", "d", "b", "a", "e", "c"))
    reversed_qrels = build_wdc_qrels(rows[::-1], catalog("a", "b", "c", "d", "e", "f"))

    assert list(qrels) == list(reversed_qrels) == ["a", "b", "c", "d", "e", "f"]
    assert qrels == reversed_qrels == {
        "a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"},
        "d": {"e"}, "e": {"d"}, "f": set(),
    }
    for query, relevant in qrels.items():
        assert query not in relevant
        assert all(query in qrels[offer_id] for offer_id in relevant)


def test_missing_positive_inside_cluster_is_rejected_without_inference():
    rows = [pair("a", "b", 1, 1), pair("b", "c", 1, 1)]
    with pytest.raises(ValueError, match=r"Missing explicit positive pair in cluster '1': 'a', 'c'"):
        build_wdc_qrels(rows, catalog("a", "b", "c"))


@pytest.mark.parametrize(("rows", "message"), [
    ([pair("a", "b", 1, 1), pair("b", "a", 1, 1, label=0)], "Conflicting labels"),
    ([pair("a", "b", 1, 1, label=0)], "Label and cluster_id disagree"),
    ([pair("a", "b", 1, 2)], "Label and cluster_id disagree"),
    ([pair("a", "b", 1, None)], "Missing cluster_id"),
    ([pair("a", "b", 1, 1), pair("a", "c", 2, 2)], "Inconsistent attributes or cluster_id"),
    ([pair("a", "b", 1, 1), pair("a", "c", 1, 1) | {"brand_left": "Changed"}],
     "Inconsistent attributes or cluster_id"),
    ([pair("a", "b", 1, 1), pair("b", "a", 1, 1)], "Duplicate unordered pairs"),
    ([pair("a", "a", 1, 1)], "Self-pairs"),
])
def test_inconsistent_annotations_are_rejected(rows, message):
    with pytest.raises(ValueError, match=message):
        build_wdc_qrels(rows, catalog("a", "b", "c"))


@pytest.mark.parametrize("offers", [catalog("a"), catalog("b")])
def test_positive_endpoints_must_belong_to_catalog(offers):
    with pytest.raises(ValueError, match="Annotated offer_id .* missing from the catalog"):
        build_wdc_qrels([pair("a", "b", 1, 1)], offers)


def test_catalog_attributes_must_match_annotations():
    offers = catalog("a", "b")
    offers["a"] = ProductOffer("a", title="Different title")
    with pytest.raises(ValueError, match="Catalog attributes differ"):
        build_wdc_qrels([pair("a", "b", 1, 1)], offers)


def test_catalog_offers_without_annotations_are_rejected():
    with pytest.raises(ValueError, match="Catalog offer_id 'c' has no pairwise annotations"):
        build_wdc_qrels([pair("a", "b", 1, 1)], catalog("a", "b", "c"))


def test_negative_pair_produces_empty_qrels_for_observed_singletons():
    qrels = build_wdc_qrels([pair("a", "b", 1, 2, label=0)], catalog("a", "b"))
    assert qrels == {"a": frozenset(), "b": frozenset()}
