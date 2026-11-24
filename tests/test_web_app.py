import csv
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import web_app
from fpv_dataset import OFFER_COLUMNS
from matching import FeatureExtractor
from product_catalog import load_fpv_catalog
from search_system import SearchEngine


@pytest.fixture
def paths(tmp_path, monkeypatch):
    fpv_path = tmp_path / "offers.csv"
    with fpv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*OFFER_COLUMNS, "label", "qrels"])
        writer.writeheader()
        for offer_id, title, source in [
            ("A", "ROBO <script>motor</script> 0802", "Джерело A"),
            ("B", "ROBO motor 0802", "Джерело B"),
            ("C", "ROBO motor 0802", ""),
            ("D", "", "Джерело A"),
        ]:
            writer.writerow({
                "offer_id": offer_id, "raw_title": title, "store": source,
                "stator_code": "0802" if offer_id == "A" else "",
                "source_url": "https://example.org/motor" if offer_id == "A" else "javascript:alert(1)",
                "entity_id": "must-not-leak", "label": "must-not-leak", "qrels": "must-not-leak",
            })

    wdc_path = tmp_path / "validation.jsonl"
    wdc_path.write_text(json.dumps({
        "id_left": "W1", "title_left": "Camera body",
        "id_right": "W2", "title_right": "Camera body kit",
        "label": "must-not-leak", "cluster_id_left": "must-not-leak",
    }) + "\n", encoding="utf-8")
    monkeypatch.setenv("FPV_CATALOG_PATH", str(fpv_path))
    monkeypatch.setenv("WDC_CATALOG_PATH", str(wdc_path))

    def missing_bundle():
        raise FileNotFoundError("fixture without bundle")

    monkeypatch.setattr(web_app, "load_corrected_scorer", missing_bundle)
    return fpv_path, wdc_path


@pytest.fixture
def client(paths):
    with TestClient(web_app.app) as browser:
        yield browser


def test_home_catalog_and_title_or_id_lookup(client):
    home = client.get("/")
    assert home.status_code == 200
    assert 'lang="uk"' in home.text
    assert home.context["selected"] == "fpv"
    assert client.get("/static/search.css").status_code == 200
    wdc = client.get("/", params={"catalog": "wdc", "q": "  CAMERA  "})
    assert [offer.offer_id for offer in wdc.context["offers"]] == ["W1", "W2"]
    assert "offer_id=W1" in wdc.text
    assert "must-not-leak" not in wdc.text
    exact_id = client.get("/", params={"q": "d"})
    assert [offer.offer_id for offer in exact_id.context["offers"]] == ["D"]
    assert "Назва відсутня" in exact_id.text
    no_matches = client.get("/", params={"q": "unmatched"})
    assert "За цим запитом пропозицій немає" in no_matches.text


def test_selected_offer_escapes_title_and_uses_only_safe_links(client):
    selected = client.get("/search", params={"catalog": "fpv", "offer_id": "A"})
    assert selected.status_code == 200
    assert selected.context["offer"].offer_id == "A"
    assert selected.context["results"] is None
    assert "&lt;script&gt;motor&lt;/script&gt;" in selected.text
    assert '<script>motor</script>' not in selected.text
    assert "Код статора" in selected.text and "0802" in selected.text
    assert "Немає даних" in selected.text
    assert 'href="https://example.org/motor"' in selected.text
    assert 'rel="noopener noreferrer"' in selected.text
    assert 'name="other_sources_only"' in selected.text
    results = client.get("/search", params={"catalog": "fpv", "offer_id": "A", "run": "true"})
    assert "javascript:" not in results.text


@pytest.mark.parametrize("method", ["renorm", "fixed"])
def test_ranked_cards_match_python_api(client, paths, method):
    params = {"catalog": "fpv", "offer_id": "A", "run": "true", "candidate_k": 3, "top_k": 2}
    if method == "fixed":
        params["method"] = method
    response = client.get("/search", params=params)
    expected = SearchEngine(load_fpv_catalog(paths[0]), method=method).search("A", 3, 2)
    assert response.status_code == 200
    assert response.context["method"] == method
    assert response.context["results"] == expected
    assert expected
    assert response.text.count('class="panel result-card"') == len(expected)
    for result in expected:
        assert f"{result.similarity_score:.6f}" in response.text
    filtered = client.get("/search", params=params | {"other_sources_only": "true"})
    assert [result.offer.offer_id for result in filtered.context["results"]] == ["B"]


def test_wdc_requests_reuse_initialized_engine_without_annotations_or_fit(client, monkeypatch):
    engine = web_app.app.state.engines["wdc"]["renorm"]
    seen = []
    original_transform = engine.extractor.transform

    def transform(pairs):
        for pair in pairs:
            assert set(pair) == {f"{field}_{side}" for field in ("id", "title", "description", "brand")
                                 for side in ("left", "right")}
        seen.append(len(pairs))
        return original_transform(pairs)

    def forbidden(*args, **kwargs):
        pytest.fail("An HTTP request attempted to fit a feature extractor")

    monkeypatch.setattr(engine.extractor, "transform", transform)
    monkeypatch.setattr(FeatureExtractor, "fit", forbidden)
    params = {"catalog": "wdc", "offer_id": "W1", "run": "true"}
    with ThreadPoolExecutor(max_workers=2) as requests:
        responses = list(requests.map(lambda _: client.get("/search", params=params), range(2)))
    assert seen == [1, 1]
    assert responses[0].context["results"] == responses[1].context["results"]
    for response in responses:
        assert response.status_code == 200
        assert 'name="other_sources_only"' not in response.text
        assert "must-not-leak" not in response.text
        assert "Відкрити пропозицію" not in response.text


def test_unknown_id_empty_results_and_invalid_parameters(client):
    params = {"catalog": "fpv", "offer_id": "D", "run": "true"}
    unknown = client.get("/search", params=params | {"offer_id": "unknown"})
    assert unknown.status_code == 404 and "не знайдено" in unknown.text
    empty = client.get("/search", params=params)
    assert empty.status_code == 200 and empty.context["results"] == []
    assert "Схожих пропозицій за цими параметрами немає" in empty.text
    assert client.get("/search", params=params | {"top_k": 0}).status_code == 422
    assert client.get("/search", params=params | {"candidate_k": 2}).status_code == 400
    assert 'value="adaptive-corrected"' not in empty.text
    unavailable = client.get("/search", params=params | {"method": "adaptive-corrected"})
    assert unavailable.status_code == 400 and "метод зараз недоступний" in unavailable.text
    unknown_source = client.get("/search", params=params | {"offer_id": "C", "other_sources_only": "true"})
    assert unknown_source.status_code == 400


def test_missing_catalog_falls_back_then_reports_no_catalogs(paths):
    fpv_path, wdc_path = paths
    fpv_path.unlink()
    with TestClient(web_app.app) as client:
        home = client.get("/")
        assert home.context["selected"] == "wdc"
        assert "Локальний файл відсутній" in home.text
        assert client.get("/search", params={"catalog": "fpv", "offer_id": "A"}).status_code == 404
    wdc_path.unlink()
    with TestClient(web_app.app) as client:
        home = client.get("/")
        assert home.status_code == 200 and "Каталоги недоступні" in home.text
