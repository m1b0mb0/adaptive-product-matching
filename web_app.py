"""Local query-by-example web interface over the shared SearchEngine."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from frozen_v1_runtime import load_corrected_scorer
from matching import norm
from product_catalog import ProductOffer, load_fpv_catalog, load_wdc_catalog
from search_system import SearchEngine


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
CATALOGS = {
    "fpv": ("FPV-двигуни", "FPV_CATALOG_PATH",
            "data/external/fpv_motors/fpv_motors_offers_pilot.csv", load_fpv_catalog),
    "wdc": ("WDC validation small", "WDC_CATALOG_PATH",
            "data/external/wdcproducts80cc20rnd000un_valid_small.json.gz", load_wdc_catalog),
}
METHODS = {
    "renorm": "Renormalization",
    "fixed": "Fixed Weighted",
    "adaptive-corrected": "Adaptive corrected v1",
}
TECHNICAL_LABELS = {
    "family_model": "Серія / модель",
    "stator_code": "Код статора",
    "kv_rpm_per_volt": "KV (об/хв/В)",
    "voltage_s": "Живлення (S)",
    "units_in_listing": "Одиниць у пропозиції",
    "seller_sku": "Артикул",
}
LOOKUP_LIMIT = 12
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engines: dict[str, dict[str, SearchEngine]] = {}
    unavailable: dict[str, str] = {}
    for key, (_, env_name, default_path, load_catalog) in CATALOGS.items():
        path = Path(os.environ.get(env_name) or ROOT / default_path)
        try:
            catalog = load_catalog(path)
            engines[key] = {
                method: SearchEngine(catalog, method=method) for method in ("renorm", "fixed")
            }
        except FileNotFoundError:
            unavailable[key] = "Локальний файл відсутній."
        except (ValueError, OSError) as error:
            logger.warning("Could not load catalog %s: %s", key, error)
            unavailable[key] = "Не вдалося прочитати каталог. Перевірте файл і його формат."

    if engines:
        try:
            scorer = load_corrected_scorer()
        except (ValueError, OSError) as error:
            logger.warning("Adaptive corrected is unavailable: %s", error)
        else:
            for methods in engines.values():
                methods["adaptive-corrected"] = SearchEngine(
                    methods["renorm"].catalog, method="adaptive-corrected", pair_scorer=scorer,
                )
    app.state.engines = engines
    app.state.unavailable = unavailable
    yield


def safe_url(value: str) -> str | None:
    if any(char.isspace() or ord(char) < 32 or char in "\\\x7f" for char in value):
        return None
    try:
        parts = urlsplit(value)
        if parts.scheme in {"http", "https"} and parts.hostname and parts.username is None:
            _ = parts.port  # urlsplit validates an explicit port when it is read.
            return value
    except ValueError:
        pass
    return None


def catalog_price(offer: ProductOffer) -> str:
    price_uah = offer.attributes.get("price_uah", "").strip()
    if price_uah:
        return f"{price_uah} грн"
    price = offer.attributes.get("price", "").strip()
    currency = offer.attributes.get("priceCurrency", "").strip()
    return " ".join(value for value in (price, currency) if value) if price else ""


def comparison_rows(left: ProductOffer, right: ProductOffer) -> list[tuple[str, str, str, str | None]]:
    same_source = bool(norm(left.source)) and norm(left.source) == norm(right.source)
    fields = [
        ("Назва", left.title, right.title, False),
        ("ID пропозиції", left.offer_id, right.offer_id, False),
        ("Бренд", left.brand, right.brand, True),
        ("Джерело", left.source, right.source, True),
    ]
    for key, label in TECHNICAL_LABELS.items():
        left_value = left.technical_attributes.get(key, "")
        right_value = right.technical_attributes.get(key, "")
        if left_value.strip() or right_value.strip():
            fields.append((label, left_value, right_value, key != "seller_sku" or same_source))
    left_price, right_price = catalog_price(left), catalog_price(right)
    if left_price or right_price:
        fields.append(("Ціна в каталозі", left_price, right_price, False))
    fields.append(("Опис", left.description, right.description, False))

    rows = []
    for label, left_value, right_value, comparable in fields:
        left_value, right_value = left_value.strip(), right_value.strip()
        state = None
        if not left_value or not right_value:
            state = "missing"
        elif comparable:
            state = "same" if left_value.casefold() == right_value.casefold() else "different"
        rows.append((label, left_value, right_value, state))
    return rows


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=WEB / "templates")
templates.env.filters["safe_url"] = safe_url
templates.env.filters["catalog_price"] = catalog_price
templates.env.globals["technical_labels"] = TECHNICAL_LABELS
templates.env.globals["method_names"] = METHODS


def render(request: Request, template: str, status_code: int = 200, **context):
    return templates.TemplateResponse(
        request=request, name=template, context=context, status_code=status_code,
    )


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, error: RequestValidationError):
    return render(request, "error.html", 422, message="Перевірте параметри пошуку та спробуйте ще раз.")


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    catalog: Literal["fpv", "wdc"] | None = None,
    q: Annotated[str, Query(max_length=256)] = "",
):
    engines = request.app.state.engines
    selected = catalog or next(iter(engines), None)
    offers = engines[selected]["renorm"].catalog if selected in engines else {}
    q = q.strip()
    needle = q.casefold()
    matches = [
        offer for offer in offers.values()
        if needle in offer.title.casefold() or needle == offer.offer_id.casefold()
    ]
    matches.sort(key=lambda offer: (offer.offer_id.casefold() != needle,
                                    offer.title.casefold(), offer.offer_id))
    return render(
        request, "index.html", catalogs=CATALOGS, selected=selected, q=q,
        unavailable=request.app.state.unavailable, offers=matches[:LOOKUP_LIMIT],
        count=len(offers), match_count=len(matches),
    )


@app.get("/search", response_class=HTMLResponse)
def search(
    request: Request,
    catalog: Literal["fpv", "wdc"],
    offer_id: Annotated[str, Query(min_length=1, max_length=256)],
    q: Annotated[str, Query(max_length=256)] = "",
    run: bool = False,
    method: Literal["renorm", "fixed", "adaptive-corrected"] = "renorm",
    candidate_k: Annotated[int, Query(ge=1, le=100)] = 20,
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    other_sources_only: bool = False,
    compare_to: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
):
    methods = request.app.state.engines.get(catalog)
    if methods is None:
        return render(request, "error.html", 404, message="Цей каталог недоступний.")
    offers = methods["renorm"].catalog
    offer = offers.get(offer_id)
    if offer is None:
        return render(request, "error.html", 404, message="Пропозицію з таким ID не знайдено.")

    results_url = request.url.remove_query_params("compare_to")
    if compare_to is not None:
        if not run:
            return render(request, "error.html", 400, message="Спочатку виконайте пошук відповідностей.",
                          back_url=results_url, back_label="До початкової пропозиції")
        if compare_to == offer_id or compare_to not in offers:
            return render(request, "error.html", 404, message="Цю пропозицію неможливо порівняти в обраному каталозі.",
                          back_url=results_url, back_label="До результатів")

    source = norm(offer.source)
    source_filter = bool(source) and any(norm(other.source) not in {"", source}
                                         for other in offers.values())
    message, status, results = "", 200, None
    if method not in methods:
        message, status = "Обраний метод зараз недоступний.", 400
        method = "renorm"
    elif top_k > candidate_k:
        message, status = "Кількість результатів має бути не більшою за кількість кандидатів.", 400
    elif other_sources_only and not source_filter:
        message, status = "Для цієї пропозиції фільтр за іншими джерелами недоступний.", 400
    elif run:
        results = methods[method].search(
            offer_id, candidate_k=candidate_k, top_k=top_k, other_sources_only=other_sources_only,
        )
        if compare_to is not None:
            result = next((row for row in results if row.offer.offer_id == compare_to), None)
            if result is None:
                return render(request, "error.html", 404, message="Пропозиції немає в поточній видачі.",
                              back_url=results_url, back_label="До результатів")
            return render(
                request, "compare.html", offer=offer, result=result,
                rows=comparison_rows(offer, result.offer), results_url=results_url, method=method,
            )
    return render(
        request, "search.html", status, catalog=catalog, catalog_name=CATALOGS[catalog][0],
        offer=offer, q=q, methods=methods, method=method, results=results, message=message,
        candidate_k=candidate_k, top_k=top_k, source_filter=source_filter,
        other_sources_only=other_sources_only, results_url=results_url,
    )
