"""Label-free candidate retrieval and pairwise ranking for a local catalog.

The index is built from the supplied catalog, not from pair labels. This is a
transductive search demonstration, not an independent matching experiment.
Retrieval uses title overlap to choose a bounded set; reranking applies the
existing grouped Fixed/Renormalization formulas to each query-candidate pair.
All scores are algorithm outputs, not calibrated identity probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from logistic_baseline import SimilarityLogisticRegression
from matching import FIELDS, FeatureExtractor, evidence_scores, norm
from product_catalog import ProductOffer, offer_to_pair


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _catalog_snapshot(catalog: Mapping[str, ProductOffer]) -> dict[str, ProductOffer]:
    if not catalog:
        raise ValueError("Cannot search an empty catalog")
    if any(not isinstance(key, str) or not isinstance(offer, ProductOffer)
           or key != offer.offer_id for key, offer in catalog.items()):
        raise ValueError("Catalog keys must equal ProductOffer.offer_id")
    return dict(sorted(catalog.items()))


@dataclass(frozen=True)
class Candidate:
    offer: ProductOffer
    retrieval_score: float


@dataclass(frozen=True)
class SearchResult:
    offer: ProductOffer
    retrieval_score: float
    similarity_score: float
    # None means that the external scorer does not expose this diagnostic.
    # A missing individual built-in feature is represented by a None value.
    feature_similarities: dict[str, float | None] | None


class TfidfCandidateRetriever:
    """Deterministic title-only retrieval, fitted once at construction.

    Empty/one-character/zero-vector queries yield no candidates. Candidates
    with zero title cosine are omitted: top-K is a maximum, not a promise to
    return arbitrary rows. Source filtering occurs before taking top-K.
    """

    def __init__(self, catalog: Mapping[str, ProductOffer]) -> None:
        self.catalog = _catalog_snapshot(catalog)
        self.offer_ids = tuple(self.catalog)
        self._positions = {offer_id: index for index, offer_id in enumerate(self.offer_ids)}
        titles = [norm(offer.title) for offer in self.catalog.values()]
        self.vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        if any(len(title) >= 2 for title in titles):
            self.vectorizer = TfidfVectorizer(
                analyzer="char", ngram_range=(2, 4), max_features=30000,
                sublinear_tf=True,
            )
            # Match FeatureExtractor's vocabulary/IDF convention: empty titles
            # are not statistical documents, but remain catalog rows to omit.
            self.vectorizer.fit([title for title in titles if title])
            self._matrix = self.vectorizer.transform(titles)

    def retrieve(self, query_id: str, candidate_k: int = 20, *,
                 other_sources_only: bool = False) -> list[Candidate]:
        candidate_k = _positive_integer(candidate_k, "candidate_k")
        if not isinstance(other_sources_only, bool):
            raise ValueError("other_sources_only must be boolean")
        if not isinstance(query_id, str) or query_id not in self.catalog:
            raise ValueError(f"Unknown offer_id: {query_id!r}")
        query = self.catalog[query_id]
        query_source = norm(query.source)
        if other_sources_only and not query_source:
            raise ValueError("Cannot filter other sources: the query source is unknown")
        if self._matrix is None:
            return []
        query_vector = self._matrix[self._positions[query_id]]
        if query_vector.nnz == 0:
            return []
        similarities = np.clip(
            np.asarray((self._matrix @ query_vector.T).toarray()).ravel(), 0.0, 1.0,
        )
        candidates = []
        for offer_id, similarity in zip(self.offer_ids, similarities):
            if offer_id == query_id or similarity <= 0:
                continue
            offer = self.catalog[offer_id]
            if other_sources_only:
                source = norm(offer.source)
                if not source or source == query_source:
                    continue
            candidates.append(Candidate(offer, float(similarity)))
        candidates.sort(key=lambda candidate: (-candidate.retrieval_score,
                                                candidate.offer.offer_id))
        return candidates[:candidate_k]


class PairScorer(Protocol):
    """Extension contract: an already-fitted scorer; query-time fitting forbidden.

    The caller must supply the exact paired representation and estimator used
    during training. This interface deliberately has no artifact deserializer
    or model/threshold selection. Returned scores must be finite and in [0, 1].
    """

    def score_pairs(self, pairs: list[dict]) -> np.ndarray: ...


class FrozenLogisticPairScorer:
    """Use an already-fitted FeatureExtractor + LR without refitting either.

    The caller is responsible for provenance and compatibility of the fitted
    bundle. Per-fold development coefficients alone are not a deployable
    bundle: the original fitted TF-IDF vocabulary and IDF are also required.
    A model's output has no established external calibration.
    """

    def __init__(self, extractor: FeatureExtractor,
                 model: SimilarityLogisticRegression) -> None:
        # These harmless public calls validate fittedness without fitting.
        extractor.transform([])
        model.metadata()
        self.extractor = extractor
        self.model = model

    def score_pairs(self, pairs: list[dict]) -> np.ndarray:
        return self.model.predict_score(self.extractor.transform(pairs).matrix)


class SearchEngine:
    """Build catalog representations once, retrieve first, then rerank pairs.

    ``renorm`` is a temporary existing baseline default, not a new method.
    Catalog-fit title/description IDF is an unsupervised demo representation;
    these scores are not directly comparable with historical fold-fit scores.
    Optional logistic/corrected scorers use their own frozen training extractor.
    No thresholds, identity decisions, or accuracy metrics are computed.
    """

    def __init__(self, catalog: Mapping[str, ProductOffer], method: str = "renorm", *,
                 pair_scorer: PairScorer | None = None) -> None:
        if method not in {"fixed", "renorm", "logistic", "adaptive-corrected"}:
            raise ValueError("Search method must be fixed, renorm, logistic, or adaptive-corrected")
        if method in {"logistic", "adaptive-corrected"} and pair_scorer is None:
            raise ValueError(f"{method} search requires an already-fitted pair_scorer")
        if method not in {"logistic", "adaptive-corrected"} and pair_scorer is not None:
            raise ValueError("pair_scorer is only supported with logistic or adaptive-corrected")
        if pair_scorer is not None and not callable(getattr(pair_scorer, "score_pairs", None)):
            raise ValueError("pair_scorer must implement score_pairs")
        self.catalog = _catalog_snapshot(catalog)
        self.method = method
        self.pair_scorer = pair_scorer
        self.retriever = TfidfCandidateRetriever(self.catalog)
        self.extractor: FeatureExtractor | None = None
        if pair_scorer is None:
            self.extractor = FeatureExtractor().fit([
                offer_to_pair(offer, offer) for offer in self.catalog.values()
            ])

    def retrieve(self, query_id: str, candidate_k: int = 20, *,
                 other_sources_only: bool = False) -> list[Candidate]:
        return self.retriever.retrieve(query_id, candidate_k,
                                       other_sources_only=other_sources_only)

    def search(self, query_id: str, candidate_k: int = 20, top_k: int = 5, *,
               other_sources_only: bool = False) -> list[SearchResult]:
        top_k = _positive_integer(top_k, "top_k")
        candidates = self.retrieve(query_id, candidate_k,
                                   other_sources_only=other_sources_only)
        if not candidates:
            return []
        pairs = [offer_to_pair(self.catalog[query_id], candidate.offer)
                 for candidate in candidates]
        features = None
        if self.pair_scorer is None:
            assert self.extractor is not None
            features = self.extractor.transform(pairs).matrix
            scores = evidence_scores(features, self.method)
        else:
            scores = np.asarray(self.pair_scorer.score_pairs(pairs), dtype=float)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all() \
                or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Pair scorer must return one finite score in [0, 1] per candidate")
        results = []
        for index, (candidate, score) in enumerate(zip(candidates, scores)):
            similarities = None if features is None else {
                field: None if np.isnan(value) else float(value)
                for field, value in zip(FIELDS, features[index])
            }
            results.append(SearchResult(candidate.offer, candidate.retrieval_score,
                                        float(score), similarities))
        results.sort(key=lambda result: (-result.similarity_score, -result.retrieval_score,
                                         result.offer.offer_id))
        return results[:top_k]
