"""Hybrid retrieval: BM25 + dense in Qdrant -> RRF -> adjacent expansion -> cross-encoder rerank.

Every stage is exposed separately so the evaluation can score each mode on the same queries.
"""

import time
from dataclasses import dataclass, field
from typing import Literal

from qdrant_client import models

from . import config
from .indexer import point_id
from .models import bm25_model, dense_model, qdrant_client, reranker_model

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank", "full"]
MODES: tuple[Mode, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank", "full")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    section_heading: str
    score: float
    origin: str  # "dense" | "bm25" | "hybrid" | "adjacent"
    rerank_score: float | None = None
    payload: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_point(cls, point: models.ScoredPoint | models.Record, origin: str) -> "RetrievedChunk":
        p = point.payload or {}
        return cls(
            chunk_id=p["chunk_id"],
            text=p["text"],
            source_file=p["source_file"],
            chunk_index=p["chunk_index"],
            section_heading=p.get("section_heading") or "",
            score=float(getattr(point, "score", 0.0) or 0.0),
            origin=origin,
            payload=p,
        )


@dataclass
class RetrievalResult:
    query: str
    mode: Mode
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float]
    candidates_reranked: int = 0
    cache_hit: bool = False

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 1)


class _Timer:
    def __init__(self, timings: dict[str, float], stage: str):
        self.timings, self.stage = timings, stage

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        self.timings[self.stage] = round((time.perf_counter() - self.t0) * 1000, 1)


class HybridRetriever:
    def __init__(self, collection: str = config.QDRANT_COLLECTION,
                 cache_size: int | None = None):
        from .cache import RetrievalCache

        self.client = qdrant_client()
        self.collection = collection
        # Exact LRU over retrieve(); evals pass cache_size=0 so latency is measured cold.
        self.cache = RetrievalCache(config.RETRIEVAL_CACHE_SIZE if cache_size is None
                                    else cache_size)
        if not self.client.collection_exists(collection):
            raise RuntimeError(f"Collection '{collection}' not found; run `rsa index` first.")

    # ── Query encoding ────────────────────────────────────────────────────────
    def _dense_query(self, query: str) -> list[float]:
        return next(iter(dense_model().query_embed(query))).tolist()

    def _bm25_query(self, query: str) -> models.SparseVector:
        s = next(iter(bm25_model().query_embed(query)))
        return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())

    # ── Stage 1: single retrievers ────────────────────────────────────────────
    def dense_search(self, query: str, limit: int = config.RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
        res = self.client.query_points(
            self.collection, query=self._dense_query(query), using=config.DENSE_VECTOR,
            limit=limit, with_payload=True,
        )
        return [RetrievedChunk.from_point(p, "dense") for p in res.points]

    def bm25_search(self, query: str, limit: int = config.RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
        res = self.client.query_points(
            self.collection, query=self._bm25_query(query), using=config.SPARSE_VECTOR,
            limit=limit, with_payload=True,
        )
        return [RetrievedChunk.from_point(p, "bm25") for p in res.points]

    # ── Stage 2: RRF fusion inside Qdrant ─────────────────────────────────────
    def hybrid_search(
        self, query: str, limit: int = config.RRF_LIMIT, prefetch_k: int = config.RETRIEVAL_TOP_K
    ) -> list[RetrievedChunk]:
        res = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=self._dense_query(query), using=config.DENSE_VECTOR,
                                limit=prefetch_k),
                models.Prefetch(query=self._bm25_query(query), using=config.SPARSE_VECTOR,
                                limit=prefetch_k),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=config.RRF_K)),
            limit=limit,
            with_payload=True,
        )
        return [RetrievedChunk.from_point(p, "hybrid") for p in res.points]

    # ── Stage 3: adjacent-chunk expansion ─────────────────────────────────────
    def expand_adjacent(
        self,
        chunks: list[RetrievedChunk],
        window: int = config.ADJACENT_WINDOW,
        cap: int = config.MAX_RERANK_CANDIDATES,
    ) -> list[RetrievedChunk]:
        # The cap bounds reranker cost, so it applies to fused hits too, not just to
        # the neighbours added here (RRF_LIMIT can exceed a small cap).
        chunks = list(chunks)[:cap]
        seen = {(c.source_file, c.chunk_index) for c in chunks}
        wanted: list[str] = []
        # Walk in fused-rank order so the cap keeps neighbours of the strongest hits.
        for c in chunks:
            total = c.payload.get("total_chunks")
            for offset in range(-window, window + 1):
                idx = c.chunk_index + offset
                key = (c.source_file, idx)
                if offset == 0 or idx < 0 or (total is not None and idx >= total) or key in seen:
                    continue
                seen.add(key)
                wanted.append(point_id(c.source_file, idx))

        room = max(cap - len(chunks), 0)
        wanted = wanted[:room]
        if not wanted:
            return list(chunks)

        records = self.client.retrieve(self.collection, ids=wanted, with_payload=True)
        by_id = {str(r.id): r for r in records}
        neighbours = [RetrievedChunk.from_point(by_id[i], "adjacent") for i in wanted if i in by_id]
        return list(chunks) + neighbours

    # ── Stage 4: cross-encoder rerank ─────────────────────────────────────────
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = config.RERANK_TOP_K
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = list(reranker_model().rerank(query, [c.text for c in chunks],
                                              batch_size=config.RERANK_BATCH_SIZE))
        for c, s in zip(chunks, scores, strict=True):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]

    # ── Pipeline entry point ──────────────────────────────────────────────────
    def retrieve(
        self, query: str, mode: Mode = "full", top_k: int = config.RERANK_TOP_K
    ) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'; expected one of {MODES}")
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")

        key = self.cache.key(query, mode, top_k)
        if self.cache.maxsize > 0:
            t0 = time.perf_counter()
            hit = self.cache.get(key)
            if hit is not None:
                hit.timings_ms = {"cache": round((time.perf_counter() - t0) * 1000, 2)}
                hit.cache_hit = True
                return hit

        t: dict[str, float] = {}
        reranked = 0

        if mode == "dense":
            with _Timer(t, "dense"):
                chunks = self.dense_search(query, limit=top_k)
        elif mode == "bm25":
            with _Timer(t, "bm25"):
                chunks = self.bm25_search(query, limit=top_k)
        elif mode == "hybrid":
            with _Timer(t, "hybrid_rrf"):
                chunks = self.hybrid_search(query, limit=top_k)
        else:
            with _Timer(t, "hybrid_rrf"):
                candidates = self.hybrid_search(query)
            if mode == "full":
                with _Timer(t, "expand_adjacent"):
                    candidates = self.expand_adjacent(candidates)
            reranked = len(candidates)
            with _Timer(t, "rerank"):
                chunks = self.rerank(query, candidates, top_k=top_k)

        result = RetrievalResult(query=query, mode=mode, chunks=chunks, timings_ms=t,
                                 candidates_reranked=reranked)
        self.cache.put(key, result)
        return result
