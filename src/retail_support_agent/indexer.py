"""Index policy chunks into one Qdrant collection holding a dense and a BM25 named vector."""

import json
import time
import uuid
from pathlib import Path

from qdrant_client import models

from . import config
from .models import bm25_model, dense_model, qdrant_client


def point_id(source_file: str, chunk_index: int) -> str:
    # Derived from document position so adjacent chunks can be fetched by ID without a search.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_file}#{chunk_index}"))


def load_chunks(path: Path = config.CHUNKS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def create_collection(recreate: bool = True) -> None:
    client = qdrant_client()
    if client.collection_exists(config.QDRANT_COLLECTION):
        if not recreate:
            return
        client.delete_collection(config.QDRANT_COLLECTION)

    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={
            config.DENSE_VECTOR: models.VectorParams(
                size=config.DENSE_DIM, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            # FastEmbed's BM25 emits term-frequency weights only; Qdrant supplies corpus IDF.
            config.SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def build_index(chunks_path: Path = config.CHUNKS_PATH, batch_size: int = 32) -> dict:
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}; run ingestion first.")

    texts = [c["text"] for c in chunks]
    t0 = time.perf_counter()
    dense_vecs = list(dense_model().embed(texts, batch_size=batch_size))
    t_dense = time.perf_counter() - t0

    t0 = time.perf_counter()
    sparse_vecs = list(bm25_model().embed(texts, batch_size=batch_size))
    t_sparse = time.perf_counter() - t0

    if len(dense_vecs[0]) != config.DENSE_DIM:
        raise ValueError(
            f"{config.DENSE_MODEL} produced {len(dense_vecs[0])}-dim vectors, "
            f"config expects {config.DENSE_DIM}"
        )

    points = [
        models.PointStruct(
            id=point_id(c["source_file"], c["chunk_index"]),
            vector={
                config.DENSE_VECTOR: d.tolist(),
                config.SPARSE_VECTOR: models.SparseVector(
                    indices=s.indices.tolist(), values=s.values.tolist()
                ),
            },
            payload=c,
        )
        for c, d, s in zip(chunks, dense_vecs, sparse_vecs, strict=True)
    ]

    create_collection(recreate=True)
    t0 = time.perf_counter()
    qdrant_client().upsert(config.QDRANT_COLLECTION, points=points, wait=True)
    t_upsert = time.perf_counter() - t0

    stored = qdrant_client().count(config.QDRANT_COLLECTION, exact=True).count
    return {
        "collection": config.QDRANT_COLLECTION,
        "points": stored,
        "documents": len({c["source_file"] for c in chunks}),
        "dense_model": config.DENSE_MODEL,
        "sparse_model": config.SPARSE_MODEL,
        "avg_bm25_terms": round(sum(len(s.indices) for s in sparse_vecs) / len(sparse_vecs), 1),
        "seconds": {
            "dense_embed": round(t_dense, 2),
            "bm25_embed": round(t_sparse, 2),
            "upsert": round(t_upsert, 2),
        },
    }
