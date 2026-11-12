"""Index policy chunks into one Qdrant collection holding a dense and a BM25 named vector."""

import hashlib
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


FINGERPRINT_KEY = "file_sha"   # payload field: hash of the source file's chunks


def file_fingerprints(chunks: list[dict]) -> dict[str, str]:
    """source_file -> SHA-256 of that file's chunks (text + metadata), so a change to the
    document *or* to the chunker re-indexes the file."""
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        by_file.setdefault(c["source_file"], []).append(c)
    return {
        f: hashlib.sha256(json.dumps(sorted(rows, key=lambda r: r.get("chunk_index", 0)),
                                     sort_keys=True, default=str).encode()).hexdigest()
        for f, rows in by_file.items()
    }


def indexed_fingerprints(client) -> dict[str, str | None]:
    """source_file -> fingerprint stored in the collection (None for points indexed
    before fingerprints existed, which forces that file to re-index)."""
    out: dict[str, str | None] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            config.QDRANT_COLLECTION, limit=1000, offset=offset,
            with_payload=["source_file", FINGERPRINT_KEY], with_vectors=False)
        for pt in points:
            payload = pt.payload or {}
            f = payload.get("source_file")
            if f is not None and out.get(f) is None:
                out[f] = payload.get(FINGERPRINT_KEY)
        if offset is None:
            return out


def _file_filter(source_file: str) -> models.FilterSelector:
    return models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(
        key="source_file", match=models.MatchValue(value=source_file))]))


def build_index(chunks_path: Path = config.CHUNKS_PATH, batch_size: int = 32,
                incremental: bool = False) -> dict:
    """Embed chunks and load them into Qdrant.

    Full (default): drop and rebuild the collection.
    Incremental: compare each source file's chunk fingerprint with the one stored on
    its points; re-embed only new or changed files, delete points of removed files,
    and leave unchanged files untouched. Falls back to a full build when the
    collection does not exist yet.
    """
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}; run ingestion first.")
    fingerprints = file_fingerprints(chunks)
    client = qdrant_client()

    files = {"added": [], "changed": [], "removed": [], "unchanged": []}
    if incremental and client.collection_exists(config.QDRANT_COLLECTION):
        mode = "incremental"
        stored = indexed_fingerprints(client)
        for f, sha in sorted(fingerprints.items()):
            key = "unchanged" if stored.get(f) == sha else (
                "changed" if f in stored else "added")
            files[key].append(f)
        files["removed"] = sorted(set(stored) - set(fingerprints))
        for f in files["changed"] + files["removed"]:
            # Deleting first also drops chunk indices that no longer exist.
            client.delete(config.QDRANT_COLLECTION, points_selector=_file_filter(f), wait=True)
        todo = set(files["added"]) | set(files["changed"])
        to_embed = [c for c in chunks if c["source_file"] in todo]
    else:
        mode = "full"
        files["added"] = sorted(fingerprints)
        to_embed = chunks
        create_collection(recreate=True)

    texts = [c["text"] for c in to_embed]
    t0 = time.perf_counter()
    dense_vecs = list(dense_model().embed(texts, batch_size=batch_size)) if texts else []
    t_dense = time.perf_counter() - t0

    t0 = time.perf_counter()
    sparse_vecs = list(bm25_model().embed(texts, batch_size=batch_size)) if texts else []
    t_sparse = time.perf_counter() - t0

    if dense_vecs and len(dense_vecs[0]) != config.DENSE_DIM:
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
            payload={**c, FINGERPRINT_KEY: fingerprints[c["source_file"]]},
        )
        for c, d, s in zip(to_embed, dense_vecs, sparse_vecs, strict=True)
    ]

    t0 = time.perf_counter()
    if points:
        client.upsert(config.QDRANT_COLLECTION, points=points, wait=True)
    t_upsert = time.perf_counter() - t0

    stored_points = client.count(config.QDRANT_COLLECTION, exact=True).count
    return {
        "collection": config.QDRANT_COLLECTION,
        "mode": mode,
        "points": stored_points,
        "documents": len(fingerprints),
        "chunks_embedded": len(to_embed),
        "files": {k: v if k != "unchanged" else len(v) for k, v in files.items()},
        "dense_model": config.DENSE_MODEL,
        "sparse_model": config.SPARSE_MODEL,
        "avg_bm25_terms": (round(sum(len(s.indices) for s in sparse_vecs) / len(sparse_vecs), 1)
                           if sparse_vecs else None),
        "seconds": {
            "dense_embed": round(t_dense, 2),
            "bm25_embed": round(t_sparse, 2),
            "upsert": round(t_upsert, 2),
        },
    }
