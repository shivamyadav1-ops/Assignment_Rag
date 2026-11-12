"""Lazily-loaded FastEmbed models and the Qdrant client, one instance per process."""

import atexit
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient

from . import config


@lru_cache(maxsize=1)
def dense_model() -> TextEmbedding:
    return TextEmbedding(config.DENSE_MODEL, cache_dir=str(config.MODEL_CACHE_DIR),
                         threads=config.MODEL_THREADS)


@lru_cache(maxsize=1)
def bm25_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(config.SPARSE_MODEL, cache_dir=str(config.MODEL_CACHE_DIR),
                               threads=config.MODEL_THREADS)


@lru_cache(maxsize=1)
def reranker_model() -> TextCrossEncoder:
    return TextCrossEncoder(config.RERANKER_MODEL, cache_dir=str(config.MODEL_CACHE_DIR),
                            threads=config.MODEL_THREADS)


@lru_cache(maxsize=1)
def qdrant_client() -> QdrantClient:
    # Embedded (on-disk) mode holds a file lock, so only one client may exist per process.
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(config.QDRANT_PATH))
    # Close before interpreter teardown; the client's own __del__ fails once imports are gone.
    atexit.register(client.close)
    return client
