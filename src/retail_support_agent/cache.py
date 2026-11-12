"""Caching layers: retrieval (exact) and answers (semantic).

Context / prompt caching (provider-side KV-prefix reuse) lives in `llm.py`; this module
holds the two application-side caches.

RetrievalCache — exact LRU over `HybridRetriever.retrieve(query, mode, top_k)`.
  The cross-encoder rerank dominates retrieval latency, and the agent often repeats a
  search (forced search, retries, the same policy question from many customers).
  In-process only: results are cheap to recompute after a restart and an in-memory
  map cannot go stale across a re-index (a new process re-reads the index).

SemanticCache — reuse a previous *final answer* for a question that means the same.
  A wrong hit serves a wrong answer with full confidence, so the design is
  conservative:
  * Only customer-independent answers are stored (`is_cacheable`): the agent called
    nothing but `search_policies`, the outcome is `answered`, no guardrail fired, the
    live NLI check found no contradiction, and neither question nor answer carries an
    order, customer or return ID. The agent's prompt holds no customer data, so such
    an answer is a function of the question text alone.
  * A hit needs BOTH cosine similarity >= threshold (BGE-small query embeddings) AND an
    identical "discriminator" signature: numbers, negations, and every rare policy
    term (a corpus word in <= DISCRIMINATIVE_DF of chunks: countries, categories,
    carriers, product types...) must match. "express shipping to Canada" can never be
    served "...to Mexico", nor "15 days" served "30 days", however close the vectors.
  * Every entry is keyed by a fingerprint of everything an answer depends on (chunk
    corpus, system prompt, agent model, reranker, retrieval config, as-of date); a
    change to any of them makes old entries unreachable. Entries also expire (TTL).
  * Input guardrails and rate limiting run before the lookup, and the output guardrail
    re-runs on a cached answer for the current requester (agent.py).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config

# ── Fingerprint ───────────────────────────────────────────────────────────────

def _file_digest(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def answer_fingerprint(system_prompt: str, model: str) -> str:
    """Everything a policy-only answer depends on. Change any part -> cache miss."""
    parts = {
        "chunks": _file_digest(config.CHUNKS_PATH),
        "system_prompt": hashlib.sha256(system_prompt.encode()).hexdigest()[:16],
        "model": model,
        "dense": config.DENSE_MODEL, "reranker": config.RERANKER_MODEL,
        "rerank_top_k": config.RERANK_TOP_K, "rrf_limit": config.RRF_LIMIT,
        "max_rerank": config.MAX_RERANK_CANDIDATES,
        "grounding_min": config.GROUNDING_MIN_SCORE,
        "as_of": config.AS_OF_DATE.isoformat(),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:20]


# ── Retrieval cache ───────────────────────────────────────────────────────────

class RetrievalCache:
    """Thread-safe LRU of RetrievalResult objects (stored and returned as deep copies,
    so a caller mutating chunks cannot poison the cache)."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(query: str, mode: str, top_k: int) -> tuple:
        return (" ".join(query.split()).lower(), mode, top_k)

    def get(self, key: tuple):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return copy.deepcopy(self._data[key])
            self.misses += 1
            return None

    def put(self, key: tuple, value) -> None:
        if self.maxsize <= 0:
            return
        with self._lock:
            self._data[key] = copy.deepcopy(value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            n = self.hits + self.misses
            return {"size": len(self._data), "maxsize": self.maxsize, "hits": self.hits,
                    "misses": self.misses, "hit_rate": round(self.hits / n, 3) if n else None}


# ── Semantic answer cache: text normalisation and discriminators ──────────────

_WORD = re.compile(r"[a-z][a-z'\-]*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_ID = re.compile(r"\b(?:ORD|CUST|RET|ESC)-[A-Z0-9]+\b", re.IGNORECASE)
NEGATIONS = frozenset({"not", "no", "never", "without", "non", "cannot", "can't", "don't",
                       "doesn't", "isn't", "won't", "unopened", "except"})
STOPWORDS = frozenset("""a an the and or but if of to in on at by for from with about as
into over under is are was were be been being do does did have has had i me my mine we
our you your it its this that these those there here what which who whom whose when
where why how can could should would will shall may might must just also so than then
too very any some all each per please hi hello hey thanks thank there's what's i'm it's
get got want wondering know tell let does doe doing done during while much many
many long need needs like able would like kindly still yet ever""".split())


def _stem(w: str) -> str:
    w = w.strip("'-")
    if w in STOPWORDS:
        return w
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us")):
        return w[:-1]
    return w


def _words(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower())]


def normalize(text: str) -> str:
    return " ".join(_words(text) + _NUMBER.findall(text))


@lru_cache(maxsize=1)
def _discriminative_vocab() -> frozenset:
    """Corpus words rare enough to change what a question is about.

    A word in at most DISCRIMINATIVE_DF of chunks (countries, categories, carriers,
    product types, 'unopened'...) must match between the new and cached question. Common
    policy words ('return', 'window', 'policy', 'day') appear everywhere and may differ.
    """
    if not config.CHUNKS_PATH.exists():
        return frozenset()
    df: Counter = Counter()
    n = 0
    with config.CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
                df.update(set(_words(json.loads(line)["text"])))
    limit = max(1, math.floor(config.SEMANTIC_CACHE_DISCRIMINATIVE_DF * n))
    return frozenset(w for w, c in df.items()
                     if c <= limit and w not in STOPWORDS and len(w) > 2)


def signature(text: str, vocab: frozenset | None = None) -> dict:
    vocab = _discriminative_vocab() if vocab is None else vocab
    words = set(_words(text))
    return {"numbers": sorted(set(_NUMBER.findall(text))),
            "negations": sorted(words & NEGATIONS),
            "terms": sorted((words & vocab) - STOPWORDS)}


def is_cacheable(question: str, result) -> tuple[bool, str]:
    """Only answers that depend on the question text alone may be shared."""
    if result.outcome != "answered":
        return False, f"outcome={result.outcome}"
    # Customer-independent tools only: policy search and the pure deadline calculator.
    if ("search_policies" not in result.tool_calls
            or not set(result.tool_calls) <= {"search_policies", "calculate_return_deadline"}):
        return False, f"tools={sorted(set(result.tool_calls))}"
    if not result.evidence:
        return False, "no grounded evidence"
    if any(s.kind == "guardrail" for s in result.steps):
        return False, "a guardrail fired"
    g = result.grounding or {}
    if (g.get("contradiction_source") or 0) > 0 or (g.get("self_contradiction") or 0) > 0:
        return False, "NLI contradiction"
    if _ID.search(question) or _ID.search(result.answer):
        return False, "contains an order/customer/return ID"
    if not result.answer.strip():
        return False, "empty answer"
    return True, "ok"


# ── Semantic answer cache ─────────────────────────────────────────────────────

def _default_embedder(text: str) -> np.ndarray:
    from .models import dense_model

    return np.asarray(next(iter(dense_model().query_embed(text))), dtype=np.float32)


class SemanticCache:
    """SQLite-backed answer cache with an in-memory embedding matrix per fingerprint."""

    def __init__(self, path: Path | None = None, threshold: float | None = None,
                 ttl_s: float | None = None,
                 embedder: Callable[[str], np.ndarray] | None = None,
                 vocab: frozenset | None = None, max_entries: int = 5000):
        self.path = Path(path or config.SEMANTIC_CACHE_PATH)
        self.threshold = config.SEMANTIC_CACHE_THRESHOLD if threshold is None else threshold
        self.ttl_s = config.SEMANTIC_CACHE_TTL_S if ttl_s is None else ttl_s
        self._embed = embedder or _default_embedder
        self._vocab = vocab
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("""CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT NOT NULL,
            question TEXT NOT NULL, norm TEXT NOT NULL, signature TEXT NOT NULL,
            embedding BLOB NOT NULL, answer TEXT NOT NULL, outcome TEXT NOT NULL,
            evidence TEXT NOT NULL, context TEXT NOT NULL, grounding TEXT,
            created REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0, last_hit REAL)""")
        self._db.execute("CREATE INDEX IF NOT EXISTS answers_fp ON answers(fingerprint)")
        self._db.commit()
        self._index: dict[str, tuple[list[dict], np.ndarray]] = {}
        self.stats_counter = Counter()

    def _sig(self, text: str) -> dict:
        return signature(text, self._vocab)

    # Matrix of live (non-expired) entries for one fingerprint, built lazily.
    def _load(self, fp: str) -> tuple[list[dict], np.ndarray]:
        cutoff = time.time() - self.ttl_s
        if fp not in self._index:
            rows = self._db.execute(
                "SELECT id, question, norm, signature, embedding, created FROM answers "
                "WHERE fingerprint = ? AND created >= ? ORDER BY id", (fp, cutoff)).fetchall()
            meta = [{"id": r[0], "question": r[1], "norm": r[2],
                     "signature": json.loads(r[3]), "created": r[5]} for r in rows]
            mat = (np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
                   if rows else np.zeros((0, 1), dtype=np.float32))
            self._index[fp] = (meta, mat)
        else:
            meta, mat = self._index[fp]
            valid = [i for i, m in enumerate(meta) if m.get("created", 0.0) >= cutoff]
            if len(valid) != len(meta):
                meta = [meta[i] for i in valid]
                dim = mat.shape[1] if mat.ndim > 1 else 1
                mat = mat[valid] if valid else np.zeros((0, dim), dtype=np.float32)
                self._index[fp] = (meta, mat)
        return self._index[fp]

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def lookup(self, question: str, fingerprint: str) -> dict | None:
        """Best entry that clears the threshold AND has the same signature, or None."""
        t0 = time.perf_counter()
        norm, sig = normalize(question), self._sig(question)
        with self._lock:
            meta, mat = self._load(fingerprint)
            best, best_sim, reason = None, -1.0, "empty"
            exact = next((m for m in meta if m["norm"] == norm), None)
            if exact is not None:
                best, best_sim, reason = exact, 1.0, "exact"
            elif meta:
                q = self._unit(self._embed(question).astype(np.float32))
                sims = mat @ q
                reason = "below_threshold"
                for i in np.argsort(-sims):
                    s = float(sims[i])
                    if s < self.threshold:
                        break
                    if meta[i]["signature"] == sig:
                        best, best_sim, reason = meta[i], s, "semantic"
                        break
                    reason = "signature_mismatch"
            if best is None:
                self.stats_counter["miss"] += 1
                self.stats_counter[f"miss:{reason}"] += 1
                return None
            row = self._db.execute(
                "SELECT answer, outcome, evidence, context, grounding, question FROM answers "
                "WHERE id = ?", (best["id"],)).fetchone()
            self._db.execute("UPDATE answers SET hits = hits + 1, last_hit = ? WHERE id = ?",
                             (time.time(), best["id"]))
            self._db.commit()
            self.stats_counter["hit"] += 1
            self.stats_counter[f"hit:{reason}"] += 1
        return {"id": best["id"], "similarity": round(best_sim, 4), "match": reason,
                "cached_question": row[5], "answer": row[0], "outcome": row[1],
                "evidence": json.loads(row[2]), "context": json.loads(row[3]),
                "grounding": json.loads(row[4]) if row[4] else None,
                "lookup_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def store(self, question: str, fingerprint: str, result) -> tuple[bool, str]:
        ok, why = is_cacheable(question, result)
        if not ok:
            self.stats_counter["skip"] += 1
            return False, why
        norm, sig = normalize(question), self._sig(question)
        emb = self._unit(self._embed(question).astype(np.float32))
        with self._lock:
            meta, _ = self._load(fingerprint)
            if any(m["norm"] == norm for m in meta):
                return False, "already cached"
            self._db.execute(
                "INSERT INTO answers (fingerprint, question, norm, signature, embedding, "
                "answer, outcome, evidence, context, grounding, created) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (fingerprint, question, norm, json.dumps(sig), emb.tobytes(), result.answer,
                 result.outcome, json.dumps(result.evidence),
                 json.dumps(result.context, default=str),
                 json.dumps(result.grounding, default=str) if result.grounding else None,
                 time.time()))
            # Bound the table: drop the oldest rows beyond max_entries.
            self._db.execute("DELETE FROM answers WHERE id NOT IN "
                             "(SELECT id FROM answers ORDER BY id DESC LIMIT ?)",
                             (self.max_entries,))
            self._db.commit()
            self._index.pop(fingerprint, None)   # rebuilt on next lookup
            self.stats_counter["store"] += 1
        return True, "stored"

    def clear(self) -> int:
        with self._lock:
            n = self._db.execute("DELETE FROM answers").rowcount
            self._db.commit()
            self._index.clear()
        return n

    def stats(self) -> dict:
        with self._lock:
            total, hits = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM answers").fetchone()
            fps = self._db.execute("SELECT COUNT(DISTINCT fingerprint) FROM answers").fetchone()[0]
        return {"path": str(self.path), "entries": total, "lifetime_hits": hits,
                "fingerprints": fps, "threshold": self.threshold, "ttl_s": self.ttl_s,
                "session": dict(self.stats_counter)}


_shared: SemanticCache | None = None
_shared_lock = threading.Lock()


def shared_semantic_cache() -> SemanticCache:
    """One cache per process (the SQLite connection and matrices are shared)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = SemanticCache()
        return _shared
