"""Small local models used as guardrails and metrics — no LLM call, CPU-only.

  NLI         cross-encoder/nli-deberta-v3-base     entailment / neutral / contradiction.
              The mechanism behind faithfulness, contradiction and completeness.
  Toxicity    unitary/toxic-bert                     multi-label (toxic, severe_toxic,
              obscene, threat, insult, identity_hate). Catches obfuscated abuse the
              regex blocklist misses.
  Injection   protectai/deberta-v3-base-prompt-injection-v2   SAFE / INJECTION.
              A one-shot encoder classifier instead of an LLM judge: same yes/no for
              a fraction of the cost and latency.
  Perplexity  distilgpt2                             exp(-mean log P) — fluency only.
  NSFW image  Falconsai/nsfw_image_detection         normal / nsfw, for uploaded photos.
              (toxic-bert has no sexual-content label and NLI zero-shot proved too weak
              for text NSFW, so NSFW is covered on images, where it matters.)

Models load lazily on first use and are cached under .cache/hf. Set
RSA_ML_GUARDRAILS=0 to disable the classifier guardrails (regex layers still run).
"""

from __future__ import annotations

import math
import os
import re
import threading
from functools import cache

from . import config

HF_CACHE = config.PROJECT_ROOT / ".cache" / "hf"

NLI_MODEL = os.getenv("RSA_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
TOXICITY_MODEL = os.getenv("RSA_TOXICITY_MODEL", "unitary/toxic-bert")
INJECTION_MODEL = os.getenv("RSA_INJECTION_MODEL",
                            "protectai/deberta-v3-base-prompt-injection-v2")
PERPLEXITY_MODEL = os.getenv("RSA_PPL_MODEL", "distilgpt2")
NSFW_IMAGE_MODEL = os.getenv("RSA_NSFW_MODEL", "Falconsai/nsfw_image_detection")

TOXICITY_THRESHOLD = 0.5      # any harmful label above this blocks
NSFW_THRESHOLD = 0.5          # P(nsfw) for images
# Measured on 67 benign eval questions: 0.9 gave 2 false positives ("What instructions
# come with the warranty claim?" = 0.993); 0.999 leaves 1, a cross-account probe.
INJECTION_THRESHOLD = 0.999

_lock = threading.Lock()


def ml_guardrails_enabled() -> bool:
    return os.getenv("RSA_ML_GUARDRAILS", "1").strip().lower() not in ("0", "false", "no")


def _load(name: str, kind: str):
    if kind == "image":
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        proc = AutoImageProcessor.from_pretrained(name, cache_dir=HF_CACHE)
        model = AutoModelForImageClassification.from_pretrained(name, cache_dir=HF_CACHE)
        model.eval()
        return proc, model
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    tok = AutoTokenizer.from_pretrained(name, cache_dir=HF_CACHE)
    cls = AutoModelForCausalLM if kind == "causal" else AutoModelForSequenceClassification
    model = cls.from_pretrained(name, cache_dir=HF_CACHE)
    model.eval()
    return tok, model


@cache
def _model(name: str, kind: str = "seq"):
    with _lock:
        return _load(name, kind)


def warmup() -> None:
    """Download + load every model up front (first call is otherwise slow)."""
    for name in (NLI_MODEL, TOXICITY_MODEL, INJECTION_MODEL):
        _model(name)
    _model(PERPLEXITY_MODEL, "causal")


# ── NLI ───────────────────────────────────────────────────────────────────────

def nli(pairs: list[tuple[str, str]], batch_size: int = 16) -> list[dict[str, float]]:
    """Probabilities {entailment, neutral, contradiction} for (premise, hypothesis) pairs."""
    if not pairs:
        return []
    import torch

    tok, model = _model(NLI_MODEL)
    id2label = {i: lbl.lower() for i, lbl in model.config.id2label.items()}
    out: list[dict[str, float]] = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        enc = tok([p for p, _ in chunk], [h for _, h in chunk], truncation="only_first",
                  max_length=512, padding=True, return_tensors="pt")
        with torch.inference_mode():
            probs = model(**enc).logits.softmax(-1).tolist()
        out.extend({id2label[j]: p for j, p in enumerate(row)} for row in probs)
    return out


# ── Toxicity / NSFW ───────────────────────────────────────────────────────────

def toxicity(texts: list[str]) -> list[dict[str, float]]:
    """Per-label sigmoid scores for each text."""
    if not texts:
        return []
    import torch

    tok, model = _model(TOXICITY_MODEL)
    enc = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
    with torch.inference_mode():
        probs = model(**enc).logits.sigmoid().tolist()
    labels = [model.config.id2label[i].lower() for i in range(len(probs[0]))]
    return [dict(zip(labels, row, strict=True)) for row in probs]


HARMFUL_LABELS = ("toxic", "severe_toxic", "threat", "insult", "identity_hate")


# ── Prompt injection ──────────────────────────────────────────────────────────

def injection_score(text: str) -> float:
    """P(INJECTION) from the encoder classifier."""
    import torch

    tok, model = _model(INJECTION_MODEL)
    enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.inference_mode():
        probs = model(**enc).logits.softmax(-1)[0].tolist()
    labels = {model.config.id2label[i].upper(): p for i, p in enumerate(probs)}
    return labels.get("INJECTION", 0.0)


# ── Perplexity ────────────────────────────────────────────────────────────────

def perplexity(text: str, stride: int = 512) -> float | None:
    """exp(-mean token log-likelihood) under distilgpt2. Fluency, not correctness."""
    text = (text or "").strip()
    if not text:
        return None
    import torch

    tok, model = _model(PERPLEXITY_MODEL, "causal")
    ids = tok(text, return_tensors="pt").input_ids[:, :1024]
    if ids.shape[1] < 2:
        return None
    with torch.inference_mode():
        loss = model(ids, labels=ids).loss.item()
    return math.exp(loss)


# ── NSFW images ───────────────────────────────────────────────────────────────

def nsfw_image_score(image) -> float:
    """P(nsfw) for a PIL image, raw bytes, or a file path."""
    import io

    import torch
    from PIL import Image

    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    elif not isinstance(image, Image.Image):
        image = Image.open(image)
    proc, model = _model(NSFW_IMAGE_MODEL, "image")
    with torch.inference_mode():
        probs = model(**proc(images=image.convert("RGB"), return_tensors="pt")
                      ).logits.softmax(-1)[0].tolist()
    labels = {model.config.id2label[i].lower(): p for i, p in enumerate(probs)}
    return labels.get("nsfw", 0.0)


# ── Claim splitting (shared by grounding metrics) ─────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NON_CLAIM = re.compile(
    r"^(would you like|do you want|let me know|please let|if you|i can help|i'm happy|"
    r"is there anything|feel free|hope this|thank)", re.IGNORECASE)


# Inline citations are provenance, not claims: "(Source: return_policy.pdf)",
# "Sources: a.md, b.html", "(chunk_id: af0a…)", backticked file names.
_CITATION = re.compile(
    r"\(\s*(?:sources?|chunk[_ ]id|see)\s*:[^)]*\)"
    r"|\b(?:sources?|chunk[_ ]id)\s*:\s*[^\n]*"
    r"|`?[\w\-]+\.(?:pdf|md|html?)`?(?:\s*\([^)]*\))?",
    re.IGNORECASE)


_DANGLING = re.compile(r",?\s*(?:as (?:stated|noted|described) in|according to|per)"
                       r"(?:\s+the)?\s*(?=[.,;:]|$)", re.IGNORECASE)


def strip_citations(text: str) -> str:
    t = _DANGLING.sub("", _CITATION.sub("", text or ""))
    t = re.sub(r"[ \t]+([.,;:])", r"\1", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def split_claims(text: str, min_words: int = 4) -> list[str]:
    """Answer → candidate factual sentences (drops questions, offers, headings and
    inline source citations)."""
    claims = []
    for raw in _SENT_SPLIT.split(strip_citations(text)):
        s = re.sub(r"[*_`#>]+", "", raw).strip(" -•\t")
        if (len(s.split()) < min_words or s.endswith("?") or s.endswith(":")
                or _NON_CLAIM.match(s)):
            continue
        claims.append(s)
    return claims
