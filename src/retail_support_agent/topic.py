"""Off-topic guardrail: refuse requests outside ShopSmart support before the agent runs.

Without it, "who is the prime minister of India" costs an LLM call (and often a
retrieval + rerank) before the model declines. This classifier runs at the input gate
in a few milliseconds, with no LLM call:

  1. Requests that mention an order / return / customer ID are always in scope.
  2. Otherwise the query is embedded with the retrieval model (bge-small, already
     loaded) and compared with two prototype sets: in-scope support intents and
     common off-topic requests. score = mean(top-3 off-topic sims) - mean(top-3
     in-scope sims). Above OFF_TOPIC_MARGIN -> blocked.

The prototypes are deliberately generic and disjoint from every eval/test question, so
the calibration in `topic_eval.py` measures generalisation, not memorisation. The
margin was chosen so that no in-scope question in the labelled sets is blocked:
unanswerable but on-topic questions ("student discount on laptops?") must still reach
the agent, which escalates them.
"""

from __future__ import annotations

import os
import re
import time
from functools import cache

import numpy as np

IN_SCOPE = [
    # orders
    "where is my order", "track my package", "my order hasn't arrived yet",
    "cancel my order", "change the delivery address on my order",
    "what does my order status mean", "I received the wrong item",
    # returns / refunds / exchanges
    "how do I return an item", "is this product returnable",
    "how many days do I have to send something back", "the item arrived damaged",
    "can I exchange it for a different size or colour", "when will I get my refund",
    "refund to my original payment method or store credit", "restocking fee",
    # shipping / delivery
    "how much is shipping", "how long does delivery take", "do you ship internationally",
    "express or next day delivery options", "holiday shipping deadlines",
    "import duties and customs fees on my package", "package marked delivered but missing",
    # products
    "which category is this product in", "is this laptop compatible with my monitor",
    "recommend headphones for travel", "product specifications and features",
    "is this item in stock", "difference between two phone models",
    # warranty / repair / troubleshooting
    "my device stopped working", "warranty coverage for my tv", "how do I make a warranty claim",
    "repair service for a cracked screen", "my laptop won't turn on",
    "protection plan or extended warranty",
    # pricing / payments / promotions
    "do you price match other stores", "apply a coupon or promo code", "gift card balance",
    "student or military discount", "my card was charged twice", "payment declined at checkout",
    "billing question about an invoice",
    # account / membership / services
    "reset my password", "update my account email", "how is my personal data used",
    "premium membership benefits", "cancel my membership", "in-store pickup",
    "installation or setup service", "trade in my old phone",
    # conversation
    "hello", "thanks for your help", "can I talk to a human agent",
    "what can you help me with",
]

OFF_TOPIC = [
    "who is the president of a country", "which party won the election",
    "what is the capital city of a country", "history of the roman empire",
    "explain the theory of relativity", "how far is the moon from earth",
    "solve this algebra equation", "what is 345 divided by 15",
    "write a python script", "fix the bug in my javascript code",
    "write an essay for my school assignment", "write a poem about the sea",
    "tell me a funny joke", "give me a recipe for pasta", "how to cook rice",
    "what will the weather be tomorrow", "who won the football match last night",
    "which celebrity is dating whom", "recommend a good movie to watch",
    "song lyrics of a popular band", "what are the symptoms of the flu",
    "what medicine should I take for a headache", "how do I file for divorce",
    "should I invest in bitcoin", "which stocks will go up", "plan a trip to europe",
    "best tourist places to visit", "relationship advice for my partner",
    "translate this sentence into german", "what is the meaning of life",
    "latest news headlines today", "does god exist", "how to lose weight quickly",
    "what is artificial intelligence and how does chatgpt work",
    "who painted the mona lisa", "how many players are on a cricket team",
    "teach me to play guitar", "what's your favourite colour",
    "write a cover letter for a job", "explain how vaccines work",
]

ID_RE = re.compile(r"\b(?:ORD|RET|CUST)-\d{3,}\b", re.IGNORECASE)
OFF_TOPIC_MARGIN = float(os.getenv("RSA_OFF_TOPIC_MARGIN", "0.04"))
TOP_N = 3


@cache
def _embedder():
    from fastembed import TextEmbedding

    from . import config
    return TextEmbedding(config.DENSE_MODEL, cache_dir=str(config.MODEL_CACHE_DIR))


def _embed(texts: list[str]) -> np.ndarray:
    vecs = np.array(list(_embedder().embed(texts)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@cache
def _prototypes() -> tuple[np.ndarray, np.ndarray]:
    return _embed(IN_SCOPE), _embed(OFF_TOPIC)


def topic_score(text: str) -> dict:
    """Return {score, in_sim, off_sim, nearest_in, nearest_off}; score > margin = off-topic."""
    ins, offs = _prototypes()
    q = _embed([text])[0]
    si, so = ins @ q, offs @ q
    in_sim = float(np.sort(si)[-TOP_N:].mean())
    off_sim = float(np.sort(so)[-TOP_N:].mean())
    return {"score": off_sim - in_sim, "in_sim": in_sim, "off_sim": off_sim,
            "nearest_in": IN_SCOPE[int(si.argmax())],
            "nearest_off": OFF_TOPIC[int(so.argmax())]}


def is_off_topic(text: str, margin: float | None = None) -> tuple[bool, dict]:
    t0 = time.perf_counter()
    if ID_RE.search(text or ""):
        return False, {"reason": "mentions an order/return/customer id",
                       "latency_ms": (time.perf_counter() - t0) * 1000}
    s = topic_score(text)
    s["latency_ms"] = (time.perf_counter() - t0) * 1000
    return s["score"] > (OFF_TOPIC_MARGIN if margin is None else margin), s
