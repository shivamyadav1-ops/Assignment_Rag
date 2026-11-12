"""🔎 Retrieval & index tab: ranking metrics from retrieval eval reports + vector index
stats and search-latency benchmark."""

from __future__ import annotations

import json
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from retail_support_agent import config

from .resources import REPORTS_DIR, get_retriever, load_cases
from .sidebar import Settings


def index_stats() -> dict:
    r = get_retriever()
    info = r.client.get_collection(r.collection)
    dense = info.config.params.vectors
    sparse = info.config.params.sparse_vectors or {}
    size = sum(f.stat().st_size for f in config.QDRANT_PATH.rglob("*") if f.is_file())
    return {"points": r.client.count(r.collection).count,
            "dense": {k: f"{v.size}-d {v.distance}" for k, v in dense.items()}
            if isinstance(dense, dict) else str(dense),
            "sparse": {k: str(v.modifier) for k, v in sparse.items()},
            "disk_mb": size / 1e6}


def index_latency(queries: list[str], repeats: int = 3) -> pd.DataFrame:
    r = get_retriever()
    rows = []
    for q in queries:
        for _ in range(repeats):
            for name, fn in (("dense embed", lambda q=q: r._dense_query(q)),
                             ("bm25 encode", lambda q=q: r._bm25_query(q)),
                             ("dense search", lambda q=q: r.dense_search(q)),
                             ("bm25 search", lambda q=q: r.bm25_search(q)),
                             ("hybrid (RRF)", lambda q=q: r.hybrid_search(q))):
                t0 = time.perf_counter()
                fn()
                rows.append({"stage": name, "ms": (time.perf_counter() - t0) * 1000})
    return pd.DataFrame(rows)


def render(settings: Settings) -> None:
    st.subheader("Retrieval quality")
    files = sorted(REPORTS_DIR.glob("retrieval_eval_*.json"), reverse=True) \
        if REPORTS_DIR.exists() else []
    if not files:
        st.info("No retrieval reports. Run `rsa eval-retrieval` to create one.")
    else:
        f = st.selectbox("Report", files, format_func=lambda p: p.name)
        data = json.loads(f.read_text(encoding="utf-8"))
        st.caption(json.dumps(data.get("config", {})))
        summ = pd.DataFrame({m: v["summary"] for m, v in data["modes"].items()}).T
        summ.index.name = "mode"
        metric_cols = [c for c in summ.columns if c not in ("n", "evidence_n")
                       and not c.startswith("latency")]
        st.dataframe(summ.style.format("{:.3f}", subset=metric_cols, na_rep="–"),
                     width="stretch")
        long = summ[metric_cols].reset_index().melt(id_vars="mode", var_name="metric")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(long, x="metric", y="value", color="mode", barmode="group",
                               title="Ranking metrics by mode", range_y=[0, 1.05]),
                        width="stretch")
        lat_cols = [c for c in summ.columns if c.startswith("latency")]
        c2.plotly_chart(px.bar(summ[lat_cols].reset_index().melt(id_vars="mode"),
                               x="mode", y="value", color="variable", barmode="group",
                               title="Retrieval latency (ms)"), width="stretch")

    st.markdown("##### Vector index")
    if st.button("Inspect index and benchmark search latency"):
        stats = index_stats()
        c = st.columns(4)
        c[0].metric("Indexed points", stats["points"])
        c[1].metric("Dense vectors", ", ".join(f"{k}: {v}" for k, v in stats["dense"].items())
                    if isinstance(stats["dense"], dict) else stats["dense"])
        c[2].metric("Sparse vectors", ", ".join(f"{k} ({v})" for k, v in stats["sparse"].items()))
        c[3].metric("On disk", f"{stats['disk_mb']:.1f} MB")
        qs = [c["question"] for c in load_cases()[:6]]
        lat = index_latency(qs)
        agg = lat.groupby("stage")["ms"].describe(percentiles=[0.5, 0.95])[
            ["count", "mean", "50%", "95%", "max"]]
        st.dataframe(agg.style.format("{:.1f}"), width="stretch")
        st.plotly_chart(px.box(lat, x="stage", y="ms", points="all",
                               title="Index-stage latency (ms) — reranking excluded"),
                        width="stretch")
