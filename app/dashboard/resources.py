"""Cached resources (retriever, LLM, MCP, judge, classifiers), cached data and reports."""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from retail_support_agent import classifiers, config
from retail_support_agent.eval_answers import CASES_PATH
from retail_support_agent.metrics import GenerationJudge
from retail_support_agent.security_eval import run_image_suite, run_security_suite

REPORTS_DIR = config.PROJECT_ROOT / "reports"


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading retriever (embeddings + reranker)…")
def get_retriever():
    from retail_support_agent.retriever import HybridRetriever
    return HybridRetriever()


@st.cache_resource(show_spinner="Connecting to LLM…")
def get_llm():
    from retail_support_agent.llm import build_llm
    return build_llm()


@st.cache_resource(show_spinner="Starting MCP tool server…")
def get_mcp(customer: str):
    from retail_support_agent.mcp_client import MCPToolClient
    return MCPToolClient(customer).start()


@st.cache_resource
def get_judge():
    return GenerationJudge()


@st.cache_resource(show_spinner="Loading local classifier models (first run downloads)…")
def warm_classifiers() -> bool:
    try:
        classifiers.warmup()
        return True
    except Exception:  # noqa: BLE001
        return False


@st.cache_data
def load_customers() -> pd.DataFrame:
    return pd.read_csv(config.OPERATIONAL_DIR / "customers.csv")


@st.cache_data
def load_orders() -> pd.DataFrame:
    return pd.read_csv(config.OPERATIONAL_DIR / "orders.csv")


@st.cache_data
def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


@st.cache_data(show_spinner="Running red-team suite…")
def cached_security(use_ml: bool) -> list[dict]:
    return run_security_suite(use_ml=use_ml)


@st.cache_data(show_spinner="Running image guardrail checks…")
def cached_image_suite() -> list[dict]:
    return run_image_suite()


def judge_available() -> bool:
    return bool(os.getenv("NVIDIA_API_KEY"))


# ── Saved reports (reports/*.json) ────────────────────────────────────────────

def find_reports(pattern: str) -> list:
    """Report files matching `pattern`, newest first."""
    return sorted(REPORTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime,
                  reverse=True) if REPORTS_DIR.exists() else []


def load_report(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
