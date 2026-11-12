"""Display formatting helpers shared by every page."""

from __future__ import annotations

import pandas as pd


def fmt_ms(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"


def fmt_pct(v) -> str:
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.0%}"


def fmt_num(v, d=3) -> str:
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.{d}f}"


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and pd.isna(x))]
    return sum(xs) / len(xs) if xs else None
