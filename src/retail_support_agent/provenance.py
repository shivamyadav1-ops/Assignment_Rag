"""Source provenance: one registry (data/sources.json) for every raw input file.

The registry records where each file came from (synthetic or public), its licence,
generator or download URL, and for public files the pinned version and SHA-256.
The chunker copies `synthetic` and `origin` onto every chunk from here, so lineage
travels with the evidence instead of being hardcoded.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from retail_support_agent.config import DATA_DIR, RAW_DIR

REGISTRY_PATH = DATA_DIR / "sources.json"


class UnregisteredSourceError(LookupError):
    """A raw file has no entry in data/sources.json."""


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("sources", {})


def registry(path: Path | None = None) -> dict[str, dict]:
    return _load(str(path or REGISTRY_PATH))


def lookup(file_path: Path, raw_dir: Path | None = None,
           registry_path: Path | None = None) -> dict:
    """Provenance entry for a file under data/raw/.

    Files outside data/raw/ (test fixtures in temp directories) are treated as
    synthetic. A file inside data/raw/ without an entry is an error: adding a
    source must come with its provenance.
    """
    raw_dir = (raw_dir or RAW_DIR).resolve()
    try:
        rel = str(file_path.resolve().relative_to(raw_dir))
    except ValueError:
        return {"origin": "synthetic", "synthetic": True}
    entry = registry(registry_path).get(rel)
    if entry is None:
        raise UnregisteredSourceError(
            f"{rel} has no provenance entry in {REGISTRY_PATH.name}; add one "
            "(public files: scripts/fetch_public_sources.py)")
    return entry
