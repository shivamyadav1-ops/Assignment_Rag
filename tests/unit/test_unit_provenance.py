"""Unit tests for the public-source path: eCFR XML parsing and the provenance registry."""

from __future__ import annotations

import json

import pytest

from retail_support_agent import chunker as ch
from retail_support_agent import provenance as prov
from retail_support_agent.config import DATA_DIR, RAW_DIR

ECFR_SAMPLE = """<?xml version="1.0"?>
<DIV5 N="435" TYPE="PART">
<HEAD>PART 435&#x2014;MAIL, INTERNET, OR TELEPHONE ORDER MERCHANDISE</HEAD>
<AUTH><HED>Authority:</HED><PSPACE>15 U.S.C. 57a.</PSPACE></AUTH>
<DIV8 N="435.1" TYPE="SECTION">
<HEAD>&#xA7; 435.1 Definitions.</HEAD>
<P>(a) <I>Prompt refund</I> shall mean a refund sent within seven (7) working days.</P>
<P>(b)</P>
</DIV8>
<DIV8 N="435.2" TYPE="SECTION">
<HEAD>&#xA7; 435.2 Mail, Internet, or telephone order sales.</HEAD>
<P>The seller must ship within thirty (30) days when no time is stated.</P>
</DIV8>
</DIV5>"""


def test_parse_ecfr_xml_sections_become_level2_headings():
    blocks = ch.parse_ecfr_xml(ECFR_SAMPLE)
    headings = [(b.text, b.level) for b in blocks if b.block_type == "heading"]
    assert headings[0] == ("PART 435—MAIL, INTERNET, OR TELEPHONE ORDER MERCHANDISE", 1)
    assert ("§ 435.1 Definitions.", 2) in headings
    assert ("§ 435.2 Mail, Internet, or telephone order sales.", 2) in headings


def test_parse_ecfr_xml_flattens_inline_markup_and_keeps_section_heading():
    paras = [b for b in ch.parse_ecfr_xml(ECFR_SAMPLE) if b.block_type == "paragraph"]
    refund = next(b for b in paras if "Prompt refund" in b.text)
    assert "(a) Prompt refund shall mean" in refund.text  # <I> flattened with spaces kept
    assert refund.heading == "§ 435.1 Definitions."
    assert all(b.text != "(b)" for b in paras)  # fragments under 10 chars dropped
    assert "Authority: 15 U.S.C. 57a." in paras[0].text


def test_parse_html_keeps_spaces_around_inline_tags():
    html = ("<main><p>Cancel within <strong>1 hour</strong> of placement "
            "(see <a href='#'>Order History</a>).</p>"
            "<ul><li><b>Confirmed:</b> payment authorized.</li></ul>"
            "<h2>Returns <em>and</em> refunds</h2></main>")
    blocks = ch.parse_html(html)
    texts = [b.text for b in blocks]
    assert "Cancel within 1 hour of placement (see Order History)." in texts
    assert "- Confirmed: payment authorized." in texts
    assert "Returns and refunds" in texts


def test_parse_ecfr_xml_without_part_returns_nothing():
    assert ch.parse_ecfr_xml("<?xml version='1.0'?><ROOT/>") == []


def test_chunk_document_xml_uses_part_title_and_section_chunks(tmp_path):
    # Pad each section past MIN_CHUNK_SIZE so it stands as its own chunk.
    filler = " The seller shall keep records of compliance." * 8
    xml = ECFR_SAMPLE.replace("working days.</P>", "working days." + filler + "</P>") \
                     .replace("is stated.</P>", "is stated." + filler + "</P>")
    f = tmp_path / "rule.xml"
    f.write_text(xml, encoding="utf-8")
    chunks = ch.chunk_document(f)
    assert chunks and all(c.source_type == "xml" for c in chunks)
    # H2 boundaries flush chunks, so no chunk mixes § 435.1 and § 435.2 text.
    assert not any("Prompt refund" in c.text and "thirty (30) days" in c.text for c in chunks)


def test_lookup_outside_raw_dir_defaults_to_synthetic(tmp_path):
    entry = prov.lookup(tmp_path / "x.md")
    assert entry == {"origin": "synthetic", "synthetic": True}


def test_lookup_unregistered_file_inside_raw_dir_fails(tmp_path):
    raw = tmp_path / "raw"
    (raw / "policies").mkdir(parents=True)
    f = raw / "policies" / "new.md"
    f.write_text("x")
    reg = tmp_path / "sources.json"
    reg.write_text(json.dumps({"sources": {}}))
    with pytest.raises(prov.UnregisteredSourceError):
        prov.lookup(f, raw_dir=raw, registry_path=reg)


def test_real_registry_covers_every_raw_file_and_labels_correctly():
    registry = prov.registry(DATA_DIR / "sources.json")
    files = {str(p.relative_to(RAW_DIR)) for p in RAW_DIR.rglob("*") if p.is_file()}
    assert files <= set(registry), f"unregistered: {files - set(registry)}"
    for rel, entry in registry.items():
        assert entry["origin"] in ("public", "synthetic")
        assert entry["synthetic"] is (entry["origin"] == "synthetic")
        assert entry.get("license"), rel
        if entry["origin"] == "public":
            assert entry["url"].startswith("https://") and len(entry["sha256"]) == 64


def test_public_file_on_disk_matches_registered_hash():
    import hashlib
    for rel, entry in prov.registry(DATA_DIR / "sources.json").items():
        if entry["origin"] == "public":
            data = (RAW_DIR / rel).read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"], rel


def test_every_indexed_chunk_carries_origin():
    chunks_file = DATA_DIR / "processed" / "chunks.jsonl"
    if not chunks_file.exists():
        pytest.skip("run `rsa ingest` first")
    rows = [json.loads(line) for line in chunks_file.open(encoding="utf-8")]
    origins = {r["source_file"]: r.get("origin") for r in rows}
    assert origins["ftc_mail_order_rule_16cfr435.xml"] == "public"
    assert all(o in ("public", "synthetic") for o in origins.values())
    assert all(r["synthetic"] is (r["origin"] == "synthetic") for r in rows)


def test_manifest_counts_match_disk():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    chunks_file = DATA_DIR / "processed" / "chunks.jsonl"
    if "totals" not in manifest or not chunks_file.exists():
        pytest.skip("regenerate with scripts/generate_manifest.py")
    n_chunks = sum(1 for _ in chunks_file.open(encoding="utf-8"))
    assert manifest["totals"]["chunks"] == n_chunks
    for s in manifest["sources"]:
        if "rows" in s and s["format"] == "csv":
            lines = (RAW_DIR / s["file"]).read_text(encoding="utf-8").strip().splitlines()
            assert s["rows"] == len(lines) - 1


@pytest.fixture(autouse=True)
def _fresh_registry_cache():
    prov._load.cache_clear()
    yield
    prov._load.cache_clear()
