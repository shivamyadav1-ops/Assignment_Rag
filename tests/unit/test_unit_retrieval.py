"""Unit tests for cluster R: chunker, retriever, indexer, ingest, policies.

No network, no real embedding/LLM calls, no Qdrant on disk: every model and client
is replaced with a deterministic in-memory stub. Pure functions are exercised for
real.
"""

from __future__ import annotations

import json
import types
import uuid
from datetime import date
from pathlib import Path

import pytest

from retail_support_agent import chunker as ch
from retail_support_agent import indexer as ix
from retail_support_agent import ingest as ing
from retail_support_agent import policies as pol
from retail_support_agent import retriever as rt

# ════════════════════════════════════════════════════════════════════════════
# chunker — token estimation
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "text,expected",
    [("", 1), ("a", 1), ("abcd", 1), ("abcdefgh", 2), ("x" * 2048, 512)],
)
def test_estimate_tokens_boundaries(text, expected):
    assert ch.estimate_tokens(text) == expected


def test_estimate_tokens_never_zero_for_unicode():
    # Multi-byte characters count as single characters, not bytes.
    assert ch.estimate_tokens("é" * 8) == 2
    assert ch.estimate_tokens("日本語") == 1


def test_estimate_chars_is_inverse_of_token_scale():
    assert ch.estimate_chars(0) == 0
    assert ch.estimate_chars(64) == 256
    assert ch.estimate_tokens("x" * ch.estimate_chars(10)) == 10


# ════════════════════════════════════════════════════════════════════════════
# chunker — table cell / row helpers
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, ""),
        ("", ""),
        ("  spaced   out \n again ", "spaced out again"),
        ("**bold**", "bold"),
        ("a|b", "a/b"),
        ("Ünïcode  text", "Ünïcode text"),
        ("\t\n", ""),
    ],
)
def test_clean_cell(raw, expected):
    assert ch._clean_cell(raw) == expected


def test_clean_cell_accepts_non_string():
    assert ch._clean_cell(12) == "12"


def test_parse_pipe_rows_drops_separator_and_empty_rows():
    lines = [
        "| Country | Price |",
        "|---------|-------|",
        "| Canada | $19.99 |",
        "|   |   |",
    ]
    assert ch._parse_pipe_rows(lines) == [
        ["Country", "Price"],
        ["Canada", "$19.99"],
    ]


def test_parse_pipe_rows_empty_input():
    assert ch._parse_pipe_rows([]) == []


def test_parse_pipe_rows_keeps_colon_alignment_row_without_dash():
    # The separator filter requires a "-"; a row of only colons is data, not a rule.
    assert ch._parse_pipe_rows(["| : | : |"]) == [[":", ":"]]


def test_rows_to_markdown_pads_ragged_rows():
    md = ch._rows_to_markdown([["A", "B", "C"], ["1"], ["1", "2", "3"]])
    lines = md.split("\n")
    assert lines[0] == "| A | B | C |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| 1 |  |  |"
    assert all(line.count("|") == 4 for line in lines)


def test_rows_to_markdown_single_row_has_header_and_rule_only():
    assert ch._rows_to_markdown([["only"]]) == "| only |\n| --- |"


def test_rows_to_markdown_empty_table_raises():
    with pytest.raises(ValueError):
        ch._rows_to_markdown([])


def test_unique_headers_disambiguates_and_names_blanks():
    assert ch._unique_headers(["A", "A", "", "A"]) == ["A", "A (2)", "Column 3", "A (3)"]


def test_unique_headers_empty():
    assert ch._unique_headers([]) == []


# ════════════════════════════════════════════════════════════════════════════
# chunker — heading detection and sentence splitting
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "line,level,text",
    [
        ("1. Returns Overview", 1, "1 Returns Overview"),
        ("1.3 Holiday Return Policy", 2, "1.3 Holiday Return Policy"),
        ("2.1.4 Damaged Item Handling", 3, "2.1.4 Damaged Item Handling"),
    ],
)
def test_detect_bare_heading_levels(line, level, text):
    assert ch._detect_bare_heading(line) == (level, text)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "just a sentence.",
        "1. lowercase start",
        "1. Ab",  # title shorter than the 4-char minimum
        "1. A sentence that ends with a period.",
        "1. " + "A" * 80,  # longer than the 70-char cap
    ],
)
def test_detect_bare_heading_rejects(line):
    assert ch._detect_bare_heading(line) is None


def test_split_sentences_basic():
    out = ch.split_sentences("One thing. Two things! Three?  Four is last.")
    assert out == ["One thing.", "Two things!", "Three?", "Four is last."]


def test_split_sentences_empty_and_whitespace():
    assert ch.split_sentences("") == []
    assert ch.split_sentences("   \n  ") == []


def test_split_sentences_does_not_split_after_abbreviation():
    # "e.g." / "Dr." end in a period but are not sentence ends.
    assert ch.split_sentences("Buy now e.g. items ship free.") == [
        "Buy now e.g. items ship free."
    ]
    assert ch.split_sentences("See Dr. Smith today. Then return it.") == [
        "See Dr. Smith today.", "Then return it."
    ]


def test_split_sentences_splits_lowercase_led_sentences():
    assert ch.split_sentences("first one. second one. Third one!") == [
        "first one.", "second one.", "Third one!"
    ]


# ════════════════════════════════════════════════════════════════════════════
# chunker — parse_markdown
# ════════════════════════════════════════════════════════════════════════════


def test_parse_markdown_empty_document():
    assert ch.parse_markdown("") == []
    assert ch.parse_markdown("\n\n   \n") == []


def test_parse_markdown_headings_paragraphs_and_lists():
    blocks = ch.parse_markdown(
        "# Return Policy\n"
        "\n"
        "Items may be returned.\n"
        "\n"
        "## Exceptions\n"
        "- Gift cards\n"
        "- Digital codes\n"
    )
    kinds = [(b.block_type, b.heading) for b in blocks]
    assert kinds[0] == ("heading", "Return Policy")
    assert kinds[1] == ("paragraph", "Return Policy")
    assert kinds[2] == ("heading", "Exceptions")
    # List items accumulate into one paragraph block under the current heading.
    assert blocks[3].block_type == "paragraph"
    assert blocks[3].heading == "Exceptions"
    assert blocks[3].text == "- Gift cards\n- Digital codes"


def test_parse_markdown_heading_levels():
    blocks = ch.parse_markdown("# A\n## B\n### C\n#### D\n")
    assert [b.level for b in blocks] == [1, 2, 3, 4]


def test_parse_markdown_table_block_carries_rows():
    blocks = ch.parse_markdown(
        "## Shipping\n"
        "\n"
        "| Country | Price |\n"
        "| --- | --- |\n"
        "| Canada | $19.99 |\n"
        "| Mexico | $24.99 |\n"
    )
    tables = [b for b in blocks if b.block_type == "table"]
    assert len(tables) == 1
    assert tables[0].heading == "Shipping"
    assert tables[0].rows == [
        ["Country", "Price"],
        ["Canada", "$19.99"],
        ["Mexico", "$24.99"],
    ]


def test_parse_markdown_single_row_table_has_no_rows_metadata():
    blocks = ch.parse_markdown("| Only | Header |\n")
    tables = [b for b in blocks if b.block_type == "table"]
    assert len(tables) == 1
    assert tables[0].rows is None


def test_parse_markdown_horizontal_rule_flushes_paragraph():
    blocks = ch.parse_markdown("First line\n---\nSecond line\n")
    paras = [b.text for b in blocks if b.block_type == "paragraph"]
    assert paras == ["First line", "Second line"]


def test_parse_markdown_bare_numbered_heading_from_pdf_text():
    blocks = ch.parse_markdown("1.3 Holiday Return Policy\nItems bought in November.\n")
    assert blocks[0].block_type == "heading"
    assert blocks[0].heading == "1.3 Holiday Return Policy"
    assert blocks[1].heading == "1.3 Holiday Return Policy"


def test_parse_markdown_table_terminated_by_following_text():
    blocks = ch.parse_markdown(
        "| A | B |\n| --- | --- |\n| 1 | 2 |\nAfter the table.\n"
    )
    assert [b.block_type for b in blocks] == ["table", "paragraph"]
    assert blocks[1].text == "After the table."


def test_parse_markdown_unicode_preserved():
    blocks = ch.parse_markdown("# Política de Devolución\n\nArtículos — 30 días.\n")
    assert blocks[0].heading == "Política de Devolución"
    assert "—" in blocks[1].text


# ════════════════════════════════════════════════════════════════════════════
# chunker — parse_html
# ════════════════════════════════════════════════════════════════════════════


def test_parse_html_strips_chrome_and_keeps_structure():
    blocks = ch.parse_html(
        "<html><body>"
        "<nav>menu menu menu</nav>"
        "<h1>Shipping Policy</h1>"
        "<p>Standard shipping takes three to five business days.</p>"
        "<p>tiny</p>"
        "<ul><li>Canada</li><li>Mexico</li></ul>"
        "<footer>footer text here</footer>"
        "</body></html>"
    )
    types_ = [b.block_type for b in blocks]
    assert types_ == ["heading", "paragraph", "list"]
    assert blocks[0].level == 1
    assert blocks[1].heading == "Shipping Policy"
    assert blocks[2].text == "- Canada\n- Mexico"
    joined = " ".join(b.text for b in blocks)
    assert "menu" not in joined and "footer text" not in joined
    assert "tiny" not in joined  # <= 10 chars is dropped


def test_parse_html_table_rows_survive_the_dom():
    blocks = ch.parse_html(
        "<body><h2>Fees</h2><table>"
        "<tr><th>Country</th><th>Fee</th></tr>"
        "<tr><td>Canada</td><td>$19.99</td></tr>"
        "</table></body>"
    )
    table = [b for b in blocks if b.block_type == "table"][0]
    assert table.rows == [["Country", "Fee"], ["Canada", "$19.99"]]
    assert table.heading == "Fees"
    assert table.text.startswith("| Country | Fee |")


def test_parse_html_empty_document():
    assert ch.parse_html("") == []
    assert ch.parse_html("<html><body></body></html>") == []


# ════════════════════════════════════════════════════════════════════════════
# chunker — merge_blocks_into_chunks
# ════════════════════════════════════════════════════════════════════════════


def _para(text: str, heading: str = "") -> ch.StructuralBlock:
    return ch.StructuralBlock(text=text, block_type="paragraph", heading=heading)


def _heading(text: str, level: int) -> ch.StructuralBlock:
    """A heading block as parse_markdown builds it: text keeps the "#" markers,
    `heading` holds the bare title."""
    return ch.StructuralBlock(
        text=f"{'#' * level} {text}", block_type="heading", heading=text, level=level
    )


def _sentences(n: int, word: str = "Alpha") -> str:
    return " ".join(f"{word} sentence number {i} goes here." for i in range(n))


def test_merge_empty_blocks():
    assert ch.merge_blocks_into_chunks([]) == []


def test_merge_keeps_text_below_min_when_no_previous_chunk():
    # A whole document shorter than min_tokens must still produce a chunk, or it
    # is never indexed.
    out = ch.merge_blocks_into_chunks([_para("Too short to keep.")], min_tokens=50)
    assert len(out) == 1
    assert out[0]["text"] == "Too short to keep."


def test_merge_small_tail_is_appended_to_previous_chunk():
    big = _para(_sentences(30))
    out = ch.merge_blocks_into_chunks(
        [big, _heading("Z", 1), _para("tail")], max_tokens=200, min_tokens=50
    )
    assert len(out) == 1
    assert out[0]["text"].endswith("tail")


def test_merge_packs_blocks_up_to_max_tokens():
    block = _para("x" * 120)  # exactly 30 estimated tokens each
    assert ch.estimate_tokens(block.text) == 30
    out = ch.merge_blocks_into_chunks(
        [block, block, block, block], max_tokens=60, overlap_tokens=1, min_tokens=1
    )
    # Two blocks per chunk: a third would exceed the 60-token budget.
    assert len(out) == 2
    assert out[0]["text"].count("x" * 120) == 2


def test_merge_chunk_body_stays_within_max_tokens():
    blocks = [_para(_sentences(6, w)) for w in ("Alpha", "Beta", "Gamma", "Delta")]
    out = ch.merge_blocks_into_chunks(blocks, max_tokens=60, overlap_tokens=1, min_tokens=1)
    assert len(out) > 1
    for c in out:
        body = c["text"].split("\n\n", 1)[-1] if c["text"].startswith("...") else c["text"]
        assert ch.estimate_tokens(body) <= 60


def test_merge_splits_oversized_lowercase_led_block_into_budget():
    # Lowercase-led sentences are still sentence boundaries, so an oversized block
    # of them is split instead of staying one chunk far above max_tokens.
    text = "this is one long run of text. " * 100
    out = ch.merge_blocks_into_chunks(
        [_para(text)], max_tokens=50, overlap_tokens=1, min_tokens=1
    )
    assert len(out) > 1
    for c in out:
        assert ch.estimate_tokens(c["text"]) <= 60


def test_merge_major_heading_forces_new_chunk():
    out = ch.merge_blocks_into_chunks(
        [
            _heading("One", 1),
            _para(_sentences(20, "Alpha")),
            _heading("Two", 1),
            _para(_sentences(20, "Beta")),
        ],
        max_tokens=512,
        overlap_tokens=1,
        min_tokens=1,
    )
    assert len(out) == 2
    assert "Alpha" in out[0]["text"] and "Beta" not in out[0]["text"]
    assert "Beta" in out[1]["text"]


def test_merge_minor_heading_packs_with_following_content():
    out = ch.merge_blocks_into_chunks(
        [
            _heading("One", 3),
            _para(_sentences(5, "Alpha")),
            _heading("Two", 3),
            _para(_sentences(5, "Beta")),
        ],
        max_tokens=512,
        overlap_tokens=1,
        min_tokens=1,
    )
    assert len(out) == 1
    assert "Alpha" in out[0]["text"] and "Beta" in out[0]["text"]


def test_merge_oversized_table_stands_alone_intact():
    table_text = "\n".join(f"| row{i} | value{i} |" for i in range(200))
    blocks = [_heading("Fees", 1), ch.StructuralBlock(table_text, "table", heading="Fees")]
    out = ch.merge_blocks_into_chunks(blocks, max_tokens=100, overlap_tokens=1, min_tokens=1)
    table_chunks = [c for c in out if c["has_table"]]
    assert len(table_chunks) == 1
    assert table_text in table_chunks[0]["text"]
    assert table_chunks[0]["heading"] == "Fees"


def test_merge_small_table_packs_into_buffer():
    blocks = [_para(_sentences(5)), ch.StructuralBlock("| a | b |", "table")]
    out = ch.merge_blocks_into_chunks(blocks, max_tokens=512, overlap_tokens=1, min_tokens=1)
    assert len(out) == 1
    assert out[0]["has_table"] is True


def test_merge_splits_oversized_block_at_sentence_boundaries():
    text = _sentences(60)
    out = ch.merge_blocks_into_chunks(
        [_para(text)], max_tokens=50, overlap_tokens=1, min_tokens=1
    )
    assert len(out) > 1
    # No chunk may end mid-sentence.
    for c in out[:-1]:
        assert c["text"].rstrip().endswith(".")


def test_merge_overlap_prefix_present_and_bounded():
    blocks = [_para(_sentences(20, w)) for w in ("Alpha", "Beta", "Gamma")]
    out = ch.merge_blocks_into_chunks(blocks, max_tokens=60, overlap_tokens=10, min_tokens=1)
    assert len(out) > 1
    for c in out[1:]:
        prefix = c["text"].split("\n\n")[0]
        assert prefix.startswith("...")
        # The carried-over context stays close to the requested overlap budget.
        assert len(prefix) <= ch.estimate_chars(10) + 10


def test_merge_zero_overlap_carries_no_context():
    blocks = [_para(_sentences(20, w)) for w in ("Alpha", "Beta", "Gamma")]
    out = ch.merge_blocks_into_chunks(blocks, max_tokens=60, overlap_tokens=0, min_tokens=1)
    assert len(out) > 1
    for c in out[1:]:
        assert not c["text"].startswith("...")


def test_merge_overlap_uses_whole_previous_chunk_when_short():
    # A size split (8 + 50 tokens > 50) inside one section, so overlap applies.
    short_prev = "Alpha one. " * 3
    blocks = [_para(short_prev), _para(_sentences(6, "Beta"))]
    out = ch.merge_blocks_into_chunks(
        blocks, max_tokens=50, overlap_tokens=200, min_tokens=1
    )
    assert len(out) == 2
    assert out[1]["text"].startswith(short_prev.strip()[:10])


@pytest.mark.parametrize("level", [1, 2])
def test_merge_overlap_never_crosses_a_major_section(level):
    blocks = [_heading("First", 1), _para(_sentences(4, "Alpha")),
              _heading("Second", level), _para(_sentences(4, "Beta"))]
    out = ch.merge_blocks_into_chunks(blocks, overlap_tokens=200, min_tokens=1)
    assert len(out) == 2
    assert "Alpha" not in out[1]["text"]
    assert out[1]["text"].startswith("#" * level + " Second")


def test_merge_overlap_still_applies_after_h3():
    # H3 does not flush, so a size split after it is still within one section.
    blocks = [_heading("Top", 2), _para(_sentences(5, "Alpha")),
              _heading("Sub", 3), _para(_sentences(6, "Beta"))]
    out = ch.merge_blocks_into_chunks(blocks, max_tokens=60, overlap_tokens=20,
                                      min_tokens=1)
    assert len(out) >= 2
    assert out[-1]["text"].startswith("...")


def test_merge_metadata_keys_are_stable():
    out = ch.merge_blocks_into_chunks(
        [_heading("H", 1), _para(_sentences(20))], min_tokens=1
    )
    for c in out:
        assert set(c) == {"text", "heading", "has_table"}


# ════════════════════════════════════════════════════════════════════════════
# chunker — table_to_chunks
# ════════════════════════════════════════════════════════════════════════════


def _table_block(rows, heading="Shipping"):
    return ch.StructuralBlock(
        text=ch._rows_to_markdown(rows), block_type="table", heading=heading, rows=rows
    )


def test_table_to_chunks_emits_whole_table_plus_one_per_row():
    rows = [["Country", "Fee"], ["Canada", "$19.99"], ["Mexico", "$24.99"]]
    out = ch.table_to_chunks(_table_block(rows), 1, "Shipping Policy", "shipping")
    assert len(out) == 3
    assert out[0]["chunk_type"] == "table"
    assert [c["chunk_type"] for c in out[1:]] == ["table_row", "table_row"]
    assert all(c["table_id"] == "shipping#table1" for c in out)
    assert all(c["has_table"] for c in out)
    assert [c["row_index"] for c in out[1:]] == [1, 2]


def test_table_to_chunks_row_text_and_row_dict():
    rows = [["Country", "Fee"], ["Canada", "$19.99"]]
    out = ch.table_to_chunks(_table_block(rows), 2, "Shipping Policy", "shipping")
    row = out[1]
    assert row["row"] == {"Country": "Canada", "Fee": "$19.99"}
    assert row["text"].splitlines() == [
        "[Shipping Policy — Shipping — table 2, row 1 of 1]",
        "Country: Canada",
        "Fee: $19.99",
    ]


def test_table_to_chunks_context_line_without_section_heading():
    rows = [["A", "B"], ["1", "2"]]
    out = ch.table_to_chunks(_table_block(rows, heading=""), 1, "Doc Title", "doc")
    assert out[0]["text"].startswith("[Doc Title — table 1, 1 rows]")


def test_table_to_chunks_pads_short_rows_and_skips_empty_values():
    rows = [["A", "B", "C"], ["1"]]
    out = ch.table_to_chunks(_table_block(rows), 1, "T", "t")
    row = out[1]
    assert row["row"] == {"A": "1", "B": "", "C": ""}
    # Empty cells do not produce "B: " lines.
    assert row["text"].splitlines()[1:] == ["A: 1"]


def test_table_to_chunks_duplicate_headers_do_not_collide():
    rows = [["A", "A"], ["1", "2"]]
    out = ch.table_to_chunks(_table_block(rows), 1, "T", "t")
    assert out[1]["row"] == {"A": "1", "A (2)": "2"}


def test_table_to_chunks_header_only_table_gives_only_whole_table_chunk():
    out = ch.table_to_chunks(_table_block([["A", "B"]]), 1, "T", "t")
    assert len(out) == 1
    assert "0 rows" in out[0]["text"]


def test_table_to_chunks_extra_cells_beyond_header_are_dropped():
    rows = [["A"], ["1", "2"]]
    out = ch.table_to_chunks(_table_block(rows), 1, "T", "t")
    assert out[1]["row"] == {"A": "1"}


# ════════════════════════════════════════════════════════════════════════════
# chunker — chunk_document / chunk_all_policies
# ════════════════════════════════════════════════════════════════════════════


MD_DOC = """# Shipping Policy

{prose}

## Rates

| Country | Fee |
| --- | --- |
| Canada | $19.99 |
| Mexico | $24.99 |
"""


def _write_doc(tmp_path: Path, name: str = "shipping_policy.md") -> Path:
    p = tmp_path / name
    p.write_text(MD_DOC.format(prose=_sentences(30, "Shipping")), encoding="utf-8")
    return p


def test_chunk_document_metadata_propagation(tmp_path):
    path = _write_doc(tmp_path)
    chunks = ch.chunk_document(path, synthetic=True)
    assert chunks
    assert all(c.source_file == "shipping_policy.md" for c in chunks)
    assert all(c.source_type == "md" for c in chunks)
    assert all(c.synthetic is True for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.total_chunks == len(chunks) for c in chunks)
    assert all(c.token_count == ch.estimate_tokens(c.text) for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(len(c.chunk_id) == 16 for c in chunks)


def test_chunk_document_table_rows_become_their_own_chunks(tmp_path):
    chunks = ch.chunk_document(_write_doc(tmp_path))
    row_chunks = [c for c in chunks if c.chunk_type == "table_row"]
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert len(row_chunks) == 2
    assert {c.row["Country"] for c in row_chunks} == {"Canada", "Mexico"}
    assert all(c.table_id == "shipping_policy#table1" for c in row_chunks + table_chunks)
    assert all(c.has_table for c in row_chunks + table_chunks)
    # The structured table left the prose stream.
    prose = [c for c in chunks if c.chunk_type == "text"]
    assert all("Canada" not in c.text for c in prose)


def test_chunk_document_title_taken_from_h1(tmp_path):
    chunks = ch.chunk_document(_write_doc(tmp_path))
    row = [c for c in chunks if c.chunk_type == "table_row"][0]
    assert row.text.startswith("[Shipping Policy — Rates — table 1")


def test_chunk_document_title_falls_back_to_filename(tmp_path):
    p = tmp_path / "price_match_rules.md"
    p.write_text(
        "No markdown title here.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n", encoding="utf-8"
    )
    chunks = ch.chunk_document(p)
    table = [c for c in chunks if c.chunk_type == "table"][0]
    assert table.text.startswith("[Price Match Rules — table 1")


def test_chunk_document_empty_file_returns_no_chunks(tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")
    assert ch.chunk_document(p) == []


def test_chunk_document_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ch.chunk_document(tmp_path / "nope.md")


def test_chunk_document_bad_encoding_raises(tmp_path):
    p = tmp_path / "latin1.md"
    p.write_bytes(b"# Caf\xe9 policy\n\n" + b"Some prose. " * 40)
    with pytest.raises(UnicodeDecodeError):
        ch.chunk_document(p)


def test_chunk_document_source_type_detected_from_suffix(tmp_path):
    p = tmp_path / "page.html"
    p.write_text(
        "<body><h1>Returns</h1><p>" + _sentences(20) + "</p></body>", encoding="utf-8"
    )
    chunks = ch.chunk_document(p)
    assert chunks and all(c.source_type == "html" for c in chunks)


def test_chunk_document_unknown_suffix_defaults_to_markdown(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("# Title\n\n" + _sentences(30), encoding="utf-8")
    chunks = ch.chunk_document(p)
    assert chunks and all(c.source_type == "md" for c in chunks)


def test_chunk_to_dict_round_trips_through_json(tmp_path):
    c = ch.chunk_document(_write_doc(tmp_path))[0]
    d = c.to_dict()
    assert set(d) >= {
        "chunk_id", "text", "source_file", "source_type", "chunk_index",
        "total_chunks", "section_heading", "has_table", "token_count",
        "synthetic", "chunk_type", "table_id", "row_index", "row",
    }
    assert json.loads(json.dumps(d, ensure_ascii=False)) == d


def test_chunk_all_policies_discovers_and_excludes(tmp_path, capsys):
    _write_doc(tmp_path, "shipping_policy.md")
    _write_doc(tmp_path, "return_policy.md")      # excluded source
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    chunks = ch.chunk_all_policies(tmp_path, verbose=False)
    assert chunks
    assert {c.source_file for c in chunks} == {"shipping_policy.md"}


def test_chunk_all_policies_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ch.chunk_all_policies(tmp_path, verbose=False)


def test_chunk_all_policies_missing_dir_raises(tmp_path):
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        ch.chunk_all_policies(tmp_path / "nope", verbose=False)


def test_chunk_all_policies_verbose_prints_one_line_per_doc(tmp_path, capsys):
    _write_doc(tmp_path, "a_policy.md")
    _write_doc(tmp_path, "b_policy.md")
    ch.chunk_all_policies(tmp_path, verbose=True)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2
    assert "a_policy.md" in out[0] and "b_policy.md" in out[1]


def test_chunk_all_policies_chunk_indices_restart_per_document(tmp_path):
    _write_doc(tmp_path, "a_policy.md")
    _write_doc(tmp_path, "b_policy.md")
    chunks = ch.chunk_all_policies(tmp_path, verbose=False)
    a = [c.chunk_index for c in chunks if c.source_file == "a_policy.md"]
    b = [c.chunk_index for c in chunks if c.source_file == "b_policy.md"]
    assert a == list(range(len(a))) and b == list(range(len(b)))


# ════════════════════════════════════════════════════════════════════════════
# indexer
# ════════════════════════════════════════════════════════════════════════════


def test_point_id_is_deterministic_uuid5():
    pid = ix.point_id("return_policy.pdf", 3)
    assert pid == str(uuid.uuid5(uuid.NAMESPACE_URL, "return_policy.pdf#3"))
    assert pid == ix.point_id("return_policy.pdf", 3)
    uuid.UUID(pid)  # parses


def test_point_id_differs_by_index_and_file():
    assert ix.point_id("a.pdf", 0) != ix.point_id("a.pdf", 1)
    assert ix.point_id("a.pdf", 0) != ix.point_id("b.pdf", 0)


def test_point_id_handles_unicode_filenames():
    assert ix.point_id("política.pdf", 0) != ix.point_id("politica.pdf", 0)


def test_load_chunks_skips_blank_lines(tmp_path):
    p = tmp_path / "chunks.jsonl"
    p.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert ix.load_chunks(p) == [{"a": 1}, {"a": 2}]


def test_load_chunks_empty_file(tmp_path):
    p = tmp_path / "chunks.jsonl"
    p.write_text("", encoding="utf-8")
    assert ix.load_chunks(p) == []


def test_load_chunks_preserves_unicode(tmp_path):
    p = tmp_path / "chunks.jsonl"
    p.write_text('{"text": "Café — 30 días"}\n', encoding="utf-8")
    assert ix.load_chunks(p)[0]["text"] == "Café — 30 días"


def test_load_chunks_malformed_line_raises(tmp_path):
    p = tmp_path / "chunks.jsonl"
    p.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        ix.load_chunks(p)


def test_load_chunks_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ix.load_chunks(tmp_path / "nope.jsonl")


def test_load_chunks_bad_encoding_raises(tmp_path):
    p = tmp_path / "chunks.jsonl"
    p.write_bytes(b'{"text": "caf\xe9"}\n')
    with pytest.raises(UnicodeDecodeError):
        ix.load_chunks(p)


# ── build_index with stubbed models / client ────────────────────────────────


class _FakeDenseVec:
    def __init__(self, dim):
        self._v = [0.1] * dim

    def tolist(self):
        return self._v

    def __len__(self):
        return len(self._v)


class _Sparse:
    """Mimics fastembed's SparseEmbedding enough for build_index."""

    def __init__(self, n=3):
        self.indices = _ArrayLike(list(range(n)))
        self.values = _ArrayLike([1.0] * n)


class _ArrayLike(list):
    def tolist(self):
        return list(self)


class _FakeDense(_ArrayLike):
    pass


class _FakeEmbedder:
    def __init__(self, dim=384):
        self.dim = dim

    def embed(self, texts, batch_size=32):
        return [_FakeDense([0.1] * self.dim) for _ in texts]


class _FakeSparseEmbedder:
    def embed(self, texts, batch_size=32):
        return [_Sparse(3) for _ in texts]


class _FakeQdrant:
    def __init__(self):
        self.upserted = None
        self.collections = set()

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name=None, **kw):
        self.collections.add(collection_name)

    def delete_collection(self, name):
        self.collections.discard(name)

    def upsert(self, collection, points=None, wait=True):
        self.upserted = points

    def count(self, collection, exact=True):
        return types.SimpleNamespace(count=len(self.upserted or []))


@pytest.fixture
def fake_index_env(monkeypatch):
    client = _FakeQdrant()
    dense = _FakeEmbedder()
    monkeypatch.setattr(ix, "qdrant_client", lambda: client)
    monkeypatch.setattr(ix, "dense_model", lambda: dense)
    monkeypatch.setattr(ix, "bm25_model", lambda: _FakeSparseEmbedder())
    return types.SimpleNamespace(client=client, dense=dense)


def _chunks_file(tmp_path, rows):
    p = tmp_path / "chunks.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_build_index_empty_corpus_raises(tmp_path, fake_index_env):
    p = _chunks_file(tmp_path, [])
    with pytest.raises(ValueError, match="No chunks found"):
        ix.build_index(p)


def test_build_index_dimension_mismatch_raises(tmp_path, fake_index_env):
    fake_index_env.dense.dim = 16
    p = _chunks_file(tmp_path, [{"text": "a", "source_file": "a.md", "chunk_index": 0}])
    with pytest.raises(ValueError, match="dim vectors"):
        ix.build_index(p)


def test_build_index_summary_and_point_ids(tmp_path, fake_index_env):
    rows = [
        {"text": "alpha", "source_file": "a.md", "chunk_index": 0},
        {"text": "beta", "source_file": "a.md", "chunk_index": 1},
        {"text": "gamma", "source_file": "b.md", "chunk_index": 0},
    ]
    stats = ix.build_index(_chunks_file(tmp_path, rows))
    assert stats["points"] == 3
    assert stats["documents"] == 2
    assert stats["avg_bm25_terms"] == 3.0
    assert set(stats["seconds"]) == {"dense_embed", "bm25_embed", "upsert"}
    points = fake_index_env.client.upserted
    assert [str(p.id) for p in points] == [
        ix.point_id("a.md", 0), ix.point_id("a.md", 1), ix.point_id("b.md", 0)
    ]
    assert points[0].payload == rows[0]


def test_build_index_missing_payload_key_raises(tmp_path, fake_index_env):
    p = _chunks_file(tmp_path, [{"text": "a", "chunk_index": 0}])  # no source_file
    with pytest.raises(KeyError):
        ix.build_index(p)


# ════════════════════════════════════════════════════════════════════════════
# retriever
# ════════════════════════════════════════════════════════════════════════════


def _point(chunk_id, source_file, chunk_index, score=None, text=None,
           heading="H", total_chunks=5, pid=None):
    payload = {
        "chunk_id": chunk_id,
        "text": text if text is not None else f"text-{chunk_id}",
        "source_file": source_file,
        "chunk_index": chunk_index,
        "section_heading": heading,
        "total_chunks": total_chunks,
    }
    p = types.SimpleNamespace(payload=payload, id=pid or ix.point_id(source_file, chunk_index))
    if score is not None:
        p.score = score
    return p


def test_retrieved_chunk_from_point_maps_payload():
    c = rt.RetrievedChunk.from_point(_point("c1", "a.md", 2, score=0.75), "dense")
    assert (c.chunk_id, c.source_file, c.chunk_index, c.origin) == ("c1", "a.md", 2, "dense")
    assert c.score == 0.75
    assert c.section_heading == "H"
    assert c.payload["total_chunks"] == 5


def test_retrieved_chunk_from_point_without_score_defaults_zero():
    c = rt.RetrievedChunk.from_point(_point("c1", "a.md", 0), "adjacent")
    assert c.score == 0.0


def test_retrieved_chunk_from_point_none_heading_becomes_empty_string():
    p = _point("c1", "a.md", 0)
    p.payload["section_heading"] = None
    assert rt.RetrievedChunk.from_point(p, "bm25").section_heading == ""


def test_retrieved_chunk_from_point_missing_required_key_raises():
    p = _point("c1", "a.md", 0)
    del p.payload["text"]
    with pytest.raises(KeyError):
        rt.RetrievedChunk.from_point(p, "dense")


def test_retrieved_chunk_from_point_empty_payload_raises():
    with pytest.raises(KeyError):
        rt.RetrievedChunk.from_point(types.SimpleNamespace(payload=None), "dense")


def test_retrieval_result_total_ms():
    r = rt.RetrievalResult("q", "full", [], {"a": 1.24, "b": 2.0})
    assert r.total_ms == 3.2
    assert rt.RetrievalResult("q", "full", [], {}).total_ms == 0


def test_timer_records_stage():
    t: dict[str, float] = {}
    with rt._Timer(t, "stage"):
        pass
    assert "stage" in t and t["stage"] >= 0.0


# ── stubbed retriever ───────────────────────────────────────────────────────


class _QueryResponse:
    def __init__(self, points):
        self.points = points


class _FakeRetrievalClient:
    def __init__(self, points=(), records=(), exists=True):
        self.points = list(points)
        self.records = {str(r.id): r for r in records}
        self.exists = exists
        self.calls = []

    def collection_exists(self, name):
        return self.exists

    def query_points(self, collection, **kw):
        self.calls.append(("query_points", kw))
        limit = kw.get("limit", len(self.points))
        return _QueryResponse(self.points[:limit])

    def retrieve(self, collection, ids=None, with_payload=True):
        self.calls.append(("retrieve", list(ids)))
        return [self.records[i] for i in ids if i in self.records]


class _FakeEmb:
    def query_embed(self, q):
        return iter([_FakeDense([0.1, 0.2, 0.3])])


class _FakeSparseEmb:
    def query_embed(self, q):
        return iter([_Sparse(2)])


class _FakeReranker:
    """Scores by position of a marker token so ranking is deterministic."""

    def __init__(self, scores=None):
        self.scores = scores
        self.calls = []

    def rerank(self, query, docs, batch_size=64):
        self.calls.append((query, list(docs)))
        if self.scores is not None:
            return list(self.scores)[: len(docs)]
        return [float(len(d)) for d in docs]


@pytest.fixture
def retriever_factory(monkeypatch):
    def make(points=(), records=(), exists=True, scores=None, cache_size=0):
        client = _FakeRetrievalClient(points, records, exists)
        reranker = _FakeReranker(scores)
        monkeypatch.setattr(rt, "qdrant_client", lambda: client)
        monkeypatch.setattr(rt, "dense_model", lambda: _FakeEmb())
        monkeypatch.setattr(rt, "bm25_model", lambda: _FakeSparseEmb())
        monkeypatch.setattr(rt, "reranker_model", lambda: reranker)
        r = rt.HybridRetriever(collection="test", cache_size=cache_size)
        r._fake_client, r._fake_reranker = client, reranker
        return r

    return make


def test_retriever_missing_collection_raises(retriever_factory):
    with pytest.raises(RuntimeError, match="not found"):
        retriever_factory(exists=False)


def test_dense_and_bm25_search_tag_origin_and_respect_limit(retriever_factory):
    pts = [_point(f"c{i}", "a.md", i, score=1.0 - i / 10) for i in range(5)]
    r = retriever_factory(points=pts)
    dense = r.dense_search("shipping", limit=3)
    assert len(dense) == 3 and all(c.origin == "dense" for c in dense)
    assert [c.score for c in dense] == pytest.approx([1.0, 0.9, 0.8])
    bm = r.bm25_search("shipping", limit=2)
    assert len(bm) == 2 and all(c.origin == "bm25" for c in bm)


def test_hybrid_search_sends_two_prefetches_and_rrf_query(retriever_factory):
    r = retriever_factory(points=[_point("c0", "a.md", 0, score=0.5)])
    out = r.hybrid_search("shipping", limit=5, prefetch_k=9)
    assert out and out[0].origin == "hybrid"
    kw = [c for c in r._fake_client.calls if c[0] == "query_points"][0][1]
    assert len(kw["prefetch"]) == 2
    assert [p.limit for p in kw["prefetch"]] == [9, 9]
    assert kw["query"].rrf.k == rt.config.RRF_K
    assert kw["limit"] == 5


def test_expand_adjacent_adds_neighbours_in_rank_order(retriever_factory):
    hits = [_point("c3", "a.md", 3, score=0.9, total_chunks=10)]
    neighbours = [_point("c2", "a.md", 2), _point("c4", "a.md", 4)]
    r = retriever_factory(points=hits, records=neighbours)
    chunks = [rt.RetrievedChunk.from_point(p, "hybrid") for p in hits]
    out = r.expand_adjacent(chunks, window=1, cap=10)
    assert [c.chunk_index for c in out] == [3, 2, 4]
    assert [c.origin for c in out] == ["hybrid", "adjacent", "adjacent"]


def test_expand_adjacent_skips_out_of_range_neighbours(retriever_factory):
    hit = _point("c0", "a.md", 0, score=0.9, total_chunks=1)
    r = retriever_factory(points=[hit], records=[])
    chunks = [rt.RetrievedChunk.from_point(hit, "hybrid")]
    out = r.expand_adjacent(chunks, window=1, cap=10)
    assert out == chunks
    assert not [c for c in r._fake_client.calls if c[0] == "retrieve"]


def test_expand_adjacent_never_duplicates_an_existing_hit(retriever_factory):
    hits = [_point("c1", "a.md", 1, score=0.9, total_chunks=5),
            _point("c2", "a.md", 2, score=0.8, total_chunks=5)]
    recs = [_point("c0", "a.md", 0), _point("c3", "a.md", 3)]
    r = retriever_factory(points=hits, records=recs)
    chunks = [rt.RetrievedChunk.from_point(p, "hybrid") for p in hits]
    out = r.expand_adjacent(chunks, window=1, cap=10)
    keys = [(c.source_file, c.chunk_index) for c in out]
    assert len(keys) == len(set(keys))
    assert sorted(c.chunk_index for c in out) == [0, 1, 2, 3]


def test_expand_adjacent_cap_truncates_fused_hits_first(retriever_factory):
    hits = [_point(f"c{i}", "a.md", i, score=1 - i / 10, total_chunks=20) for i in range(5)]
    r = retriever_factory(points=hits, records=[])
    chunks = [rt.RetrievedChunk.from_point(p, "hybrid") for p in hits]
    out = r.expand_adjacent(chunks, window=1, cap=3)
    assert len(out) == 3
    assert [c.chunk_index for c in out] == [0, 1, 2]


def test_expand_adjacent_room_limits_number_of_neighbours(retriever_factory):
    hits = [_point("c5", "a.md", 5, score=0.9, total_chunks=20)]
    recs = [_point("c4", "a.md", 4), _point("c6", "a.md", 6)]
    r = retriever_factory(points=hits, records=recs)
    chunks = [rt.RetrievedChunk.from_point(p, "hybrid") for p in hits]
    out = r.expand_adjacent(chunks, window=1, cap=2)
    assert len(out) == 2
    fetched = [c for c in r._fake_client.calls if c[0] == "retrieve"][0][1]
    assert len(fetched) == 1


def test_expand_adjacent_drops_ids_the_store_does_not_have(retriever_factory):
    hits = [_point("c1", "a.md", 1, score=0.9, total_chunks=5)]
    r = retriever_factory(points=hits, records=[_point("c0", "a.md", 0)])
    chunks = [rt.RetrievedChunk.from_point(p, "hybrid") for p in hits]
    out = r.expand_adjacent(chunks, window=1, cap=10)
    assert [c.chunk_index for c in out] == [1, 0]


def test_expand_adjacent_empty_input(retriever_factory):
    r = retriever_factory()
    assert r.expand_adjacent([]) == []


def test_expand_adjacent_defaults_total_chunks_when_payload_lacks_it(retriever_factory):
    p = _point("c2", "a.md", 2, score=0.9)
    del p.payload["total_chunks"]
    r = retriever_factory(points=[p], records=[_point("c1", "a.md", 1)])
    chunks = [rt.RetrievedChunk.from_point(p, "hybrid")]
    out = r.expand_adjacent(chunks, window=1, cap=10)
    # Without total_chunks the upper bound is chunk_index + 1, so only the
    # preceding neighbour can be fetched.
    assert [c.chunk_index for c in out] == [2, 1]


def test_rerank_sorts_descending_and_truncates(retriever_factory):
    r = retriever_factory(scores=[0.1, 0.9, 0.5])
    chunks = [rt.RetrievedChunk.from_point(_point(f"c{i}", "a.md", i), "hybrid")
              for i in range(3)]
    out = r.rerank("q", chunks, top_k=2)
    assert [c.chunk_id for c in out] == ["c1", "c2"]
    assert [c.rerank_score for c in out] == [0.9, 0.5]
    assert all(c.rerank_score is not None for c in chunks)


def test_rerank_empty_candidates(retriever_factory):
    r = retriever_factory()
    assert r.rerank("q", []) == []
    assert r._fake_reranker.calls == []


def test_rerank_score_count_mismatch_raises(retriever_factory):
    r = retriever_factory(scores=[0.1])
    chunks = [rt.RetrievedChunk.from_point(_point(f"c{i}", "a.md", i), "hybrid")
              for i in range(3)]
    with pytest.raises(ValueError):
        r.rerank("q", chunks)


def test_rerank_top_k_larger_than_candidates(retriever_factory):
    r = retriever_factory(scores=[0.3, 0.7])
    chunks = [rt.RetrievedChunk.from_point(_point(f"c{i}", "a.md", i), "hybrid")
              for i in range(2)]
    assert len(r.rerank("q", chunks, top_k=50)) == 2


@pytest.mark.parametrize("mode", ["nope", "", "DENSE", None])
def test_retrieve_rejects_unknown_mode(retriever_factory, mode):
    r = retriever_factory()
    with pytest.raises(ValueError, match="Unknown mode"):
        r.retrieve("q", mode=mode)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_retrieve_rejects_empty_query(retriever_factory, query):
    r = retriever_factory()
    with pytest.raises(ValueError, match="must not be empty"):
        r.retrieve(query, mode="dense")


@pytest.mark.parametrize(
    "mode,stage",
    [("dense", "dense"), ("bm25", "bm25"), ("hybrid", "hybrid_rrf")],
)
def test_retrieve_single_stage_modes(retriever_factory, mode, stage):
    pts = [_point(f"c{i}", "a.md", i, score=1 - i / 10) for i in range(4)]
    r = retriever_factory(points=pts)
    res = r.retrieve("shipping fees", mode=mode, top_k=2)
    assert res.mode == mode
    assert len(res.chunks) == 2
    assert list(res.timings_ms) == [stage]
    assert res.candidates_reranked == 0
    assert res.cache_hit is False


def test_retrieve_hybrid_rerank_mode_skips_expansion(retriever_factory):
    pts = [_point(f"c{i}", "a.md", i, score=1 - i / 10, total_chunks=20) for i in range(3)]
    r = retriever_factory(points=pts, records=[_point("c9", "a.md", 9)],
                          scores=[0.2, 0.9, 0.5])
    res = r.retrieve("q", mode="hybrid_rerank", top_k=3)
    assert "expand_adjacent" not in res.timings_ms
    assert set(res.timings_ms) == {"hybrid_rrf", "rerank"}
    assert res.candidates_reranked == 3
    assert [c.chunk_id for c in res.chunks] == ["c1", "c2", "c0"]


def test_retrieve_full_mode_runs_expansion(retriever_factory):
    pts = [_point("c1", "a.md", 1, score=0.9, total_chunks=5)]
    recs = [_point("c0", "a.md", 0), _point("c2", "a.md", 2)]
    r = retriever_factory(points=pts, records=recs, scores=[0.1, 0.9, 0.5])
    res = r.retrieve("q", mode="full", top_k=2)
    assert set(res.timings_ms) == {"hybrid_rrf", "expand_adjacent", "rerank"}
    assert res.candidates_reranked == 3
    assert [c.chunk_id for c in res.chunks] == ["c0", "c2"]
    assert res.total_ms == pytest.approx(sum(res.timings_ms.values()), abs=0.2)


def test_retrieve_cache_hit_is_returned_and_flagged(retriever_factory):
    pts = [_point("c0", "a.md", 0, score=0.9)]
    r = retriever_factory(points=pts, cache_size=4)
    first = r.retrieve("Shipping   fees", mode="dense", top_k=1)
    n_calls = len(r._fake_client.calls)
    second = r.retrieve("shipping fees", mode="dense", top_k=1)  # normalised key
    assert first.cache_hit is False and second.cache_hit is True
    assert list(second.timings_ms) == ["cache"]
    assert len(r._fake_client.calls) == n_calls  # no extra store round-trip
    assert [c.chunk_id for c in second.chunks] == [c.chunk_id for c in first.chunks]


def test_retrieve_cache_disabled_always_recomputes(retriever_factory):
    r = retriever_factory(points=[_point("c0", "a.md", 0, score=0.9)], cache_size=0)
    r.retrieve("q", mode="dense", top_k=1)
    n = len(r._fake_client.calls)
    second = r.retrieve("q", mode="dense", top_k=1)
    assert second.cache_hit is False
    assert len(r._fake_client.calls) > n


def test_retrieve_cache_is_keyed_by_mode_and_top_k(retriever_factory):
    r = retriever_factory(points=[_point("c0", "a.md", 0, score=0.9)], cache_size=4)
    r.retrieve("q", mode="dense", top_k=1)
    assert r.retrieve("q", mode="bm25", top_k=1).cache_hit is False
    assert r.retrieve("q", mode="dense", top_k=2).cache_hit is False
    assert r.retrieve("q", mode="dense", top_k=1).cache_hit is True


def test_retrieve_strips_query_before_use(retriever_factory):
    r = retriever_factory(points=[_point("c0", "a.md", 0, score=0.9)])
    assert r.retrieve("  shipping  ", mode="dense", top_k=1).query == "shipping"


def test_retrieve_empty_store_returns_no_chunks(retriever_factory):
    r = retriever_factory(points=[])
    res = r.retrieve("anything", mode="full", top_k=5)
    assert res.chunks == []
    assert res.candidates_reranked == 0


def test_modes_tuple_matches_literal():
    assert set(rt.MODES) == {"dense", "bm25", "hybrid", "hybrid_rerank", "full"}


# ════════════════════════════════════════════════════════════════════════════
# ingest
# ════════════════════════════════════════════════════════════════════════════


def _chunk(text="hello world", source="a.md", index=0, ctype="text", table=False, tokens=10):
    return ch.Chunk(
        chunk_id=f"id{index}", text=text, source_file=source, source_type="md",
        chunk_index=index, total_chunks=1, section_heading="H",
        has_table=table, token_count=tokens, synthetic=True, chunk_type=ctype,
    )


@pytest.fixture
def ingest_env(tmp_path, monkeypatch):
    out = tmp_path / "chunks.jsonl"
    monkeypatch.setattr(ing, "CHUNKS_FILE", out)
    monkeypatch.setattr(ing, "PROCESSED_DIR", tmp_path)
    return out


def test_ingest_policies_writes_jsonl_and_summary(ingest_env, monkeypatch, capsys):
    chunks = [
        _chunk("alpha", "a.md", 0, tokens=10),
        _chunk("beta", "a.md", 1, ctype="table", table=True, tokens=30),
        _chunk("gamma", "b.md", 0, ctype="table_row", table=True, tokens=20),
    ]
    monkeypatch.setattr(ing, "chunk_all_policies", lambda: chunks)
    summary = ing.ingest_policies()

    lines = ingest_env.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["text"] == "alpha"
    assert summary["total_chunks"] == 3
    assert summary["chunks_by_source"] == {"a.md": 2, "b.md": 1}
    assert summary["chunks_by_type"] == {"text": 1, "table": 1, "table_row": 1}
    assert summary["table_chunks"] == 2
    assert summary["total_tokens"] == 60
    assert summary["avg_tokens_per_chunk"] == 20
    assert summary["min_tokens"] == 10
    assert summary["max_tokens"] == 30
    assert summary["chunks_file"] == str(ingest_env)


def test_ingest_policies_empty_corpus_summary_has_safe_defaults(ingest_env, monkeypatch):
    monkeypatch.setattr(ing, "chunk_all_policies", lambda: [])
    summary = ing.ingest_policies()
    assert summary["total_chunks"] == 0
    assert summary["avg_tokens_per_chunk"] == 0  # no ZeroDivisionError
    assert summary["min_tokens"] == 0 and summary["max_tokens"] == 0
    assert ingest_env.read_text(encoding="utf-8") == ""


def test_ingest_policies_writes_unicode_unescaped(ingest_env, monkeypatch):
    monkeypatch.setattr(ing, "chunk_all_policies", lambda: [_chunk("Café — 30 días")])
    ing.ingest_policies()
    raw = ingest_env.read_text(encoding="utf-8")
    assert "Café — 30 días" in raw
    assert "\\u" not in raw


def test_ingest_policies_overwrites_previous_file(ingest_env, monkeypatch):
    ingest_env.write_text("stale\nstale\n", encoding="utf-8")
    monkeypatch.setattr(ing, "chunk_all_policies", lambda: [_chunk("fresh")])
    ing.ingest_policies()
    assert ingest_env.read_text(encoding="utf-8").strip().count("\n") == 0
    assert "stale" not in ingest_env.read_text(encoding="utf-8")


def test_ingest_policies_propagates_chunker_failure(ingest_env, monkeypatch):
    def boom():
        raise FileNotFoundError("no policies")

    monkeypatch.setattr(ing, "chunk_all_policies", boom)
    with pytest.raises(FileNotFoundError):
        ing.ingest_policies()


def test_ingest_operational_reports_warnings(monkeypatch, capsys):
    monkeypatch.setattr(
        ing, "load_operational_data",
        lambda: {"customers": 2, "orders": 3, "returns": 1, "db_path": "postgres://stub",
                 "warnings": [f"w{i}" for i in range(12)]},
    )
    summary = ing.ingest_operational()
    out = capsys.readouterr().out
    assert summary["orders"] == 3
    assert "12 validation warning(s)" in out
    assert "and 2 more" in out
    assert "w9" in out and "w10" not in out  # only the first 10 are listed


def test_ingest_operational_no_warnings(monkeypatch, capsys):
    monkeypatch.setattr(
        ing, "load_operational_data",
        lambda: {"customers": 1, "orders": 1, "returns": 0, "db_path": "postgres://stub",
                 "warnings": []},
    )
    ing.ingest_operational()
    assert "no validation warnings" in capsys.readouterr().out


def test_ingest_main_writes_summary_json(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(ing, "INGEST_SUMMARY_FILE", tmp_path / "ingestion_summary.json")
    monkeypatch.setattr(ing, "ingest_policies", lambda: {"total_chunks": 2})
    monkeypatch.setattr(ing, "ingest_operational", lambda: {"orders": 1})
    full = ing.main()
    on_disk = json.loads((tmp_path / "ingestion_summary.json").read_text())
    assert on_disk == full
    assert on_disk["policies"] == {"total_chunks": 2}
    assert on_disk["operational"] == {"orders": 1}
    assert on_disk["elapsed_seconds"] >= 0
    assert on_disk["started_at"].endswith("+00:00")


# ════════════════════════════════════════════════════════════════════════════
# policies
# ════════════════════════════════════════════════════════════════════════════


def test_human_date_formats_and_handles_none():
    assert pol.human_date(date(2025, 1, 31)) == "January 31, 2025"
    assert pol.human_date(date(2025, 12, 5)) == "December 5, 2025"  # no zero padding
    assert pol.human_date(None) is None


@pytest.mark.parametrize(
    "category,premium,expected",
    [
        ("Laptops & Computers", False, 30),
        ("Laptops & Computers", True, 45),
        ("Smartphones & Tablets", False, 15),
        ("Smartphones & Tablets", True, 30),
        ("Cables & Adapters", True, 60),
        ("Gift Cards", False, None),
        ("Gift Cards", True, None),
        ("Unicorns", False, None),
    ],
)
def test_effective_window(category, premium, expected):
    assert pol.effective_window(category, premium) == expected


@pytest.mark.parametrize(
    "purchase,expected",
    [
        (date(2024, 11, 14), None),
        (date(2024, 11, 15), date(2025, 1, 31)),
        (date(2024, 12, 1), date(2025, 1, 31)),
        (date(2024, 12, 31), date(2025, 1, 31)),
        (date(2025, 1, 1), None),
        (date(2024, 6, 1), None),
        (None, None),
    ],
)
def test_holiday_deadline_boundaries(purchase, expected):
    assert pol.holiday_deadline(purchase) == expected


def test_return_deadline_unknown_category():
    out = pol.return_deadline("Unicorns", False, date(2025, 3, 1))
    assert out == {"ok": False, "code": "UNKNOWN_CATEGORY",
                   "message": "Category 'Unicorns' has no return rule on file."}


def test_return_deadline_non_returnable():
    out = pol.return_deadline("Gift Cards", True, date(2025, 3, 1))
    assert out["ok"] is False and out["code"] == "NON_RETURNABLE"


def test_return_deadline_standard_window_from_delivery():
    out = pol.return_deadline("Laptops & Computers", False,
                              purchase_date=date(2025, 3, 1),
                              delivery_date=date(2025, 3, 5))
    assert out["ok"] is True
    assert out["window_counted_from"] == "delivery"
    assert out["deadline"] == "2025-04-04"
    assert out["rule_applied"] == "standard_window"
    assert out["holiday_period_purchase"] is False
    assert out["effective_window_days"] == 30
    assert out["premium_extension_days"] == 0
    assert out["deadline_text"] == "April 4, 2025"


def test_return_deadline_counts_from_purchase_without_delivery():
    out = pol.return_deadline("Audio & Headphones", False, date(2025, 3, 1))
    assert out["window_counted_from"] == "purchase"
    assert out["deadline"] == "2025-03-31"


def test_return_deadline_premium_extension_applies():
    out = pol.return_deadline("Audio & Headphones", True, date(2025, 3, 1))
    assert out["effective_window_days"] == 45
    assert out["premium_extension_days"] == 15
    assert out["deadline"] == "2025-04-15"
    assert "+ 15 Premium days" in out["explanation"]


def test_return_deadline_holiday_extension_wins_when_later():
    out = pol.return_deadline("Smartphones & Tablets", False,
                              purchase_date=date(2024, 12, 1),
                              delivery_date=date(2024, 12, 3))
    assert out["holiday_period_purchase"] is True
    assert out["holiday_deadline"] == "2025-01-31"
    assert out["standard_deadline"] == "2024-12-18"
    assert out["deadline"] == "2025-01-31"
    assert out["rule_applied"] == "holiday_extension"
    assert "January 31, 2025" in out["explanation"]


def test_return_deadline_never_shortens_a_window():
    # Standard window runs past Jan 31, so the standard rule must stay.
    out = pol.return_deadline("Cables & Adapters", True,
                              purchase_date=date(2024, 12, 31),
                              delivery_date=date(2024, 12, 31))
    assert out["rule_applied"] == "standard_window"
    assert out["deadline"] == "2025-03-01"
    assert out["holiday_deadline"] == "2025-01-31"


def test_return_deadline_without_any_date():
    out = pol.return_deadline("Laptops & Computers", False, None)
    assert out["ok"] is True
    assert out["deadline"] is None
    assert out["standard_deadline"] is None
    assert out["deadline_text"] is None
    assert out["rule_applied"] == "standard_window"


def _decide(**kw):
    base = dict(
        category="Laptops & Computers",
        order_status="delivered",
        days_since_delivery=5,
        is_premium=False,
        existing_return_statuses=[],
    )
    base.update(kw)
    return pol.evaluate_return(**base)


def test_evaluate_return_eligible():
    d = _decide()
    assert d.eligible is True
    assert d.reason_code == "ELIGIBLE"
    assert d.window_days == 30
    assert d.standard_window_days == 30
    assert d.days_remaining == 25
    assert d.days_since_delivery == 5
    assert d.premium_extension_applied is False


def test_evaluate_return_premium_fields():
    d = _decide(is_premium=True, days_since_delivery=40)
    assert d.eligible is True
    assert d.window_days == 45
    assert d.standard_window_days == 30
    assert d.premium_extension_applied is True
    assert d.days_remaining == 5


def test_evaluate_return_unknown_category():
    d = _decide(category="Unicorns")
    assert (d.eligible, d.reason_code) == (False, "UNKNOWN_CATEGORY")
    assert d.window_days is None and d.standard_window_days is None


def test_evaluate_return_non_returnable():
    d = _decide(category="Software & Digital Codes")
    assert (d.eligible, d.reason_code) == (False, "NON_RETURNABLE")
    assert d.standard_window_days is None


@pytest.mark.parametrize("status", ["shipped", "processing", "cancelled", "Delivered", ""])
def test_evaluate_return_requires_delivered_status(status):
    d = _decide(order_status=status)
    assert (d.eligible, d.reason_code) == (False, "ORDER_NOT_DELIVERED")
    assert d.standard_window_days == 30


@pytest.mark.parametrize("statuses", [["pending"], ["approved"], ["refunded"],
                                      ["rejected", "pending"]])
def test_evaluate_return_blocked_by_active_return(statuses):
    d = _decide(existing_return_statuses=statuses)
    assert (d.eligible, d.reason_code) == (False, "ALREADY_RETURNED")


@pytest.mark.parametrize("statuses", [[], ["rejected"], ["rejected", "rejected"]])
def test_evaluate_return_rejected_return_does_not_block(statuses):
    assert _decide(existing_return_statuses=statuses).eligible is True


def test_evaluate_return_missing_delivery_date():
    d = _decide(days_since_delivery=None)
    assert (d.eligible, d.reason_code) == (False, "ORDER_NOT_DELIVERED")
    assert "No delivery date on record" in d.message
    assert d.window_days == 30


def test_evaluate_return_non_returnable_checked_before_status():
    d = _decide(category="Gift Cards", order_status="shipped")
    assert d.reason_code == "NON_RETURNABLE"


def test_evaluate_return_status_checked_before_window():
    d = _decide(order_status="shipped", days_since_delivery=999)
    assert d.reason_code == "ORDER_NOT_DELIVERED"


@pytest.mark.parametrize(
    "days,eligible,remaining",
    [(0, True, 30), (29, True, 1), (30, True, 0), (31, False, -1), (365, False, -335)],
)
def test_evaluate_return_window_boundary(days, eligible, remaining):
    d = _decide(days_since_delivery=days)
    assert d.eligible is eligible
    assert d.days_remaining == remaining
    if not eligible:
        assert d.reason_code == "RETURN_WINDOW_EXPIRED"
        assert f"closed {-remaining} day(s) ago" in d.message


def test_evaluate_return_holiday_extension_rescues_expired_window():
    # Delivered Dec 5, 2024; the 15-day window closed Dec 20 but the holiday rule
    # keeps the order returnable until Jan 31, 2025.
    d = _decide(
        category="Smartphones & Tablets",
        days_since_delivery=40,
        purchase_date=date(2024, 12, 1),
        delivery_date=date(2024, 12, 5),
        as_of=date(2025, 1, 14),
    )
    assert d.eligible is True
    assert d.reason_code == "ELIGIBLE"
    assert d.days_remaining == 17


def test_evaluate_return_holiday_extension_needs_as_of_and_delivery_date():
    # Without as_of/delivery_date the holiday rule cannot be applied, so a
    # holiday purchase past its standard window reads as expired.
    d = _decide(
        category="Smartphones & Tablets",
        days_since_delivery=40,
        purchase_date=date(2024, 12, 1),
    )
    assert d.eligible is False
    assert d.reason_code == "RETURN_WINDOW_EXPIRED"


def test_evaluate_return_holiday_extension_not_applied_when_standard_is_later():
    d = _decide(
        category="Cables & Adapters",
        is_premium=True,
        days_since_delivery=10,
        purchase_date=date(2024, 12, 31),
        delivery_date=date(2024, 12, 31),
        as_of=date(2025, 1, 10),
    )
    assert d.eligible is True
    assert d.days_remaining == 50  # standard arithmetic, not the Jan 31 date


def test_evaluate_return_past_holiday_deadline_expires():
    d = _decide(
        category="Smartphones & Tablets",
        days_since_delivery=100,
        purchase_date=date(2024, 12, 1),
        delivery_date=date(2024, 12, 5),
        as_of=date(2025, 2, 10),
    )
    assert d.eligible is False
    assert d.reason_code == "RETURN_WINDOW_EXPIRED"
    assert d.days_remaining == -10


def test_evaluate_return_is_keyword_only():
    with pytest.raises(TypeError):
        pol.evaluate_return("Laptops & Computers", "delivered", 1, False, [])


def test_category_table_is_coherent_with_evaluate_return():
    for category, window in pol.CATEGORY_WINDOW_DAYS.items():
        d = _decide(category=category, days_since_delivery=0)
        if window is None:
            assert d.reason_code == "NON_RETURNABLE"
        else:
            assert d.eligible is True
            assert d.window_days == window


# ════════════════════════════════════════════════════════════════════════════
# chunker — parse_pdf against the local corpus (no network, pdfplumber only)
# ════════════════════════════════════════════════════════════════════════════

_PDF = ch.Path(__file__).resolve().parents[2] / "data" / "raw" / "policies" / "return_policy.pdf"
_needs_pdf = pytest.mark.skipif(not _PDF.exists(), reason="return_policy.pdf not in corpus")


@_needs_pdf
def test_parse_pdf_recovers_numbered_headings_and_tables():
    blocks = ch.parse_pdf(_PDF)
    assert blocks
    headings = [b.heading for b in blocks if b.block_type == "heading"]
    assert any(h.startswith("1.1 ") for h in headings)
    tables = [b for b in blocks if b.block_type == "table" and b.rows]
    assert tables
    first = tables[0]
    assert first.rows[0] == ["Product Category", "Return Window", "Condition Required"]
    assert first.heading.startswith("1.1 ")
    # Every row has the same width as the header: column structure survived.
    assert {len(r) for r in first.rows} == {3}


@_needs_pdf
def test_parse_pdf_table_text_is_not_repeated_as_prose():
    blocks = ch.parse_pdf(_PDF)
    prose = "\n".join(b.text for b in blocks if b.block_type != "table")
    assert "Original packaging, all accessories" not in prose


@_needs_pdf
def test_chunk_document_pdf_emits_row_chunks_matching_the_policy_table():
    chunks = ch.chunk_document(_PDF, source_type="pdf")
    rows = [c for c in chunks if c.chunk_type == "table_row"]
    assert rows
    windows = {
        c.row["Product Category"]: c.row["Return Window"]
        for c in rows
        if c.row and "Product Category" in c.row
    }
    assert windows["Laptops & Computers"] == "30 days"
    assert windows["Gift Cards"] == "Non-returnable"
    assert all(c.source_file == "return_policy.pdf" for c in chunks)
    assert all(c.source_type == "pdf" for c in chunks)


@_needs_pdf
def test_pdf_category_table_agrees_with_policies_module():
    rows = [c for c in ch.chunk_document(_PDF, source_type="pdf")
            if c.chunk_type == "table_row" and c.row and "Product Category" in c.row]
    documented = {c.row["Product Category"]: c.row["Return Window"] for c in rows}
    for category, days in pol.CATEGORY_WINDOW_DAYS.items():
        if category not in documented:
            continue
        expected = "Non-returnable" if days is None else f"{days} days"
        assert documented[category] == expected, category


def test_parse_pdf_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ch.parse_pdf(tmp_path / "nope.pdf")
