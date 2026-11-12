"""
Structure-aware recursive chunker with overlap, plus row-level table chunks.

Strategy:
  1. Parse each document into structural blocks (headings, paragraphs, tables, lists)
  2. Respect boundaries: never split mid-list-item or mid-sentence
  3. Merge small consecutive prose blocks until they approach CHUNK_SIZE
  4. When a block exceeds CHUNK_SIZE, split at sentence boundaries
  5. Add CHUNK_OVERLAP tokens of trailing context from the previous chunk
  6. Tables are taken out of the prose stream and emitted as:
       - one chunk per row, rendered as "Column: value" lines under a context line
         naming the document, section and table, with the row as a JSON dict in
         metadata (`row`), and
       - one whole-table chunk, for questions that span several rows.
     A fact in a table row then competes for rank as a short, single-fact chunk
     instead of one line inside a long chunk full of other rows and prose.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from retail_support_agent.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE
from retail_support_agent.provenance import lookup

# ── Token estimation ──────────────────────────────────────────────────────────
# Approximate: 1 token ≈ 4 characters for English text.
# Good enough for chunking; exact counts happen at the LLM/embedding stage.

def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def estimate_chars(tokens: int) -> int:
    """Convert token count back to approximate character count."""
    return tokens * 4


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StructuralBlock:
    """A semantically meaningful unit from the document."""
    text: str
    block_type: str   # "heading", "paragraph", "table", "list", "meta"
    heading: str = ""  # nearest parent heading
    level: int = 0     # heading level (1, 2, 3) or 0 for non-headings
    rows: list[list[str]] | None = None  # tables only: header row first


@dataclass
class Chunk:
    """A chunk ready for embedding and storage."""
    chunk_id: str
    text: str
    source_file: str
    source_type: str          # "pdf", "html", "md"
    chunk_index: int
    total_chunks: int = 0     # filled after all chunks generated
    section_heading: str = ""
    page_number: int | None = None
    has_table: bool = False
    token_count: int = 0
    synthetic: bool = False
    chunk_type: str = "text"  # "text" | "table" | "table_row"
    table_id: str | None = None
    row_index: int | None = None       # 1-based data-row number within its table
    row: dict | None = None            # table_row only: {column header: cell value}
    origin: str = "synthetic"          # "synthetic" | "public" (from data/sources.json)
    license: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Table helpers ─────────────────────────────────────────────────────────────

def _clean_cell(value: str | None) -> str:
    """Collapse wrapped lines and Markdown emphasis inside one table cell."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.replace("**", "").replace("|", "/")


_SEPARATOR_ROW = re.compile(r"^\|?[\s:\-|]+\|?$")


def _parse_pipe_rows(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        s = line.strip()
        if _SEPARATOR_ROW.match(s) and "-" in s:
            continue
        rows.append([_clean_cell(c) for c in s.strip("|").split("|")])
    return [r for r in rows if any(r)]


def _rows_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(padded[0]) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in padded[1:]]
    return "\n".join(lines)


# ── Parsers ───────────────────────────────────────────────────────────────────

# PDF text extraction strips Markdown "#" markers, so headings arrive as bare
# numbered lines ("1.3 Holiday Return Policy"). Detect those structurally.
NUMBERED_HEADING = re.compile(r'^(\d+(?:\.\d+)*)\.?\s+([A-Z][^.!?]{3,70})$')


def _detect_bare_heading(line: str) -> tuple[int, str] | None:
    """Return (level, text) if a non-Markdown line looks like a heading."""
    m = NUMBERED_HEADING.match(line)
    if not m:
        return None
    number, title = m.group(1), m.group(2).strip()
    # "1." -> level 1, "1.1" -> level 2, "1.1.1" -> level 3
    level = number.count(".") + 1
    return level, f"{number} {title}"


def parse_markdown(text: str) -> list[StructuralBlock]:
    """Parse Markdown into structural blocks."""
    blocks: list[StructuralBlock] = []
    current_heading = ""
    accumulator: list[str] = []
    in_table = False
    table_lines: list[str] = []

    def flush_accumulator():
        nonlocal accumulator
        if accumulator:
            joined = "\n".join(accumulator).strip()
            if joined:
                blocks.append(StructuralBlock(
                    text=joined, block_type="paragraph", heading=current_heading
                ))
            accumulator = []

    def flush_table():
        nonlocal table_lines, in_table
        if table_lines:
            rows = _parse_pipe_rows(table_lines)
            blocks.append(StructuralBlock(
                text="\n".join(table_lines), block_type="table", heading=current_heading,
                rows=rows if len(rows) >= 2 else None,
            ))
            table_lines = []
        in_table = False

    for line in text.split("\n"):
        stripped = line.strip()

        # Table detection
        if stripped.startswith("|") and "|" in stripped[1:]:
            if not in_table:
                flush_accumulator()
                in_table = True
            table_lines.append(stripped)
            continue
        elif in_table:
            flush_table()

        # Headings
        heading_match = re.match(r'^(#{1,4})\s+(.+)$', stripped)
        if heading_match:
            flush_accumulator()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            current_heading = heading_text
            blocks.append(StructuralBlock(
                text=stripped, block_type="heading",
                heading=heading_text, level=level
            ))
            continue

        # Horizontal rules
        if re.match(r'^-{3,}$', stripped) or re.match(r'^\*{3,}$', stripped):
            flush_accumulator()
            continue

        # Bare numbered headings (PDF-extracted text has no "#" markers)
        bare = _detect_bare_heading(stripped)
        if bare:
            level, heading_text = bare
            flush_accumulator()
            current_heading = heading_text
            blocks.append(StructuralBlock(
                text=stripped, block_type="heading",
                heading=heading_text, level=level
            ))
            continue

        # List items — keep together with accumulated text
        if re.match(r'^[-*]\s+', stripped) or re.match(r'^\d+\.\s+', stripped):
            accumulator.append(stripped)
            continue

        # Empty line — flush paragraph
        if not stripped:
            flush_accumulator()
            continue

        # Regular text
        accumulator.append(stripped)

    # Final flush
    flush_accumulator()
    flush_table()

    return blocks


def _inline_text(element) -> str:
    """Element text with inline tags joined by spaces, as a browser renders them.

    get_text(strip=True) strips every text node and joins them with nothing, so
    "within <strong>1 hour</strong> of" became "within1 hourof", which breaks BM25
    matching and the text the model reads. Join with spaces instead, then remove
    the spaces that creates before punctuation and inside brackets.
    """
    text = re.sub(r"\s+", " ", element.get_text(" ", strip=True))
    text = re.sub(r"\s+([.,;:!?)\]])", r"\1", text)
    return re.sub(r"([(\[])\s+", r"\1", text).strip()


def parse_html(html_content: str) -> list[StructuralBlock]:
    """Parse HTML into structural blocks using BeautifulSoup."""
    import html2text
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "lxml")

    # Remove nav, footer, script, style
    for tag in soup.find_all(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    blocks: list[StructuralBlock] = []
    current_heading = ""

    main = soup.find("main") or soup.find("body") or soup

    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "table", "section"]):
        tag_name = element.name

        if tag_name in ("h1", "h2", "h3", "h4"):
            heading_text = _inline_text(element)
            current_heading = heading_text
            level = int(tag_name[1])
            blocks.append(StructuralBlock(
                text=heading_text, block_type="heading",
                heading=heading_text, level=level
            ))

        elif tag_name == "table":
            # Read cells straight from the DOM so row/column structure survives.
            rows = []
            for tr in element.find_all("tr"):
                cells = [_clean_cell(td.get_text(" ", strip=True))
                         for td in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(cells)
            if len(rows) >= 2:
                blocks.append(StructuralBlock(
                    text=_rows_to_markdown(rows), block_type="table",
                    heading=current_heading, rows=rows,
                ))
            else:
                h = html2text.HTML2Text()
                h.body_width = 0
                h.ignore_links = True
                table_md = h.handle(str(element)).strip()
                if table_md:
                    blocks.append(StructuralBlock(
                        text=table_md, block_type="table", heading=current_heading
                    ))

        elif tag_name in ("ul", "ol"):
            # Only process if not nested inside another list we already captured
            if not element.find_parent(["ul", "ol"]):
                items = []
                for li in element.find_all("li", recursive=False):
                    items.append(f"- {_inline_text(li)}")
                if items:
                    blocks.append(StructuralBlock(
                        text="\n".join(items), block_type="list", heading=current_heading
                    ))

        elif tag_name == "p":
            text = _inline_text(element)
            if text and len(text) > 10:  # skip tiny fragments
                blocks.append(StructuralBlock(
                    text=text, block_type="paragraph", heading=current_heading
                ))

        elif tag_name == "section":
            # Sections are containers; their children are already processed
            continue

    return blocks


def parse_ecfr_xml(xml_content: str) -> list[StructuralBlock]:
    """Parse an eCFR regulation (the ecfr.gov versioner XML) into structural blocks.

    The part heading (DIV5/HEAD) becomes a level-1 heading and each section
    (DIV8/HEAD, e.g. "§ 435.2 ...") a level-2 heading, so the chunker starts a new
    chunk at every section and each chunk's heading carries the citation. Body
    paragraphs are <P> elements; inline markup such as <I> is flattened.
    Authority/source notes (AUTH, SOURCE, CITA) are kept as one metadata paragraph
    under the part heading so lineage is searchable but does not mix into sections.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(xml_content, "xml")
    blocks: list[StructuralBlock] = []

    text_of = _inline_text

    part = soup.find("DIV5")
    if part is None:
        return blocks
    part_head = part.find("HEAD", recursive=False)
    part_title = text_of(part_head) if part_head else "Regulation"
    blocks.append(StructuralBlock(text=part_title, block_type="heading",
                                  heading=part_title, level=1))
    notes = [text_of(el) for el in part.find_all(["AUTH", "SOURCE"], recursive=False)]
    if notes:
        blocks.append(StructuralBlock(text=" ".join(notes), block_type="paragraph",
                                      heading=part_title))

    for section in part.find_all("DIV8"):
        head = section.find("HEAD", recursive=False)
        heading = text_of(head) if head else part_title
        blocks.append(StructuralBlock(text=heading, block_type="heading",
                                      heading=heading, level=2))
        for p in section.find_all("P"):
            text = text_of(p)
            if len(text) > 10:
                blocks.append(StructuralBlock(text=text, block_type="paragraph",
                                              heading=heading))
    return blocks


# Vertical gap (points) between consecutive PDF text lines that marks a paragraph break.
PDF_PARAGRAPH_GAP = 4.0


def parse_pdf(pdf_path: Path) -> list[StructuralBlock]:
    """Parse a PDF with pdfplumber, keeping each table at its position on the page.

    Text lines that fall inside a detected table's bounding box are dropped from the
    prose; the table itself is re-inserted at the same position as Markdown pipe rows
    (wrapped cells collapsed to one line). Without this, the plain-text layer repeats
    every table as one cell per line with no column structure, and the structured copy
    would end up detached from its section.
    """
    import pdfplumber

    lines_out: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            tables = page.find_tables()
            boxes = [t.bbox for t in tables]

            def in_table(line: dict, boxes=boxes) -> bool:
                mid_y = (line["top"] + line["bottom"]) / 2
                return any(x0 - 2 <= line["x0"] and line["x1"] <= x1 + 2 and top <= mid_y <= bottom
                           for x0, top, x1, bottom in boxes)

            items: list[tuple[float, str, object]] = [
                (ln["top"], "line", ln) for ln in page.extract_text_lines() if not in_table(ln)
            ]
            items += [(t.bbox[1], "table", t.extract()) for t in tables]
            items.sort(key=lambda item: item[0])

            last_bottom: float | None = None
            for _, kind, payload in items:
                if kind == "line":
                    if last_bottom is not None and payload["top"] - last_bottom > PDF_PARAGRAPH_GAP:
                        lines_out.append("")
                    lines_out.append(payload["text"])
                    last_bottom = payload["bottom"]
                else:
                    rows = [[_clean_cell(c) for c in row] for row in payload]
                    rows = [r for r in rows if any(r)]
                    if len(rows) >= 2:
                        lines_out.append("")
                        lines_out.append(_rows_to_markdown(rows))
                        lines_out.append("")
                    last_bottom = None
            lines_out.append("")

    return parse_markdown("\n".join(lines_out))


# ── Sentence splitter ─────────────────────────────────────────────────────────

# A boundary is terminal punctuation + whitespace + the start of a new word, in
# either case: requiring a capital meant prose whose sentences start lowercase
# could never be split, leaving oversized chunks far above CHUNK_SIZE.
SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Za-z0-9"\'(\[])')

# Abbreviations whose trailing period is not a sentence end. Accepting lowercase
# starts above would otherwise split "e.g. items ship free" mid-sentence.
_ABBREVIATIONS = re.compile(
    r'(?:\b(?:e\.g|i\.e|etc|vs|approx|incl|excl|no|nos|dept|est|fig|min|max'
    r'|Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Inc|Ltd|Co|Corp|U\.S|a\.m|p\.m)\.)$',
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping each sentence intact."""
    pieces = [s.strip() for s in SENTENCE_ENDINGS.split(text) if s.strip()]
    sentences: list[str] = []
    for piece in pieces:
        if sentences and _ABBREVIATIONS.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


# ── Core chunking logic ──────────────────────────────────────────────────────

def merge_blocks_into_chunks(
    blocks: list[StructuralBlock],
    max_tokens: int = CHUNK_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP,
    min_tokens: int = MIN_CHUNK_SIZE,
) -> list[dict]:
    """
    Merge structural blocks into overlapping chunks.

    Rules:
      1. Tables are kept as single chunks (never split mid-table)
      2. Headings are attached to the next block
      3. Consecutive small blocks are merged up to max_tokens
      4. Large blocks are split at sentence boundaries
      5. Each chunk gets overlap_tokens from the end of the previous chunk
    """
    raw_chunks: list[dict] = []  # {"text": str, "heading": str, "has_table": bool}

    # Buffer accumulates blocks until adding the next one would exceed max_tokens.
    # Headings do NOT force a flush — they are packed in with the content that
    # follows them, so a chunk can span several short subsections.
    buffer_parts: list[str] = []
    buffer_tokens = 0
    buffer_heading = ""      # heading active when the buffer started
    buffer_has_table = False
    buffer_section_start = False  # buffer opens with an H1/H2 heading
    current_heading = ""     # most recent heading seen in the stream

    def flush_buffer() -> None:
        nonlocal buffer_parts, buffer_tokens, buffer_heading, buffer_has_table
        nonlocal buffer_section_start
        text = "\n\n".join(buffer_parts).strip()
        if text and estimate_tokens(text) >= min_tokens:
            raw_chunks.append({
                "text": text,
                "heading": buffer_heading or current_heading,
                "has_table": buffer_has_table,
                "section_start": buffer_section_start,
            })
        elif text and raw_chunks:
            # Too small to stand alone — append to the previous chunk rather
            # than emitting a fragment or dropping content.
            raw_chunks[-1]["text"] += "\n\n" + text
            raw_chunks[-1]["has_table"] = raw_chunks[-1]["has_table"] or buffer_has_table
        elif text:
            # Nothing to append to (e.g. a whole document shorter than
            # min_tokens): keep it as an under-sized chunk. Dropping it would
            # leave the document unindexed and unretrievable.
            raw_chunks.append({
                "text": text,
                "heading": buffer_heading or current_heading,
                "has_table": buffer_has_table,
                "section_start": buffer_section_start,
            })
        buffer_parts = []
        buffer_tokens = 0
        buffer_heading = ""
        buffer_has_table = False
        buffer_section_start = False

    def add_to_buffer(text: str, is_table: bool = False,
                      section_start: bool = False) -> None:
        nonlocal buffer_tokens, buffer_heading, buffer_has_table, buffer_section_start
        if not buffer_parts:
            buffer_heading = current_heading
            buffer_section_start = section_start
        buffer_parts.append(text)
        buffer_tokens += estimate_tokens(text)
        buffer_has_table = buffer_has_table or is_table

    for block in blocks:
        block_tokens = estimate_tokens(block.text)

        # Headings: update context; flush at major section boundaries (H1/H2)
        # so that distinct policy sections never merge into one chunk.  Minor
        # headings (H3+) are packed in with the content that follows them.
        if block.block_type == "heading":
            if block.level <= 2 and buffer_parts:
                flush_buffer()
            elif buffer_tokens + block_tokens > max_tokens:
                flush_buffer()
            current_heading = block.heading
            add_to_buffer(block.text, section_start=block.level <= 2)
            continue

        # Tables: never split. Pack whole if it fits, else give it its own chunk.
        if block.block_type == "table":
            if buffer_tokens + block_tokens <= max_tokens:
                add_to_buffer(block.text, is_table=True)
            else:
                flush_buffer()
                if block_tokens > max_tokens:
                    # Oversized table stands alone, intact, even past the target.
                    heading_prefix = f"{current_heading}\n\n" if current_heading else ""
                    raw_chunks.append({
                        "text": heading_prefix + block.text,
                        "heading": current_heading,
                        "has_table": True,
                    })
                else:
                    add_to_buffer(block.text, is_table=True)
            continue

        # Regular content that fits.
        if buffer_tokens + block_tokens <= max_tokens:
            add_to_buffer(block.text)
            continue

        # Doesn't fit: flush, then handle the block on its own.
        flush_buffer()

        if block_tokens <= max_tokens:
            add_to_buffer(block.text)
            continue

        # Oversized block: split at sentence boundaries, never mid-sentence.
        sentence_buffer: list[str] = []
        sentence_tokens = 0
        for sentence in split_sentences(block.text):
            s_tokens = estimate_tokens(sentence)
            if sentence_tokens + s_tokens > max_tokens and sentence_buffer:
                raw_chunks.append({
                    "text": " ".join(sentence_buffer).strip(),
                    "heading": current_heading,
                    "has_table": False,
                })
                sentence_buffer = [sentence]
                sentence_tokens = s_tokens
            else:
                sentence_buffer.append(sentence)
                sentence_tokens += s_tokens
        # Trailing sentences continue into the buffer so the next block can pack in.
        if sentence_buffer:
            add_to_buffer(" ".join(sentence_buffer).strip())

    flush_buffer()

    # ── Apply overlap ─────────────────────────────────────────────────────
    overlap_chars = estimate_chars(overlap_tokens)
    overlapped_chunks = []

    for i, chunk in enumerate(raw_chunks):
        if i == 0 or chunk.get("section_start"):
            # A new H1/H2 section starts clean: overlap only continues a section
            # that was split for size, never bleeds one section into the next.
            chunk.pop("section_start", None)
            overlapped_chunks.append(chunk)
            continue

        prev_text = raw_chunks[i - 1]["text"]
        # Take the last overlap_chars characters from previous chunk
        if overlap_chars <= 0:
            overlap_text = ""
        elif len(prev_text) > overlap_chars:
            overlap_text = "..." + prev_text[-overlap_chars:].lstrip()
            # Find the start of a sentence in the overlap
            sent_start = re.search(r'(?<=[.!?])\s+', overlap_text)
            if sent_start:
                overlap_text = "..." + overlap_text[sent_start.end():]
        else:
            overlap_text = prev_text

        chunk_with_overlap = {
            "text": overlap_text + "\n\n" + chunk["text"],
            "heading": chunk["heading"],
            "has_table": chunk["has_table"],
        }
        overlapped_chunks.append(chunk_with_overlap)

    return overlapped_chunks


# ── Table chunks ──────────────────────────────────────────────────────────────

def _document_title(file_path: Path, source_type: str, blocks: list[StructuralBlock]) -> str:
    for b in blocks:
        if b.block_type != "heading":
            continue
        if source_type == "md" and b.text.startswith("# "):
            return b.heading
        if source_type in ("html", "xml") and b.level == 1:
            return b.heading
    # PDF text carries no reliable title marker.
    return file_path.stem.replace("_", " ").title()


def _unique_headers(header: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, h in enumerate(header):
        name = h or f"Column {i + 1}"
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return out


def table_to_chunks(block: StructuralBlock, table_no: int, title: str,
                    source_stem: str) -> list[dict]:
    """One whole-table chunk plus one chunk per data row.

    Row text is "Column: value" lines rather than raw JSON: embedding and reranking
    models are trained on prose, and the context line keeps a bare row
    ("Canada / $19.99") tied to the document and section it came from. The row is
    also kept as a dict in metadata for exact lookups and citations.
    """
    header = _unique_headers(block.rows[0])
    data_rows = block.rows[1:]
    table_id = f"{source_stem}#table{table_no}"
    where = f"{title} — {block.heading}" if block.heading else title

    chunks = [{
        "text": f"[{where} — table {table_no}, {len(data_rows)} rows]\n"
                f"{_rows_to_markdown(block.rows)}",
        "heading": block.heading, "has_table": True,
        "chunk_type": "table", "table_id": table_id,
    }]
    for i, cells in enumerate(data_rows, 1):
        cells = cells + [""] * (len(header) - len(cells))
        row = dict(zip(header, cells, strict=False))
        lines = [f"[{where} — table {table_no}, row {i} of {len(data_rows)}]"]
        lines += [f"{col}: {val}" for col, val in row.items() if val]
        chunks.append({
            "text": "\n".join(lines),
            "heading": block.heading, "has_table": True,
            "chunk_type": "table_row", "table_id": table_id,
            "row_index": i, "row": row,
        })
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(
    file_path: Path,
    source_type: str | None = None,
    synthetic: bool = False,
) -> list[Chunk]:
    """
    Parse and chunk a document file.

    Args:
        file_path: Path to the document (PDF, HTML, or Markdown)
        source_type: Override format detection ("pdf", "html", "md")
        synthetic: Whether this is synthetic data

    Returns:
        List of Chunk objects ready for embedding
    """
    if source_type is None:
        ext = file_path.suffix.lower()
        source_type = {"pdf": "pdf", ".pdf": "pdf", ".html": "html", ".htm": "html",
                       ".xml": "xml", ".md": "md", ".markdown": "md"}.get(ext, "md")

    text = ""
    if source_type == "pdf":
        blocks = parse_pdf(file_path)
    elif source_type == "html":
        text = file_path.read_text(encoding="utf-8")
        blocks = parse_html(text)
    elif source_type == "xml":
        text = file_path.read_text(encoding="utf-8")
        blocks = parse_ecfr_xml(text)
    else:  # markdown
        text = file_path.read_text(encoding="utf-8")
        blocks = parse_markdown(text)

    if not blocks:
        return []

    # Structured tables leave the prose stream and become table + row chunks.
    table_blocks = [b for b in blocks if b.block_type == "table" and b.rows]
    prose_blocks = [b for b in blocks if not (b.block_type == "table" and b.rows)]

    raw_chunks = merge_blocks_into_chunks(prose_blocks)
    title = _document_title(file_path, source_type, blocks)
    for table_no, block in enumerate(table_blocks, 1):
        raw_chunks.extend(table_to_chunks(block, table_no, title, file_path.stem))

    chunks: list[Chunk] = []
    source_name = file_path.name

    for i, raw in enumerate(raw_chunks):
        chunk_text = raw["text"]
        chunk_id = hashlib.sha256(
            f"{source_name}:{i}:{chunk_text[:100]}".encode()
        ).hexdigest()[:16]

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            source_file=source_name,
            source_type=source_type,
            chunk_index=i,
            section_heading=raw.get("heading", ""),
            has_table=raw.get("has_table", False),
            token_count=estimate_tokens(chunk_text),
            synthetic=synthetic,
            chunk_type=raw.get("chunk_type", "text"),
            table_id=raw.get("table_id"),
            row_index=raw.get("row_index"),
            row=raw.get("row"),
        ))

    # Fill total_chunks
    for c in chunks:
        c.total_chunks = len(chunks)

    return chunks


# Markdown sources that are published as PDF and ingested in that form only.
# Ingesting both would put near-duplicate chunks in the corpus, which makes
# retrieval ground truth ambiguous and inflates evaluation scores.
EXCLUDED_SOURCES = {
    "return_policy.md",
    "price_match_policy.md",
    "troubleshooting_guide.md",
    "product_category_guide.md",
}

SUPPORTED_SUFFIXES = {".pdf": "pdf", ".html": "html", ".htm": "html", ".md": "md",
                      ".xml": "xml"}


def chunk_all_policies(policies_dir: Path | None = None, verbose: bool = True) -> list[Chunk]:
    """
    Chunk every supported policy document in the policies directory.

    Documents are discovered from disk rather than hardcoded, so adding a new
    policy file to data/raw/policies/ is enough to include it in the corpus.
    """
    from retail_support_agent.config import POLICIES_DIR
    policies_dir = policies_dir or POLICIES_DIR

    all_chunks: list[Chunk] = []

    discovered = sorted(
        p for p in policies_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and p.name not in EXCLUDED_SOURCES
    )

    if not discovered:
        raise FileNotFoundError(f"No policy documents found in {policies_dir}")

    for file_path in discovered:
        fmt = SUPPORTED_SUFFIXES[file_path.suffix.lower()]
        # Lineage comes from the provenance registry (data/sources.json), not the file.
        prov = lookup(file_path)
        chunks = chunk_document(file_path, source_type=fmt,
                                synthetic=bool(prov.get("synthetic", True)))
        for c in chunks:
            c.origin = prov.get("origin", "synthetic")
            c.license = prov.get("license")
        if verbose:
            avg = sum(c.token_count for c in chunks) // max(len(chunks), 1)
            print(f"  {file_path.name:32} {fmt:4}  {len(chunks):3} chunks  avg {avg:3} tokens")
        all_chunks.extend(chunks)

    return all_chunks
