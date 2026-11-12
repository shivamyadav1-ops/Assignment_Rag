"""
Convert Markdown policy documents to PDF using reportlab.

Usage:
    python scripts/convert_policy_to_pdf.py                 # convert the configured set
    python scripts/convert_policy_to_pdf.py foo.md bar.md   # convert specific files

The PDF becomes the ingested artifact; the source .md is then excluded from
ingestion (see POLICY_SOURCES in chunker.py) so the corpus holds no duplicates.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "raw" / "policies"

# Markdown documents that are published as PDF
DEFAULT_TARGETS = [
    "return_policy.md",
    "price_match_policy.md",
    "troubleshooting_guide.md",
]


def convert(md_file: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    pdf_file = md_file.with_suffix(".pdf")
    text = md_file.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(pdf_file), pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("DocH1", parent=styles["Heading1"], fontSize=18, spaceAfter=6))
    styles.add(ParagraphStyle("DocH2", parent=styles["Heading2"], fontSize=14, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle("DocH3", parent=styles["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle("DocBody", parent=styles["BodyText"], fontSize=10, leading=13))
    styles.add(ParagraphStyle("DocBullet", parent=styles["BodyText"], fontSize=10, leftIndent=20, leading=13))
    styles.add(ParagraphStyle("DocNote", parent=styles["BodyText"], alignment=TA_CENTER,
                              fontSize=8, textColor=colors.grey))

    def bold(s: str) -> str:
        while "**" in s:
            s = s.replace("**", "<b>", 1).replace("**", "</b>", 1)
        return s

    elements: list = []
    table_rows: list[list[str]] = []

    def flush_table():
        nonlocal table_rows
        if not table_rows:
            return
        width = len(table_rows[0])
        rows = [r + [""] * (width - len(r)) if len(r) < width else r[:width]
                for r in table_rows]
        col_w = (letter[0] - 1.5 * inch) / width
        cells = [[Paragraph(c, styles["DocBody"]) for c in r] for r in rows]
        t = Table(cells, colWidths=[col_w] * width, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f6fa")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 8))
        table_rows = []

    for line in text.split("\n"):
        s = line.strip()

        if s.startswith("|") and set(s) <= set("|-: "):
            continue                                   # table separator row
        if s.startswith("|"):
            table_rows.append([c.strip().replace("**", "").replace("✅", "Yes").replace("❌", "No")
                               for c in s.split("|")[1:-1]])
            continue

        flush_table()

        if s.startswith("[SYNTHETIC"):
            elements.append(Paragraph(s, styles["DocNote"]))
        elif s.startswith("# "):
            elements.append(Paragraph(bold(s[2:]), styles["DocH1"]))
        elif s.startswith("## "):
            elements.append(Paragraph(bold(s[3:]), styles["DocH2"]))
        elif s.startswith("### "):
            elements.append(Paragraph(bold(s[4:]), styles["DocH3"]))
        elif s.startswith("---"):
            elements.append(Spacer(1, 6))
        elif s.startswith("> "):
            elements.append(Paragraph(bold(s[2:]), styles["DocBullet"]))
        elif s.startswith(("- ", "* ")):
            elements.append(Paragraph("• " + bold(s[2:]), styles["DocBullet"]))
        elif s and s[0].isdigit() and "." in s[:3]:
            elements.append(Paragraph(bold(s), styles["DocBullet"]))
        elif not s:
            elements.append(Spacer(1, 4))
        else:
            elements.append(Paragraph(bold(s), styles["DocBody"]))

    flush_table()
    doc.build(elements)
    return pdf_file


def main():
    targets = sys.argv[1:] or DEFAULT_TARGETS
    print("Converting Markdown policies to PDF\n")
    for name in targets:
        md_file = BASE / name if not Path(name).is_absolute() else Path(name)
        if not md_file.exists():
            print(f"  SKIP {name} (not found)")
            continue
        pdf = convert(md_file)
        print(f"  {md_file.name:34} -> {pdf.name:34} ({pdf.stat().st_size / 1024:6.1f} KB)")
    print("\nDone.")


if __name__ == "__main__":
    main()
