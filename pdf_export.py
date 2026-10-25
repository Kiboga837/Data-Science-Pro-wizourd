"""PDF export for reports.

Takes the same list of blocks the main report uses (headings, text, bullets, tables, images, code, callouts) and
produces a paginated A4 PDF with a title page, contents list, repeating table headers and page numbers.
Needs the `reportlab` package (pure Python, no system libraries).
"""
import textwrap
from io import BytesIO
from xml.sax.saxutils import escape


def pdf_available() -> bool:
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


_UNICODE = False  # True when the DejaVu fonts (wide Unicode coverage) were registered for this build


def _fonts() -> dict:
    """Register DejaVu Sans (bundled with matplotlib) for Greek, Cyrillic and math symbols.
    Falls back to the built-in Helvetica/Courier, which only cover basic Latin."""
    global _UNICODE
    try:
        from pathlib import Path

        from matplotlib import get_data_path
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        folder = Path(get_data_path()) / "fonts" / "ttf"
        for name, file in {"DejaVu": "DejaVuSans.ttf", "DejaVu-Bold": "DejaVuSans-Bold.ttf",
                           "DejaVu-Italic": "DejaVuSans-Oblique.ttf", "DejaVu-Mono": "DejaVuSansMono.ttf"}.items():
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(folder / file)))
        _UNICODE = True
        return {"regular": "DejaVu", "bold": "DejaVu-Bold", "italic": "DejaVu-Italic", "mono": "DejaVu-Mono"}
    except Exception:  # noqa: BLE001
        _UNICODE = False
        return {"regular": "Helvetica", "bold": "Helvetica-Bold", "italic": "Helvetica-Oblique", "mono": "Courier"}


def _text(value) -> str:
    """Escape markup; without Unicode fonts also replace characters the built-in fonts cannot show."""
    out = escape(str(value))
    return out if _UNICODE else out.encode("cp1252", "replace").decode("cp1252")


def _wrap_code(code: str, width: int = 105) -> str:
    lines = []
    for line in str(code).expandtabs(4).splitlines() or [""]:
        indent = len(line) - len(line.lstrip())
        lines += textwrap.wrap(line, width, subsequent_indent=" " * (indent + 2), drop_whitespace=False,
                               replace_whitespace=False) or [""]
    joined = "\n".join(lines)
    return joined if _UNICODE else joined.encode("cp1252", "replace").decode("cp1252")


def to_pdf(blocks: list, meta: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (CondPageBreak, Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    f = _fonts()
    blue, light = colors.HexColor("#1f4e79"), colors.HexColor("#f5f8fb")
    base = getSampleStyleSheet()["BodyText"]
    body = ParagraphStyle("body", parent=base, fontName=f["regular"], fontSize=9.5, leading=13, spaceAfter=5)
    h1 = ParagraphStyle("h1", parent=body, fontName=f["bold"], fontSize=16, leading=20, textColor=blue,
                        spaceBefore=10, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=body, fontName=f["bold"], fontSize=12.5, leading=16,
                        textColor=colors.HexColor("#2f6f9f"), spaceBefore=8, spaceAfter=4)
    h3 = ParagraphStyle("h3", parent=body, fontName=f["bold"], fontSize=10.5, spaceBefore=5, spaceAfter=3)
    cap = ParagraphStyle("cap", parent=body, fontName=f["italic"], fontSize=8, textColor=colors.grey, spaceAfter=9)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.5, leading=9.5, spaceAfter=0)
    cellh = ParagraphStyle("cellh", parent=cell, fontName=f["bold"], textColor=colors.white)
    code_style = ParagraphStyle("code", parent=body, fontName=f["mono"], fontSize=7, leading=8.6, backColor=colors.HexColor("#f4f4f4"),
                                borderColor=colors.HexColor("#dddddd"), borderWidth=0.5, borderPadding=4, spaceAfter=8)
    title_style = ParagraphStyle("title", parent=h1, fontSize=26, leading=30, spaceAfter=10)

    margin = 1.8 * cm
    avail = A4[0] - 2 * margin
    story = []

    # ---- title page ---------------------------------------------------------
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph(_text(meta.get("title", "Report")), title_style))
    sub = " | ".join(str(x) for x in [meta.get("author"), meta.get("date"), f"Dataset: {meta['dataset']}" if meta.get("dataset") else ""] if x)
    story.append(Paragraph(_text(sub), body))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("Contents", h2))
    for i, b in enumerate([b for b in blocks if b[0] == "h1"], 1):
        story.append(Paragraph(f"{i}. {_text(b[1])}", body))
    story.append(PageBreak())

    # ---- content ------------------------------------------------------------
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            story.append(CondPageBreak(5 * cm))
            story.append(Paragraph(_text(b[1]), h1))
        elif kind == "h2":
            story.append(CondPageBreak(3 * cm))
            story.append(Paragraph(_text(b[1]), h2))
        elif kind == "h3":
            story.append(Paragraph(_text(b[1]), h3))
        elif kind == "p":
            story.append(Paragraph(_text(b[1]).replace("\n", "<br/>"), body))
        elif kind == "bullets":
            for item in b[1]:
                story.append(Paragraph(_text(item), body, bulletText="\u2022"))
        elif kind == "callout":
            box = Table([[Paragraph(_text(b[1]), body)]], colWidths=[avail])
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8e1")),
                                     ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#f0b400")),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story += [box, Spacer(1, 6)]
        elif kind == "code":
            story.append(Preformatted(_wrap_code(b[1]), code_style))
        elif kind == "table":
            df = b[1]
            if df is None or df.empty:
                continue
            header = [Paragraph(_text(c), cellh) for c in df.columns]
            rows = [[Paragraph(_text(v), cell) for v in row] for row in df.itertuples(index=False)]
            weights = [min(max(len(str(c)), *(len(str(v)) for v in df[c])) + 2, 32) for c in df.columns]
            widths = [avail * w / sum(weights) for w in weights]
            table = Table([header] + rows, colWidths=widths, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f6f9f")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            story.append(table)
            story.append(Paragraph(_text(b[2]) if b[2] else "", cap))
        elif kind == "image":
            width_px, height_px = ImageReader(BytesIO(b[1])).getSize()
            scale = min(avail / width_px, 17 * cm / height_px)
            story.append(Image(BytesIO(b[1]), width=width_px * scale, height=height_px * scale))
            story.append(Paragraph(_text(b[2]) if b[2] else "", cap))

    title = _text(meta.get("title", "Report"))[:70]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(f["regular"], 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(margin, 1.0 * cm, title)
        canvas.drawRightString(A4[0] - margin, 1.0 * cm, f"Page {document.page}")
        canvas.restoreState()

    out = BytesIO()
    SimpleDocTemplate(out, pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                      title=str(meta.get("title", "Report")), author=str(meta.get("author", ""))
                      ).build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()
