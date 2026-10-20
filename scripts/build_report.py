"""Render the ten-section Markdown report to a paginated PDF."""

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report" / "assignment_report.md"
OUTPUT = ROOT / "report" / "assignment_report.pdf"


def _page_chrome(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5DEE5"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52616B"))
    canvas.drawString(18 * mm, 10 * mm, "Heart Disease Classification | MLOps Assignment 01")
    canvas.drawRightString(192 * mm, 10 * mm, str(document.page))
    canvas.restoreState()


def _styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
        fontSize=24, leading=29, alignment=TA_CENTER,
        textColor=colors.HexColor("#163B50"), spaceAfter=14,
    ))
    base.add(ParagraphStyle(
        name="Section", parent=base["Heading1"], fontName="Helvetica-Bold",
        fontSize=19, leading=23, textColor=colors.HexColor("#163B50"),
        spaceAfter=12,
    ))
    base.add(ParagraphStyle(
        name="Subsection", parent=base["Heading2"], fontSize=12, leading=15,
        textColor=colors.HexColor("#227C78"), spaceBefore=7, spaceAfter=5,
    ))
    base.add(ParagraphStyle(
        name="BodyReport", parent=base["BodyText"], fontSize=9.4, leading=13.2,
        textColor=colors.HexColor("#24333D"), spaceAfter=7,
    ))
    base.add(ParagraphStyle(
        name="BulletReport", parent=base["BodyText"], fontSize=9.1, leading=12.4,
        leftIndent=14, firstLineIndent=-9, bulletIndent=0, spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        name="MetaReport", parent=base["BodyText"], fontSize=11, leading=17,
        alignment=TA_CENTER, textColor=colors.HexColor("#52616B"), spaceAfter=5,
    ))
    base.add(ParagraphStyle(
        name="CodeReport", parent=base["Code"], fontName="Courier",
        fontSize=8, leading=10, backColor=colors.HexColor("#F0F4F4"),
        borderPadding=8, spaceBefore=4, spaceAfter=8,
    ))
    return base


def _inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text


def build():
    styles = _styles()
    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    in_code = False
    code_lines = []
    table_lines = []

    def flush_table():
        nonlocal table_lines
        if not table_lines:
            return
        rows = [[_inline(cell.strip()) for cell in line.strip().strip("|").split("|")] for line in table_lines]
        rows = [row for row in rows if not all(set(cell) <= {"-", ":", " "} for cell in row)]
        if rows:
            table = Table(rows, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163B50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD5D9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F4")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.extend([Spacer(1, 4), table, Spacer(1, 8)])
        table_lines = []

    for line in lines:
        if line.startswith("<!-- PAGE BREAK -->"):
            flush_table()
            story.append(PageBreak())
            continue
        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["CodeReport"]))
                code_lines = []
                in_code = False
            else:
                flush_table()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        if not line.strip():
            continue
        if line.startswith("# "):
            title = _inline(line[2:])
            if not story:
                story.extend([Spacer(1, 42 * mm), Paragraph(title, styles["ReportTitle"]), Spacer(1, 7 * mm)])
            else:
                story.append(Paragraph(title, styles["Section"]))
        elif line.startswith("## "):
            story.append(Paragraph(_inline(line[3:]), styles["Subsection"]))
        elif line.startswith("- "):
            story.append(Paragraph("• " + _inline(line[2:]), styles["BulletReport"]))
        elif re.match(r"\d+\. ", line):
            story.append(Paragraph(_inline(line), styles["BulletReport"]))
        elif line.startswith("**") and "  " in line:
            story.append(Paragraph(_inline(line), styles["MetaReport"]))
        else:
            story.append(Paragraph(_inline(line), styles["BodyReport"]))

        if line.startswith("The EDA script produces"):
            image_path = ROOT / "artifacts" / "eda" / "numeric_distributions.png"
            if image_path.exists():
                story.extend([Spacer(1, 5), Image(str(image_path), width=155 * mm, height=108 * mm), Spacer(1, 6)])

    flush_table()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=21 * mm,
        title="Heart Disease Classification: Reproducible MLOps Workflow",
        author="Student submission",
    )
    document.build(story, onFirstPage=_page_chrome, onLaterPages=_page_chrome)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()