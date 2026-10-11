"""Rebuild selectable-text workshop PDFs without adding app dependencies.

Run from the project folder:
    uv run --link-mode copy --with reportlab python demo/generate_samples.py

Expected outcomes stay in scenarios.json, never in the source documents.
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


PROJECT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#17344B")
TEAL = colors.HexColor("#137F82")
INK = colors.HexColor("#213747")
MUTED = colors.HexColor("#596D7B")
PALE = colors.HexColor("#EDF5F5")
LINE = colors.HexColor("#D8E4EA")


def paragraph(canvas, text, x, top, width, *, size=11, color=INK, bold=False):
    """Draw wrapped text with a top edge; return the next vertical position."""
    style = ParagraphStyle(
        "text", fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size, leading=size * 1.45, textColor=color, alignment=TA_LEFT,
    )
    item = Paragraph(text, style)
    _, height = item.wrap(width, 1000)
    item.drawOn(canvas, x, top - height)
    return top - height


def write_pdf(scenario, number):
    destination = PROJECT / scenario["pdf"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(destination), pagesize=A4, invariant=1, pageCompression=1)
    canvas.setTitle("Synthetic contract intake - " + scenario["intake"]["customer_name"])
    canvas.setAuthor("Contract Intake & Group Allocation workshop")
    width, height = A4
    left, right = 23 * mm, width - 23 * mm
    content_width = right - left

    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 47 * mm, width, 47 * mm, fill=1, stroke=0)
    paragraph(canvas, "WORKSHOP / SYNTHETIC DOCUMENT", left, height - 17 * mm,
              content_width, size=9, color=colors.HexColor("#A9E2DF"), bold=True)
    paragraph(canvas, "Contract intake brief", left, height - 25 * mm,
              content_width, size=25, color=colors.white, bold=True)

    y = height - 60 * mm
    paragraph(canvas, f"SAMPLE {number:02d}", left, y, content_width,
              size=9, color=TEAL, bold=True)
    y -= 12 * mm
    y = paragraph(canvas, scenario["intake"]["customer_name"], left, y,
                  content_width, size=20, color=NAVY, bold=True)
    y -= 8 * mm
    y = paragraph(canvas, "Fictional customer and request. For software training only.",
                  left, y, content_width, size=10, color=MUTED)
    y -= 12 * mm

    # Explicit labels are readable by people, text extraction, and the limited mock.
    # The conflict belongs to the form; the PDF consistently describes Government.
    intake = scenario["intake"]
    segment = "Government" if number in (1, 2) else "Medium Business"
    rows = [
        f"Customer legal name: {intake['customer_name']}",
        f"Customer segment: {segment}",
        "Contract type: Services agreement",
        f"Handling approach: {intake['handling_approach']}",
        "Digital eligibility: Unknown",
    ]
    panel_height = 64 * mm
    canvas.setFillColor(PALE)
    canvas.roundRect(left, y - panel_height, content_width, panel_height,
                     3 * mm, fill=1, stroke=0)
    row_y = y - 8 * mm
    for row in rows:
        row_y = paragraph(canvas, row, left + 6 * mm, row_y,
                          content_width - 12 * mm, size=10.5)
        row_y -= 5 * mm
    y -= panel_height + 13 * mm

    y = paragraph(canvas, "Request description", left, y, content_width,
                  size=12, color=NAVY, bold=True)
    y -= 5 * mm
    y = paragraph(canvas, intake["description"], left, y, content_width)
    y -= 11 * mm
    y = paragraph(canvas, "Information provided", left, y, content_width,
                  size=12, color=NAVY, bold=True)
    y -= 5 * mm
    paragraph(canvas,
              "This brief records the synthetic intake information above. "
              "It contains no agreed commercial terms and is not a signed agreement.",
              left, y, content_width, size=10.5, color=MUTED)

    canvas.setStrokeColor(LINE)
    canvas.line(left, 25 * mm, right, 25 * mm)
    paragraph(canvas, "SYNTHETIC WORKSHOP FIXTURE", left, 20 * mm,
              content_width, size=8, color=MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(right, 16 * mm, "1 / 1")
    canvas.showPage()
    canvas.save()
    print(destination.relative_to(PROJECT).as_posix())


if __name__ == "__main__":
    scenarios = json.loads((PROJECT / "demo/scenarios.json").read_text(encoding="utf-8"))
    for index, scenario in enumerate(scenarios, start=1):
        write_pdf(scenario, index)
