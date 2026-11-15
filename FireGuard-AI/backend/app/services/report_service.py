"""Downloadable reports: a per-prediction PDF and a CSV export of the history."""

from __future__ import annotations

import csv
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.models.prediction import PredictionRecord
from app.services.storage_service import StorageService
from ml.config import CLASS_DISPLAY_NAMES

FIRE_RED = colors.HexColor("#dc2626")
SAFE_GREEN = colors.HexColor("#16a34a")


def format_probability(value: float) -> str:
    """Model probabilities never truly reach 0 or 1, so rounding must not claim they do."""
    if value >= 0.99995:
        return "> 99.99%"
    if value <= 0.00005:
        return "< 0.01%"
    return f"{value * 100:.2f}%"


def _scaled_image(path, max_width: float, max_height: float) -> RLImage:
    img = RLImage(str(path))
    ratio = min(max_width / img.imageWidth, max_height / img.imageHeight)
    img.drawWidth, img.drawHeight = img.imageWidth * ratio, img.imageHeight * ratio
    return img


def build_pdf_report(record: PredictionRecord, storage: StorageService, model_metadata: dict | None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title=f"FireGuard AI report {record.id}")
    styles = getSampleStyleSheet()
    is_fire = record.predicted_class == "fire"
    accent = FIRE_RED if is_fire else SAFE_GREEN
    story = [
        Paragraph("FireGuard AI - Image Analysis Report", styles["Title"]),
        Spacer(1, 4),
        Paragraph(CLASS_DISPLAY_NAMES[record.predicted_class],
                  ParagraphStyle("result", parent=styles["Heading1"], textColor=accent, alignment=1)),
        Spacer(1, 8),
    ]

    rows = [
        ["Prediction ID", record.id],
        ["File name", record.filename],
        ["Analysed at (UTC)", record.created_at.strftime("%Y-%m-%d %H:%M:%S")],
        ["Predicted class", CLASS_DISPLAY_NAMES[record.predicted_class]],
        ["Confidence", format_probability(record.confidence)],
        ["Probability of fire", format_probability(record.fire_probability)],
        ["Probability of no fire", format_probability(record.no_fire_probability)],
        ["Decision threshold (P(fire))", f"{record.threshold:.2f}"],
        ["Image size", f"{record.image_width} x {record.image_height} px"],
        ["Inference time", f"{record.inference_ms:.1f} ms"],
        ["Model version", record.model_version],
    ]
    if model_metadata:
        rows.append(["Model architecture", Paragraph(model_metadata["architecture"], styles["BodyText"])])
    table = Table(rows, colWidths=[55 * mm, 115 * mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR", (1, 3), (1, 3), accent),
        ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
    ]))
    story += [table, Spacer(1, 10)]

    images = [("Uploaded image", record.image_path), ("Grad-CAM overlay", record.overlay_path)]
    cells = [[_scaled_image(storage.absolute(p), 80 * mm, 75 * mm) for _, p in images if p],
             [Paragraph(title, styles["Italic"]) for title, p in images if p]]
    image_table = Table(cells)
    image_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story += [image_table, Spacer(1, 10)]
    story.append(Paragraph(
        "Grad-CAM highlights the image regions that most influenced the model's decision. "
        "This automated result is a decision-support aid and must not replace human verification "
        "or official fire alarm systems.", styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()


def build_history_csv(records: list[PredictionRecord]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["id", "filename", "predicted_class", "confidence", "fire_probability", "no_fire_probability",
                     "threshold", "inference_ms", "model_version", "created_at_utc"])
    for r in records:
        writer.writerow([r.id, r.filename, r.predicted_class, f"{r.confidence:.6f}", f"{r.fire_probability:.6f}",
                         f"{r.no_fire_probability:.6f}", r.threshold, r.inference_ms, r.model_version,
                         r.created_at.isoformat()])
    return out.getvalue()
