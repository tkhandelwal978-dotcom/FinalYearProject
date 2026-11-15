"""Aggregate statistics computed from the persisted predictions (never hardcoded)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.models.prediction import PredictionRecord
from app.schemas.system import AnalyticsOut, ConfidenceBucket, DailyTrend
from app.services.prediction_service import to_schema

CONFIDENCE_EDGES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def compute_analytics(db: Session, days: int = 14) -> AnalyticsOut:
    P = PredictionRecord
    is_fire = case((P.predicted_class == "fire", 1), else_=0)
    total, fire, avg_conf, avg_ms = db.execute(
        select(func.count(P.id), func.coalesce(func.sum(is_fire), 0), func.avg(P.confidence), func.avg(P.inference_ms))
    ).one()

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    last24_total, last24_fire = db.execute(
        select(func.count(P.id), func.coalesce(func.sum(is_fire), 0)).where(P.created_at >= now - timedelta(hours=24))
    ).one()

    start_day = (now - timedelta(days=days - 1)).date()
    day = func.date(P.created_at)
    rows = db.execute(
        select(day, func.coalesce(func.sum(is_fire), 0), func.count(P.id))
        .where(P.created_at >= datetime.combine(start_day, datetime.min.time()))
        .group_by(day)
    ).all()
    by_day = {str(d): (int(f), int(t)) for d, f, t in rows}
    trend = []
    for offset in range(days):
        key = (start_day + timedelta(days=offset)).isoformat()
        f, t = by_day.get(key, (0, 0))
        trend.append(DailyTrend(date=key, fire=f, non_fire=t - f, total=t))

    buckets = []
    for low, high in zip(CONFIDENCE_EDGES, CONFIDENCE_EDGES[1:]):
        upper = P.confidence <= high if high == 1.0 else P.confidence < high
        f, t = db.execute(
            select(func.coalesce(func.sum(is_fire), 0), func.count(P.id)).where(P.confidence >= low, upper)
        ).one()
        buckets.append(ConfidenceBucket(range=f"{int(low * 100)}-{int(high * 100)}%", fire=int(f), non_fire=int(t) - int(f)))

    recent = db.scalars(select(P).order_by(P.created_at.desc()).limit(6)).all()
    total, fire = int(total), int(fire)
    return AnalyticsOut(
        total_predictions=total,
        fire_detections=fire,
        non_fire_detections=total - fire,
        fire_detection_rate=round(fire / total * 100, 2) if total else 0.0,
        average_confidence=round(float(avg_conf), 4) if avg_conf is not None else None,
        average_inference_ms=round(float(avg_ms), 2) if avg_ms is not None else None,
        predictions_last_24h=int(last24_total),
        fire_last_24h=int(last24_fire),
        daily_trend=trend,
        confidence_distribution=buckets,
        recent_predictions=[to_schema(r) for r in recent],
    )
