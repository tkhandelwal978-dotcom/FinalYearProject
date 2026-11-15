"""Database layer tests (no model needed)."""

from datetime import datetime, timedelta

from sqlalchemy import select

from app.database.session import Database
from app.models.prediction import PredictionRecord
from app.services.analytics_service import compute_analytics


def make_record(**overrides) -> PredictionRecord:
    values = dict(filename="a.jpg", image_path="uploads/a.jpg", thumbnail_path="uploads/a_thumb.jpg",
                  predicted_class="fire", fire_probability=0.9, no_fire_probability=0.1, confidence=0.9,
                  threshold=0.5, inference_ms=12.0, image_width=100, image_height=100, file_size_bytes=1000,
                  model_version="1.0.0")
    values.update(overrides)
    return PredictionRecord(**values)


def test_records_persist_across_engine_restarts(tmp_path):
    url = f"sqlite:///{(tmp_path / 'persist.db').as_posix()}"
    db = Database(url)
    db.create_tables()
    with db.session_factory() as session:
        session.add(make_record())
        session.commit()
        record_id = session.scalars(select(PredictionRecord.id)).one()
    db.engine.dispose()

    reopened = Database(url)
    with reopened.session_factory() as session:
        stored = session.get(PredictionRecord, record_id)
        assert stored is not None and stored.predicted_class == "fire"
        assert len(stored.id) == 36 and isinstance(stored.created_at, datetime)


def test_analytics_aggregations(tmp_path):
    db = Database(f"sqlite:///{(tmp_path / 'stats.db').as_posix()}")
    db.create_tables()
    now = datetime.utcnow()
    with db.session_factory() as session:
        session.add_all([
            make_record(confidence=0.95),
            make_record(predicted_class="non_fire", fire_probability=0.2, no_fire_probability=0.8, confidence=0.8),
            make_record(predicted_class="non_fire", fire_probability=0.45, no_fire_probability=0.55, confidence=0.55,
                        created_at=now - timedelta(days=3)),
        ])
        session.commit()
        stats = compute_analytics(session, days=7)

    assert (stats.total_predictions, stats.fire_detections, stats.non_fire_detections) == (3, 1, 2)
    assert stats.fire_detection_rate == 33.33
    assert stats.predictions_last_24h == 2
    assert sum(d.total for d in stats.daily_trend) == 3
    buckets = {b.range: b for b in stats.confidence_distribution}
    assert buckets["90-100%"].fire == 1 and buckets["50-60%"].non_fire == 1 and buckets["80-90%"].non_fire == 1
