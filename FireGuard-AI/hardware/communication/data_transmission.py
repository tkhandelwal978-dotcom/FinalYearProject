"""JSON serialization and local JSONL logging for sensor telemetry."""
import json
from pathlib import Path
from datetime import datetime


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def to_json(readings):
    return json.dumps(readings, default=_json_default, separators=(",", ":"))


def append_jsonl(readings, output_path="artifacts/sensor_telemetry.jsonl"):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(to_json(readings) + "\n")
    return path
