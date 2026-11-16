"""Pydantic schemas for Raspberry Pi sensor telemetry."""
from datetime import datetime
from pydantic import BaseModel, Field


class SensorTelemetryIn(BaseModel):
    timestamp: datetime
    temperature_c: float | None = Field(default=None, ge=-40, le=125)
    humidity_percent: float | None = Field(default=None, ge=0, le=100)
    mq2_adc: int | None = Field(default=None, ge=0, le=1023)
    mq2_voltage: float | None = Field(default=None, ge=0, le=3.3)
    flame_detected: bool | None = None


class SensorTelemetryOut(BaseModel):
    status: str
    received_at: datetime
