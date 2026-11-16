"""Prototype multi-sensor rule evaluation; not a certified fire alarm."""
from hardware.config import CONFIG


def evaluate_readings(readings):
    reasons = []

    temperature = readings.get("temperature_c")
    mq2_adc = readings.get("mq2_adc")
    flame = readings.get("flame_detected")

    if temperature is not None and temperature >= CONFIG.temperature_alert_c:
        reasons.append("high_temperature")
    if mq2_adc is not None and mq2_adc >= CONFIG.mq2_adc_alert:
        reasons.append("elevated_mq2_signal")
    if flame is True:
        reasons.append("flame_detected")

    return {
        "alert": bool(reasons),
        "reasons": reasons,
        "sensor_readings": readings,
    }
