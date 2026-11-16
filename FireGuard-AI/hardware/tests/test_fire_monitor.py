from hardware.monitoring.fire_monitor import evaluate_readings


def test_no_alert_for_normal_readings():
    result = evaluate_readings({
        "temperature_c": 25.0,
        "mq2_adc": 100,
        "flame_detected": False,
    })
    assert result["alert"] is False
    assert result["reasons"] == []


def test_flame_detection_triggers_alert():
    result = evaluate_readings({
        "temperature_c": 25.0,
        "mq2_adc": 100,
        "flame_detected": True,
    })
    assert result["alert"] is True
    assert "flame_detected" in result["reasons"]


def test_multiple_conditions_are_reported():
    result = evaluate_readings({
        "temperature_c": 70.0,
        "mq2_adc": 800,
        "flame_detected": True,
    })
    assert result["alert"] is True
    assert set(result["reasons"]) == {
        "high_temperature", "elevated_mq2_signal", "flame_detected"
    }
