"""Collect readings from configured sensors into one timestamped record."""
from datetime import datetime, timezone


class SensorManager:
    def __init__(self, mq2=None, dht11=None, flame=None):
        self.mq2 = mq2
        self.dht11 = dht11
        self.flame = flame

    def read_all(self):
        record = {
            "timestamp": datetime.now(timezone.utc),
            "temperature_c": None,
            "humidity_percent": None,
            "mq2_adc": None,
            "mq2_voltage": None,
            "flame_detected": None,
        }
        if self.dht11 is not None:
            record.update(self.dht11.read())
        if self.mq2 is not None:
            record["mq2_adc"] = round(self.mq2.read_average())
            record["mq2_voltage"] = round(self.mq2.read_voltage(), 3)
        if self.flame is not None:
            record["flame_detected"] = self.flame.flame_detected()
        return record

    def close(self):
        for sensor in (self.mq2, self.dht11, self.flame):
            if sensor is not None:
                try:
                    sensor.close()
                except Exception:
                    pass
