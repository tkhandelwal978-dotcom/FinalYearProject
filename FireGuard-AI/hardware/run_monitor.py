"""Run configured Raspberry Pi sensors and log rule-evaluation telemetry.

Configure GPIO pins in hardware/config.py before running on the physical Pi.
"""
import time

from hardware.communication.data_transmission import append_jsonl
from hardware.config import CONFIG
from hardware.monitoring.fire_monitor import evaluate_readings
from hardware.sensors.manager import SensorManager
from hardware.sensors.mq2 import MQ2Sensor


def main():
    from hardware.sensors.dht11 import DHT11Sensor
    from hardware.sensors.flame import FlameSensor

    mq2 = MQ2Sensor(
        channel=CONFIG.mq2_adc_channel,
        bus=CONFIG.spi_bus,
        device=CONFIG.spi_device,
        reference_voltage=CONFIG.adc_reference_voltage,
    )
    dht11 = DHT11Sensor(CONFIG.dht11_gpio_bcm) if CONFIG.dht11_gpio_bcm is not None else None
    flame = (
        FlameSensor(CONFIG.flame_gpio_bcm, CONFIG.flame_active_low)
        if CONFIG.flame_gpio_bcm is not None else None
    )
    manager = SensorManager(mq2=mq2, dht11=dht11, flame=flame)

    try:
        print("FireGuard hardware monitoring started. Press Ctrl+C to stop.")
        while True:
            readings = manager.read_all()
            result = evaluate_readings(readings)
            append_jsonl(result)
            print(result)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
