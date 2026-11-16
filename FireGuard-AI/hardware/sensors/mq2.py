"""MQ-2 analog sensor driver using MCP3008 over SPI on Raspberry Pi."""
from __future__ import annotations

import time

try:
    import spidev
except ImportError:
    spidev = None


class MQ2Sensor:
    """Read raw MQ-2 signal through MCP3008; no ppm calibration is implied."""

    def __init__(self, channel=0, bus=0, device=0, reference_voltage=3.3):
        if not 0 <= channel <= 7:
            raise ValueError("MCP3008 channel must be in range 0..7")
        if not 0 < reference_voltage <= 3.3:
            raise ValueError("reference_voltage must be > 0 and <= 3.3V for Pi-safe setup")
        if spidev is None:
            raise RuntimeError("spidev missing; install with: sudo apt install python3-spidev")
        self.channel = channel
        self.reference_voltage = reference_voltage
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = 1_000_000

    def read_raw(self):
        """Return 10-bit ADC count (0..1023)."""
        response = self.spi.xfer2([1, (8 + self.channel) << 4, 0])
        return ((response[1] & 0b11) << 8) | response[2]

    def read_average(self, samples=10):
        if samples < 1:
            raise ValueError("samples must be at least 1")
        return sum(self.read_raw() for _ in range(samples)) / samples

    def read_voltage(self, samples=10):
        return self.read_average(samples) * self.reference_voltage / 1023.0

    def close(self):
        self.spi.close()


def main():
    from hardware.config import CONFIG

    sensor = MQ2Sensor(CONFIG.mq2_adc_channel, CONFIG.spi_bus,
                       CONFIG.spi_device, CONFIG.adc_reference_voltage)
    try:
        print("MQ-2 / MCP3008 readings (Ctrl+C to stop)")
        while True:
            raw = sensor.read_average()
            voltage = raw * CONFIG.adc_reference_voltage / 1023.0
            print(f"ADC={raw:.1f}/1023 | voltage={voltage:.3f} V")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
