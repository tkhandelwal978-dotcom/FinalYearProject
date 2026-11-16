"""Central configuration for Raspberry Pi sensor hardware.

Set GPIO pins to match the actual wiring before starting the sensor drivers.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    # MCP3008 SPI0 / CE0; MQ-2 analog signal connected to CH0.
    spi_bus: int = 0
    spi_device: int = 0
    mq2_adc_channel: int = 0
    adc_reference_voltage: float = 3.3

    # Configure these only after confirming physical wiring (BCM numbering).
    dht11_gpio_bcm: int | None = None
    flame_gpio_bcm: int | None = None
    flame_active_low: bool = True

    # Prototype alert thresholds; tune using controlled tests.
    temperature_alert_c: float = 55.0
    mq2_adc_alert: int = 400


CONFIG = HardwareConfig()
