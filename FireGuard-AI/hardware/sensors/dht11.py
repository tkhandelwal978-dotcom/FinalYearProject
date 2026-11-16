"""DHT11 temperature/humidity driver using Adafruit CircuitPython."""
class DHT11Sensor:
    def __init__(self, gpio_bcm):
        if gpio_bcm is None:
            raise ValueError("Set dht11_gpio_bcm in hardware/config.py after checking wiring")
        try:
            import board
            import adafruit_dht
        except ImportError as exc:
            raise RuntimeError(
                "Install adafruit-blinka and adafruit-circuitpython-dht"
            ) from exc

        pin = getattr(board, f"D{gpio_bcm}", None)
        if pin is None:
            raise ValueError(f"BCM GPIO {gpio_bcm} is not exposed by this board mapping")
        self.device = adafruit_dht.DHT11(pin)

    def read(self):
        """Return temperature in Celsius and relative humidity percentage."""
        try:
            return {
                "temperature_c": self.device.temperature,
                "humidity_percent": self.device.humidity,
            }
        except RuntimeError:
            # DHT sensors can produce occasional transient read/checksum errors.
            return {"temperature_c": None, "humidity_percent": None}

    def close(self):
        self.device.exit()
