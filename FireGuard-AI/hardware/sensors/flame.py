"""Digital IR flame-sensor driver for Raspberry Pi GPIO."""
class FlameSensor:
    def __init__(self, gpio_bcm, active_low=True):
        if gpio_bcm is None:
            raise ValueError("Set flame_gpio_bcm in hardware/config.py after checking wiring")
        try:
            from gpiozero import DigitalInputDevice
        except ImportError as exc:
            raise RuntimeError("Install gpiozero with: sudo apt install python3-gpiozero") from exc

        self.device = DigitalInputDevice(gpio_bcm, pull_up=None)
        self.active_low = active_low

    def flame_detected(self):
        level = bool(self.device.value)
        return (not level) if self.active_low else level

    def close(self):
        self.device.close()
