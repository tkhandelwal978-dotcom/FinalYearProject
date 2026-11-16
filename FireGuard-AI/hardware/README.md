# Raspberry Pi IoT Hardware Module

This folder contains the optional sensor-side extension for FireGuard-AI. The existing
computer-vision application remains intact and continues to use its EfficientNet-B0
image classifier.

## Target hardware
- Raspberry Pi 4 Model B (8GB) running Raspberry Pi OS
- DHT11 temperature/humidity sensor
- MQ-2 smoke/combustible-gas module
- MCP3008 10-bit ADC (MQ-2 analog output on CH0)
- Digital IR flame sensor

## Data flow
Sensors -> Raspberry Pi acquisition -> rule-based sensor monitor -> JSON telemetry
-> optional FastAPI telemetry endpoint. The camera/image classifier is a complementary
detection path; sensor thresholds do not replace or retrain the image model.

## MCP3008 / MQ-2 wiring
- MQ-2 AO -> suitable voltage divider/level conditioning -> MCP3008 CH0.
- MCP3008 VDD and VREF -> 3.3V; AGND and DGND -> GND.
- CLK -> GPIO11/SCLK; DOUT -> GPIO9/MISO; DIN -> GPIO10/MOSI; CS/SHDN -> GPIO8/CE0.
- MQ-2 module supply is commonly 5V; verify the exact breakout board specification.
- Ensure the conditioned AO voltage never exceeds MCP3008 VREF (3.3V here).
- Raspberry Pi GPIO is not 5V tolerant. Verify flame module DO voltage before GPIO connection.

## Configuration
DHT11 and flame GPIO BCM pins are deliberately unset until the actual wiring is confirmed.
Set them in `hardware/config.py`. Confirm flame module active-low/active-high behaviour
experimentally before enabling monitoring.

## Raspberry Pi setup
Enable SPI: `sudo raspi-config` -> Interface Options -> SPI -> Enable, then reboot.

Install packages:
```bash
sudo apt update
sudo apt install python3-spidev python3-gpiozero -y
python3 -m pip install adafruit-blinka adafruit-circuitpython-dht
```

From the `FireGuard-AI` directory, test the MQ-2/MCP3008 driver:
```bash
python3 -m hardware.sensors.mq2
```

The MQ-2 driver reports raw ADC counts and voltage only. It does not calculate calibrated
ppm. MQ-series sensors need heater warm-up, calibration and controlled testing before
quantitative gas estimates can be claimed. This prototype is not a certified fire alarm.
