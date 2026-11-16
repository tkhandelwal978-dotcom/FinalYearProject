# Raspberry Pi Hardware Changes — Commit Guide

These commits were created on branch `tanmay/iot-embedded-hardware` in chronological order.
They add an optional sensor module and a separate telemetry endpoint without replacing the
existing EfficientNet-B0 image classifier.

1. `5338034 docs: document Raspberry Pi hardware architecture`
2. `0f79a13 feat: add Raspberry Pi hardware configuration`
3. `a1b57c9 feat: add DHT11 MQ2 and flame sensor drivers`
4. `7f5cec1 feat: add sensor manager and fire monitoring logic`
5. `2b06476 feat: add sensor telemetry API endpoint`
6. `051d43a docs: add hardware runner and document IoT integration`
7. `28c45e6 test: add hardware monitoring tests and ignore telemetry logs`

## Validation performed
- Python compile check passed for the hardware module and backend application.
- Hardware rule-monitor unit tests passed (3 tests).
- Physical Raspberry Pi, sensor wiring, and SPI readings have not been tested in this environment.

## Before running on the Pi
1. Confirm MQ-2 analog output is voltage-limited to 3.3V or below before MCP3008 CH0.
2. Enable SPI and install `python3-spidev`.
3. Set actual DHT11 and flame sensor BCM pins in `hardware/config.py`.
4. Verify flame sensor output polarity and voltage.
5. Calibrate MQ-2 after its required warm-up; the ADC count is not a ppm reading.
6. Run from `FireGuard-AI`: `python3 -m hardware.run_monitor`.

## Push the branch
After extracting this archive, open a terminal in the repository and run:
`git push origin tanmay/iot-embedded-hardware`

The commit author email in this offline working copy is a local placeholder. Configure your
own Git identity before creating additional commits if you want GitHub to associate them
with your profile.
