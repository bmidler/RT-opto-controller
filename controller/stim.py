"""PC side of the stim path.

Three layers:
  * SerialLink     -- thin pyserial wrapper with a hardware-free dry-run mode.
  * StimArduino    -- speaks the stim_controller.ino protocol (config / S/X/K).
  * StimController  -- the decision state machine: maps per-frame class
                       predictions to edge-triggered START/STOP plus keepalive,
                       with optional onset/offset debouncing.
"""

from __future__ import annotations

import time
from typing import Optional

from .config import ControllerConfig


class SerialLink:
    """Opens a serial port, or logs to stdout in dry-run mode."""

    def __init__(self, port: str, baud: int, dry_run: bool, label: str = "serial"):
        self.port = port
        self.baud = baud
        self.dry_run = dry_run
        self.label = label
        self._ser = None

    def open(self, settle_sec: float = 2.0) -> None:
        if self.dry_run:
            print(f"[{self.label}] DRY-RUN (no port opened)")
            return
        import serial  # lazy: pyserial only needed for real hardware
        self._ser = serial.Serial(port=self.port, baudrate=self.baud, timeout=0.1)
        # Arduino/Teensy typically resets when the port opens; let it boot.
        time.sleep(settle_sec)

    def write(self, data: bytes, echo: bool = True) -> None:
        if self.dry_run:
            if echo:
                print(f"[{self.label}] -> {data!r}")
            return
        self._ser.write(data)
        self._ser.flush()

    def readline(self) -> str:
        if self.dry_run or self._ser is None:
            return ""
        try:
            return self._ser.readline().decode(errors="replace").strip()
        except Exception:
            return ""

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None


class StimArduino:
    """Implements the stim_controller.ino serial protocol."""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.link = SerialLink(config.stim_serial_port, config.stim_baud,
                               config.serial_dry_run, label="stim")

    def open(self) -> None:
        self.link.open()
        self.configure()

    def configure(self) -> None:
        c = self.config
        line = (f"{c.stim_pin},{c.pulse_width_us},{c.frequency_hz},"
                f"{c.max_pulses},{c.watchdog_ms}\n")
        self.link.write(line.encode())

    def start(self) -> None:
        self.link.write(b"S", echo=True)

    def stop(self) -> None:
        self.link.write(b"X", echo=True)

    def keepalive(self) -> None:
        # Suppress per-keepalive echo (fires often) to avoid log spam.
        self.link.write(b"K", echo=False)

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.link.close()


class StimController:
    """Maps class predictions to laser START/STOP with debounce + keepalive.

    Mode: continuous-while-detected. The laser stays on (the Arduino keeps
    pulsing at the configured frequency/width) as long as the predicted class
    is in `trigger_classes`; it stops when the behaviour ends. Only state
    transitions are sent over serial, plus a periodic keepalive that refreshes
    the Arduino's safety watchdog while on.
    """

    def __init__(self, config: ControllerConfig, arduino: Optional[StimArduino]):
        self.arduino = arduino
        self.trigger_set = set(int(x) for x in config.trigger_classes)
        self.onset_frames = max(1, int(config.onset_frames))
        self.offset_frames = max(1, int(config.offset_frames))
        self.keepalive_sec = config.keepalive_ms / 1000.0

        self.is_on = False
        self._consec_on = 0
        self._consec_off = 0
        self._last_keepalive = 0.0
        self.n_activations = 0

    def update(self, pred_class: int, now: float) -> bool:
        """Feed one frame's prediction. Returns whether stim is ON afterwards."""
        in_trigger = pred_class in self.trigger_set
        if in_trigger:
            self._consec_on += 1
            self._consec_off = 0
        else:
            self._consec_off += 1
            self._consec_on = 0

        if not self.is_on and self._consec_on >= self.onset_frames:
            self.is_on = True
            self.n_activations += 1
            self._last_keepalive = now
            if self.arduino is not None:
                self.arduino.start()
        elif self.is_on and self._consec_off >= self.offset_frames:
            self.is_on = False
            if self.arduino is not None:
                self.arduino.stop()
        elif self.is_on and (now - self._last_keepalive) >= self.keepalive_sec:
            self._last_keepalive = now
            if self.arduino is not None:
                self.arduino.keepalive()

        return self.is_on

    def shutdown(self) -> None:
        if self.is_on and self.arduino is not None:
            self.arduino.stop()
        self.is_on = False
