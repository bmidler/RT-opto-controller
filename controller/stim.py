"""PC side of the stim path.

Three layers:
  * SerialLink     -- thin pyserial wrapper with a hardware-free dry-run mode.
  * StimArduino    -- speaks the stim_controller.ino protocol (config / S/X/K).
  * StimController  -- the decision state machine: maps per-frame class
                       predictions to edge-triggered START/STOP plus keepalive,
                       with optional onset/offset debouncing.
"""

from __future__ import annotations

import threading
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
    """Implements the stim_controller.ino serial protocol.

    A background reader thread drains the stim board's USB serial and records
    the arrival time of every pulse marker ("P,<index>,<micros>") the board
    emits on each rising edge. The PC arrival time (perf_counter) is the
    authoritative stim time used to align pulses to the recorded video; the
    board's own micros() is kept alongside for jitter/drop diagnostics. Reading
    happens off the hot path so the closed-loop latency is unaffected.
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.link = SerialLink(config.stim_serial_port, config.stim_baud,
                               config.serial_dry_run, label="stim")
        # One entry per pulse reported by the board:
        #   {"t": perf_counter_sec, "board_index": int, "board_us": int}
        self.pulse_marks: list[dict] = []
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()

    def open(self) -> None:
        self.link.open()
        self.configure()
        self._start_pulse_listener()

    # -- pulse marker listener ---------------------------------------------
    def _start_pulse_listener(self) -> None:
        # Nothing to read from in dry-run (no port); skip the thread entirely.
        if self.config.serial_dry_run:
            return
        self._reader_thread = threading.Thread(
            target=self._read_loop, name="stim-reader", daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        while not self._reader_stop.is_set():
            line = self.link.readline()        # blocks up to the serial timeout
            if not line:
                continue
            t = time.perf_counter()            # timestamp at arrival
            if line[0] in ("P", "p"):
                self._record_pulse(line, t)
            # Any other line (CONFIG/WATCHDOG/banner) is ignored.

    def _record_pulse(self, line: str, t: float) -> None:
        board_index = -1
        board_us = -1
        parts = line.split(",")
        try:
            if len(parts) >= 2:
                board_index = int(parts[1])
            if len(parts) >= 3:
                board_us = int(parts[2])
        except ValueError:
            pass  # malformed marker: still record the arrival time
        self.pulse_marks.append(
            {"t": t, "board_index": board_index, "board_us": board_us})

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
            # Stop the reader before closing the port so it isn't mid-read.
            self._reader_stop.set()
            if self._reader_thread is not None:
                self._reader_thread.join(timeout=2.0)
            self.link.close()


class StimController:
    """Maps class predictions to a fixed-duration, retriggerable laser train.

    Mode: fixed-duration-on-onset. Each detected event onset starts a pulse
    train that runs for `stim_duration_sec` (the Arduino pulses at the
    configured frequency/width for that whole window), INDEPENDENT of how long
    the behaviour itself lasts -- a one-frame event and a 100-frame event both
    produce the same train. If a new onset is detected while a train is still
    running, the train RESETS to the new onset (a fresh START re-zeroes the
    Arduino's pulse counter and phase). Only edges are sent over serial, plus a
    periodic keepalive that refreshes the Arduino's safety watchdog for the full
    window so the event ending early can't cut the train short.
    """

    def __init__(self, config: ControllerConfig, arduino: Optional[StimArduino]):
        self.arduino = arduino
        self.trigger_set = set(int(x) for x in config.trigger_classes)
        self.onset_frames = max(1, int(config.onset_frames))
        # Retained for metadata/back-compat; unused in fixed-duration mode.
        self.offset_frames = max(1, int(config.offset_frames))
        self.stim_duration_sec = max(0.0, float(config.stim_duration_sec))
        self.keepalive_sec = config.keepalive_ms / 1000.0

        self.is_on = False
        self._consec_on = 0
        self._stim_until = 0.0       # perf_counter time the current train ends
        self._last_keepalive = 0.0
        self.n_activations = 0

        # Edge log: one entry per START / STOP the controller actually commands.
        # Each is {"kind": "START"|"STOP", "t": perf_counter_sec, "frame": int}.
        # The individual laser pulses are reconstructed from these windows at
        # shutdown (the stim Arduino does not report pulses back over serial).
        self.events: list[dict] = []

    def _record(self, kind: str, now: float, frame: Optional[int]) -> None:
        self.events.append({"kind": kind, "t": now, "frame": frame})

    def update(self, pred_class: int, now: float,
               frame: Optional[int] = None) -> bool:
        """Feed one frame's prediction. Returns whether stim is ON afterwards.

        Fixed-duration + retriggerable: a detected onset starts a train that
        runs for `stim_duration_sec` regardless of the event's length; a fresh
        onset while a train is running resets it to the new onset.

        `frame` is the inference-camera frame number, logged on each START/STOP
        edge so stimulation can be aligned back to the recorded video.
        """
        if pred_class in self.trigger_set:
            self._consec_on += 1
        else:
            self._consec_on = 0

        # Rising edge: _consec_on reaches the threshold exactly once per
        # contiguous run of trigger frames (a non-trigger frame resets it to 0,
        # so a later run is a NEW onset that retriggers the train).
        onset = (self._consec_on == self.onset_frames)

        if onset:
            if self.is_on:
                # Retrigger before the previous train finished: close the old
                # window in the log so pulse reconstruction restarts here too,
                # matching the Arduino re-zeroing its counter on the new START.
                self._record("STOP", now, frame)
            self.is_on = True
            self.n_activations += 1
            self._stim_until = now + self.stim_duration_sec
            self._last_keepalive = now
            self._record("START", now, frame)
            if self.arduino is not None:
                self.arduino.start()        # re-zeroes pulse counter + phase
        elif self.is_on:
            if now >= self._stim_until:
                # Fixed window elapsed -> stop, independent of the event length.
                self.is_on = False
                self._record("STOP", now, frame)
                if self.arduino is not None:
                    self.arduino.stop()
            elif (now - self._last_keepalive) >= self.keepalive_sec:
                # Keep the watchdog fed for the whole window even after the
                # behaviour has ended, so an early offset can't cut the train.
                self._last_keepalive = now
                if self.arduino is not None:
                    self.arduino.keepalive()

        return self.is_on

    def shutdown(self, now: Optional[float] = None,
                 frame: Optional[int] = None) -> None:
        if self.is_on:
            self._record("STOP", now if now is not None else 0.0, frame)
            if self.arduino is not None:
                self.arduino.stop()
        self.is_on = False


def build_activation_windows(events, t_zero: float, t_end_session: float):
    """Pair START/STOP edges into [on, off] activation windows.

    `events` are StimController.events; `t_zero` is the perf_counter value of the
    first frame (so window times come out session-relative, matching the
    classifications.csv `t_arrival_s` column). A START still open at shutdown is
    closed at `t_end_session` (the last frame's arrival time).

    Returns a list of dicts with onset/offset time (s) and frame number.
    """
    windows = []
    open_start = None
    for ev in events:
        ts = ev["t"] - t_zero
        if ev["kind"] == "START":
            if open_start is None:
                open_start = (ts, ev["frame"])
        else:  # STOP
            if open_start is not None:
                windows.append({
                    "onset_t_s": open_start[0], "onset_frame": open_start[1],
                    "offset_t_s": ts, "offset_frame": ev["frame"],
                })
                open_start = None
    if open_start is not None:
        windows.append({
            "onset_t_s": open_start[0], "onset_frame": open_start[1],
            "offset_t_s": t_end_session, "offset_frame": None,
        })
    return windows


def expand_pulses(windows, frequency_hz: float, pulse_width_us: float,
                  max_pulses: int):
    """Reconstruct individual laser pulses inside each activation window.

    The stim Arduino fires the first pulse immediately on START, then one every
    1/frequency_hz, each HIGH for pulse_width_us, until STOP (or until max_pulses
    is reached when max_pulses > 0). These are modelled from the commanded
    windows + configured timing, since the board does not echo per-pulse events.

    Returns a flat list of pulse dicts (onset/offset time in session seconds).
    """
    pulses = []
    if frequency_hz <= 0:
        return pulses
    period = 1.0 / frequency_hz
    width_s = pulse_width_us / 1e6
    for wi, w in enumerate(windows):
        t0, t1 = w["onset_t_s"], w["offset_t_s"]
        k = 0
        while True:
            if max_pulses > 0 and k >= max_pulses:
                break
            onset = t0 + k * period
            if onset > t1 + 1e-9:   # pulse would begin after the window closed
                break
            pulses.append({
                "activation_index": wi,
                "pulse_in_activation": k,
                "t_onset_s": onset,
                "t_offset_s": onset + width_s,
            })
            k += 1
    return pulses
