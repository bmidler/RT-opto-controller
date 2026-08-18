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
    """Opens a serial port, or logs to stdout in dry-run mode.

    Writes are mutex-protected: the stim board is written to from the frame
    thread (S/X/K) and from the reader thread (config re-send after a board
    reset), and a config line interleaved with a command byte would be parsed
    as garbage.
    """

    def __init__(self, port: str, baud: int, dry_run: bool, label: str = "serial"):
        self.port = port
        self.baud = baud
        self.dry_run = dry_run
        self.label = label
        self._ser = None
        self._write_lock = threading.Lock()

    def open(self, settle_sec: float = 2.0) -> None:
        if self.dry_run:
            print(f"[{self.label}] DRY-RUN (no port opened)")
            return
        import serial  # lazy: pyserial only needed for real hardware
        try:
            self._ser = serial.Serial(port=self.port, baudrate=self.baud,
                                      timeout=0.1)
        except Exception as e:
            # Name the ports that DO exist: after re-flashing, a native-USB
            # board (Nano ESP32) commonly re-enumerates onto a different port.
            try:
                from serial.tools import list_ports
                found = ", ".join(f"{p.device} ({p.description})"
                                  for p in list_ports.comports()) or "none"
            except Exception:
                found = "could not enumerate"
            raise RuntimeError(
                f"cannot open {self.label} serial port {self.port!r}: {e}\n"
                f"  ports currently available: {found}\n"
                f"  If the port name changed, update the config. If it is "
                f"correct, close anything else holding it (Arduino IDE Serial "
                f"Monitor, SpinView, another controller run).") from None
        # NOTE: do NOT assume the board resets here. An AVR Arduino resets on the
        # DTR toggle, but the Nano ESP32 enumerates over native USB and keeps
        # running across a port open/close. The stim board therefore has to
        # accept configuration at any time (it does), and we must never rely on
        # opening the port to put it into a known state.
        time.sleep(settle_sec)

    def write(self, data: bytes, echo: bool = True) -> None:
        if self.dry_run:
            if echo:
                print(f"[{self.label}] -> {data!r}")
            return
        if self._ser is None:
            return          # port never opened / already closed: nothing to do
        with self._write_lock:
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
        # Every non-pulse line the board emits, kept for the session record:
        #   {"t": perf_counter_sec, "text": str}
        self.board_messages: list[dict] = []
        self.n_board_resets = 0      # boot banners seen AFTER the initial open
        self.n_config_errors = 0     # "CONFIG ERR ..." replies
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._config_ok = threading.Event()
        self._config_err = threading.Event()
        self._opened = False         # True once the initial handshake is done

    def open(self) -> None:
        self.link.open()
        # The reader must be running BEFORE the first config line is sent: the
        # board's "CONFIG OK"/"CONFIG ERR" reply is what the handshake waits on.
        self._start_pulse_listener()
        self.configure(wait=True)
        self._opened = True

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
                continue
            self._handle_board_message(line, t)

    def _handle_board_message(self, line: str, t: float) -> None:
        """Record and act on a non-pulse line from the board.

        These used to be discarded, which made a mid-session board reset
        completely invisible from the PC. They are now kept (and written to
        stim_board_log.csv at shutdown) and, critically, acted on.
        """
        self.board_messages.append({"t": t, "text": line})
        print(f"[stim/board] {line}")

        if line.startswith("CONFIG OK"):
            # Verify the board echoed back exactly what we asked for. The board
            # validates ranges, but a byte corrupted in transit can still yield
            # a DIFFERENT yet internally-valid config (e.g. "5000" arriving as
            # "500"), which would silently halve the pulse width. Comparing the
            # echo is what catches that.
            bad = self._echo_mismatch(line)
            if bad:
                self.n_config_errors += 1
                print(f"[stim] CONFIG echo mismatch: {bad}")
                self._config_err.set()
            else:
                self._config_ok.set()
        elif line.startswith("CONFIG ERR"):
            self.n_config_errors += 1
            self._config_err.set()
        elif line.startswith("STIM controller ready"):
            # The board rebooted. It does NOT reset when we open the port, so
            # nothing else would ever re-configure it -- without this the board
            # stays unconfigured (and refuses to stim) until it is re-flashed.
            if self._opened:
                self.n_board_resets += 1
                print("[stim] board reset detected -> re-sending config")
                self._send_config()

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

    def _config_line(self) -> bytes:
        """The config line, formatted so the board's strict parser accepts it.

        Field types are pinned here (ints as ints, frequency with a fixed number
        of decimals) because the firmware rejects any token with trailing
        characters -- a YAML value of `5000.0` would otherwise be sent as
        "5000.0" and refused by the integer parser.
        """
        c = self.config
        return (f"{int(c.stim_pin)},{int(c.pulse_width_us)},"
                f"{float(c.frequency_hz):.4f},{int(c.max_pulses)},"
                f"{int(c.watchdog_ms)}\n").encode()

    def _echo_mismatch(self, line: str) -> str:
        """Return a description of any field the board echoed back differently.

        The board's reply is "CONFIG OK pin=9 pulseWidthUs=5000 ..."; empty
        return means every field we care about matches what we sent.
        """
        got = {}
        for tok in line.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                got[k] = v
        c = self.config
        want = {
            "pin": int(c.stim_pin),
            "pulseWidthUs": int(c.pulse_width_us),
            "maxPulses": int(c.max_pulses),
            "watchdogMs": int(c.watchdog_ms),
        }
        bad = []
        for k, v in want.items():
            if k not in got:
                bad.append(f"{k} missing from echo")
            else:
                try:
                    if int(got[k]) != v:
                        bad.append(f"{k}: sent {v}, board has {got[k]}")
                except ValueError:
                    bad.append(f"{k}: unparseable echo {got[k]!r}")
        # frequency is a float; compare with tolerance
        if "frequencyHz" in got:
            try:
                if abs(float(got["frequencyHz"]) - float(c.frequency_hz)) > 1e-3:
                    bad.append(f"frequencyHz: sent {c.frequency_hz}, "
                               f"board has {got['frequencyHz']}")
            except ValueError:
                bad.append(f"frequencyHz: unparseable echo {got['frequencyHz']!r}")
        else:
            bad.append("frequencyHz missing from echo")
        return "; ".join(bad)

    def _send_config(self) -> None:
        self._config_ok.clear()
        self._config_err.clear()
        self.link.write(self._config_line())

    def configure(self, wait: bool = True, attempts: int = 3,
                  timeout: float = 3.0) -> None:
        """Send the config line and (by default) verify the board accepted it.

        The board validates every field and replies CONFIG OK / CONFIG ERR. A
        silent failure here used to be invisible AND dangerous: a rejected or
        truncated config left the board running stale parameters. Now an
        unacknowledged config is a hard error before any frame is grabbed.
        """
        if self.config.serial_dry_run:
            self.link.write(self._config_line())
            return
        for attempt in range(1, attempts + 1):
            self._send_config()
            if not wait:
                return
            if self._config_ok.wait(timeout):
                return
            if self._config_err.is_set():
                raise RuntimeError(
                    "stim board REJECTED the config line "
                    f"({self._config_line()!r}); see the CONFIG ERR reason "
                    "printed above")
            print(f"[stim] no CONFIG OK from the board "
                  f"(attempt {attempt}/{attempts}), retrying")
        raise RuntimeError(
            f"stim board did not acknowledge its config after {attempts} "
            f"attempts on {self.config.stim_serial_port}. Check the port, and "
            "that stim_controller.ino is flashed and built with "
            "'USB CDC On Boot: Enabled'.")

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
        # Retrigger debounce. With onset_frames=1 a classifier that flickers
        # 1,0,1,0,... produces an onset on every other frame, so the board is
        # re-STARTed (and its pulse counter re-zeroed) many times a second --
        # the animal then receives pulses at the retrigger rate rather than at
        # frequency_hz, and max_pulses can never be reached. Onsets inside the
        # refractory window are ignored. 0.0 disables it (previous behaviour).
        self.refractory_sec = max(0.0, float(config.stim_refractory_sec))

        self.is_on = False
        self._consec_on = 0
        self._stim_until = 0.0       # perf_counter time the current train ends
        self._last_keepalive = 0.0
        self._last_start = float("-inf")   # perf_counter of the last START sent
        self.n_activations = 0
        self.n_onsets_suppressed = 0

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

        # Debounce retriggers: an onset too soon after the last START is
        # counted and dropped rather than re-zeroing the board's pulse train.
        if onset and self.refractory_sec > 0.0 and \
                (now - self._last_start) < self.refractory_sec:
            onset = False
            self.n_onsets_suppressed += 1

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
            self._last_start = now
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
