"""Drives the existing campy camera-trigger Teensy (campy/trigger/trigger.ino).

This board pulses the camera's hardware trigger line at the target frame rate,
which is what paces the whole real-time loop. We speak its serial protocol
directly:

    "<numPins>,<pin0>,<pin1>,...,<frameRate>,<numPulses>\n"

and stop it by resending the command with numPulses = -1 (campy convention).
"""

from __future__ import annotations

from .config import ControllerConfig
from .stim import SerialLink

# trigger.ino counts pulses with a 32-bit int; this is ~2.4 years at 30 fps.
_CONTINUOUS_PULSES = 2_000_000_000


class CameraTrigger:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.link = SerialLink(config.cam_trigger_serial_port,
                               config.cam_trigger_baud,
                               config.serial_dry_run, label="cam-trigger")
        self._serial_list = None

    def _num_pulses(self) -> int:
        if self.config.rec_time_sec > 0:
            return int(round(self.config.frame_rate * self.config.rec_time_sec))
        if self.config.max_frames > 0:
            return int(self.config.max_frames)
        return _CONTINUOUS_PULSES

    def start(self) -> None:
        # Open the port and let the Teensy boot (it resets on serial open and
        # waits for the config string), mirroring campy's open-then-sleep.
        self.link.open(settle_sec=3.0)
        pins = list(self.config.cam_trigger_pins)
        serial_list = ([len(pins)] + pins
                       + [self.config.frame_rate, self._num_pulses()])
        self._serial_list = serial_list
        line = ",".join(str(x) for x in serial_list) + "\n"
        self.link.write(line.encode())
        print(f"[cam-trigger] pins={pins} @ {self.config.frame_rate} fps, "
              f"{self._num_pulses()} pulses")

    def stop(self) -> None:
        if self._serial_list is not None:
            serial_list = list(self._serial_list)
            serial_list[-1] = -1            # numPulses = -1 -> stop pulsing
            line = ",".join(str(x) for x in serial_list) + "\n"
            self.link.write(line.encode())
        self.link.close()
