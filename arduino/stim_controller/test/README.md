# Host-side test harness for `stim_controller.ino`

This compiles the **real firmware source** against a stub Arduino API
(`arduino_stub.h`) so its safety behaviour can be exercised on a PC, with a
fake clock and a scripted serial stream, without a board or a laser.

```sh
cd arduino/stim_controller/test
g++ -std=c++17 -Wall -I.. -o harness harness.cpp && ./harness
```

Exit status is non-zero if any check fails.

What it checks:

* the gate pin is an OUTPUT driven LOW before anything else in `setup()`
* `'S'` is refused until a config line has been accepted
* malformed config lines are rejected whole (truncated, out-of-range,
  wrong pin, >=100% duty, noise byte or command byte inside the line) and
  leave the previous config intact
* the config line `controller/stim.py` actually sends is accepted, and is
  still accepted *after* `loop()` is running (the Nano ESP32 does not reset
  when the PC opens the port)
* pulse shape: HIGH for `pulseWidth_us`, repeating at `1/frequency_Hz`
* a stalled loop skips pulses instead of firing a catch-up burst with the
  gate held HIGH
* the watchdog drops the gate when keepalives stop
* `maxPulses` and `MAX_TRAIN_MS` both end a train
* stray bytes (including the old `'C'` command) never latch the pin or block

The stub is deliberately minimal: it is not an ESP32 emulator, and it cannot
tell you anything about USB CDC behaviour, timing under real load, or the
electrical state of the pin. Those still need the board and a scope.
