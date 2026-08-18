// =====================================================================
//  Optogenetics stim controller  --  Arduino Nano ESP32 (ESP32-S3)
//  ---------------------------------------------------------------------
//  Receives commands from the RT-opto controller PC and drives a single
//  digital TTL pin that gates a laser. This is a SEPARATE board from the
//  camera-trigger Teensy (arduino/camera_controller/trigger.ino).
//
//  BOARD REQUIREMENTS (Arduino IDE -> Tools)
//  -----------------------------------------
//    Board        : Arduino Nano ESP32
//    USB CDC On Boot: Enabled   <-- REQUIRED. With it disabled, `Serial` becomes
//                                UART0 on D0/D1 and this firmware would fight
//                                the very pins it is talking over.
//  The Nano ESP32 is a 3.3 V part: STIM_PIN drives 0/3.3 V, and the pin is NOT
//  5 V tolerant. Check that 3.3 V clears your laser driver's input threshold.
//
//  HARDWARE REQUIREMENT
//  --------------------
//  Fit a ~10 kohm pulldown from STIM_PIN to GND. ESP32-S3 GPIOs come out of
//  reset as floating inputs, and no firmware can drive the pin during the boot
//  window. The pulldown is what guarantees the laser is OFF while the chip is
//  in reset, in the bootloader, or unpowered.
//
//  SERIAL PROTOCOL  (115200 baud)
//  ------------------------------
//  1) Config line -- accepted AT ANY TIME, not just at startup:
//
//         pin,pulseWidth_us,frequency_Hz,maxPulses,watchdog_ms\n
//
//     e.g.  "9,5000,3.0000,75,500"
//       pin          : must equal STIM_PIN below, else the line is rejected
//       pulseWidth_us: HIGH time of each pulse, microseconds
//       frequency_Hz : pulse repetition rate (period = 1e6 / frequency_Hz)
//       maxPulses    : pulses per activation; 0 = unlimited (still bounded by
//                      MAX_TRAIN_MS)
//       watchdog_ms  : auto-OFF if no command/keepalive within this window.
//                      Must be > 0 -- there is deliberately no way to disable it.
//
//     Every field is validated. A line that fails validation is REJECTED whole:
//     the previous (known-good) config is kept and the board replies
//     "CONFIG ERR <reason>". A good line replies "CONFIG OK ...". The board
//     refuses to stimulate until it has accepted one config line.
//
//  2) Runtime single-character commands:
//       'S' : START stimulating (resets pulse counter + watchdog)
//       'X' : STOP  stimulating immediately (pin LOW)
//       'K' : KEEPALIVE (refreshes the watchdog)
//     Command characters and the config-line charset are disjoint, so both
//     share the stream safely. There is no 'C' command any more: a config line
//     identifies itself, so nothing can put the board into a blocking read.
//
//  3) Board -> PC reporting:
//       "P,<pulseIndex>,<micros>"  on each pulse rising edge
//       "STIM controller ready..." banner on every boot (the PC watches for
//                                  this and re-sends its config -- see below)
//       "CONFIG OK ..." / "CONFIG ERR ..." / "WATCHDOG: ..." / "LIMIT: ..."
//
//  WHY CONFIG IS ACCEPTED AT ANY TIME
//  ----------------------------------
//  Unlike an AVR Arduino, the Nano ESP32 does NOT reset when the PC opens the
//  serial port -- it enumerates over native USB. An earlier version of this
//  firmware only read its config inside setup(), which meant (a) the config
//  line sent by every session after the first was silently discarded, so the
//  board kept running whatever parameters it was given at power-on, and (b) a
//  board that reset mid-session was stranded until it was re-flashed. Parsing
//  config lines from the main loop fixes both.
//
//  SAFETY INVARIANTS
//  -----------------
//   * loop() never blocks. There is no busy-wait anywhere (a spin here starves
//     the ESP32 idle task and trips the task watchdog, rebooting the board).
//   * The pin is driven LOW as the very first statement in setup(), before
//     Serial.begin(), so the floating window is as short as the silicon allows.
//   * The watchdog keys off the PIN, not off `stimulating`, so it still rescues
//     states where the two disagree.
//   * MAX_TRAIN_MS caps any single activation regardless of maxPulses.
//   * Pulse state is DERIVED from elapsed time, never accumulated, so a stalled
//     loop skips pulses (correct) instead of firing a catch-up burst with the
//     gate held HIGH.
// =====================================================================

#include <stdlib.h>
#include <string.h>

const uint32_t BAUD = 115200;

// The gate pin is physical wiring, so it is a compile-time constant: no serial
// input can ever retarget it and orphan the real pin in a HIGH state.
const uint8_t STIM_PIN = 9;

// Absolute ceiling on one activation, independent of maxPulses and of the
// watchdog. Last-resort backstop; normal trains are far shorter.
const uint32_t MAX_TRAIN_MS = 60000;

// Validation bounds for the config line.
const uint32_t MIN_PULSE_US   = 1;
const uint32_t MAX_PULSE_US   = 1000000;   // 1 s
const float    MIN_FREQ_HZ    = 0.01f;
const float    MAX_FREQ_HZ    = 1000.0f;
const uint32_t MIN_WATCHDOG_MS = 1;
const uint32_t MAX_WATCHDOG_MS = 10000;

// --- Configurable parameters (set over serial) -----------------------
uint32_t pulseWidthUs  = 5000;
float    frequencyHz   = 20.0;
uint32_t periodUs      = 50000;  // derived = 1e6 / frequencyHz
uint32_t maxPulses     = 0;      // 0 = unlimited
uint32_t watchdogMs    = 500;
bool     configured    = false;  // no stim until a config line is accepted

// --- Runtime state ----------------------------------------------------
bool     stimulating   = false;
uint32_t trainStartUs  = 0;      // micros() at the start of the activation
uint32_t trainStartMs  = 0;      // millis() equivalent, for MAX_TRAIN_MS
uint32_t lastCmdMs     = 0;      // millis() of last S/K (watchdog reference)
uint32_t lastPulseIdx  = 0;      // last pulse index reported
bool     pinIsHigh     = false;

// --- Config line accumulator -----------------------------------------
char     cfgBuf[64];
uint8_t  cfgLen        = 0;
bool     cfgOverflow   = false;
bool     cfgInvalid    = false;  // an out-of-charset byte landed mid-line

void setPin(bool high) {
  digitalWrite(STIM_PIN, high ? HIGH : LOW);
  pinIsHigh = high;
}

// Report a pulse's rising edge back to the PC. Guarded by availableForWrite():
// on the ESP32's native USB CDC a write can block for up to the TX timeout when
// the host stops draining, which would stall the pulse loop. A dropped marker
// costs a diagnostic; a stalled loop costs pulse timing.
void emitPulseMark(uint32_t idx) {
  if (Serial.availableForWrite() < 32) return;
  Serial.print('P');
  Serial.print(',');
  Serial.print(idx);
  Serial.print(',');
  Serial.println(micros());
}

void stopStim() {
  stimulating = false;
  setPin(false);
}

void startStim() {
  if (!configured) {
    Serial.println("ERR unconfigured -- send a config line before 'S'");
    return;
  }
  trainStartUs = micros();
  trainStartMs = millis();
  lastCmdMs    = trainStartMs;
  lastPulseIdx = 0;
  stimulating  = true;
  setPin(true);                  // first pulse begins immediately
  emitPulseMark(0);
}

// --- Config parsing ---------------------------------------------------
// Split `s` in place on ',' into exactly `want` fields. Returns false on any
// other count, so a truncated or over-long line is rejected rather than
// silently zero-filled.
static bool splitFields(char *s, char *out[], uint8_t want) {
  uint8_t n = 1;
  out[0] = s;
  for (char *p = s; *p; ++p) {
    if (*p == ',') {
      if (n >= want) return false;
      *p = '\0';
      out[n++] = p + 1;
    }
  }
  return n == want;
}

// strtoul/strtod with full-token validation: any trailing garbage, or an empty
// token, is an error. This is the difference that matters -- Serial.parseInt()
// returns 0 both for "the PC sent 0" and for "nothing parseable arrived", and
// that ambiguity is how a truncated line used to turn into watchdogMs = 0.
static bool toU32(const char *s, uint32_t *v) {
  if (*s == '\0') return false;
  char *end;
  unsigned long x = strtoul(s, &end, 10);
  if (*end != '\0') return false;
  *v = (uint32_t)x;
  return true;
}

static bool toF32(const char *s, float *v) {
  if (*s == '\0') return false;
  char *end;
  double x = strtod(s, &end);
  if (*end != '\0') return false;
  *v = (float)x;
  return true;
}

static void configErr(const char *reason) {
  Serial.print("CONFIG ERR ");
  Serial.println(reason);
}

// Parse + validate a complete config line. On success the new values are
// committed and the board is left STOPPED; on failure nothing changes.
void applyConfig(char *line) {
  char *f[5];
  if (!splitFields(line, f, 5)) { configErr("fields"); return; }

  uint32_t pin, pw, mp, wd;
  float fr;
  if (!toU32(f[0], &pin)) { configErr("pin"); return; }
  if (!toU32(f[1], &pw))  { configErr("width"); return; }
  if (!toF32(f[2], &fr))  { configErr("freq"); return; }
  if (!toU32(f[3], &mp))  { configErr("maxpulses"); return; }
  if (!toU32(f[4], &wd))  { configErr("watchdog"); return; }

  if (pin != STIM_PIN)                        { configErr("pin-mismatch"); return; }
  if (pw < MIN_PULSE_US || pw > MAX_PULSE_US) { configErr("width-range"); return; }
  if (fr < MIN_FREQ_HZ  || fr > MAX_FREQ_HZ)  { configErr("freq-range"); return; }
  if (wd < MIN_WATCHDOG_MS || wd > MAX_WATCHDOG_MS) { configErr("watchdog-range"); return; }

  uint32_t newPeriodUs = (uint32_t)(1e6f / fr);
  // Duty cycle >= 100% would make the "end of HIGH" condition unreachable and
  // hold the laser on continuously. Reject rather than clamp: silently running
  // a different pulse width than requested is worse than refusing.
  if (pw >= newPeriodUs) { configErr("duty>=100%"); return; }

  // Commit. Stop first so a reconfiguration can never straddle a pulse.
  stopStim();
  pulseWidthUs = pw;
  frequencyHz  = fr;
  periodUs     = newPeriodUs;
  maxPulses    = mp;
  watchdogMs   = wd;
  configured   = true;

  Serial.print("CONFIG OK pin=");      Serial.print(STIM_PIN);
  Serial.print(" pulseWidthUs=");      Serial.print(pulseWidthUs);
  Serial.print(" frequencyHz=");       Serial.print(frequencyHz, 4);
  Serial.print(" periodUs=");          Serial.print(periodUs);
  Serial.print(" maxPulses=");         Serial.print(maxPulses);
  Serial.print(" watchdogMs=");        Serial.print(watchdogMs);
  Serial.print(" dutyPct=");           Serial.println(100.0f * pulseWidthUs / periodUs, 2);

  if (pulseWidthUs * 2 > periodUs) {
    Serial.println("CONFIG WARN duty cycle above 50%");
  }
}

void setup() {
  // FIRST, before anything else: drive the gate LOW. Everything below this can
  // block, fail, or take time; none of it may happen with the laser floating.
  pinMode(STIM_PIN, OUTPUT);
  digitalWrite(STIM_PIN, LOW);
  pinIsHigh = false;

  Serial.begin(BAUD);
  // Non-blocking USB writes: drop output rather than stall the pulse loop when
  // the host is not draining the CDC endpoint.
  Serial.setTxTimeoutMs(0);
  Serial.setTimeout(50);

  lastCmdMs = millis();

  // Banner. The PC watches for this line and (re-)sends its config, so a board
  // that resets mid-session recovers on its own instead of needing a re-flash.
  Serial.println("STIM controller ready. Send config line: "
                 "pin,pulseWidth_us,frequency_Hz,maxPulses,watchdog_ms");
}

void loop() {
  // --- Service incoming bytes (never blocks) ---------------------------
  while (Serial.available() > 0) {
    int c = Serial.read();
    // A command byte landing INSIDE a partially-received config line means the
    // line is corrupt (our writes are mutex-serialised on the PC side, so a
    // command can never legitimately appear mid-line). Poison it -- otherwise
    // the command char is simply consumed and the surviving digits parse as a
    // valid but WRONG config. The command itself is still honoured: refusing a
    // real 'X' would be worse than acting on a spurious one, which the watchdog
    // bounds anyway.
    if (cfgLen > 0 && strchr("SsXxKk", c) != NULL) cfgInvalid = true;
    switch (c) {
      case 'S': case 's':
        startStim();
        break;
      case 'X': case 'x':
        stopStim();
        lastCmdMs = millis();
        break;
      case 'K': case 'k':
        lastCmdMs = millis();
        break;
      case '\n': case '\r':
        if (cfgLen > 0) {
          if (cfgOverflow)      configErr("too-long");
          else if (cfgInvalid)  configErr("bad-char");
          else {
            cfgBuf[cfgLen] = '\0';
            applyConfig(cfgBuf);
          }
        }
        cfgLen = 0;
        cfgOverflow = false;
        cfgInvalid = false;
        break;
      default:
        if ((c >= '0' && c <= '9') || c == ',' || c == '.' || c == '-') {
          if (cfgLen < sizeof(cfgBuf) - 1) {
            cfgBuf[cfgLen++] = (char)c;
          } else {
            cfgOverflow = true;
          }
        } else if (cfgLen > 0) {
          // A byte outside the config charset landed INSIDE a config line.
          // It must poison the line, not be silently skipped: dropping it
          // would turn a corrupted "5x000" into a perfectly valid 5000, i.e.
          // a wrong-but-accepted config. Noise arriving between lines
          // (cfgLen == 0) is just ignored.
          cfgInvalid = true;
        }
        break;
    }
  }

  uint32_t nowMs = millis();

  // --- Watchdog: fail safe to OFF --------------------------------------
  // Keys off the PIN as well as `stimulating`, so it still fires in any state
  // where the two have diverged -- which is exactly when it is needed.
  if ((stimulating || pinIsHigh) &&
      (uint32_t)(nowMs - lastCmdMs) > watchdogMs) {
    stopStim();
    Serial.println("WATCHDOG: no keepalive -> stim OFF");
  }

  // --- Absolute train cap ----------------------------------------------
  if (stimulating && (uint32_t)(nowMs - trainStartMs) > MAX_TRAIN_MS) {
    stopStim();
    Serial.println("LIMIT: max train time -> stim OFF");
  }

  // --- Pulse train: state DERIVED from elapsed time --------------------
  // No accumulated phase, so a delayed loop skips the pulses that elapsed
  // rather than firing them back-to-back with the gate held HIGH.
  if (stimulating) {
    uint32_t elapsed = micros() - trainStartUs;   // unsigned: wrap-safe
    uint32_t idx     = elapsed / periodUs;        // periodUs > 0 by validation
    uint32_t phase   = elapsed - idx * periodUs;

    if (maxPulses > 0 && idx >= maxPulses) {
      stopStim();
    } else {
      if (idx != lastPulseIdx) {
        if (idx > lastPulseIdx + 1) {
          Serial.print("WARN skipped pulses: ");
          Serial.println(idx - lastPulseIdx - 1);
        }
        lastPulseIdx = idx;
        emitPulseMark(idx);
      }
      setPin(phase < pulseWidthUs);
    }
  }
}
