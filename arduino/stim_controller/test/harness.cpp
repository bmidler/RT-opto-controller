#include "arduino_stub.h"
#include "../stim_controller.ino"

static int failures = 0;
static void check(bool ok, const char *what) {
  printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) failures++;
}
static bool sawLine(const char *prefix) {
  for (auto &l : Serial.lines) if (l.rfind(prefix, 0) == 0) return true;
  return false;
}
static void clearOut() { Serial.lines.clear(); Serial.cur.clear(); }
static void advance(uint32_t us, uint32_t stepUs = 100) {
  for (uint32_t i = 0; i < us; i += stepUs) { g_micros += stepUs; loop(); }
}
static int pin() { return g_pinState[STIM_PIN]; }

int main() {
  setup();
  printf("\n=== after setup(), before any config ===\n");
  check(g_pinIsOutput[STIM_PIN], "STIM_PIN configured as OUTPUT in setup()");
  check(pin() == LOW,            "STIM_PIN driven LOW in setup() (no floating window)");
  check(!configured,             "board starts unconfigured");

  printf("\n=== 'S' before any config is refused ===\n");
  clearOut(); Serial.feed("S"); loop();
  check(pin() == LOW,               "pin stays LOW");
  check(sawLine("ERR unconfigured"), "board reports ERR unconfigured");

  printf("\n=== malformed config lines are REJECTED (old code accepted these) ===\n");
  struct { const char *line; const char *why; } bad[] = {
    {"9,5000\n",            "truncated line"},
    {"9,5000,3.0,75,0\n",   "watchdog_ms = 0 (would disable the watchdog)"},
    {"8,5000,3.0,75,500\n", "wrong pin"},
    {"9,400000,3.0,75,500\n","duty cycle >= 100%"},
    {"9,5000,0,75,500\n",   "frequency 0"},
    {"9,5000,3.0,75,500,7\n","too many fields"},
    {"9,5z00,3.0,75,500\n", "noise byte inside a field"},
    {"9,5K000,3.0,75,500\n","command byte inside a config line"},
    {"9,5000,3.0,75,50000\n","watchdog above the allowed range"},
  };
  for (auto &b : bad) {
    clearOut(); Serial.feed(b.line); loop();
    bool rejected = sawLine("CONFIG ERR") && !configured;
    printf("  %-40s -> %s\n", b.why, rejected ? "rejected" : "ACCEPTED <-- BUG");
    if (!rejected) failures++;
  }

  printf("\n=== the real config line from stim.py is accepted ===\n");
  clearOut(); Serial.feed("9,5000,3.0000,75,500\n"); loop();
  check(sawLine("CONFIG OK"), "board replies CONFIG OK");
  check(configured,           "board is now configured");
  check(periodUs == 333333,   "periodUs derived from 3 Hz");
  check(pulseWidthUs == 5000, "pulseWidthUs = 5000");
  check(watchdogMs == 500,    "watchdogMs = 500");
  check(maxPulses == 75,      "maxPulses = 75");
  check(pin() == LOW,         "still LOW after configuring");

  printf("\n=== config accepted mid-run (the Nano ESP32 never resets on open) ===\n");
  clearOut(); Serial.feed("9,2000,10.0000,50,400\n"); loop();
  check(sawLine("CONFIG OK") && pulseWidthUs == 2000 && periodUs == 100000,
        "second config line applied while loop() was already running");
  Serial.feed("9,5000,3.0000,75,500\n"); loop();   // restore

  printf("\n=== pulse shape: 5 ms HIGH every 333.3 ms ===\n");
  clearOut(); Serial.feed("S"); loop();
  check(pin() == HIGH, "pin HIGH immediately on 'S'");
  advance(4000);  Serial.feed("K"); loop();
  check(pin() == HIGH, "still HIGH at t=4 ms (< 5 ms pulse width)");
  advance(2000);  Serial.feed("K"); loop();
  check(pin() == LOW,  "LOW at t=6 ms (past the 5 ms pulse)");
  // walk to just after the second period boundary, feeding keepalives
  for (int i = 0; i < 400; i++) { advance(1000); Serial.feed("K"); loop(); }
  check(pin() == LOW,  "LOW mid-period at t=406 ms");

  printf("\n=== stalled loop must NOT produce a stuck-HIGH catch-up burst ===\n");
  // Jump the clock 10 periods forward in one go, as a blocked loop would.
  g_micros += 10 * 333333; Serial.feed("K"); loop();
  check(pin() == LOW, "pin LOW after a 10-period stall (phase-derived, not accumulated)");
  check(sawLine("WARN skipped pulses: "), "board reports the skipped pulses");

  printf("\n=== watchdog turns the laser off when keepalives stop ===\n");
  clearOut(); Serial.feed("S"); loop();
  check(pin() == HIGH, "HIGH after 'S'");
  advance(600000);   // 600 ms with no keepalive
  check(pin() == LOW,                  "pin LOW after the 500 ms watchdog");
  check(sawLine("WATCHDOG"),           "board reports the watchdog trip");
  check(!stimulating,                  "stimulating cleared");

  printf("\n=== maxPulses ends the train ===\n");
  clearOut(); Serial.feed("S"); loop();
  for (int i = 0; i < 26000; i++) { advance(1000); Serial.feed("K"); loop(); }
  check(!stimulating, "train stopped by maxPulses=75 at ~25 s");
  check(pin() == LOW, "pin LOW after the train ends");

  printf("\n=== absolute train cap with maxPulses = 0 (unlimited) ===\n");
  clearOut(); Serial.feed("9,5000,3.0000,0,500\n"); loop();
  Serial.feed("S"); loop();
  for (int i = 0; i < 61000; i++) { advance(1000); Serial.feed("K"); loop(); }
  check(!stimulating,        "MAX_TRAIN_MS stopped an otherwise unlimited train");
  check(pin() == LOW,        "pin LOW after the cap");
  check(sawLine("LIMIT"),    "board reports the limit");

  printf("\n=== line noise never latches the pin or blocks ===\n");
  clearOut(); Serial.feed("9,5000,3.0000,75,500\n"); loop();
  Serial.feed("S"); loop();
  Serial.feed("C");            // the old firmware entered a blocking readConfig here
  Serial.feed("\x01\xff?zQ");  // random bytes
  loop();
  check(pin() == HIGH, "still pulsing normally; stray 'C'/noise ignored");
  advance(600000);
  check(pin() == LOW,  "watchdog still able to fire afterwards");

  printf("\n%s  (%d failure%s)\n", failures ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED",
         failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
