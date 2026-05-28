// =====================================================================
//  Optogenetics stim controller
//  ---------------------------------------------------------------------
//  Receives commands from the RT-opto controller PC and drives a single
//  digital TTL pin that gates a laser. This is a SEPARATE board from the
//  camera-trigger Teensy (campy/trigger/trigger.ino).
//
//  SERIAL PROTOCOL  (115200 baud)
//  ------------------------------
//  1) Config line (sent once at startup, and again after a 'C' command):
//
//         pin,pulseWidth_us,frequency_Hz,maxPulses,watchdog_ms\n
//
//     e.g.  "9,5000,20,0,500"
//       pin          : digital output pin wired to the laser driver gate
//       pulseWidth_us: HIGH time of each pulse, microseconds
//       frequency_Hz : pulse repetition rate (period = 1e6 / frequency_Hz)
//       maxPulses    : pulses per activation; 0 = unlimited (continuous)
//       watchdog_ms  : auto-OFF if no command/keepalive within this window
//                      (0 disables the watchdog -- not recommended)
//
//  2) Runtime single-character commands:
//       'S' : START stimulating (resets pulse counter + watchdog)
//       'X' : STOP  stimulating immediately (pin LOW)
//       'K' : KEEPALIVE (refreshes the watchdog while stimulating)
//       'C' : reconfigure -- read a new config line
//
//  Pulse generation is non-blocking (micros() based) so commands are always
//  serviced promptly. The watchdog is the key safety feature: if the PC
//  controller crashes or the cable is pulled, the laser turns itself off.
// =====================================================================

const uint32_t BAUD = 115200;

// --- Configurable parameters (set over serial) -----------------------
int      stimPin       = 13;     // overwritten by config line
uint32_t pulseWidthUs  = 5000;
float    frequencyHz   = 20.0;
uint32_t periodUs      = 50000;  // derived = 1e6 / frequencyHz
long     maxPulses     = 0;      // 0 = unlimited
uint32_t watchdogMs    = 500;    // 0 = disabled

// --- Runtime state ----------------------------------------------------
bool     stimulating   = false;
uint32_t periodStartUs = 0;      // micros() at start of current pulse period
uint32_t lastCmdMs     = 0;      // millis() of last S/K (watchdog reference)
long     pulseCount    = 0;      // pulses emitted in current activation
bool     pinIsHigh     = false;

void setPin(bool high) {
  digitalWrite(stimPin, high ? HIGH : LOW);
  pinIsHigh = high;
}

void computePeriod() {
  if (frequencyHz <= 0.0) {
    periodUs = 0xFFFFFFFF;       // effectively never pulses
  } else {
    periodUs = (uint32_t)(1e6 / frequencyHz);
  }
}

void stopStim() {
  stimulating = false;
  setPin(false);
}

void startStim() {
  // Re-init the pin in case a previous config changed it.
  pinMode(stimPin, OUTPUT);
  pulseCount = 0;
  periodStartUs = micros();
  lastCmdMs = millis();
  stimulating = true;
  setPin(true);                  // first pulse begins immediately
}

// Block until a full config line (5 numbers) has been parsed.
void readConfig() {
  while (Serial.available() == 0) {}
  stopStim();                    // safety: never read config while firing

  int newPin       = (int)Serial.parseInt();
  uint32_t pw      = (uint32_t)Serial.parseFloat();
  float fr         = Serial.parseFloat();
  long mp          = (long)Serial.parseInt();
  uint32_t wd      = (uint32_t)Serial.parseInt();

  stimPin      = newPin;
  pulseWidthUs = pw;
  frequencyHz  = fr;
  maxPulses    = mp;
  watchdogMs   = wd;
  computePeriod();

  pinMode(stimPin, OUTPUT);
  setPin(false);

  Serial.println();
  Serial.print("CONFIG pin=");        Serial.print(stimPin);
  Serial.print(" pulseWidthUs=");     Serial.print(pulseWidthUs);
  Serial.print(" frequencyHz=");      Serial.print(frequencyHz);
  Serial.print(" periodUs=");         Serial.print(periodUs);
  Serial.print(" maxPulses=");        Serial.print(maxPulses);
  Serial.print(" watchdogMs=");       Serial.println(watchdogMs);
}

void setup() {
  Serial.begin(BAUD);
  Serial.setTimeout(2000);
  delay(200);
  Serial.println("STIM controller ready. Send config line: "
                 "pin,pulseWidth_us,frequency_Hz,maxPulses,watchdog_ms");
  readConfig();
}

void loop() {
  // --- Service incoming commands --------------------------------------
  while (Serial.available() > 0) {
    int c = Serial.read();
    switch (c) {
      case 'S': case 's':
        startStim();
        break;
      case 'X': case 'x':
        stopStim();
        break;
      case 'K': case 'k':
        lastCmdMs = millis();
        break;
      case 'C': case 'c':
        readConfig();
        break;
      default:
        break;                   // ignore stray bytes / newlines
    }
  }

  // --- Watchdog: fail safe to OFF -------------------------------------
  if (stimulating && watchdogMs > 0 &&
      (uint32_t)(millis() - lastCmdMs) > watchdogMs) {
    stopStim();
    Serial.println("WATCHDOG: no keepalive -> stim OFF");
  }

  // --- Non-blocking pulse train ---------------------------------------
  if (stimulating) {
    uint32_t now = micros();
    uint32_t elapsed = now - periodStartUs;   // unsigned: wrap-safe

    if (elapsed >= periodUs) {
      // One full period elapsed -> count the pulse just completed.
      pulseCount++;
      if (maxPulses > 0 && pulseCount >= maxPulses) {
        stopStim();
        return;
      }
      periodStartUs += periodUs;                // advance phase (no drift)
      setPin(true);                             // begin next pulse
    } else if (elapsed >= pulseWidthUs && pinIsHigh) {
      setPin(false);                            // end of HIGH portion
    } else if (elapsed < pulseWidthUs && !pinIsHigh) {
      setPin(true);                             // (re)assert HIGH within pulse
    }
  }
}
