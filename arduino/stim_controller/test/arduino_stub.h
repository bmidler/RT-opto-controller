// Minimal Arduino API stub so stim_controller.ino can be compiled and driven
// on a host with a fake clock and a scripted serial stream.
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <deque>
#include <vector>

#define HIGH 1
#define LOW 0
#define OUTPUT 1

static uint32_t g_micros = 0;
static inline uint32_t micros() { return g_micros; }
static inline uint32_t millis() { return g_micros / 1000; }

static int  g_pinState[64];
static bool g_pinIsOutput[64];
static std::vector<std::pair<uint32_t,int>> g_pinHistory;  // (micros, level)

static inline void pinMode(uint8_t p, uint8_t m) { g_pinIsOutput[p] = (m == OUTPUT); }
static inline void digitalWrite(uint8_t p, uint8_t v) {
  if (g_pinState[p] != v) g_pinHistory.push_back({g_micros, v});
  g_pinState[p] = v;
}

struct FakeSerial {
  std::deque<char> rx;
  std::vector<std::string> lines;
  std::string cur;

  void begin(uint32_t) {}
  void setTxTimeoutMs(uint32_t) {}
  void setTimeout(uint32_t) {}
  int  available() { return (int)rx.size(); }
  int  availableForWrite() { return 256; }
  int  read() { if (rx.empty()) return -1; char c = rx.front(); rx.pop_front(); return c; }
  void feed(const char *s) { while (*s) rx.push_back(*s++); }

  void emit(const std::string &s) { cur += s; }
  void flushLine() { lines.push_back(cur); cur.clear(); }

  void print(char c)          { emit(std::string(1, c)); }
  void print(const char *s)   { emit(s); }
  void print(int v)           { char b[32]; snprintf(b,32,"%d",v); emit(b); }
  void print(unsigned v)      { char b[32]; snprintf(b,32,"%u",v); emit(b); }
  void print(unsigned long v) { char b[32]; snprintf(b,32,"%lu",v); emit(b); }
  void print(float v, int d)  { char b[64]; snprintf(b,64,"%.*f",d,v); emit(b); }
  void println()                  { flushLine(); }
  void println(const char *s)     { emit(s); flushLine(); }
  void println(int v)             { print(v); flushLine(); }
  void println(unsigned v)        { print(v); flushLine(); }
  void println(unsigned long v)   { print(v); flushLine(); }
  void println(float v, int d)    { print(v,d); flushLine(); }
};
static FakeSerial Serial;
