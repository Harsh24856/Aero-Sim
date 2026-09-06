// Single source of truth for converting our compressed simulation timescale into
// real-world-meaningful units. Used everywhere RUL hours or flight duration are
// displayed - defined ONCE here specifically because duplicating this constant
// across files already caused a real bug (the /telemetry list page was showing
// raw un-scaled values while the detail page had been fixed separately).
//
// Real Rotax 914 TBO (Time Between Overhauls): 2,000 flight hours or 15 years
// calendar life, whichever comes first (Rotax service data / rotax-owner.com).
export const ROTAX_914_TBO_HOURS = 2000;

// generate_multi_engine_data_v2.py's MAX_DURATION (20,000s) is the censoring
// cutoff used when generating training data - the longest a single simulated
// flight runs before being cut off as "censored" rather than failed. Treating
// this as representing a full wear-to-failure cycle mapped onto the real 2,000h
// TBO gives a single, consistent scale factor for both RUL and elapsed time,
// since they are both "time" in the same compressed simulation timescale.
const MAX_SIMULATED_DURATION_S = 20000;
const MAX_SIMULATED_DURATION_H = MAX_SIMULATED_DURATION_S / 3600;

// = 360: one simulated second represents 360 real seconds (6 real minutes) of
// equivalent flight time. A 20-second simulate session therefore represents
// ~2 real hours of flight - meaningful, without ever needing to actually run a
// session for 2,000 hours.
export const TIME_SCALE = ROTAX_914_TBO_HOURS / MAX_SIMULATED_DURATION_H;

// ai.py's rul_hours_internal (stored as simulations.final_rul_hours) is
// explicitly documented THERE as simulated-timescale, NOT real-world hours -
// this is the same TIME_SCALE, named for its use at that specific call site.
export const RUL_SCALE = TIME_SCALE;

export function simRulHoursToReal(simRulHours: number): number {
  return simRulHours * RUL_SCALE;
}

// REAL BUG FIXED HERE: this claimed to match ai.py's formula but never did -
// the elapsedSimSeconds parameter was accepted and then ignored (note it was
// prefixed with _, the convention for "intentionally unused"), so this
// silently used a completely different formula instead. ai.py's run_inference()
// computes a SELF-NORMALIZING percent: 100 * rul_hours / (elapsed_hours +
// rul_hours) - what fraction of this flight's own implied total life remains.
// A short session with a small elapsed_hours and a healthy rul_hours correctly
// reports close to 100% under that formula. The old code instead computed
// rul_hours as a flat fraction of the fixed 5.556h max simulated duration,
// which is a genuinely different (and for short sessions, much lower) number -
// confirmed: a live 98% (ai.py) was showing as 65% here for the identical run.
export function simRulHoursToPercent(simRulHours: number, elapsedSimSeconds?: number): number {
  if (elapsedSimSeconds == null) {
    // No saved elapsed time (older rows, or a snapshot without it) - fall back
    // to the flat-ratio approximation rather than divide by an unknown elapsed time.
    const clamped = Math.max(0, Math.min(simRulHours, MAX_SIMULATED_DURATION_H));
    return (clamped / MAX_SIMULATED_DURATION_H) * 100;
  }
  const elapsedHours = elapsedSimSeconds / 3600;
  const impliedTotalLife = elapsedHours + Math.max(simRulHours, 1e-6);
  return (100 * simRulHours) / impliedTotalLife;
}

// Converts simulated elapsed seconds (e.g. from raw telemetry's "time" field,
// or a scenario's time_offset_s) into real-world-equivalent HOURS of flight.
export function simSecondsToRealHours(simSeconds: number): number {
  return (simSeconds * TIME_SCALE) / 3600;
}
