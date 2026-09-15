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

// ---- Physics v3 -------------------------------------------------------------
// v3 runs (simulations.model_version = 'v3') store RUL in REAL ENGINE HOURS against
// the engine's TBO, and their clock is not compressed, so none of the v2 scaling
// above applies to them. Every page decides through these helpers, never by
// calling the v2 functions directly on a row whose version it has not checked.
// Rows written before the v3 migration default to 'v2' and render as legacy.
export type ModelVersion = "v2" | "v3";

export function modelVersionOf(row: { model_version?: string | null } | null | undefined): ModelVersion {
  return row?.model_version === "v3" ? "v3" : "v2";
}

/** TBO for a v3 run: the column, else the physics snapshot that carries it. */
export function tboHoursOf(row: {
  tbo_hours?: number | null;
  final_telemetry?: Record<string, unknown> | null;
} | null | undefined): number | null {
  if (row?.tbo_hours != null) return row.tbo_hours;
  const t = row?.final_telemetry?.tbo_hours;
  return typeof t === "number" ? t : null;
}

/**
 * v3: engine hours the physics has put on the engine, wear x TBO. This - not the
 * simulated clock - is the time that matters on v3: wear is time-compressed so a
 * demo flight ages the engine by tens of hours in a few minutes, and RUL counts
 * down on this same scale. Showing simulated seconds as "flight time" (0.06 h)
 * next to an RUL that just dropped 62 engine hours read as broken.
 */
export function engineHoursOfWear(wear: number | null | undefined, tboHours: number | null | undefined): number | null {
  if (typeof wear !== "number" || !Number.isFinite(wear) || !tboHours) return null;
  return Math.max(0, wear) * tboHours;
}

/** Engine hours on the engine at the end of a stored v3 run (final_telemetry wear x TBO). */
export function engineHoursOf(row: {
  tbo_hours?: number | null;
  final_telemetry?: Record<string, unknown> | null;
} | null | undefined): number | null {
  const wear = row?.final_telemetry?.wear;
  return engineHoursOfWear(typeof wear === "number" ? wear : null, tboHoursOf(row));
}

/** Simulated clock as m:ss or h:mm:ss. */
export function formatSimClock(simSeconds: number | null | undefined): string {
  if (typeof simSeconds !== "number" || !Number.isFinite(simSeconds)) return "--";
  const t = Math.max(0, Math.floor(simSeconds));
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), sec = t % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

/** Flight hours from simulated seconds, on the run's own timescale. v3 callers should
 *  prefer engine hours (engineHoursOfWear / engineHoursOf). */
export function flightHours(simSeconds: number, version: ModelVersion): number {
  return version === "v3" ? simSeconds / 3600 : simSecondsToRealHours(simSeconds);
}

/** RUL percent for display. v3: percent of TBO. v2: the self-normalising legacy formula. */
export function rulPercentOf(
  rulHours: number,
  version: ModelVersion,
  elapsedSimSeconds?: number,
  tboHours?: number | null,
): number | null {
  if (version === "v3") {
    return tboHours ? Math.max(0, Math.min(100, (100 * rulHours) / tboHours)) : null;
  }
  return Math.min(100, simRulHoursToPercent(rulHours, elapsedSimSeconds));
}
