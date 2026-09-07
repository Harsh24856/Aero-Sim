/**
 * Display-unit conversions. FRONTEND-ONLY, deliberately.
 *
 * The physics twin, the API and the database all speak SI throughout: metres,
 * m/s, degrees C. Nothing here changes any of that. Converting at the display
 * layer keeps one canonical unit system in the data and one aviation-familiar
 * unit system in front of the operator, which is how a real GCS is built - the
 * moment stored values start carrying display units, every consumer has to know
 * which flavour it received.
 *
 * Airspeed is shown in KNOTS because that is what aircrew, ICAO and every UAV
 * ground control station actually use. Altitude keeps metres as the primary
 * (the airframe envelope in physics.py is defined in metres) with feet
 * alongside, since flight levels are universally read in feet.
 */

import { simRulHoursToReal, ROTAX_914_TBO_HOURS } from "./timeScale";

export const MS_TO_KNOTS = 1.943844;   // 1 m/s = 1.943844 kt
export const MS_TO_KMH = 3.6;
export const M_TO_FEET = 3.280839895;

export function msToKnots(ms: number): number { return ms * MS_TO_KNOTS; }
export function msToKmh(ms: number): number { return ms * MS_TO_KMH; }
export function mToFeet(m: number): number { return m * M_TO_FEET; }

const nf = (v: number, d = 0) =>
  v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

/** Airspeed as knots, e.g. "89 kt". Null-safe. */
export function formatAirspeed(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "--";
  return `${nf(msToKnots(ms))} kt`;
}

/** Secondary airspeed readout, e.g. "46.0 m/s / 166 km/h". */
export function formatAirspeedSecondary(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "";
  return `${nf(ms, 1)} m/s / ${nf(msToKmh(ms))} km/h`;
}

/** Altitude in metres, e.g. "2,400 m". */
export function formatAltitude(m: number | null | undefined): string {
  if (m == null || Number.isNaN(m)) return "--";
  return `${nf(m)} m`;
}

/** Altitude in feet, e.g. "7,874 ft". */
export function formatAltitudeFeet(m: number | null | undefined): string {
  if (m == null || Number.isNaN(m)) return "";
  return `${nf(mToFeet(m))} ft`;
}

/**
 * RUL in REAL-WORLD hours.
 *
 * ai.py's rul_hours_internal (stored as simulations.final_rul_hours) is on the
 * COMPRESSED simulated timescale and is documented as such at its source.
 * Printing it raw as "4.17 h" reads as four hours of flight left when it
 * actually represents ~1,501 real hours against a 2,000 h TBO - off by a factor
 * of 360. Always convert before showing it to a person.
 */
export function formatRulRealHours(simRulHours: number | null | undefined): string {
  if (simRulHours == null || Number.isNaN(simRulHours)) return "--";
  return `${nf(simRulHoursToReal(simRulHours))} h`;
}

/** e.g. "of 2,000 h TBO" - the reference the number above should be read against. */
export function tboReference(): string {
  return `of ${nf(ROTAX_914_TBO_HOURS)} h TBO`;
}

/** RUL as a share of the engine's real TBO, e.g. "75%". */
export function formatRulTboPercent(simRulHours: number | null | undefined): string {
  if (simRulHours == null || Number.isNaN(simRulHours)) return "--";
  const pct = (simRulHoursToReal(simRulHours) / ROTAX_914_TBO_HOURS) * 100;
  return `${nf(Math.max(0, Math.min(100, pct)), 0)}%`;
}
