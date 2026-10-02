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
 * RUL on the model's OWN scale, e.g. "4.17 sim h".
 *
 * Deliberately NOT converted to real-world hours. timeScale.ts carries a
 * TIME_SCALE that maps the training generator's 20,000s censoring cutoff onto
 * the Rotax 2,000h TBO, but that equivalence is an assumption, not a measured
 * property of the model - and applying it to RUL breaks badly at the top end:
 * ai.py's meaningful ceiling is MAX_SIM_LIFE_HOURS = 20000/3600 = 5.556 sim h,
 * yet 29 of 143 recorded runs sit ABOVE it (max 546.6) because the model
 * extrapolates. Scaling 546.6 by 360 would print ~197,000 hours of remaining
 * life with a straight face.
 *
 * So hours are shown on the model's native scale and the percentage - which
 * ai.py computes itself, self-normalising against the flight's own implied
 * life - is the single number used for comparison everywhere.
 */
export function formatRulSimHours(simRulHours: number | null | undefined): string {
  if (simRulHours == null || Number.isNaN(simRulHours)) return "--";
  return `${nf(simRulHours, 2)} sim h`;
}

/** ai.py's MAX_SIM_LIFE_HOURS - beyond this the model is extrapolating. */
export const MAX_SIM_LIFE_HOURS = 20000 / 3600;

export function isRulExtrapolated(simRulHours: number | null | undefined): boolean {
  return simRulHours != null && simRulHours > MAX_SIM_LIFE_HOURS;
}

/** RUL text on the run's own scale: "1,234 engine h" (v3) or "4.17 sim h" (v2). */
export function formatRul(rulHours: number | null | undefined, version: "v2" | "v3" | "v4" | "v5"): string {
  if (rulHours == null || Number.isNaN(rulHours)) return "--";
  return version === "v2" ? formatRulSimHours(rulHours) : `${nf(rulHours)} engine h`;
}

/** v3: beyond 105% of TBO (aiv3.py RUL_OUT_OF_RANGE_FRAC). v2: beyond MAX_SIM_LIFE_HOURS. */
export function isRulOutOfRange(
  rulHours: number | null | undefined,
  version: "v2" | "v3" | "v4" | "v5",
  tboHours?: number | null,
): boolean {
  if (rulHours == null) return false;
  return version === "v2" ? isRulExtrapolated(rulHours) : tboHours != null && rulHours > 1.05 * tboHours;
}
