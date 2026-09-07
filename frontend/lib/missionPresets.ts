/**
 * Mission profiles required by PS 26054 section E ("Engine behavior simulation
 * during: High Altitude, Endurance mission, Hot-weather operation, Rapid
 * throttle transitions").
 *
 * A profile is a list of legs played in order by the simulator page. Each leg
 * holds the setpoints to hold for `seconds` of SIMULATED time - the same units
 * rawTelemetry.time reports, so a leg boundary is checked against the twin's own
 * clock rather than wall-clock, and stays correct if physics ever falls behind
 * real time.
 *
 * `isaDevC` is the ISA temperature deviation in degrees C (hot-day operation).
 * The backend ignores it until the ambient model lands; sending it early is
 * harmless because ParamUpdate is a pydantic model and pydantic v2 drops unknown
 * fields by default.
 */

export type MissionLeg = {
  label: string;
  seconds: number;
  altitude: number;   // m
  throttle: number;   // 0..1
  airspeed: number;   // m/s
  aoa: number;        // deg
  isaDevC?: number;   // ISA deviation, deg C
};

export type MissionPreset = {
  id: string;
  name: string;
  tagline: string;
  description: string;
  /** What a judge/operator should watch for while it plays. */
  watchFor: string;
  legs: MissionLeg[];
};

export const MISSION_PRESETS: MissionPreset[] = [
  {
    id: "high_altitude",
    name: "High Altitude",
    tagline: "Service ceiling ingress",
    description:
      "Climb to a high-altitude ISR station and hold. Air density falls with altitude, so the same throttle setting yields progressively less power and thrust.",
    watchFor:
      "Thrust margin narrowing as air density drops, and EGT rising on the sustained climb.",
    legs: [
      { label: "Climb to 3000 m", seconds: 120, altitude: 3000, throttle: 0.85, airspeed: 45, aoa: 4 },
      { label: "Climb to 5500 m", seconds: 150, altitude: 5500, throttle: 0.9, airspeed: 50, aoa: 3 },
      { label: "Station keeping", seconds: 240, altitude: 5500, throttle: 0.62, airspeed: 55, aoa: 2 },
    ],
  },
  {
    id: "endurance",
    name: "Endurance",
    tagline: "Long-loiter ISR",
    description:
      "Extended low-power cruise at best-endurance settings, the dominant regime for a MALE UAV on a long ISR tasking. Wear accumulates slowly and steadily.",
    watchFor:
      "Wear and RUL trending down gradually with no fault signature - the baseline a degradation trend is measured against.",
    legs: [
      { label: "Climb to cruise", seconds: 90, altitude: 2500, throttle: 0.75, airspeed: 45, aoa: 4 },
      { label: "Loiter leg 1", seconds: 300, altitude: 2500, throttle: 0.45, airspeed: 38, aoa: 3 },
      { label: "Loiter leg 2", seconds: 300, altitude: 2800, throttle: 0.42, airspeed: 38, aoa: 3 },
      { label: "Loiter leg 3", seconds: 300, altitude: 2500, throttle: 0.45, airspeed: 40, aoa: 3 },
    ],
  },
  {
    id: "hot_weather",
    name: "Hot Weather",
    tagline: "ISA +30 C, hot and high",
    description:
      "Operation from a hot airfield. Elevated ambient temperature lowers air density at every altitude - the classic hot-and-high problem - so available power and climb performance degrade.",
    watchFor:
      "Reduced power and thrust at the same throttle versus a standard day, plus elevated oil temperature.",
    legs: [
      { label: "Hot-day departure", seconds: 120, altitude: 1500, throttle: 0.9, airspeed: 40, aoa: 5, isaDevC: 30 },
      { label: "Hot-day climb", seconds: 150, altitude: 3500, throttle: 0.88, airspeed: 48, aoa: 4, isaDevC: 30 },
      { label: "Hot-day cruise", seconds: 240, altitude: 3500, throttle: 0.6, airspeed: 52, aoa: 2, isaDevC: 30 },
    ],
  },
  {
    id: "rapid_throttle",
    name: "Rapid Throttle",
    tagline: "Transient response",
    description:
      "Repeated large, fast throttle transitions. Thermal states lag the commanded power by design (EGT, CHT and oil temperature all have different time constants), so this profile stresses transient behaviour rather than steady state.",
    watchFor:
      "EGT reacting fast while CHT and oil temperature lag well behind - the three time constants pulling apart.",
    legs: [
      { label: "Establish cruise", seconds: 90, altitude: 2000, throttle: 0.55, airspeed: 45, aoa: 3 },
      { label: "Full power", seconds: 45, altitude: 2000, throttle: 1.0, airspeed: 55, aoa: 3 },
      { label: "Idle", seconds: 45, altitude: 2000, throttle: 0.15, airspeed: 35, aoa: 4 },
      { label: "Full power", seconds: 45, altitude: 2000, throttle: 1.0, airspeed: 55, aoa: 3 },
      { label: "Idle", seconds: 45, altitude: 2000, throttle: 0.15, airspeed: 35, aoa: 4 },
      { label: "Recover to cruise", seconds: 120, altitude: 2000, throttle: 0.6, airspeed: 48, aoa: 3 },
    ],
  },
];

export function getPreset(id: string | null | undefined): MissionPreset | null {
  if (!id) return null;
  return MISSION_PRESETS.find((p) => p.id === id) ?? null;
}

/** Total simulated seconds a profile runs for. */
export function presetDuration(p: MissionPreset): number {
  return p.legs.reduce((sum, l) => sum + l.seconds, 0);
}

/** Which leg is active at simulated time `t`, or the last leg once past the end. */
export function legAt(p: MissionPreset, t: number): { leg: MissionLeg; index: number } {
  let acc = 0;
  for (let i = 0; i < p.legs.length; i++) {
    acc += p.legs[i].seconds;
    if (t < acc) return { leg: p.legs[i], index: i };
  }
  return { leg: p.legs[p.legs.length - 1], index: p.legs.length - 1 };
}
