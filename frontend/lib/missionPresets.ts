/**
 * MALE-UAV mission profiles for the physics-v4 twin (PS 26054).
 *
 * Section E asks for engine behaviour during high altitude, endurance, hot-weather
 * operation and rapid throttle transitions; the v4 twin adds the question a MALE
 * operator actually has to answer mid-sortie - is that the engine or a sensor, and
 * can the aircraft finish the mission? Each profile is one sortie:
 *
 *   scenario  which engine flies (backend/scenarios_v4.py preset), chosen before Start
 *   env       its atmosphere, fuel and installation (the "Inputs to the engine"),
 *             mission-wide, and optionally changed per leg
 *   legs      setpoints held for `seconds` of FLIGHT time (rawTelemetry.time), so a
 *             boundary follows the twin's own clock, not the wall clock
 *   events    faults and sensor failures injected - and repaired - at mission times
 *
 * The mission clock starts when the AI gives its first answer (after its 128 s
 * window), so an event at t = 240 s is seen by an AI that is already watching.
 * Every leg stays inside the envelope the v4 models were trained on
 * (generate_dataset_v4.py MISSIONS): 28-58 m/s, up to 7,200 m, throttle 0.30-1.00.
 * The first leg is at least 150 s above that airspeed floor, so the post-takeoff
 * alert hold (main.py hold_alerts_outside_envelope) has cleared before any event.
 */

/** A subset of the v4 engine inputs (Meters.tsx ENGINE_INPUTS keys). */
export type MissionEnv = Partial<{
  isa_dev_c: number; qnh_offset_pa: number; humidity_frac: number; fuel_octane_mon: number;
  fuel_ethanol_frac: number; target_lambda: number; cooling_airflow_factor: number;
  electrical_load_a: number; oil_thermostat_open: boolean;
}>;

export type MissionLeg = {
  label: string;
  seconds: number;
  altitude: number;   // m
  throttle: number;   // 0..1
  airspeed: number;   // m/s
  aoa: number;        // deg
  env?: MissionEnv;   // changes from this leg on
};

/** POST /inject or /inject/clear bodies, fired once at `at` mission seconds. */
export type MissionAction =
  | { kind: "fault"; name: string; severity: number }
  | { kind: "sensor"; channel: string; type: string; severity: number }
  | { kind: "clear_fault"; name: string }
  | { kind: "clear_sensor"; channel: string };

export type MissionEvent = {
  at: number;
  label: string;
  action: MissionAction;
  /** Engines it applies to; omitted = all. The backend refuses a turbo fault on a 912. */
  engines?: string[];
};

export type MissionPreset = {
  id: string;
  name: string;
  tagline: string;
  /** Which PS 26054 section E condition (or v4 capability) it demonstrates. */
  requirement: string;
  description: string;
  /** What to watch in the cockpit while it plays. */
  watchFor: string;
  scenario: string;
  env?: MissionEnv;
  legs: MissionLeg[];
  events?: MissionEvent[];
};

const TURBO = ["Rotax_914_ULF", "Rotax_915_iS", "Rotax_916_iS"];

export const MISSION_PRESETS: MissionPreset[] = [
  {
    id: "isr_endurance",
    name: "ISR Endurance Sortie",
    tagline: "Transit, long loiter, return",
    requirement: "Endurance mission",
    description:
      "The MALE UAV's bread-and-butter tasking: climb out, transit to the area, orbit at best-endurance power for most of the sortie, and come home. A healthy engine, a standard day.",
    watchFor:
      "Nominal all the way. Engine vs healthy twin stays on top of each other, wear condition sits near the ground truth, and RUL follows the calendar countdown while the hour meter ages the engine about 3 h per real minute.",
    scenario: "healthy",
    legs: [
      { label: "Climb to patrol altitude", seconds: 180, altitude: 3000, throttle: 0.88, airspeed: 45, aoa: 5 },
      { label: "Transit to area", seconds: 240, altitude: 3500, throttle: 0.72, airspeed: 52, aoa: 2.5 },
      { label: "Loiter orbit 1", seconds: 360, altitude: 3500, throttle: 0.55, airspeed: 40, aoa: 5 },
      { label: "Loiter orbit 2", seconds: 360, altitude: 3200, throttle: 0.52, airspeed: 40, aoa: 5.5 },
      { label: "Return to base", seconds: 240, altitude: 1500, throttle: 0.45, airspeed: 48, aoa: 1 },
    ],
  },
  {
    id: "high_altitude",
    name: "High-Altitude Surveillance",
    tagline: "Station keeping at 6,500 m",
    requirement: "High altitude",
    description:
      "Climb to a high ISR station. Air density falls by half, so a naturally aspirated 912 loses power with every metre, while the turbo engines hold manifold pressure until the wastegate runs out of authority. On a turbo engine the turbocharger starts to degrade on station.",
    watchFor:
      "MAP and boost ratio against altitude, and the thrust margin narrowing. On a turbo engine, turbo degradation is only observable above its critical altitude (15,000 ft) - this is where the AI can see it.",
    scenario: "healthy",
    legs: [
      { label: "Climb to 3,000 m", seconds: 180, altitude: 3000, throttle: 0.95, airspeed: 45, aoa: 5 },
      { label: "Climb to 6,000 m", seconds: 240, altitude: 6000, throttle: 0.95, airspeed: 52, aoa: 3 },
      { label: "Station keeping, 6,500 m", seconds: 420, altitude: 6500, throttle: 0.85, airspeed: 55, aoa: 2.5 },
      { label: "Descend to 2,500 m", seconds: 240, altitude: 2500, throttle: 0.5, airspeed: 50, aoa: 1 },
    ],
    events: [
      { at: 480, label: "Turbocharger degrades (45%)", engines: TURBO,
        action: { kind: "fault", name: "turbo_degradation", severity: 0.45 } },
    ],
  },
  {
    id: "hot_high",
    name: "Hot-and-High Operations",
    tagline: "ISA +28 °C, humid, then the radiator clogs",
    requirement: "Hot-weather operation",
    description:
      "Departure from a hot, humid forward airfield. Density altitude cuts power and climb, and every temperature runs high. Mid-sortie the radiator starts to foul. The healthy twin flies the same hot day, so the day cancels out and only the cooling fault is left in the residuals.",
    watchFor:
      "CHT and oil temperature above a standard day on both the engine and its twin; then, after the fault, the engine pulling away from the twin while the twin stays put. Which part? should name cooling degradation, and the advisory asks for more airspeed and less power.",
    scenario: "healthy",
    env: { isa_dev_c: 28, humidity_frac: 0.7, cooling_airflow_factor: 0.95 },
    legs: [
      { label: "Hot-day departure", seconds: 180, altitude: 1200, throttle: 0.98, airspeed: 42, aoa: 6 },
      { label: "Hot-day climb", seconds: 240, altitude: 2500, throttle: 0.9, airspeed: 46, aoa: 4 },
      { label: "Cruise in the heat", seconds: 420, altitude: 2500, throttle: 0.72, airspeed: 50, aoa: 3 },
      { label: "Return, reduced power", seconds: 180, altitude: 1000, throttle: 0.45, airspeed: 46, aoa: 1 },
    ],
    events: [
      { at: 480, label: "Radiator fouling (45%)", action: { kind: "fault", name: "cooling_degradation", severity: 0.45 } },
    ],
  },
  {
    id: "rapid_throttle",
    name: "Tactical Repositioning",
    tagline: "Dash, throttle back, dash again",
    requirement: "Rapid throttle transitions",
    description:
      "Repeated full-power dashes and throttle-backs, as when an ISR aircraft repositions on a moving target. EGT, CHT and oil temperature have very different time constants, so transients pull them apart - exactly what must not be mistaken for a fault.",
    watchFor:
      "EGT reacting within seconds while CHT and oil temperature lag behind. The twin follows the same transients, so the residuals stay small and the AI should stay nominal throughout.",
    scenario: "healthy",
    legs: [
      { label: "Establish cruise", seconds: 180, altitude: 1500, throttle: 0.65, airspeed: 46, aoa: 3 },
      { label: "Dash", seconds: 60, altitude: 1500, throttle: 1.0, airspeed: 56, aoa: 2 },
      { label: "Throttle back", seconds: 60, altitude: 1500, throttle: 0.35, airspeed: 40, aoa: 5 },
      { label: "Dash", seconds: 60, altitude: 1600, throttle: 1.0, airspeed: 56, aoa: 2 },
      { label: "Throttle back", seconds: 60, altitude: 1600, throttle: 0.35, airspeed: 40, aoa: 5 },
      { label: "Dash", seconds: 60, altitude: 1500, throttle: 1.0, airspeed: 56, aoa: 2 },
      { label: "Recover to cruise", seconds: 240, altitude: 1500, throttle: 0.65, airspeed: 46, aoa: 3 },
    ],
  },
  {
    id: "fault_isolation",
    name: "In-Flight Fault Isolation",
    tagline: "Engine or sensor? Then a real fault",
    requirement: "Diagnosis and health prognosis",
    description:
      "A routine cruise where the CHT sender fails, is reset, and then the oil pump starts to wear. The first must not ground the aircraft; the second decides whether it can finish the sortie.",
    watchFor:
      "CHT dropout shows under Engine or sensor? with no engine fault named. After the repair it clears. When the oil pump degrades, oil pressure separates from the twin, Which part? names it, health falls and the advisory changes.",
    scenario: "healthy",
    legs: [
      { label: "Climb to cruise", seconds: 180, altitude: 2500, throttle: 0.88, airspeed: 45, aoa: 5 },
      { label: "Cruise", seconds: 660, altitude: 2500, throttle: 0.72, airspeed: 50, aoa: 3 },
      { label: "Return to base", seconds: 240, altitude: 1200, throttle: 0.5, airspeed: 48, aoa: 1 },
    ],
    events: [
      { at: 240, label: "CHT sender fails (dropout)", action: { kind: "sensor", channel: "cht", type: "dropout", severity: 0.8 } },
      { at: 420, label: "CHT sender reset", action: { kind: "clear_sensor", channel: "cht" } },
      { at: 540, label: "Oil pump degrades (50%)", action: { kind: "fault", name: "oil_pump_degradation", severity: 0.5 } },
    ],
  },
];

export function getPreset(id: string | null | undefined): MissionPreset | null {
  if (!id) return null;
  return MISSION_PRESETS.find((p) => p.id === id) ?? null;
}

/** The sortie's name for a recorded mission id (simulations.mission), or null. */
export function missionName(id: string | null | undefined): string | null {
  return getPreset(id)?.name ?? null;
}

/** Total flight seconds a profile runs for. */
export function presetDuration(p: MissionPreset): number {
  return p.legs.reduce((sum, l) => sum + l.seconds, 0);
}

/** Engine hours a profile puts on the engine at a given life scale (v4: flight s x scale). */
export function presetEngineHours(p: MissionPreset, lifeScale: number): number {
  return (presetDuration(p) * lifeScale) / 3600;
}

/** The events that apply to an engine. */
export function eventsFor(p: MissionPreset, engineModel: string): MissionEvent[] {
  return (p.events ?? []).filter((e) => !e.engines || e.engines.includes(engineModel));
}

/** The env in force during leg `index`: mission-wide, then each leg's changes in order. */
export function envAt(p: MissionPreset, index: number): MissionEnv {
  return p.legs.slice(0, index + 1).reduce<MissionEnv>((env, l) => ({ ...env, ...l.env }), { ...p.env });
}

/** Which leg is active at flight time `t`, or the last leg once past the end. */
export function legAt(p: MissionPreset, t: number): { leg: MissionLeg; index: number } {
  let acc = 0;
  for (let i = 0; i < p.legs.length; i++) {
    acc += p.legs[i].seconds;
    if (t < acc) return { leg: p.legs[i], index: i };
  }
  return { leg: p.legs[p.legs.length - 1], index: p.legs.length - 1 };
}

/** POST path and body for a mission event. */
export function eventRequest(a: MissionAction): { path: string; body: Record<string, unknown> } {
  switch (a.kind) {
    case "fault": return { path: "/inject", body: { kind: "fault", name: a.name, severity: a.severity } };
    case "sensor": return { path: "/inject", body: { kind: "sensor", channel: a.channel, type: a.type, severity: a.severity } };
    case "clear_fault": return { path: "/inject/clear", body: { kind: "fault", name: a.name } };
    case "clear_sensor": return { path: "/inject/clear", body: { kind: "sensor", channel: a.channel } };
  }
}
