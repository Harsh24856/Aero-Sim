// Physics v4 and v5 on the frontend. backend/aiv5.py answers in aiv4's shape plus the
// v5 fields below, so one panel and one set of labels serve both.
// Physics v4 on the frontend: the AI result shape (backend/aiv4.py), the twin's
// per-frame extras (backend/twin_v4.py) and the names shown for faults and sensors.
// One place, so the live cockpit, the replay and the reports all read the same words.

export type V4FaultMode = { probability: number; present: boolean; threshold: number; severity: number; family?: string };
export type V4Sensor = { condition: string; confidence: number };

/** The v4 fields of an AI result (on top of the v2/v3 AiResult in Diagnostics.tsx). */
export type AiResultV4Fields = {
  placeholder_models?: boolean;       // served by the 914's models until this engine's train
  models_engine?: string;
  fault_modes?: Record<string, V4FaultMode>;
  faults_present?: string[];
  sensors?: Record<string, V4Sensor>;
  faulty_sensors?: string[];
  wear_condition?: number | null;     // 1 = as new; the health head
  margin_min?: number;                // operating margin to the nearest certified limit, 0..1
  rul_calendar_hours?: number;        // TBO minus engine hours: the "just count down" answer
  wear_limited?: boolean;             // wear ends life before the overhaul date
  // ---- v5 only ----
  model_version?: string;
  labels?: string;                    // "v5b": severity is EFFECTIVE, small sensor offsets count as none
  families?: Record<string, number>;  // fault-family probabilities
  families_present?: string[];        // families called - the answer when no single fault clears its cut-off
  severity_kind?: string;             // "effective": the share of the fault's full effect the engine shows
  context_settling?: boolean;         // the 60-minute context is still filling
};

export const FAMILY_LABELS: Record<string, string> = {
  induction: "Induction / turbo", cylinder: "Cylinders / valves", fuel_ignition: "Fuel & ignition",
  oil: "Oil system", cooling: "Cooling", propeller: "Propeller",
};
export const familyLabel = (k: string) => FAMILY_LABELS[k] ?? k.replace(/_/g, " ");

/** What the twin knows and the AI does not - shown as the ground-truth panel. */
export type V4Truth = {
  fault_severity?: Record<string, number>;
  faults_present?: string[];
  sensor_faults?: Record<string, string>;
  wear_condition?: number;
  margin_min?: number;
  true_values?: Record<string, number>;
  rul_hours?: number;
  rul_calendar_hours?: number;
  wear_limited?: boolean;
  severity_kind?: string;             // v5: "effective"
};

/** The engine being flown (main.py engine_summary, in every v4 frame). */
export type V4Engine = {
  engine_id?: number | null;
  scenario?: string | null;
  start_engine_hours?: number | null;
  tbo_hours?: number | null;
  life_scale?: number;
  placeholder_models?: boolean;
};

export const FAULT_LABELS: Record<string, string> = {
  air_filter_fouling: "Air filter fouling",
  compression_loss: "Compression loss",
  valve_leakage: "Valve leakage",
  turbo_degradation: "Turbo degradation",
  wastegate_fault: "Wastegate fault",
  intercooler_fouling: "Intercooler fouling",
  injector_fouling: "Fuel metering fouling",
  ignition_degradation: "Ignition degradation",
  combustion_instability: "Combustion instability",
  bearing_wear: "Bearing wear",
  oil_pump_degradation: "Oil pump degradation",
  oil_degradation: "Oil degradation",
  cooling_degradation: "Cooling degradation",
  prop_erosion: "Prop erosion",
};

// Faults the physics hides below the critical altitude (the wastegate closes further
// and holds boost) - shown as a hint, not as a weakness of the model.
export const ALTITUDE_MASKED = new Set(["turbo_degradation", "wastegate_fault"]);

export const SENSOR_LABELS: Record<string, string> = {
  egt: "EGT", cht: "CHT", coolant_temp: "Coolant", oil_temp: "Oil temp",
  oil_pressure: "Oil press", engine_rpm: "RPM", fuel_flow: "Fuel flow",
  manifold_pressure_kpa: "MAP", vibx: "Vib X", viby: "Vib Y", vibz: "Vib Z",
  battery_voltage: "Battery",
};

// The six channels the on-board twin gives residuals on, with display units.
export const RESIDUAL_CHANNELS: { key: string; label: string; unit: string }[] = [
  { key: "egt", label: "EGT", unit: "°C" },
  { key: "cht", label: "CHT", unit: "°C" },
  { key: "oil_temp", label: "Oil temp", unit: "°C" },
  { key: "oil_pressure", label: "Oil press", unit: "bar" },
  { key: "engine_rpm", label: "RPM", unit: "rpm" },
  { key: "fuel_flow", label: "Fuel flow", unit: "L/h" },
];

export const faultLabel = (k: string) => FAULT_LABELS[k] ?? k.replace(/_/g, " ");
export const sensorLabel = (k: string) => SENSOR_LABELS[k] ?? k;
