"use client";

import { useMemo, useState } from "react";
import { ArrowLeftRight, Pause, Play } from "lucide-react";
import SlidingBar from "@/components/SlidingBar";
import InputBar from "@/components/SlidingBar1(a)";

// ─── Shared types & helpers ─────────────────────────────────────────────────
export type RawTelemetry = {
  time?: number; altitude?: number; throttle?: number; airspeed?: number; aoa?: number;
  air_density?: number; torque_available_nm?: number; engine_rpm?: number; prop_rpm?: number;
  prop_torque?: number; power_kw?: number; fuel_flow?: number; thrust?: number; lift?: number;
  drag?: number; thrust_margin?: number; lift_weight_margin?: number; egt?: number; cht?: number;
  oil_pressure?: number; oil_temp?: number; vibx?: number; viby?: number; vibz?: number;
  rpm_fault?: number;
  // Electrical and injection channels (PS section B). Monitored by advisory.py limit checks;
  // deliberately not AI inputs, so they are shown and alarmed, never diagnosed by a model.
  battery_voltage?: number; battery_current?: number; alternator_output?: number;
  injection_timing?: number;
  // physics v4 (backend/twin_v4.py): what the twelve instruments read, plus the engine's
  // own induction and combustion state.
  physics_version?: string; torque_nm?: number; coolant_temp?: number;
  manifold_pressure_kpa?: number; boost_pressure_ratio?: number; ambient_temp_c?: number;
  lambda?: number; knock_retard?: number; wastegate_position?: number; engine_hours?: number;
  // physics v4 inputs to the engine besides the flight controls (twin_v4.DEFAULT_ENV)
  isa_dev_c?: number; qnh_offset_pa?: number; humidity_frac?: number; fuel_octane_mon?: number;
  fuel_ethanol_frac?: number; target_lambda?: number; cooling_airflow_factor?: number;
  electrical_load_a?: number; oil_thermostat_open?: boolean;
};

// The v4 atmosphere, fuel and installation inputs. Ranges are safety.PARAM_LIMITS, which are
// the ranges the training data sampled, so the models never see an engine they weren't trained on.
export const ENGINE_INPUTS: { key: keyof RawTelemetry; label: string; min: number; max: number; step: number; show: (v: number) => string }[] = [
  { key: "isa_dev_c", label: "ISA Dev", min: -30, max: 50, step: 1, show: (v) => `${v > 0 ? "+" : ""}${v.toFixed(0)} °C` },
  { key: "qnh_offset_pa", label: "QNH", min: -2700, max: 2700, step: 100, show: (v) => `${(1013.25 + v / 100).toFixed(0)} hPa` },
  { key: "humidity_frac", label: "Humidity", min: 0, max: 1, step: 0.05, show: (v) => `${Math.round(v * 100)}%` },
  { key: "fuel_octane_mon", label: "Octane", min: 91, max: 100, step: 1, show: (v) => `${v.toFixed(0)} MON` },
  { key: "fuel_ethanol_frac", label: "Ethanol", min: 0, max: 0.1, step: 0.01, show: (v) => `E${Math.round(v * 100)}` },
  { key: "target_lambda", label: "Target λ", min: 0.78, max: 1.02, step: 0.01, show: (v) => v.toFixed(2) },
  { key: "cooling_airflow_factor", label: "Cooling Air", min: 0.65, max: 1.3, step: 0.05, show: (v) => `×${v.toFixed(2)}` },
  { key: "electrical_load_a", label: "Elec Load", min: 3, max: 34, step: 1, show: (v) => `${v.toFixed(0)} A` },
];

export function mpsToKmh(mps: number): number { return mps * 3.6; }
// Airspeed is displayed in KNOTS - the unit aircrew, ICAO and real ground
// control stations use. Physics, API and DB stay in m/s; this converts for
// display only.
export function mpsToKnots(mps: number): number { return mps * 1.943844; }
export function kmhToMps(kmh: number): number { return kmh / 3.6; }
export function knotsToMps(kt: number): number { return kt / 1.943844; }

export const SENSOR_FIELDS: { key: keyof RawTelemetry; label: string; unit: string; decimals?: number; convert?: (v: number) => number }[] = [
  { key: "altitude", label: "Altitude", unit: "m", decimals: 0 },
  { key: "throttle", label: "Throttle", unit: "", decimals: 2 },
  { key: "airspeed", label: "Airspeed", unit: "kt", decimals: 0, convert: mpsToKnots },
  { key: "aoa", label: "AoA", unit: "deg", decimals: 1 },
  { key: "air_density", label: "Air Density", unit: "kg/m3", decimals: 3 },
  { key: "torque_available_nm", label: "Torque Avail", unit: "Nm", decimals: 1 },
  { key: "engine_rpm", label: "Engine RPM", unit: "", decimals: 0 },
  { key: "prop_rpm", label: "Prop RPM", unit: "", decimals: 0 },
  { key: "prop_torque", label: "Prop Torque", unit: "Nm", decimals: 1 },
  { key: "power_kw", label: "Power", unit: "kW", decimals: 1 },
  { key: "fuel_flow", label: "Fuel Flow", unit: "kg/h", decimals: 2 },
  { key: "thrust", label: "Thrust", unit: "N", decimals: 1 },
  { key: "lift", label: "Lift", unit: "N", decimals: 0 },
  { key: "drag", label: "Drag", unit: "N", decimals: 1 },
  { key: "thrust_margin", label: "Thrust Margin", unit: "N", decimals: 1 },
  { key: "lift_weight_margin", label: "Lift/Weight Margin", unit: "N", decimals: 0 },
  { key: "egt", label: "EGT", unit: "C", decimals: 1 },
  { key: "cht", label: "CHT", unit: "C", decimals: 1 },
  { key: "oil_pressure", label: "Oil Pressure", unit: "", decimals: 1 },
  { key: "oil_temp", label: "Oil Temp", unit: "C", decimals: 1 },
  { key: "vibx", label: "Vib X", unit: "", decimals: 3 },
  { key: "viby", label: "Vib Y", unit: "", decimals: 3 },
  { key: "vibz", label: "Vib Z", unit: "", decimals: 3 },
  { key: "rpm_fault", label: "RPM Fault", unit: "", decimals: 0 },
  { key: "battery_voltage", label: "Battery", unit: "V", decimals: 1 },
  { key: "battery_current", label: "Batt Current", unit: "A", decimals: 1 },
  { key: "alternator_output", label: "Alternator", unit: "A", decimals: 1 },
  { key: "injection_timing", label: "Injection", unit: "deg", decimals: 2 },
];

// Physics v4: oil pressure is in bar, torque is brake torque, and the cockpit reads the
// twelve instruments the AI is trained on (plus the induction state that explains them).
export const SENSOR_FIELDS_V4: typeof SENSOR_FIELDS = [
  { key: "altitude", label: "Altitude", unit: "m", decimals: 0 },
  { key: "throttle", label: "Throttle", unit: "", decimals: 2 },
  { key: "airspeed", label: "Airspeed", unit: "kt", decimals: 0, convert: mpsToKnots },
  { key: "aoa", label: "AoA", unit: "deg", decimals: 1 },
  { key: "ambient_temp_c", label: "OAT", unit: "C", decimals: 1 },
  { key: "air_density", label: "Air Density", unit: "kg/m3", decimals: 3 },
  { key: "engine_rpm", label: "Engine RPM", unit: "", decimals: 0 },
  { key: "prop_rpm", label: "Prop RPM", unit: "", decimals: 0 },
  { key: "torque_nm", label: "Torque", unit: "Nm", decimals: 1 },
  { key: "power_kw", label: "Power", unit: "kW", decimals: 1 },
  { key: "manifold_pressure_kpa", label: "MAP", unit: "kPa", decimals: 1 },
  { key: "boost_pressure_ratio", label: "Boost PR", unit: "", decimals: 2 },
  { key: "wastegate_position", label: "Wastegate", unit: "", decimals: 2 },
  { key: "fuel_flow", label: "Fuel Flow", unit: "L/h", decimals: 1 },
  { key: "lambda", label: "Lambda", unit: "", decimals: 2 },
  { key: "knock_retard", label: "Knock Retard", unit: "", decimals: 2 },
  { key: "egt", label: "EGT", unit: "C", decimals: 0 },
  { key: "cht", label: "CHT", unit: "C", decimals: 0 },
  { key: "coolant_temp", label: "Coolant", unit: "C", decimals: 0 },
  { key: "oil_temp", label: "Oil Temp", unit: "C", decimals: 0 },
  { key: "oil_pressure", label: "Oil Press", unit: "bar", decimals: 2 },
  { key: "battery_voltage", label: "Battery", unit: "V", decimals: 2 },
  { key: "vibx", label: "Vib X", unit: "g", decimals: 3 },
  { key: "viby", label: "Vib Y", unit: "g", decimals: 3 },
  { key: "vibz", label: "Vib Z", unit: "g", decimals: 3 },
  { key: "thrust", label: "Thrust", unit: "N", decimals: 0 },
];

export type MetersProps = {
  /** Live airspeed in KNOTS for the semicircle gauge */
  speedKnots?: number;
  /** Live altitude in meters */
  altitude?: number;
  /** Throttle 0–100 */
  throttle: number;
  onThrottleChange: (v: number) => void;
  airspeedTarget: number;
  onAirspeedTargetChange: (v: number) => void;
  started: boolean;
  onStart: () => void;
  paused: boolean;
  onTogglePause: () => void;
  isSignedIn?: boolean | null;   // null = auth check still in flight
  /** The aircraft is on the CAN bus and owns the set-points: sliders lock, gauges show its values */
  canLive?: boolean;
  /** v4 only: the engine's current inputs, shown as the "Inputs to the engine" controls */
  engineInputs?: RawTelemetry | null;
  onEngineInputChange?: (key: keyof RawTelemetry, value: number | boolean) => void;
};

const TICK_COUNT = 52;
const MAX_SPEED = 135; // kt (= 250 km/h, the airframe ceiling in Simulator.tsx)
const MIN_SPEED = 68;  // kt (= 126 km/h / 35 m/s, the enforced stall floor)

// ─── Component ──────────────────────────────────────────────────────────────
export default function Meters({
  speedKnots = 0,
  altitude = 0,
  throttle,
  onThrottleChange,
  airspeedTarget,
  onAirspeedTargetChange,
  started,
  onStart,
  paused,
  onTogglePause,
  isSignedIn = true,
  canLive = false,
  engineInputs,
  onEngineInputChange,
}: MetersProps) {
  const fraction = Math.max(0, Math.min(1, speedKnots / MAX_SPEED));
  // v4: the panel slides between the flight controls and the engine's other inputs.
  const [view, setView] = useState<"flight" | "engine">("flight");
  const hasInputs = !!(engineInputs && onEngineInputChange);
  const showEngine = view === "engine" && hasInputs;

  // Compute 180° dome/semicircle tick positions in SVG coordinates (0 0 300 150)
  const ticks = useMemo(() => {
    const cx = 150;
    const cy = 135;
    const rIn = 98;
    const rOut = 118;

    return Array.from({ length: TICK_COUNT }, (_, i) => {
      const frac = i / (TICK_COUNT - 1);
      const angle = Math.PI - frac * Math.PI; // 180° to 0°
      const x1 = Number((cx + rIn * Math.cos(angle)).toFixed(2));
      const y1 = Number((cy - rIn * Math.sin(angle)).toFixed(2));
      const x2 = Number((cx + rOut * Math.cos(angle)).toFixed(2));
      const y2 = Number((cy - rOut * Math.sin(angle)).toFixed(2));
      const isLit = frac <= fraction;

      return {
        key: i,
        x1,
        y1,
        x2,
        y2,
        isLit,
      };
    });
  }, [fraction]);

  return (
    <section className="panel-shell relative flex h-full min-h-0 flex-col justify-between overflow-hidden p-3 md:p-4 bg-[#080808]">
      {/* Background ambient glow */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_60%_80%_at_50%_35%,rgba(232,118,58,0.12),transparent_70%)]" />

      {hasInputs && (
        <button
          type="button"
          onClick={() => setView(showEngine ? "flight" : "engine")}
          className="absolute right-2 top-2 z-20 flex items-center gap-1 rounded-sm border border-[#4c3025] bg-[#1a110d] px-1.5 py-0.5 text-[7px] font-semibold uppercase tracking-[0.12em] text-[#d9c0ae] hover:text-[#ff8050] md:text-[9px]"
          aria-pressed={showEngine}
        >
          <ArrowLeftRight size={11} /> {showEngine ? "Flight controls" : "Engine inputs"}
        </button>
      )}

      {/* Two panes side by side in a 200%-wide strip; the swap slides it by half. */}
      <div className="relative z-10 min-h-0 flex-1 overflow-hidden">
      <div
        className="flex h-full w-[200%] transition-transform duration-300 ease-out"
        style={{ transform: showEngine ? "translateX(-50%)" : "translateX(0)" }}
      >
      <div className="flex h-full w-1/2 min-h-0 flex-col justify-between" inert={showEngine} aria-hidden={showEngine}>
      {/* ── Top section: Throttle Stat | Semicircle Speedometer | Altitude Stat ── */}
      <div className="relative z-10 flex min-h-0 flex-1 items-center justify-between px-2 md:px-4">
        {/* Left: Throttle Stat */}
        <div className="flex flex-col items-start min-w-[70px]">
          <span className="text-2xl font-bold tracking-tight text-white md:text-3xl">
            {Math.round(throttle)}%
          </span>
          <span className="mt-0.5 text-[9px] font-medium uppercase tracking-wider text-[#9b8577] md:text-[11px]">
            Throttle
          </span>
        </div>

        {/* Center: Semicircle Arch Speedometer Gauge */}
        <div className="relative flex h-full max-h-[170px] w-full max-w-[280px] md:max-w-[320px] items-center justify-center">
          <svg viewBox="0 0 300 150" className="h-full w-full overflow-visible">
            <defs>
              <filter id="tick-glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2" result="glow" />
                <feMerge>
                  <feMergeNode in="glow" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Render Ticks */}
            {ticks.map((t) => (
              <line
                key={t.key}
                x1={t.x1}
                y1={t.y1}
                x2={t.x2}
                y2={t.y2}
                stroke={t.isLit ? "#e8763a" : "rgba(255, 255, 255, 0.18)"}
                strokeWidth={t.isLit ? 2.5 : 2}
                strokeLinecap="round"
                filter={t.isLit ? "url(#tick-glow)" : undefined}
              />
            ))}
          </svg>

          {/* Speed number & unit in the center arch */}
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-end pb-3 md:pb-4">
            <span className="text-4xl font-bold leading-none tracking-tight text-white drop-shadow-[0_0_25px_rgba(255,255,255,0.25)] md:text-5xl">
              {Math.round(speedKnots)}
            </span>
            <span className="mt-1 text-[9px] font-semibold uppercase tracking-[0.2em] text-[#9b8577] md:text-[11px]">
              KT
            </span>
          </div>
        </div>

        {/* Right: Altitude Stat */}
        <div className="flex flex-col items-end min-w-[70px]">
          <span className="text-2xl font-bold tracking-tight text-white md:text-3xl">
            {Math.round(altitude).toLocaleString()}
          </span>
          <span className="mt-0.5 text-[9px] font-medium uppercase tracking-wider text-[#9b8577] md:text-[11px]">
            Alt (m)
          </span>
        </div>
      </div>

      {canLive && (
        <div role="status" className="relative z-10 mt-1 border border-[#2f4a30] bg-[#0d150e] px-2 py-1 text-center text-[8px] uppercase tracking-[0.12em] text-[#a8e0a8] md:text-[9px]">
          Aircraft on CAN bus - throttle and speed come from the aircraft
        </div>
      )}

      {/* ── Middle section: Throttle & Speed Target sliders side by side ── */}
      {/* The speed track starts at the stall floor, not at zero, so its end labels are
          spelled out; without them a near-empty bar reads as a fault. */}
      <div className="relative z-10 my-2 grid grid-cols-2 gap-4 md:gap-6">
        <SlidingBar label="Throttle" value={throttle} onChange={onThrottleChange} disabled={canLive}
          format={(v) => `${Math.round(v)}%`} />
        <SlidingBar label="Speed Target" min={MIN_SPEED} max={MAX_SPEED} disabled={canLive}
          value={Math.round(mpsToKnots(airspeedTarget))}
          onChange={(kt) => onAirspeedTargetChange(knotsToMps(kt))}
          format={(v) => `${Math.round(v)} KT`} />
      </div>

      </div>

      {/* ── v4: atmosphere, fuel and installation inputs ── */}
      <div className={`flex h-full w-1/2 min-h-0 flex-col ${canLive ? "opacity-60" : ""}`} inert={!showEngine} aria-hidden={!showEngine}>
        {engineInputs && onEngineInputChange && (
          <>
            <div className="px-1 pb-2 pt-0.5 text-[8px] font-semibold uppercase tracking-[0.12em] text-[#d9c0ae] md:text-[10px]">
              Inputs to the engine
            </div>
            {/* px-2 / pt-3: room for the thumb at either end and the drag tooltip above it. */}
            <div className="grid min-h-0 flex-1 grid-cols-2 content-start gap-x-4 gap-y-2 overflow-y-auto px-2 pt-3 md:grid-cols-3">
              {ENGINE_INPUTS.map(({ key, label, min, max, step, show }) => (
                <InputBar key={key} compact label={label} min={min} max={max} step={step} format={show}
                  value={Number(engineInputs[key] ?? min)} disabled={canLive}
                  onChange={(v) => onEngineInputChange(key, Number(v.toFixed(4)))} />
              ))}
              <label className="flex items-center gap-1.5 text-[7px] uppercase tracking-[0.1em] text-[#d9c0ae] md:text-[9px]">
                <input type="checkbox" checked={engineInputs.oil_thermostat_open !== false} disabled={canLive}
                  onChange={(e) => onEngineInputChange("oil_thermostat_open", e.target.checked)} />
                Oil thermostat {engineInputs.oil_thermostat_open === false ? "stuck (bypass)" : "working"}
              </label>
            </div>
          </>
        )}
      </div>
      </div>
      </div>

      {/* ── Bottom section: Full width Start/Pause Simulation button ── */}
      <div className="relative z-10 pt-1">
        {!started ? (
          <button
            className={`flex w-full items-center justify-center gap-2 rounded-sm py-2.5 text-[9px] font-bold uppercase tracking-[0.14em] transition-all active:scale-[0.99] md:py-3 md:text-[11px] ${
              isSignedIn
                ? "bg-[#ff611b] text-[#180b06] shadow-[0_0_15px_rgba(255,97,27,0.3)] hover:bg-[#ff7c37]"
                : "border border-[#7b3c21] bg-[#25130d] text-[#ff9b69] hover:bg-[#361b11]"
            }`}
            onClick={onStart}
          >
            {/* Clicking while signed out still calls onStart - the actual page.tsx
                handler redirects to /login itself, this is just the visual cue. */}
            <Play size={14} className="fill-current" /> {isSignedIn ? "Start Simulation" : "Sign In to Simulate"}
          </button>
        ) : (
          <button
            className={`flex w-full items-center justify-center gap-2 rounded-sm py-2.5 text-[9px] font-bold uppercase tracking-[0.14em] transition-all active:scale-[0.99] md:py-3 md:text-[11px] ${
              paused
                ? "bg-[#ff611b] text-[#180b06] shadow-[0_0_15px_rgba(255,97,27,0.3)] hover:bg-[#ff7c37]"
                : "border border-[#7b3c21] bg-[#25130d] text-[#ff9b69] hover:bg-[#361b11]"
            }`}
            onClick={onTogglePause}
          >
            {paused ? <Play size={14} className="fill-current" /> : <Pause size={14} className="fill-current" />}
            {paused ? "Resume Simulation" : "Pause Simulation"}
          </button>
        )}
      </div>
    </section>
  );
}
