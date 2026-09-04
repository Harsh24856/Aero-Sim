"use client";

import { useMemo } from "react";
import { Pause, Play } from "lucide-react";

// ─── Shared types & helpers ─────────────────────────────────────────────────
export type RawTelemetry = {
  time?: number; altitude?: number; throttle?: number; airspeed?: number; aoa?: number;
  air_density?: number; torque_available_nm?: number; engine_rpm?: number; prop_rpm?: number;
  prop_torque?: number; power_kw?: number; fuel_flow?: number; thrust?: number; lift?: number;
  drag?: number; thrust_margin?: number; lift_weight_margin?: number; egt?: number; cht?: number;
  oil_pressure?: number; oil_temp?: number; vibx?: number; viby?: number; vibz?: number;
  rpm_fault?: number;
};

export function mpsToKmh(mps: number): number { return mps * 3.6; }
export function kmhToMps(kmh: number): number { return kmh / 3.6; }

export const SENSOR_FIELDS: { key: keyof RawTelemetry; label: string; unit: string; decimals?: number; convert?: (v: number) => number }[] = [
  { key: "altitude", label: "Altitude", unit: "m", decimals: 0 },
  { key: "throttle", label: "Throttle", unit: "", decimals: 2 },
  { key: "airspeed", label: "Airspeed", unit: "km/h", decimals: 0, convert: mpsToKmh },
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
];

export type MetersProps = {
  /** Live airspeed in km/h for the semicircle gauge */
  speedKmh?: number;
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
};

const TICK_COUNT = 52;
const MAX_SPEED = 250; // km/h

// ─── Component ──────────────────────────────────────────────────────────────
export default function Meters({
  speedKmh = 0,
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
}: MetersProps) {
  const fraction = Math.max(0, Math.min(1, speedKmh / MAX_SPEED));

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

          {/* Speed Number & KM/H in the center arch */}
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-end pb-3 md:pb-4">
            <span className="text-4xl font-bold leading-none tracking-tight text-white drop-shadow-[0_0_25px_rgba(255,255,255,0.25)] md:text-5xl">
              {Math.round(speedKmh)}
            </span>
            <span className="mt-1 text-[9px] font-semibold uppercase tracking-[0.2em] text-[#9b8577] md:text-[11px]">
              KM/H
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

      {/* ── Middle section: Throttle & Speed Target sliders side by side ── */}
      <div className="relative z-10 my-2 grid grid-cols-2 gap-4 md:gap-6">
        {/* Throttle Slider */}
        <div>
          <div className="flex items-center justify-between text-[8px] uppercase tracking-[0.12em] text-[#d9c0ae] md:text-[10px]">
            <label htmlFor="throttle-input" className="font-semibold">Throttle</label>
            <output className="font-mono text-[#ff8050] font-semibold">{Math.round(throttle)}%</output>
          </div>
          <div className="mt-1.5 h-3.5 border border-[#4c3025] bg-[#1a110d] p-[2px] rounded-sm">
            <div
              className="h-full bg-gradient-to-r from-[#ff971e] via-[#ff5b1c] to-[#ed3919] rounded-[1px] transition-all duration-75"
              style={{ width: `${throttle}%` }}
            />
          </div>
          <input
            id="throttle-input"
            className="cockpit-range mt-1.5 w-full"
            type="range"
            min="0"
            max="100"
            value={throttle}
            onChange={(e) => onThrottleChange(Number(e.target.value))}
          />
        </div>

        {/* Speed Target Slider */}
        <div>
          <div className="flex items-center justify-between text-[8px] uppercase tracking-[0.12em] text-[#d9c0ae] md:text-[10px]">
            <label htmlFor="airspeed-input" className="font-semibold">Speed Target</label>
            <output className="font-mono text-[#ff8050] font-semibold">{Math.round(mpsToKmh(airspeedTarget))} KM/H</output>
          </div>
          <div className="mt-1.5 h-3.5 border border-[#4c3025] bg-[#1a110d] p-[2px] rounded-sm">
            <div
              className="h-full bg-gradient-to-r from-[#ff971e] via-[#ff5b1c] to-[#ed3919] rounded-[1px] transition-all duration-75"
              style={{ width: `${Math.max(0, ((mpsToKmh(airspeedTarget) - 126) / (MAX_SPEED - 126)) * 100)}%` }} // 126 km/h (35 m/s) floor, matching Simulator.tsx's enforced minimum flight speed
            />
          </div>
          <input
            id="airspeed-input"
            className="cockpit-range mt-1.5 w-full"
            type="range"
            min="126"
            max={MAX_SPEED}
            step="5"
            value={Math.round(mpsToKmh(airspeedTarget))}
            onChange={(e) => onAirspeedTargetChange(kmhToMps(Number(e.target.value)))}
          />
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
