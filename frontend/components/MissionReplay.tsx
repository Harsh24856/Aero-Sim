"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { Play, Pause, SkipBack, SkipForward, AlertTriangle } from "lucide-react";
import { flightHours, formatSimClock, type ModelVersion } from "@/lib/timeScale";
import { faultLabel } from "@/lib/v4";
import { formatAirspeed, formatAirspeedSecondary, formatAltitude, formatAltitudeFeet } from "@/lib/units";

/**
 * Mission replay (PS 26054 section E, "Replay of historical mission data").
 *
 * Reads nothing itself - the page hands it the already-fetched telemetry_logs
 * rows, which the backend has been writing one-per-simulated-second all along.
 * No new schema and no new backend endpoint was needed for this.
 *
 * Null handling matters here and is not defensive padding: the AI needs a full
 * 128-simulated-second window before it predicts anything, so the first ~128
 * rows of EVERY run have null health/RUL. Those render as "--". Coercing them
 * to 0 would draw a fake cliff at the start of every mission.
 */

export type ReplayRow = {
  time_offset_s: number;
  altitude: number | null;
  throttle: number | null;
  airspeed: number | null;
  aoa: number | null;
  engine_rpm: number | null;
  prop_rpm: number | null;
  power_kw: number | null;
  fuel_flow: number | null;
  thrust: number | null;
  lift: number | null;
  drag: number | null;
  egt: number | null;
  cht: number | null;
  oil_pressure: number | null;
  oil_temp: number | null;
  vibx: number | null;
  viby: number | null;
  vibz: number | null;
  fault_detected: boolean | null;
  detection_confidence: number | null;
  health_percent: number | null;
  rul_percent_remaining: number | null;
  rul_hours: number | null;
  // physics v4 (backend/dbv4.py) - absent on v2/v3 rows
  engine_hours?: number | null;
  coolant_temp?: number | null;
  manifold_pressure_kpa?: number | null;
  battery_voltage?: number | null;
  margin_min?: number | null;
  rul_calendar_hours?: number | null;
  rul_band_hours?: number | null;
  wear_limited?: boolean | null;
  fault_modes?: Record<string, { probability: number | null; present: boolean; severity: number | null }> | null;
  residuals?: Record<string, number | null> | null;
  truth?: { faults_present?: string[] | null; sensor_faults?: Record<string, string> | null;
            wear_condition?: number | null; rul_hours?: number | null } | null;
};

const SPEEDS = [1, 2, 4, 8];

function fmt(v: number | null | undefined, digits = 1, suffix = ""): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "--";
  return `${v.toFixed(digits)}${suffix}`;
}

export default function MissionReplay({ rows, modelVersion = "v2" }: { rows: ReplayRow[]; modelVersion?: ModelVersion }) {
  const v4 = modelVersion === "v4";
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const last = Math.max(rows.length - 1, 0);
  const cur = rows[Math.min(idx, last)];

  // Playback ticks one logged row (= one simulated second) per 1000/speed ms.
  useEffect(() => {
    if (timer.current) { clearInterval(timer.current); timer.current = null; }
    if (!playing || rows.length === 0) return;
    timer.current = setInterval(() => {
      setIdx((i) => {
        if (i >= last) { setPlaying(false); return last; }
        return i + 1;
      });
    }, 1000 / speed);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [playing, speed, last, rows.length]);

  const chartData = useMemo(
    () => rows.map((r) => ({
      t: r.time_offset_s,
      health: r.health_percent,
      rul: r.rul_percent_remaining,
      alt: r.altitude,
    })),
    [rows]
  );

  if (rows.length === 0) {
    return (
      <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 text-on-surface-variant text-[13px]">
        No per-second telemetry was recorded for this mission.
      </div>
    );
  }

  // [label, value, optional secondary readout]
  const groups: { title: string; items: [string, string, string?][] }[] = [
    // Airspeed in knots and altitude with a feet readout: the units aircrew and
    // every real GCS actually use. The stored values stay SI - this converts at
    // the display layer only.
    { title: "Flight Parameters", items: [
      ["Altitude", formatAltitude(cur.altitude), formatAltitudeFeet(cur.altitude)],
      ["Airspeed", formatAirspeed(cur.airspeed), formatAirspeedSecondary(cur.airspeed)],
      ["Throttle", cur.throttle === null ? "--" : `${(cur.throttle * 100).toFixed(0)}%`],
      ["AoA", cur.aoa === null ? "--" : `${cur.aoa.toFixed(1)}\u00b0`],
    ]},
    { title: "Powerplant", items: [
      ["Engine RPM", fmt(cur.engine_rpm, 0)],
      ["Prop RPM", fmt(cur.prop_rpm, 0)],
      ["Power (kW)", fmt(cur.power_kw, 1)],
      ["Fuel Flow", fmt(cur.fuel_flow, 2)],
      ...(v4 ? [["Manifold", fmt(cur.manifold_pressure_kpa, 1, " kPa")] as [string, string]] : []),
    ]},
    { title: "Thermal & Oil", items: [
      ["EGT", cur.egt === null ? "--" : `${cur.egt.toFixed(0)} \u00b0C`],
      ["CHT", cur.cht === null ? "--" : `${cur.cht.toFixed(0)} \u00b0C`],
      ["Oil Press", v4 ? fmt(cur.oil_pressure, 2, " bar") : fmt(cur.oil_pressure, 1)],
      ["Oil Temp", cur.oil_temp === null ? "--" : `${cur.oil_temp.toFixed(0)} \u00b0C`],
      ...(v4 ? [["Coolant", fmt(cur.coolant_temp, 0, " \u00b0C")] as [string, string],
                ["Battery", fmt(cur.battery_voltage, 2, " V")] as [string, string]] : []),
    ]},
    { title: "Aero & Vibration", items: [
      ["Thrust (N)", fmt(cur.thrust, 0)],
      ["Drag (N)", fmt(cur.drag, 0)],
      ["Vib X / Y", `${fmt(cur.vibx, 2)} / ${fmt(cur.viby, 2)}`],
      ["Vib Z", fmt(cur.vibz, 2)],
    ]},
  ];

  const aiPending = cur.health_percent === null;

  return (
    <div className="space-y-4">
      {/* transport */}
      <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em]">Mission Replay</h2>
          <div className="flex items-center gap-1.5">
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`rounded px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.1em] transition-all ${
                  speed === s ? "bg-tertiary text-black" : "bg-surface/60 border border-outline-variant/30 text-on-surface-variant hover:text-primary"
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => { setIdx(0); setPlaying(false); }}
            className="p-2 rounded bg-surface/60 border border-outline-variant/30 text-on-surface-variant hover:text-primary transition-colors"
            aria-label="Restart"
          >
            <SkipBack className="h-4 w-4" />
          </button>
          <button
            onClick={() => setPlaying((p) => !p)}
            className="p-2.5 rounded bg-tertiary text-black hover:brightness-110 transition-all"
            aria-label={playing ? "Pause" : "Play"}
          >
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </button>
          <button
            onClick={() => { setIdx(last); setPlaying(false); }}
            className="p-2 rounded bg-surface/60 border border-outline-variant/30 text-on-surface-variant hover:text-primary transition-colors"
            aria-label="Jump to end"
          >
            <SkipForward className="h-4 w-4" />
          </button>

          <input
            type="range"
            min={0}
            max={last}
            value={Math.min(idx, last)}
            onChange={(e) => { setIdx(Number(e.target.value)); setPlaying(false); }}
            className="cockpit-range w-full"
            aria-label="Scrub mission time"
          />
        </div>

        <div className="mt-3 flex items-center justify-between text-[11px] font-mono">
          <span className="text-primary">
            T+{cur.time_offset_s.toFixed(0)}s
            <span className="text-on-surface-variant ml-2">
              {/* v3 replay rows carry no wear, so show the simulated clock rather than
                  pretending seconds are engine hours. */}
              {v4
                ? `(flight ${formatSimClock(cur.time_offset_s)} \u00b7 ${fmt(cur.engine_hours, 1)} engine h)`
                : modelVersion === "v3"
                ? `(sim ${formatSimClock(cur.time_offset_s)})`
                : `(${flightHours(cur.time_offset_s, modelVersion).toFixed(2)} h real)`}
            </span>
          </span>
          <span className="text-on-surface-variant">
            frame {Math.min(idx, last) + 1} / {rows.length}
          </span>
        </div>
      </div>

      {/* health / rul at scrub position */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
          <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant">{v4 ? "Wear condition at T" : "Health at T"}</div>
          <div className="text-3xl font-bold text-primary">{fmt(cur.health_percent, 1, "%")}</div>
        </div>
        <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
          <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant">RUL at T</div>
          <div className="text-3xl font-bold text-primary">{fmt(cur.rul_percent_remaining, 1, "%")}</div>
          {v4 && cur.rul_hours != null && (
            <div className="text-[10px] uppercase tracking-[0.1em] text-on-surface-variant mt-1">
              {cur.rul_hours.toFixed(0)}{cur.rul_band_hours != null ? ` \u00b1${cur.rul_band_hours.toFixed(0)}` : ""} h
              {cur.rul_calendar_hours != null ? ` \u00b7 calendar ${cur.rul_calendar_hours.toFixed(0)} h` : ""}
              {cur.wear_limited ? " \u00b7 wear-limited" : ""}
            </div>
          )}
        </div>
        <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
          <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant">Fault Status</div>
          <div className={`text-3xl font-bold ${cur.fault_detected ? "text-[#ff9a72]" : "text-[#7fc87f]"}`}>
            {cur.fault_detected === null ? "--" : cur.fault_detected ? "DETECTED" : "CLEAR"}
          </div>
          {cur.detection_confidence !== null && (
            <div className="text-[10px] uppercase tracking-[0.1em] text-on-surface-variant mt-1">
              {(cur.detection_confidence * 100).toFixed(0)}% conf
            </div>
          )}
        </div>
      </div>

      {v4 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
            <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant mb-2">AI named at T</div>
            {(() => {
              const named = Object.entries(cur.fault_modes ?? {}).filter(([, f]) => f.present);
              return named.length === 0
                ? <div className="text-[13px] text-[#7fc87f]">{cur.fault_modes ? "No component fault" : "--"}</div>
                : named.map(([n, f]) => (
                    <div key={n} className="flex justify-between text-[13px] text-[#ff9a72]">
                      <span>{faultLabel(n)}</span>
                      <span className="font-mono">{f.severity != null ? `${(f.severity * 100).toFixed(0)}%` : ""}</span>
                    </div>));
            })()}
            {cur.margin_min != null && (
              <div className="text-[10px] uppercase tracking-[0.1em] text-on-surface-variant mt-2">
                operating margin {(cur.margin_min * 100).toFixed(0)}%
              </div>
            )}
          </div>
          <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
            <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant mb-2">Ground truth at T (injected)</div>
            <div className="text-[13px] text-primary">
              {(cur.truth?.faults_present ?? []).length > 0
                ? (cur.truth?.faults_present ?? []).map(faultLabel).join(", ")
                : "No engine fault"}
            </div>
            {Object.keys(cur.truth?.sensor_faults ?? {}).length > 0 && (
              <div className="text-[12px] text-[#ffd27a] mt-1">
                Sensor: {Object.entries(cur.truth?.sensor_faults ?? {}).map(([c, k]) => `${c} ${k}`).join(", ")}
              </div>
            )}
            {cur.truth?.wear_condition != null && (
              <div className="text-[10px] uppercase tracking-[0.1em] text-on-surface-variant mt-2">
                true wear condition {(cur.truth.wear_condition * 100).toFixed(0)}%
                {cur.truth.rul_hours != null ? ` \u00b7 true RUL ${cur.truth.rul_hours.toFixed(0)} h` : ""}
              </div>
            )}
          </div>
        </div>
      )}

      {aiPending && (
        <div className="bg-surface/60 border border-outline-variant/30 rounded-lg p-3 flex items-start gap-2.5">
          <AlertTriangle className="h-4 w-4 text-tertiary shrink-0 mt-0.5" />
          <p className="text-[12px] text-on-surface-variant leading-relaxed">
            No AI output at this instant. The models need a full 128 simulated seconds of
            history before their first prediction, so the opening stretch of every mission
            is diagnostically blank by design &mdash; not missing data.
          </p>
        </div>
      )}

      {/* trajectory with scrub marker */}
      <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
        <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-4">Flight Profile</h2>
        <div style={{ width: "100%", height: 260 }}>
          <ResponsiveContainer>
            <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#352722" />
              <XAxis dataKey="t" stroke="#8a7f6a" fontSize={11} tickFormatter={(v) => `${v}s`} />
              <YAxis stroke="#8a7f6a" fontSize={11} domain={[0, 100]} />
              <Tooltip contentStyle={{ background: "#0d0e0d", border: "1px solid #352722", borderRadius: 6, fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="health" name={v4 ? "Wear condition %" : "Health %"} stroke="#7fc87f" dot={false} strokeWidth={2} connectNulls={false} />
              <Line type="monotone" dataKey="rul" name="RUL %" stroke="#ff9f42" dot={false} strokeWidth={2} connectNulls={false} />
              <ReferenceLine x={cur.time_offset_s} stroke="#ff6a22" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* sensor state at scrub position */}
      <div>
        <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-3">Sensor State at T</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {groups.map((g) => (
            <div key={g.title} className="bg-surface/60 border border-outline-variant/30 rounded-lg p-4">
              <div className="text-[10px] uppercase tracking-[0.1em] text-tertiary/80 mb-2">{g.title}</div>
              <dl className="space-y-1.5">
                {g.items.map(([label, value, secondary]) => (
                  <div key={label} className="flex items-start justify-between gap-2 text-[12px]">
                    <dt className="text-on-surface-variant">{label}</dt>
                    <dd className="text-right">
                      <span className="text-primary font-mono">{value}</span>
                      {secondary && (
                        <span className="block text-[10px] text-on-surface-variant/70 font-mono">{secondary}</span>
                      )}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
