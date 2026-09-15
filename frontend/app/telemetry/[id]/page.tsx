"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import { supabase } from "@/lib/supabase";
import { ArrowLeft, Play, Gauge, Thermometer, Activity, Fuel } from "lucide-react";
import { modelVersionOf, rulPercentOf, tboHoursOf } from "@/lib/timeScale";
import ModelBadge from "@/components/ModelBadge";
import { formatAirspeed, formatAirspeedSecondary, formatAltitude, formatAltitudeFeet, formatRul, isRulOutOfRange } from "@/lib/units";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

const API = "http://localhost:8000";

type Simulation = {
  id: number;
  user_id: string;
  engine_model: string;
  started_at: string;
  ended_at: string | null;
  outcome: string | null;
  final_health_percent: number | null;
  final_rul_hours: number | null;
  final_telemetry: Record<string, unknown> | null;
  model_version: string | null;
  tbo_hours: number | null;
  groq_result: {
    status?: string; headline?: string; summary?: string; risk?: string; model?: string;
    findings?: string[]; recommendations?: string[];
  } | null;
};

type LogRow = {
  time_offset_s: number;
  altitude: number | null;
  health_percent: number | null;
  rul_percent_remaining: number | null;
  egt: number | null;
  cht: number | null;
};

export default function TelemetryDetailPage() {
  const router = useRouter();
  const params = useParams();
  const id = params.id as string;

  const [loading, setLoading] = useState(true);
  const [sim, setSim] = useState<Simulation | null>(null);
  const [logs, setLogs] = useState<LogRow[]>([]);
  const [notFound, setNotFound] = useState(false);
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const { data: sessionData } = await supabase.auth.getSession();
      if (!sessionData.session) {
        router.push("/login");
        return;
      }
      // RLS means this returns nothing (not an error) if the id belongs to a
      // DIFFERENT user - .single() then throws, which we treat as "not found"
      // rather than leaking whether the id exists at all.
      const { data, error } = await supabase
        .from("simulations")
        .select("id, user_id, engine_model, started_at, ended_at, outcome, final_health_percent, final_rul_hours, final_telemetry, groq_result, model_version, tbo_hours")
        .eq("id", id)
        .single();
      if (cancelled) return;
      if (error || !data) {
        setNotFound(true);
        setLoading(false);
        return;
      }
      setSim(data);

      const { data: logRows } = await supabase
        .from("telemetry_logs")
        .select("time_offset_s, altitude, health_percent, rul_percent_remaining, egt, cht")
        .eq("simulation_id", id)
        .order("time_offset_s", { ascending: true });
      if (!cancelled && logRows) setLogs(logRows);
      if (!cancelled) setLoading(false);
    }
    load();
    return () => { cancelled = true; };
  }, [id, router]);

  const handleContinue = async () => {
    if (!sim) return;
    setResuming(true);
    setResumeError(null);
    try {
      const { data: sessionData } = await supabase.auth.getSession();
      const userId = sessionData.session?.user?.id;
      if (!userId) {
        router.push("/login");
        return;
      }
      const res = await fetch(`${API}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ simulation_id: sim.id, user_id: userId }),
      });
      const result = await res.json();
      if (result.status !== "ok") {
        setResumeError(result.message || "Could not resume this simulation.");
        setResuming(false);
        return;
      }
      // engine= picks the correct themed Simulator variant (see SIMULATOR_BY_ENGINE
      // in /simulate); resumed=1 tells that page the physics twin was ALREADY
      // restored server-side by the /resume call above, so it must adopt the twin's
      // current altitude/throttle/airspeed instead of running its fresh-takeoff
      // auto-climb - which used to overwrite the restored state within a second or
      // two of arriving, and was the reason "Continue Simulation" did not continue.
      router.push(`/simulate?engine=${result.engine_model}&resumed=1`);
    } catch {
      setResumeError("Could not reach the backend - is it running?");
      setResuming(false);
    }
  };

  // Optional 4th element is a display formatter; fields without one keep the
  // plain toFixed(decimals) path. Airspeed is shown in knots and altitude gains a
  // feet readout - the units a real ground control station uses. Conversion is
  // display-only; the stored telemetry stays SI.
  type SensorField = [string, string, number, ((v: number) => { value: string; secondary?: string })?];
  const sensorGroups: { title: string; icon: typeof Gauge; fields: SensorField[] }[] = [
    { title: "Flight Parameters", icon: Gauge, fields: [
      ["altitude", "Altitude", 0, (v) => ({ value: formatAltitude(v), secondary: formatAltitudeFeet(v) })],
      ["throttle", "Throttle", 2],
      ["airspeed", "Airspeed", 1, (v) => ({ value: formatAirspeed(v), secondary: formatAirspeedSecondary(v) })],
      ["aoa", "AoA (deg)", 1],
    ] },
    { title: "Powerplant", icon: Activity, fields: [["engine_rpm", "Engine RPM", 0], ["prop_rpm", "Prop RPM", 0], ["power_kw", "Power (kW)", 1], ["fuel_flow", "Fuel Flow", 2]] },
    { title: "Thermal", icon: Thermometer, fields: [["egt", "EGT (C)", 0], ["cht", "CHT (C)", 0], ["oil_pressure", "Oil Pressure", 1], ["oil_temp", "Oil Temp (C)", 1]] },
    { title: "Vibration & Wear", icon: Fuel, fields: [["vibx", "Vib X", 3], ["viby", "Vib Y", 3], ["vibz", "Vib Z", 3], ["wear", "Wear", 4]] },
  ];

  const chartData = logs.map((l) => ({
    t: Math.round(l.time_offset_s),
    altitude: l.altitude ?? undefined,
    health: l.health_percent ?? undefined,
    rul: l.rul_percent_remaining ?? undefined,
  }));

  return (
    <>
      <Navbar />
      <main className="pt-16 relative min-h-screen">
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />

        <div className="relative z-10 max-w-5xl mx-auto px-4 md:px-6 py-12">
          <Link href="/telemetry" className="inline-flex items-center gap-1.5 text-[13px] text-on-surface-variant hover:text-tertiary transition-colors mb-6">
            <ArrowLeft size={15} /> Back to history
          </Link>

          {loading ? (
            <div className="text-on-surface-variant text-sm py-16 text-center border border-outline-variant/30 rounded-lg">Loading...</div>
          ) : notFound || !sim ? (
            <div className="text-on-surface-variant text-sm py-16 text-center border border-outline-variant/30 rounded-lg">
              Simulation not found (or it belongs to a different account).
            </div>
          ) : (
            <>
              <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h1 className="font-headline-display text-[26px] md:text-[32px] font-bold text-primary uppercase tracking-tight">
                      {sim.engine_model.replace(/_/g, " ")}
                      <ModelBadge version={modelVersionOf(sim)} />
                    </h1>
                    <span className="text-[11px] font-mono text-on-surface-variant/60 border border-outline-variant/30 rounded px-2 py-0.5">
                      ID #{sim.id}
                    </span>
                  </div>
                  <p className="text-on-surface-variant text-[13px]">
                    {new Date(sim.started_at).toLocaleString()}
                    {sim.ended_at && ` -> ${new Date(sim.ended_at).toLocaleString()}`}
                    {" - "}<span className="uppercase">{sim.outcome ?? "in progress"}</span>
                  </p>
                </div>

                <div className="flex flex-col items-end gap-1.5">
                  <button
                    onClick={handleContinue}
                    disabled={resuming || !sim.final_telemetry}
                    className="flex items-center gap-2 rounded bg-tertiary px-5 py-2.5 text-[11px] font-bold uppercase tracking-[0.1em] text-black hover:brightness-110 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <Play size={14} className="fill-current" /> {resuming ? "Resuming..." : "Continue Simulation"}
                  </button>
                  {!sim.final_telemetry && (
                    <span className="text-[11px] text-on-surface-variant/60">No saved state to resume from</span>
                  )}
                  {resumeError && <span className="text-[11px] text-red-400">{resumeError}</span>}
                  <div className="flex gap-3">
                    <Link href={`/mission/replay/${sim.id}`} className="text-[10px] font-bold uppercase tracking-[0.1em] text-tertiary hover:brightness-125">
                      Replay
                    </Link>
                    <Link href={`/mission/report/${sim.id}`} className="text-[10px] font-bold uppercase tracking-[0.1em] text-on-surface-variant hover:text-primary">
                      Full report
                    </Link>
                  </div>
                </div>
              </header>

              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
                  <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant mb-1">Final Health</div>
                  <div className="text-3xl font-bold text-primary">{sim.final_health_percent != null ? `${sim.final_health_percent.toFixed(0)}%` : "--"}</div>
                </div>
                <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
                  <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant mb-1">Final RUL</div>
                  <div className="text-3xl font-bold text-primary">
                    {(() => {
                      const pct = sim.final_rul_hours != null
                        ? rulPercentOf(sim.final_rul_hours, modelVersionOf(sim), typeof sim.final_telemetry?.time === "number" ? sim.final_telemetry.time : undefined, tboHoursOf(sim))
                        : null;
                      return pct == null ? "--" : `${pct.toFixed(0)}%`;
                    })()}
                  </div>
                  {sim.final_rul_hours != null && (
                    <div className="text-[10px] uppercase tracking-[0.1em] text-on-surface-variant mt-1">
                      {formatRul(sim.final_rul_hours, modelVersionOf(sim))}
                      {isRulOutOfRange(sim.final_rul_hours, modelVersionOf(sim), tboHoursOf(sim)) && (
                        <span className="text-tertiary"> &middot; extrapolated</span>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {sim.groq_result?.status === "ok" && (
                <div className="mb-8 bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
                  <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
                    <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em]">Post-Flight Analysis</h2>
                    {sim.groq_result.risk && (
                      <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-tertiary">
                        {sim.groq_result.risk} risk
                      </span>
                    )}
                  </div>
                  {/* Plain text - third-party model output. */}
                  <p className="text-[14px] font-bold text-primary leading-snug mb-1.5">{sim.groq_result.headline}</p>
                  <p className="text-[13px] leading-relaxed text-on-surface-variant">{sim.groq_result.summary}</p>
                  <Link href={`/mission/report/${sim.id}`} className="mt-3 inline-block text-[10px] font-bold uppercase tracking-[0.1em] text-tertiary hover:brightness-125">
                    Findings and recommendations &rarr;
                  </Link>
                </div>
              )}

              {/* The health/RUL traces below break wherever the AI produced no
                  output. That happens for the first 128 simulated seconds of a
                  flight while the model's window fills, and again after every
                  resume, because continuing a flight restarts that window. The
                  gaps are real absences of data, not dropouts - so the lines are
                  drawn broken rather than interpolated across them. */}
              {chartData.length > 1 && chartData.some((d) => d.health === undefined) && (
                <div className="mb-4 text-[11px] text-on-surface-variant/80 leading-relaxed">
                  Gaps in the health and RUL traces are the AI&rsquo;s 128-second window fill &mdash;
                  at the start of the flight, and again after each resume. No diagnosis exists
                  for those stretches, so the lines are broken rather than interpolated.
                </div>
              )}

              {chartData.length > 1 && (
                <div className="mb-8 bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
                  <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-4">Flight Profile</h2>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#352722" />
                      <XAxis dataKey="t" stroke="#8a7f6a" fontSize={11} tickFormatter={(v) => `${v}s`} />
                      <YAxis stroke="#8a7f6a" fontSize={11} domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{ background: "#0d0e0d", border: "1px solid #352722", borderRadius: 6, fontSize: 12 }}
                        labelFormatter={(v) => `t = ${v}s`}
                      />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="health" name="Health %" stroke="#7fc87f" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="rul" name="RUL %" stroke="#ff9f42" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {sim.final_telemetry && (
                <div className="mb-8">
                  <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-3">Final Sensor State</h2>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    {sensorGroups.map(({ title, icon: Icon, fields }) => (
                      <div key={title} className="bg-surface/60 border border-outline-variant/30 rounded-lg p-4">
                        <div className="flex items-center gap-1.5 mb-3 text-[10px] uppercase tracking-[0.1em] text-tertiary/80">
                          <Icon size={13} /> {title}
                        </div>
                        <div className="space-y-2">
                          {fields.map(([key, label, decimals, format]) => {
                            const val = sim.final_telemetry?.[key];
                            const shown = typeof val === "number"
                              ? (format ? format(val) : { value: val.toFixed(decimals) })
                              : { value: "--" };
                            return (
                              <div key={key} className="flex items-start justify-between gap-2 text-[12px]">
                                <span className="text-on-surface-variant">{label}</span>
                                <span className="text-right">
                                  <span className="text-primary font-mono">{shown.value}</span>
                                  {shown.secondary && (
                                    <span className="block text-[10px] text-on-surface-variant/70 font-mono">{shown.secondary}</span>
                                  )}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </>
  );
}
