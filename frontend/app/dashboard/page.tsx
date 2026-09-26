"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import { supabase } from "@/lib/supabase";
import { engineHoursOf, flightHours, modelVersionOf, rulPercentOf, tboHoursOf, usesEngineHours } from "@/lib/timeScale";
import EnginesCard from "@/components/EnginesCard";
import ModelBadge from "@/components/ModelBadge";
import { Rocket, History, ListChecks, TrendingUp, Clock, ChevronRight, LogOut } from "lucide-react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  BarChart, Bar, Cell,
} from "recharts";

type SimulationRow = {
  id: number;
  engine_model: string;
  started_at: string;
  ended_at: string | null;
  outcome: string | null;
  final_health_percent: number | null;
  final_rul_hours: number | null;
  final_telemetry: Record<string, unknown> | null;
  model_version: string | null;
  tbo_hours: number | null;
};

const ENGINE_COLORS: Record<string, string> = {
  Rotax_914_ULF: "#ff9f42",
  Rotax_912_ULS: "#7fc87f",
  Rotax_915_iS: "#5aa9e6",
  Rotax_916_iS: "#d47fe6",
};

function formatEngine(engine: string) {
  return engine.replace(/_/g, " ");
}

function elapsedSecondsOf(sim: SimulationRow): number | undefined {
  const t = sim.final_telemetry?.time;
  return typeof t === "number" ? t : undefined;
}

export default function DashboardPage() {
  const router = useRouter();
  const [userName, setUserName] = useState<string | undefined>(undefined);
  const [simulations, setSimulations] = useState<SimulationRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function loadDashboard() {
      const { data } = await supabase.auth.getSession();
      const user = data.session?.user;
      if (!user) {
        router.push("/login");
        return;
      }

      const [{ data: profile }, { data: rows }] = await Promise.all([
        supabase.from("profiles").select("name").eq("id", user.id).single(),
        // Enough rows for a meaningful trend chart, not just the most recent
        // handful - RLS already scopes this to the signed-in user's own runs.
        supabase
          .from("simulations")
          .select("id, engine_model, started_at, ended_at, outcome, final_health_percent, final_rul_hours, final_telemetry, model_version, tbo_hours")
          .order("started_at", { ascending: false })
          .limit(30),
      ]);
      if (!cancelled) {
        setUserName(profile?.name || user.user_metadata?.name || user.email || "Pilot");
        setSimulations(rows ?? []);
        setLoading(false);
      }
    }
    loadDashboard();
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session) router.push("/login");
    });
    return () => { cancelled = true; listener.subscription.unsubscribe(); };
  }, [router]);

  const stats = useMemo(() => {
    const withHealth = simulations.filter((s) => s.final_health_percent != null);
    const avgHealth = withHealth.length
      ? withHealth.reduce((sum, s) => sum + (s.final_health_percent ?? 0), 0) / withHealth.length
      : null;

    const rulPercents = simulations
      .map((s) => s.final_rul_hours != null
        ? rulPercentOf(s.final_rul_hours, modelVersionOf(s), elapsedSecondsOf(s), tboHoursOf(s))
        : null)
      .filter((v): v is number => v != null);
    const avgRulPercent = rulPercents.length
      ? rulPercents.reduce((sum, v) => sum + v, 0) / rulPercents.length
      : null;

    const totalFlightHours = simulations.reduce((sum, s) => {
      // v3 runs count engine hours (wear x TBO); legacy v2 runs their real-world equivalent.
      if (usesEngineHours(modelVersionOf(s))) return sum + (engineHoursOf(s) ?? 0);
      const secs = elapsedSecondsOf(s);
      return secs != null ? sum + flightHours(secs, "v2") : sum;
    }, 0);

    return { totalRuns: simulations.length, avgHealth, avgRulPercent, totalFlightHours };
  }, [simulations]);

  // Chronological (oldest -> newest, left to right) for the trend line -
  // simulations itself stays newest-first for the table below.
  const trendData = useMemo(() => {
    return [...simulations]
      .filter((s) => s.final_health_percent != null)
      .reverse()
      .map((s, i) => ({
        run: i + 1,
        health: s.final_health_percent,
        rul: s.final_rul_hours != null ? rulPercentOf(s.final_rul_hours, modelVersionOf(s), elapsedSecondsOf(s), tboHoursOf(s)) : null,
        date: new Date(s.started_at).toLocaleDateString(),
      }));
  }, [simulations]);

  const engineUsage = useMemo(() => {
    const counts = new Map<string, number>();
    for (const s of simulations) counts.set(s.engine_model, (counts.get(s.engine_model) ?? 0) + 1);
    return Array.from(counts.entries()).map(([engine, count]) => ({ engine: formatEngine(engine), count, raw: engine }));
  }, [simulations]);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.push("/home");
  };

  if (userName === undefined) {
    return (
      <>
        <Navbar />
        <main className="pt-16 min-h-screen flex items-center justify-center text-on-surface-variant text-sm">Loading...</main>
      </>
    );
  }

  return (
    <>
      <Navbar />
      <main className="pt-16 relative min-h-screen">
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />

        <div className="relative z-10 max-w-6xl mx-auto px-4 md:px-6 py-12">
          <header className="mb-10">
            <h1 className="font-headline-display text-[28px] md:text-[36px] font-bold text-primary uppercase tracking-tight mb-1">
              Welcome back, {userName}
            </h1>
            <p className="text-tertiary/80 text-[13px] tracking-[0.15em] uppercase font-mono">
              Mission control overview
            </p>
          </header>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
            <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 flex items-center gap-4">
              <div className="p-2.5 bg-tertiary/10 rounded"><ListChecks size={20} className="text-tertiary" /></div>
              <div>
                <div className="text-2xl font-bold text-primary">{stats.totalRuns}</div>
                <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant">Total Runs</div>
              </div>
            </div>
            <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 flex items-center gap-4">
              <div className="p-2.5 bg-tertiary/10 rounded"><TrendingUp size={20} className="text-tertiary" /></div>
              <div>
                <div className="text-2xl font-bold text-primary">{stats.avgHealth != null ? `${stats.avgHealth.toFixed(0)}%` : "--"}</div>
                <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant">Avg. Health</div>
              </div>
            </div>
            <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 flex items-center gap-4">
              <div className="p-2.5 bg-tertiary/10 rounded"><TrendingUp size={20} className="text-tertiary" /></div>
              <div>
                <div className="text-2xl font-bold text-primary">{stats.avgRulPercent != null ? `${stats.avgRulPercent.toFixed(0)}%` : "--"}</div>
                <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant">Avg. RUL</div>
              </div>
            </div>
            <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 flex items-center gap-4">
              <div className="p-2.5 bg-tertiary/10 rounded"><Clock size={20} className="text-tertiary" /></div>
              <div>
                <div className="text-2xl font-bold text-primary">{stats.totalFlightHours.toFixed(1)}h</div>
                <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant" title="v3/v4 runs: engine hours on the engine. Legacy runs: real-world equivalent flight time.">Engine Hours Flown</div>
              </div>
            </div>
          </div>

          {trendData.length > 1 && (
            <div className="mb-8 bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
              <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-4">Health &amp; RUL Trend (last {trendData.length} runs)</h2>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={trendData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#352722" />
                  <XAxis dataKey="run" stroke="#8a7f6a" fontSize={11} tickFormatter={(v) => `#${v}`} />
                  <YAxis stroke="#8a7f6a" fontSize={11} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ background: "#0d0e0d", border: "1px solid #352722", borderRadius: 6, fontSize: 12 }}
                    labelFormatter={(v, p) => p?.[0]?.payload?.date ?? `run #${v}`}
                  />
                  <Line type="monotone" dataKey="health" name="Health %" stroke="#7fc87f" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="rul" name="RUL %" stroke="#ff9f42" dot={false} strokeWidth={2} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {engineUsage.length > 0 && (
            <div className="mb-8 bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
              <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-4">Runs by Engine</h2>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={engineUsage} layout="vertical" margin={{ top: 5, right: 20, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#352722" horizontal={false} />
                  <XAxis type="number" stroke="#8a7f6a" fontSize={11} allowDecimals={false} />
                  <YAxis type="category" dataKey="engine" stroke="#8a7f6a" fontSize={11} width={110} />
                  <Tooltip contentStyle={{ background: "#0d0e0d", border: "1px solid #352722", borderRadius: 6, fontSize: 12 }} />
                  <Bar dataKey="count" name="Runs" radius={[0, 4, 4, 0]}>
                    {engineUsage.map((entry) => (
                      <Cell key={entry.raw} fill={ENGINE_COLORS[entry.raw] ?? "#8a7f6a"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          <EnginesCard />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-10">
            <Link href="/engine" className="group bg-surface/80 border border-tertiary/30 rounded-lg p-6 hover:border-tertiary transition-colors flex items-center justify-between">
              <div>
                <div className="text-lg font-bold text-primary uppercase mb-1">Start New Simulation</div>
                <div className="text-[13px] text-on-surface-variant">Choose an engine and begin a fresh flight</div>
              </div>
              <Rocket size={24} className="text-tertiary group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link href="/telemetry" className="group bg-surface/80 border border-outline-variant/30 rounded-lg p-6 hover:border-tertiary/50 transition-colors flex items-center justify-between">
              <div>
                <div className="text-lg font-bold text-primary uppercase mb-1">View Telemetry History</div>
                <div className="text-[13px] text-on-surface-variant">Browse and resume past simulation runs</div>
              </div>
              <History size={24} className="text-on-surface-variant group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>

          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em]">Recent Simulations</h2>
              <Link href="/telemetry" className="text-[12px] text-tertiary hover:underline">View all</Link>
            </div>

            {loading ? (
              <div className="text-on-surface-variant text-sm py-8 text-center border border-outline-variant/30 rounded-lg">Loading...</div>
            ) : simulations.length === 0 ? (
              <div className="text-on-surface-variant text-sm py-8 text-center border border-outline-variant/30 rounded-lg">
                No simulations yet - <Link href="/engine" className="text-tertiary hover:underline">start your first flight</Link>
              </div>
            ) : (
              <div className="border border-outline-variant/30 rounded-lg overflow-hidden">
                {simulations.slice(0, 8).map((sim) => {
                  const rulPct = sim.final_rul_hours != null
                    ? rulPercentOf(sim.final_rul_hours, modelVersionOf(sim), elapsedSecondsOf(sim), tboHoursOf(sim))
                    : null;
                  return (
                    <Link
                      key={sim.id}
                      href={`/telemetry/${sim.id}`}
                      className="flex items-center justify-between px-4 py-3 border-b border-outline-variant/20 last:border-b-0 hover:bg-surface-container-highest/40 transition-colors"
                    >
                      <div>
                        <div className="text-[13px] text-primary font-medium">{formatEngine(sim.engine_model)}<ModelBadge version={modelVersionOf(sim)} /></div>
                        <div className="text-[11px] text-on-surface-variant">{new Date(sim.started_at).toLocaleString()}</div>
                      </div>
                      <div className="flex items-center gap-5 text-[12px]">
                        <span className="text-on-surface-variant uppercase">{sim.outcome ?? "in progress"}</span>
                        <span className="text-primary w-10 text-right">{sim.final_health_percent != null ? `${sim.final_health_percent.toFixed(0)}%` : "--"}</span>
                        <span className="text-primary w-10 text-right">{rulPct != null ? `${rulPct.toFixed(0)}%` : "--"}</span>
                        <ChevronRight size={16} className="text-on-surface-variant" />
                      </div>
                    </Link>
                  );
                })}
              </div>
            )}
          </section>

          <div className="mt-10 flex justify-end border-t border-outline-variant/30 pt-6">
            <button
              onClick={handleSignOut}
              className="inline-flex items-center gap-2 border border-outline-variant/50 px-4 py-2 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-on-surface-variant transition-colors hover:border-tertiary hover:text-tertiary"
            >
              <LogOut size={14} /> Sign out
            </button>
          </div>
        </div>
      </main>
    </>
  );
}
