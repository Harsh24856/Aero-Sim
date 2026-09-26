"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { missionName } from "@/lib/missionPresets";
import Navbar from "@/components/Navbar";
import { supabase } from "@/lib/supabase";
import { ChevronRight, TrendingUp, Activity, ListChecks } from "lucide-react";
import { modelVersionOf, rulPercentOf, tboHoursOf } from "@/lib/timeScale";
import ModelBadge from "@/components/ModelBadge";

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
  mission?: string | null;
};

function outcomeBadgeClass(outcome: string | null): string {
  if (!outcome) return "border-tertiary/40 bg-tertiary/10 text-tertiary";
  if (outcome === "stopped") return "border-[#3a5a3a] bg-[#122015] text-[#7fc87f]";
  // A resumed run is still open - it is being flown again under the same id.
  if (outcome === "resumed") return "border-tertiary/40 bg-tertiary/10 text-tertiary";
  if (outcome === "reset" || outcome === "engine_switched") return "border-outline-variant/40 bg-surface-container-highest/40 text-on-surface-variant";
  return "border-outline-variant/40 bg-surface-container-highest/40 text-on-surface-variant";
}

export default function TelemetryListPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [simulations, setSimulations] = useState<SimulationRow[]>([]);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const { data: sessionData } = await supabase.auth.getSession();
      if (!sessionData.session) {
        router.push("/login");
        return;
      }
      // RLS already scopes this to the signed-in user's own rows - no need for
      // an explicit .eq("user_id", ...) filter, the policy enforces it either way.
      const { data, error } = await supabase
        .from("simulations")
        .select("id, engine_model, started_at, ended_at, outcome, final_health_percent, final_rul_hours, final_telemetry, model_version, tbo_hours, mission")
        .order("started_at", { ascending: false });
      if (!cancelled) {
        if (!error && data) setSimulations(data);
        setLoading(false);
      }
    }
    load();

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session) router.push("/login");
    });
    return () => {
      cancelled = true;
      listener.subscription.unsubscribe();
    };
  }, [router]);

  const totalRuns = simulations.length;
  const avgHealth = (() => {
    const withHealth = simulations.filter((s) => s.final_health_percent != null);
    if (withHealth.length === 0) return null;
    return withHealth.reduce((sum, s) => sum + (s.final_health_percent ?? 0), 0) / withHealth.length;
  })();
  const bestRulPercent = simulations.reduce<number | null>((best, s) => {
    if (s.final_rul_hours == null) return best;
    const pct = rulPercentOf(s.final_rul_hours, modelVersionOf(s), typeof s.final_telemetry?.time === "number" ? s.final_telemetry.time : undefined, tboHoursOf(s));
    if (pct == null) return best;
    return best == null ? pct : Math.max(best, pct);
  }, null);

  return (
    <>
      <Navbar />
      <main className="pt-16 relative min-h-screen">
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />

        <div className="relative z-10 max-w-6xl mx-auto px-4 md:px-6 py-12">
          <header className="mb-8">
            <h1 className="font-headline-display text-[28px] md:text-[36px] font-bold text-primary uppercase tracking-tight mb-1">
              Telemetry History
            </h1>
            <p className="text-tertiary/80 text-[13px] tracking-[0.15em] uppercase font-mono">
              Every simulation run, all engines
            </p>
          </header>

          {!loading && simulations.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
              <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 flex items-center gap-4">
                <div className="p-2.5 bg-tertiary/10 rounded"><ListChecks size={20} className="text-tertiary" /></div>
                <div>
                  <div className="text-2xl font-bold text-primary">{totalRuns}</div>
                  <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant">Total Runs</div>
                </div>
              </div>
              <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 flex items-center gap-4">
                <div className="p-2.5 bg-tertiary/10 rounded"><Activity size={20} className="text-tertiary" /></div>
                <div>
                  <div className="text-2xl font-bold text-primary">{avgHealth != null ? `${avgHealth.toFixed(0)}%` : "--"}</div>
                  <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant">Avg. Final Health</div>
                </div>
              </div>
              <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 flex items-center gap-4">
                <div className="p-2.5 bg-tertiary/10 rounded"><TrendingUp size={20} className="text-tertiary" /></div>
                <div>
                  <div className="text-2xl font-bold text-primary">{bestRulPercent != null ? `${Math.min(100, bestRulPercent).toFixed(0)}%` : "--"}</div>
                  <div className="text-[11px] uppercase tracking-[0.1em] text-on-surface-variant">Best RUL</div>
                </div>
              </div>
            </div>
          )}

          {loading ? (
            <div className="text-on-surface-variant text-sm py-16 text-center border border-outline-variant/30 rounded-lg">
              Loading...
            </div>
          ) : simulations.length === 0 ? (
            <div className="text-on-surface-variant text-sm py-16 text-center border border-outline-variant/30 rounded-lg">
              No simulations yet - <Link href="/engine" className="text-tertiary hover:underline">start your first flight</Link>
            </div>
          ) : (
            <div className="border border-outline-variant/30 rounded-lg overflow-hidden">
              <div className="grid grid-cols-[1fr_auto_auto_auto_auto] gap-4 px-4 py-2.5 bg-surface-container-highest/40 text-[10px] uppercase tracking-[0.1em] text-on-surface-variant border-b border-outline-variant/30">
                <span>Engine / Started</span>
                <span className="w-24 text-right">Outcome</span>
                <span className="w-16 text-right">Health</span>
                <span className="w-16 text-right">RUL</span>
                <span className="w-5" />
              </div>
              {simulations.map((sim) => {
                const rulPct = sim.final_rul_hours != null
                  ? rulPercentOf(sim.final_rul_hours, modelVersionOf(sim), typeof sim.final_telemetry?.time === "number" ? sim.final_telemetry.time : undefined, tboHoursOf(sim))
                  : null;
                return (
                  <Link
                    key={sim.id}
                    href={`/telemetry/${sim.id}`}
                    className="grid grid-cols-[1fr_auto_auto_auto_auto] gap-4 items-center px-4 py-3.5 border-b border-outline-variant/20 last:border-b-0 hover:bg-surface-container-highest/40 transition-colors"
                  >
                    <div>
                      <div className="text-[13px] text-primary font-medium">{sim.engine_model.replace(/_/g, " ")}<ModelBadge version={modelVersionOf(sim)} /></div>
                      <div className="text-[11px] text-on-surface-variant">
                        {missionName(sim.mission) && <span className="text-tertiary">{missionName(sim.mission)} &middot; </span>}
                        {new Date(sim.started_at).toLocaleString()}
                      </div>
                    </div>
                    <span className={`w-24 justify-self-end text-center rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.05em] ${outcomeBadgeClass(sim.outcome)}`}>
                      {sim.outcome ?? "active"}
                    </span>
                    <span className="w-16 text-right text-[13px] text-primary font-mono">{sim.final_health_percent != null ? `${sim.final_health_percent.toFixed(0)}%` : "--"}</span>
                    <span className="w-16 text-right text-[13px] text-primary font-mono">{rulPct != null ? `${rulPct.toFixed(0)}%` : "--"}</span>
                    <ChevronRight size={16} className="text-on-surface-variant" />
                  </Link>
                );
              })}
            </div>
          )}
        </div>
      </main>
    </>
  );
}
