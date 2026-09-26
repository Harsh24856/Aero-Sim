"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, FileText } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { modelVersionOf } from "@/lib/timeScale";
import ModelBadge from "@/components/ModelBadge";
import { missionName } from "@/lib/missionPresets";
import Navbar from "@/components/Navbar";
import MissionReplay, { type ReplayRow } from "@/components/MissionReplay";

const REPLAY_COLUMNS =
  "time_offset_s, altitude, throttle, airspeed, aoa, engine_rpm, prop_rpm, power_kw, " +
  "fuel_flow, thrust, lift, drag, egt, cht, oil_pressure, oil_temp, vibx, viby, vibz, " +
  "fault_detected, detection_confidence, health_percent, rul_percent_remaining, rul_hours";
// physics v4 rows (backend/dbv4.py) carry both clocks, three more instruments, the
// twin residuals, the AI's component calls and the injected ground truth.
const V4_REPLAY_COLUMNS =
  ", engine_hours, coolant_temp, manifold_pressure_kpa, battery_voltage, margin_min, " +
  "rul_calendar_hours, rul_band_hours, wear_limited, fault_modes, residuals, truth";

type SimMeta = {
  id: number;
  engine_model: string;
  started_at: string;
  ended_at: string | null;
  outcome: string | null;
  model_version: string | null;
  mission?: string | null;
};

export default function MissionReplayPage() {
  const params = useParams();
  const id = Array.isArray(params?.id) ? params.id[0] : params?.id;
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [sim, setSim] = useState<SimMeta | null>(null);
  const [rows, setRows] = useState<ReplayRow[]>([]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) { router.push("/login"); return; }

      // Treat any error/empty as not-found rather than surfacing detail: with RLS
      // on, "exists but not yours" and "does not exist" should look identical.
      const { data: simRow, error } = await supabase
        .from("simulations")
        .select("id, engine_model, started_at, ended_at, outcome, model_version, mission")
        .eq("id", id)
        .single();
      if (cancelled) return;
      if (error || !simRow) { setNotFound(true); setLoading(false); return; }
      setSim(simRow as SimMeta);

      // RLS policy "select own telemetry" already scopes this via the parent run.
      const { data: logRows } = await supabase
        .from("telemetry_logs")
        .select(simRow.model_version === "v4" ? REPLAY_COLUMNS + V4_REPLAY_COLUMNS : REPLAY_COLUMNS)
        .eq("simulation_id", id)
        .order("time_offset_s", { ascending: true });
      if (cancelled) return;
      setRows((logRows as unknown as ReplayRow[]) ?? []);
      setLoading(false);
    })();
    return () => { cancelled = true; };
  }, [id, router]);

  return (
    <>
      <Navbar />
      <main className="pt-16 relative min-h-screen">
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />
        <div className="relative z-10 max-w-5xl mx-auto px-4 md:px-6 py-12">

          <Link href="/mission" className="inline-flex items-center gap-2 text-on-surface-variant hover:text-primary transition-colors text-[12px] mb-6">
            <ArrowLeft className="h-4 w-4" /> Back to missions
          </Link>

          {loading && <div className="text-on-surface-variant text-[13px]">Loading mission&hellip;</div>}
          {notFound && <div className="text-on-surface-variant text-[13px]">Mission not found.</div>}

          {!loading && !notFound && sim && (
            <>
              <header className="flex flex-wrap items-start justify-between gap-4 mb-8">
                <div>
                  <h1 className="font-headline-display text-[26px] md:text-[32px] font-bold text-primary uppercase tracking-tight">
                    {sim.engine_model.replace(/_/g, " ")}
                    <ModelBadge version={modelVersionOf(sim)} />
                    <span className="ml-3 text-[12px] font-mono text-tertiary/70 align-middle">ID #{sim.id}</span>
                  </h1>
                  <div className="text-[12px] text-on-surface-variant mt-1">
                    {missionName(sim.mission) && <span className="text-tertiary">{missionName(sim.mission)} &middot; </span>}
                    {new Date(sim.started_at).toLocaleString()}
                    {sim.ended_at && <> &rarr; {new Date(sim.ended_at).toLocaleString()}</>}
                    {sim.outcome && <> &middot; {sim.outcome.toUpperCase()}</>}
                  </div>
                </div>
                <Link
                  href={`/mission/report/${sim.id}`}
                  className="rounded bg-surface/80 border border-outline-variant/30 px-5 py-2.5 text-[11px] font-bold uppercase tracking-[0.1em] text-on-surface-variant hover:text-primary transition-all flex items-center gap-2"
                >
                  <FileText className="h-4 w-4" /> Mission Report
                </Link>
              </header>

              <MissionReplay rows={rows} modelVersion={modelVersionOf(sim)} />
            </>
          )}
        </div>
      </main>
    </>
  );
}
