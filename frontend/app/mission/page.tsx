"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { supabase } from "@/lib/supabase";
import Navbar from "@/components/Navbar";
import { ENGINE_INPUTS } from "@/components/Meters";
import {
  MISSION_PRESETS, eventsFor, missionName, presetDuration, presetEngineHours,
  type MissionEnv, type MissionEvent, type MissionPreset,
} from "@/lib/missionPresets";
import { formatSimClock } from "@/lib/timeScale";
import { Mountain, Clock, Thermometer, Activity, Crosshair, Play, Film, FileText, ChevronRight, Zap } from "lucide-react";

const API = "http://localhost:8000";

const ENGINES = [
  { id: "Rotax_914_ULF", label: "914 ULF" },
  { id: "Rotax_912_ULS", label: "912 ULS" },
  { id: "Rotax_915_iS", label: "915 iS" },
  { id: "Rotax_916_iS", label: "916 iS" },
];
const engineLabel = (id: string) => ENGINES.find((e) => e.id === id)?.label ?? id.replace(/_/g, " ");

const PRESET_ICON: Record<string, React.ComponentType<{ className?: string }>> = {
  isr_endurance: Clock,
  high_altitude: Mountain,
  hot_high: Thermometer,
  rapid_throttle: Activity,
  fault_isolation: Crosshair,
};

const SCENARIO_LABEL: Record<string, string> = { healthy: "Healthy engine, 300 h" };

const OUTCOME: Record<string, { label: string; tone: string }> = {
  engine_failure: { label: "Engine failure", tone: "text-[#ff6a4d]" },
  stopped: { label: "Completed", tone: "text-[#7fc87f]" },
  reset: { label: "Reset", tone: "text-on-surface-variant" },
  engine_switched: { label: "Engine switched", tone: "text-on-surface-variant" },
  server_shutdown: { label: "Interrupted", tone: "text-[#ffd27a]" },
};

/** "ISA +28 °C", "Humidity 70%" - the mission's inputs, in the cockpit's own units. */
function envChips(env: MissionEnv | undefined): string[] {
  if (!env) return [];
  const chips = ENGINE_INPUTS
    .filter((f) => typeof env[f.key as keyof MissionEnv] === "number")
    .map((f) => `${f.label} ${f.show(env[f.key as keyof MissionEnv] as number)}`);
  if (env.oil_thermostat_open === false) chips.push("Oil thermostat stuck");
  return chips;
}

type SimRow = {
  id: number;
  engine_model: string;
  started_at: string;
  outcome: string | null;
  final_health_percent: number | null;
  scenario: string | null;
  mission: string | null;
};

export default function MissionPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [engine, setEngine] = useState(ENGINES[0].id);
  // Preset awaiting an engine choice. Null = no launch dialog open.
  const [pending, setPending] = useState<MissionPreset | null>(null);
  const [runs, setRuns] = useState<SimRow[]>([]);
  // The backend's life scale (engine hours per flight hour); null when it isn't running v4.
  const [lifeScale, setLifeScale] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) { router.push("/login"); return; }
      // RLS scopes this to the signed-in user; no .eq("user_id") needed.
      const { data } = await supabase
        .from("simulations")
        .select("id, engine_model, started_at, outcome, final_health_percent, scenario, mission")
        .order("started_at", { ascending: false })
        .limit(6);
      if (cancelled) return;
      setRuns((data as SimRow[]) ?? []);
      setLoading(false);
    })();
    fetch(`${API}/scenarios`).then((r) => r.json())
      .then((r) => { if (!cancelled && r?.status === "ok") setLifeScale(r.life_scale ?? null); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [router]);

  const launch = (preset: MissionPreset, engineId: string) => {
    router.push(`/simulate?engine=${encodeURIComponent(engineId)}&preset=${encodeURIComponent(preset.id)}`);
  };

  const eventLine = (e: MissionEvent) => (
    <li key={`${e.at}:${e.label}`} className="flex items-center justify-between gap-3 text-[11px]">
      <span className="text-on-surface-variant truncate">
        <span className="text-[#ffd27a] font-mono mr-1.5">T+{formatSimClock(e.at)}</span>
        {e.label}
      </span>
      {e.engines && (
        <span className="shrink-0 text-[10px] text-on-surface-variant/70">{e.engines.map(engineLabel).join(" / ")} only</span>
      )}
    </li>
  );

  return (
    <>
      <Navbar />
      <main className="pt-16 relative min-h-screen">
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />
        <div className="relative z-10 max-w-6xl mx-auto px-4 md:px-6 py-12">

          <header className="mb-8">
            <h1 className="font-headline-display text-[28px] md:text-[36px] font-bold text-primary uppercase tracking-tight mb-1">
              Missions
            </h1>
            <p className="text-tertiary/80 text-[13px] tracking-[0.15em] uppercase font-mono">
              MALE-UAV sorties &middot; PS 26054 section E
            </p>
            <p className="mt-3 max-w-3xl text-[12px] leading-relaxed text-on-surface-variant">
              Each sortie picks the engine&rsquo;s condition, sets its weather and fuel, flies scripted legs and
              injects faults on the mission clock, so you can watch the twin and the AI respond. The clock starts
              when the AI comes online (after its 128 s window); flight time is real time, while engine hours
              {lifeScale != null ? ` run ${lifeScale}× faster` : " run faster"}, so a sortie visibly ages the engine.
            </p>
          </header>

          {/* Default engine. Each profile still asks before launching, so this
              is a convenience preselection rather than the only choice. */}
          <section className="mb-8">
            <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-3">Default Engine</h2>
            <div className="flex flex-wrap gap-2">
              {ENGINES.map((e) => (
                <button
                  key={e.id}
                  onClick={() => setEngine(e.id)}
                  className={`rounded px-4 py-2 text-[11px] font-bold uppercase tracking-[0.1em] transition-all ${
                    engine === e.id
                      ? "bg-tertiary text-black"
                      : "bg-surface/80 border border-outline-variant/30 text-on-surface-variant hover:text-primary"
                  }`}
                >
                  {e.label}
                </button>
              ))}
            </div>
          </section>

          {/* profiles */}
          <section className="mb-12">
            <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-4">Sorties</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {MISSION_PRESETS.map((p) => {
                const Icon = PRESET_ICON[p.id] ?? Activity;
                const dur = presetDuration(p);
                const chips = envChips(p.env);
                return (
                  <article key={p.id} className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 flex flex-col">
                    <div className="flex items-start gap-3 mb-3">
                      <div className="p-2.5 bg-tertiary/10 rounded shrink-0">
                        <Icon className="h-5 w-5 text-tertiary" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <h3 className="text-primary font-bold text-[15px] leading-tight">{p.name}</h3>
                          <span className="rounded border border-tertiary/40 px-1.5 py-0.5 text-[9px] uppercase tracking-[0.1em] text-tertiary">
                            {p.requirement}
                          </span>
                        </div>
                        <div className="text-[10px] uppercase tracking-[0.1em] text-tertiary/80 mt-0.5">{p.tagline}</div>
                      </div>
                    </div>

                    <p className="text-[12px] leading-relaxed text-on-surface-variant mb-3">{p.description}</p>

                    <div className="mb-3 flex flex-wrap gap-1.5 text-[10px]">
                      <span className="rounded bg-black/40 px-2 py-0.5 text-on-surface-variant">
                        {SCENARIO_LABEL[p.scenario] ?? p.scenario.replace(/_/g, " ")}
                      </span>
                      {(chips.length ? chips : ["Standard day"]).map((c) => (
                        <span key={c} className="rounded bg-black/40 px-2 py-0.5 text-on-surface-variant">{c}</span>
                      ))}
                    </div>

                    <div className="bg-surface/60 border border-outline-variant/30 rounded p-3 mb-3">
                      <div className="text-[10px] uppercase tracking-[0.1em] text-tertiary/80 mb-2">Legs</div>
                      <ol className="space-y-1">
                        {p.legs.map((l, i) => (
                          <li key={i} className="flex items-center justify-between gap-3 text-[11px]">
                            <span className="text-on-surface-variant truncate">
                              <span className="text-tertiary/70 font-mono mr-1.5">{String(i + 1).padStart(2, "0")}</span>
                              {l.label}
                            </span>
                            <span className="text-primary font-mono shrink-0">
                              {Math.round(l.altitude).toLocaleString()} m &middot; {Math.round(l.throttle * 100)}% &middot; {Math.round(l.airspeed * 1.943844)} kt &middot; {formatSimClock(l.seconds)}
                            </span>
                          </li>
                        ))}
                      </ol>
                      {(p.events?.length ?? 0) > 0 && (
                        <>
                          <div className="mt-3 mb-2 flex items-center gap-1.5 text-[10px] uppercase tracking-[0.1em] text-tertiary/80">
                            <Zap className="h-3 w-3" /> Events
                          </div>
                          <ol className="space-y-1">{p.events!.map(eventLine)}</ol>
                        </>
                      )}
                    </div>

                    <div className="text-[11px] text-on-surface-variant mb-4">
                      <span className="text-[10px] uppercase tracking-[0.1em] text-tertiary/80">Watch for </span>
                      {p.watchFor}
                    </div>

                    <div className="mt-auto flex items-center justify-between gap-3">
                      <span className="text-[10px] uppercase tracking-[0.1em] text-on-surface-variant">
                        {formatSimClock(dur)} flight
                        {lifeScale != null && <> &middot; +{Math.round(presetEngineHours(p, lifeScale))} engine h</>}
                      </span>
                      <button
                        onClick={() => setPending(p)}
                        className="rounded bg-tertiary px-5 py-2.5 text-[11px] font-bold uppercase tracking-[0.1em] text-black hover:brightness-110 transition-all flex items-center gap-2"
                      >
                        <Play className="h-3.5 w-3.5" /> Launch
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>

          {/* recent missions -> replay / report */}
          <section>
            <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-4">Recent Missions</h2>
            {loading ? (
              <div className="text-on-surface-variant text-[13px]">Loading&hellip;</div>
            ) : runs.length === 0 ? (
              <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 text-on-surface-variant text-[13px]">
                No missions recorded yet. Launch a sortie above.
              </div>
            ) : (
              <div className="border border-outline-variant/30 rounded-lg overflow-hidden">
                {runs.map((r) => {
                  const outcome = r.outcome ? OUTCOME[r.outcome] ?? { label: r.outcome, tone: "text-on-surface-variant" } : null;
                  return (
                    <div
                      key={r.id}
                      className="grid grid-cols-[auto_1fr_auto_auto_auto] gap-4 items-center px-4 py-3 border-b border-outline-variant/20 last:border-b-0 hover:bg-surface-container-highest/40 transition-colors"
                    >
                      <span className="text-tertiary/70 font-mono text-[11px]">#{r.id}</span>
                      <span className="text-primary text-[13px] truncate">
                        {missionName(r.mission) && <span className="text-tertiary mr-2">{missionName(r.mission)}</span>}
                        {engineLabel(r.engine_model)}
                        {r.scenario && <span className="text-on-surface-variant ml-2 text-[11px]">{r.scenario.replace(/_/g, " ")}</span>}
                        <span className="text-on-surface-variant ml-2 text-[11px]">
                          {new Date(r.started_at).toLocaleString()}
                        </span>
                      </span>
                      <span className="text-right text-[11px]">
                        <span className={outcome?.tone ?? "text-on-surface-variant"}>{outcome?.label ?? "In progress"}</span>
                        {r.final_health_percent != null && (
                          <span className="ml-2 font-mono text-on-surface-variant">{r.final_health_percent.toFixed(0)}%</span>
                        )}
                      </span>
                      <Link
                        href={`/mission/replay/${r.id}`}
                        className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.1em] text-tertiary hover:brightness-125"
                      >
                        <Film className="h-3.5 w-3.5" /> Replay
                      </Link>
                      <Link
                        href={`/mission/report/${r.id}`}
                        className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.1em] text-on-surface-variant hover:text-primary"
                      >
                        <FileText className="h-3.5 w-3.5" /> Report <ChevronRight className="h-3 w-3" />
                      </Link>
                    </div>
                  );
                })}
              </div>
            )}
          </section>

        </div>

        {/* Engine selection for a chosen profile. Asked per launch so the same
            profile can be flown on different engines without going back up to
            the page-level default. */}
        {pending && (() => {
          const applies = eventsFor(pending, engine);
          const skipped = (pending.events ?? []).filter((e) => !applies.includes(e));
          const chips = envChips(pending.env);
          return (
            <div
              className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
              role="dialog"
              aria-modal="true"
              aria-label={`Select engine for ${pending.name}`}
              onClick={() => setPending(null)}
            >
              <div
                className="w-full max-w-md rounded-lg border border-tertiary/40 bg-[#0d0e0d] p-6"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="text-[10px] uppercase tracking-[0.1em] text-tertiary mb-1">Launch sortie &middot; {pending.requirement}</div>
                <h3 className="text-primary font-bold text-[18px] mb-1">{pending.name}</h3>
                <p className="text-[12px] text-on-surface-variant mb-5">{pending.tagline}</p>

                <div className="text-[10px] uppercase tracking-[0.1em] text-tertiary/80 mb-2">Select engine</div>
                <div className="grid grid-cols-2 gap-2 mb-5">
                  {ENGINES.map((e) => (
                    <button
                      key={e.id}
                      onClick={() => setEngine(e.id)}
                      className={`rounded px-3 py-2.5 text-[11px] font-bold uppercase tracking-[0.1em] transition-all ${
                        engine === e.id
                          ? "bg-tertiary text-black"
                          : "bg-surface/60 border border-outline-variant/30 text-on-surface-variant hover:text-primary"
                      }`}
                    >
                      {e.label}
                    </button>
                  ))}
                </div>

                <div className="rounded border border-outline-variant/30 bg-black/40 p-3 mb-5 space-y-1.5 text-[11px] text-on-surface-variant">
                  <div>
                    {pending.legs.length} legs &middot; {formatSimClock(presetDuration(pending))} flight
                    {lifeScale != null && <> &middot; +{Math.round(presetEngineHours(pending, lifeScale))} engine h</>}
                  </div>
                  <div>{SCENARIO_LABEL[pending.scenario] ?? pending.scenario} &middot; {chips.length ? chips.join(", ") : "standard day"}</div>
                  {applies.length > 0 && (
                    <div>Events: {applies.map((e) => `${e.label} at T+${formatSimClock(e.at)}`).join("; ")}</div>
                  )}
                  {skipped.length > 0 && (
                    <div className="text-[#ffd27a]">
                      Not on a {engineLabel(engine)}: {skipped.map((e) => e.label).join("; ")} - it has no such hardware.
                    </div>
                  )}
                </div>

                <div className="flex flex-col gap-2 sm:flex-row">
                  <button
                    onClick={() => launch(pending, engine)}
                    className="w-full rounded bg-tertiary py-2.5 text-[11px] font-bold uppercase tracking-[0.1em] text-black hover:brightness-110 transition-all flex items-center justify-center gap-2"
                  >
                    <Play className="h-3.5 w-3.5" /> Launch on {engineLabel(engine)}
                  </button>
                  <button
                    onClick={() => setPending(null)}
                    className="w-full rounded border border-outline-variant/30 bg-black/40 py-2.5 text-[11px] font-bold uppercase tracking-[0.1em] text-on-surface-variant hover:text-primary transition-all"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          );
        })()}
      </main>
    </>
  );
}
