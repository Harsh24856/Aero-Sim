"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { supabase } from "@/lib/supabase";
import Navbar from "@/components/Navbar";
import { MISSION_PRESETS, presetDuration, type MissionPreset } from "@/lib/missionPresets";
import { simSecondsToRealHours } from "@/lib/timeScale";
import { Mountain, Clock, Thermometer, Activity, Play, Film, FileText, ChevronRight } from "lucide-react";

const ENGINES = [
  { id: "Rotax_914_ULF", label: "914 F/UL" },
  { id: "Rotax_912_ULS", label: "912 ULS" },
  { id: "Rotax_915_iS", label: "915 iS" },
  { id: "Rotax_916_iS", label: "916 iS" },
];

const PRESET_ICON: Record<string, React.ComponentType<{ className?: string }>> = {
  high_altitude: Mountain,
  endurance: Clock,
  hot_weather: Thermometer,
  rapid_throttle: Activity,
};

type SimRow = {
  id: number;
  engine_model: string;
  started_at: string;
  outcome: string | null;
  final_health_percent: number | null;
};

export default function MissionPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [engine, setEngine] = useState(ENGINES[0].id);
  const [runs, setRuns] = useState<SimRow[]>([]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) { router.push("/login"); return; }
      // RLS scopes this to the signed-in user; no .eq("user_id") needed.
      const { data } = await supabase
        .from("simulations")
        .select("id, engine_model, started_at, outcome, final_health_percent")
        .order("started_at", { ascending: false })
        .limit(6);
      if (cancelled) return;
      setRuns((data as SimRow[]) ?? []);
      setLoading(false);
    })();
    return () => { cancelled = true; };
  }, [router]);

  const launch = (preset: MissionPreset) => {
    router.push(`/simulate?engine=${encodeURIComponent(engine)}&preset=${encodeURIComponent(preset.id)}`);
  };

  return (
    <>
      <Navbar />
      <main className="pt-16 relative min-h-screen">
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />
        <div className="relative z-10 max-w-6xl mx-auto px-4 md:px-6 py-12">

          <header className="mb-8">
            <h1 className="font-headline-display text-[28px] md:text-[36px] font-bold text-primary uppercase tracking-tight mb-1">
              Mission Profiles
            </h1>
            <p className="text-tertiary/80 text-[13px] tracking-[0.15em] uppercase font-mono">
              Scripted environmental and operating scenarios
            </p>
          </header>

          {/* engine selection */}
          <section className="mb-8">
            <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-3">Engine</h2>
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
            <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-4">Profiles</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {MISSION_PRESETS.map((p) => {
                const Icon = PRESET_ICON[p.id] ?? Activity;
                const dur = presetDuration(p);
                return (
                  <article key={p.id} className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 flex flex-col">
                    <div className="flex items-start gap-3 mb-3">
                      <div className="p-2.5 bg-tertiary/10 rounded shrink-0">
                        <Icon className="h-5 w-5 text-tertiary" />
                      </div>
                      <div className="min-w-0">
                        <h3 className="text-primary font-bold text-[15px] leading-tight">{p.name}</h3>
                        <div className="text-[10px] uppercase tracking-[0.1em] text-tertiary/80 mt-0.5">{p.tagline}</div>
                      </div>
                    </div>

                    <p className="text-[12px] leading-relaxed text-on-surface-variant mb-3">{p.description}</p>

                    <div className="bg-surface/60 border border-outline-variant/30 rounded p-3 mb-3">
                      <div className="text-[10px] uppercase tracking-[0.1em] text-tertiary/80 mb-2">Profile legs</div>
                      <ol className="space-y-1">
                        {p.legs.map((l, i) => (
                          <li key={i} className="flex items-center justify-between gap-3 text-[11px]">
                            <span className="text-on-surface-variant truncate">
                              <span className="text-tertiary/70 font-mono mr-1.5">{String(i + 1).padStart(2, "0")}</span>
                              {l.label}
                            </span>
                            <span className="text-primary font-mono shrink-0">
                              {Math.round(l.altitude)} m &middot; {Math.round(l.throttle * 100)}% &middot; {l.seconds}s
                            </span>
                          </li>
                        ))}
                      </ol>
                    </div>

                    <div className="text-[11px] text-on-surface-variant mb-4">
                      <span className="text-[10px] uppercase tracking-[0.1em] text-tertiary/80">Watch for </span>
                      {p.watchFor}
                    </div>

                    <div className="mt-auto flex items-center justify-between gap-3">
                      <span className="text-[10px] uppercase tracking-[0.1em] text-on-surface-variant">
                        {dur}s sim &middot; {simSecondsToRealHours(dur).toFixed(2)} h real
                      </span>
                      <button
                        onClick={() => launch(p)}
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
                No missions recorded yet. Launch a profile above.
              </div>
            ) : (
              <div className="border border-outline-variant/30 rounded-lg overflow-hidden">
                {runs.map((r) => (
                  <div
                    key={r.id}
                    className="grid grid-cols-[auto_1fr_auto_auto] gap-4 items-center px-4 py-3 border-b border-outline-variant/20 last:border-b-0 hover:bg-surface-container-highest/40 transition-colors"
                  >
                    <span className="text-tertiary/70 font-mono text-[11px]">#{r.id}</span>
                    <span className="text-primary text-[13px] truncate">
                      {r.engine_model.replace(/_/g, " ")}
                      <span className="text-on-surface-variant ml-2 text-[11px]">
                        {new Date(r.started_at).toLocaleString()}
                      </span>
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
                ))}
              </div>
            )}
          </section>

        </div>
      </main>
    </>
  );
}
