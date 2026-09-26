"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { supabase } from "@/lib/supabase";

/**
 * The user's physics-v4 engines: each one's hour meter against its TBO, its status,
 * and the latest entry in its maintenance history. An engine's hours carry from one
 * flight to the next (the engines table), so this is where the life clock is seen
 * across flights. Renders nothing for a user with no v4 engines yet.
 */
type EngineRow = {
  id: number; name: string; engine_model: string; engine_hours: number;
  tbo_hours: number; status: string; scenario: string | null;
};
type EventRow = { engine_id: number; engine_hours: number; level: string; subject: string; created_at: string };

const LEVEL_TONE: Record<string, string> = {
  warning: "text-[#ff9a72]", caution: "text-[#ffd27a]", advisory: "text-[#e8c9a0]",
};

export default function EnginesCard() {
  const [engines, setEngines] = useState<EngineRow[]>([]);
  const [latest, setLatest] = useState<Record<number, EventRow>>({});

  useEffect(() => {
    let cancelled = false;
    (async () => {
      // RLS scopes both tables to the signed-in user's own engines.
      const { data: rows } = await supabase
        .from("engines")
        .select("id, name, engine_model, engine_hours, tbo_hours, status, scenario")
        .order("updated_at", { ascending: false })
        .limit(12);
      const list = (rows as EngineRow[] | null) ?? [];
      if (cancelled || list.length === 0) { setEngines(list); return; }
      const { data: events } = await supabase
        .from("maintenance_events")
        .select("engine_id, engine_hours, level, subject, created_at")
        .in("engine_id", list.map((e) => e.id))
        .order("created_at", { ascending: false })
        .limit(50);
      const byEngine: Record<number, EventRow> = {};
      for (const ev of (events as EventRow[] | null) ?? []) byEngine[ev.engine_id] ??= ev;
      if (!cancelled) { setEngines(list); setLatest(byEngine); }
    })();
    return () => { cancelled = true; };
  }, []);

  if (engines.length === 0) return null;

  return (
    <div className="mb-8 bg-surface/80 border border-outline-variant/30 rounded-lg p-5">
      <h2 className="text-sm font-bold text-primary uppercase tracking-[0.1em] mb-1">Engines</h2>
      <p className="text-[11px] text-on-surface-variant mb-4">
        Each engine keeps its hour meter and its faults from flight to flight. Continue one from the cockpit&apos;s engine menu.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {engines.map((e) => {
          const life = Math.max(0, Math.min(1, e.engine_hours / e.tbo_hours));
          const ev = latest[e.id];
          return (
            <div key={e.id} className="border border-outline-variant/30 rounded p-3 bg-black/30">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-[13px] font-medium text-primary">
                  {e.name} <span className="text-[11px] text-on-surface-variant">{e.engine_model.replace(/_/g, " ")}</span>
                </span>
                <span className={`text-[10px] uppercase tracking-[0.1em] ${e.status === "worn_out" ? "text-[#ff9a72]" : "text-[#7fc87f]"}`}>
                  {e.status.replace(/_/g, " ")}
                </span>
              </div>
              <div className="mt-2 flex items-baseline justify-between text-[11px] font-mono text-on-surface-variant">
                <span>{e.engine_hours.toFixed(1)} h</span>
                <span>TBO {e.tbo_hours.toFixed(0)} h</span>
              </div>
              <div className="mt-1 h-1.5 w-full overflow-hidden rounded bg-[#241a16]">
                <div className={`h-full ${life > 0.85 ? "bg-[#ff8050]" : "bg-tertiary"}`} style={{ width: `${life * 100}%` }} />
              </div>
              {ev && (
                <div className={`mt-2 text-[11px] leading-snug ${LEVEL_TONE[ev.level] ?? "text-on-surface-variant"}`}>
                  {ev.level} at {ev.engine_hours.toFixed(1)} h: {ev.subject}
                </div>
              )}
              <Link href={`/simulate?engine=${e.engine_model}`}
                    className="mt-2 inline-block text-[10px] font-bold uppercase tracking-[0.1em] text-tertiary hover:brightness-125">
                Fly this engine &rarr;
              </Link>
            </div>
          );
        })}
      </div>
    </div>
  );
}
