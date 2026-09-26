"use client";

import { useEffect, useRef, useState } from "react";
import { RESIDUAL_CHANNELS } from "@/lib/v4";

/**
 * The digital twin, made visible. For each of the six channels the on-board twin
 * covers: what the sensor reads (orange) against what a perfectly healthy engine
 * reads at the same moment (grey dashed), over the last 128 s - the same window the
 * AI looks at. The shaded gap IS the residual the models are fed.
 *
 * Keeps its own 1 Hz history from the live frames (one point per flight second);
 * nothing extra is sent by the backend.
 */
const WINDOW = 128;

type Point = { t: number; m: Record<string, number>; w: Record<string, number> };

export default function TwinResidualChart({
  frame,
  suspect = [],
}: {
  frame?: Record<string, unknown> | null;
  suspect?: string[];     // channels the AI says are faulty sensors: drawn muted
}) {
  const hist = useRef<Point[]>([]);
  const [, redraw] = useState(0);

  useEffect(() => {
    const t = typeof frame?.time === "number" ? Math.floor(frame.time as number) : null;
    const twin = frame?.twin as Record<string, number> | undefined;
    if (t === null || !twin) return;
    const h = hist.current;
    if (h.length && t < h[h.length - 1].t) h.length = 0;           // a new flight
    if (h.length && t === h[h.length - 1].t) return;
    const m: Record<string, number> = {}, w: Record<string, number> = {};
    for (const { key } of RESIDUAL_CHANNELS) {
      if (typeof frame?.[key] === "number" && typeof twin[key] === "number") {
        m[key] = frame[key] as number;
        w[key] = twin[key];
      }
    }
    h.push({ t, m, w });
    if (h.length > WINDOW) h.splice(0, h.length - WINDOW);
    redraw((n) => n + 1);
  }, [frame]);

  const h = hist.current;
  if (h.length < 2) {
    return <div className="text-[8px] text-[#aa8f7f] md:text-[9px]">Collecting the twin comparison...</div>;
  }
  const W = 200, H = 26;
  return (
    <div className="space-y-1">
      {RESIDUAL_CHANNELS.map(({ key, label, unit }) => {
        const pts = h.filter((p) => key in p.m);
        if (pts.length < 2) return null;
        const vals = pts.flatMap((p) => [p.m[key], p.w[key]]);
        let lo = Math.min(...vals), hi = Math.max(...vals);
        if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
        const x = (i: number) => (i / (pts.length - 1)) * W;
        const y = (v: number) => H - ((v - lo) / (hi - lo)) * H;
        const line = (k: "m" | "w") => pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[k][key]).toFixed(1)}`).join("");
        const gap = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.m[key]).toFixed(1)}`).join("")
          + pts.slice().reverse().map((p, j) => `L${x(pts.length - 1 - j).toFixed(1)},${y(p.w[key]).toFixed(1)}`).join("") + "Z";
        const last = pts[pts.length - 1];
        const res = last.m[key] - last.w[key];
        const muted = suspect.includes(key);
        const digits = unit === "bar" ? 2 : unit === "L/h" ? 1 : 0;
        return (
          <div key={key} className={muted ? "opacity-40" : ""}
               title={muted ? "The AI flags this sensor itself as faulty - its gap from the twin is not an engine signal" : undefined}>
            <div className="flex items-baseline justify-between text-[7px] uppercase tracking-[0.1em] text-[#bca18e] md:text-[8px]">
              <span>{label}</span>
              <span className="font-mono normal-case text-[#e0c2ae]">
                {last.m[key].toFixed(digits)} vs twin {last.w[key].toFixed(digits)} {unit}
                <span className={Math.abs(res) > 0 ? "ml-1 text-[#ff9a72]" : "ml-1"}>
                  ({res >= 0 ? "+" : ""}{res.toFixed(digits)})
                </span>
              </span>
            </div>
            <svg viewBox={`0 0 ${W} ${H}`} className="h-[26px] w-full" preserveAspectRatio="none" role="img"
                 aria-label={`${label}: measured against the healthy twin over the last ${pts.length} seconds`}>
              <path d={gap} fill="#ff8050" fillOpacity={0.14} />
              <path d={line("w")} fill="none" stroke="#8a7f6a" strokeWidth={1} strokeDasharray="3 2" vectorEffect="non-scaling-stroke" />
              <path d={line("m")} fill="none" stroke="#ff8050" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
            </svg>
          </div>
        );
      })}
      <div className="flex gap-3 text-[7px] text-[#aa8f7f] md:text-[8px]">
        <span><span className="text-[#ff8050]">&#9472;</span> sensor</span>
        <span><span className="text-[#8a7f6a]">&#9476;</span> healthy twin</span>
        <span>last {h.length} s</span>
      </div>
    </div>
  );
}
