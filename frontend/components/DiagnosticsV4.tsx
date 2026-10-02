"use client";

import { useState, type ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Eye, EyeOff } from "lucide-react";
import TwinResidualChart from "@/components/TwinResidualChart";
import {
  ALTITUDE_MASKED, faultLabel, familyLabel, sensorLabel,
  type AiResultV4Fields, type V4Engine, type V4Truth,
} from "@/lib/v4";

/**
 * The physics-v4 AI panel: the five questions the models answer, in the order an
 * operator asks them, then the digital-twin comparison, then the ground truth.
 *
 *   1 Is anything wrong?     detection
 *   2 Which part?            component faults this engine can have
 *   3 Engine or sensor?      condition of each of the 12 instruments
 *   4 How worn?              wear condition - kept apart from the operating margin
 *   5 Hours left?            RUL against the calendar countdown
 *
 * GROUND TRUTH is shown by default (decided for the demo): what was actually injected
 * sits beside what the AI worked out without seeing it. It never reaches the AI.
 */
type AiV4 = AiResultV4Fields & {
  status: string;
  settling?: boolean;
  settle_seconds_left?: number;
  settle_window_s?: number;
  fault_detected?: boolean;
  detection_confidence?: number;
  rul_hours?: number | null;
  rul_ready?: boolean;
  rul_mae_hours?: number | null;
  tbo_hours?: number;
};

const card = "border border-[#352722] bg-[#0d0e0d] p-2";
const head = "text-[7px] uppercase tracking-[0.11em] text-[#bca18e] md:text-[9px]";
const pct = (v: number | null | undefined) => (v == null ? "--" : `${Math.round(v * 100)}%`);

function Bar({ value, tone }: { value: number; tone: string }) {
  return (
    <div className="mt-0.5 h-1 w-full overflow-hidden bg-[#241a16]">
      <div className={`h-full ${tone} transition-[width] duration-500`} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
    </div>
  );
}

export default function DiagnosticsV4({
  ai, truth, engine, frame, advisoryCard,
}: {
  ai: AiV4 | null;
  truth?: V4Truth | null;
  engine?: V4Engine | null;
  frame?: Record<string, unknown> | null;
  advisoryCard?: ReactNode;
}) {
  const [showTruth, setShowTruth] = useState(true);
  const ok = ai?.status === "ok";
  const settling = ok && !!ai?.settling;
  const faults = Object.entries(ai?.fault_modes ?? {});
  const suspect = ok && !settling ? ai?.faulty_sensors ?? [] : [];
  // v5: when a family is called but none of its faults clears its own cut-off, the
  // honest answer is the family (e.g. "Oil system") rather than a guessed part.
  const v5 = ai?.model_version === "v5";
  const named = new Set((ai?.faults_present ?? []).map((f) => ai?.fault_modes?.[f]?.family));
  const familyOnly = v5 && !settling ? (ai?.families_present ?? []).filter((f) => !named.has(f)) : [];
  const sevWord = ai?.severity_kind === "effective" ? "eff. sev" : "sev";

  return (
    <>
      {engine?.placeholder_models && (
        <div className="border border-[#4c3025] bg-[#14100d] p-2 text-[8px] text-[#e8c9a0] md:text-[9px]"
             title="This engine's own models are still training; the Rotax 914's models answer until they are exported.">
          Placeholder AI: running the 914&apos;s models until this engine&apos;s own finish training.
        </div>
      )}

      {ok && !settling && ai?.context_settling && (
        <div className="border border-[#352722] bg-[#0d0e0d] p-2 text-[8px] text-[#bca18e] md:text-[9px]"
             title="The models also read 10- and 60-minute averages of every residual. They fill from the start of the flight; slow sensor drift is judged more surely once they have.">
          Long-term trends still building (first hour of flight).
        </div>
      )}

      {settling && (
        <div role="status" className="border border-[#4c3025] bg-[#0d0e0d] p-2 text-[8px] text-[#d9c0ae] md:text-[9px]">
          AI settling after takeoff - alerts are held until the last 128 s of flight are above 28 m/s
          {ai?.settle_seconds_left != null ? ` (${Math.ceil(ai.settle_seconds_left)} s)` : ""}.
        </div>
      )}

      {ok && (
        <>
          {/* 1 - Is anything wrong? */}
          <article className={`border p-2 ${ai?.fault_detected ? "border-[#84432c] bg-[#21130f]" : "border-[#2f4a30] bg-[#0d150e]"}`}>
            <div className={`${head} flex items-center justify-between`}>
              <span>1 &middot; Anything wrong?</span>
              {ai?.fault_detected ? <AlertTriangle size={11} className="text-[#ff844d]" /> : <CheckCircle2 size={11} className="text-[#7fc87f]" />}
            </div>
            <div className="mt-1 flex items-end justify-between">
              <strong className={`text-[11px] font-normal md:text-[14px] ${ai?.fault_detected ? "text-[#ff9a72]" : "text-[#a8e0a8]"}`}>
                {ai?.fault_detected ? "FAULT" : "NOMINAL"}
              </strong>
              <span className="text-[7px] uppercase text-[#aa8f7f] md:text-[8px]">{pct(ai?.detection_confidence)} likely</span>
            </div>
            <Bar value={ai?.detection_confidence ?? 0} tone={ai?.fault_detected ? "bg-[#ff8050]" : "bg-[#7fc87f]"} />
          </article>

          {/* Health: the wear condition as a percentage, the number the screen-edge
              vignette (HealthVignette) follows. Same bands: caution < 80, warning < 50. */}
          {(() => {
            const h = ai?.wear_condition;
            const tone = h == null ? "text-[#aa8f7f]" : h < 0.5 ? "text-[#ff6a4d]" : h < 0.8 ? "text-[#ffd27a]" : "text-[#a8e0a8]";
            const bar = h == null ? "bg-[#4c3025]" : h < 0.5 ? "bg-[#ff5b3c]" : h < 0.8 ? "bg-[#e0a040]" : "bg-[#7fc87f]";
            return (
              <article className={`border p-2 ${h != null && h < 0.5 ? "border-[#84432c] bg-[#21130f]" : "border-[#352722] bg-[#0d0e0d]"}`}
                       title="Engine health from the AI's health head: 100% = as new, 0% = end of life. Falls with wear and with any fault.">
                <div className={`${head} flex items-center justify-between`}>
                  <span>Health</span>
                  <span className="text-[#aa8f7f]">{h == null ? "" : h < 0.5 ? "WARNING" : h < 0.8 ? "CAUTION" : "GOOD"}</span>
                </div>
                <strong className={`text-[14px] font-normal md:text-[18px] ${tone}`}>
                  {h == null ? "--" : `${(h * 100).toFixed(1)}%`}
                </strong>
                <Bar value={h ?? 0} tone={bar} />
              </article>
            );
          })()}

          {/* 2 - Which part? */}
          <article className={card}>
            <div className={head}>2 &middot; Which part?</div>
            {familyOnly.length > 0 && (
              <div className="mt-1 border border-[#84642c] bg-[#1c1710] px-1 py-0.5 text-[8px] text-[#ffd27a] md:text-[9px]"
                   title="The fault family is clear, but no single part in it clears its own threshold yet.">
                Suspected: {familyOnly.map(familyLabel).join(", ")} - part not yet clear
              </div>
            )}
            <div className="mt-1 space-y-0.5">
              {faults.map(([name, f]) => (
                <div key={name}
                     title={`${Math.round(f.probability * 100)}% likely (called at ${Math.round(f.threshold * 100)}%)` +
                            (ALTITUDE_MASKED.has(name) ? ". Only observable above the turbo's critical altitude (15,000 ft)." : "")}>
                  <div className="flex items-center justify-between gap-1 text-[8px] md:text-[9px]">
                    <span className={f.present ? "text-[#ff9a72]" : "text-[#aa8f7f]"}>
                      {faultLabel(name)}{ALTITUDE_MASKED.has(name) ? " ↑" : ""}
                    </span>
                    <span className={`font-mono ${f.present ? "text-[#ff9a72]" : "text-[#7fc87f]"}`}>
                      {f.present ? `${sevWord} ${Math.round(f.severity * 100)}%` : "OK"}
                    </span>
                  </div>
                  <div className="h-[2px] w-full bg-[#241a16]">
                    <div className={`h-full ${f.present ? "bg-[#ff8050]" : "bg-[#4c3025]"}`}
                         style={{ width: `${Math.round(f.probability * 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </article>

          {/* 3 - Engine or sensor? */}
          <article className={card}>
            <div className={head}>3 &middot; Engine or sensor?</div>
            <div className="mt-1 grid grid-cols-3 gap-1">
              {Object.entries(ai?.sensors ?? {}).map(([ch, s]) => {
                const bad = s.condition !== "none";
                return (
                  <div key={ch} title={`${sensorLabel(ch)}: ${bad ? s.condition : "fine"} (${Math.round(s.confidence * 100)}%)`}
                       className={`border px-1 py-0.5 ${bad ? "border-[#84642c] bg-[#1c1710]" : "border-[#2a201b]"}`}>
                    <div className="truncate text-[6px] uppercase tracking-[0.08em] text-[#bca18e] md:text-[7px]">{sensorLabel(ch)}</div>
                    <div className={`text-[8px] md:text-[9px] ${bad ? "text-[#ffd27a]" : "text-[#7fc87f]"}`}>{bad ? s.condition : "fine"}</div>
                  </div>
                );
              })}
            </div>
          </article>

          {/* 4 - How worn? (wear and operating margin are different questions) */}
          <article className={card}>
            <div className={head}>4 &middot; How worn?</div>
            <div className="mt-1 flex items-baseline justify-between text-[8px] md:text-[9px]">
              <span className="text-[#d9c4b4]" title="The health head: how worn the engine IS, 100% = as new. Falls over life, faster with a fault.">Wear condition</span>
              <span className="font-mono text-[#efe0d5]">{ai?.wear_condition == null ? "--" : pct(ai.wear_condition)}</span>
            </div>
            <Bar value={ai?.wear_condition ?? 0} tone="bg-[#7fc87f]" />
            <div className="mt-1.5 flex items-baseline justify-between text-[8px] md:text-[9px]">
              <span className="text-[#d9c4b4]" title="How far the MEASURED CHT, EGT, oil temperature and oil pressure are from their certified limits right now. Depends on what the engine is doing, not how worn it is.">Operating margin</span>
              <span className="font-mono text-[#efe0d5]">{pct(ai?.margin_min)}</span>
            </div>
            <Bar value={ai?.margin_min ?? 0} tone={(ai?.margin_min ?? 1) < 0.15 ? "bg-[#ff8050]" : "bg-[#8ab4d8]"} />
          </article>

          {/* 5 - Hours left? */}
          <article className={card}>
            <div className={`${head} flex items-center justify-between`}>
              <span>5 &middot; Hours left?</span>
              {ai?.wear_limited && ai.rul_ready !== false && (
                <span className="border border-[#84432c] px-1 text-[6px] text-[#ff9a72] md:text-[7px]"
                      title="The model puts this engine's end of life before its scheduled overhaul, by more than its own measured error.">
                  WEAR-LIMITED
                </span>
              )}
            </div>
            {ai?.rul_ready === false || ai?.rul_hours == null ? (
              <div className="mt-1 text-[8px] text-[#aa8f7f] md:text-[9px]">Stabilising...</div>
            ) : (
              <>
                <strong className="text-[11px] font-normal text-[#efe0d5] md:text-[14px]">
                  {Math.round(ai.rul_hours).toLocaleString()} h
                  {ai.rul_mae_hours != null && <span className="ml-1 text-[8px] text-[#aa8f7f] md:text-[9px]">&plusmn;{Math.round(ai.rul_mae_hours)}</span>}
                </strong>
                {ai.rul_calendar_hours != null && (
                  <div className="text-[7px] uppercase tracking-[0.1em] text-[#aa8f7f] md:text-[8px]">
                    calendar {Math.round(ai.rul_calendar_hours).toLocaleString()} h to overhaul
                    {ai.tbo_hours ? ` (TBO ${ai.tbo_hours.toLocaleString()} h)` : ""}
                  </div>
                )}
              </>
            )}
          </article>

          {advisoryCard}
        </>
      )}

      {/* The twin itself - model-free, so it runs while the AI warms up too. */}
      {frame?.twin != null && (
        <article className={card}>
          <div className={head}>Engine vs healthy twin</div>
          <div className="mt-1">
            <TwinResidualChart frame={frame} suspect={suspect} />
          </div>
        </article>
      )}

      {truth && (
        <article className="border border-[#2a3a4c] bg-[#0c1116] p-2">
          <div className={`${head} flex items-center justify-between`}>
            <span className="text-[#9cc3e6]">Ground truth (injected)</span>
            <button type="button" onClick={() => setShowTruth((v) => !v)} aria-pressed={showTruth}
                    className="flex items-center gap-1 text-[#9cc3e6] hover:text-white"
                    title="What was actually injected into the engine and its sensors. The AI never sees this.">
              {showTruth ? <EyeOff size={10} /> : <Eye size={10} />}
              {showTruth ? "Hide" : "Show"}
            </button>
          </div>
          {showTruth && (
            <div className="mt-1 space-y-0.5 text-[8px] text-[#cfe0ef] md:text-[9px]">
              <div>
                Engine: {(truth.faults_present ?? []).length
                  ? (truth.faults_present ?? []).map((f) =>
                      `${faultLabel(f)} ${Math.round(100 * (truth.fault_severity?.[f] ?? 0))}%`).join(", ")
                  : "no fault"}
              </div>
              <div>
                Sensors: {Object.keys(truth.sensor_faults ?? {}).length
                  ? Object.entries(truth.sensor_faults ?? {}).map(([c, k]) => `${sensorLabel(c)} ${k}`).join(", ")
                  : "all fine"}
              </div>
              {truth.severity_kind === "effective" && (
                <div className="text-[#7f9db8]">severities are effective: the share of each fault&apos;s full effect this engine shows</div>
              )}
              <div className="font-mono text-[#9cc3e6]">
                wear {pct(truth.wear_condition)} &middot; RUL {truth.rul_hours != null ? `${Math.round(truth.rul_hours)} h` : "--"}
                {truth.wear_limited ? " (wear-limited)" : ""}
              </div>
            </div>
          )}
        </article>
      )}
    </>
  );
}
