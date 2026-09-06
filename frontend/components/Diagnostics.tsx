import { AlertTriangle, CheckCircle2, FileText } from "lucide-react";
import { simSecondsToRealHours } from "@/lib/timeScale";

export type AiDiagnosisChannel = { fault_type: string; confidence: number };
export type AiResult = {
  status: string;
  fault_detected?: boolean;
  detection_confidence?: number;
  faulty_channels?: string[];
  diagnosis?: Record<string, AiDiagnosisChannel>;
  severity_percent?: Record<string, number>;
  health_percent?: number;
  rul_percent_remaining?: number;
  rul_hours_internal?: number;
  steps_collected?: number;
  steps_needed?: number;
};

export type DiagnosticsProps = {
  ai?: AiResult | null;
  simSeconds?: number;   // live elapsed simulated flight time (rawTelemetry.time)
};

const AI_CHANNELS = [
  { key: "egt", label: "EGT" },
  { key: "cht", label: "CHT" },
  { key: "oil_pressure", label: "Oil Pressure" },
  { key: "oil_temp", label: "Oil Temp" },
  { key: "vibx", label: "Vib X" },
  { key: "viby", label: "Vib Y" },
  { key: "vibz", label: "Vib Z" },
  { key: "rpm", label: "RPM" },
];

// Pure AI output now - the throttle-derived engine metrics (RPM/temp/fuel/vibration)
// were removed since Sensr's "All Sensors" list already shows the real
// telemetry, and this panel's actual job is the AI's diagnosis, not duplicating
// raw sensor readouts.
export default function Diagnostics({ ai = null, simSeconds }: DiagnosticsProps) {
  return (
    <aside className="panel-shell flex h-full min-h-0 flex-col overflow-hidden p-2.5 md:p-3.5">
      <h2 className="panel-heading flex items-center justify-between">
        Diagnostics <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#ff722f] shadow-[0_0_7px_#ff5b1d]" />
      </h2>

      {/* Live, continuously moving - real-world equivalent flight time, ticking
          up as telemetry streams in. Same compression scale as RUL (2000h TBO /
          ~5.56h max simulated duration), just shown as a live counter here
          instead of a static value only visible after stopping. */}
      {simSeconds !== undefined && (
        <div className="mt-2 flex items-center justify-between border border-[#352722] bg-[#0d0e0d] px-2 py-1.5 text-[8px] uppercase tracking-[0.1em] text-[#bca18e] md:text-[9px]">
          <span>Flight Time (real-world eq.)</span>
          <span className="font-mono text-[#efe0d5]">{simSecondsToRealHours(simSeconds).toFixed(2)}h</span>
        </div>
      )}

      <div className="mt-3 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
        {!ai && (
          <div className="border border-[#352722] bg-[#0d0e0d] p-2 text-[8px] text-[#aa8f7f] md:text-[9px]">
            Waiting for AI service...
          </div>
        )}

        {ai?.status === "warming_up" && ai.steps_collected !== undefined && ai.steps_needed !== undefined && (
          <div className="border border-[#4c3025] bg-[#0d0e0d] p-2">
            <div className="text-[8px] text-[#d9c0ae] md:text-[9px]">
              Buffering: {ai.steps_collected} / {ai.steps_needed}s
            </div>
            <div className="mt-1 h-1 overflow-hidden rounded bg-[#211510]">
              <div className="h-full bg-[#ff8050]" style={{ width: `${(100 * ai.steps_collected) / ai.steps_needed}%` }} />
            </div>
          </div>
        )}

        {ai?.status === "ai_service_unavailable" && (
          <div className="border border-[#84432c] bg-[#21130f] p-2 text-[8px] text-[#ff9a72] md:text-[9px]">
            AI service unavailable
          </div>
        )}

        {ai?.status === "ok" && (
          <>
            <article className={`border p-2 ${ai.fault_detected ? "border-[#84432c] bg-[#21130f]" : "border-[#2f4a30] bg-[#0d150e]"}`}>
              <div className="flex items-center justify-between text-[7px] uppercase tracking-[0.11em] text-[#bca18e] md:text-[9px]">
                <span>Fault Status</span>
                {ai.fault_detected ? (
                  <AlertTriangle size={11} className="text-[#ff844d]" />
                ) : (
                  <CheckCircle2 size={11} className="text-[#7fc87f]" />
                )}
              </div>
              <div className="mt-1 flex items-end justify-between gap-1">
                <strong className={`text-[11px] font-normal md:text-[14px] ${ai.fault_detected ? "text-[#ff9a72]" : "text-[#a8e0a8]"}`}>
                  {ai.fault_detected ? "DETECTED" : "NOMINAL"}
                </strong>
                <span className="text-[7px] uppercase text-[#aa8f7f] md:text-[8px]">
                  {Math.round((ai.detection_confidence ?? 0) * 100)}% conf
                </span>
              </div>
            </article>

            <div className="grid grid-cols-2 gap-2">
              <article className="border border-[#352722] bg-[#0d0e0d] p-2">
                <div className="text-[7px] uppercase tracking-[0.11em] text-[#bca18e] md:text-[9px]">Health</div>
                <strong className="text-[11px] font-normal text-[#efe0d5] md:text-[14px]">
                  {(ai.health_percent ?? 0).toFixed(1)}%
                </strong>
              </article>
              <article className="border border-[#352722] bg-[#0d0e0d] p-2">
                <div className="text-[7px] uppercase tracking-[0.11em] text-[#bca18e] md:text-[9px]">RUL</div>
                <strong className="text-[11px] font-normal text-[#efe0d5] md:text-[14px]">
                  {(ai.rul_percent_remaining ?? 0).toFixed(1)}%
                </strong>
              </article>
            </div>

            <div className="space-y-1.5">
              {AI_CHANNELS.map((ch) => {
                const faultType = ai.diagnosis?.[ch.key]?.fault_type ?? "none";
                const pct = Math.round(ai.severity_percent?.[ch.key] ?? 0);
                const active = faultType !== "none";
                return (
                  <article key={ch.key} className={`border p-1.5 ${active ? "border-[#84432c] bg-[#21130f]" : "border-[#352722] bg-[#0d0e0d]"}`}>
                    <div className="flex items-center justify-between text-[7px] uppercase tracking-[0.1em] text-[#bca18e] md:text-[8px]">
                      <span>{ch.label}</span>
                      <span>{pct}%</span>
                    </div>
                    <div className={`mt-0.5 text-[9px] font-normal md:text-[10px] ${active ? "text-[#ff9a72]" : "text-[#7fc87f]"}`}>
                      {active ? faultType : "OK"}
                    </div>
                  </article>
                );
              })}
            </div>
          </>
        )}
      </div>

      <div className="mt-2 shrink-0 border border-[#382b25] bg-[#0c0d0c] p-2 text-center text-[7px] uppercase tracking-[0.12em] text-[#aa8f7f] md:text-[9px]">
        <FileText className="mr-1 inline-block h-3 w-3 text-[#e68450]" /> System logs
      </div>
    </aside>
  );
}
