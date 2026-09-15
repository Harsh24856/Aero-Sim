import { AlertTriangle, CheckCircle2, FileText, Wrench } from "lucide-react";
import { flightHours, formatSimClock } from "@/lib/timeScale";

export type AiDiagnosisChannel = { fault_type: string; confidence: number; reliable?: boolean };
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
  // physics v3 (backend/aiv3.py): RUL in real engine hours, plus engine failure modes
  model_version?: string;
  rul_hours?: number;
  rul_mae_hours?: number | null;   // test-split mean absolute error, the RUL uncertainty band
  tbo_hours?: number;
  failure_modes?: Record<string, { severity_percent: number; present: boolean; threshold?: number }>;
  steps_collected?: number;
  steps_needed?: number;
  // main.py: fault alerts held while the AI window still contains ground-roll samples
  settling?: boolean;
};

// Mirrors backend/advisory.py build_advisory(). Deterministic, computed
// backend-side every broadcast - no network call of its own.
export type AdvisoryItem = {
  code: string;
  channel: string | null;
  subsystem: string;
  severity: AdvisorySeverity;
  message: string;
  action: string;
};
export type AdvisorySeverity = "nominal" | "advisory" | "caution" | "warning";
export type Advisory = {
  severity: AdvisorySeverity;
  headline: string;
  insufficient_data?: boolean;
  items?: AdvisoryItem[];
};

// Mirrors backend/residual.py ResidualMonitor.update(). Model-free physics
// residuals - present only on physics v3, and available while the AI warms up.
export type ResidualChannel = {
  label: string; unit: string; measured: number; expected: number;
  residual: number; z: number; status: string; signature: string | null;
};
export type Residuals = {
  enabled: boolean;
  degradation_index?: number;
  deviations?: string[];
  saturated?: string[];
  channels?: Record<string, ResidualChannel>;
  zeroing?: boolean;                    // sender-offset calibration in progress
  samples_left?: number;
  offsets?: Record<string, number>;     // applied sender offsets from zeroing
};

// Safety net (backend/main.py sim_status and the page's WebSocket watchdog).
export type LinkState = "connecting" | "open" | "stale" | "lost";
export type SimStatus = {
  health: "idle" | "running" | "recovered" | "halted";
  recoveries: number;
  last_error: string | null;
  ai: "ok" | "warming_up" | "unavailable" | "unsupported" | "pending";
  ai_breaker: "closed" | "open" | "half_open";
  loop_lag_ms: number;
};

export type DiagnosticsProps = {
  ai?: AiResult | null;
  advisory?: Advisory | null;
  residuals?: Residuals | null;
  // physics_version from the live telemetry. Decides the flight-time scale even
  // when the AI service is down and there is no ai.model_version to read.
  physicsVersion?: string;
  link?: LinkState;            // only passed while a flight is running
  engineHours?: number | null; // v3: engine hour meter (wear x TBO)
  simStatus?: SimStatus | null;
  simSeconds?: number;   // live elapsed simulated flight time (rawTelemetry.time)
  dataSource?: string;   // "can" when sensors arrive from the aircraft over CAN (main.py /measured)
  onZeroSensors?: () => void;   // POST /residuals/zero - only offered while a flight is running
};

// Cockpit palette per severity. Kept local to this file on purpose: the
// telemetry pages use design tokens (bg-surface/outline-variant) while this
// panel uses hardcoded cockpit hex, and mixing the two looks wrong.
const ADVISORY_STYLE: Record<AdvisorySeverity, { box: string; text: string; label: string }> = {
  nominal:  { box: "border-[#2f4a30] bg-[#0d150e]", text: "text-[#a8e0a8]", label: "Nominal" },
  advisory: { box: "border-[#4c3025] bg-[#14100d]", text: "text-[#e8c9a0]", label: "Advisory" },
  caution:  { box: "border-[#84642c] bg-[#1c1710]", text: "text-[#ffd27a]", label: "Caution" },
  warning:  { box: "border-[#84432c] bg-[#21130f]", text: "text-[#ff9a72]", label: "Warning" },
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
export default function Diagnostics({ ai = null, advisory = null, residuals = null, physicsVersion, link, engineHours, simStatus = null, simSeconds, dataSource, onZeroSensors }: DiagnosticsProps) {
  const v3 = ai?.model_version === "v3" || physicsVersion === "v3";
  const labelOf = (c: string) => residuals?.channels?.[c]?.label ?? c;
  return (
    <aside className="panel-shell flex h-full min-h-0 flex-col overflow-hidden p-2.5 md:p-3.5">
      <h2 className="panel-heading flex items-center justify-between">
        Diagnostics
        <span className="flex items-center gap-1.5">
          {dataSource === "can" && (
            <span
              title="Sensor readings are arriving from the aircraft over the CAN bus; the twin's physics supplies the expected values."
              className="border border-[#2f4a30] px-1 text-[7px] tracking-[0.1em] text-[#a8e0a8]"
            >
              CAN LIVE
            </span>
          )}
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#ff722f] shadow-[0_0_7px_#ff5b1d]" />
        </span>
      </h2>

      {/* Live, continuously moving - real-world equivalent flight time, ticking
          up as telemetry streams in. Same compression scale as RUL (2000h TBO /
          ~5.56h max simulated duration), just shown as a live counter here
          instead of a static value only visible after stopping. */}
      {simSeconds !== undefined && (
        <div className="mt-2 flex items-center justify-between border border-[#352722] bg-[#0d0e0d] px-2 py-1.5 text-[8px] uppercase tracking-[0.1em] text-[#bca18e] md:text-[9px]">
          {v3 ? (
            <>
              {/* v3: engine hours are the clock RUL counts down on; the simulated
                  clock is shown beside them for reference. */}
              <span title="Engine hour meter: hours on this engine (wear x TBO), the same clock RUL counts down on. Wear is time-compressed in the simulation.">Engine Hours</span>
              <span className="text-right font-mono text-[#efe0d5]">
                {engineHours != null ? `${engineHours.toFixed(1)}h` : "--"}
                <span className="ml-1.5 text-[#aa8f7f]">sim {formatSimClock(simSeconds)}</span>
              </span>
            </>
          ) : (
            <>
              <span>Flight Time (real-world eq.)</span>
              <span className="font-mono text-[#efe0d5]">{flightHours(simSeconds, "v2").toFixed(2)}h</span>
            </>
          )}
        </div>
      )}

      <div className="mt-3 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
        {(link === "lost" || link === "stale") && (
          <div role="status" className="border border-[#84432c] bg-[#21130f] p-2 text-[8px] text-[#ff9a72] md:text-[9px]">
            {link === "lost"
              ? "Connection to the simulator lost - reconnecting automatically..."
              : "No telemetry for a few seconds - waiting for the simulator..."}
          </div>
        )}

        {simStatus?.health === "halted" && (
          <div role="alert" className="border border-[#84432c] bg-[#21130f] p-2 text-[8px] text-[#ff9a72] md:text-[9px]">
            Simulation halted safely after repeated physics faults. Press Start to continue from the last good state.
            {simStatus.last_error && <div className="mt-1 text-[#aa8f7f]">{simStatus.last_error}</div>}
          </div>
        )}

        {simStatus?.health === "recovered" && (
          <div role="status" className="border border-[#84642c] bg-[#1c1710] p-2 text-[8px] text-[#ffd27a] md:text-[9px]">
            Physics fault recovered automatically - the flight continued from the last good state.
          </div>
        )}

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

        {ai?.status === "ok" && ai.settling && (
          <div role="status" className="border border-[#4c3025] bg-[#0d0e0d] p-2 text-[8px] text-[#d9c0ae] md:text-[9px]">
            AI settling after takeoff - fault alerts are held until the last 128 s of flight are above 32 m/s.
          </div>
        )}

        {ai?.status === "ai_unsupported_engine" && (
          <div className="border border-[#4c3025] bg-[#14100d] p-2 text-[8px] text-[#e8c9a0] md:text-[9px]">
            No AI model for this engine on physics v3 yet. Physics, residuals and the advisory keep working.
          </div>
        )}

        {ai?.status === "ai_service_unavailable" && (
          <div className="border border-[#84432c] bg-[#21130f] p-2 text-[8px] text-[#ff9a72] md:text-[9px]">
            AI service unavailable{simStatus?.ai_breaker && simStatus.ai_breaker !== "closed" ? " - retrying automatically" : ""}. The simulation keeps running.
          </div>
        )}

        {residuals?.enabled && (
          <article className="border border-[#352722] bg-[#0d0e0d] p-2">
            <div className="flex items-center justify-between text-[7px] uppercase tracking-[0.11em] text-[#bca18e] md:text-[9px]">
              <span>Physics Residuals</span>
              <span className="font-mono text-[#efe0d5]">wear idx {(residuals.degradation_index ?? 0).toFixed(2)}</span>
            </div>
            {residuals.zeroing ? (
              <div role="status" className="mt-1 text-[8px] text-[#ffd27a] md:text-[9px]">
                Zeroing sensors - hold steady{residuals.samples_left != null ? ` (${residuals.samples_left} s)` : ""}
              </div>
            ) : (
              <div className={`mt-1 text-[8px] md:text-[9px] ${(residuals.deviations?.length ?? 0) > 0 ? "text-[#ff9a72]" : "text-[#7fc87f]"}`}>
                {(residuals.deviations?.length ?? 0) > 0
                  ? `Disagrees with physics: ${(residuals.deviations ?? []).map(labelOf).join(", ")}`
                  : "All sensors agree with physics"}
                {(residuals.saturated?.length ?? 0) > 0 && ` · at range limit: ${(residuals.saturated ?? []).map(labelOf).join(", ")}`}
              </div>
            )}
            <div className="mt-1 flex items-center justify-between gap-2 text-[7px] text-[#aa8f7f] md:text-[8px]">
              <span title="Sender offsets measured on a known-healthy run and subtracted from every reading">
                {residuals.offsets && Object.keys(residuals.offsets).length > 0
                  ? `Offsets: ${Object.entries(residuals.offsets).map(([c, v]) => `${labelOf(c)} ${v > 0 ? "+" : ""}${v.toFixed(1)}`).join(", ")}`
                  : "Sensors not zeroed"}
              </span>
              {onZeroSensors && !residuals.zeroing && (
                <button
                  type="button"
                  onClick={onZeroSensors}
                  title="Only on a known-healthy engine held at a steady operating point: measures each sender's offset over 60 s"
                  className="shrink-0 border border-[#4c3025] px-1.5 py-0.5 uppercase tracking-[0.1em] text-[#d9c0ae] hover:border-[#ff8050]"
                >
                  Zero sensors
                </button>
              )}
            </div>
          </article>
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
                {v3 && ai.rul_hours != null && (
                  <div
                    className="mt-0.5 text-[7px] uppercase tracking-[0.1em] text-[#aa8f7f] md:text-[8px]"
                    title={ai.rul_mae_hours != null ? `Uncertainty band: the model's measured error on held-out data for an engine at this life stage (±${Math.round(ai.rul_mae_hours)} h). Nearly-new engines are under-predicted - by 37-109 h depending on the engine - so the band widens there.` : undefined}
                  >
                    {Math.round(ai.rul_hours).toLocaleString()}
                    {ai.rul_mae_hours != null ? ` ±${Math.round(ai.rul_mae_hours)}` : ""} engine h{ai.tbo_hours ? ` of ${ai.tbo_hours.toLocaleString()} TBO` : ""}
                  </div>
                )}
              </article>
            </div>

            {v3 && ai.failure_modes && (
              <article className="border border-[#352722] bg-[#0d0e0d] p-2">
                <div className="text-[7px] uppercase tracking-[0.11em] text-[#bca18e] md:text-[9px]">Engine Failure Modes</div>
                <div className="mt-1 grid grid-cols-2 gap-x-2 gap-y-0.5">
                  {Object.entries(ai.failure_modes).map(([mode, fm]) => (
                    <div key={mode} className="flex items-center justify-between gap-1 text-[8px] md:text-[9px]">
                      <span className={`capitalize ${fm.present ? "text-[#ff9a72]" : "text-[#aa8f7f]"}`}>{mode.replace(/_/g, " ")}</span>
                      <span className={fm.present ? "text-[#ff9a72]" : "text-[#7fc87f]"}>
                        {fm.present ? `${Math.round(fm.severity_percent)}%` : "OK"}
                      </span>
                    </div>
                  ))}
                </div>
              </article>
            )}

            {advisory && !advisory.insufficient_data && (
              <article className={`border p-2 ${ADVISORY_STYLE[advisory.severity].box}`}>
                <div className="flex items-center justify-between text-[7px] uppercase tracking-[0.11em] text-[#bca18e] md:text-[9px]">
                  <span className="flex items-center gap-1">
                    <Wrench className="h-3 w-3" /> Maintenance Advisory
                  </span>
                  <span className={ADVISORY_STYLE[advisory.severity].text}>
                    {ADVISORY_STYLE[advisory.severity].label}
                  </span>
                </div>
                <div className={`mt-1 text-[9px] font-normal md:text-[11px] ${ADVISORY_STYLE[advisory.severity].text}`}>
                  {advisory.headline}
                </div>
                {(advisory.items ?? []).slice(0, 3).map((item) => (
                  <div key={item.code} className="mt-1.5 border-t border-[#2a201b] pt-1.5">
                    <div className="text-[8px] text-[#d9c4b4] md:text-[9px]">{item.message}</div>
                    <div className="mt-0.5 text-[8px] text-[#aa8f7f] md:text-[9px]">
                      <span className="text-[#e68450]">&#8594;</span> {item.action}
                    </div>
                  </div>
                ))}
                {(advisory.items ?? []).length > 3 && (
                  <div className="mt-1 text-[7px] uppercase tracking-[0.1em] text-[#aa8f7f] md:text-[8px]">
                    +{(advisory.items ?? []).length - 3} more
                  </div>
                )}
              </article>
            )}

            <div className="space-y-1.5">
              {AI_CHANNELS.map((ch) => {
                const faultType = ai.diagnosis?.[ch.key]?.fault_type ?? "none";
                const pct = Math.round(ai.severity_percent?.[ch.key] ?? 0);
                // reliable === false: this engine's model was measured no better than chance
                // on this channel, so its call is not shown as a fault.
                const uncalibrated = ai.diagnosis?.[ch.key]?.reliable === false;
                // v3: a call below the 70% confidence floor (the same one the health cap and the
                // advisory use) is not a fault. v2 display is left exactly as it was.
                const confident = !v3 || (ai.diagnosis?.[ch.key]?.confidence ?? 0) >= 0.7;
                const active = faultType !== "none" && !uncalibrated && confident;
                return (
                  <article key={ch.key} className={`border p-1.5 ${active ? "border-[#84432c] bg-[#21130f]" : "border-[#352722] bg-[#0d0e0d]"}`}>
                    <div className="flex items-center justify-between text-[7px] uppercase tracking-[0.1em] text-[#bca18e] md:text-[8px]">
                      <span>{ch.label}</span>
                      <span>{pct}%</span>
                    </div>
                    <div className={`mt-0.5 text-[9px] font-normal md:text-[10px] ${active ? "text-[#ff9a72]" : "text-[#7fc87f]"}`}>
                      {uncalibrated ? <span className="text-[#aa8f7f]" title="This engine's model is not reliable on this channel">Uncalibrated</span> : active ? faultType : "OK"}
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
