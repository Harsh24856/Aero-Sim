"use client";

import { useEffect, useRef, useState } from "react";
import { supabase } from "@/lib/supabase";
import { formatLifeScale } from "@/lib/timeScale";
import { faultLabel, sensorLabel, SENSOR_LABELS, type V4Truth } from "@/lib/v4";

/**
 * Physics v4: which engine flies, and faults on demand.
 *
 * Before a flight - a demo preset (a fresh engine placed at the point in its life
 * where the fault is visible, backend/scenarios_v4.py) or one of the user's saved
 * engines, which continues from its own hour meter with its own faults.
 * During a flight - one click on a fault chip injects it into the engine being flown
 * (POST /inject), a second click takes it back out (POST /inject/clear). Sensors: pick
 * the channel, one click on the fault kind breaks it; the x on its chip repairs it.
 *
 * One fixed-height row before and during the flight, so starting doesn't shift the page.
 * Renders nothing unless the backend runs physics v4 (GET /scenarios answers ok).
 */
type Preset = { name: string; label: string; start_engine_hours: number };
type SavedEngine = { id: number; name: string; engine_hours: number; tbo_hours: number; status: string };

const SENSOR_KINDS = ["bias", "drift", "stuck", "spike", "noise", "dropout"];
const sel = "shrink-0 rounded border border-[#4c3025] bg-[#1a110d] px-1.5 py-1 text-[9px] text-[#e8c9a0] md:text-[10px]";
const chipBase = "shrink-0 rounded border px-2 py-1 text-[8px] transition-colors disabled:opacity-50 md:text-[9px]";
const chipOff = "border-[#4c3025] bg-[#1a110d] text-[#e8c9a0] hover:border-[#ff8050] hover:text-[#ff8050]";
const chipOn = "border-[#ff8050] bg-[#3a1a0e] text-[#ffb08a]";

export default function ScenarioPanel({
  api, engineModel, started, applicableFaults = [], onEngineChoice, truth, onInputs, initialScenario,
}: {
  api: string;
  engineModel: string;
  started: boolean;
  applicableFaults?: string[];
  onEngineChoice: (engineId: number | null) => void;   // a saved engine for /start, or null = preset
  truth?: V4Truth | null;                              // what is active now
  onInputs?: (inputs: Record<string, number | boolean> | null) => void;   // the chosen engine's inputs, before a flight
  initialScenario?: string;                            // a mission's engine, selected once on arrival
}) {
  const [presets, setPresets] = useState<Preset[] | null>(null);
  const [lifeScale, setLifeScale] = useState<number | null>(null);
  const [choice, setChoice] = useState<string>("");
  const [saved, setSaved] = useState<SavedEngine[]>([]);
  const [faultList, setFaultList] = useState<string[]>([]);
  const [severity, setSeverity] = useState(0.4);
  const [channel, setChannel] = useState("cht");
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const appliedInitial = useRef(false);

  // Presets for this engine, and the user's own saved engines of the same model.
  useEffect(() => {
    if (started) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${api}/scenarios`).then((x) => x.json());
        if (cancelled) return;
        if (r.status !== "ok" || r.engine_model !== engineModel) { setPresets(null); return; }
        setPresets(r.scenarios);
        setLifeScale(r.life_scale ?? null);
        setFaultList(r.applicable_faults ?? []);
        let inputs = r.inputs ?? null;
        let selected = r.selected;
        // A mission names the engine it flies; select it once, then the pilot may change it.
        if (initialScenario && !appliedInitial.current
            && r.scenarios.some((p: Preset) => p.name === initialScenario)) {
          appliedInitial.current = true;
          if (selected !== initialScenario) {
            const sel = await fetch(`${api}/scenario`, {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name: initialScenario }),
            }).then((x) => x.json()).catch(() => null);
            if (cancelled) return;
            if (sel?.status === "ok") { selected = initialScenario; inputs = sel.inputs ?? inputs; }
          }
          setChoice(`preset:${selected ?? initialScenario}`);
          onEngineChoice(null);
        }
        onInputs?.(inputs);
        setChoice((c) => c || `preset:${selected ?? r.scenarios[0]?.name ?? "healthy"}`);
      } catch {
        setPresets(null);
      }
      const { data: session } = await supabase.auth.getSession();
      if (!session.session || cancelled) return;
      const { data } = await supabase
        .from("engines")
        .select("id, name, engine_hours, tbo_hours, status")
        .eq("engine_model", engineModel)
        .order("updated_at", { ascending: false });
      if (!cancelled) setSaved((data as SavedEngine[] | null) ?? []);
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, engineModel, started]);

  if (!presets) return null;

  const choose = async (value: string) => {
    setChoice(value);
    setNote(null);
    if (value.startsWith("saved:")) {
      onEngineChoice(Number(value.slice(6)));
      return;
    }
    onEngineChoice(null);
    const r = await fetch(`${api}/scenario`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: value.slice(7) }),
    }).then((x) => x.json()).catch(() => null);
    if (r?.status !== "ok") setNote(r?.detail ?? "Could not choose that engine");
    else onInputs?.(r.inputs ?? null);
  };

  // One request at a time per control; the note says what happened.
  const send = async (path: string, body: Record<string, unknown>, id: string, ok: string) => {
    setBusy(id);
    const r = await fetch(`${api}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }).then((x) => x.json()).catch(() => null);
    setBusy(null);
    setNote(r?.status === "ok" ? ok : r?.detail ?? "Request failed");
  };
  const faultSev = truth?.fault_severity ?? {};
  const sensorFaults = truth?.sensor_faults ?? {};
  const faults = applicableFaults.length ? applicableFaults : faultList;
  const divider = <span className="mx-1 h-4 w-px shrink-0 bg-[#4c3025]" />;

  return (
    <div className="shrink-0 flex h-10 min-w-0 items-center gap-2 overflow-x-auto overflow-y-hidden whitespace-nowrap rounded border border-[#4c3025] bg-[#140f0c] px-2.5">
      <span className="shrink-0 text-[8px] uppercase tracking-[0.12em] text-[#bca18e] md:text-[9px]">
        {started ? "Inject" : "Engine"}
      </span>

      {!started && (
        <>
          <select className={sel} value={choice} onChange={(e) => void choose(e.target.value)} aria-label="Engine to fly">
            <optgroup label="Demo scenarios (a fresh engine)">
              {presets.map((p) => (
                <option key={p.name} value={`preset:${p.name}`}>{p.label}</option>
              ))}
            </optgroup>
            {saved.length > 0 && (
              <optgroup label="Your engines (continue from their hour meter)">
                {saved.map((e) => (
                  <option key={e.id} value={`saved:${e.id}`} disabled={e.status === "worn_out"}>
                    {e.name} &middot; {e.engine_hours.toFixed(1)} / {e.tbo_hours.toFixed(0)} h{e.status === "worn_out" ? " (worn out)" : ""}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
          {lifeScale != null && (
            <span className="shrink-0 text-[8px] text-[#aa8f7f] md:text-[9px]"
                  title="Engine hours advance this much faster than flight time, so wear builds up visibly during a demo flight. The AI still sees real seconds.">
              {formatLifeScale(lifeScale)} &middot; 1 real min = {((60 * lifeScale) / 3600).toFixed(0)} engine h
            </span>
          )}
          {divider}
          <span className="shrink-0 text-[8px] text-[#7d6758] md:text-[9px]">Fault injection opens once the flight starts</span>
        </>
      )}

      {started && (
        <>
          <input type="range" min={0.1} max={0.9} step={0.05} value={severity}
                 onChange={(e) => setSeverity(Number(e.target.value))} className="cockpit-range w-16 shrink-0"
                 aria-label="Severity for the next fault" />
          <span className="shrink-0 font-mono text-[8px] text-[#e8c9a0] md:text-[9px]" title="Severity for the next fault you click">
            {Math.round(severity * 100)}%
          </span>
          {faults.map((f) => {
            const on = faultSev[f] != null;
            return (
              <button key={f} disabled={busy === f} aria-pressed={on}
                      title={on ? "Click to remove this fault" : `Click to inject at ${Math.round(severity * 100)}%`}
                      className={`${chipBase} ${on ? chipOn : chipOff}`}
                      onClick={() => void (on
                        ? send("/inject/clear", { kind: "fault", name: f }, f, `${faultLabel(f)} removed`)
                        : send("/inject", { kind: "fault", name: f, severity }, f, `${faultLabel(f)} injected - watch whether the AI finds it`))}>
                {faultLabel(f)}{on && <> {Math.round(faultSev[f] * 100)}% &times;</>}
              </button>
            );
          })}

          {divider}

          <select className={sel} value={channel} onChange={(e) => setChannel(e.target.value)} aria-label="Sensor to break">
            {Object.keys(SENSOR_LABELS).map((c) => (
              <option key={c} value={c}>{sensorLabel(c)}{sensorFaults[c] ? ` (${sensorFaults[c]})` : ""}</option>
            ))}
          </select>
          {SENSOR_KINDS.map((k) => (
            <button key={k} disabled={busy === `s:${k}`} title={`Break the ${sensorLabel(channel)} sensor: ${k}`}
                    className={`${chipBase} ${sensorFaults[channel] === k ? chipOn : chipOff}`}
                    onClick={() => void send("/inject", { kind: "sensor", channel, type: k, severity: 0.8 }, `s:${k}`,
                                             `${sensorLabel(channel)} ${k} - watch whether the AI finds it`)}>
              {k}
            </button>
          ))}
          {Object.entries(sensorFaults).map(([ch, type]) => (
            <button key={ch} disabled={busy === `c:${ch}`} title="Click to repair this sensor"
                    className={`${chipBase} ${chipOn}`}
                    onClick={() => void send("/inject/clear", { kind: "sensor", channel: ch }, `c:${ch}`,
                                             `${sensorLabel(ch)} repaired`)}>
              {sensorLabel(ch)} {type} &times;
            </button>
          ))}
        </>
      )}

      {note && <span role="status" className="shrink-0 text-[8px] text-[#ffd27a] md:text-[9px]">{note}</span>}
    </div>
  );
}
