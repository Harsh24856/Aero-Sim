"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import Sensr from "@/components/Sensr";
import SimulatorDefault, { type SimTelemetry, type ResumeState } from "@/components/Simulator";
import Simulator912 from "@/components/Simulator_912";
import Simulator915 from "@/components/Simulator_915";
import Simulator916 from "@/components/Simulator_916";
import Meters, { type RawTelemetry, mpsToKnots } from "@/components/Meters";
import Diagnostics, { type AiResult, type Advisory, type Residuals, type LinkState, type SimStatus } from "@/components/Diagnostics";
import { getPreset, legAt, presetDuration } from "@/lib/missionPresets";
import Link from "next/link";
import { supabase } from "@/lib/supabase";
import { engineHoursOfWear, flightHours, formatSimClock } from "@/lib/timeScale";
import HealthVignette from "@/components/HealthVignette";
import SoundToggle from "@/components/SoundToggle";
import { EngineSound } from "@/lib/engineSound";

// Which themed Simulator variant to render, based on the ?engine= query param
// set by /engine's selectEngine() navigation. Falls back to the base (914)
// variant for the 914 itself or any unrecognized/missing value.
const SIMULATOR_BY_ENGINE: Record<string, typeof SimulatorDefault> = {
  Rotax_912_ULS: Simulator912,
  Rotax_915_iS: Simulator915,
  Rotax_916_iS: Simulator916,
};

const API = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws";

function SimulatePageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const engineParam = searchParams.get("engine") ?? "";
  const ActiveSimulator = SIMULATOR_BY_ENGINE[engineParam] ?? SimulatorDefault;
  // Set by /telemetry/[id]'s "Continue Simulation" button, which has already
  // called POST /resume (restoring the backend twin's exact altitude/throttle/
  // airspeed/wear/RUL) and started the physics loop before navigating here. This
  // page must therefore ADOPT that state rather than begin a fresh takeoff.
  const isResume = searchParams.get("resumed") === "1";
  // Mission profile (PS section E). A resumed session never plays a profile:
  // resuming must adopt the stored state, and a profile would immediately
  // overwrite it - the same class of bug the auto-climb guard above fixes.
  const preset = isResume ? null : getPreset(searchParams.get("preset"));

  // Simulating requires being signed in - null while the initial session check is
  // still in flight (so Start does not briefly appear usable before we actually
  // know), then true/false once resolved.
  const [isSignedIn, setIsSignedIn] = useState<boolean | null>(null);
  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => setIsSignedIn(!!data.session));
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsSignedIn(!!session);
    });
    return () => listener.subscription.unsubscribe();
  }, []);

  // Keep the backend on the engine this page shows. /engine's Select button calls
  // /select_engine before navigating here, but a direct link, a bookmark, a refresh
  // or the footer "Live Cockpit" link did not - and the twin then flew whichever
  // engine was selected last (found in testing: a 915 run recorded under the 914
  // cockpit). Never while resuming, and never over a session that is running.
  useEffect(() => {
    if (isResume || !engineParam) return;
    (async () => {
      try {
        const st = await fetch(`${API}/state`).then((r) => r.json());
        if (!st.running && st.engine_model !== engineParam) {
          await fetch(`${API}/select_engine`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ engine_model: engineParam }),
          });
        }
      } catch {
        // Backend down - the cockpit already reports that on its own.
      }
    })();
  }, [engineParam, isResume]);

  // 35% - a realistic post-run-up idle/taxi setting, and critically ABOVE the
  // training data's throttle minimum of 0.118. The previous 5% default sat
  // entirely outside the AI's training distribution (0% of training rows are
  // below 0.10 throttle), which made every prediction an extrapolation.
  const [throttle, setThrottle] = useState(35);
  // 35 m/s = SPEED_MIN, the stall floor the simulator enforces in manual flight
  // (Simulator.tsx). The old 30 m/s default sat BELOW that floor, so the speed
  // slider opened with its readout and its thumb disagreeing - the thumb pinned
  // at the floor while the output showed a value the control could not express.
  const [airspeedTarget, setAirspeedTarget] = useState(35); // m/s
  const [started, setStarted] = useState(false);
  const [paused, setPaused] = useState(false);
  useEffect(() => { flyingRef.current = started && !paused; }, [started, paused]);
  const [liveTelemetry, setLiveTelemetry] = useState<SimTelemetry | null>(null);
  const [aiResult, setAiResult] = useState<AiResult | null>(null);
  const [advisory, setAdvisory] = useState<Advisory | null>(null);
  const [residuals, setResiduals] = useState<Residuals | null>(null);
  // Safety net: the WebSocket link to main.py and the backend's own health block.
  const [link, setLink] = useState<LinkState>("connecting");
  const [simStatus, setSimStatus] = useState<SimStatus | null>(null);
  const [legIndex, setLegIndex] = useState(0);
  // Mission autopilot. Engaged by default for a profile run; the pilot can hand
  // themselves the controls at any moment, and once disengaged the profile stops
  // commanding anything - taking over must actually mean taking over.
  const [autopilotOn, setAutopilotOn] = useState(true);
  // Simulated time at which the profile's clock starts.
  //
  // main.py fast-forwards the first 128 simulated seconds so the AI's window
  // fills without waiting 128 real ones. rawTelemetry.time therefore leaps
  // during warmup, and keying legs straight off it burned through the opening
  // legs of every profile in a few real seconds - a High Altitude run reached
  // leg 3 of 3 in about twenty. The profile clock starts when the AI goes live,
  // so each leg gets the duration it asks for.
  const missionT0 = useRef<number | null>(null);
  const [rawTelemetry, setRawTelemetry] = useState<RawTelemetry | null>(null);
  // Physics generation of the live twin, from the telemetry itself - available even
  // when the AI service is down and there is no ai.model_version to read.
  const livePhysicsVersion = (rawTelemetry as { physics_version?: string } | null)?.physics_version;
  // v3 "flight time" is the engine hour meter: wear x TBO, exactly TBO minus the true
  // RUL. A per-session delta was tried first and read 0.0 h - its baseline was taken
  // from the previous flight's last frame (higher wear) before the new twin's first
  // frame arrived. The meter needs no baseline and matches the mission report.
  const liveWear = (rawTelemetry as { wear?: number } | null)?.wear;
  const liveTbo = (rawTelemetry as { tbo_hours?: number } | null)?.tbo_hours;
  const wsRef = useRef<WebSocket | null>(null);
  // Read by the WebSocket watchdog, which lives in a mount-once effect.
  const flyingRef = useRef(false);

  // ---- engine sound (lib/engineSound.ts) ----
  // Created lazily inside a click (browsers block audio until a user gesture).
  // Mute and volume persist per browser; every storage access is guarded.
  const soundRef = useRef<EngineSound | null>(null);
  const [soundMuted, setSoundMuted] = useState(false);
  const [soundVolume, setSoundVolume] = useState(0.6);
  useEffect(() => {
    try {
      const m = localStorage.getItem("aero.sound.muted");
      const v = localStorage.getItem("aero.sound.volume");
      if (m !== null) setSoundMuted(m === "1");
      if (v !== null && Number.isFinite(Number(v))) setSoundVolume(Math.min(1, Math.max(0, Number(v))));
    } catch { /* storage unavailable - defaults */ }
  }, []);
  useEffect(() => {
    soundRef.current?.setMuted(soundMuted);
    try { localStorage.setItem("aero.sound.muted", soundMuted ? "1" : "0"); } catch { /* ignore */ }
  }, [soundMuted]);
  useEffect(() => {
    soundRef.current?.setVolume(soundVolume);
    try { localStorage.setItem("aero.sound.volume", String(soundVolume)); } catch { /* ignore */ }
  }, [soundVolume]);
  const ensureSound = useCallback(async () => {
    try {
      if (!soundRef.current) soundRef.current = new EngineSound();
      soundRef.current.setVolume(soundVolume);
      soundRef.current.setMuted(soundMuted);
      await soundRef.current.start();
    } catch {
      // Audio is a nicety - never let it break Start.
    }
  }, [soundMuted, soundVolume]);
  useEffect(() => () => soundRef.current?.dispose(), []);

  // Non-null once the backend's restored state has been read back. Passed to the
  // Simulator as its FIRST-render initial state, which is why the simulator is
  // withheld from the tree until this resolves (see resumePending below) - the
  // component seeds its physics refs from it, and useRef only honours the value
  // it is given on the very first render.
  const [resumeInit, setResumeInit] = useState<ResumeState | null>(null);
  // True from mount until the /resume handshake above has either succeeded or
  // been ruled out. Rendering the simulator during this window would mount it
  // with fresh-takeoff defaults, which is precisely what must not happen.
  const [resumePending, setResumePending] = useState(isResume);

  // Reads the ALREADY-RESTORED state straight back off the backend rather than
  // passing it through the URL: /state's params are the physics twin's own live
  // values, so there is no way for them to disagree with what /resume actually
  // restored. Deliberately does NOT call /start - /resume already set
  // session_active and launched the simulation loop, and a /start here would at
  // best be a no-op ("already_running") and at worst re-enter the fresh-session
  // branch and discard the restored twin.
  useEffect(() => {
    if (!isResume) return;
    let cancelled = false;
    // Hard deadline on the handshake. Without it a backend that accepts the
    // connection but never answers leaves this page stranded on "Restoring flight
    // state..." with no way forward - strictly worse than the fresh-start UI it is
    // standing in for. Aborting falls through to that UI instead.
    const abort = new AbortController();
    const deadline = setTimeout(() => abort.abort(), 4000);
    (async () => {
      try {
        const res = await fetch(`${API}/state`, { signal: abort.signal });
        const s = await res.json();
        if (cancelled) return;
        // running === false means there is no resumed session to adopt (a stale
        // ?resumed=1 URL, a reloaded tab, a backend restarted since). Fall back
        // to the normal "click Start for a fresh flight" UI rather than pretending.
        if (s?.running && s.params) {
          setThrottle(Math.round((s.params.throttle ?? 0.35) * 100));
          setAirspeedTarget(s.params.airspeed ?? 30);
          setResumeInit({
            altitude: s.params.altitude ?? 0,
            speed: s.params.airspeed ?? 0,
            pitch: s.params.aoa ?? 0,
          });
          setStarted(true);
        }
      } catch {
        // Backend unreachable, or the deadline above fired - same graceful
        // degradation as everywhere else on this page.
      } finally {
        clearTimeout(deadline);
        if (!cancelled) setResumePending(false);
      }
    })();
    return () => { cancelled = true; clearTimeout(deadline); abort.abort(); };
  }, [isResume]);

  // Set only by an explicit Stop (never by Pause) - showing this is what makes
  // Stop feel genuinely different from Pause, and confirms the run was actually
  // persisted rather than just leaving the UI silently reset to "waiting".
  const [stoppedSummary, setStoppedSummary] = useState<{
    healthPercent: number | null;
    rulPercent: number | null;
    engineHours: number | null;   // v3: engine hour meter at stop (wear x TBO)
    simSeconds: number;   // actual simulated flight time (rawTelemetry.time), not
                          // wall-clock testing time - meaningless numbers otherwise,
                          // since a 20-real-second test session should read as real
                          // equivalent flight hours, not literally "20 seconds"
    outcome: string;
    simulationId?: number | null;
  } | null>(null);

  // main.py already merges the AI service's response into every WebSocket
  // broadcast (see main.py's simulation_loop) - this just reads that 'ai' field
  // back out, so Diagnostics can show the REAL model output, not a placeholder.
  //
  // SAFETY NET: the socket reconnects on its own with backoff (0.5 s doubling to 5 s),
  // so a backend restart or a dropped connection recovers without a page reload. A
  // frame that fails to parse is skipped rather than throwing inside onmessage. And
  // the AI panel follows the payload's "ai" key exactly, including null, so a stale
  // "AI service unavailable" can never outlive the session that produced it.
  useEffect(() => {
    let closedByPage = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let lastFrame = Date.now();

    const connect = () => {
      setLink((l) => (l === "open" ? "connecting" : l));
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => { retry = 0; lastFrame = Date.now(); setLink("open"); };
      ws.onmessage = (event) => {
        let data: Record<string, unknown>;
        try {
          data = JSON.parse(event.data);
        } catch {
          return;   // one bad frame must not take the cockpit down
        }
        lastFrame = Date.now();
        setLink((l) => (l === "open" ? l : "open"));
        if ("ai" in data) setAiResult((data.ai as AiResult | null) ?? null);
        if (data.advisory) setAdvisory(data.advisory as Advisory);
        if ("residuals" in data) setResiduals((data.residuals as Residuals | null) ?? null);
        if (data.sim_status) setSimStatus(data.sim_status as SimStatus);
        setRawTelemetry(data as unknown as RawTelemetry);   // full payload - all raw features for Sensr
      };
      ws.onerror = () => { /* onclose follows and handles the retry */ };
      ws.onclose = () => {
        if (closedByPage) return;
        setLink("lost");
        const delay = Math.min(5000, 500 * 2 ** retry);
        retry += 1;
        timer = setTimeout(connect, delay);
      };
    };
    connect();

    // A socket can stay "open" while nothing arrives (a half-dead connection, a backend
    // that dropped this client). No frame for 4 s while flying: show it; for 8 s: close
    // the socket ourselves so onclose reconnects. Idle (not flying) sends no frames, so
    // only act while a flight is live.
    const watchdog = setInterval(() => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || !flyingRef.current) { lastFrame = Date.now(); return; }
      const silent = Date.now() - lastFrame;
      if (silent > 8000) {
        lastFrame = Date.now();
        ws.close();
      } else if (silent > 4000) {
        setLink((l) => (l === "open" ? "stale" : l));
      }
    }, 1000);

    return () => {
      closedByPage = true;
      if (timer) clearTimeout(timer);
      clearInterval(watchdog);
      wsRef.current?.close();
    };
  }, []);

  const onTelemetryChange = useCallback((t: SimTelemetry) => setLiveTelemetry(t), []);

  // Engine sound follows the physics stream while flying; pause/stop fade it out.
  useEffect(() => {
    const snd = soundRef.current;
    if (!snd) return;
    if (!started || paused) { if (snd.isRunning) snd.stop(); return; }
    if (!snd.isRunning) { void snd.start(); }
    const t = rawTelemetry as unknown as Record<string, unknown> | null;
    if (!t) return;
    const num = (k: string) => (typeof t[k] === "number" ? (t[k] as number) : NaN);
    const fm = aiResult?.status === "ok" ? aiResult.failure_modes : undefined;
    const maxKw = ({ Rotax_912_ULS: 73.5, Rotax_914_ULF: 84.8, Rotax_915_iS: 104, Rotax_916_iS: 117 } as Record<string, number>)[engineParam] ?? 85;
    snd.update({
      rpm: num("engine_rpm"),
      propRpm: num("prop_rpm"),
      throttle: num("throttle"),
      load: num("power_kw") / maxKw,
      airspeed: num("airspeed"),
      misfire: !!fm?.misfire?.present,
      instability: !!fm?.combustion_instability?.present,
    });
  }, [rawTelemetry, started, paused, aiResult, engineParam]);

  // Mission-profile leg advance. Keyed off rawTelemetry.time - the twin's own
  // simulated clock - not wall-clock, so a leg boundary stays correct even if
  // physics briefly falls behind real time. Setting throttle/airspeed here is
  // exactly what a human would do with the cockpit controls; nothing bypasses
  // the normal /params path.
  // Latch the profile's zero once the AI is producing output.
  useEffect(() => {
    if (!preset || !started) { return; }
    if (missionT0.current === null && aiResult?.status === "ok" && rawTelemetry?.time != null) {
      missionT0.current = rawTelemetry.time;
    }
  }, [preset, started, aiResult?.status, rawTelemetry?.time]);
  useEffect(() => { if (!started) missionT0.current = null; }, [started]);
  const sessionEngineHours = livePhysicsVersion === "v3" ? engineHoursOfWear(liveWear, liveTbo) : null;

  const missionElapsed =
    missionT0.current === null ? 0 : Math.max(0, (rawTelemetry?.time ?? 0) - missionT0.current);
  const activeLeg = preset ? legAt(preset, missionElapsed) : null;
  useEffect(() => {
    if (!preset || !autopilotOn || !started || paused || resumePending) return;
    const { leg, index } = legAt(preset, missionElapsed);
    if (index === legIndex) return;
    setLegIndex(index);
    setThrottle(Math.round(leg.throttle * 100));
    setAirspeedTarget(leg.airspeed);
  }, [preset, autopilotOn, started, paused, resumePending, missionElapsed, legIndex]);

  // Apply the opening leg's setpoints as soon as a profile run starts.
  useEffect(() => {
    if (!preset || !autopilotOn || !started || resumePending) return;
    const first = preset.legs[0];
    setThrottle(Math.round(first.throttle * 100));
    setAirspeedTarget(first.airspeed);
    // Intentionally only on transition into `started` for a profile run.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset, autopilotOn, started, resumePending]);

  // AoA comes directly from Simulator's pitch (see Simulator.tsx's formatPitch -
  // altitude changes there produce a pitch angle, which IS our AoA). Forwarded to
  // the real backend alongside altitude/throttle/airspeed, throttled to match the
  // ~10/sec rate telemetry already arrives at.
  useEffect(() => {
    if (!started || paused || !liveTelemetry) return;
    // Never write params while a resume is still being read back - the values in
    // flight at that moment are the simulator's fresh-takeoff defaults, and
    // sending them would overwrite the very state we are about to adopt.
    if (resumePending) return;
    fetch(`${API}/params`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        altitude: liveTelemetry.altitude,
        throttle: throttle / 100,
        airspeed: liveTelemetry.speed,
        aoa: liveTelemetry.pitch,
        // Mission-profile environment. Undefined outside a profile run, and
        // ParamUpdate leaves the twin's value untouched when it is null, so a
        // normal flight stays on a standard day.
        isa_dev_c: (autopilotOn ? activeLeg?.leg.isaDevC : undefined) ?? 0,
      }),
    }).catch(() => {
      // Backend not running is not a reason to break the local simulator display -
      // degrade gracefully, same pattern used in ai.py's own error handling.
    });
  }, [started, paused, liveTelemetry, throttle, resumePending, autopilotOn, activeLeg?.leg.isaDevC]);

  // Simulating now requires being signed in - checked here (the actual
  // enforcement point) rather than only hiding/disabling the button, since a
  // disabled button alone would not stop a direct call to this handler. Not
  // signed in -> redirect to /login instead of starting anything, matching how
  // the login page itself already has a "continue without an account" escape
  // hatch for anyone who genuinely does not want to sign up.
  const onStart = useCallback(async () => {
    const { data } = await supabase.auth.getSession();
    const userId = data.session?.user?.id ?? null;
    if (!userId) {
      router.push("/login");
      return;
    }
    setStoppedSummary(null);   // dismiss any previous run's summary card
    void ensureSound();
    setStarted(true);
    try {
      await fetch(`${API}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
      });
    } catch {
      // backend optional for the local game/gauges to work
    }
  }, [router, ensureSound]);

  // Pause must ACTUALLY stop the backend's physics loop, not just the frontend's
  // own game/display state - main.py's simulation_loop() runs independently once
  // started, and keeps calling ai.py every simulated second FOREVER until /stop is
  // called, regardless of what the frontend UI shows. Without this, "pausing" only
  // stopped the local game while the backend kept silently hammering the AI service.
  const onTogglePause = useCallback(async () => {
    // Read the session BEFORE the setPaused updater - getSession() is async and
    // the updater function itself must stay synchronous.
    const { data: sessionData } = await supabase.auth.getSession();
    const userId = sessionData.session?.user?.id ?? null;
    setPaused((p) => {
      const next = !p;
      if (next) {
        // final: false - this is a PAUSE, not a genuine stop. The backend keeps
        // the physics session alive (session_active stays true) so the matching
        // /start below resumes rather than resetting the twin's time/altitude/wear.
        fetch(`${API}/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ final: false }),
        }).catch(() => {});
      } else {
        // Backend reuses the existing simulation_id on resume (see main.py's
        // /start) rather than fragmenting one flight into multiple DB rows - the
        // user_id here only matters the first time a fresh one gets created.
        fetch(`${API}/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: userId }),
        }).catch(() => {});
      }
      return next;
    });
  }, []);

  // Explicit "Stop Simulation" - genuinely ends the flight (not a pause): calls
  // /stop, which main.py already persists to Supabase (closes the simulations row
  // with outcome="stopped" and the last known health/RUL values), then resets the
  // frontend UI back to its pre-start state so the game/gauges show "stopped"
  // rather than leaving a paused-looking UI that implies resuming is still possible.
  const onStopClick = useCallback(async () => {
    let stopSucceeded = false;
    let closedId: number | null = null;
    try {
      const res = await fetch(`${API}/stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ final: true }),   // explicit, matches the backend default - a genuine Stop
      });
      stopSucceeded = res.ok;
      if (res.ok) {
        const json = await res.json().catch(() => null);
        closedId = json?.simulation_id ?? null;
      }
    } catch {
      // backend optional - local game/gauges still get reset below regardless
    }
    setStoppedSummary({
      simulationId: closedId,
      healthPercent: aiResult?.status === "ok" ? aiResult.health_percent ?? null : null,
      rulPercent: aiResult?.status === "ok" ? aiResult.rul_percent_remaining ?? null : null,
      simSeconds: rawTelemetry?.time ?? 0,
      engineHours: sessionEngineHours,
      outcome: stopSucceeded ? "Saved" : "Stopped locally (backend unreachable - not saved)",
    });
    soundRef.current?.stop();
    setStarted(false);
    setPaused(false);
  }, [aiResult, rawTelemetry, sessionEngineHours]);

  // The post-flight summary is generated by a fire-and-forget task on /stop, so
  // it is never ready the instant this card appears. Poll the row until it lands,
  // then stop. Gives up after ~60s rather than polling forever.
  const [stopGroq, setStopGroq] = useState<{ headline?: string; summary?: string; risk?: string; status?: string } | null>(null);
  useEffect(() => {
    const simId = stoppedSummary?.simulationId;
    if (!simId) { setStopGroq(null); return; }
    let cancelled = false;
    let ticks = 0;
    const tick = async () => {
      ticks += 1;
      const { data } = await supabase
        .from("simulations").select("groq_result").eq("id", simId).single();
      if (cancelled) return;
      if (data?.groq_result) { setStopGroq(data.groq_result); clearInterval(handle); }
      else if (ticks >= 20) clearInterval(handle);
    };
    const handle = setInterval(tick, 3000);
    tick();
    return () => { cancelled = true; clearInterval(handle); };
  }, [stoppedSummary?.simulationId]);

  // Safety net: if the user navigates away entirely (not just pausing) while the
  // simulation is running, the backend loop would otherwise keep running forever
  // with no UI left to show it or pause it. Two mechanisms, for two different
  // ways of "leaving":
  //  - pagehide + sendBeacon: fires on hard refresh, tab close, or typing a new
  //    URL - a regular fetch() is NOT guaranteed to complete once the page
  //    actually starts unloading, but sendBeacon is specifically designed by
  //    browsers to reliably deliver a small POST in exactly this situation.
  //  - the React effect cleanup below: fires on CLIENT-SIDE navigation (clicking
  //    a Navbar link, browser back/forward within this app) - the page never
  //    actually unloads in that case, so pagehide would not fire, but the
  //    component genuinely unmounts and a normal fetch completes fine.
  // Read through a ref: a [started] dependency would run the cleanup whenever Stop
  // flips started to false, sending a second /stop right after the explicit one.
  const startedRef = useRef(started);
  startedRef.current = started;
  useEffect(() => {
    const sendFinalStop = () => {
      if (!startedRef.current) return;
      const body = new Blob([JSON.stringify({ final: true })], { type: "application/json" });
      navigator.sendBeacon(`${API}/stop`, body);
    };
    window.addEventListener("pagehide", sendFinalStop);
    return () => {
      window.removeEventListener("pagehide", sendFinalStop);
      if (startedRef.current) {
        // final: true (explicit) - navigating away entirely is a genuine end of
        // session, not a pause, regardless of whether it happened to be paused
        // at the moment of navigating away.
        fetch(`${API}/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ final: true }),
        }).catch(() => {});
      }
    };
  }, []);

  return (
    <>
      <Navbar />
      {/* mt-16 clears the fixed Navbar (h-16 = 4rem) WITHOUT eating into
          the grid's own height. h-[calc(100vh-4rem)] is the actual
          content area; overflow-hidden prevents page-level scroll. */}
      <div className="mt-16 grid h-[calc(100vh-4rem)] grid-cols-1 gap-2 overflow-hidden bg-[#050403] p-2 md:grid-cols-[240px_1fr_240px]">
        <Sensr
          rawTelemetry={rawTelemetry}
        />

        {/* Center column: simulator on top, meters (ring gauge + controls) below.
            min-h-0 on each so flex children actually shrink to fit. */}
        <div className="flex min-h-0 flex-col gap-2">
          {/* Mission strip: which profile is flying, which leg, and who has
              control. Only present on a profile run. */}
          {preset && (
            <div className="shrink-0 flex flex-wrap items-center justify-between gap-2 rounded border border-[#4c3025] bg-[#140f0c] px-2.5 py-1.5">
              <div className="flex items-baseline gap-2 min-w-0">
                <span className="text-[8px] uppercase tracking-[0.12em] text-[#bca18e] md:text-[9px]">Mission</span>
                <span className="truncate text-[11px] font-bold text-[#efe0d5] md:text-[13px]">{preset.name}</span>
                {activeLeg && (
                  <span className="truncate text-[9px] text-[#aa8f7f] md:text-[11px]">
                    Leg {activeLeg.index + 1}/{preset.legs.length} &middot; {activeLeg.leg.label}
                    {" \u00b7 "}{Math.round(activeLeg.leg.altitude)} m
                    {" \u00b7 "}{Math.round(activeLeg.leg.throttle * 100)}%
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <span className={`text-[8px] font-bold uppercase tracking-[0.12em] md:text-[9px] ${autopilotOn ? "text-[#7fc87f]" : "text-[#ff9a72]"}`}>
                  {autopilotOn ? "Autopilot engaged" : "Manual control"}
                </span>
                <button
                  onClick={() => setAutopilotOn((v) => !v)}
                  className="rounded border border-[#4c3025] bg-[#1a110d] px-2.5 py-1 text-[8px] font-bold uppercase tracking-[0.12em] text-[#e8c9a0] transition-colors hover:text-[#ff8050] md:text-[9px]"
                >
                  {autopilotOn ? "Take control" : "Re-engage"}
                </button>
              </div>
            </div>
          )}
          <div className="relative min-h-0" style={{ flex: '50 1 0%' }}>
            {/* Bottom-right: the top corners carry the HUD's HDG/ALT and TAS/R-C readouts. */}
            <div className="absolute bottom-2 right-2 z-20">
              <SoundToggle
                muted={soundMuted}
                volume={soundVolume}
                onToggle={() => { setSoundMuted((m) => !m); if (started && !paused) void ensureSound(); }}
                onVolume={(v) => { setSoundVolume(v); if (soundMuted) setSoundMuted(false); }}
              />
            </div>
            {resumePending ? (
              <div className="flex h-full items-center justify-center rounded border border-outline-variant/30 bg-black/40 text-[11px] uppercase tracking-[0.15em] text-tertiary">
                Restoring flight state...
              </div>
            ) : (
            <ActiveSimulator
              autopilot={!!preset && autopilotOn}
              altitudeTarget={activeLeg?.leg.altitude}
              onTelemetryChange={onTelemetryChange}
              throttle={throttle}
              onThrottleChange={setThrottle}
              airspeedTarget={airspeedTarget}
              onAirspeedTargetChange={setAirspeedTarget}
              started={started}
              paused={paused}
              onStop={onStopClick}
              initialState={resumeInit}
            />
            )}
          </div>
          <div className="min-h-0" style={{ flex: '50 1 0%' }}>
            <Meters
              speedKnots={liveTelemetry ? mpsToKnots(liveTelemetry.speed) : 0}
              altitude={liveTelemetry?.altitude ?? 0}
              throttle={throttle}
              onThrottleChange={setThrottle}
              airspeedTarget={airspeedTarget}
              onAirspeedTargetChange={setAirspeedTarget}
              started={started}
              onStart={onStart}
              paused={paused}
              onTogglePause={onTogglePause}
              isSignedIn={isSignedIn}
            />
          </div>
        </div>

        {/* Only pass simSeconds while genuinely running - otherwise this reflects
            whatever the backend telemetry stream happens to contain (which could
            be leftover/unrelated to this frontend session entirely), showing a
            "moving" flight time even while paused or never started. */}
        <Diagnostics ai={aiResult} advisory={advisory} residuals={residuals} physicsVersion={livePhysicsVersion} link={started && !paused ? link : undefined} engineHours={started ? sessionEngineHours : undefined} simStatus={simStatus} simSeconds={started && !paused ? rawTelemetry?.time : undefined} />
      </div>

      <HealthVignette
        active={started && !paused && aiResult?.status === "ok"}
        health={aiResult?.status === "ok" ? aiResult.health_percent : null}
      />

      {/* Shown ONLY after an explicit Stop, never after Pause - this is what makes
          the two feel genuinely distinct instead of both just freezing the game. */}
      {stoppedSummary && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="max-h-[88vh] w-full max-w-lg overflow-y-auto rounded-lg border border-tertiary/40 bg-[#0d0e0d] p-6 text-center">
            <div className="mb-1 text-[11px] uppercase tracking-[0.15em] text-tertiary">Simulation {stoppedSummary.outcome}</div>
            <div className="mb-5 text-[12px] text-on-surface-variant">
              {/* Real-world equivalent flight time, not wall-clock test duration -
                  a 20-second local session is meaningless as "20s" against a
                  2,000h TBO; scaled through the same compression factor as RUL,
                  it correctly reads as ~2 real hours of flight. */}
              {livePhysicsVersion === "v3"
                ? <>Engine hours: {stoppedSummary.engineHours != null ? `${stoppedSummary.engineHours.toFixed(1)} h` : "--"} &middot; sim {formatSimClock(stoppedSummary.simSeconds)}</>
                : <>Flight time: {flightHours(stoppedSummary.simSeconds, "v2").toFixed(1)}h (real-world equivalent)</>}
            </div>
            <div className="mb-6 grid grid-cols-2 gap-3">
              <div className="rounded border border-outline-variant/30 bg-black/40 p-3">
                <div className="text-[10px] uppercase tracking-[0.1em] text-on-surface-variant">Final Health</div>
                <div className="text-2xl font-bold text-primary">
                  {stoppedSummary.healthPercent != null ? `${stoppedSummary.healthPercent.toFixed(0)}%` : "--"}
                </div>
              </div>
              <div className="rounded border border-outline-variant/30 bg-black/40 p-3">
                <div className="text-[10px] uppercase tracking-[0.1em] text-on-surface-variant">Final RUL</div>
                <div className="text-2xl font-bold text-primary">
                  {stoppedSummary.rulPercent != null ? `${stoppedSummary.rulPercent.toFixed(0)}%` : "--"}
                </div>
              </div>
            </div>
            {/* Post-flight analysis. Only rendered for a run that actually
                persisted - an unsaved local stop has no row to summarize. */}
            {stoppedSummary.simulationId != null && (
              <div className="mb-6 rounded border border-outline-variant/30 bg-black/40 p-4 text-left">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="text-[10px] uppercase tracking-[0.1em] text-tertiary">Post-Flight Analysis</span>
                  {stopGroq?.risk && (
                    <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-tertiary">
                      {stopGroq.risk} risk
                    </span>
                  )}
                </div>

                {!stopGroq && (
                  <p className="text-[11px] leading-relaxed text-on-surface-variant">
                    Generating analysis&hellip; this runs in the background after the mission
                    ends and lands in a few seconds.
                  </p>
                )}

                {stopGroq && stopGroq.status !== "ok" && (
                  <p className="text-[11px] leading-relaxed text-on-surface-variant">
                    Analysis unavailable for this mission.
                  </p>
                )}

                {stopGroq && stopGroq.status === "ok" && (
                  <>
                    {/* Plain text only - third-party model output. */}
                    <p className="mb-1.5 text-[12px] font-bold leading-snug text-primary">{stopGroq.headline}</p>
                    <p className="text-[11px] leading-relaxed text-on-surface-variant">{stopGroq.summary}</p>
                    <Link
                      href={`/mission/report/${stoppedSummary.simulationId}`}
                      className="mt-2 inline-block text-[10px] font-bold uppercase tracking-[0.1em] text-tertiary hover:brightness-125"
                    >
                      Full mission report &rarr;
                    </Link>
                  </>
                )}
              </div>
            )}

            <div className="flex flex-col gap-2 sm:flex-row">
              <button
                onClick={() => setStoppedSummary(null)}
                className="w-full rounded bg-tertiary py-2.5 text-[11px] font-bold uppercase tracking-[0.1em] text-black hover:brightness-110 transition-all"
              >
                Start New Simulation
              </button>
              <Link
                href="/engine"
                className="w-full rounded border border-outline-variant/30 bg-black/40 py-2.5 text-center text-[11px] font-bold uppercase tracking-[0.1em] text-on-surface-variant transition-all hover:text-primary"
              >
                Back to Engines
              </Link>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default function SimulatePage() {
  return (
    <Suspense fallback={null}>
      <SimulatePageInner />
    </Suspense>
  );
}
