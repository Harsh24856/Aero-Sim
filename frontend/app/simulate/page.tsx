"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import Sensr from "@/components/Sensr";
import SimulatorDefault, { type SimTelemetry, type ResumeState } from "@/components/Simulator";
import Simulator912 from "@/components/Simulator_912";
import Simulator915 from "@/components/Simulator_915";
import Simulator916 from "@/components/Simulator_916";
import Meters, { type RawTelemetry, mpsToKmh } from "@/components/Meters";
import Diagnostics, { type AiResult, type Advisory } from "@/components/Diagnostics";
import { getPreset, legAt, presetDuration } from "@/lib/missionPresets";
import { supabase } from "@/lib/supabase";
import { simSecondsToRealHours } from "@/lib/timeScale";

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

  // 35% - a realistic post-run-up idle/taxi setting, and critically ABOVE the
  // training data's throttle minimum of 0.118. The previous 5% default sat
  // entirely outside the AI's training distribution (0% of training rows are
  // below 0.10 throttle), which made every prediction an extrapolation.
  const [throttle, setThrottle] = useState(35);
  const [airspeedTarget, setAirspeedTarget] = useState(30); // m/s
  const [started, setStarted] = useState(false);
  const [paused, setPaused] = useState(false);
  const [liveTelemetry, setLiveTelemetry] = useState<SimTelemetry | null>(null);
  const [aiResult, setAiResult] = useState<AiResult | null>(null);
  const [advisory, setAdvisory] = useState<Advisory | null>(null);
  const [legIndex, setLegIndex] = useState(0);
  const [rawTelemetry, setRawTelemetry] = useState<RawTelemetry | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

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
    simSeconds: number;   // actual simulated flight time (rawTelemetry.time), not
                          // wall-clock testing time - meaningless numbers otherwise,
                          // since a 20-real-second test session should read as real
                          // equivalent flight hours, not literally "20 seconds"
    outcome: string;
  } | null>(null);

  // main.py already merges the AI service's response into every WebSocket
  // broadcast (see main.py's simulation_loop) - this just reads that 'ai' field
  // back out, so Diagnostics can show the REAL model output, not a placeholder.
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.ai) setAiResult(data.ai);
      if (data.advisory) setAdvisory(data.advisory);
      setRawTelemetry(data);   // full payload - all 24 raw features for Sensr
    };
    ws.onerror = () => console.log("WebSocket error - is main.py running on :8000?");
    return () => ws.close();
  }, []);

  const onTelemetryChange = useCallback((t: SimTelemetry) => setLiveTelemetry(t), []);

  // Mission-profile leg advance. Keyed off rawTelemetry.time - the twin's own
  // simulated clock - not wall-clock, so a leg boundary stays correct even if
  // physics briefly falls behind real time. Setting throttle/airspeed here is
  // exactly what a human would do with the cockpit controls; nothing bypasses
  // the normal /params path.
  const activeLeg = preset ? legAt(preset, rawTelemetry?.time ?? 0) : null;
  useEffect(() => {
    if (!preset || !started || paused || resumePending) return;
    const { leg, index } = legAt(preset, rawTelemetry?.time ?? 0);
    if (index === legIndex) return;
    setLegIndex(index);
    setThrottle(Math.round(leg.throttle * 100));
    setAirspeedTarget(leg.airspeed);
  }, [preset, started, paused, resumePending, rawTelemetry?.time, legIndex]);

  // Apply the opening leg's setpoints as soon as a profile run starts.
  useEffect(() => {
    if (!preset || !started || resumePending) return;
    const first = preset.legs[0];
    setThrottle(Math.round(first.throttle * 100));
    setAirspeedTarget(first.airspeed);
    // Intentionally only on transition into `started` for a profile run.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset, started, resumePending]);

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
        isa_dev_c: activeLeg?.leg.isaDevC ?? 0,
      }),
    }).catch(() => {
      // Backend not running is not a reason to break the local simulator display -
      // degrade gracefully, same pattern used in ai.py's own error handling.
    });
  }, [started, paused, liveTelemetry, throttle, resumePending, activeLeg?.leg.isaDevC]);

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
  }, [router]);

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
    try {
      const res = await fetch(`${API}/stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ final: true }),   // explicit, matches the backend default - a genuine Stop
      });
      stopSucceeded = res.ok;
    } catch {
      // backend optional - local game/gauges still get reset below regardless
    }
    setStoppedSummary({
      healthPercent: aiResult?.status === "ok" ? aiResult.health_percent ?? null : null,
      rulPercent: aiResult?.status === "ok" ? aiResult.rul_percent_remaining ?? null : null,
      simSeconds: rawTelemetry?.time ?? 0,
      outcome: stopSucceeded ? "Saved" : "Stopped locally (backend unreachable - not saved)",
    });
    setStarted(false);
    setPaused(false);
  }, [aiResult, rawTelemetry]);

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
  useEffect(() => {
    const sendFinalStop = () => {
      if (!started) return;
      const body = new Blob([JSON.stringify({ final: true })], { type: "application/json" });
      navigator.sendBeacon(`${API}/stop`, body);
    };
    window.addEventListener("pagehide", sendFinalStop);
    return () => {
      window.removeEventListener("pagehide", sendFinalStop);
      if (started) {
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
  }, [started]);

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
          <div className="min-h-0" style={{ flex: '50 1 0%' }}>
            {resumePending ? (
              <div className="flex h-full items-center justify-center rounded border border-outline-variant/30 bg-black/40 text-[11px] uppercase tracking-[0.15em] text-tertiary">
                Restoring flight state...
              </div>
            ) : (
            <ActiveSimulator
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
              speedKmh={liveTelemetry ? mpsToKmh(liveTelemetry.speed) : 0}
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
        <Diagnostics ai={aiResult} advisory={advisory} simSeconds={started && !paused ? rawTelemetry?.time : undefined} />
      </div>

      {/* Shown ONLY after an explicit Stop, never after Pause - this is what makes
          the two feel genuinely distinct instead of both just freezing the game. */}
      {stoppedSummary && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-sm rounded-lg border border-tertiary/40 bg-[#0d0e0d] p-6 text-center">
            <div className="mb-1 text-[11px] uppercase tracking-[0.15em] text-tertiary">Simulation {stoppedSummary.outcome}</div>
            <div className="mb-5 text-[12px] text-on-surface-variant">
              {/* Real-world equivalent flight time, not wall-clock test duration -
                  a 20-second local session is meaningless as "20s" against a
                  2,000h TBO; scaled through the same compression factor as RUL,
                  it correctly reads as ~2 real hours of flight. */}
              Flight time: {simSecondsToRealHours(stoppedSummary.simSeconds).toFixed(1)}h (real-world equivalent)
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
            <button
              onClick={() => setStoppedSummary(null)}
              className="w-full rounded bg-tertiary py-2.5 text-[11px] font-bold uppercase tracking-[0.1em] text-black hover:brightness-110 transition-all"
            >
              Start New Simulation
            </button>
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
