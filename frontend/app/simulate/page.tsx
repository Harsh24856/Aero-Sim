"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import Sensr from "@/components/Sensr";
import SimulatorDefault, { type SimTelemetry } from "@/components/Simulator";
import Simulator912 from "@/components/Simulator_912";
import Simulator915 from "@/components/Simulator_915";
import Simulator916 from "@/components/Simulator_916";
import Meters, { type RawTelemetry, mpsToKmh } from "@/components/Meters";
import Diagnostics, { type AiResult } from "@/components/Diagnostics";
import { supabase } from "@/lib/supabase";

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

  const [throttle, setThrottle] = useState(5); // initial throttle for every engine - pilot ramps up manually from here
  const [airspeedTarget, setAirspeedTarget] = useState(30); // m/s
  const [started, setStarted] = useState(false);
  const [paused, setPaused] = useState(false);
  const [liveTelemetry, setLiveTelemetry] = useState<SimTelemetry | null>(null);
  const [aiResult, setAiResult] = useState<AiResult | null>(null);
  const [rawTelemetry, setRawTelemetry] = useState<RawTelemetry | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // main.py already merges the AI service's response into every WebSocket
  // broadcast (see main.py's simulation_loop) - this just reads that 'ai' field
  // back out, so Diagnostics can show the REAL model output, not a placeholder.
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.ai) setAiResult(data.ai);
      setRawTelemetry(data);   // full payload - all 24 raw features for Sensr
    };
    ws.onerror = () => console.log("WebSocket error - is main.py running on :8000?");
    return () => ws.close();
  }, []);

  const onTelemetryChange = useCallback((t: SimTelemetry) => setLiveTelemetry(t), []);

  // AoA comes directly from Simulator's pitch (see Simulator.tsx's formatPitch -
  // altitude changes there produce a pitch angle, which IS our AoA). Forwarded to
  // the real backend alongside altitude/throttle/airspeed, throttled to match the
  // ~10/sec rate telemetry already arrives at.
  useEffect(() => {
    if (!started || paused || !liveTelemetry) return;
    fetch(`${API}/params`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        altitude: liveTelemetry.altitude,
        throttle: throttle / 100,
        airspeed: liveTelemetry.speed,
        aoa: liveTelemetry.pitch,
      }),
    }).catch(() => {
      // Backend not running is not a reason to break the local simulator display -
      // degrade gracefully, same pattern used in ai.py's own error handling.
    });
  }, [started, paused, liveTelemetry, throttle]);

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
        fetch(`${API}/stop`, { method: "POST" }).catch(() => {});
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
    try {
      await fetch(`${API}/stop`, { method: "POST" });
    } catch {
      // backend optional - local game/gauges still get reset below regardless
    }
    setStarted(false);
    setPaused(false);
  }, []);

  // Safety net: if the user navigates away entirely (not just pausing) while the
  // simulation is running, the backend loop would otherwise keep running forever
  // with no UI left to show it or pause it.
  useEffect(() => {
    return () => {
      if (started) {
        fetch(`${API}/stop`, { method: "POST" }).catch(() => {});
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
            <ActiveSimulator
              onTelemetryChange={onTelemetryChange}
              throttle={throttle}
              onThrottleChange={setThrottle}
              airspeedTarget={airspeedTarget}
              onAirspeedTargetChange={setAirspeedTarget}
              started={started}
              paused={paused}
              onStop={onStopClick}
            />
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

        <Diagnostics ai={aiResult} />
      </div>
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
