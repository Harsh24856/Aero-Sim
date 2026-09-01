"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Navbar from "@/components/Navbar";
import Sensr from "@/components/Sensr";
import Simulator, { type SimTelemetry } from "@/components/Simulator";
import Meters, { type RawTelemetry, mpsToKmh } from "@/components/Meters";
import Diagnostics, { type AiResult } from "@/components/Diagnostics";

const API = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws";

export default function SimulatePage() {
  const [throttle, setThrottle] = useState(75);
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

  const onStart = useCallback(async () => {
    setStarted(true);
    try {
      await fetch(`${API}/start`, { method: "POST" });
    } catch {
      // backend optional for the local game/gauges to work
    }
  }, []);

  // Pause must ACTUALLY stop the backend's physics loop, not just the frontend's
  // own game/display state - main.py's simulation_loop() runs independently once
  // started, and keeps calling ai.py every simulated second FOREVER until /stop is
  // called, regardless of what the frontend UI shows. Without this, "pausing" only
  // stopped the local game while the backend kept silently hammering the AI service.
  const onTogglePause = useCallback(async () => {
    setPaused((p) => {
      const next = !p;
      if (next) {
        fetch(`${API}/stop`, { method: "POST" }).catch(() => {});
      } else {
        fetch(`${API}/start`, { method: "POST" }).catch(() => {});
      }
      return next;
    });
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
            <Simulator
              onTelemetryChange={onTelemetryChange}
              throttle={throttle}
              onThrottleChange={setThrottle}
              airspeedTarget={airspeedTarget}
              onAirspeedTargetChange={setAirspeedTarget}
              started={started}
              paused={paused}
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
            />
          </div>
        </div>

        <Diagnostics ai={aiResult} />
      </div>
    </>
  );
}
