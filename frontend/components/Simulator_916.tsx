"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import WhitePlane from "@/components/WhitePlane";

type RidgePoint = [number, number];
type Building = { x: number; w: number; h: number };
type FlightStatus = "flying" | "landed" | "crashed";
type ControlKey = "up" | "down" | "left" | "right";

/* ───────────────────────────────────────────────────────────────
   Seeded PRNG (Mulberry32) - deterministic terrain every load
─────────────────────────────────────────────────────────────── */
function mulberry32(seed: number): () => number {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ───────────────────────────────────────────────────────────────
   Terrain generation
─────────────────────────────────────────────────────────────── */
function buildRidgePoints(seed: number, width: number, segments: number, baseline: number, minPeak: number, maxPeak: number): RidgePoint[] {
  const rand = mulberry32(seed);
  const points: RidgePoint[] = [[0, baseline - (minPeak + rand() * (maxPeak - minPeak))]];
  const step = width / segments;
  for (let i = 1; i <= segments; i++) {
    const x = Math.round(i * step);
    const prevY = points[i - 1][1];
    const maxDelta = (maxPeak - minPeak) * 0.4;
    let y = prevY + (rand() - 0.5) * maxDelta;
    const minY = baseline - maxPeak;
    const maxY = baseline - minPeak;
    y = Math.max(minY, Math.min(maxY, y));
    points.push([x, y]);
  }
  return points;
}

// Closed path for filling (baseline → crest → baseline → Z)
function catmullRomPath(points: RidgePoint[], baseline: number, totalWidth: number): string {
  const pts = points;
  let d = `M0,${baseline.toFixed(1)} L${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] || p2;
    const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
    const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
    const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
    const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
  }
  d += ` L${totalWidth.toFixed(1)},${baseline.toFixed(1)} Z`;
  return d;
}

// Open path tracing ONLY the ridge crest — no baseline closure.
// Used for snow-cap highlights and ridge-edge strokes without
// outlining the flat bottom edge of the mountain fill.
function catmullRomCrestPath(points: RidgePoint[]): string {
  const pts = points;
  let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] || p2;
    const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
    const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
    const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
    const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
  }
  return d;
}

// Returns BOTH the closed fill path and the open crest-only path
// for each tiled ridge layer.
function tiledRidgePathData(
  seed: number, width: number, segments: number,
  baseline: number, minPeak: number, maxPeak: number
): { fill: string; crest: string } {
  const pts = buildRidgePoints(seed, width, segments, baseline, minPeak, maxPeak);
  pts[pts.length - 1][1] = pts[0][1]; // seamless tile join
  const pts2 = pts.slice(1).map(([x, y]): RidgePoint => [x + width, y]);
  const all = pts.concat(pts2);
  return {
    fill: catmullRomPath(all, baseline, width * 2),
    crest: catmullRomCrestPath(all),
  };
}

function buildBuildings(seed: number, width: number, count: number): Building[] {
  const rand = mulberry32(seed);
  const rects: Building[] = [];
  let x = 20;
  for (let i = 0; i < count; i++) {
    const w = 18 + rand() * 34;
    const h = 10 + rand() * 26;
    x += 20 + rand() * 70;
    if (x > width - 40) break;
    rects.push({ x, w, h });
  }
  return rects;
}

/* ───────────────────────────────────────────────────────────────
   Tunable constants — real UAV physics envelope (meters, m/s),
   matching the backend-connected dashboards ranges established
   earlier in this project.
─────────────────────────────────────────────────────────────── */
const WORLD_W = 1200;
const SKY_H = 500;
const ALT_MIN = 0;
const ALT_MAX = 8000;       // meters
const ALT_CURVE_K = 1000;
const SPEED_MIN = 35;       // m/s (= 126 km/h) - stall-speed floor enforced during manual flight (not during auto-climb ramp-up, which starts from 0)
const SPEED_MAX = 69.4;     // m/s (= 250 km/h) - real MALE UAV ceiling: Hermes 900 max is 220 km/h
const ALT_RATE = 260;       // m / second held
const THROTTLE_RATE = 40;   // percent / second held (Sensr slider only, no longer keyboard-driven)
const SPEED_RATE = 8;       // m/s / second held - left/right speed adjustment rate
const SAFE_LANDING_SPEED = 20; // m/s (= 72 km/h) - was 35 m/s (126 km/h), FASTER than
                                // real cruise speed (112 km/h) - not a sensible landing
                                // threshold. 72 km/h is well below cruise, as landing
                                // speed should be.
const SAFE_LANDING_PITCH = 12;
const PLANE_TOP_AT_GROUND = 80.5; // moved down so the plane starts ON the runway, not slightly above it
const PLANE_TOP_AT_MAX_ALT = 11;
const HEADING = 45;
const INITIAL_ALT = 0;      // start on the ground

// Auto-climb takeoff sequence parameters
const AUTO_CLIMB_TARGET = 1000;   // meters — altitude to climb to during takeoff
const AUTO_CLIMB_DURATION = 50;   // seconds to reach target
const AUTO_CLIMB_RATE = AUTO_CLIMB_TARGET / AUTO_CLIMB_DURATION; // ~40 m/s vertical
const AUTO_CLIMB_PITCH = 6;      // degrees nose-up during climb
// A resumed flight whose saved altitude is at/below this is genuinely still on
// the ground, so the takeoff sequence below is the correct thing to run for it.
const RESUME_GROUND_EPS = 0.5;   // metres

/* ───────────────────────────────────────────────────────────────
   Formatting helpers
─────────────────────────────────────────────────────────────── */
function formatAlt(alt: number): string {
  return `${Math.round(alt).toLocaleString()} M`;
}
function formatSpeed(spdMps: number): string {
  return `${Math.round(spdMps * 3.6)} KM/H`;
}
function formatRC(vsK: number): string {
  return `${vsK >= 0 ? "+" : ""}${vsK.toFixed(2)}`;
}
function formatPitch(p: number): string {
  const r = Math.round(p);
  return `${r >= 0 ? "+" : ""}${r}° AOA`;
}

/* ───────────────────────────────────────────────────────────────
   Exported types
─────────────────────────────────────────────────────────────── */
export type SimTelemetry = {
  altitude: number;
  speed: number;
  pitch: number; // this IS our AoA — see the AOA display label above
  verticalSpeed: number;
  status: FlightStatus;
};

// Passed by /simulate when the user clicked "Continue Simulation" on a past run.
// main.py's /resume has ALREADY restored the physics twin to this exact state
// before this component ever mounts, so the simulator must adopt it verbatim
// rather than running its usual from-the-ground takeoff. Null/absent = a
// genuinely fresh flight.
export type ResumeState = {
  altitude: number;   // metres, as restored on the backend twin
  speed: number;      // m/s true airspeed
  pitch: number;      // degrees - this IS the twin's aoa, see SimTelemetry.pitch
};

export type SimulatorProps = {
  onTelemetryChange?: (t: SimTelemetry) => void;
  throttle: number;
  onThrottleChange: (v: number) => void;
  airspeedTarget: number;
  onAirspeedTargetChange: (v: number) => void;
  started: boolean;
  paused: boolean;
  onStop: () => void;
  initialState?: ResumeState | null;
  /** Mission-profile climb ceiling. Defaults to AUTO_CLIMB_TARGET so an
   *  ordinary (non-preset) launch behaves exactly as before. */
  altitudeTarget?: number;
};

/* ───────────────────────────────────────────────────────────────
   Main component
─────────────────────────────────────────────────────────────── */
function FlightApproachGame({
  onTelemetryChange,
  throttle,
  onThrottleChange,
  airspeedTarget,
  onAirspeedTargetChange,
  started,
  paused,
  onStop,
  initialState,
  altitudeTarget,
}: SimulatorProps) {
  // No ridge data needed for this theme (explicit "no hills" request) - just
  // buildings + the starfield/moon rendered directly in JSX below.
  const buildings = useMemo(() => buildBuildings(5, WORLD_W, 8), []);

  const [status, setStatus] = useState<FlightStatus>("flying");
  const [focused, setFocused] = useState(false);

  /* ── Physics refs ──────────────────────────────────────────── */
  // A resumed flight starts from the state the backend twin was ALREADY restored
  // to, not from the ground. useRef only honours its argument on the very first
  // render, which is exactly the semantics wanted here: /simulate withholds this
  // component from the tree until initialState is known, so the first render is
  // the one that carries the real values.
  const initialAltitude = initialState?.altitude ?? INITIAL_ALT;
  const altitudeRef = useRef(initialAltitude);
  const speedRef = useRef(initialState?.speed ?? 0);
  const pitchRef = useRef(initialState?.pitch ?? 0);
  const verticalSpeedRef = useRef(0);
  const statusRef = useRef<FlightStatus>("flying");
  const distanceRef = useRef(0);
  const lastTimeRef = useRef<number | null>(null);
  const keysRef = useRef({ up: false, down: false, left: false, right: false });
  // Auto-climb is the FRESH-takeoff sequence: it drives altitude from 0 up to
  // AUTO_CLIMB_TARGET, and /simulate mirrors that altitude straight back to the
  // backend via POST /params. On a resumed flight that dragged the just-restored
  // altitude/throttle/airspeed back to a ground takeoff within a second or two -
  // the single largest reason "Continue Simulation" did not actually continue.
  // Resuming from a snapshot taken ON the ground is the one case where running a
  // takeoff sequence is still correct.
  const autoClimbRef = useRef(
    initialState == null || initialState.altitude <= RESUME_GROUND_EPS
  ); // active during takeoff sequence

  // Ref-synced mirrors of externally-controlled props so the rAF
  // loop reads current values without restarting its effect.
  const throttleRef = useRef(throttle);
  const airspeedTargetRef = useRef(airspeedTarget);
  const altitudeTargetRef = useRef(altitudeTarget ?? AUTO_CLIMB_TARGET);
  const startedRef = useRef(started);
  const pausedRef = useRef(paused);
  useEffect(() => { throttleRef.current = throttle; }, [throttle]);
  useEffect(() => { airspeedTargetRef.current = airspeedTarget; }, [airspeedTarget]);
  useEffect(() => { altitudeTargetRef.current = altitudeTarget ?? AUTO_CLIMB_TARGET; }, [altitudeTarget]);
  useEffect(() => { startedRef.current = started; }, [started]);
  useEffect(() => { pausedRef.current = paused; }, [paused]);

  /* ── DOM refs ──────────────────────────────────────────────── */
  const containerRef = useRef<HTMLDivElement | null>(null);
  const farRef = useRef<HTMLDivElement | null>(null);
  const midRef = useRef<HTMLDivElement | null>(null);
  const nearRef = useRef<HTMLDivElement | null>(null);
  const buildingsRef = useRef<HTMLDivElement | null>(null);
  const dashRef = useRef<HTMLDivElement | null>(null);
  const slabRef = useRef<HTMLDivElement | null>(null);
  const planeRef = useRef<HTMLDivElement | null>(null);
  const altValueRef = useRef<HTMLSpanElement | null>(null);
  const tasValueRef = useRef<HTMLSpanElement | null>(null);
  const rcValueRef = useRef<HTMLSpanElement | null>(null);
  const pitchValueRef = useRef<HTMLSpanElement | null>(null);
  const climbBadgeRef = useRef<HTMLDivElement | null>(null);

  /* ── Restart ───────────────────────────────────────────────── */
  const restart = useCallback(() => {
    altitudeRef.current = INITIAL_ALT; // back to ground
    speedRef.current = 0;
    pitchRef.current = 0;
    verticalSpeedRef.current = 0;
    statusRef.current = "flying";
    distanceRef.current = 0;
    lastTimeRef.current = null;
    autoClimbRef.current = true; // reset takeoff sequence
    setStatus("flying");
    if (containerRef.current) containerRef.current.focus();
  }, []);

  /* ── Keyboard handlers ─────────────────────────────────────── */
  useEffect(() => {
    const isNav = (key: string) => ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(key);
    const down = (event: KeyboardEvent) => {
      if (isNav(event.key)) event.preventDefault();
      if (event.key === "ArrowUp") keysRef.current.up = true;
      if (event.key === "ArrowDown") keysRef.current.down = true;
      if (event.key === "ArrowLeft") keysRef.current.left = true;
      if (event.key === "ArrowRight") keysRef.current.right = true;
      if (event.key === "r" || event.key === "R") restart();
    };
    const up = (event: KeyboardEvent) => {
      if (event.key === "ArrowUp") keysRef.current.up = false;
      if (event.key === "ArrowDown") keysRef.current.down = false;
      if (event.key === "ArrowLeft") keysRef.current.left = false;
      if (event.key === "ArrowRight") keysRef.current.right = false;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [restart]);

  useEffect(() => {
    if (containerRef.current) containerRef.current.focus();
  }, []);

  /* ── Main animation / physics loop ─────────────────────────── */
  const lastTelemetryEmitRef = useRef(0);
  useEffect(() => {
    let raf: number;
    const tick = (now: number) => {
      const last = lastTimeRef.current ?? now;
      const dt = Math.min(0.05, (now - last) / 1000);
      lastTimeRef.current = now;

      if (statusRef.current === "flying" && startedRef.current && !pausedRef.current) {
        let alt = altitudeRef.current;

        // Speed keys — active during both auto-climb and manual flight. Reverted
        // from an earlier throttle-control experiment: left/right now adjusts
        // airspeed target directly (the same shared value Sensr's speed
        // slider controls), not throttle.
        if (keysRef.current.right || keysRef.current.left) {
          const delta = (keysRef.current.right ? 1 : 0) - (keysRef.current.left ? 1 : 0);
          const newTarget = Math.max(SPEED_MIN, Math.min(SPEED_MAX, airspeedTargetRef.current + delta * SPEED_RATE * dt));
          airspeedTargetRef.current = newTarget;
          onAirspeedTargetChange(newTarget);
        }

        if (autoClimbRef.current) {
          // ── Auto-climb phase ──────────────────────────────────
          // Altitude rises automatically at a steady rate; speed
          // ramps up to the airspeed target; pitch holds a gentle
          // nose-up angle.
          alt += AUTO_CLIMB_RATE * dt;
          if (alt >= altitudeTargetRef.current) {
            alt = altitudeTargetRef.current;
            autoClimbRef.current = false;
          }

          // Speed ramp-up (slightly slower smoothing than manual)
          const spd = speedRef.current + (airspeedTargetRef.current - speedRef.current) * Math.min(1, dt * 1.0);
          speedRef.current = spd;

          // Gentle nose-up during climb
          const newPitch = pitchRef.current + (AUTO_CLIMB_PITCH - pitchRef.current) * Math.min(1, dt * 3);
          pitchRef.current = newPitch;
        } else {
          // ── Manual flight ─────────────────────────────────────
          if (keysRef.current.up) alt += ALT_RATE * dt;
          if (keysRef.current.down) alt -= ALT_RATE * dt;
          alt = Math.max(ALT_MIN, Math.min(ALT_MAX, alt));

          // Speed tracks the externally-set airspeed target, floored at SPEED_MIN
          // once actually in manual flight - a genuine stall-speed protection, not
          // just a slider/keyboard input clamp (which only limits the TARGET, not
          // the tracked value itself).
          const spd = speedRef.current + (airspeedTargetRef.current - speedRef.current) * Math.min(1, dt * 1.5);
          speedRef.current = Math.max(SPEED_MIN, spd);

          // Pitch response
          let pitchTarget = 0;
          if (keysRef.current.up) pitchTarget = 10;
          else if (keysRef.current.down) pitchTarget = -8;
          const newPitch = pitchRef.current + (pitchTarget - pitchRef.current) * Math.min(1, dt * 5);
          pitchRef.current = newPitch;
        }

        const deltaAlt = alt - altitudeRef.current;
        const vs = dt > 0 ? deltaAlt / dt : 0;

        altitudeRef.current = alt;
        verticalSpeedRef.current = vs;
        distanceRef.current += speedRef.current * dt;

        // Ground check — SKIP during auto-climb, since starting at
        // altitude 0 would otherwise immediately trigger a false
        // "crashed" state on the very first tick.
        if (!autoClimbRef.current && alt <= 0.5) {
          altitudeRef.current = 0;
          if (speedRef.current <= SAFE_LANDING_SPEED && Math.abs(pitchRef.current) <= SAFE_LANDING_PITCH) {
            statusRef.current = "landed";
          } else {
            statusRef.current = "crashed";
          }
          speedRef.current = 0;
          setStatus(statusRef.current);
        }
      }

      /* ── Visual updates (every frame) ──────────────────────── */
      const d = distanceRef.current;
      if (farRef.current) farRef.current.style.transform = `translateX(-${(((d * 0.012) % 1) * 50).toFixed(3)}%)`;
      if (midRef.current) midRef.current.style.transform = `translateX(-${(((d * 0.022) % 1) * 50).toFixed(3)}%)`;
      if (nearRef.current) nearRef.current.style.transform = `translateX(-${(((d * 0.04) % 1) * 50).toFixed(3)}%)`;
      if (buildingsRef.current) buildingsRef.current.style.transform = `translateX(-${(((d * 0.065) % 1) * 50).toFixed(3)}%)`;
      if (dashRef.current) dashRef.current.style.backgroundPositionX = `${(-(d * 1.6) % 100).toFixed(2)}px`;
      if (slabRef.current) slabRef.current.style.backgroundPositionX = `${(-(d * 1.6) % 140).toFixed(2)}px`;

      const altCurve = altitudeRef.current / (altitudeRef.current + ALT_CURVE_K);
      const topPercent = PLANE_TOP_AT_GROUND - altCurve * (PLANE_TOP_AT_GROUND - PLANE_TOP_AT_MAX_ALT);
      if (planeRef.current) {
        planeRef.current.style.top = `${topPercent}%`;
        planeRef.current.style.transform = `translateY(-100%) rotate(${-pitchRef.current}deg)`;
      }

      if (altValueRef.current) altValueRef.current.textContent = formatAlt(altitudeRef.current);
      if (tasValueRef.current) tasValueRef.current.textContent = formatSpeed(speedRef.current);
      if (rcValueRef.current) rcValueRef.current.textContent = formatRC(verticalSpeedRef.current / 1000);
      if (pitchValueRef.current) pitchValueRef.current.textContent = formatPitch(pitchRef.current);

      // Auto-climb badge visibility
      if (climbBadgeRef.current) {
        climbBadgeRef.current.style.display =
          autoClimbRef.current && startedRef.current ? "block" : "none";
      }

      if (onTelemetryChange && now - lastTelemetryEmitRef.current > 100) {
        lastTelemetryEmitRef.current = now;
        onTelemetryChange({
          altitude: altitudeRef.current,
          speed: speedRef.current,
          pitch: pitchRef.current,
          verticalSpeed: verticalSpeedRef.current,
          status: statusRef.current,
        });
      }

      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [onTelemetryChange, onThrottleChange, onAirspeedTargetChange]);

  /* ── Touch / mouse button helpers ──────────────────────────── */
  const heldPress = (key: ControlKey) => ({
    onMouseDown: (event: React.MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      keysRef.current[key] = true;
      if (containerRef.current) containerRef.current.focus();
    },
    onMouseUp: () => (keysRef.current[key] = false),
    onMouseLeave: () => (keysRef.current[key] = false),
    onTouchStart: (event: React.TouchEvent<HTMLButtonElement>) => {
      event.preventDefault();
      keysRef.current[key] = true;
    },
    onTouchEnd: (event: React.TouchEvent<HTMLButtonElement>) => {
      event.preventDefault();
      keysRef.current[key] = false;
    },
  });

  const labelStyle = { color: "#dd9a5c", textShadow: "0 0 6px rgba(0,0,0,0.5)" };
  const valueStyle = { color: "#f8efdd", textShadow: "0 0 6px rgba(0,0,0,0.5)" };

  const initAltCurve = initialAltitude / (initialAltitude + ALT_CURVE_K);
  const initTopPercent = PLANE_TOP_AT_GROUND - initAltCurve * (PLANE_TOP_AT_GROUND - PLANE_TOP_AT_MAX_ALT);

  /* ── JSX ───────────────────────────────────────────────────── */
  return (
    <div style={{ width: "100%", height: "100%", fontFamily: "'VT323', monospace", display: "flex", flexDirection: "column", minHeight: 0 }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
        .fg-screen { box-shadow: 0 12px 34px rgba(0,0,0,0.5); }
        .fg-screen:focus-visible { outline: none; box-shadow: 0 0 0 3px #f2a955, 0 12px 34px rgba(0,0,0,0.5); }
        .fg-screen:focus { outline: none; }
        @keyframes fg-flicker { 0%,89%,100% { opacity: 1; } 91% { opacity: 0.72; } 93% { opacity: 1; } 96% { opacity: 0.85; } }
        .fg-telemetry { animation: fg-flicker 4.2s infinite; }
        @keyframes fg-climb-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.45; } }
        .fg-climb-badge { animation: fg-climb-pulse 1.8s ease-in-out infinite; }
        .fg-btn {
          font-family: 'VT323', monospace;
          font-size: 18px;
          letter-spacing: 0.05em;
          background: #23201b;
          color: #f2e4c4;
          border: 1px solid #55483a;
          border-radius: 4px;
          padding: 5px 10px;
          cursor: pointer;
          user-select: none;
        }
        .fg-btn:active { background: #3a3226; }
        .fg-btn:focus-visible { outline: 2px solid #f2a955; }
      `}</style>

      {/* ── Game canvas ──────────────────────────────────────────── */}
      <div
        ref={containerRef}
        className="fg-screen"
        tabIndex={0}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onClick={() => containerRef.current && containerRef.current.focus()}
        style={{
          position: "relative",
          width: "100%",
          flex: 1,
          minHeight: 0,
          overflow: "hidden",
          borderRadius: 6,
          border: "6px solid #1a120f",
          userSelect: "none",
          willChange: "contents",
        }}
      >
        {/* Sky gradient */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            background:
              "linear-gradient(to bottom, #050818 0%, #0a1030 25%, #131c42 48%, #1c2850 68%, #26305a 82%, #2f3a62 92%)",
          }}
        />

        {/* ── No hills for this theme (explicit request) - night sky with
            stars and a moon instead. The far/mid/near refs are still
            attached to empty parallax layers so the tick loop's existing
            style-mutation code (unchanged) has somewhere harmless to write. */}
        <div ref={farRef} style={{ position: "absolute", left: 0, top: 0, width: "200%", height: "76%", willChange: "transform" }}>
          <svg viewBox={`0 0 ${WORLD_W * 2} ${SKY_H}`} preserveAspectRatio="none" width="100%" height="100%">
            {/* Starfield - deterministic scatter via the same seeded PRNG used for terrain */}
            {Array.from({ length: 90 }).map((_, i) => {
              const rand = mulberry32(1000 + i);
              const x = rand() * WORLD_W * 2;
              const y = rand() * SKY_H * 0.65;
              const r = 0.6 + rand() * 1.4;
              const op = 0.4 + rand() * 0.6;
              return <circle key={i} cx={x} cy={y} r={r} fill="#ffffff" opacity={op} />;
            })}
          </svg>
        </div>

        <div ref={midRef} style={{ position: "absolute", left: 0, top: 0, width: "200%", height: "76%", willChange: "transform" }} />

        <div ref={nearRef} style={{ position: "absolute", left: 0, top: 0, width: "200%", height: "76%", willChange: "transform" }} />

        {/* Moon */}
        <div
          style={{
            position: "absolute", top: "9%", left: "20%", width: "7%", aspectRatio: "1",
            borderRadius: "50%",
            background: "radial-gradient(circle at 38% 38%, #f5f6fa 0%, #d8dce8 55%, #b8bfd4 85%)",
            boxShadow: "0 0 30px 8px rgba(220,225,245,0.35)",
          }}
        />

        {/* ── Buildings ──────────────────────────────────────────── */}
        <div style={{ position: "absolute", left: 0, top: "69%", width: "100%", height: "10.5%", background: "linear-gradient(to bottom, #1c2030, #12141f)", overflow: "hidden" }}>
          <div ref={buildingsRef} style={{ position: "absolute", left: 0, top: 0, width: "200%", height: "100%", willChange: "transform" }}>
            <svg viewBox={`0 0 ${WORLD_W * 2} 100`} preserveAspectRatio="none" width="100%" height="100%">
              {[0, WORLD_W].map((offset) =>
                buildings.map((b, i) => (
                  <rect key={`${offset}-${i}`} x={b.x + offset} y={100 - b.h} width={b.w} height={b.h} fill="rgba(15,15,25,0.9)" />
                ))
              )}
            </svg>
          </div>
        </div>

        {/* ── Road / runway ──────────────────────────────────────── */}
        <div style={{ position: "absolute", left: 0, top: "79.5%", width: "100%", height: "12.5%", background: "linear-gradient(to bottom, #4c4f51, #2a2c2d)", borderTop: "3px solid #dcd6c6", borderBottom: "3px solid #dcd6c6", overflow: "hidden" }}>
          <div ref={slabRef} style={{ position: "absolute", inset: 0, backgroundImage: "repeating-linear-gradient(to right, rgba(255,255,255,0.07) 0 2px, transparent 2px 140px)", willChange: "background-position" }} />
          <div ref={dashRef} style={{ position: "absolute", left: 0, right: 0, top: "44%", height: "10%", backgroundImage: "repeating-linear-gradient(to right, #ece5d3 0 46px, transparent 46px 100px)", willChange: "background-position" }} />
        </div>

        {/* ── Pitch / AoA strip ──────────────────────────────────── */}
        <div style={{ position: "absolute", left: 0, top: "92%", width: "100%", height: "8%", background: "rgba(48,42,26,0.62)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 6 }}>
          <span ref={pitchValueRef} style={{ fontSize: "clamp(14px,3vw,22px)", letterSpacing: "0.18em", color: "#efe0b8" }}>
            {formatPitch(0)}
          </span>
        </div>

        {/* ── UAV: real 3D model (BlackoutPlane), not an SVG silhouette.
            interactive=false disables its auto-rotate/drag controls, since THIS
            wrapper controls orientation via pitch (rotate) to reflect actual
            flight attitude - an uncontrolled auto-rotate would fight against that.
            The inner scaleX(-1) flips the model to face right, matching the
            world's scroll direction (mountains move left = the UAV flies right).
            If the model's default orientation turns out to already face right,
            this flip should simply be removed. */}
        <div
          ref={planeRef}
          style={{
            position: "absolute",
            left: "2%",
            width: "22%",
            aspectRatio: "21 / 8",
            top: `${initTopPercent}%`,
            transform: "translateY(-100%) rotate(0deg)",
            transformOrigin: "58% 68%",
            zIndex: 5,
            pointerEvents: "none",
            filter: status === "crashed" ? "drop-shadow(0 0 8px rgba(255,70,40,0.85))" : "none",
          }}
        >
          {/* scaleX(-1) flip removed - confirmed wrong via screenshot, natural
              orientation faces the correct way. cameraPosition=[5,0.4,0] views
              the model from the SIDE (along X) instead of R3F's default head-on
              view down -Z, which was showing the nose instead of the wing
              profile - also confirmed via screenshot. */}
          <div style={{ width: "100%", height: "100%" }}>
            <WhitePlane />
          </div>
        </div>

        {/* ── HUD telemetry overlays ─────────────────────────────── */}
        <div style={{ position: "absolute", top: "4%", left: "3%", zIndex: 10, lineHeight: 1.3 }}>
          <div style={{ fontSize: "clamp(14px,2.6vw,20px)" }}>
            <span style={labelStyle}>HDG </span>
            <span style={valueStyle}>{String(HEADING).padStart(3, "0")}°</span>
          </div>
          <div style={{ fontSize: "clamp(14px,2.6vw,20px)" }}>
            <span style={labelStyle}>ALT </span>
            <span ref={altValueRef} style={valueStyle}>{formatAlt(initialAltitude)}</span>
          </div>
        </div>

        <div style={{ position: "absolute", top: "4%", right: "3%", zIndex: 10, lineHeight: 1.3, textAlign: "right" }}>
          <div style={{ fontSize: "clamp(14px,2.6vw,20px)" }}>
            <span style={labelStyle}>TAS </span>
            <span ref={tasValueRef} style={valueStyle}>{formatSpeed(0)}</span>
          </div>
          <div style={{ fontSize: "clamp(14px,2.6vw,20px)" }}>
            <span style={labelStyle}>R/C </span>
            <span ref={rcValueRef} style={valueStyle}>{formatRC(0)}</span>
          </div>
        </div>

        <div className="fg-telemetry" style={{ position: "absolute", top: "2%", left: "50%", transform: "translateX(-50%)", zIndex: 10, background: "rgba(32,12,10,0.78)", border: "1px solid rgba(255,190,150,0.35)", padding: "2px 14px", fontSize: "clamp(12px,2.2vw,17px)", letterSpacing: "0.22em", color: "#f3c9b0", whiteSpace: "nowrap" }}>
          LIVE TELEMETRY FEED
        </div>

        {/* Auto-climb indicator — shown during takeoff sequence */}
        <div
          ref={climbBadgeRef}
          className="fg-climb-badge"
          style={{
            display: "none",
            position: "absolute",
            bottom: "12%",
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 21,
            background: "rgba(0,0,0,0.55)",
            border: "1px solid rgba(100,220,140,0.45)",
            padding: "3px 14px",
            fontSize: "clamp(12px,2.2vw,16px)",
            letterSpacing: "0.14em",
            color: "#a8f0c0",
            whiteSpace: "nowrap",
            pointerEvents: "none",
          }}
        >
          AUTO CLIMB · {AUTO_CLIMB_TARGET.toLocaleString()} M
        </div>

        {/* CRT scanlines */}
        <div style={{ position: "absolute", inset: 0, zIndex: 20, pointerEvents: "none", background: "repeating-linear-gradient(to bottom, rgba(0,0,0,0.12) 0px, rgba(0,0,0,0.12) 1px, transparent 2px, transparent 4px)", mixBlendMode: "multiply" }} />
        {/* Vignette */}
        <div style={{ position: "absolute", inset: 0, zIndex: 20, pointerEvents: "none", boxShadow: "inset 0 0 90px rgba(0,0,0,0.55)" }} />

        {/* ── Status overlays ────────────────────────────────────── */}
        {!started && status === "flying" && (
          <div style={{ position: "absolute", inset: 0, zIndex: 22, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.45)" }}>
            <div style={{ background: "rgba(0,0,0,0.6)", color: "#f8efdd", fontSize: "clamp(14px,2.6vw,20px)", letterSpacing: "0.12em", padding: "10px 20px", borderRadius: 4, border: "1px solid rgba(255,190,150,0.35)" }}>
              WAITING - CLICK START SIMULATION
            </div>
          </div>
        )}

        {started && paused && status === "flying" && (
          <div style={{ position: "absolute", inset: 0, zIndex: 22, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.5)" }}>
            <div style={{ background: "rgba(0,0,0,0.65)", color: "#f2c98a", fontSize: "clamp(18px,3.4vw,28px)", letterSpacing: "0.2em", padding: "12px 24px", borderRadius: 4, border: "1px solid rgba(255,190,150,0.45)" }}>
              PAUSED
            </div>
          </div>
        )}

        {started && !paused && !focused && status === "flying" && (
          <div style={{ position: "absolute", bottom: "10%", left: "50%", transform: "translateX(-50%)", zIndex: 21, background: "rgba(0,0,0,0.5)", color: "#f8efdd", fontSize: "clamp(12px,2.2vw,16px)", letterSpacing: "0.1em", padding: "4px 12px", borderRadius: 4, pointerEvents: "none" }}>
            CLICK TO FOCUS · ↑↓ ALTITUDE · ←→ SPEED
          </div>
        )}

        {status !== "flying" && (
          <div style={{ position: "absolute", inset: 0, zIndex: 25, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 14, background: status === "crashed" ? "rgba(110,15,10,0.4)" : "rgba(20,60,30,0.28)" }}>
            <div style={{ fontSize: "clamp(26px,6vw,46px)", letterSpacing: "0.2em", color: status === "crashed" ? "#ff7a54" : "#c8f2cf", textShadow: "0 0 14px rgba(0,0,0,0.7)" }}>
              {status === "crashed" ? "CRASHED" : "SAFE LANDING"}
            </div>
            <button className="fg-btn" onClick={restart}>RESTART (R)</button>
          </div>
        )}
      </div>

      {/* ── Control buttons ──────────────────────────────────────── */}
      <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 8, color: "#8a7f6a", fontSize: 13, flexShrink: 0 }}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button className="fg-btn" {...heldPress("down")}>▼ ALT</button>
          <button className="fg-btn" {...heldPress("up")}>▲ ALT</button>
          <button className="fg-btn" {...heldPress("left")}>◀ SPD</button>
          <button className="fg-btn" {...heldPress("right")}>SPD ▶</button>
          <button className="fg-btn" onClick={restart}>RESTART</button>
          <button className="fg-btn" style={{ background: "#5a2020", borderColor: "#7a3030" }} onClick={onStop}>STOP SIMULATION</button>
        </div>
        <div>Click the display, then ↑↓ = altitude, ←→ = speed. Land under {Math.round(SAFE_LANDING_SPEED * 3.6)} km/h, gently.</div>
      </div>
    </div>
  );
}

export default FlightApproachGame;