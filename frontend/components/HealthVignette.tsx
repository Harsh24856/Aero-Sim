"use client";

// Low-health screen effect, in the style of modern AAA games (The Last of Us Part II,
// Call of Duty, Battlefield): the EDGES of the view darken and tint while the centre
// stays clear, so the pilot is warned without losing the flight picture. Deliberately
// not a full-screen flash, not a heartbeat on every frame.
//
//   health >= 80   nothing
//   80 .. 50       faint amber edge (advisory.py "caution" band)
//   < 50           red edge, stronger as health falls ("warning" band)
//   < 25           adds a slow 2.4 s breathing pulse - the only motion, and only when critical
//
// Fades over ~1.5 s in both directions so a noisy health value never strobes.
// Motion is removed entirely under prefers-reduced-motion. pointer-events: none, so
// it can never block a control.

type Props = {
  health: number | null | undefined;   // 0..100, AI composite health
  active: boolean;                     // only while flying with a live AI result
};

const CAUTION = 80;
const WARNING = 50;
const CRITICAL = 25;

export default function HealthVignette({ health, active }: Props) {
  const h = active && typeof health === "number" && Number.isFinite(health) ? health : 100;
  const level = h >= CAUTION ? 0 : h >= WARNING ? 1 : h >= CRITICAL ? 2 : 3;

  // 0 at the caution threshold, 1 at zero health - drives spread and opacity.
  const depth = Math.min(1, Math.max(0, (CAUTION - h) / CAUTION));
  const red = level >= 2;
  const rgb = red ? "176, 24, 18" : "196, 120, 20";
  const edge = red ? 0.35 + 0.4 * depth : 0.22;          // alpha at the very edge
  const clearRadius = red ? 62 - 22 * depth : 70;        // % of the view kept untouched

  return (
    <div
      aria-hidden
      className="pointer-events-none fixed inset-0 z-[90]"
      style={{
        opacity: level === 0 ? 0 : 1,
        transition: "opacity 1.5s ease, background 1.5s ease",
        background: `radial-gradient(ellipse at center, rgba(${rgb},0) ${clearRadius}%, rgba(${rgb},${(edge * 0.55).toFixed(3)}) ${Math.min(92, clearRadius + 18)}%, rgba(${rgb},${edge.toFixed(3)}) 100%)`,
        // Slight colour drain of what is behind the edge at warning and below.
        backdropFilter: level >= 2 ? `saturate(${(1 - 0.35 * depth).toFixed(2)})` : undefined,
        WebkitMaskImage: level >= 2 ? `radial-gradient(ellipse at center, transparent ${clearRadius}%, black 100%)` : undefined,
        maskImage: level >= 2 ? `radial-gradient(ellipse at center, transparent ${clearRadius}%, black 100%)` : undefined,
      }}
    >
      {level === 3 && <div className="aero-vignette-pulse absolute inset-0" style={{ background: `radial-gradient(ellipse at center, rgba(${rgb},0) 55%, rgba(${rgb},0.35) 100%)` }} />}
      <style>{`
        .aero-vignette-pulse { animation: aero-vignette-breathe 2.4s ease-in-out infinite; }
        @keyframes aero-vignette-breathe { 0%, 100% { opacity: 0.15; } 50% { opacity: 0.7; } }
        @media (prefers-reduced-motion: reduce) { .aero-vignette-pulse { animation: none; opacity: 0.4; } }
      `}</style>
    </div>
  );
}
