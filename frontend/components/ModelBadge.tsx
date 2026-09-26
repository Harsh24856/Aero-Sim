import type { ModelVersion } from "@/lib/timeScale";

// Marks which physics generation a run was recorded on. Legacy (v2) runs use the
// compressed simulated timescale and must not be compared number-for-number with
// v3 runs, whose RUL is in real engine hours.
export default function ModelBadge({ version }: { version: ModelVersion }) {
  if (version === "v4") {
    return (
      <span
        title="Physics v4 - component-level faults, on-board healthy twin, RUL in real engine hours; engine hours advance on the run's life scale"
        className="ml-2 inline-block rounded border border-tertiary/60 px-1.5 py-0.5 align-middle text-[9px] font-bold uppercase tracking-[0.1em] text-tertiary"
      >
        v4
      </span>
    );
  }
  if (version === "v3") {
    return (
      <span
        title="Physics v3 - RUL in real engine hours against TBO"
        className="ml-2 inline-block rounded border border-primary/40 px-1.5 py-0.5 align-middle text-[9px] font-bold uppercase tracking-[0.1em] text-primary"
      >
        v3
      </span>
    );
  }
  return (
    <span
      title="Legacy physics v2 run - RUL on the compressed simulated timescale"
      className="ml-2 inline-block rounded border border-outline-variant/50 px-1.5 py-0.5 align-middle text-[9px] font-bold uppercase tracking-[0.1em] text-on-surface-variant"
    >
      Legacy
    </span>
  );
}
