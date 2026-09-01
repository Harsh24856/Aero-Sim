"use client";

import { type RawTelemetry, SENSOR_FIELDS } from "@/components/Meters";

export type SensrProps = {
  rawTelemetry?: RawTelemetry | null;
};

// All 24 raw sensor readouts panel — the instrument-readout sidebar.
// Flight controls (throttle, speed target, start/pause) now live in Meters.tsx.
export default function Sensr({
  rawTelemetry = null,
}: SensrProps) {
  return (
    <aside className="panel-shell flex h-full min-h-0 flex-col overflow-hidden p-2.5 md:p-3.5">
      <PanelHeading>Sensors</PanelHeading>

      <div className="mt-3 min-h-0 flex-1 overflow-y-auto">
        <div className="mb-1.5 text-[8px] uppercase tracking-[0.12em] text-[#927363] md:text-[10px]">
          All Sensors {rawTelemetry ? "" : "(waiting for data)"}
        </div>
        <div className="grid grid-cols-2 gap-1.5 pr-1">
          {SENSOR_FIELDS.map(({ key, label, unit, decimals = 1, convert }) => {
            const raw = rawTelemetry?.[key];
            const value = typeof raw === "number" && convert ? convert(raw) : raw;
            return (
              <div key={key} className="border border-[#352722] bg-[#0d0e0d] px-2 py-1.5">
                <div className="text-[7px] uppercase tracking-[0.08em] text-[#bca18e] md:text-[8px]">{label}</div>
                <div className="mt-0.5 text-[10px] text-[#efe0d5] md:text-[11px]">
                  {typeof value === "number" ? value.toFixed(decimals) : "--"}
                  {unit && <span className="ml-1 text-[#aa8f7f]">{unit}</span>}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}

function PanelHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="panel-heading flex items-center justify-between">
      {children} <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#ff722f] shadow-[0_0_7px_#ff5b1d]" />
    </h2>
  );
}
