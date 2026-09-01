"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import EngineViewer from "@/components/EngineViewer";
import { Zap, Cog, AlertTriangle } from "lucide-react";

const API = "http://localhost:8000";

// Real specs for all 4 selectable engines - computed from the same ENGINE_CONFIGS
// data that drives physics.py (values match the JSON spec sheet exactly). Only
// Rotax_914_ULF has a real 3D model (uav-engine-914.glb) and AI predictions -
// the other 3 are genuinely simulated with correct, different physics, but the AI
// was trained exclusively on 914 data and would not give meaningful output for them.
const ENGINES = [
  {
    id: "Rotax_914_ULF",
    name: "ROTAX 914 UL/F",
    tagline: "Turbocharged 4-stroke piston engine - the real digital twin this simulator is built around.",
    maxRpm: "5,800",
    maxPower: "84 kW",
    config: "4-CYL TURBO",
    aiValid: true,
    hasModel: true,
  },
  {
    id: "Rotax_912_ULS",
    name: "ROTAX 912 ULS",
    tagline: "Naturally-aspirated workhorse - lower power, proven reliability, no turbo complexity.",
    maxRpm: "5,800",
    maxPower: "74 kW",
    config: "4-CYL N/A",
    aiValid: false,
    hasModel: false,
  },
  {
    id: "Rotax_915_iS",
    name: "ROTAX 915 iS",
    tagline: "Fuel-injected, higher boost - significantly more power than the 914 at the same RPM.",
    maxRpm: "5,800",
    maxPower: "104 kW",
    config: "4-CYL TURBO iS",
    aiValid: false,
    hasModel: false,
  },
  {
    id: "Rotax_916_iS",
    name: "ROTAX 916 iS",
    tagline: "The most powerful in the lineup - highest torque and power output across the whole range.",
    maxRpm: "5,800",
    maxPower: "117 kW",
    config: "4-CYL TURBO iS",
    aiValid: false,
    hasModel: false,
  },
];

export default function EnginePage() {
  const router = useRouter();
  const [selecting, setSelecting] = useState<string | null>(null);

  const selectEngine = async (engineId: string) => {
    setSelecting(engineId);
    try {
      await fetch(`${API}/select_engine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine_model: engineId }),
      });
    } catch {
      // Backend down is not a reason to block navigation - /simulate handles a
      // missing backend gracefully already (shows "waiting for data" states).
    }
    router.push("/simulate");
  };

  return (
    <>
      <Navbar />

      <main className="flex-grow pt-16 relative">
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />

        <div className="relative z-10 max-w-7xl mx-auto px-4 md:px-6 py-12 md:py-16">
          <header className="mb-10 text-center md:text-left">
            <h1 className="font-headline-display text-[30px] md:text-[40px] leading-[1.15] font-bold text-primary uppercase tracking-tight mb-2">
              SELECT PROPULSION SYSTEM
            </h1>
            <p className="text-tertiary/80 text-[13px] tracking-[0.15em] uppercase font-mono">
              Configure your UAV for mission-specific performance parameters.
            </p>
          </header>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {ENGINES.map((engine) => (
              <article
                key={engine.id}
                className={`bg-surface/80 backdrop-blur-xl border rounded-lg flex flex-col overflow-hidden transition-all duration-300 hover:-translate-y-1 ${
                  engine.aiValid
                    ? "border-tertiary/30 hover:border-tertiary hover:shadow-[2px_2px_0px_#FF9100]"
                    : "border-outline-variant/40 hover:border-outline-variant"
                }`}
              >
                <div className="relative">
                  <div className={`absolute top-3 right-3 z-10 px-2 py-1 text-[10px] tracking-[0.1em] font-bold uppercase rounded flex items-center gap-1 ${
                    engine.aiValid
                      ? "bg-black/90 text-tertiary border border-tertiary/50"
                      : "bg-black/90 text-on-surface-variant border border-on-surface-variant/40"
                  }`}>
                    <span className={`w-1.5 h-1.5 rounded-full ${engine.aiValid ? "bg-tertiary animate-pulse" : "bg-on-surface-variant/60"}`} />
                    {engine.aiValid ? "AI READY" : "SIM ONLY"}
                  </div>
                  <div className="h-44 border-b border-outline-variant/30 bg-surface-container-high/40">
                    {engine.hasModel ? (
                      <EngineViewer />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center gap-2 text-on-surface-variant/40">
                        <Cog size={40} />
                        <span className="text-[10px] uppercase tracking-[0.1em]">No 3D model yet</span>
                      </div>
                    )}
                  </div>
                </div>
                <div className="p-5 flex-grow flex flex-col">
                  <h2 className={`font-headline-display text-lg mb-2 uppercase ${engine.aiValid ? "text-tertiary" : "text-on-surface-variant"}`}>
                    {engine.name}
                  </h2>
                  <p className="text-on-surface-variant/80 text-xs mb-4 flex-grow">
                    {engine.tagline}
                  </p>
                  <div className="space-y-2 mb-4 text-[12px] font-mono">
                    <div className="flex justify-between border-b border-outline-variant/20 pb-1.5">
                      <span className="text-on-surface-variant/60">MAX RPM</span>
                      <span className="text-on-surface-variant font-bold">{engine.maxRpm}</span>
                    </div>
                    <div className="flex justify-between border-b border-outline-variant/20 pb-1.5">
                      <span className="text-on-surface-variant/60">MAX POWER</span>
                      <span className="text-on-surface-variant font-bold">{engine.maxPower}</span>
                    </div>
                    <div className="flex justify-between pb-1.5">
                      <span className="text-on-surface-variant/60">CONFIG</span>
                      <span className="text-on-surface-variant font-bold">{engine.config}</span>
                    </div>
                  </div>

                  {!engine.aiValid && (
                    <div className="mb-4 flex items-start gap-1.5 text-[10px] text-on-surface-variant/60">
                      <AlertTriangle size={12} className="mt-0.5 shrink-0" />
                      <span>Real physics simulation, but AI predictions are trained only on the 914 and will not be meaningful here.</span>
                    </div>
                  )}

                  <button
                    onClick={() => selectEngine(engine.id)}
                    disabled={selecting !== null}
                    className={`w-full py-2.5 text-[11px] tracking-[0.1em] font-bold uppercase rounded flex items-center justify-center gap-2 mt-auto transition-all disabled:opacity-50 ${
                      engine.aiValid
                        ? "bg-tertiary text-black hover:brightness-110"
                        : "border border-outline-variant/50 text-on-surface-variant hover:border-tertiary hover:text-tertiary"
                    }`}
                  >
                    <Zap size={14} /> {selecting === engine.id ? "SELECTING..." : "SELECT ENGINE"}
                  </button>
                </div>
              </article>
            ))}
          </div>
        </div>
      </main>

      <footer className="bg-surface-container-lowest w-full border-t border-outline-variant/30 z-10 relative">
        <div className="flex flex-col md:flex-row justify-between items-center px-6 py-6 gap-4 w-full">
          <div className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant text-center md:text-left">
            &copy; 2026 AERO-SIM AEROSPACE. ALL SYSTEMS NOMINAL.
          </div>
          <div className="flex flex-wrap justify-center gap-4">
            <a className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant hover:text-tertiary transition-colors" href="#">Terms of Flight</a>
            <a className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant hover:text-tertiary transition-colors" href="#">Privacy Policy</a>
            <a className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant hover:text-tertiary transition-colors" href="#">API Docs</a>
          </div>
        </div>
      </footer>
    </>
  );
}
