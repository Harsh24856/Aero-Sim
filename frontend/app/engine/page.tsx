"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import EngineViewer from "@/components/EngineViewer";
import EngineViewer912 from "@/components/EngineViewer912";
import EngineViewer915 from "@/components/EngineViewer915";
import EngineViewer916 from "@/components/EngineViewer916";
import { Zap, FileText, ChevronRight, Gauge, Scale, X } from "lucide-react";

const API = "http://localhost:8000";

const ENGINES = [
  {
    id: "Rotax_914_ULF",
    code: "01 / T-BASE",
    name: "ROTAX 914 UL/F",
    role: "Turbocharged Altitude Standard",
    tag: "TURBOCHARGED",
    tagline: "Turbocharged 4-stroke boxer with automatic TCU wastegate — the digital twin this simulator is built around.",
    maxRpm: "5,800 RPM",
    maxPower: "84.8 kW / 115 HP",
    maxTorque: "144 Nm",
    dryMass: "64.0 kg",
    powerPct: 71.9,
    induction: "Turbo + TCU Wastegate",
    ceiling: "16,000 ft critical altitude",
    aiValid: true,
    Viewer: EngineViewer,
  },
  {
    id: "Rotax_912_ULS",
    code: "02 / NA-LGT",
    name: "ROTAX 912 ULS",
    role: "Naturally Aspirated Benchmark",
    tag: "NAT. ASPIRATED",
    tagline: "Lightweight workhorse — no turbo complexity, zero lag, and a simple dual-carburetor induction system.",
    maxRpm: "5,800 RPM",
    maxPower: "73.5 kW / 100 HP",
    maxTorque: "128 Nm",
    dryMass: "56.6 kg",
    powerPct: 62.5,
    induction: "Naturally Aspirated",
    ceiling: "Air-density limited ~10,000 ft",
    aiValid: true,
    Viewer: EngineViewer912,
  },
  {
    id: "Rotax_915_iS",
    code: "03 / EFI-ADV",
    name: "ROTAX 915 iS",
    role: "Turbocharged + Intercooler + EFI",
    tag: "TURBO + EFI",
    tagline: "Intercooled turbocharging with redundant dual-channel EFI — +41% over 912, full power to 15,000 ft.",
    maxRpm: "5,800 RPM",
    maxPower: "104 kW / 141 HP",
    maxTorque: "172 Nm",
    dryMass: "82.2 kg",
    powerPct: 88.1,
    induction: "Turbo + Air-to-Air Intercooler",
    ceiling: "Takeoff power to 15,000 ft",
    aiValid: true,
    Viewer: EngineViewer915,
  },
  {
    id: "Rotax_916_iS",
    code: "04 / APEX",
    name: "ROTAX 916 iS",
    role: "Flagship High-Performance Apex",
    tag: "APEX TURBO + EFI",
    tagline: "Pinnacle of Rotax aerospace — 160 HP, 187 Nm peak torque, and full dual FADEC engine management.",
    maxRpm: "5,800 RPM",
    maxPower: "117 kW / 160 HP",
    maxTorque: "187 Nm",
    dryMass: "85.8 kg",
    powerPct: 100.0,
    induction: "High-Boost Turbo + FADEC EFI",
    ceiling: "Takeoff power to 15,000 ft",
    aiValid: true,
    Viewer: EngineViewer916,
  },
];

export default function EnginePage() {
  const router = useRouter();
  const [selecting, setSelecting] = useState<string | null>(null);
  const [compareOpen, setCompareOpen] = useState(false);

  const selectEngine = async (engineId: string) => {
    setSelecting(engineId);
    try {
      await fetch(`${API}/select_engine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine_model: engineId }),
      });
    } catch {
      // Backend down is handled gracefully in /simulate
    }
    router.push(`/simulate?engine=${engineId}`);
  };

  return (
    <>
      <Navbar />

      <main className="flex-grow pt-16 relative min-h-screen">
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />

        <div className="relative z-10 max-w-7xl mx-auto px-4 md:px-6 py-10">

          {/* Header */}
          <header className="mb-8">
            <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-5">
              <div>
                <div className="mb-2 flex items-center gap-3">
                  <span className="h-px w-8 bg-tertiary" />
                  <span className="font-mono text-[10px] font-bold tracking-[0.24em] text-tertiary uppercase">
                    PROPULSION CATALOGUE // REV 4.2
                  </span>
                </div>
                <h1 className="font-headline-display text-[28px] md:text-[38px] font-bold text-primary uppercase tracking-tight leading-tight">
                  SELECT PROPULSION SYSTEM
                </h1>
                <p className="mt-1.5 text-on-surface-variant text-[13px] max-w-xl leading-relaxed">
                  Four certified digital twin engines. Select one to configure your UAV airframe and open the live cockpit simulator.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3 font-mono text-[10px] uppercase tracking-[0.14em]">
                <button
                  onClick={() => setCompareOpen(true)}
                  className="flex items-center gap-2 border border-outline-variant/50 bg-surface-container-highest/60 text-on-surface-variant px-3.5 py-2 rounded hover:border-tertiary hover:text-tertiary transition-colors"
                >
                  <Scale size={13} />
                  <span>Compare All</span>
                </button>
                <Link
                  href="/engine_info"
                  className="flex items-center gap-2 border border-tertiary/40 bg-tertiary/10 text-tertiary px-3.5 py-2 rounded font-bold hover:bg-tertiary hover:text-black transition-all shadow-[1px_1px_0px_#000000]"
                >
                  <FileText size={13} />
                  <span>Technical Report</span>
                </Link>
                <div className="flex items-center gap-2 text-on-surface-variant/60 border border-outline-variant/30 px-3 py-2 rounded bg-black/30">
                  <span className="h-1.5 w-1.5 rounded-full bg-tertiary animate-pulse shadow-[0_0_8px_#ff9100]" />
                  4 AI twins ready
                </div>
              </div>
            </div>
          </header>

          {/* Engine Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
            {ENGINES.map((engine, index) => {
              const isSelected = selecting === engine.id;
              const isOtherSelecting = selecting !== null && !isSelected;

              return (
                <article
                  key={engine.id}
                  className={`group relative flex flex-col overflow-hidden border bg-surface/80 backdrop-blur-xl rounded-lg transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[4px_4px_0px_#000000] shadow-[2px_2px_0px_#000000] ${
                    isSelected
                      ? "border-tertiary shadow-[4px_4px_0px_#000000]"
                      : "border-outline-variant/30 hover:border-tertiary/50"
                  } ${isOtherSelecting ? "opacity-40 pointer-events-none" : ""}`}
                >
                  {/* Top accent line */}
                  <div className={`absolute inset-x-0 top-0 h-px ${isSelected ? "bg-tertiary" : "bg-gradient-to-r from-transparent via-tertiary/50 to-transparent"}`} />

                  {/* Card header meta */}
                  <div className="flex items-center justify-between px-4 pt-3.5 pb-2 border-b border-outline-variant/20">
                    <span className="font-mono text-[10px] font-bold tracking-[0.16em] text-on-surface-variant/50 uppercase">
                      {engine.code}
                    </span>
                    <span className="font-mono text-[9px] font-bold tracking-[0.12em] uppercase text-tertiary bg-tertiary/10 border border-tertiary/30 px-2 py-0.5 rounded flex items-center gap-1">
                      <span className="h-1 w-1 rounded-full bg-tertiary animate-pulse" />
                      AI READY
                    </span>
                  </div>

                  {/* 3D Viewer */}
                  <div className="relative h-44 bg-[radial-gradient(ellipse_at_center,rgba(103,51,22,0.3),transparent_65%),linear-gradient(145deg,#18130f,#0b0c0c)] border-b border-outline-variant/20 overflow-hidden">
                    <engine.Viewer />
                    <span className="pointer-events-none absolute bottom-2 left-3 font-mono text-[9px] uppercase tracking-wider text-on-surface-variant/40 opacity-0 group-hover:opacity-100 transition-opacity">
                      drag to rotate
                    </span>
                  </div>

                  {/* Body */}
                  <div className="flex flex-col flex-grow p-4">
                    {/* Engine name */}
                    <div className="mb-3">
                      <p className="font-mono text-[9px] font-bold tracking-[0.18em] text-on-surface-variant/45 uppercase mb-0.5">
                        ROTAX // {engine.tag}
                      </p>
                      <h2 className="font-headline-display text-xl font-bold tracking-tight uppercase text-primary group-hover:text-tertiary transition-colors">
                        {engine.name}
                      </h2>
                      <p className="font-mono text-[10px] text-on-surface-variant/60 mt-0.5">{engine.role}</p>
                    </div>

                    {/* Power gauge */}
                    <div className="mb-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-mono text-[9px] uppercase tracking-wider text-on-surface-variant/50 flex items-center gap-1">
                          <Gauge size={10} className="text-tertiary" /> Power Output
                        </span>
                        <span className="font-mono text-[11px] font-bold text-white">{engine.maxPower}</span>
                      </div>
                      <div className="w-full h-1 bg-outline-variant/30 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-tertiary to-amber-300 rounded-full"
                          style={{ width: `${engine.powerPct}%` }}
                        />
                      </div>
                      <div className="flex justify-between mt-0.5 font-mono text-[8px] text-on-surface-variant/40">
                        <span>{engine.maxRpm}</span>
                        <span>{engine.powerPct.toFixed(0)}% of lineup max</span>
                      </div>
                    </div>

                    {/* Tagline */}
                    <p className="text-[12px] leading-relaxed text-on-surface-variant/80 mb-4 min-h-[3.5rem]">
                      {engine.tagline}
                    </p>

                    {/* Spec row */}
                    <div className="grid grid-cols-3 border-y border-outline-variant/20 py-2.5 mb-4 font-mono">
                      <div className="border-r border-outline-variant/20 pr-2">
                        <span className="block text-[8px] tracking-[0.12em] text-on-surface-variant/50 uppercase">Torque</span>
                        <span className="block text-[11px] font-bold text-white mt-0.5">{engine.maxTorque}</span>
                      </div>
                      <div className="border-r border-outline-variant/20 px-2">
                        <span className="block text-[8px] tracking-[0.12em] text-on-surface-variant/50 uppercase">Dry Mass</span>
                        <span className="block text-[11px] font-bold text-white mt-0.5">{engine.dryMass}</span>
                      </div>
                      <div className="pl-2">
                        <span className="block text-[8px] tracking-[0.12em] text-on-surface-variant/50 uppercase">Ceiling</span>
                        <span className="block text-[10px] font-bold text-white mt-0.5 truncate leading-tight" title={engine.ceiling}>
                          {engine.ceiling.split(" ")[0]} {engine.ceiling.split(" ")[1]}
                        </span>
                      </div>
                    </div>

                    {/* CTA */}
                    <div className="mt-auto">
                      <button
                        onClick={() => selectEngine(engine.id)}
                        disabled={selecting !== null}
                        className={`w-full flex items-center justify-center gap-2 py-2.5 text-[11px] font-bold uppercase tracking-[0.14em] rounded transition-all disabled:opacity-50 ${
                          isSelected
                            ? "bg-tertiary text-black shadow-[2px_2px_0px_#000000]"
                            : "bg-tertiary text-black shadow-[2px_2px_0px_#000000] hover:-translate-y-0.5 hover:shadow-[4px_4px_0px_#000000] active:translate-y-0 active:shadow-[1px_1px_0px_#000000]"
                        }`}
                      >
                        {isSelected ? (
                          <>
                            <span className="h-2 w-2 rounded-full bg-black animate-ping" />
                            <span>INITIALIZING...</span>
                          </>
                        ) : (
                          <>
                            <Zap size={13} className="fill-black" />
                            <span>SELECT ENGINE</span>
                            <ChevronRight size={13} className="group-hover:translate-x-0.5 transition-transform" />
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>

          {/* Quick actions row */}
          <div className="mt-8 flex flex-col sm:flex-row items-center justify-between gap-4 p-4 bg-surface/80 border border-outline-variant/30 rounded-lg shadow-[2px_2px_0px_#000000]">
            <div className="flex items-center gap-3 font-mono text-[11px] text-on-surface-variant uppercase tracking-wider">
              <span className="h-2 w-2 rounded-full bg-tertiary animate-pulse shadow-[0_0_8px_#ff9100]" />
              <span>Physics Engine: <span className="text-primary font-bold">100 Hz Kinematics Active</span></span>
              <span className="hidden md:inline text-outline-variant">|</span>
              <span className="hidden md:inline">Neural Surrogates: <span className="text-primary font-bold">4 Checkpoints Loaded</span></span>
            </div>
            <button
              onClick={() => setCompareOpen(true)}
              className="flex items-center gap-2 font-mono text-[11px] text-tertiary hover:underline uppercase tracking-wider"
            >
              <Scale size={13} />
              Compare all 4 engines side-by-side →
            </button>
          </div>
        </div>
      </main>

      {/* Footer — matches home page style */}
      <footer className="bg-surface-container-lowest w-full border-t border-outline-variant/30 z-10 relative">
        <div className="flex flex-col md:flex-row justify-between items-center px-6 py-6 gap-4 w-full">
          <div className="text-[11px] tracking-[0.1em] font-bold uppercase text-tertiary text-center md:text-left flex flex-col md:flex-row gap-2 md:gap-4 items-center font-mono">
            <span>&copy; 2026 AERO-SIM PRECISION SYSTEMS.</span>
            <span className="hidden md:inline text-outline-variant">|</span>
            <span className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-tertiary animate-pulse" />
              ALL PROPULSION SYSTEMS NOMINAL
            </span>
          </div>
          <div className="flex flex-wrap justify-center gap-4">
            <Link className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant hover:text-tertiary transition-colors" href="/engine_info">Spec Sheets</Link>
            <Link className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant hover:text-tertiary transition-colors" href="/simulate">Live Cockpit</Link>
            <Link className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant hover:text-tertiary transition-colors" href="/telemetry">Telemetry Records</Link>
            <Link className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant hover:text-tertiary transition-colors" href="/about_us">About Us</Link>
          </div>
        </div>
      </footer>

      {/* COMPARE MODAL */}
      {compareOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm overflow-y-auto"
          onClick={() => setCompareOpen(false)}
        >
          <div
            className="relative w-full max-w-5xl bg-surface border border-outline-variant/40 rounded-lg shadow-[4px_4px_0px_#000000] text-white my-8 overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-outline-variant/30 bg-surface-container-highest/40">
              <div>
                <div className="font-mono text-[10px] text-tertiary font-bold tracking-[0.2em] uppercase mb-0.5 flex items-center gap-2">
                  <Scale size={12} />
                  AERO-SIM // SPECIFICATION MATRIX
                </div>
                <h3 className="font-headline-display text-xl font-bold uppercase text-primary tracking-tight">
                  Engine Comparison
                </h3>
              </div>
              <button
                onClick={() => setCompareOpen(false)}
                className="p-2 rounded border border-outline-variant/30 bg-surface-container-highest/60 text-on-surface-variant hover:text-primary hover:border-outline-variant transition-colors"
                aria-label="Close"
              >
                <X size={16} />
              </button>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
              <table className="w-full font-mono text-xs border-collapse">
                <thead>
                  <tr className="border-b border-outline-variant/25 bg-surface-container-highest/30">
                    <th className="py-3 px-5 text-left text-on-surface-variant/50 uppercase tracking-widest text-[10px] font-bold w-32">Spec</th>
                    {ENGINES.map((e) => (
                      <th key={e.id} className="py-3 px-4 text-left">
                        <div className="font-mono text-[9px] text-tertiary uppercase tracking-wider">{e.code}</div>
                        <div className="font-headline-display text-sm font-bold text-white mt-0.5">{e.name}</div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/15">
                  {[
                    { label: "Max Power", key: "maxPower" as const },
                    { label: "Max Torque", key: "maxTorque" as const },
                    { label: "Dry Mass", key: "dryMass" as const },
                    { label: "Max RPM", key: "maxRpm" as const },
                    { label: "Induction", key: "induction" as const },
                    { label: "Ceiling", key: "ceiling" as const },
                  ].map(({ label, key }) => (
                    <tr key={key} className="hover:bg-surface-container-highest/20 transition-colors">
                      <td className="py-3 px-5 text-on-surface-variant/60 font-bold uppercase text-[10px] tracking-wider">{label}</td>
                      {ENGINES.map((e) => (
                        <td key={e.id} className={`py-3 px-4 text-[11px] font-semibold ${key === "maxPower" ? "text-tertiary" : "text-on-surface"}`}>
                          {e[key]}
                        </td>
                      ))}
                    </tr>
                  ))}
                  {/* Power bar row */}
                  <tr className="hover:bg-surface-container-highest/20 transition-colors">
                    <td className="py-3 px-5 text-on-surface-variant/60 font-bold uppercase text-[10px] tracking-wider">vs Apex</td>
                    {ENGINES.map((e) => (
                      <td key={e.id} className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 bg-outline-variant/30 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-tertiary to-amber-300 rounded-full"
                              style={{ width: `${e.powerPct}%` }}
                            />
                          </div>
                          <span className="text-[10px] text-white/60 w-8">{e.powerPct.toFixed(0)}%</span>
                        </div>
                      </td>
                    ))}
                  </tr>
                  {/* Action row */}
                  <tr className="bg-surface-container-highest/20">
                    <td className="py-4 px-5 text-on-surface-variant/60 font-bold uppercase text-[10px] tracking-wider">Launch</td>
                    {ENGINES.map((e) => (
                      <td key={e.id} className="py-4 px-4">
                        <button
                          onClick={() => { setCompareOpen(false); selectEngine(e.id); }}
                          className="w-full flex items-center justify-center gap-1.5 py-2 px-3 rounded bg-tertiary text-black font-bold uppercase tracking-wider text-[10px] shadow-[1px_1px_0px_#000000] hover:-translate-y-0.5 hover:shadow-[2px_2px_0px_#000000] transition-all"
                        >
                          <Zap size={11} className="fill-black" />
                          <span>{e.name.replace("ROTAX ", "")}</span>
                        </button>
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
