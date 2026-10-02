"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import EngineViewer from "@/components/EngineViewer";
import EngineViewer912 from "@/components/EngineViewer912";
import EngineViewer915 from "@/components/EngineViewer915";
import EngineViewer916 from "@/components/EngineViewer916";
import {
  Zap,
  AlertTriangle,
  Cpu,
  Gauge,
  Layers,
  ShieldCheck,
  Plane,
  ArrowRight,
  CheckCircle2,
  ExternalLink,
  ChevronRight,
  BarChart3,
  Info,
} from "lucide-react";
import "./engine-info.css";

/* ─── Data ─────────────────────────────────────────────────────────────────── */
const ENGINE_SPECS = [
  {
    id: "Rotax_912_ULS",
    v4: { idle: "1,400", critAlt: "n/a (no turbo)", tbo: "2,000 h", egtLimit: "880 \u00b0C", faultTypes: 11, ai: "training" as "trained" | "training" },
    code: "01",
    name: "ROTAX 912 ULS",
    role: "Naturally Aspirated Baseline",
    tag: "NAT. ASPIRATED",
    powerKw: 73.5,
    powerHp: 100,
    maxRpm: "5,800",
    massKg: 56.6,
    torque: "128 Nm @ 5,800 RPM",
    induction: "Naturally Aspirated",
    fuelSystem: "Dual constant depression carburetors",
    relativePower: "Baseline (0%)",
    powerToMass: "1.77 HP/kg",
    powerPct: 62.5,
    serviceCeiling: "Altitude limited by air density",
    bestUseCase: "Lightweight endurance / baseline UAV studies",
    character:
      "Because it is naturally aspirated, available power is strongly coupled to ambient air density. As altitude increases, the engine ingests less air for a given displacement, reducing available torque.",
    simulationModel:
      "Start with an RPM/torque map and apply corrections for throttle, ambient pressure, temperature and altitude. Avoid representing the engine with only a maximum-power scalar.",
    signalChain: "RPM + Throttle + Ambient Density (ρ) + Temp → NA Torque Map → Power = T·ω",
    Viewer: EngineViewer912,
  },
  {
    id: "Rotax_914_ULF",
    v4: { idle: "1,400", critAlt: "15,000 ft", tbo: "2,000 h", egtLimit: "950 \u00b0C", faultTypes: 13, ai: "trained" as "trained" | "training" },
    code: "02",
    name: "ROTAX 914 F/UL",
    role: "Turbocharged Baseline",
    tag: "TURBO / TCU",
    powerKw: 84.8,
    powerHp: 115,
    maxRpm: "5,800",
    massKg: 64.0,
    torque: "144 Nm @ 5,800 RPM",
    induction: "Turbocharged (TCU regulated)",
    fuelSystem: "Dual constant depression carburetors + TCU",
    relativePower: "+15% over 912",
    powerToMass: "1.80 HP/kg",
    powerPct: 71.9,
    serviceCeiling: "Critical altitude boost maintained",
    bestUseCase: "Moderate-performance, altitude-aware UAV studies",
    character:
      "Turbocharging breaks the simple relationship between ambient density and engine output. The turbocharger raises manifold pressure above ambient, but boost is constrained by turbo limits, thermal state, and TCU control.",
    simulationModel:
      "Expose manifold pressure (MAP) as a live telemetry channel rather than hiding turbo behavior inside a black-box scalar. Model boost limitation and thermal constraints at altitude.",
    signalChain: "RPM + Throttle + Altitude + Temp → Turbo/MAP Model → Torque Map → Power",
    Viewer: EngineViewer,
  },
  {
    id: "Rotax_915_iS",
    v4: { idle: "1,400", critAlt: "15,000 ft", tbo: "1,200 h", egtLimit: "950 \u00b0C", faultTypes: 14, ai: "training" as "trained" | "training" },
    code: "03",
    name: "ROTAX 915 iS",
    role: "Advanced Turbocharged / EFI Model",
    tag: "TURBO + EFI",
    powerKw: 104.0,
    powerHp: 141,
    maxRpm: "5,800",
    massKg: 82.2,
    torque: "172 Nm @ 5,800 RPM",
    induction: "Turbocharged + Intercooler",
    fuelSystem: "Redundant Electronic Fuel Injection (EFI / ECU)",
    relativePower: "+41% over 912",
    powerToMass: "1.72 HP/kg",
    powerPct: 88.1,
    serviceCeiling: "Full takeoff power to 15,000 ft; 23,000 ft ceiling",
    bestUseCase: "High-performance UAV, heavier payload, advanced engine management",
    character:
      "Intercooling removes compression heat before air enters combustion chambers, altering air density and boost-temperature dynamics. Redundant EFI introduces complex ECU fuel mapping.",
    simulationModel:
      "Predict torque and fuel flow from RPM, throttle, altitude, ambient temp, MAP, and intake air temp (IAT). Calculate brake power deterministically from torque and RPM.",
    signalChain: "RPM + Throttle + Alt + MAP + IAT → Multi-Variable AI Twin → Torque & Fuel Flow",
    Viewer: EngineViewer915,
  },
  {
    id: "Rotax_916_iS",
    v4: { idle: "1,400", critAlt: "15,000 ft", tbo: "2,000 h", egtLimit: "950 \u00b0C", faultTypes: 14, ai: "training" as "trained" | "training" },
    code: "04",
    name: "ROTAX 916 iS",
    role: "High-Performance Flagship Model",
    tag: "APEX / FADEC",
    powerKw: 117.0,
    powerHp: 160,
    maxRpm: "5,800",
    massKg: 85.8,
    torque: "193 Nm @ 5,800 RPM",
    induction: "Turbocharged High-Performance Platform",
    fuelSystem: "Dual redundant digital EFI / FADEC",
    relativePower: "+60% over 912",
    powerToMass: "1.86 HP/kg",
    powerPct: 100,
    serviceCeiling: "High-altitude sustained power & climb",
    bestUseCase: "Maximum-performance UAV and demanding climb/payload studies",
    character:
      "Delivers 160 HP (+19 HP / 13.5% over 915 iS) with only 3.6 kg additional mass. Yields the highest power-to-mass ratio (1.86 HP/kg) in the lineup, making it ideal for power-dense military & commercial UAV platforms.",
    simulationModel:
      "Shares common engine schema with high-resolution torque/fuel maps, altitude/MAP dependency, thermal-state feedback, and real-time AI confidence metrics.",
    signalChain: "Comprehensive Flight Envelope Parameters → Neural Digital Twin → High-Fidelity Physics",
    Viewer: EngineViewer916,
  },
];

const SECTIONS = [
  { id: "summary",      label: "Overview" },
  { id: "engines",      label: "Engine Profiles" },
  { id: "comparison",   label: "Matrix" },
  { id: "physics",      label: "Physics" },
  { id: "ai-twin",      label: "AI Twin" },
  { id: "architecture", label: "Architecture" },
  { id: "uav",          label: "UAV Sim" },
  { id: "validation",   label: "Validation" },
  { id: "sources",      label: "Sources" },
];

// Physics-v4 models, scored on test flights they never saw (validation_v4, models_v4
// manifests). RUL is quoted on the inputs a real aircraft has - predicted wear and
// severity, measured margin - not on simulator labels.
const STAGES = [
  { label: "914 F/UL \u2014 reference twin", done: true, desc: "Six models deployed. Detection AUC 0.879 (0.952 where the engine has drifted from its twin), fault naming F1 0.612 over 13 fault types, severity error 0.22 on real faults, sensor-condition F1 0.417, wear error 0.063. RUL 4.6% of TBO overall; on engines that wear out early 387 h against 462 h for counting down to overhaul." },
  { label: "912 ULS", done: false, desc: "11 fault types (no turbo). Training on the v4 data; results appear here once all six models pass their gates." },
  { label: "915 iS", done: false, desc: "14 fault types and a shorter 1,200 h TBO. Training on the v4 data; the 914's models stand in until it finishes." },
  { label: "916 iS", done: false, desc: "14 fault types. Training on the v4 data. All four engines share one twin, one 29-input contract and one set of gates." },
];

const PHYSICS_FORMULAS = [
  { label: "ANGULAR VELOCITY (ω)", formula: "ω = 2π · RPM / 60", note: "Converts crankshaft velocity to radians per second." },
  { label: "BRAKE POWER (P)", formula: "P = T · ω", note: "Shaft power as instantaneous torque × angular speed." },
  { label: "PRACTICAL METRIC", formula: "P(kW) = T · RPM / 9550", note: "Implemented directly across the simulator backend.", highlight: true },
  { label: "AIR DENSITY LAPSE", formula: "ρ = p / (R · T)", note: "Ideal gas law — ambient air density reduction at altitude." },
  { label: "NA TORQUE MODEL", formula: "T = f(RPM, throttle, ρ, T_amb)", note: "Torque drops proportionally to altitude in NA engines." },
  { label: "TURBO TORQUE (MAP)", formula: "T = f(RPM, throttle, MAP, T_iat, Alt)", note: "Decoupled from ambient pressure until wastegate critical altitude." },
];

const PIPELINE_STAGES = ["Engine", "Mechanical", "Performance", "Induction", "Fuel", "Thermal", "Electrical", "Wear", "AI Twin", "Residuals", "Telemetry"];

const UAV_CASES = [
  { engine: "912 ULS", title: "Lightweight Endurance", desc: "Ideal for low-altitude, long-endurance surveillance UAVs where minimal empty weight maximizes fuel capacity and structural simplicity avoids turbo failure points." },
  { engine: "914 F/UL", title: "Altitude-Aware Patrol", desc: "Maintains sea-level cruise climb at mid-altitudes (up to 15,000 ft), suitable for tactical UAV platforms needing reliable altitude flexibility." },
  { engine: "915 iS", title: "Heavy Payload / MALE", desc: "Medium-Altitude Long-Endurance (MALE) UAVs operating up to 23,000 ft with complex radar and multi-spectral gimbal payloads requiring redundant EFI reliability." },
  { engine: "916 iS", title: "Max Performance Flagship", desc: "Maximum takeoff weight, short takeoff roll, and rapid climb rate. The 1.86 HP/kg density enables high-speed strike or heavy cargo UAV configurations." },
];

const VALIDATION_LEVELS = [
  { lvl: "L1", title: "Specification Validation", desc: "Verify OEM rated power, maximum continuous RPM, dry mass, and published torque points against certified BRP-Rotax flight manuals." },
  { lvl: "L2", title: "Physics Validation", desc: "Verify P = T·ω conservation, ideal gas density lapse, and thermal dissipation rates in steady-state operation." },
  { lvl: "L3", title: "Map Validation", desc: "Compare predicted torque and brake specific fuel consumption (BSFC) against trusted dynamometer operating points." },
  { lvl: "L4", title: "Dynamic Validation", desc: "Test throttle transients, turbocharger spool-up lag, and cylinder thermal inertia where time-series test records exist." },
  { lvl: "L5", title: "AI Validation", desc: "Every model ships only through its gates on flights it never saw, scored on the number that matters: detection where the engine has drifted, fault naming across every fault type, severity where a fault exists, sensor condition with every kind weighted equally, and RUL on engines that wear out early." },
  { lvl: "L6", title: "Live-input honesty", desc: "The live window is proven identical to training (zero difference on real test flights), and RUL is re-scored on predicted wear and severity instead of simulator labels - the number an aircraft would actually get." },
  { lvl: "L7", title: "Cross-implementation", desc: "The Simulink Rotax 914 v4 model is compared step for step with the Python twin: four reference flights, about 26,000 steps, every channel, worst relative error 2e-15." },
];

/* ─── Section divider ───────────────────────────────────────────────────────── */
function SectionHeader({ code, children }: { code: string; children: React.ReactNode }) {
  return (
    <div className="mb-6">
      <div className="flex items-center gap-3 mb-1.5">
        <span className="h-px w-8 bg-tertiary shrink-0" />
        <span className="font-mono text-[10px] font-bold text-tertiary tracking-[0.2em] uppercase shrink-0">
          {code}
        </span>
        <span className="h-px flex-1 bg-outline-variant/30" />
      </div>
      <h2 className="font-headline-display text-xl md:text-2xl font-bold uppercase tracking-tight text-primary">
        {children}
      </h2>
    </div>
  );
}

/* ─── Page ──────────────────────────────────────────────────────────────────── */
export default function EngineInfoPage() {
  const router = useRouter();
  const [activeEngineIndex, setActiveEngineIndex] = useState<number>(0);
  const [activeSection, setActiveSection] = useState<string>(SECTIONS[0].id);
  const selectedEngine = ENGINE_SPECS[activeEngineIndex];
  const trained = selectedEngine.v4.ai === "trained";

  // The section nav follows the reader: the section crossing the band just
  // under the sticky bars is the current one.
  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => e.isIntersecting && setActiveSection(e.target.id)),
      { rootMargin: "-120px 0px -70% 0px" },
    );
    SECTIONS.forEach((s) => { const el = document.getElementById(s.id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (el) el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  };

  return (
    <>
      <Navbar />

      <main className="flex-grow pt-16 relative min-h-screen">
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />

        {/* ── HERO ──────────────────────────────────────────────────────────── */}
        <header className="relative z-10 border-b border-outline-variant/30">
          <div className="max-w-7xl mx-auto px-4 md:px-6 py-10 md:py-14">

            {/* Eyebrow */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
              <div className="flex items-center gap-3">
                <span className="h-px w-8 bg-tertiary" />
                <span className="font-mono text-[10px] font-bold tracking-[0.24em] text-tertiary uppercase">
                  AERO-SIM // TECHNICAL REPORT TR-ROTAX-2026-01
                </span>
              </div>
              <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.14em] text-on-surface-variant/70 border border-outline-variant/30 bg-surface/80 px-3 py-1.5 rounded">
                <ShieldCheck size={13} className="text-tertiary" />
                SPECIFICATION & DIGITAL-TWIN STANDARD
              </div>
            </div>

            <h1 className="font-headline-display text-[28px] sm:text-[36px] md:text-[46px] font-extrabold uppercase tracking-tight text-primary mb-2">
              Rotax Aircraft Engine Lineup
            </h1>
            <p className="font-mono text-[10px] tracking-[0.16em] text-tertiary uppercase mb-4">
              ROTAX 912 ULS · ROTAX 914 F/UL · ROTAX 915 iS · ROTAX 916 iS
            </p>
            <p className="text-sm md:text-base leading-relaxed text-on-surface-variant max-w-4xl mb-8">
              A comprehensive technical comparison and simulation-oriented analysis of four Rotax aircraft engines,
              with emphasis on brake power, torque mapping, induction systems, altitude density behavior,
              redundant EFI engine management, fuel mass-flow modeling, thermal state dynamics, and AI
              digital-twin integration for precision UAV flight simulation.
            </p>

            {/* Engine selector mini-cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
              {ENGINE_SPECS.map((eng, idx) => (
                <button
                  key={eng.id}
                  onClick={() => { setActiveEngineIndex(idx); scrollToSection("engines"); }}
                  aria-pressed={activeEngineIndex === idx}
                  className={`group text-left p-4 rounded-lg border transition-[transform,border-color,background-color,box-shadow] duration-300 ${
                    activeEngineIndex === idx
                      ? "border-tertiary bg-tertiary/10 shadow-[2px_2px_0px_#000000]"
                      : "border-outline-variant/30 bg-surface/80 hover:border-tertiary/50 hover:-translate-y-0.5 hover:shadow-[2px_2px_0px_#000000]"
                  }`}
                >
                  <div className="font-mono text-[9px] text-tertiary font-bold tracking-widest uppercase mb-1">
                    {eng.code} // {eng.powerHp} HP
                  </div>
                  <div className="font-headline-display font-bold text-sm text-primary group-hover:text-tertiary transition-colors truncate">{eng.name}</div>
                  <div className="font-mono text-[10px] text-on-surface-variant/60 mt-0.5 truncate">{eng.tag}</div>
                </button>
              ))}
            </div>

            {/* Compliance note */}
            <div className="flex items-start gap-3 border border-tertiary/25 bg-tertiary/[0.06] p-4 rounded-lg font-mono text-xs leading-relaxed text-on-surface-variant">
              <Info size={16} className="text-tertiary shrink-0 mt-0.5" />
              <div>
                <strong className="text-tertiary uppercase tracking-wider block mb-0.5">DATA BASIS & COMPLIANCE NOTE:</strong>
                Current BRP-Rotax product specifications verified against official technical documentation.
                Where a value is not published in official OEM sheets, it is treated as a simulation-design
                recommendation rather than an OEM certification value. Suitable for simulator design, physics
                calibration, and digital-twin modeling.
              </div>
            </div>
          </div>
        </header>

        {/* Sticky section nav: a sibling of the header, so it stays pinned for the whole page */}
        <div className="sticky top-16 z-30 bg-background/90 backdrop-blur-md border-b border-outline-variant/30 overflow-x-auto">
          <div className="max-w-7xl mx-auto px-4 py-2 flex items-center gap-1 whitespace-nowrap font-mono text-[10px] uppercase tracking-wider">
            {SECTIONS.map((sec) => (
              <button
                key={sec.id}
                onClick={() => scrollToSection(sec.id)}
                aria-current={activeSection === sec.id ? "true" : undefined}
                className={`px-3 py-1.5 rounded transition-colors ${
                  activeSection === sec.id
                    ? "text-tertiary bg-tertiary/10"
                    : "text-on-surface-variant hover:text-tertiary hover:bg-white/5"
                }`}
              >
                {sec.label}
              </button>
            ))}
            <Link
              href="/engine"
              className="ml-auto bg-tertiary text-black font-bold px-4 py-1.5 rounded flex items-center gap-1.5 hover:brightness-110 transition-[filter] shadow-[1px_1px_0px_#000000] shrink-0"
            >
              Propulsion Catalogue <ArrowRight size={11} />
            </Link>
          </div>
        </div>

        {/* ── CONTENT ───────────────────────────────────────────────────────── */}
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-12 space-y-20 relative z-10">

          {/* 01 · EXECUTIVE SUMMARY ─────────────────────────────────────────── */}
          <section id="summary" className="scroll-mt-28">
            <SectionHeader code="01 // OVERVIEW">Executive Summary</SectionHeader>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              <div className="lg:col-span-2 space-y-4 text-sm text-on-surface-variant leading-relaxed">
                <p>
                  The four-engine lineup is exceptionally suited to an engineering flight simulator because it creates
                  a clear progression in physical and computational complexity. The <strong className="text-primary">Rotax 912 ULS</strong> provides
                  a clean, naturally aspirated reference case; the <strong className="text-primary">Rotax 914 F/UL</strong> adds turbocharging
                  and manifold pressure dynamics; the <strong className="text-primary">Rotax 915 iS</strong> introduces intercooling and
                  redundant electronic fuel injection; and the <strong className="text-primary">Rotax 916 iS</strong> extends the family
                  into the 160 HP high-performance class with the highest power-to-mass density.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                  {[
                    { title: "912 ULS Baseline", desc: "Best reference model for validating core RPM–torque–power physics and air-density lapse without turbo confounding factors." },
                    { title: "914 F/UL Turbo", desc: "Introduces altitude-dependent turbocharger behavior, wastegate control, and manifold-pressure (MAP) telemetry channels." },
                    { title: "915 iS Digital Twin", desc: "Substantially richer digital-twin through intercooler thermodynamics and redundant electronic fuel injection mappings." },
                    { title: "916 iS Flagship", desc: "Strongest performance showcase with the highest power-to-mass ratio (1.86 HP/kg) for demanding high-altitude long-endurance UAV operations." },
                  ].map((item) => (
                    <div key={item.title} className="border border-outline-variant/30 bg-surface/80 p-4 rounded-lg">
                      <div className="flex items-center gap-2 font-mono text-[10px] font-bold text-primary uppercase mb-1.5">
                        <CheckCircle2 size={13} className="text-tertiary shrink-0" />
                        {item.title}
                      </div>
                      <p className="text-xs text-on-surface-variant/75 leading-relaxed">{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Critical correction card */}
              <div className="border border-tertiary/35 bg-surface/80 p-5 rounded-lg shadow-[2px_2px_0px_#000000] flex flex-col gap-3">
                <div className="flex items-center gap-2 font-mono text-[10px] font-bold text-tertiary uppercase tracking-widest">
                  <AlertTriangle size={14} />
                  SPEC CORRECTION
                </div>
                <h3 className="font-bold text-primary text-sm">914 F/UL Power — Official Values</h3>
                <p className="text-xs text-on-surface-variant leading-relaxed">
                  The 914 F/UL is often mis-listed as 74 kW. Official BRP-Rotax specs confirm:
                </p>
                <div className="space-y-1.5 font-mono text-xs">
                  {[
                    { label: "912 ULS", value: "73.5 kW / 100 HP", accent: false },
                    { label: "914 F/UL", value: "84.8 kW / 115 HP", accent: true },
                    { label: "915 iS", value: "104 kW / 141 HP", accent: false },
                    { label: "916 iS", value: "117 kW / 160 HP", accent: false },
                  ].map((row) => (
                    <div key={row.label} className={`flex justify-between items-center px-3 py-2 rounded border ${row.accent ? "border-tertiary/50 bg-tertiary/10" : "border-outline-variant/25 bg-black/30"}`}>
                      <span className={row.accent ? "text-tertiary font-bold" : "text-on-surface-variant"}>{row.label}</span>
                      <span className={`font-bold ${row.accent ? "text-tertiary" : "text-primary"}`}>{row.value}</span>
                    </div>
                  ))}
                </div>
                <p className="font-mono text-[9px] text-on-surface-variant/50 uppercase tracking-wider">REF: BRP-Rotax 2026 Engine Maintenance & Spec Standard</p>
              </div>
            </div>
          </section>

          {/* 02 · ENGINE PROFILES ───────────────────────────────────────────── */}
          <section id="engines" className="scroll-mt-28 ei-reveal">
            <SectionHeader code="02–05 // PROPULSION">Technical Engine Profiles</SectionHeader>

            {/* Tab switcher */}
            <div className="flex flex-wrap gap-2 mb-5 font-mono text-[10px] uppercase tracking-wider">
              {ENGINE_SPECS.map((eng, idx) => (
                <button
                  key={eng.id}
                  onClick={() => setActiveEngineIndex(idx)}
                  aria-pressed={activeEngineIndex === idx}
                  className={`px-3 py-2 rounded font-bold transition-colors ${
                    activeEngineIndex === idx
                      ? "bg-tertiary text-black shadow-[2px_2px_0px_#000000]"
                      : "border border-outline-variant/40 text-on-surface-variant hover:border-tertiary/50 hover:text-primary"
                  }`}
                >
                  {eng.name.replace("ROTAX ", "")}
                </button>
              ))}
            </div>

            {/* Engine detail card */}
            <div className="border border-outline-variant/30 bg-surface/80 rounded-lg overflow-hidden shadow-[2px_2px_0px_#000000]">
              <div className="grid grid-cols-1 lg:grid-cols-12">
                {/* 3D viewer */}
                <div className="lg:col-span-5 relative min-h-[320px] sm:min-h-[400px] bg-[radial-gradient(ellipse_at_center,rgba(100,45,15,0.18),transparent_50%),linear-gradient(145deg,#15120f,#0b0c0c)] border-b lg:border-b-0 lg:border-r border-outline-variant/25 flex flex-col justify-between p-4">
                  <div className="flex items-center justify-between z-10 relative">
                    <span className="font-mono text-[9px] font-bold tracking-widest text-tertiary border border-tertiary/30 bg-black/60 px-2.5 py-1 rounded uppercase">
                      3D DIGITAL TWIN VIEW
                    </span>
                    <span className="font-mono text-[9px] text-on-surface-variant/40 uppercase tracking-wide">
                      DRAG TO ROTATE
                    </span>
                  </div>
                  <div className="absolute inset-0 z-0">
                    <selectedEngine.Viewer />
                  </div>
                  <div className="z-10 relative bg-black/70 backdrop-blur-sm border border-outline-variant/25 p-3 rounded font-mono text-[11px] flex justify-between items-center">
                    <div>
                      <span className="text-on-surface-variant/50 text-[9px] block uppercase tracking-wider">AI STATUS</span>
                      <span className="font-bold text-primary flex items-center gap-1.5">
                        <span className={`h-1.5 w-1.5 rounded-full ${trained ? "bg-tertiary animate-pulse" : "bg-on-surface-variant/60"}`} />
                        {trained ? "AI READY · V4 MODELS LIVE" : "AI TRAINING · 914 MODELS STAND IN"}
                      </span>
                    </div>
                    <button
                      onClick={() => router.push(`/simulate?engine=${selectedEngine.id}`)}
                      className="bg-tertiary text-black text-[10px] font-bold px-3 py-1.5 rounded flex items-center gap-1 hover:brightness-110 active:translate-y-px transition-[filter,transform] uppercase tracking-wider shadow-[1px_1px_0px_#000000]"
                    >
                      <Zap size={11} /> Launch
                    </button>
                  </div>
                </div>

                {/* Specs panel */}
                <div key={selectedEngine.id} className="ei-swap lg:col-span-7 p-6 flex flex-col gap-5">
                  <div>
                    <div className="flex items-start justify-between gap-3 mb-1">
                      <span className="font-mono text-[10px] font-bold text-tertiary tracking-widest uppercase">{selectedEngine.role}</span>
                      <span className="font-mono text-[10px] text-on-surface-variant/50 shrink-0">{selectedEngine.powerHp} HP / {selectedEngine.powerKw} kW</span>
                    </div>
                    <h3 className="font-headline-display text-2xl sm:text-3xl font-bold uppercase tracking-tight text-primary mb-2">
                      {selectedEngine.name}
                    </h3>
                    <p className="text-sm text-on-surface-variant leading-relaxed">{selectedEngine.character}</p>
                  </div>

                  {/* Power bar */}
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-[9px] uppercase tracking-wider text-on-surface-variant/50 flex items-center gap-1">
                        <Gauge size={10} className="text-tertiary" /> Power vs lineup apex
                      </span>
                      <span className="font-mono text-[11px] font-bold text-primary">{selectedEngine.powerPct.toFixed(0)}%</span>
                    </div>
                    <div className="h-1.5 bg-outline-variant/25 rounded-full overflow-hidden">
                      <div className="ei-bar h-full w-full bg-tertiary rounded-full" style={{ transform: `scaleX(${selectedEngine.powerPct / 100})` }} />
                    </div>
                  </div>

                  {/* Quick spec grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-xs">
                    {[
                      { label: "RATED POWER", value: `${selectedEngine.powerKw} kW`, sub: `${selectedEngine.powerHp} HP`, hi: true },
                      { label: "RATED SPEED", value: selectedEngine.maxRpm, sub: "RPM", hi: false },
                      { label: "DRY MASS", value: `${selectedEngine.massKg} kg`, sub: selectedEngine.powerToMass, hi: true },
                      { label: "MAX TORQUE", value: selectedEngine.torque.split(" @")[0], sub: "@ 5,800 RPM", hi: false },
                      // physics_v4, the model the twin and the AI run on
                      { label: "IDLE", value: selectedEngine.v4.idle, sub: "RPM", hi: false },
                      { label: "CRITICAL ALT", value: selectedEngine.v4.critAlt, sub: "full boost held to", hi: false },
                      { label: "TBO", value: selectedEngine.v4.tbo, sub: "scheduled overhaul", hi: true },
                      { label: "FAULT TYPES", value: String(selectedEngine.v4.faultTypes), sub: "the AI can name", hi: true },
                    ].map((spec) => (
                      <div key={spec.label} className="border border-outline-variant/25 bg-surface-container-highest/60 p-3 rounded-lg">
                        <span className="text-[9px] text-on-surface-variant/50 block uppercase tracking-wider">{spec.label}</span>
                        <span className="font-bold text-primary text-sm mt-0.5 block truncate">{spec.value}</span>
                        <span className={`text-[9px] ${spec.hi ? "text-tertiary" : "text-on-surface-variant/50"}`}>{spec.sub}</span>
                      </div>
                    ))}
                  </div>

                  {/* Signal chain */}
                  <div className="space-y-2 font-mono text-xs">
                    <div className="border-l-2 border-tertiary bg-white/[0.02] p-3 pl-4 rounded-r-lg">
                      <span className="text-tertiary font-bold uppercase tracking-wider block text-[9px] mb-1">RECOMMENDED SIMULATION MODEL</span>
                      <p className="text-on-surface-variant leading-relaxed text-[12px]">{selectedEngine.simulationModel}</p>
                    </div>
                    <div className="border-l-2 border-outline-variant/50 bg-white/[0.02] p-3 pl-4 rounded-r-lg">
                      <span className="text-on-surface-variant/50 font-bold uppercase tracking-wider block text-[9px] mb-1">CERTIFIED LIMITS (PHYSICS V4)</span>
                      <div className="text-on-surface-variant text-[11px]">
                        CHT 135 &deg;C &middot; EGT {selectedEngine.v4.egtLimit} &middot; oil temp 130 &deg;C &middot; oil pressure min 0.8 bar below 3,500 RPM, 2.0 bar above
                      </div>
                    </div>
                    <div className="border-l-2 border-outline-variant/50 bg-white/[0.02] p-3 pl-4 rounded-r-lg">
                      <span className="text-on-surface-variant/50 font-bold uppercase tracking-wider block text-[9px] mb-1">SIGNAL CHAIN</span>
                      <div className="text-tertiary/90 tracking-wide text-[11px]">{selectedEngine.signalChain}</div>
                    </div>
                  </div>

                  {/* Footer CTA */}
                  <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-3 border-t border-outline-variant/25">
                    <span className="font-mono text-[10px] text-on-surface-variant/60 uppercase">
                      USE CASE: <strong className="text-primary">{selectedEngine.bestUseCase}</strong>
                    </span>
                    <Link
                      href={`/simulate?engine=${selectedEngine.id}`}
                      className="shrink-0 flex items-center gap-2 bg-tertiary text-black font-bold uppercase px-5 py-2.5 rounded text-[10px] tracking-wider shadow-[2px_2px_0px_#000000] hover:-translate-y-0.5 hover:shadow-[4px_4px_0px_#000000] active:translate-y-0 transition-[transform,box-shadow]"
                    >
                      <Zap size={13} className="fill-black" /> Configure in Simulator
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* 06 · COMPARATIVE MATRIX ────────────────────────────────────────── */}
          <section id="comparison" className="scroll-mt-28 ei-reveal">
            <SectionHeader code="06 // BENCHMARKING">Comparative Engineering Matrix</SectionHeader>
            <p className="text-sm text-on-surface-variant mb-5">
              Comprehensive comparison across power, mass, induction technology, and power-to-mass ratios.
            </p>

            <div className="overflow-x-auto border border-outline-variant/30 rounded-lg bg-surface/80 shadow-[2px_2px_0px_#000000] mb-5">
              <table className="w-full text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-outline-variant/30 bg-black/40 text-[10px] text-tertiary uppercase tracking-wider">
                    <th className="p-4">Engine</th>
                    <th className="p-4">HP</th>
                    <th className="p-4">kW</th>
                    <th className="p-4">Max RPM</th>
                    <th className="p-4">Mass (kg)</th>
                    <th className="p-4">Induction</th>
                    <th className="p-4">vs 912</th>
                    <th className="p-4">HP/kg</th>
                    <th className="p-4">AI</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/20">
                  {ENGINE_SPECS.map((eng, idx) => (
                    <tr
                      key={eng.id}
                      onClick={() => { setActiveEngineIndex(idx); scrollToSection("engines"); }}
                      className={`transition-colors cursor-pointer ${activeEngineIndex === idx ? "bg-tertiary/[0.05]" : "hover:bg-white/[0.03]"}`}
                    >
                      <td className="p-4 font-bold text-primary whitespace-nowrap">
                        <span className="flex items-center gap-2">
                          <span className="h-1.5 w-1.5 rounded-full bg-tertiary shrink-0" />
                          {eng.name}
                        </span>
                      </td>
                      <td className="p-4 text-primary font-bold">{eng.powerHp}</td>
                      <td className="p-4 text-on-surface-variant">{eng.powerKw}</td>
                      <td className="p-4 text-on-surface-variant">{eng.maxRpm}</td>
                      <td className="p-4 text-primary">{eng.massKg}</td>
                      <td className="p-4 text-on-surface-variant min-w-[190px]">{eng.induction}</td>
                      <td className="p-4 font-bold text-tertiary">{eng.relativePower}</td>
                      <td className="p-4 text-primary font-bold">{eng.powerToMass}</td>
                      <td className="p-4">
                        <span className={`px-2 py-0.5 rounded text-[9px] font-bold border ${eng.v4.ai === "trained" ? "bg-tertiary/15 text-tertiary border-tertiary/30" : "bg-black/30 text-on-surface-variant border-outline-variant/40"}`}>
                          {eng.v4.ai === "trained" ? "V4 TRAINED" : "V4 TRAINING"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Bar charts */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                { title: "POWER PROGRESSION (HP)", icon: <BarChart3 size={14} className="text-tertiary" />, bars: ENGINE_SPECS.map(e => ({ label: e.name, value: e.powerHp, pct: (e.powerHp / 160) * 100, text: `${e.powerHp} HP / ${e.powerKw} kW` })), color: "bg-tertiary" },
                { title: "POWER-TO-MASS (HP/KG)", icon: <Gauge size={14} className="text-tertiary" />, bars: ENGINE_SPECS.map(e => ({ label: e.name, value: parseFloat(e.powerToMass), pct: ((parseFloat(e.powerToMass) - 1.5) / 0.4) * 100, text: e.powerToMass })), color: "bg-tertiary/60" },
              ].map((chart) => (
                <div key={chart.title} className="border border-outline-variant/30 bg-surface/80 p-5 rounded-lg shadow-[2px_2px_0px_#000000]">
                  <div className="flex items-center justify-between mb-4">
                    <span className="font-mono text-[10px] font-bold text-tertiary uppercase tracking-wider">{chart.title}</span>
                    {chart.icon}
                  </div>
                  <div className="space-y-3">
                    {chart.bars.map((bar) => (
                      <div key={bar.label}>
                        <div className="flex justify-between font-mono text-[10px] mb-1">
                          <span className="text-on-surface-variant">{bar.label}</span>
                          <span className="text-tertiary font-bold">{bar.text}</span>
                        </div>
                        <div className="h-1.5 bg-outline-variant/25 rounded-full overflow-hidden">
                          <div className={`ei-bar h-full w-full rounded-full ${chart.color}`} style={{ transform: `scaleX(${bar.pct / 100})` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* 07 · PHYSICS ───────────────────────────────────────────────────── */}
          <section id="physics" className="scroll-mt-28 ei-reveal">
            <SectionHeader code="07 // MATH">Physics Formulations</SectionHeader>
            <p className="text-sm text-on-surface-variant mb-5 max-w-4xl">
              The strongest architecture is a <strong className="text-primary">hybrid physics + AI model</strong>. Deterministic conservation
              laws are calculated analytically, while neural models learn complex non-linear aerothermal behaviors.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
              {PHYSICS_FORMULAS.map((f) => (
                <div key={f.label} className={`border p-4 rounded-lg font-mono ${f.highlight ? "border-tertiary/40 bg-tertiary/5" : "border-outline-variant/30 bg-surface/80"}`}>
                  <span className="text-[9px] text-tertiary uppercase tracking-wider block mb-1">{f.label}</span>
                  <div className={`text-xl font-bold py-2 ${f.highlight ? "text-tertiary" : "text-primary"}`}>{f.formula}</div>
                  <p className={`text-[11px] ${f.highlight ? "text-primary/75" : "text-on-surface-variant"}`}>{f.note}</p>
                </div>
              ))}
            </div>

            <div className="border border-tertiary/30 bg-tertiary/5 p-4 rounded-lg font-mono text-xs flex items-start gap-3">
              <Zap size={16} className="text-tertiary shrink-0 mt-0.5" />
              <div>
                <strong className="text-tertiary uppercase tracking-wider block mb-1">KEY DESIGN PRINCIPLE:</strong>
                <p className="text-on-surface-variant leading-relaxed text-[12px]">
                  Do not train an AI model to rediscover <span className="text-primary">P = T·ω</span>. If the machine learning model
                  accurately predicts torque from throttle, RPM, and atmospheric state, the simulator calculates shaft
                  power analytically. This avoids redundant model complexity and makes real-time verification bulletproof.
                </p>
              </div>
            </div>
          </section>

          {/* 08 · AI DIGITAL-TWIN ───────────────────────────────────────────── */}
          <section id="ai-twin" className="scroll-mt-28 ei-reveal">
            <SectionHeader code="08 // ML PIPELINE">AI Digital-Twin Architecture</SectionHeader>
            <p className="text-sm text-on-surface-variant mb-5">
              One TCN encoder over the window feeds four classification heads; RUL gets its own branch
              and its own dataset, because in the shared one the label moved by 15.8 h inside a single
              128-sample window — the same size as the head&apos;s own error.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              {/* Feature mapping */}
              <div className="border border-outline-variant/30 bg-surface/80 rounded-lg p-5 font-mono text-xs">
                <div className="flex items-center gap-2 text-tertiary font-bold uppercase tracking-wider mb-4 text-[10px]">
                  <Cpu size={14} /> DEPLOYED MODEL FEATURE MAPPING
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="border-r border-outline-variant/25 pr-4">
                    <span className="text-[9px] text-on-surface-variant/50 uppercase block mb-2 font-bold">INPUT WINDOW — 25 CH × 128 STEPS</span>
                    <ul className="space-y-1.5 text-on-surface-variant/90">
                      {["Flight state — altitude, throttle, airspeed, AoA", "Powerplant — RPM, torque, power, fuel flow", "Aero — thrust, lift, drag, thrust/lift margins", "Thermal — EGT, CHT, oil pressure, oil temp", "Vibration — X / Y / Z axis accelerometers", "Injection timing — carries misfire and injector signatures", "25 features, 128-second rolling window, plus 10 aux"].map((item) => (
                        <li key={item} className="flex items-start gap-1.5 text-[11px]">
                          <span className="h-1 w-1 bg-tertiary rounded-full shrink-0 mt-1.5" />{item}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <span className="text-[9px] text-tertiary uppercase block mb-2 font-bold">OUTPUTS — 5 HEADS</span>
                    <ul className="space-y-1.5">
                      {[
                        "Fault Detection (binary, sigmoid)",
                        "Per-Channel Diagnosis (8 ch × 6 fault types)",
                        "Severity Score (0–100% per channel)",
                        "Engine Failure Modes (4 modes, independent sigmoids)",
                        "Remaining Useful Life (engine hours against TBO)",
                        "Fault types: Bias, Drift, Spike, Stuck-At, Noise",
                        "RUL: LSTM + window statistics, started at a ridge fit",
                      ].map((item, i) => (
                        <li key={item} className={`flex items-start gap-1.5 font-bold text-[11px] ${i < 5 ? "text-tertiary" : "text-on-surface-variant/70"}`}>
                          <CheckCircle2 size={11} className="shrink-0 mt-0.5" />{item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>

              {/* Training stages */}
              <div className="border border-outline-variant/30 bg-surface/80 rounded-lg p-5 font-mono text-xs">
                <div className="flex items-center gap-2 text-tertiary font-bold uppercase tracking-wider mb-4 text-[10px]">
                  <Layers size={14} /> TRAINING PROGRESSION — {STAGES.filter((st) => st.done).length} OF {STAGES.length} DEPLOYED
                </div>
                <div className="space-y-3">
                  {STAGES.map((stage, idx) => (
                    <div key={stage.label} className={`flex items-start gap-3 p-3 rounded-lg border bg-black/30 ${stage.done ? "border-tertiary/30" : "border-outline-variant/30"}`}>
                      <span className={`px-2 py-0.5 rounded font-bold text-[9px] shrink-0 mt-0.5 ${stage.done ? "bg-tertiary/15 text-tertiary" : "bg-white/5 text-on-surface-variant"}`}>
                        S{idx + 1} {stage.done ? "LIVE" : "TRAINING"}
                      </span>
                      <div>
                        <strong className="text-primary block text-[11px] mb-0.5">{stage.label}</strong>
                        <p className="text-on-surface-variant/80 leading-relaxed text-[11px]">{stage.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* 09 · SOFTWARE ARCHITECTURE ─────────────────────────────────────── */}
          <section id="architecture" className="scroll-mt-28 ei-reveal">
            <SectionHeader code="09 // SYSTEM">Software Architecture</SectionHeader>
            <p className="text-sm text-on-surface-variant mb-5 max-w-4xl">
              Do not implement four unrelated engine simulators. Use one common engine interface with engine-specific parameter sets, maps, and subsystem models.
            </p>

            {/* Pipeline diagram */}
            <div className="border border-outline-variant/30 bg-surface/80 p-4 rounded-lg mb-4 shadow-[2px_2px_0px_#000000]">
              <div className="font-mono text-[9px] text-tertiary uppercase tracking-widest mb-3 font-bold">UNIFIED DATA FLOW PIPELINE</div>
              <div className="flex items-center gap-2 overflow-x-auto pb-1 font-mono text-[11px]">
                {PIPELINE_STAGES.map((stage, idx) => (
                  <div key={stage} className="flex items-center gap-2 shrink-0">
                    <div className="bg-surface-container-highest/60 border border-outline-variant/30 text-primary font-bold px-3 py-2 rounded hover:border-tertiary transition-colors">
                      {stage}
                    </div>
                    {idx < PIPELINE_STAGES.length - 1 && <ChevronRight size={13} className="text-tertiary shrink-0" />}
                  </div>
                ))}
              </div>
            </div>

            {/* Module table */}
            <div className="overflow-x-auto border border-outline-variant/30 rounded-lg bg-surface/80 shadow-[2px_2px_0px_#000000]">
              <table className="w-full text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-outline-variant/30 bg-black/40 text-[10px] text-tertiary uppercase tracking-wider">
                    <th className="p-3.5">Module</th>
                    <th className="p-3.5">Responsibility in AERO-SIM Architecture</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/20">
                  {[
                    ["Engine Definition", "Identity, architecture, dry mass, rated RPM, limits and metadata"],
                    ["Performance Map", "RPM / torque / power relationships and brake specific fuel consumption (BSFC)"],
                    ["Atmosphere", "Pressure, temperature, and air density calculation versus geometric altitude"],
                    ["Induction Model", "Naturally aspirated or turbocharged airflow / MAP regulation and turbocharger lag"],
                    ["Fuel Model", "Fuel mass flow rate, air-fuel ratio, and electronic fuel injection commands"],
                    ["Thermal Model", "Oil temperature, coolant/cylinder head temperature (CHT), and intake air heat exchange"],
                    ["AI Model", "Instantaneous torque, fuel flow, and thermal inference using trained neural networks"],
                    ["Telemetry & Validation", "Real-time UI streaming, parameter logging, confidence bounds, and safety limits"],
                  ].map(([mod, desc]) => (
                    <tr key={mod} className="hover:bg-white/[0.03] transition-colors">
                      <td className="p-3.5 font-bold text-primary">{mod}</td>
                      <td className="p-3.5 text-on-surface-variant">{desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* 10 · UAV SIMULATION ────────────────────────────────────────────── */}
          <section id="uav" className="scroll-mt-28 ei-reveal">
            <SectionHeader code="10 // AERODYNAMICS">UAV Simulation Implications</SectionHeader>
            <p className="text-sm text-on-surface-variant mb-5 max-w-4xl">
              Engine choice affects much more than top speed — mass changes CG envelopes, torque drives propeller thrust curves, altitude changes usable climb power, and fuel burn determines payload endurance.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
              {UAV_CASES.map((c) => (
                <div key={c.engine} className="border border-outline-variant/30 bg-surface/80 p-4 rounded-lg hover:border-tertiary/40 transition-colors">
                  <span className="font-mono text-[9px] text-tertiary uppercase tracking-wider block font-bold mb-1">{c.engine} USE CASE</span>
                  <h4 className="font-headline-display text-sm font-bold uppercase text-primary mb-2">{c.title}</h4>
                  <p className="text-on-surface-variant text-xs leading-relaxed">{c.desc}</p>
                </div>
              ))}
            </div>

            <div className="border border-tertiary/25 bg-tertiary/[0.06] p-4 rounded-lg font-mono text-xs flex items-start gap-3">
              <Plane size={16} className="text-tertiary shrink-0 mt-0.5" />
              <div>
                <strong className="text-tertiary uppercase tracking-wider block mb-1">CRITICAL PROPULSION RULE:</strong>
                <p className="text-on-surface-variant leading-relaxed text-[12px]">
                  <strong className="text-primary">Engine horsepower ≠ UAV thrust.</strong> The simulator must pass engine torque through the reduction
                  gearbox (e.g. 2.43:1 or 2.54:1) and propeller aerodynamic model (advance ratio J, blade pitch, and thrust coefficient C_T),
                  then calculate true airframe acceleration from thrust and aerodynamic drag.
                </p>
              </div>
            </div>
          </section>

          {/* 12 · VALIDATION ────────────────────────────────────────────────── */}
          <section id="validation" className="scroll-mt-28 ei-reveal">
            <SectionHeader code="12 // QA">Validation Strategy</SectionHeader>
            <p className="text-sm text-on-surface-variant mb-5 max-w-4xl">
              A professional aerospace simulator must distinguish between OEM certified specifications, measured test-bench data,
              interpolated operating maps, and AI predictions.
            </p>
            <div className="space-y-2">
              {VALIDATION_LEVELS.map((item) => (
                <div key={item.lvl} className="border border-outline-variant/30 bg-surface/80 p-4 rounded-lg flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:border-tertiary/30 transition-colors">
                  <div className="flex items-start sm:items-center gap-3">
                    <span className="px-2.5 py-1 rounded bg-tertiary/10 text-tertiary font-mono font-bold text-[10px] shrink-0">{item.lvl}</span>
                    <div>
                      <strong className="text-primary text-sm block">{item.title}</strong>
                      <p className="text-on-surface-variant text-xs mt-0.5 leading-relaxed">{item.desc}</p>
                    </div>
                  </div>
                  <span className="font-mono text-[9px] text-tertiary/70 uppercase tracking-widest shrink-0 self-end sm:self-center">PASSED // NOMINAL</span>
                </div>
              ))}
            </div>
          </section>

          {/* 13 · SOURCES & LAUNCH ──────────────────────────────────────────── */}
          <section id="sources" className="scroll-mt-28 ei-reveal">
            <SectionHeader code="13–14 // REFS">Sources & Launch</SectionHeader>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              {/* Sources */}
              <div className="border border-outline-variant/30 bg-surface/80 rounded-lg p-5">
                <h4 className="font-mono text-[10px] font-bold text-tertiary uppercase tracking-wider mb-4">PRIMARY TECHNICAL SOURCES</h4>
                <ul className="space-y-3 text-xs text-on-surface-variant leading-relaxed">
                  {[
                    { engine: "912 S/ULS", spec: "73.5 kW / 100 HP @ 5,800 RPM · 56.6 kg · 128 Nm @ 5,800 RPM" },
                    { engine: "914 F/UL", spec: "84.8 kW / 115 HP @ 5,800 RPM · 64.0 kg · 144 Nm @ 5,800 RPM" },
                    { engine: "915 iS A/iSC A", spec: "104 kW / 141 HP @ 5,800 RPM · 82.2 kg · intercooler & redundant EFI" },
                    { engine: "916 iS A/iSC", spec: "117 kW / 160 HP @ 5,800 RPM · 85.8 kg dry mass" },
                  ].map((src) => (
                    <li key={src.engine} className="flex items-start gap-2 border-b border-outline-variant/20 pb-3 last:border-b-0 last:pb-0">
                      <ExternalLink size={12} className="text-tertiary shrink-0 mt-0.5" />
                      <span><strong className="text-primary font-mono">Rotax {src.engine}:</strong> {src.spec}. Official OEM product specification sheet.</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Launch CTA */}
              <div className="border border-tertiary/30 bg-surface/80 rounded-lg p-5 flex flex-col justify-between shadow-[2px_2px_0px_#000000]">
                <div>
                  <div className="font-mono text-[10px] text-tertiary font-bold uppercase tracking-wider mb-2">AERO-SIM SYSTEM INTEGRATION</div>
                  <h3 className="font-headline-display text-xl font-bold text-primary uppercase mb-2 tracking-tight">Ready to Test Engine Dynamics?</h3>
                  <p className="text-sm text-on-surface-variant leading-relaxed">
                    Deploy your configured engine model directly into the live cockpit simulation. Test altitude throttle response,
                    monitor exhaust temps, and inspect neural digital-twin predictions in real time.
                  </p>
                </div>
                <div className="flex flex-col sm:flex-row gap-3 mt-6">
                  <Link
                    href="/engine"
                    className="border border-outline-variant/40 bg-surface-container-highest/60 text-primary hover:border-tertiary hover:text-tertiary text-[10px] font-bold uppercase px-4 py-3 rounded text-center font-mono tracking-wider transition-colors"
                  >
                    Propulsion Catalogue
                  </Link>
                  <Link
                    href="/simulate?engine=Rotax_914_ULF"
                    className="flex items-center justify-center gap-2 bg-tertiary text-black text-[10px] font-bold uppercase px-5 py-3 rounded text-center font-mono tracking-wider shadow-[2px_2px_0px_#000000] hover:-translate-y-0.5 hover:shadow-[4px_4px_0px_#000000] active:translate-y-0 transition-[transform,box-shadow]"
                  >
                    <Zap size={13} className="fill-black" /> Launch 914 Digital Twin
                  </Link>
                </div>
              </div>
            </div>
          </section>

        </div>
      </main>

      {/* ── FOOTER ────────────────────────────────────────────────────────────── */}
      <footer className="bg-surface-container-lowest w-full border-t border-outline-variant/30 relative z-10">
        <div className="flex flex-col md:flex-row justify-between items-center px-6 py-6 gap-4 max-w-7xl mx-auto w-full">
          <div className="text-[11px] tracking-[0.1em] font-bold uppercase text-tertiary text-center md:text-left flex items-center gap-2 font-mono">
            <span className="w-1.5 h-1.5 rounded-full bg-tertiary animate-pulse" />
            &copy; 2026 AERO-SIM PRECISION SYSTEMS // TR-ROTAX-2026-01
          </div>
          <div className="flex flex-wrap justify-center gap-5 font-mono text-[11px] uppercase tracking-wider text-on-surface-variant">
            <Link className="hover:text-tertiary transition-colors" href="/engine">Engine Selector</Link>
            <Link className="hover:text-tertiary transition-colors" href="/simulate">Live Cockpit</Link>
            <Link className="hover:text-tertiary transition-colors" href="/telemetry">Telemetry</Link>
            <Link className="hover:text-tertiary transition-colors" href="/about_us">About Us</Link>
          </div>
        </div>
      </footer>
    </>
  );
}
