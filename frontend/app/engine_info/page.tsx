"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import EngineViewer from "@/components/EngineViewer";
import EngineViewer912 from "@/components/EngineViewer912";
import EngineViewer915 from "@/components/EngineViewer915";
import EngineViewer916 from "@/components/EngineViewer916";
import { ArrowRight, ArrowUpRight, Check, Minus, X } from "lucide-react";
import "./engine-info.css";

/* ─── Data ────────────────────────────────────────────────────────────────────
   Specs: BRP-Rotax sheets (Sources). Limits, TBO and fault counts: physics v5
   (backend/physics_v5.py ENGINE_SPECS). AI numbers: each engine's v5 test card,
   about 900 held-out flights (validation_v5/artifacts/<engine>b_specialists). */
type Engine = {
  id: string;
  short: string;
  name: string;
  induction: string;
  hp: number;
  kw: number;
  massKg: number;
  torqueNm: number;
  rpm: string;
  critAlt: string;
  tbo: string;
  egt: string;
  faults: number;
  fuel: string;
  useCase: string;
  character: string;
  model: string;
  Viewer: React.ComponentType;
};

const ENGINES: Engine[] = [
  {
    id: "Rotax_912_ULS", short: "912 ULS", name: "Rotax 912 ULS", induction: "Naturally aspirated",
    hp: 100, kw: 73.5, massKg: 56.6, torqueNm: 128, rpm: "5,800", critAlt: "None", tbo: "2,000 h", egt: "880 °C",
    faults: 11, fuel: "Dual constant-depression carburettors",
    useCase: "Light, long-endurance airframes where empty weight matters most.",
    character: "Power follows air density. Climb and the engine breathes less air per stroke, so torque falls with altitude. The cleanest case for checking the core RPM, torque and power physics.",
    model: "RPM, throttle and ambient density into a naturally aspirated torque map, then power from torque and speed.",
    Viewer: EngineViewer912,
  },
  {
    id: "Rotax_914_ULF", short: "914 F/UL", name: "Rotax 914 F/UL", induction: "Turbocharged, TCU regulated",
    hp: 115, kw: 84.8, massKg: 64.0, torqueNm: 144, rpm: "5,800", critAlt: "15,000 ft", tbo: "2,000 h", egt: "950 °C",
    faults: 13, fuel: "Dual carburettors with turbo control unit",
    useCase: "Patrol airframes that need sea-level power up to mid altitude.",
    character: "The turbo raises manifold pressure above ambient, so power no longer tracks density until the wastegate runs out of authority at critical altitude.",
    model: "Manifold pressure is a live channel: boost limit, turbo lag and charge heat are modelled, not folded into one power number.",
    Viewer: EngineViewer,
  },
  {
    id: "Rotax_915_iS", short: "915 iS", name: "Rotax 915 iS", induction: "Turbocharged and intercooled",
    hp: 141, kw: 104.0, massKg: 82.2, torqueNm: 172, rpm: "5,800", critAlt: "15,000 ft", tbo: "1,200 h", egt: "950 °C",
    faults: 14, fuel: "Redundant electronic fuel injection",
    useCase: "Medium-altitude, long-endurance airframes carrying heavy payloads.",
    character: "The intercooler pulls compression heat out before combustion, which changes charge density and the boost-temperature balance. Redundant EFI meters fuel electronically.",
    model: "Torque and fuel flow from RPM, throttle, altitude, MAP and intake air temperature; brake power computed from torque.",
    Viewer: EngineViewer915,
  },
  {
    id: "Rotax_916_iS", short: "916 iS", name: "Rotax 916 iS", induction: "Turbocharged high-output",
    hp: 160, kw: 117.0, massKg: 85.8, torqueNm: 193, rpm: "5,800", critAlt: "15,000 ft", tbo: "2,000 h", egt: "950 °C",
    faults: 14, fuel: "Dual redundant EFI",
    useCase: "Maximum climb and payload; the most power per kilogram in the lineup.",
    character: "19 hp more than the 915 iS for 3.6 kg more mass. Same engine schema as the 915, with its own torque and fuel maps and thermal feedback.",
    model: "The shared turbo schema with high-resolution torque and fuel maps, altitude and MAP dependence, and thermal-state feedback.",
    Viewer: EngineViewer916,
  },
];

const HP_MAX = 160;
const ppk = (e: Engine) => e.hp / e.massKg;

// Test card per engine (held-out flights). Lower is better where `lower` is set.
const METRICS: { key: string; label: string; unit: string; lower?: boolean }[] = [
  { key: "det", label: "Detection", unit: "recall at 95% precision" },
  { key: "diag", label: "Fault naming", unit: "macro F1" },
  { key: "sev", label: "Severity error", unit: "on real faults", lower: true },
  { key: "sensor", label: "Sensor faults", unit: "macro F1" },
  { key: "health", label: "Health error", unit: "wear condition", lower: true },
  { key: "rul", label: "Remaining life", unit: "error, % of TBO", lower: true },
];
const SCORES: Record<string, { v: Record<string, string>; gates: string; miss?: string }> = {
  Rotax_912_ULS: { v: { det: "0.967", diag: "0.834", sev: "0.082", sensor: "0.901", health: "0.028", rul: "10.9%" }, gates: "5 of 6", miss: "severity bias" },
  Rotax_914_ULF: { v: { det: "0.978", diag: "0.842", sev: "0.073", sensor: "0.898", health: "0.029", rul: "9.3%" }, gates: "6 of 6" },
  Rotax_915_iS: { v: { det: "0.975", diag: "0.808", sev: "0.090", sensor: "0.900", health: "0.030", rul: "10.5%" }, gates: "5 of 6", miss: "severity bias" },
  Rotax_916_iS: { v: { det: "0.971", diag: "0.803", sev: "0.077", sensor: "0.892", health: "0.027", rul: "11.1%" }, gates: "4 of 6", miss: "air-filter recall, severity bias" },
};

const PIPELINE = [
  { title: "128-second window", body: "39 channels: flight state, atmosphere, thermal, powerplant, vibration, and 14 residuals against the healthy on-board twin." },
  { title: "Five specialists", body: "One network per question: is anything wrong, which fault and family, how bad, which sensor is lying, how worn." },
  { title: "Remaining life", body: "A gated gradient-boosted model reads the window, the engine's history and its inspection logbook." },
  { title: "Advisory", body: "Deterministic rules turn the outputs into a maintenance call a pilot can act on." },
];

const FORMULAS = [
  { f: "ω = 2π · RPM / 60", n: "Crankshaft speed in radians per second." },
  { f: "P = T · ω", n: "Shaft power from torque and angular speed." },
  { f: "P(kW) = T · RPM / 9550", n: "The form the simulator backend computes.", key: true },
  { f: "ρ = p / (R · T)", n: "Air density from pressure and temperature." },
  { f: "T = f(RPM, throttle, ρ, T_amb)", n: "Naturally aspirated torque falls with density." },
  { f: "T = f(RPM, throttle, MAP, T_iat, alt)", n: "Turbo torque holds until critical altitude." },
];

const VALIDATION = [
  { lvl: "L1", t: "Specifications", d: "Power, RPM, mass and torque from BRP-Rotax specification sheets.", s: "Sourced", ok: true },
  { lvl: "L2", t: "Physics", d: "19 checks: rated power, idle, fuel flow against Rotax 75% figures, boost to critical altitude, temperature and pressure limits.", s: "Tested", ok: true },
  { lvl: "L3", t: "Performance maps", d: "Torque and BSFC against dynamometer points.", s: "Not run: no dyno data", ok: false },
  { lvl: "L4", t: "Dynamics", d: "Throttle transients, spool-up lag and thermal inertia against test records.", s: "Not run: no test records", ok: false },
  { lvl: "L5", t: "AI models", d: "Every engine scored on about 900 unseen flights. 20 of 24 gates pass; each engine replaced its stand-in only after a 22-flight A/B.", s: "20 of 24 gates", ok: true },
  { lvl: "L6", t: "Live input", d: "The live window matches training exactly, on GPU and CPU.", s: "Parity tested", ok: true },
  { lvl: "L7", t: "Cross-implementation", d: "Simulink 914 v4 model against the Python twin: 26,000 steps, worst relative error 2e-15.", s: "Passed (914, v4)", ok: true },
];

const SECTIONS = [
  { id: "lineup", label: "Lineup" },
  { id: "engine", label: "Engine" },
  { id: "compare", label: "Compare" },
  { id: "ai", label: "AI twin" },
  { id: "physics", label: "Physics" },
  { id: "validation", label: "Validation" },
  { id: "sources", label: "Sources" },
];

const COMPARE_ROWS: [string, (x: Engine) => string][] = [
  ["Power", (x) => `${x.hp} hp · ${x.kw} kW`],
  ["Torque", (x) => `${x.torqueNm} Nm`],
  ["Dry mass", (x) => `${x.massKg} kg`],
  ["Induction", (x) => x.induction],
  ["vs 912", (x) => (x.hp === 100 ? "baseline" : `+${Math.round((x.hp / 100 - 1) * 100)}%`)],
  ["TBO", (x) => x.tbo],
];

/* ─── Page ────────────────────────────────────────────────────────────────── */
export default function EngineInfoPage() {
  const [sel, setSel] = useState(1); // the 914, the reference twin
  const [active, setActive] = useState(SECTIONS[0].id);
  const nav = useRef<HTMLElement>(null);
  const e = ENGINES[sel];

  // The section crossing the band under the sticky bars is the current one.
  useEffect(() => {
    const io = new IntersectionObserver(
      (es) => es.forEach((x) => x.isIntersecting && setActive(x.target.id)),
      { rootMargin: "-130px 0px -65% 0px" },
    );
    SECTIONS.forEach((s) => { const el = document.getElementById(s.id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);

  // On a phone the section nav scrolls sideways: keep the current tab in view without
  // scrollIntoView, which would cancel the page's own smooth scroll.
  useEffect(() => {
    const n = nav.current;
    const tab = n?.querySelector<HTMLElement>(`[data-sec="${active}"]`);
    if (n && tab && n.scrollWidth > n.clientWidth) n.scrollTo({ left: tab.offsetLeft - (n.clientWidth - tab.offsetWidth) / 2 });
  }, [active]);

  const go = (id: string) => {
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    document.getElementById(id)?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  };
  const pick = (i: number, jump = false) => { setSel(i); if (jump) go("engine"); };

  return (
    <>
      <Navbar />
      <main className="ei relative pt-16">
        {/* ── Hero: the selected engine in 3D beside one line on what this page is ── */}
        <header className="relative mx-auto grid max-w-7xl grid-cols-1 items-center gap-10 px-4 pb-14 pt-10 md:px-6 lg:grid-cols-12 lg:pt-16">
          <div className="lg:col-span-5">
            <h1 className="font-headline-display text-4xl font-bold uppercase leading-[0.95] tracking-tight text-primary md:text-6xl">
              Four Rotax engines, one twin.
            </h1>
            <p className="mt-5 max-w-[46ch] text-base leading-relaxed text-on-surface-variant">
              Specifications, physics and the AI models behind every engine in the simulator, with the scores each earned on flights it never saw.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href={`/simulate?engine=${e.id}`} className="ei-btn ei-btn-primary">
                Fly the {e.short} <ArrowRight size={16} aria-hidden />
              </Link>
              <button type="button" onClick={() => go("compare")} className="ei-btn ei-btn-ghost">Compare engines</button>
            </div>
          </div>

          <figure className="ei-stage lg:col-span-7">
            <div className="absolute inset-0"><e.Viewer key={e.id} /></div>
            <figcaption className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-4 p-5">
              <div>
                <p className="font-headline-display text-2xl font-bold tracking-tight text-primary">{e.name}</p>
                <p className="mt-0.5 text-sm text-on-surface-variant">{e.induction}</p>
              </div>
              <p className="font-mono text-sm text-primary"><span className="text-3xl font-bold text-tertiary">{e.hp}</span> hp</p>
            </figcaption>
          </figure>
        </header>

        {/* ── Section nav ── */}
        <nav ref={nav} aria-label="Report sections" className="sticky top-16 z-30 overflow-x-auto border-y border-outline-variant/40 bg-background/85 backdrop-blur-md">
          <div className="mx-auto flex max-w-7xl items-center gap-1 whitespace-nowrap px-4 md:px-6">
            {SECTIONS.map((s) => (
              <button key={s.id} type="button" data-sec={s.id} onClick={() => go(s.id)}
                aria-current={active === s.id ? "true" : undefined}
                className={`ei-tab ${active === s.id ? "is-on" : ""}`}>
                {s.label}
              </button>
            ))}
          </div>
        </nav>

        <div className="mx-auto max-w-7xl space-y-28 px-4 py-20 md:px-6">
          {/* ── Lineup: one power ladder; a rung selects its engine ── */}
          <section id="lineup" className="ei-sec">
            <h2 className="ei-h2">Power climbs with complexity.</h2>
            <p className="ei-lead">Each step up adds a system to model: a turbo, then an intercooler and injection, then more output from the same schema.</p>
            <ol className="mt-10 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-outline-variant/40 bg-outline-variant/40 lg:grid-cols-4">
              {ENGINES.map((x, i) => (
                <li key={x.id} className="bg-background">
                  <button type="button" onClick={() => pick(i, true)} aria-pressed={sel === i}
                    className={`ei-rung h-full w-full text-left ${sel === i ? "is-on" : ""}`}>
                    <span className="block h-1 origin-left bg-tertiary" style={{ transform: `scaleX(${x.hp / HP_MAX})` }} aria-hidden />
                    <span className="block p-5 md:p-6">
                      <span className="block font-headline-display text-lg font-bold tracking-tight text-primary">{x.short}</span>
                      <span className="mt-6 block font-mono text-5xl font-bold tracking-tight text-primary md:text-6xl">
                        {x.hp}<span className="ml-1 text-base font-medium text-on-surface-variant">hp</span>
                      </span>
                      <span className="mt-3 block font-mono text-sm text-on-surface-variant">{x.kw} kW · {x.massKg} kg</span>
                      <span className="mt-1 block text-sm text-on-surface-variant">{x.induction}</span>
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          </section>

          {/* ── Selected engine ── */}
          <section id="engine" className="ei-sec">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <h2 className="ei-h2 normal-case">{e.name}</h2>
              <div role="group" aria-label="Engine" className="ei-seg">
                {ENGINES.map((x, i) => (
                  <button key={x.id} type="button" aria-pressed={sel === i} onClick={() => pick(i)} className={sel === i ? "is-on" : ""}>{x.short}</button>
                ))}
              </div>
            </div>

            <div key={e.id} className="ei-swap mt-10 grid grid-cols-1 gap-12 lg:grid-cols-12">
              <div className="space-y-8 lg:col-span-5">
                <p className="text-lg leading-relaxed text-primary/90">{e.character}</p>
                <dl className="space-y-5 text-sm">
                  <div><dt className="ei-k">How the twin models it</dt><dd className="mt-1.5 leading-relaxed text-on-surface-variant">{e.model}</dd></div>
                  <div><dt className="ei-k">Fuel system</dt><dd className="mt-1.5 text-on-surface-variant">{e.fuel}</dd></div>
                  <div><dt className="ei-k">Best fit</dt><dd className="mt-1.5 text-on-surface-variant">{e.useCase}</dd></div>
                </dl>
                <Link href={`/simulate?engine=${e.id}`} className="ei-link">Fly the {e.short} <ArrowUpRight size={16} aria-hidden /></Link>
              </div>

              <div className="grid grid-cols-2 gap-4 self-start sm:grid-cols-3 lg:col-span-7">
                <Spec label="Rated power" value={`${e.kw}`} unit="kW" note={`${e.hp} hp at ${e.rpm} RPM`} big />
                <Spec label="Torque" value={`${e.torqueNm}`} unit="Nm" note={`at ${e.rpm} RPM`} big />
                <Spec label="Dry mass" value={`${e.massKg}`} unit="kg" note={`${ppk(e).toFixed(2)} hp/kg`} big />
                <Spec label="Overhaul (TBO)" value={e.tbo} />
                <Spec label="Full boost to" value={e.critAlt} />
                <Spec label="Faults the AI names" value={`${e.faults}`} />
                <div className="col-span-2 rounded-xl border border-outline-variant/40 p-5 sm:col-span-3">
                  <p className="ei-k">Limits in the twin</p>
                  <p className="mt-2 font-mono text-sm leading-relaxed text-primary/90">
                    EGT {e.egt}, CHT 135 °C, oil 130 °C, oil pressure min 0.8 bar below 3,500 RPM and 2.0 bar above
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* ── Compare ── */}
          <section id="compare" className="ei-sec">
            <h2 className="ei-h2">Side by side.</h2>
            <div className="mt-10 overflow-x-auto rounded-xl border border-outline-variant/40">
              <table className="w-full min-w-[640px] table-fixed text-left text-sm">
                <caption className="sr-only">Rotax engine comparison</caption>
                <thead>
                  <tr className="border-b border-outline-variant/40">
                    <th scope="col" className="ei-th w-44">Spec</th>
                    {ENGINES.map((x, i) => (
                      <th key={x.id} scope="col" className={`ei-th ${sel === i ? "text-tertiary" : "text-primary"}`}>
                        <button type="button" onClick={() => pick(i)} className="font-headline-display text-base font-bold normal-case tracking-tight">{x.short}</button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {COMPARE_ROWS.map(([label, f]) => (
                    <tr key={label} className="border-b border-outline-variant/25">
                      <th scope="row" className="ei-td font-sans font-medium text-on-surface-variant">{label}</th>
                      {ENGINES.map((x, i) => <td key={x.id} className={`ei-td ${sel === i ? "bg-tertiary/[0.06] text-primary" : "text-primary/85"}`}>{f(x)}</td>)}
                    </tr>
                  ))}
                  <tr>
                    <th scope="row" className="ei-td font-sans font-medium text-on-surface-variant">Power per kg</th>
                    {ENGINES.map((x, i) => (
                      <td key={x.id} className={`ei-td ${sel === i ? "bg-tertiary/[0.06]" : ""}`}>
                        <span className="text-primary">{ppk(x).toFixed(2)} hp/kg</span>
                        {/* the bar spans 1.6-1.9 hp/kg so the differences are visible */}
                        <span className="mt-2 block h-1 origin-left rounded-full bg-tertiary" style={{ transform: `scaleX(${(ppk(x) - 1.6) / 0.3})` }} aria-hidden />
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-sm text-on-surface-variant">Power-per-kg bars start at 1.6 hp/kg.</p>
          </section>

          {/* ── AI twin ── */}
          <section id="ai" className="ei-sec">
            <h2 className="ei-h2">Every engine has its own models.</h2>
            <p className="ei-lead">Scores on about 900 flights each engine never trained on. The gate bars are fixed, and an engine only replaced its 914 stand-in after beating it on the same 22 flights.</p>

            <ol className="mt-10 grid grid-cols-1 gap-px overflow-hidden rounded-xl border border-outline-variant/40 bg-outline-variant/40 md:grid-cols-4">
              {PIPELINE.map((p) => (
                <li key={p.title} className="bg-background p-5 md:p-6">
                  <p className="font-headline-display text-lg font-bold uppercase tracking-tight text-primary">{p.title}</p>
                  <p className="mt-2 text-sm leading-relaxed text-on-surface-variant">{p.body}</p>
                </li>
              ))}
            </ol>

            <div className="mt-6 overflow-x-auto rounded-xl border border-outline-variant/40">
              <table className="w-full min-w-[720px] table-fixed text-left text-sm">
                <caption className="sr-only">AI test scores per engine</caption>
                <thead>
                  <tr className="border-b border-outline-variant/40">
                    <th scope="col" className="ei-th w-56">Test card</th>
                    {ENGINES.map((x, i) => (
                      <th key={x.id} scope="col" className={`ei-th font-headline-display text-base font-bold normal-case tracking-tight ${sel === i ? "text-tertiary" : "text-primary"}`}>{x.short}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {METRICS.map((m) => (
                    <tr key={m.key} className="border-b border-outline-variant/25">
                      <th scope="row" className="ei-td">
                        <span className="block font-medium text-primary">{m.label}</span>
                        <span className="block font-normal text-on-surface-variant">{m.unit}{m.lower ? ", lower is better" : ""}</span>
                      </th>
                      {ENGINES.map((x, i) => (
                        <td key={x.id} className={`ei-td font-mono text-base ${sel === i ? "bg-tertiary/[0.06] text-primary" : "text-primary/85"}`}>{SCORES[x.id].v[m.key]}</td>
                      ))}
                    </tr>
                  ))}
                  <tr>
                    <th scope="row" className="ei-td font-medium text-primary">Gates passed</th>
                    {ENGINES.map((x, i) => {
                      const s = SCORES[x.id];
                      return (
                        <td key={x.id} className={`ei-td ${sel === i ? "bg-tertiary/[0.06]" : ""}`}>
                          <span className="inline-flex items-center gap-1.5 font-mono text-primary">
                            {s.miss ? <Minus size={14} className="text-on-surface-variant" aria-hidden /> : <Check size={14} className="text-tertiary" aria-hidden />}
                            {s.gates}
                          </span>
                          {s.miss && <span className="mt-1 block text-on-surface-variant">Missed: {s.miss}</span>}
                        </td>
                      );
                    })}
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* ── Physics ── */}
          <section id="physics" className="ei-sec grid grid-cols-1 gap-12 lg:grid-cols-12">
            <div className="lg:col-span-4">
              <h2 className="ei-h2">The model learns torque. Physics does the rest.</h2>
              <p className="ei-lead">Conservation laws are computed, not learned. Once the twin predicts torque from throttle, RPM and air state, shaft power follows exactly.</p>
            </div>
            <dl className="grid grid-cols-1 gap-x-10 gap-y-8 sm:grid-cols-2 lg:col-span-8">
              {FORMULAS.map((x) => (
                <div key={x.f}>
                  <dt className={`font-mono text-xl font-bold md:text-2xl ${x.key ? "text-tertiary" : "text-primary"}`}>{x.f}</dt>
                  <dd className="mt-2 text-sm text-on-surface-variant">{x.n}</dd>
                </div>
              ))}
            </dl>
          </section>

          {/* ── Validation ── */}
          <section id="validation" className="ei-sec">
            <h2 className="ei-h2">What has been checked, and what has not.</h2>
            <ol className="mt-10 border-l border-outline-variant/50">
              {VALIDATION.map((v) => (
                <li key={v.lvl} className="relative grid grid-cols-1 gap-2 pb-8 pl-8 last:pb-0 md:grid-cols-12 md:gap-6">
                  <span className={`absolute -left-[5px] top-1.5 h-2.5 w-2.5 rounded-full ${v.ok ? "bg-tertiary" : "border border-on-surface-variant bg-background"}`} aria-hidden />
                  <p className="font-mono text-sm text-on-surface-variant md:col-span-1">{v.lvl}</p>
                  <div className="md:col-span-8">
                    <p className="font-medium text-primary">{v.t}</p>
                    <p className="mt-1 text-sm leading-relaxed text-on-surface-variant">{v.d}</p>
                  </div>
                  <p className={`inline-flex items-start gap-1.5 text-sm md:col-span-3 md:justify-end ${v.ok ? "text-tertiary" : "text-on-surface-variant"}`}>
                    {v.ok ? <Check size={16} aria-hidden /> : <X size={16} aria-hidden />}{v.s}
                  </p>
                </li>
              ))}
            </ol>
          </section>

          {/* ── Sources and the way out ── */}
          <section id="sources" className="ei-sec grid grid-cols-1 gap-12 border-t border-outline-variant/40 pt-16 lg:grid-cols-12">
            <div className="lg:col-span-7">
              <h2 className="ei-h2">Sources</h2>
              <ul className="mt-6 space-y-3 text-sm text-on-surface-variant">
                {ENGINES.map((x) => (
                  <li key={x.id}><span className="font-medium text-primary">{x.name}:</span> {x.kw} kW / {x.hp} hp at {x.rpm} RPM, {x.massKg} kg, {x.torqueNm} Nm. BRP-Rotax product specification sheet.</li>
                ))}
                <li>Values BRP-Rotax does not publish are simulation-design choices, not certified figures.</li>
              </ul>
            </div>
            <div className="rounded-xl border border-tertiary/40 bg-tertiary/[0.06] p-6 md:p-8 lg:col-span-5">
              <p className="font-headline-display text-2xl font-bold uppercase tracking-tight text-primary">Take one up.</p>
              <p className="mt-2 text-sm leading-relaxed text-on-surface-variant">Fly any engine in the cockpit, inject a fault, and watch the twin and the AI respond.</p>
              <div className="mt-6 flex flex-wrap gap-3">
                <Link href={`/simulate?engine=${e.id}`} className="ei-btn ei-btn-primary">Fly the {e.short} <ArrowRight size={16} aria-hidden /></Link>
                <Link href="/engine" className="ei-btn ei-btn-ghost">Engine catalogue</Link>
              </div>
            </div>
          </section>
        </div>
      </main>
    </>
  );
}

function Spec({ label, value, unit, note, big }: { label: string; value: string; unit?: string; note?: string; big?: boolean }) {
  return (
    <div className="rounded-xl border border-outline-variant/40 p-5">
      <p className="ei-k">{label}</p>
      <p className={`mt-3 font-mono font-bold tracking-tight text-primary ${big ? "text-3xl" : "text-xl"}`}>
        {value}{unit && <span className="ml-1 text-sm font-medium text-on-surface-variant">{unit}</span>}
      </p>
      {note && <p className="mt-1 text-sm text-on-surface-variant">{note}</p>}
    </div>
  );
}
