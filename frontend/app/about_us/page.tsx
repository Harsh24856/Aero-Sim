"use client";

import Link from "next/link";
import type { FormEvent } from "react";
import { useState } from "react";
import emailjs from "@emailjs/browser";
import {
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  Copy,
  Cpu,
  Crosshair,
  Database,
  Gauge,
  Mail,
  Radio,
  ShieldCheck,
  Wrench,
  Zap,
} from "lucide-react";
import Navbar from "@/components/Navbar";

const capabilities = [
  {
    icon: Cpu,
    label: "01 // PHYSICS TWIN",
    title: "A real-physics engine",
    text: "Physics v4 models each Rotax engine layer by layer at 100 Hz - atmosphere, induction, combustion and knock, friction, propeller, shaft, thermal network, lubrication, electrical and vibration - with its own hardware: carburettors or injection, naturally aspirated or turbocharged. Weather, fuel and installation are live inputs.",
  },
  {
    icon: Copy,
    label: "02 // HEALTHY TWIN",
    title: "A perfect copy flies beside it",
    text: "Every second, a healthy copy of the same engine flies the same throttle, altitude and weather. The gap between what the instruments read and what the healthy engine would read is the clearest sign of damage there is - and the heat of the day cancels out.",
  },
  {
    icon: BrainCircuit,
    label: "03 // SIX AI HEADS",
    title: "From signals to decisions",
    text: "One dilated-TCN model per engine reads the last 128 seconds of 29 inputs and answers six questions: is anything wrong, which part, how bad, is it a sensor, how worn is the engine, and how many hours are left.",
  },
  {
    icon: Radio,
    label: "04 // ENGINE OR SENSOR",
    title: "A broken sender is not a broken engine",
    text: "Twelve instruments, each of which can fail six ways - bias, drift, stuck, spike, noise or dropout. The AI tells a failed thermocouple from a failing engine, so a bad sensor never grounds a healthy aircraft.",
  },
  {
    icon: Crosshair,
    label: "05 // MISSIONS",
    title: "Fly the sorties that matter",
    text: "MALE-UAV sorties - endurance, high altitude, hot and high, rapid throttle, in-flight fault isolation - set the weather and inject faults on the mission clock. Any fault can also be injected or removed by hand, with the ground truth shown beside the AI's answer.",
  },
  {
    icon: Database,
    label: "06 // ENGINE HISTORY",
    title: "Every engine keeps its hours",
    text: "Engines have hour meters that age 180 times faster than flight time, so wear builds up visibly during a flight. Flights, telemetry, maintenance events and reports are kept per engine, and any flight can be replayed second by second.",
  },
];

const workflow = [
  "Pick a Rotax 912, 914, 915 or 916 and its condition - a demo scenario, or one of your own engines continuing from its hour meter.",
  "Fly it by hand, or launch a mission sortie. Weather and fuel are set under Engine inputs.",
  "Watch the engine against its healthy twin. Inject a fault or break a sensor whenever you like.",
  "After 128 seconds the AI names what is wrong, how bad it is, whether it is a sensor, the engine's health and its hours left - beside the ground truth.",
  "Stop: the report, the replay and the engine's hour meter carry on to its next flight.",
];

const traceabilityPoints = [
  "The 29 model inputs are the same, in the same order, in the training pipeline, the AI service and the twin - the AI service refuses to start if they drift.",
  "Each engine ships its own models, scalers and manifest: per-fault cut-offs, test results, and the RUL error band at each stage of life.",
  "Every number is measured on flights the models never saw, split by flight so no flight is in two sets.",
  "Physics keeps flying if the AI service or the database is offline; a flight whose record could not be saved at takeoff is saved when the connection returns.",
  "Remaining life is in real engine hours against that engine's overhaul interval, and never rises during a flight.",
  "Flight time is real seconds, as the AI was trained on; engine hours run 180 times faster. The scale is one declared constant, recorded on every flight.",
];

// Test-flight results per engine (backend/models_v4/<engine>/manifest.json, docs/model_cards_v4.md).
const results = [
  { engine: "Rotax 914 ULF", auc: "0.879", f1: "0.61", health: "0.063", note: "" },
  { engine: "Rotax 912 ULS", auc: "0.925", f1: "0.73", health: "0.075", note: "" },
  { engine: "Rotax 915 iS", auc: "0.900", f1: "0.63", health: "0.063", note: "" },
  { engine: "Rotax 916 iS", auc: "-", f1: "-", health: "-", note: "Training; flies on the 914's models meanwhile" },
];

// physics_v4 layer chain, in signal order, with the engines that have each block.
const layers: { name: string; on?: string }[] = [
  { name: "Atmosphere" }, { name: "Air filter" },
  { name: "Turbocharger + wastegate", on: "914 · 915 · 916" }, { name: "Intercooler", on: "915 · 916" },
  { name: "Throttle body" }, { name: "Volumetric efficiency" }, { name: "Air mass flow" },
  { name: "Carburettor", on: "912 · 914" }, { name: "Fuel injection", on: "915 · 916" },
  { name: "Combustion" }, { name: "Otto cycle + knock" }, { name: "Friction + pumping" },
  { name: "Exhaust" }, { name: "Propeller" }, { name: "Shaft" },
  { name: "Thermal network" }, { name: "Lubrication" }, { name: "Electrical" },
  { name: "Aerodynamics" }, { name: "Vibration" }, { name: "Life + margins" },
];

const EMAILJS_SERVICE_ID = process.env.NEXT_PUBLIC_EMAILJS_SERVICE_ID;
const EMAILJS_TEMPLATE_ID = process.env.NEXT_PUBLIC_EMAILJS_TEMPLATE_ID;
const EMAILJS_PUBLIC_KEY = process.env.NEXT_PUBLIC_EMAILJS_PUBLIC_KEY;

function ContactForm() {
  const [isSending, setIsSending] = useState(false);
  const [status, setStatus] = useState<"idle" | "success" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState("");

  async function handleContactSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStatus("idle");
    setErrorMessage("");

    if (!EMAILJS_SERVICE_ID || !EMAILJS_TEMPLATE_ID || !EMAILJS_PUBLIC_KEY) {
      setStatus("error");
      setErrorMessage("Email service is not configured yet. Please contact the site administrator.");
      return;
    }

    setIsSending(true);
    const formData = new FormData(event.currentTarget);
    const name = String(formData.get("name") ?? "");
    const email = String(formData.get("email") ?? "");
    const subject = String(formData.get("subject") ?? "AERO-SIM enquiry");
    const message = String(formData.get("message") ?? "");

    try {
      await emailjs.send(
        EMAILJS_SERVICE_ID,
        EMAILJS_TEMPLATE_ID,
        { title: subject, name, email, message, time: new Date().toLocaleString() },
        { publicKey: EMAILJS_PUBLIC_KEY },
      );
      event.currentTarget.reset();
      setStatus("success");
    } catch (error) {
      setStatus("error");
      setErrorMessage(error instanceof Error ? error.message : "EmailJS could not send this message.");
    } finally {
      setIsSending(false);
    }
  }

  const inputClass =
    "mt-1.5 w-full border border-outline-variant/50 bg-surface-container-highest/60 px-3 py-2.5 text-sm text-white placeholder:text-on-surface-variant/40 focus:outline-none focus:border-tertiary focus:ring-1 focus:ring-tertiary/30 rounded transition-colors";
  const labelClass = "block font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-on-surface-variant";

  return (
    <form onSubmit={handleContactSubmit} className="bg-surface/80 border border-outline-variant/30 rounded-lg p-6 shadow-[2px_2px_0px_#000000] grid gap-4 md:grid-cols-2">
      <label className="block">
        <span className={labelClass}>Name</span>
        <input required name="name" type="text" placeholder="Your name" className={inputClass} />
      </label>
      <label className="block">
        <span className={labelClass}>Email</span>
        <input required name="email" type="email" placeholder="you@example.com" className={inputClass} />
      </label>
      <label className="block md:col-span-2">
        <span className={labelClass}>Subject</span>
        <input required name="subject" type="text" placeholder="What would you like to discuss?" className={inputClass} />
      </label>
      <label className="block md:col-span-2">
        <span className={labelClass}>Message</span>
        <textarea
          required name="message" rows={5}
          placeholder="Tell us about your project, engine model, or question..."
          className={`${inputClass} resize-y`}
        />
      </label>
      <div className="flex flex-col justify-between gap-3 md:col-span-2 sm:flex-row sm:items-center">
        <div className="text-[12px] leading-relaxed">
          {status === "success" && <p className="text-emerald-400">Message sent. We&apos;ll get back to you soon.</p>}
          {status === "error" && <p className="text-red-400">{errorMessage}</p>}
          {status === "idle" && <p className="text-on-surface-variant/60 font-mono text-[11px]">Delivered securely via EmailJS.</p>}
        </div>
        <button
          disabled={isSending}
          type="submit"
          className="flex items-center justify-center gap-2 bg-tertiary px-6 py-3 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-black rounded shadow-[2px_2px_0px_#000000] hover:-translate-y-0.5 hover:shadow-[4px_4px_0px_#000000] active:translate-y-0 active:shadow-[1px_1px_0px_#000000] transition-all disabled:opacity-60 disabled:cursor-wait"
        >
          {isSending ? "Sending..." : "Send Message"}
          <ArrowRight size={14} />
        </button>
      </div>
    </form>
  );
}

export default function AboutUsPage() {
  return (
    <>
      <Navbar />

      <main className="relative flex-grow pt-16 min-h-screen">
        <div className="pointer-events-none absolute inset-0 grid-bg opacity-50 z-0" />

        {/* Hero */}
        <header className="relative z-10 border-b border-outline-variant/30">
          <div className="mx-auto max-w-7xl px-4 md:px-6 py-14 md:py-20">
            <div className="max-w-4xl">
              <div className="mb-3 flex items-center gap-3">
                <span className="h-px w-8 bg-tertiary" />
                <span className="font-mono text-[10px] font-bold tracking-[0.24em] text-tertiary uppercase">
                  ABOUT AERO-SIM // SYSTEM BRIEF
                </span>
              </div>
              <h1 className="font-headline-display text-[32px] sm:text-[42px] md:text-[54px] font-extrabold uppercase leading-[1.08] tracking-tight text-white">
                Understand the machine
                <span className="block text-tertiary">before it becomes a problem.</span>
              </h1>
              <p className="mt-5 max-w-3xl text-sm md:text-base leading-relaxed text-on-surface-variant">
                AERO-SIM is a digital twin of the Rotax engines that fly MALE-class UAVs. A
                real-physics engine flies beside a healthy copy of itself, and an AI watching both
                names the failing part, tells a broken sensor from a broken engine, and counts down
                the hours the engine has left.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link
                  href="/engine_info"
                  className="inline-flex items-center gap-2 bg-tertiary px-6 py-3 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-black rounded shadow-[2px_2px_0px_#000000] hover:-translate-y-0.5 hover:shadow-[4px_4px_0px_#000000] active:translate-y-0 active:shadow-[1px_1px_0px_#000000] transition-all"
                >
                  Explore engine data <ArrowRight size={14} />
                </Link>
                <Link
                  href="/engine"
                  className="inline-flex items-center gap-2 border border-outline-variant/50 bg-surface-container-highest/60 px-6 py-3 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-white rounded hover:border-tertiary hover:text-tertiary transition-colors"
                >
                  Open simulator <Gauge size={14} />
                </Link>
              </div>
            </div>
          </div>
        </header>

        {/* Capabilities */}
        <section className="relative z-10 mx-auto max-w-7xl px-4 md:px-6 py-12 md:py-16">
          <div className="mb-8 flex flex-col md:flex-row md:items-end md:justify-between gap-3">
            <div>
              <span className="font-mono text-[10px] font-bold tracking-[0.2em] text-tertiary uppercase">WHAT THE PLATFORM DOES</span>
              <h2 className="mt-1.5 font-headline-display text-2xl md:text-3xl font-bold uppercase text-white tracking-tight">
                One system, six connected jobs
              </h2>
            </div>
            <p className="max-w-sm text-xs leading-relaxed text-on-surface-variant/70">
              Built for simulation, calibration, demonstration, and repeatable engineering analysis.
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {capabilities.map((cap) => {
              const Icon = cap.icon;
              return (
                <article
                  key={cap.label}
                  className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 shadow-[2px_2px_0px_#000000] hover:border-tertiary/50 hover:-translate-y-0.5 hover:shadow-[4px_4px_0px_#000000] transition-all min-h-[220px] flex flex-col"
                >
                  <Icon size={20} className="text-tertiary mb-5" />
                  <p className="font-mono text-[9px] font-bold tracking-[0.16em] text-tertiary/70 uppercase mb-1">{cap.label}</p>
                  <h3 className="font-headline-display text-base font-bold uppercase text-white mb-2">{cap.title}</h3>
                  <p className="text-xs leading-relaxed text-on-surface-variant/80 mt-auto">{cap.text}</p>
                </article>
              );
            })}
          </div>
        </section>

        {/* Workflow + Traceability */}
        <section className="relative z-10 border-y border-outline-variant/30 bg-black/20">
          <div className="mx-auto grid max-w-7xl gap-8 px-4 md:px-6 py-12 md:py-16 md:grid-cols-[1fr_0.85fr]">
            <div>
              <span className="font-mono text-[10px] font-bold tracking-[0.2em] text-tertiary uppercase">HOW A SESSION WORKS</span>
              <h2 className="mt-1.5 font-headline-display text-2xl md:text-3xl font-bold uppercase text-white tracking-tight mb-7">
                From setup to insight
              </h2>
              <div className="space-y-4">
                {workflow.map((step, index) => (
                  <div key={step} className="flex items-start gap-4 border-b border-outline-variant/25 pb-4 last:border-b-0">
                    <span className="font-mono text-sm font-bold text-tertiary shrink-0">0{index + 1}</span>
                    <p className="text-sm leading-relaxed text-on-surface-variant">{step}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-6 shadow-[2px_2px_0px_#000000]">
              <div className="flex items-center gap-3 border-b border-outline-variant/25 pb-4 mb-5">
                <ShieldCheck size={18} className="text-tertiary shrink-0" />
                <h3 className="font-mono text-xs font-bold uppercase tracking-[0.14em] text-white">Built around traceability</h3>
              </div>
              <ul className="space-y-3.5">
                {traceabilityPoints.map((point) => (
                  <li key={point} className="flex items-start gap-3 text-xs leading-relaxed text-on-surface-variant">
                    <CheckCircle2 size={14} className="mt-0.5 shrink-0 text-tertiary" />
                    {point}
                  </li>
                ))}
              </ul>

              <div className="mt-6 pt-5 border-t border-outline-variant/25 flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-on-surface-variant/60">
                <Wrench size={13} className="text-tertiary" />
                PHYSICS V4 // FOUR ROTAX ENGINES
              </div>
            </div>
          </div>
        </section>

        {/* Runtime architecture - what actually runs, as opposed to the plant model below */}
        <section className="relative z-10 mx-auto max-w-7xl px-4 md:px-6 py-12 md:py-16">
          <div className="mb-8">
            <span className="font-mono text-[10px] font-bold tracking-[0.2em] text-tertiary uppercase">HOW IT FITS TOGETHER</span>
            <h2 className="mt-1.5 font-headline-display text-2xl md:text-3xl font-bold uppercase text-white tracking-tight">
              System architecture
            </h2>
            <p className="mt-3 max-w-3xl text-sm leading-relaxed text-on-surface-variant">
              Three processes and one shared contract. The physics twin flies the engine and its
              healthy copy; the inference service turns the last 128 seconds into six answers; the
              cockpit shows both, next to the ground truth the AI never sees.
            </p>
          </div>

          <div className="grid gap-4 lg:grid-cols-4">
            {[
              {
                n: "01", name: "Physics twin", file: "backend/twin_v4.py",
                lines: ["Rotax 912 / 914 / 915 / 916", "100 Hz physics, 1 Hz instruments", "healthy twin, same inputs", "wear on the engine-hour clock"],
                note: "Each engine is also a Simulink model built from primitive blocks, matching the Python to about 2e-15.",
              },
              {
                n: "02", name: "Simulation loop", file: "backend/main.py",
                lines: ["WebSocket at 20 Hz", "bounded AI queue + breaker", "fault injection and removal", "missions, CAN bus, Supabase"],
                note: "Guards every stage: a failure degrades one feature, not the flight.",
              },
              {
                n: "03", name: "Inference service", file: "backend/aiv4.py",
                lines: ["dilated TCN, six heads", "128 s x 29-input window", "per-fault cut-offs", "on the Metal GPU"],
                note: "Separate process and environment, so the test numbers are what serves.",
              },
              {
                n: "04", name: "Cockpit", file: "frontend/",
                lines: ["gauges + engine inputs", "engine vs healthy twin", "AI beside ground truth", "missions, replay, reports"],
                note: "Shows the model's own numbers, and says when it is not ready.",
              },
            ].map((b) => (
              <div key={b.n} className="bg-surface/80 border border-outline-variant/30 rounded-lg p-5 shadow-[2px_2px_0px_#000000] flex flex-col">
                <div className="flex items-baseline justify-between">
                  <span className="font-mono text-[10px] tracking-[0.18em] text-tertiary font-bold">{b.n}</span>
                  <span className="font-mono text-[9px] text-on-surface-variant/50">{b.file}</span>
                </div>
                <h3 className="font-headline-display text-lg text-white uppercase tracking-tight mt-2">{b.name}</h3>
                <ul className="mt-3 space-y-1.5">
                  {b.lines.map((l) => (
                    <li key={l} className="font-mono text-[11px] text-on-surface-variant leading-snug flex gap-2">
                      <span className="text-tertiary/60">&rsaquo;</span>{l}
                    </li>
                  ))}
                </ul>
                <p className="mt-4 pt-3 border-t border-outline-variant/20 text-[11px] leading-relaxed text-on-surface-variant/70">{b.note}</p>
              </div>
            ))}
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="bg-surface/60 border border-outline-variant/25 rounded-lg p-5">
              <h3 className="font-mono text-[10px] uppercase tracking-[0.16em] text-tertiary font-bold">Physics against physics</h3>
              <p className="mt-2 text-[13px] leading-relaxed text-on-surface-variant">
                The healthy twin is the same engine with no wear and no faults, flown on the same
                inputs. Its difference from the instruments on six channels - EGT, CHT, oil
                temperature and pressure, rpm and fuel flow - needs no model to read, so the cockpit
                charts it directly; the AI reads the same six residuals as part of its input.
              </p>
            </div>
            <div className="bg-surface/60 border border-outline-variant/25 rounded-lg p-5">
              <h3 className="font-mono text-[10px] uppercase tracking-[0.16em] text-tertiary font-bold">On real hardware</h3>
              <p className="mt-2 text-[13px] leading-relaxed text-on-surface-variant">
                <span className="font-mono text-white">can_ingest.py</span> bridges a SocketCAN bus into the
                same endpoints the simulator uses, so the twin can shadow a real engine without any
                change to the pipeline that reads it. While the aircraft is on the bus it owns the
                set-points, and the cockpit's controls lock.
              </p>
            </div>
          </div>
        </section>

        {/* Results */}
        <section className="relative z-10 mx-auto max-w-7xl px-4 md:px-6 py-12 md:py-16">
          <div className="mb-8">
            <span className="font-mono text-[10px] font-bold tracking-[0.2em] text-tertiary uppercase">HOW WELL IT WORKS</span>
            <h2 className="mt-1.5 font-headline-display text-2xl md:text-3xl font-bold uppercase text-white tracking-tight">
              Results on unseen flights
            </h2>
            <p className="mt-3 max-w-3xl text-sm leading-relaxed text-on-surface-variant">
              About 4,000 simulated flights per engine, split by flight. Every figure is from test
              flights the models never saw during training or selection.
            </p>
          </div>
          <div className="overflow-x-auto bg-surface/80 border border-outline-variant/30 rounded-lg shadow-[2px_2px_0px_#000000]">
            <table className="w-full min-w-[560px] text-left">
              <thead>
                <tr className="border-b border-outline-variant/30 font-mono text-[10px] uppercase tracking-[0.14em] text-on-surface-variant">
                  <th className="px-5 py-3 font-bold">Engine</th>
                  <th className="px-5 py-3 font-bold">Anything wrong? <span className="normal-case tracking-normal text-on-surface-variant/60">AUC</span></th>
                  <th className="px-5 py-3 font-bold">Which part? <span className="normal-case tracking-normal text-on-surface-variant/60">F1</span></th>
                  <th className="px-5 py-3 font-bold">How worn? <span className="normal-case tracking-normal text-on-surface-variant/60">error</span></th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.engine} className="border-b border-outline-variant/20 last:border-b-0">
                    <td className="px-5 py-3 font-headline-display text-sm uppercase text-white">{r.engine}</td>
                    {r.note ? (
                      <td colSpan={3} className="px-5 py-3 text-[12px] text-on-surface-variant/70">{r.note}</td>
                    ) : (
                      <>
                        <td className="px-5 py-3 font-mono text-sm text-tertiary">{r.auc}</td>
                        <td className="px-5 py-3 font-mono text-sm text-tertiary">{r.f1}</td>
                        <td className="px-5 py-3 font-mono text-sm text-tertiary">{r.health}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 max-w-3xl text-[12px] leading-relaxed text-on-surface-variant/70">
            Said plainly: remaining life is the open problem. On engines that wear out before their
            overhaul, the estimate made from what an aircraft can actually measure is off by about 19-21% of
            the overhaul interval - better than counting down the calendar, but not yet our 15% target.
          </p>
        </section>

        {/* Inside the engine */}
        <section className="relative z-10 mx-auto max-w-7xl px-4 md:px-6 py-12 md:py-16">
          <div className="mb-8">
            <span className="font-mono text-[10px] font-bold tracking-[0.2em] text-tertiary uppercase">INSIDE THE ENGINE</span>
            <h2 className="mt-1.5 font-headline-display text-2xl md:text-3xl font-bold uppercase text-white tracking-tight">
              Physics v4, layer by layer
            </h2>
            <p className="mt-3 max-w-3xl text-sm leading-relaxed text-on-surface-variant">
              The signal chain every engine runs, in order. Each engine&rsquo;s hardware decides its
              blocks: the 912 breathes without a turbo, the 914 adds one, and the 915 and 916 add an
              intercooler and fuel injection. The same chain exists block for block in Simulink.
            </p>
          </div>
          <ol className="flex flex-wrap items-center gap-2">
            {layers.map((l, i) => (
              <li key={l.name} className="flex items-center gap-2">
                <span className={`rounded border px-3 py-2 ${l.on ? "border-tertiary/40 bg-tertiary/5" : "border-outline-variant/30 bg-surface/80"}`}>
                  <span className="block font-mono text-[11px] text-white">{l.name}</span>
                  {l.on && <span className="block font-mono text-[9px] text-tertiary/80">{l.on} only</span>}
                </span>
                {i < layers.length - 1 && <span className="text-tertiary/50">&rsaquo;</span>}
              </li>
            ))}
          </ol>
          <p className="mt-4 font-mono text-[10px] uppercase tracking-wider text-on-surface-variant/60">
            The earlier (v3) plant model is still here:{" "}
            <a href="/uav-piston-engine-diagram.html" target="_blank" rel="noopener noreferrer" className="text-tertiary hover:underline">
              open its full diagram
            </a>
          </p>
        </section>

        {/* Contact CTA + Direct */}
        <section className="relative z-10 mx-auto max-w-7xl px-4 md:px-6 py-12 md:py-16 grid grid-cols-1 md:grid-cols-[1fr_1.2fr] gap-8">
          <div>
            <span className="font-mono text-[10px] font-bold tracking-[0.2em] text-tertiary uppercase">CONTACT AERO-SIM</span>
            <h2 className="mt-1.5 font-headline-display text-3xl font-bold uppercase text-white tracking-tight">
              Let&apos;s talk propulsion.
            </h2>
            <p className="mt-4 max-w-md text-sm leading-relaxed text-on-surface-variant">
              Have a question about the digital twin, engine data, model behavior, or a potential collaboration?
              Send us a note and include the engine model or simulation scenario you are working with.
            </p>

            <div className="mt-8 bg-surface/80 border border-outline-variant/30 rounded-lg p-5 shadow-[2px_2px_0px_#000000]">
              <p className="font-mono text-[9px] font-bold tracking-[0.18em] text-tertiary/70 uppercase mb-2">PRIMARY CONTACT</p>
              <a
                href="mailto:contact@aero-sim.com"
                className="flex items-center gap-3 text-base font-bold text-white hover:text-tertiary transition-colors"
              >
                <Mail size={18} className="text-tertiary shrink-0" />
                contact@aero-sim.com
              </a>
              <p className="mt-1.5 text-xs text-on-surface-variant/70">For technical questions, demonstrations, and project collaboration.</p>
            </div>
          </div>

          <ContactForm />
        </section>
      </main>

      <footer className="bg-surface-container-lowest w-full border-t border-outline-variant/30 z-10 relative">
        <div className="flex flex-col md:flex-row justify-between items-center px-6 py-6 gap-4 max-w-7xl mx-auto w-full">
          <div className="text-[11px] tracking-[0.1em] font-bold uppercase text-tertiary text-center md:text-left flex items-center gap-3 font-mono">
            <span className="w-1.5 h-1.5 rounded-full bg-tertiary animate-pulse" />
            <span>&copy; 2026 AERO-SIM // Precision UAV Simulation &amp; Engine Intelligence</span>
          </div>
          <div className="flex flex-wrap justify-center gap-5 font-mono text-[11px] uppercase tracking-wider">
            <Link className="text-on-surface-variant hover:text-tertiary transition-colors" href="/engine">Propulsion Catalogue</Link>
            <Link className="text-on-surface-variant hover:text-tertiary transition-colors" href="/engine_info">Spec Sheets</Link>
            <Link className="text-on-surface-variant hover:text-tertiary transition-colors" href="/simulate">Live Cockpit</Link>
          </div>
        </div>
      </footer>
    </>
  );
}
