"use client";

import Link from "next/link";
import type { FormEvent } from "react";
import { useState } from "react";
import emailjs from "@emailjs/browser";
import {
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  Cpu,
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
    title: "A living engine model",
    text: "Run a real-time UAV propulsion simulation with engine torque, propeller response, aerodynamics, fuel flow, thermal behavior, oil systems, vibration, wear, and automatically derived fault stress.",
  },
  {
    icon: Radio,
    label: "02 // LIVE TELEMETRY",
    title: "See every important signal",
    text: "The dashboard streams engine, propeller, aerodynamic, thermal, oil, vibration, and fault telemetry over a live connection so operating changes are visible as they happen.",
  },
  {
    icon: BrainCircuit,
    label: "03 // AI DIAGNOSTICS",
    title: "From signals to decisions",
    text: "Dedicated model heads detect faults, identify affected channels, estimate severity, and calculate remaining useful life for the Rotax 912, 914, 915, and 916 engine sets.",
  },
  {
    icon: Database,
    label: "04 // FLIGHT HISTORY",
    title: "Keep the story of a run",
    text: "Optional Supabase persistence stores simulation runs, telemetry, diagnostics, and final engine state, making it possible to review or resume a previous scenario.",
  },
];

const workflow = [
  "Choose a Rotax engine model from the technical catalogue.",
  "Set altitude, throttle, airspeed, and angle of attack.",
  "Start the simulation and watch the twin respond in real time.",
  "Review health, diagnosis, severity, and RUL as the AI window fills.",
];

const traceabilityPoints = [
  "Input features follow the model training contract.",
  "Each engine uses its own scaler and model checkpoints.",
  "Physics continues running if optional AI or persistence is offline.",
  "RUL is clearly presented as a simulated-timescale estimate.",
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
                AERO-SIM is a UAV engine digital-twin platform for exploring propulsion behavior,
                monitoring live operating data, and turning complex sensor signals into useful
                engineering decisions. It connects validated physics with model-driven diagnostics
                in one workspace.
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
                One system, four connected jobs
              </h2>
            </div>
            <p className="max-w-sm text-xs leading-relaxed text-on-surface-variant/70">
              Built for simulation, calibration, demonstration, and repeatable engineering analysis.
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
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
                UAV ENGINE DIGITAL TWIN // STATUS: ONLINE
              </div>
            </div>
          </div>
        </section>

        {/* System diagram */}
        <section className="relative z-10 mx-auto max-w-7xl px-4 md:px-6 py-12 md:py-16">
          <div className="mb-8 flex flex-col md:flex-row md:items-end md:justify-between gap-3">
            <div>
              <span className="font-mono text-[10px] font-bold tracking-[0.2em] text-tertiary uppercase">INSIDE THE MODEL</span>
              <h2 className="mt-1.5 font-headline-display text-2xl md:text-3xl font-bold uppercase text-white tracking-tight">
                Full system diagram
              </h2>
              <p className="mt-3 max-w-2xl text-sm leading-relaxed text-on-surface-variant">
                The complete signal flow of <span className="font-mono text-tertiary">UAV_Piston_Engine.slx</span> — the
                top-level architecture plus every one of the nine subsystems, redrawn from roughly 500 Simulink blocks.
              </p>
            </div>
            <a
              href="/uav-piston-engine-diagram.html"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex shrink-0 items-center gap-2 border border-outline-variant/50 bg-surface-container-highest/60 px-5 py-3 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-white rounded hover:border-tertiary hover:text-tertiary transition-colors"
            >
              Open full diagram <ArrowRight size={14} />
            </a>
          </div>

          <div className="bg-surface/80 border border-outline-variant/30 rounded-lg p-2 shadow-[2px_2px_0px_#000000]">
            <iframe
              src="/uav-piston-engine-diagram.html"
              title="UAV Piston Engine — Full System Diagram"
              loading="lazy"
              className="w-full h-[600px] md:h-[760px] rounded bg-background"
            />
          </div>
          <p className="mt-3 font-mono text-[10px] uppercase tracking-wider text-on-surface-variant/60">
            Scroll inside the frame to move through sections 01 &ndash; 09.
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
