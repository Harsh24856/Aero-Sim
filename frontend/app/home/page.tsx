import Image from "next/image";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import Navbar from "@/components/Navbar";
import BlackoutPlane from "@/components/BlackoutPlane";

// Every number on this page is real: engine specs from backend/physics_v4.ENGINE_SPECS_V4,
// AI results from validation_v5/artifacts/914b_specialists/model_card.md (900 held-out
// test flights of the Rotax 914, v5 models). Update both when they change.
const QUESTIONS: { q: string; a: string }[] = [
  { q: "Anything wrong?", a: "Catches 95.7% of faults while 95% of its alarms are real." },
  { q: "Which part?", a: "Names one of 13 component faults, or the fault family when the part is not yet clear." },
  { q: "Engine or sensor?", a: "Checks 12 instruments for bias, drift, stuck, spike, noise and dropout, with 0.10% false alarms." },
  { q: "How worn?", a: "Reads the engine's wear condition to within 3.3%." },
  { q: "Hours left?", a: "Counts down to overhaul, within 10.9% of TBO on engines wearing out early." },
];

const ENGINES: { id: string; name: string; kw: number; tbo: number; turbo: boolean }[] = [
  { id: "Rotax_912_ULS", name: "Rotax 912 ULS", kw: 73.5, tbo: 2000, turbo: false },
  { id: "Rotax_914_ULF", name: "Rotax 914 ULF", kw: 84.5, tbo: 2000, turbo: true },
  { id: "Rotax_915_iS", name: "Rotax 915 iS", kw: 104, tbo: 1200, turbo: true },
  { id: "Rotax_916_iS", name: "Rotax 916 iS", kw: 118, tbo: 2000, turbo: true },
];

const FLOW: { verb: string; body: string; href: string; link: string }[] = [
  { verb: "Pick an engine", body: "Start healthy, or from a preset with wear or a fault already under way.", href: "/engine", link: "Start a flight" },
  { verb: "Fly and break it", body: "Set altitude, speed and throttle, then inject engine or sensor faults mid-flight.", href: "/mission", link: "Missions" },
  { verb: "Read the report", body: "Replay any flight second by second, with the AI's calls beside the truth.", href: "/telemetry", link: "Past flights" },
];

const cta = "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded px-6 py-3 text-[11px] font-bold uppercase tracking-[0.1em] transition-all duration-200 active:translate-y-px";

export default function HomePage() {
  return (
    <>
      <Navbar />

      <main className="relative flex-grow pt-16">
        <div className="pointer-events-none absolute inset-0 z-0 grid-bg opacity-50" />

        {/* Hero: message left, the aircraft right; stacks on mobile. */}
        <section className="relative z-10 mx-auto grid w-full max-w-7xl grid-cols-1 items-center gap-8 px-4 pb-12 pt-10 md:px-6 lg:min-h-[calc(100dvh-4rem)] lg:grid-cols-[5fr_7fr] lg:pt-0">
          <div className="home-reveal flex flex-col items-start gap-5">
            <h1 className="max-w-[18ch] font-headline-display text-4xl font-bold leading-[1.05] tracking-tight text-primary md:text-5xl lg:text-[3.25rem]">
              A digital twin for UAV piston engines
            </h1>
            <p className="max-w-[46ch] text-base leading-relaxed text-on-surface-variant">
              Fly a Rotax engine, inject faults, and watch the AI name the part, its severity and the hours left.
            </p>
            <div className="flex w-full flex-col gap-3 pt-2 sm:w-auto sm:flex-row">
              <Link href="/engine" className={`${cta} bg-tertiary text-black shadow-[2px_2px_0px_#000000] hover:-translate-y-0.5 hover:shadow-[4px_4px_0px_#000000]`}>
                Start a flight <ArrowRight size={14} strokeWidth={2} />
              </Link>
              <Link href="/telemetry" className={`${cta} border border-outline-variant/60 bg-surface-container-highest/60 text-primary hover:border-tertiary hover:text-tertiary`}>
                Past flights
              </Link>
            </div>
          </div>
          <div className="home-reveal w-full [animation-delay:120ms]">
            <BlackoutPlane frameClassName="aspect-[4/3] lg:aspect-[5/4]" />
          </div>
        </section>

        {/* What the AI answers: the real v5 panel beside the five questions it answers. */}
        <section className="home-scroll relative z-10 mx-auto grid w-full max-w-7xl grid-cols-1 gap-10 px-4 py-16 md:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] md:px-6 md:py-24">
          <figure className="mx-auto w-full max-w-[240px] self-start md:sticky md:top-24 md:mx-0 md:justify-self-end">
            <Image
              src="/home/diagnostics-v5.png"
              alt="The cockpit's diagnostics panel during a flight with cooling degradation: fault detected, cooling degradation at 24% effective severity, all twelve sensors fine, 535 hours left, wear-limited."
              width={456}
              height={1512}
              sizes="240px"
              className="h-auto w-full rounded border border-[#4c3328] shadow-[0_24px_60px_rgba(20,8,2,0.6)]"
            />
            <figcaption className="mt-3 text-[12px] leading-relaxed text-on-surface-variant">
              The live panel, mid-flight, with a cooling fault injected.
            </figcaption>
          </figure>
          <div className="flex flex-col justify-center">
            <h2 className="font-headline-display text-3xl font-bold leading-tight tracking-tight text-primary md:text-4xl">
              Five questions, answered every second
            </h2>
            <ol className="mt-8 space-y-6">
              {QUESTIONS.map(({ q, a }) => (
                <li key={q} className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)] gap-4 md:grid-cols-[minmax(0,11rem)_minmax(0,1fr)]">
                  <span className="font-headline-md text-base font-semibold text-tertiary">{q}</span>
                  <span className="max-w-[52ch] text-[15px] leading-relaxed text-on-surface-variant">{a}</span>
                </li>
              ))}
            </ol>
            <p className="mt-8 text-[12px] text-on-surface-variant/80">
              Measured on 900 test flights the models never saw.
            </p>
          </div>
        </section>

        {/* The engines: one tile each, straight into the simulator. */}
        <section className="home-scroll relative z-10 mx-auto w-full max-w-7xl px-4 py-16 md:px-6 md:py-20">
          <h2 className="font-headline-display text-3xl font-bold leading-tight tracking-tight text-primary md:text-4xl">
            Four Rotax engines
          </h2>
          <div className="mt-8 grid grid-cols-1 gap-px overflow-hidden rounded border border-outline-variant/40 bg-outline-variant/40 sm:grid-cols-2 lg:grid-cols-4">
            {ENGINES.map((e) => (
              <Link
                key={e.id}
                href={`/simulate?engine=${e.id}`}
                className="group flex flex-col gap-4 bg-[#0e0c0b] p-6 transition-colors duration-200 hover:bg-[#1a1210]"
              >
                <span className="font-headline-md text-lg font-semibold text-primary">{e.name}</span>
                <dl className="grid grid-cols-3 gap-2 font-data-sm text-[12px]">
                  <div><dt className="text-on-surface-variant">Power</dt><dd className="text-primary">{e.kw} kW</dd></div>
                  <div><dt className="text-on-surface-variant">TBO</dt><dd className="text-primary">{e.tbo.toLocaleString("en-US")} h</dd></div>
                  <div><dt className="text-on-surface-variant">Intake</dt><dd className="text-primary">{e.turbo ? "Turbo" : "Natural"}</dd></div>
                </dl>
                <span className="mt-auto inline-flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-[0.1em] text-tertiary">
                  Fly it <ArrowRight size={13} strokeWidth={2} className="transition-transform duration-200 group-hover:translate-x-0.5" />
                </span>
              </Link>
            ))}
          </div>
        </section>

        {/* How a flight goes: verbs, not "step 1/2/3". */}
        <section className="home-scroll relative z-10 mx-auto w-full max-w-7xl px-4 pb-20 pt-8 md:px-6">
          <div className="grid grid-cols-1 gap-8 border-t border-outline-variant/40 pt-10 md:grid-cols-3">
            {FLOW.map((f) => (
              <div key={f.verb} className="flex flex-col gap-2">
                <h3 className="font-headline-md text-xl font-semibold text-primary">{f.verb}</h3>
                <p className="max-w-[40ch] text-[15px] leading-relaxed text-on-surface-variant">{f.body}</p>
                <Link href={f.href} className="mt-1 inline-flex w-fit items-center gap-1.5 text-[12px] font-semibold text-tertiary hover:underline">
                  {f.link} <ArrowRight size={13} strokeWidth={2} />
                </Link>
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer className="relative z-10 w-full border-t border-outline-variant/30 bg-surface-container-lowest">
        <div className="mx-auto flex w-full max-w-7xl flex-col items-center justify-between gap-4 px-4 py-8 md:flex-row md:px-6">
          <span className="text-[12px] text-on-surface-variant">&copy; 2026 AERO-SIM</span>
          <nav aria-label="Footer" className="flex flex-wrap justify-center gap-6">
            {[["/engine_info", "Engines"], ["/mission", "Missions"], ["/telemetry", "Telemetry"], ["/about_us", "About us"]].map(([href, label]) => (
              <Link key={href} href={href} className="text-[12px] text-on-surface-variant transition-colors hover:text-tertiary">
                {label}
              </Link>
            ))}
          </nav>
        </div>
      </footer>
    </>
  );
}
