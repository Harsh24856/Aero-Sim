"use client";

import Link from "next/link";
import { Fragment, useEffect, useRef, type CSSProperties } from "react";
import Navbar from "@/components/Navbar";
import type { World } from "./world";
import "./landing.css";

/* Word-level masks: each word rises out of its own clip at reading pace.
   The visible words are presentational; the heading keeps the real text. */
function Words({ text, from = 0 }: { text: string; from?: number }) {
  return (
    <>
      {text.split(" ").map((w, i) => (
        <Fragment key={i}>
          {i ? " " : null}
          <span className="lp-wmask" aria-hidden="true">
            <span className="lp-word" style={{ "--wd": `${(from + i) * 72}ms` } as CSSProperties}>{w}</span>
          </span>
        </Fragment>
      ))}
    </>
  );
}

const Arrow = () => (
  <svg viewBox="0 0 14 14" fill="none" width="13" height="13" aria-hidden="true">
    <path d="M3 11 11 3M5 3h6v6" stroke="currentColor" strokeWidth="1.3" />
  </svg>
);

const CHAPTERS = [
  { n: "01", id: "twin", t: "Physics Twin", p: "Eight coupled models, stepped once a second." },
  { n: "02", id: "fleet", t: "Engine Fleet", p: "912, 914, 915 and 916 flown side by side." },
  { n: "03", id: "watch", t: "AI Watch", p: "Five heads that hear a fault coming." },
  { n: "04", id: "launch", t: "Launch", p: "Take the stick. Break something safely." },
];

const FLEET = [
  { model: "/models/uav-engine-914.glb", name: "914 F/UL", tag: "Turbo · reference twin", meta: ["115 HP · 84.8 kW", "13 fault types"], lead: true },
  { model: "/models/rotax-style-912-boxer-uav-engine.glb", name: "912 ULS", tag: "Naturally aspirated", meta: ["100 HP · 73.5 kW", "11 fault types"] },
  { model: "/models/rotax-style-915-turbo-uav-engine.glb", name: "915 iS", tag: "Intercooled · EFI", meta: ["141 HP · 104 kW", "14 fault types"] },
  { model: "/models/rotax-916-engine.glb", name: "916 iS", tag: "Flagship · 1.86 HP/kg", meta: ["160 HP · 117 kW", "14 fault types"] },
];

const HEADS = [
  { k: "01", h: "Detection", code: "DET", p: "Is anything wrong? A watch over every one-second window, tuned for engines drifting from their twin.", out: "p(fault)" },
  { k: "02", h: "Diagnosis", code: "DIA", p: "Which channel is lying — per-sensor attribution, so a stuck probe is not mistaken for a sick engine.", out: "channel" },
  { k: "03", h: "Severity", code: "SEV", p: "How bad, graded continuously rather than as a traffic light.", out: "0 → 1" },
  { k: "04", h: "Failure modes", code: "FMD", p: "Misfire, injector fouling, cooling loss, combustion — named across every fault type.", out: "13 classes" },
  { k: "05", h: "Remaining life", code: "RUL", p: "Engine hours left before overhaul, re-scored on predicted wear instead of simulator labels.", out: "hours" },
];

const RAIL = ["Pre-flight", "Physics Twin", "Engine Fleet", "AI Watch", "Launch", "Colophon"];

export default function HomeLanding() {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = rootRef.current!;
    const $ = <T extends Element = HTMLElement>(s: string) => root.querySelector<T>(s)!;
    const $$ = <T extends Element = HTMLElement>(s: string) => [...root.querySelectorAll<T>(s)];
    const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const COARSE = matchMedia("(hover: none)").matches;
    const vpH = () => document.documentElement.clientHeight || innerHeight;
    const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
    const smooth = (e0: number, e1: number, x: number) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
    const cleanups: (() => void)[] = [];
    const listen = (t: EventTarget, ev: string, f: EventListener) => {
      t.addEventListener(ev, f, { passive: true } as AddEventListenerOptions);
      cleanups.push(() => t.removeEventListener(ev, f));
    };
    const timers: number[] = [];
    const later = (f: () => void, ms: number) => timers.push(window.setTimeout(f, ms));

    /* ---------------------------------------------- scroll ↔ chapters */
    const SECS = $$("[data-cam]");
    let anchors: number[] = [];
    const measure = () => {
      const max = Math.max(1, document.documentElement.scrollHeight - vpH());
      anchors = SECS.map((el, i) => {
        if (i === 0) return 0;
        if (i === SECS.length - 1) return max;
        const r = el.getBoundingClientRect();
        return clamp(r.top + scrollY + r.height * 0.5 - vpH() * 0.5, 0, max);
      });
      for (let i = 1; i < anchors.length; i++) anchors[i] = Math.max(anchors[i], anchors[i - 1] + 1);
    };
    const progressFor = (y: number) => {
      if (y <= anchors[0]) return 0;
      for (let i = 0; i < anchors.length - 1; i++)
        if (y <= anchors[i + 1]) return i + (y - anchors[i]) / (anchors[i + 1] - anchors[i]);
      return anchors.length - 1;
    };
    measure();
    listen(window, "resize", measure);
    document.fonts?.ready.then(measure);

    /* ---------------------------------------------- reveals */
    const items = $$("[data-rv]");
    const groups = new Map<Element | null, HTMLElement[]>();
    items.forEach((el) => { const k = el.parentElement; groups.set(k, [...(groups.get(k) || []), el]); });
    groups.forEach((arr) => arr.forEach((el, i) => (el.dataset.rvd = String(i * 85))));
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (!e.isIntersecting) return;
      io.unobserve(e.target);
      const el = e.target as HTMLElement;
      later(() => el.classList.add("rv-in"), REDUCE ? 0 : +(el.dataset.rvd || 0));
    }), { rootMargin: "0px 0px -10% 0px", threshold: 0.04 });
    items.forEach((el) => { if (!el.closest(".lp-hero")) io.observe(el); });
    cleanups.push(() => io.disconnect());

    /* ---------------------------------------------- nav, rail, hash links */
    const navwrap = $(".lp-navwrap");
    const rail = $(".lp-rail");
    const dots = $$<HTMLButtonElement>(".lp-rail button");
    dots.forEach((b, i) => listen(b, "click", () => scrollTo({ top: anchors[i], behavior: REDUCE ? "auto" : "smooth" })));
    let last = 0, active = -1;
    const onScroll = () => {
      const y = scrollY;
      navwrap.classList.toggle("stuck", y > 40);
      navwrap.classList.toggle("hide", y > last + 4 && y > vpH() * 0.8);
      if (y < last - 4) navwrap.classList.remove("hide");
      last = y;
      const a = Math.round(progressFor(y));
      if (a !== active) { active = a; dots.forEach((d, i) => d.classList.toggle("on", i === a)); rail.dataset.at = String(a); }
    };
    listen(window, "scroll", onScroll);
    onScroll();
    $$<HTMLAnchorElement>('a[href^="#"]').forEach((a) => listen(a, "click", ((e: Event) => {
      const t = document.querySelector<HTMLElement>(a.getAttribute("href")!);
      if (!t) return;
      e.preventDefault();
      scrollTo({ top: t.offsetTop - 40, behavior: REDUCE ? "auto" : "smooth" });
    }) as EventListener));

    /* ---------------------------------------------- the hero stands down
       Each piece along the foot gets its own window in the exit, so the
       foot empties one element at a time instead of sliding off as a block. */
    const seq = [
      { el: $(".lp-peek"), at: 0, span: 0.26, blur: 10, shift: false },
      { el: $(".lp-cue"), at: 0.1, span: 0.3, blur: 0, shift: true },
      ...$$(".lp-chip").map((el, i) => ({ el, at: 0.2 + i * 0.1, span: 0.3, blur: 0, shift: true })),
      { el: $(".lp-side"), at: 0.6, span: 0.3, blur: 0, shift: false },
    ];
    let exiting = false;
    const exit = () => {
      const t = clamp(scrollY / Math.max(1, vpH() * 0.58), 0, 1);
      if (t <= 0) {
        if (!exiting) return;
        seq.forEach(({ el }) => { el.style.opacity = ""; el.style.transform = ""; el.style.filter = ""; el.style.pointerEvents = ""; el.style.transition = ""; });
        exiting = false; return;
      }
      exiting = true;
      seq.forEach((o) => {
        o.el.style.transition = "none";
        const a = 1 - smooth(o.at, o.at + o.span, t);
        o.el.style.opacity = a.toFixed(3);
        if (o.shift) o.el.style.transform = `translate3d(0,${((1 - a) * 15).toFixed(1)}px,0)`;
        if (o.blur) o.el.style.filter = a > 0.999 ? "" : `blur(${((1 - a) * o.blur).toFixed(1)}px)`;
        o.el.style.pointerEvents = a < 0.05 ? "none" : "";
      });
    };
    listen(window, "scroll", exit);

    /* ---------------------------------------------- cursor */
    const dot = $(".lp-cursor");
    let raf = 0;
    if (!COARSE) {
      let x = innerWidth / 2, y = innerHeight / 2, tx = x, ty = y;
      listen(window, "pointermove", ((e: PointerEvent) => { tx = e.clientX; ty = e.clientY; }) as EventListener);
      $$("a, button, [data-card]").forEach((el) => {
        listen(el, "mouseenter", () => dot.classList.add("act"));
        listen(el, "mouseleave", () => dot.classList.remove("act"));
      });
      const tick = () => {
        x += (tx - x) * 0.18; y += (ty - y) * 0.18;
        dot.style.transform = `translate3d(${x.toFixed(1)}px,${y.toFixed(1)}px,0)`;
        raf = requestAnimationFrame(tick);
      };
      tick();
    }

    /* ---------------------------------------------- grain */
    {
      const S = 180, c = document.createElement("canvas");
      c.width = c.height = S;
      const x = c.getContext("2d")!, im = x.createImageData(S, S);
      for (let i = 0; i < S * S; i++) { const v = 110 + Math.random() * 90; im.data.set([v, v, v, 255], i * 4); }
      x.putImageData(im, 0, 0);
      $(".lp-grain").style.backgroundImage = `url(${c.toDataURL("image/png")})`;
    }

    /* ---------------------------------------------- the world */
    const pre = $(".lp-pre"), fill = $(".lp-pre-fill"), pct = $(".lp-pre-pct"), label = $(".lp-pre-label");
    let world: World | null = null, cancelled = false;
    const revealHero = () => {
      pre.classList.add("done");
      document.documentElement.classList.remove("lp-locked");
      $$(".lp-hero [data-rv]").forEach((el, i) => later(() => el.classList.add("rv-in"), REDUCE ? 0 : 340 + i * 95));
    };
    const fallback = (err?: unknown) => {
      if (err) console.error("[aero-sim] WebGL unavailable, showing the still page", err);
      root.classList.add("no-webgl");
      $$("[data-rv]").forEach((el) => el.classList.add("rv-in"));
      pre.classList.add("done");
      document.documentElement.classList.remove("lp-locked");
    };
    document.documentElement.classList.add("lp-locked");
    listen(window, "resize", () => layoutFix());
    const layoutFix = () => document.documentElement.style.setProperty("--vw", document.documentElement.clientWidth + "px");
    layoutFix();

    import("./world")
      .then(({ createWorld }) => createWorld({
        canvas: $<HTMLCanvasElement>(".lp-gl"),
        root,
        getProgress: () => progressFor(scrollY),
        onStep: (f, l) => { fill.style.right = `${((1 - f) * 100).toFixed(1)}%`; pct.textContent = String(Math.round(f * 100)); label.textContent = l; },
        onLost: () => fallback(new Error("context lost")),
        reduce: REDUCE,
        coarse: COARSE,
      }))
      .then((w) => {
        if (cancelled) { w.dispose(); return; }
        world = w;
        later(() => { revealHero(); w.start(); measure(); }, 240);
      })
      .catch(fallback);

    $$("[data-chip], .lp-plate").forEach((el) => {
      listen(el, "mouseenter", () => world?.setFocus(true));
      listen(el, "mouseleave", () => world?.setFocus(false));
    });

    return () => {
      cancelled = true;
      cleanups.forEach((f) => f());
      timers.forEach(clearTimeout);
      cancelAnimationFrame(raf);
      world?.dispose();
      document.documentElement.classList.remove("lp-locked");
    };
  }, []);

  return (
    <div className="lp" ref={rootRef}>
      <canvas className="lp-gl" aria-hidden="true" />
      <div className="lp-vignette" />
      <div className="lp-grain" />
      <div className="lp-cursor" />

      {/* ============================================ preloader */}
      <div className="lp-pre">
        <div className="lp-pre-in">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="lp-pre-mark" src="/aero-sim-icon.svg" alt="" width={44} height={44} />
          <div className="lp-pre-k">AERO·SIM — DIGITAL TWIN</div>
          <div className="lp-pre-bar"><i className="lp-pre-fill" /></div>
          <div className="lp-pre-meta"><span className="lp-pre-label">Spooling up</span><b><span className="lp-pre-pct">0</span>%</b></div>
        </div>
      </div>

      <div className="lp-navwrap"><Navbar /></div>

      <div className="lp-page" id="top">
        {/* ============================================ hero */}
        <section className="lp-hero" data-cam="0">
          <div className="lp-hero-top">
            <div className="lp-eyebrow" data-rv="fade"><span className="lp-dot" /> Chapter 00 — Pre-flight</div>
            <h1 className="lp-display lp-h-hero">
              <span className="lp-sr">Every engine speaks before it fails.</span>
              <span className="lp-line lp-wr" data-rv="line"><Words text="Every engine" /></span>
              <span className="lp-line lp-wr" data-rv="line"><Words text="speaks before" from={2} /></span>
              <span className="lp-line lp-wr" data-rv="line"><Words text="it fails." from={4} /></span>
            </h1>
            <p className="lp-body lp-hero-sub" data-rv="up">
              A live physics twin of four Rotax-class UAV engines, read by neural heads that hear a fault coming before the gauges do.
            </p>
          </div>

          <div className="lp-hero-spacer" />

          <div className="lp-hero-foot">
            <div className="lp-cue" data-rv="fade"><span>Scroll to take off</span><span className="lp-track"><i /></span></div>
            <div className="lp-chapters">
              {CHAPTERS.map((c, i) => (
                <a key={c.n} href={`#${c.id}`} className="lp-chip" data-chip={i} data-rv="up">
                  <span className="lp-num">{c.n}</span>
                  <span className="lp-tx"><b>{c.t}</b><span>{c.p}</span></span>
                </a>
              ))}
            </div>
          </div>

          <a className="lp-peek" href="#fleet" data-rv="fade" aria-label="Chase cam: the twin in flight">
            <span className="lp-peek-fr" data-view="chase">
              <span className="lp-rec"><i /> Live</span>
            </span>
            <span className="lp-peek-cap"><b>Chase cam</b><i>914 F/UL · circuit 01</i></span>
          </a>

          <div className="lp-word-fb" aria-hidden="true">AERO<em>-</em>SIM</div>

          <div className="lp-side" data-rv="up"><span>UAV engine digital twin</span></div>
        </section>

        {/* ============================================ 01 · the twin */}
        <section className="lp-sec" id="twin" data-cam="1">
          <div className="lp-sec-head" data-rv="fade">
            <span className="lp-k"><b>01</b> — The Physics Twin</span><span className="lp-rule" /><span className="lp-k">TWIN·914</span>
          </div>
          <div className="lp-gate">
            <h2 className="lp-display lp-h-sec">
              <span className="lp-sr">Physics you can fly, faults you can feel.</span>
              <span className="lp-wr lp-wr-block" data-rv="line"><Words text="Physics you can fly, faults you can feel." /></span>
            </h2>
            <div className="lp-gate-copy">
              <p className="lp-lead" data-rv="up">
                The twin runs the engine the way the engine runs itself: combustion, propeller load, airflow, heat soak, oil film,
                vibration and wear, stepped once per simulated second. Inject a misfire at altitude and watch it ripple through every channel.
              </p>
              <p className="lp-body" data-rv="up">
                Measured data arrives the way a real aircraft sends it — over the CAN bus — so the twin can shadow a live engine and
                flag the gap between what it reads and what physics expects. Every step is checked against a Simulink model of the same engine.
              </p>
              <Link className="lp-arrowlink" href="/engine" data-rv="fade">
                <span>Enter the simulator</span><span className="lp-ar"><Arrow /></span>
              </Link>
            </div>
          </div>
          <div className="lp-stats" data-rv="up">
            <div><b>04</b><span>Engines modelled</span></div>
            <div><b>08</b><span>Coupled physics models</span></div>
            <div><b>13</b><span>Fault types · 914</span></div>
            <div><b>2·10<sup>−15</sup></b><span>Worst error vs Simulink</span></div>
          </div>
        </section>

        {/* ============================================ 02 · the fleet */}
        <section className="lp-sec" id="fleet" data-cam="2">
          <div className="lp-sec-head" data-rv="fade">
            <span className="lp-k"><b>02</b> — The Engine Fleet</span><span className="lp-rule" /><span className="lp-k">ROTAX 91X</span>
          </div>
          <div className="lp-cards">
            {FLEET.map((e) => (
              <Link key={e.name} href="/engine_info" className={`lp-card${e.lead ? " lead" : ""}`} data-card data-rv="up">
                <span className="lp-card-fr" data-view="engine" data-model={e.model}>
                  <span className="lp-card-load">Spinning up {e.name}</span>
                  <span className="lp-card-ar"><Arrow /></span>
                  <span className="lp-card-lab"><b>{e.name}</b><span>{e.tag}</span></span>
                </span>
                <span className="lp-card-meta"><span>{e.meta[0]}</span><span>{e.meta[1]}</span></span>
              </Link>
            ))}
          </div>
        </section>

        {/* ============================================ 03 · the watch */}
        <section className="lp-sec" id="watch" data-cam="3">
          <div className="lp-sec-head" data-rv="fade">
            <span className="lp-k"><b>03</b> — The AI Watch</span><span className="lp-rule" /><span className="lp-k">5 HEADS · 29 INPUTS</span>
          </div>
          <div className="lp-cur-head">
            <h2 className="lp-display lp-h-sec">
              <span className="lp-sr">Five heads. One verdict. Before it breaks.</span>
              <span className="lp-wr lp-wr-block" data-rv="line"><Words text="Five heads. One verdict. Before it breaks." /></span>
            </h2>
            <p className="lp-body-lg" data-rv="up">
              Every one-second window passes through five neural heads trained on the twin. Physics residuals keep them honest, and a
              deterministic advisory turns their output into a maintenance call a pilot can act on.
            </p>
          </div>
          <div className="lp-atlas">
            {HEADS.map((h) => (
              <div key={h.k} className="lp-plate" data-rv="up">
                <span className="lp-k">{h.k}</span>
                <h3>{h.h}<em>{h.code}</em></h3>
                <p>{h.p}</p>
                <span className="lp-t">out → {h.out}</span>
                <i className="lp-bar" />
              </div>
            ))}
          </div>
        </section>

        {/* ============================================ 04 · launch */}
        <section className="lp-sec lp-fin" id="launch" data-cam="4">
          <div className="lp-eyebrow" data-rv="fade">Chapter 04 — Launch</div>
          <h2 className="lp-display">
            <span className="lp-sr">Take off</span>
            <span className="lp-wr lp-wr-block" data-rv="line"><Words text="Take off" /></span>
          </h2>
          <p className="lp-body-lg" data-rv="up">
            The runway is lit. Pick an engine, set a mission, and push it until the twin tells you it is about to give.
          </p>
          <div className="lp-ctas" data-rv="fade">
            <Link className="lp-cta" href="/engine"><i /><span>Launch simulation</span><Arrow /></Link>
            <Link className="lp-cta ghost" href="/telemetry"><i /><span>Browse telemetry</span><Arrow /></Link>
          </div>
        </section>

        {/* ============================================ footer */}
        <footer className="lp-foot" data-cam="5">
          <div className="lp-foot-grid">
            <div className="lp-foot-brand">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/aero-sim-icon.svg" alt="" width={54} height={54} />
              <p>AERO-SIM is a UAV engine digital twin: physics, telemetry and neural diagnostics in one cockpit, built to fail safely on the ground.</p>
            </div>
            <div><h4>Chapters</h4><ul>
              <li><a href="#twin">Physics Twin</a></li><li><a href="#fleet">Engine Fleet</a></li>
              <li><a href="#watch">AI Watch</a></li><li><a href="#launch">Launch</a></li>
            </ul></div>
            <div><h4>Platform</h4><ul>
              <li><Link href="/engine">Simulator</Link></li><li><Link href="/engine_info">Engines</Link></li>
              <li><Link href="/mission">Missions</Link></li><li><Link href="/telemetry">Telemetry</Link></li>
            </ul></div>
            <div><h4>Crew</h4><ul>
              <li><Link href="/about_us">About us</Link></li><li><Link href="/dashboard">Dashboard</Link></li>
              <li><Link href="/login">Sign in</Link></li>
            </ul></div>
          </div>
          <div className="lp-foot-base">
            <span>© 2026 AERO-SIM Precision Systems</span>
            <span className="lp-status"><i /> System status: optimal</span>
            <span>three.js · Next.js · FastAPI</span>
          </div>
        </footer>
      </div>

      <nav className="lp-rail" aria-label="Chapters">
        {RAIL.map((n) => <button key={n} type="button" title={n} aria-label={n}><i /></button>)}
      </nav>
    </div>
  );
}
