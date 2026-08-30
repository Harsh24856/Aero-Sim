import Link from "next/link";
import Navbar from "@/components/Navbar";
import BlackoutPlane from "@/components/BlackoutPlane";

export default function HomePage() {
  return (
    <>
      <Navbar />

      <main className="flex-grow pt-16 relative flex flex-col justify-center items-center">
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />

        <section className="relative z-10 w-full max-w-7xl mx-auto px-4 md:px-6 py-8 md:py-10 flex flex-col items-center justify-center min-h-fit">
          {/* Text now leads - headline, copy, and CTAs sit above the 3D model */}
          <div className="text-center max-w-3xl mx-auto flex flex-col items-center space-y-4 mb-6 md:mb-8">
            <span className="text-[11px] tracking-[0.2em] font-bold uppercase text-tertiary font-mono">
              UAV ENGINE DIGITAL TWIN
            </span>
            <h1 className="font-headline-display text-[32px] md:text-[44px] leading-[1.15] font-bold text-primary uppercase tracking-tight">
              PRECISION UAV SIMULATION
            </h1>
            <p className="text-base leading-relaxed text-on-surface-variant max-w-2xl">
              Advanced telemetry and flight control systems for the next generation of
              aerospace engineering.
            </p>
          </div>

          {/* 3D model below the text, fixed framing (see BlackoutPlane component) */}
          <div className="w-full max-w-4xl mx-auto">
            <BlackoutPlane />
          </div>

          {/* CTAs below the model, not grouped with the text block */}
          <div className="flex flex-col sm:flex-row gap-4 mt-6 w-full justify-center">
            <Link
              href="/engine"
              className="bg-tertiary text-black text-[11px] tracking-[0.1em] font-bold uppercase px-8 py-4 rounded shadow-[2px_2px_0px_#000000] hover:-translate-y-0.5 hover:shadow-[4px_4px_0px_#000000] active:translate-y-0 active:shadow-[1px_1px_0px_#000000] transition-all duration-200 text-center"
            >
              LAUNCH SIMULATION
            </Link>
            <button className="bg-surface-container-highest/60 backdrop-blur-md border border-outline-variant/50 text-primary text-[11px] tracking-[0.1em] font-bold uppercase px-8 py-4 rounded hover:bg-surface-container-high transition-colors duration-200 hover:border-tertiary hover:text-tertiary">
              ACCESS TELEMETRY
            </button>
          </div>

          <div className="absolute bottom-8 left-8 hidden lg:flex items-center space-x-2 bg-surface/80 backdrop-blur-md border border-tertiary/30 px-4 py-2 rounded shadow-[1px_1px_0px_#000000]">
            <div className="w-2 h-2 rounded-full bg-tertiary animate-pulse" />
            <span className="text-[13px] tracking-wide text-tertiary font-mono">
              SYS.LINK_ESTABLISHED
            </span>
          </div>
          <div className="absolute bottom-8 right-8 hidden lg:flex flex-col items-end space-y-1">
            <span className="text-[13px] tracking-wide text-on-surface-variant font-mono">
              ALTITUDE_SIM: <span className="text-tertiary">32,000 FT</span>
            </span>
            <span className="text-[13px] tracking-wide text-on-surface-variant font-mono">
              THRUST_V: <span className="text-tertiary">NOMINAL</span>
            </span>
          </div>
        </section>
      </main>

      <footer className="bg-surface-container-lowest w-full mt-12 border-t border-outline-variant/30 z-10 relative">
        <div className="flex flex-col md:flex-row justify-between items-center px-6 py-8 gap-4 w-full">
          <div className="text-[11px] tracking-[0.1em] font-bold uppercase text-tertiary text-center md:text-left flex flex-col md:flex-row gap-2 md:gap-4 items-center">
            <span>&copy; 2026 AERO-SIM PRECISION SYSTEMS.</span>
            <span className="hidden md:inline text-outline-variant">|</span>
            <span className="text-tertiary flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-tertiary animate-pulse" />
              SYSTEM STATUS: OPTIMAL
            </span>
          </div>
          <div className="flex flex-wrap justify-center gap-4">
            <a className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant hover:text-tertiary transition-colors" href="#">Privacy Policy</a>
            <a className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant hover:text-tertiary transition-colors" href="#">Terms of Service</a>
            <a className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant hover:text-tertiary transition-colors" href="#">Security Protocol</a>
            <a className="text-[11px] tracking-[0.1em] font-bold uppercase text-on-surface-variant hover:text-tertiary transition-colors" href="#">API Documentation</a>
          </div>
        </div>
      </footer>
    </>
  );
}
