import Navbar from "@/components/Navbar";
import EngineViewer from "@/components/EngineViewer";
import { Zap, Wind, Fuel } from "lucide-react";
import TurbofanIllustration from "@/components/TurbofanIllustration";
import CombustionIllustration from "@/components/CombustionIllustration";

export default function EnginePage() {
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

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
            {/* ROTAX 914 - the real engine this whole project is built around, with the
                actual 3D model + real physics-model specs. */}
            <article className="bg-surface/80 backdrop-blur-xl border border-tertiary/30 hover:border-tertiary rounded-lg flex flex-col overflow-hidden transition-all duration-300 hover:-translate-y-1 hover:shadow-[2px_2px_0px_#FF9100]">
              <div className="absolute" />
              <div className="relative">
                <div className="absolute top-4 right-4 z-10 px-2 py-1 bg-black/90 text-tertiary border border-tertiary/50 text-[11px] tracking-[0.1em] font-bold uppercase rounded flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-tertiary animate-pulse" /> ONLINE
                </div>
                <div className="h-56 border-b border-tertiary/30">
                  <EngineViewer />
                </div>
              </div>
              <div className="p-6 flex-grow flex flex-col">
                <h2 className="font-headline-display text-xl text-tertiary mb-2 uppercase">
                  ROTAX 914 UL/F
                </h2>
                <p className="text-on-surface-variant text-sm mb-6 flex-grow">
                  Turbocharged 4-stroke piston engine - the real digital twin this
                  simulator is built around.
                </p>
                <div className="space-y-3 mb-6 text-[13px] font-mono">
                  <div className="flex justify-between border-b border-tertiary/20 pb-2">
                    <span className="text-on-surface-variant">MAX ENGINE RPM</span>
                    <span className="text-tertiary font-bold">5,800</span>
                  </div>
                  <div className="flex justify-between border-b border-tertiary/20 pb-2">
                    <span className="text-on-surface-variant">MAX POWER</span>
                    <span className="text-tertiary font-bold">85 kW</span>
                  </div>
                  <div className="flex justify-between pb-2">
                    <span className="text-on-surface-variant">CONFIGURATION</span>
                    <span className="text-tertiary font-bold">4-CYL TURBO</span>
                  </div>
                </div>
                <button className="w-full py-3 bg-tertiary text-black text-[11px] tracking-[0.1em] font-bold uppercase rounded flex items-center justify-center gap-2 mt-auto hover:brightness-110 transition-all">
                  <Zap size={16} /> SELECT ENGINE
                </button>
              </div>
            </article>

            {/* Conceptual alternatives - no real geometry exists for these yet, so
                they get an honest placeholder rather than a faked render. */}
            <article className="bg-surface/60 backdrop-blur-xl border border-outline-variant/40 rounded-lg flex flex-col overflow-hidden opacity-80">
              <div className="relative">
                <div className="absolute top-4 right-4 z-10 px-2 py-1 bg-black/90 text-on-surface-variant border border-on-surface-variant/40 text-[11px] tracking-[0.1em] font-bold uppercase rounded flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-on-surface-variant/60" /> STANDBY
                </div>
                <div className="h-56 border-b border-outline-variant/40 bg-surface-container-high/40 p-6">
                  <TurbofanIllustration />
                </div>
              </div>
              <div className="p-6 flex-grow flex flex-col">
                <h2 className="font-headline-display text-xl text-on-surface-variant mb-2 uppercase">
                  KINETIC TURBOFAN
                </h2>
                <p className="text-on-surface-variant/70 text-sm mb-6 flex-grow">
                  High-speed, high-altitude concept - planned for a future digital twin.
                </p>
                <div className="space-y-3 mb-6 text-[13px] font-mono">
                  <div className="flex justify-between border-b border-outline-variant/30 pb-2">
                    <span className="text-on-surface-variant/60">MAX SPEED</span>
                    <span className="text-on-surface-variant font-bold">850 KTS</span>
                  </div>
                  <div className="flex justify-between border-b border-outline-variant/30 pb-2">
                    <span className="text-on-surface-variant/60">CEILING</span>
                    <span className="text-on-surface-variant font-bold">45,000 FT</span>
                  </div>
                  <div className="flex justify-between pb-2">
                    <span className="text-on-surface-variant/60">STATUS</span>
                    <span className="text-on-surface-variant font-bold">CONCEPT</span>
                  </div>
                </div>
                <button disabled className="w-full py-3 border border-outline-variant/40 text-on-surface-variant/50 text-[11px] tracking-[0.1em] font-bold uppercase rounded flex items-center justify-center gap-2 mt-auto cursor-not-allowed">
                  <Wind size={16} /> NOT AVAILABLE
                </button>
              </div>
            </article>

            <article className="bg-surface/60 backdrop-blur-xl border border-outline-variant/40 rounded-lg flex flex-col overflow-hidden opacity-70">
              <div className="relative">
                <div className="absolute top-4 right-4 z-10 px-2 py-1 bg-black/90 text-on-surface-variant border border-on-surface-variant/40 text-[11px] tracking-[0.1em] font-bold uppercase rounded flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-on-surface-variant/40" /> OFFLINE
                </div>
                <div className="h-56 border-b border-outline-variant/40 bg-surface-container-high/40 p-6">
                  <CombustionIllustration />
                </div>
              </div>
              <div className="p-6 flex-grow flex flex-col">
                <h2 className="font-headline-display text-xl text-on-surface-variant/70 mb-2 uppercase">
                  HYBRID-X COMBUSTION
                </h2>
                <p className="text-on-surface-variant/60 text-sm mb-6 flex-grow">
                  Extended-range concept for rugged environments - not yet in development.
                </p>
                <div className="space-y-3 mb-6 text-[13px] font-mono">
                  <div className="flex justify-between border-b border-outline-variant/30 pb-2">
                    <span className="text-on-surface-variant/60">ENDURANCE</span>
                    <span className="text-on-surface-variant/70 font-bold">24h</span>
                  </div>
                  <div className="flex justify-between border-b border-outline-variant/30 pb-2">
                    <span className="text-on-surface-variant/60">FUEL</span>
                    <span className="text-on-surface-variant/70 font-bold">MULTI</span>
                  </div>
                  <div className="flex justify-between pb-2">
                    <span className="text-on-surface-variant/60">STATUS</span>
                    <span className="text-on-surface-variant/70 font-bold">CONCEPT</span>
                  </div>
                </div>
                <button disabled className="w-full py-3 border border-outline-variant/30 text-on-surface-variant/40 text-[11px] tracking-[0.1em] font-bold uppercase rounded flex items-center justify-center gap-2 mt-auto cursor-not-allowed">
                  <Fuel size={16} /> NOT AVAILABLE
                </button>
              </div>
            </article>
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
