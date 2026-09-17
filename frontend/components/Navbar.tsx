"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { CircleUserRound } from "lucide-react";
import { supabase } from "@/lib/supabase";

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const [isSignedIn, setIsSignedIn] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setIsSignedIn(Boolean(data.session));
    });

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsSignedIn(Boolean(session));
    });

    return () => listener.subscription.unsubscribe();
  }, []);

  const isSimulatorActive =
    pathname === "/home" ||
    pathname === "/" ||
    pathname === "/simulate" ||
    pathname === "/engine";

  const isEnginesActive = pathname === "/engine_info";

  return (
    <nav className="fixed top-0 w-full z-50 bg-background/80 backdrop-blur-xl border-b border-outline-variant/30 shadow-[2px_2px_0px_#000000]">
      <div className="flex justify-between items-center px-6 h-16 w-full max-w-full">
        {/* aero-sim-logo.svg is the full lockup and stays the asset of record; the bar
            variant is the same artwork with the strapline dropped, because at the 40px
            the navigation allows that line renders 2.2px tall - a smudge, not type.
            The wordmark lives in the asset, so the accessible name comes from alt.
            unoptimized: next/image refuses to run SVGs through the optimiser unless
            dangerouslyAllowSVG is set, and a static local mark needs neither. */}
        <Link href="/home" aria-label="AERO-SIM home" className="flex items-center shrink-0">
          <Image
            src="/aero-sim-logo-bar.svg"
            alt="AERO-SIM"
            width={1060}
            height={328}
            priority
            unoptimized
            className="h-9 w-auto md:h-10"
          />
        </Link>

        <div className="hidden md:flex space-x-8 items-center">
          <Link
            href="/home"
            className={`font-label-caps text-[11px] tracking-[0.1em] font-bold uppercase pb-1 flex items-center transition-colors duration-200 ${
              isSimulatorActive
                ? "text-tertiary border-b-2 border-tertiary"
                : "text-on-surface-variant hover:text-primary"
            }`}
          >
            Simulator
          </Link>

          <Link
            href="/engine_info"
            className={`font-label-caps text-[11px] tracking-[0.1em] font-bold uppercase pb-1 flex items-center transition-colors duration-200 ${
              isEnginesActive
                ? "text-tertiary border-b-2 border-tertiary"
                : "text-on-surface-variant hover:text-primary"
            }`}
          >
            Engines
          </Link>

          <Link
            href="/mission"
            className={`font-label-caps text-[11px] tracking-[0.1em] font-bold uppercase pb-1 flex items-center transition-colors duration-200 ${
              pathname?.startsWith("/mission")
                ? "text-tertiary border-b-2 border-tertiary"
                : "text-on-surface-variant hover:text-primary"
            }`}
          >
            Mission
          </Link>

          <Link
            href="/telemetry"
            className={`font-label-caps text-[11px] tracking-[0.1em] font-bold uppercase pb-1 flex items-center transition-colors duration-200 ${
              pathname?.startsWith("/telemetry")
                ? "text-tertiary border-b-2 border-tertiary"
                : "text-on-surface-variant hover:text-primary"
            }`}
          >
            Telemetry
          </Link>

          <Link
            href="/about_us"
            className={`font-label-caps text-[11px] tracking-[0.1em] font-bold uppercase pb-1 flex items-center transition-colors duration-200 ${
              pathname === "/about_us"
                ? "text-tertiary border-b-2 border-tertiary"
                : "text-on-surface-variant hover:text-primary"
            }`}
          >
            About Us
          </Link>
        </div>

        <div className="flex items-center text-primary">
          <button
            className="hover:text-tertiary transition-colors"
            aria-label={isSignedIn ? "Open dashboard" : "Sign in"}
            onClick={() => router.push(isSignedIn ? "/dashboard" : "/login")}
          >
            <CircleUserRound size={22} className={isSignedIn ? "text-tertiary" : undefined} />
          </button>
        </div>
      </div>
    </nav>
  );
}
