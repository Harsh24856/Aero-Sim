"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Settings, CircleUserRound } from "lucide-react";

const NAV_LINKS = [
  { label: "Simulator", href: "/home" },
  { label: "Simulate", href: "/simulate" },
  { label: "Telemetry", href: "#" },
  { label: "Missions", href: "#" },
];

export default function Navbar() {
  const pathname = usePathname();

  // /engine is part of the simulator flow, so "Simulator" stays highlighted there too.
  // /simulate has its own dedicated "Simulate" tab (the live cockpit dashboard),
  // separate from the AERO-SIM landing/config pages.
  const isActive = (href: string) =>
    href === "/home" ? pathname === "/home" || pathname === "/engine" : pathname === href;

  return (
    <nav className="fixed top-0 w-full z-50 bg-background/80 backdrop-blur-xl border-b border-outline-variant/30 shadow-[2px_2px_0px_#000000]">
      <div className="flex justify-between items-center px-6 h-16 w-full max-w-full">
        <Link href="/home" className="font-headline-md text-2xl font-bold text-primary tracking-tighter">
          AERO-SIM
        </Link>

        <div className="hidden md:flex space-x-8">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.label}
              href={link.href}
              className={`font-label-caps text-[11px] tracking-[0.1em] font-bold uppercase pb-1 flex items-center transition-colors duration-200 ${
                isActive(link.href)
                  ? "text-tertiary border-b-2 border-tertiary"
                  : "text-on-surface-variant hover:text-primary"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>

        <div className="flex items-center space-x-4 text-primary">
          <button className="hover:text-tertiary transition-colors" aria-label="Settings">
            <Settings size={22} />
          </button>
          <button className="hover:text-tertiary transition-colors" aria-label="Account">
            <CircleUserRound size={22} />
          </button>
        </div>
      </div>
    </nav>
  );
}
