"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Settings, CircleUserRound, LogOut } from "lucide-react";
import { supabase } from "@/lib/supabase";

const NAV_LINKS = [
  { label: "Simulator", href: "/home" },
  { label: "Engines", href: "/engines" },
  { label: "Telemetry", href: "#" },
  { label: "Missions", href: "#" },
  { label: "About Us", href: "#" },
];

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setUserEmail(data.session?.user?.email ?? null);
    });
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUserEmail(session?.user?.email ?? null);
    });
    return () => listener.subscription.unsubscribe();
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    setMenuOpen(false);
    router.push("/home");
  };

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

          <div className="relative">
            <button
              className="hover:text-tertiary transition-colors"
              aria-label={userEmail ? "Account menu" : "Sign in"}
              onClick={() => (userEmail ? setMenuOpen((v) => !v) : router.push("/login"))}
            >
              <CircleUserRound size={22} className={userEmail ? "text-tertiary" : undefined} />
            </button>

            {menuOpen && userEmail && (
              <div className="absolute right-0 top-full mt-2 w-56 bg-surface border border-outline-variant/40 rounded-lg shadow-lg py-2 z-50">
                <div className="px-3 py-2 text-[11px] uppercase tracking-[0.08em] text-on-surface-variant border-b border-outline-variant/30 truncate">
                  {userEmail}
                </div>
                <button
                  onClick={handleSignOut}
                  className="w-full flex items-center gap-2 px-3 py-2 text-[13px] text-primary hover:bg-surface-container-highest/60 transition-colors"
                >
                  <LogOut size={15} /> Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
