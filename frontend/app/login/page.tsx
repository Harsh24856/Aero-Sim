"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import { supabase } from "@/lib/supabase";
import { ShieldCheck, Zap, Eye, EyeOff } from "lucide-react";

function getAuthErrorMessage(message: string, mode: "login" | "signup") {
  if (mode === "signup" && message.toLowerCase().includes("rate limit")) {
    return "Email delivery is temporarily rate-limited. Please wait before trying again, or ask the project owner to configure a custom SMTP provider in Supabase.";
  }
  return message;
}

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [githubLoading, setGithubLoading] = useState(false);

  const handleGoogleSignIn = async () => {
    setError(null);
    setGoogleLoading(true);
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}/home` },
    });
    if (error) {
      setError(error.message);
      setGoogleLoading(false);
    }
  };

  const handleGithubSignIn = async () => {
    setError(null);
    setGithubLoading(true);
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "github",
      options: { redirectTo: `${window.location.origin}/home` },
    });
    if (error) {
      setError(error.message);
      setGithubLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    if (mode === "signup") {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: { data: { name } },
      });
      if (error) {
        setError(getAuthErrorMessage(error.message, mode));
      } else {
        router.push("/home");
      }
    } else {
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) {
        setError(getAuthErrorMessage(error.message, mode));
      } else {
        router.push("/home");
      }
    }
    setLoading(false);
  };

  return (
    <>
      <Navbar />
      <main className="flex-grow pt-16 relative flex items-center justify-center min-h-screen">
        {/* Grid + ambient glow */}
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />
        <div className="absolute inset-0 pointer-events-none z-0">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-64 bg-tertiary/10 rounded-full blur-[80px]" />
        </div>

        <div className="relative z-10 w-full max-w-5xl mx-auto px-4 py-16 grid gap-10 lg:grid-cols-[1.05fr_minmax(0,380px)] lg:items-center">

          {/* Identity panel. Hidden below lg: on a phone the card should be the
              whole screen, not something to scroll past. */}
          <section className="hidden lg:block">
            <span className="font-mono text-[10px] tracking-[0.24em] text-tertiary font-bold uppercase">
              DRDO PS 26054 // SIH 26054
            </span>
            <h2 className="font-headline-display text-[42px] leading-[1.05] font-bold text-primary uppercase tracking-tight mt-3">
              AI digital twin for<br />aero piston engines
            </h2>
            <p className="text-on-surface-variant text-[13px] leading-relaxed mt-4 max-w-md">
              A physics twin of the Rotax powerplants in MALE-class UAVs, with five AI heads
              reading its telemetry in real time: fault detection, per-channel diagnosis,
              severity, engine failure modes and remaining useful life.
            </p>
            <dl className="mt-8 grid grid-cols-3 gap-px bg-outline-variant/20 border border-outline-variant/20 rounded overflow-hidden max-w-md">
              {[
                { k: "Engines", v: "912 · 914", s: "915 · 916 iS" },
                { k: "Channels", v: "25", s: "model inputs" },
                { k: "RUL error", v: "0.6%", s: "of TBO" },
              ].map((it) => (
                <div key={it.k} className="bg-surface/70 px-3 py-3">
                  <dt className="font-mono text-[9px] uppercase tracking-[0.14em] text-on-surface-variant/70">{it.k}</dt>
                  <dd className="font-headline-display text-[17px] text-primary mt-1 leading-none">{it.v}</dd>
                  <dd className="font-mono text-[9px] text-on-surface-variant/50 mt-1">{it.s}</dd>
                </div>
              ))}
            </dl>
          </section>

          {/* Auth column */}
          <div className="w-full max-w-sm mx-auto lg:mx-0">

          {/* Brand mark above card */}
          <div className="text-center mb-6">
            <span className="font-mono text-[10px] tracking-[0.24em] text-tertiary font-bold uppercase lg:hidden">
              AERO-SIM // PILOT ACCESS SYSTEM
            </span>
            <h1 className="font-headline-display text-3xl font-bold text-primary uppercase tracking-tight mt-1">
              {mode === "login" ? "Sign In" : "Create Account"}
            </h1>
            <p className="text-on-surface-variant text-[13px] mt-1">
              {mode === "login"
                ? "Access your simulation history and telemetry records."
                : "Track your simulation runs and flight data."}
            </p>
          </div>

          {/* Card */}
          <div className="bg-surface/80 backdrop-blur-xl border border-outline-variant/30 rounded-lg p-7 shadow-[2px_2px_0px_#000000]">

            {/* OAuth buttons */}
            <div className="space-y-3 mb-6">
              <button
                type="button"
                onClick={handleGoogleSignIn}
                disabled={googleLoading}
                className="w-full flex items-center justify-center gap-3 py-2.5 bg-white text-[#1f1f1f] text-[13px] font-semibold rounded hover:bg-gray-100 transition-all disabled:opacity-50 shadow-[1px_1px_0px_#000000]"
              >
                <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" className="shrink-0">
                  <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.9c1.7-1.57 2.7-3.87 2.7-6.62z"/>
                  <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.9-2.26c-.8.54-1.84.86-3.06.86-2.35 0-4.34-1.59-5.05-3.72H.96v2.33A9 9 0 0 0 9 18z"/>
                  <path fill="#FBBC05" d="M3.95 10.7A5.4 5.4 0 0 1 3.67 9c0-.59.1-1.17.28-1.7V4.97H.96A9 9 0 0 0 0 9c0 1.45.35 2.83.96 4.03z"/>
                  <path fill="#EA4335" d="M9 3.58c1.32 0 2.51.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0A9 9 0 0 0 .96 4.97L3.95 7.3C4.66 5.17 6.65 3.58 9 3.58z"/>
                </svg>
                {googleLoading ? "Redirecting..." : "Continue with Google"}
              </button>

              <button
                type="button"
                onClick={handleGithubSignIn}
                disabled={githubLoading}
                className="w-full flex items-center justify-center gap-3 py-2.5 bg-[#171513] border border-outline-variant/50 text-[13px] font-semibold text-white rounded hover:border-tertiary hover:bg-[#211b17] transition-all disabled:opacity-50 shadow-[1px_1px_0px_#000000]"
              >
                {/* GitHub icon */}
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className="shrink-0 text-white/80">
                  <path d="M12 0C5.37 0 0 5.373 0 12c0 5.303 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222 0 1.606-.015 2.896-.015 3.286 0 .315.216.69.825.572C20.565 21.795 24 17.298 24 12c0-6.627-5.373-12-12-12z"/>
                </svg>
                {githubLoading ? "Redirecting..." : "Continue with GitHub"}
              </button>
            </div>

            {/* Divider */}
            <div className="flex items-center gap-3 mb-6">
              <div className="flex-1 h-px bg-outline-variant/30" />
              <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-on-surface-variant/60">or</span>
              <div className="flex-1 h-px bg-outline-variant/30" />
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              {mode === "signup" && (
                <div>
                  <label className="block font-mono text-[10px] uppercase tracking-[0.14em] text-on-surface-variant mb-1.5">
                    Pilot Name
                  </label>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. John Doe"
                    className="w-full bg-surface-container-highest/60 border border-outline-variant/50 rounded px-3 py-2.5 text-primary text-sm focus:outline-none focus:border-tertiary focus:ring-1 focus:ring-tertiary/30 placeholder:text-on-surface-variant/30 transition-colors"
                  />
                </div>
              )}

              <div>
                <label className="block font-mono text-[10px] uppercase tracking-[0.14em] text-on-surface-variant mb-1.5">
                  Email Address
                </label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="pilot@aero-sim.io"
                  className="w-full bg-surface-container-highest/60 border border-outline-variant/50 rounded px-3 py-2.5 text-primary text-sm focus:outline-none focus:border-tertiary focus:ring-1 focus:ring-tertiary/30 placeholder:text-on-surface-variant/30 transition-colors"
                />
              </div>

              <div>
                <label className="block font-mono text-[10px] uppercase tracking-[0.14em] text-on-surface-variant mb-1.5">
                  Password
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    required
                    minLength={6}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder={mode === "signup" ? "At least 6 characters" : "••••••••"}
                    className="w-full bg-surface-container-highest/60 border border-outline-variant/50 rounded px-3 py-2.5 pr-10 text-primary text-sm focus:outline-none focus:border-tertiary focus:ring-1 focus:ring-tertiary/30 placeholder:text-on-surface-variant/30 transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((v) => !v)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-on-surface-variant/50 hover:text-on-surface-variant transition-colors"
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
              </div>

              {error && (
                <div className="text-[12px] text-red-400 bg-red-500/10 border border-red-500/30 rounded px-3 py-2 leading-relaxed">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 py-3 bg-tertiary text-black text-[11px] tracking-[0.12em] font-bold uppercase rounded shadow-[2px_2px_0px_#000000] hover:-translate-y-0.5 hover:shadow-[4px_4px_0px_#000000] active:translate-y-0 active:shadow-[1px_1px_0px_#000000] transition-all disabled:opacity-50 disabled:hover:translate-y-0 disabled:hover:shadow-[2px_2px_0px_#000000]"
              >
                <Zap size={14} className="fill-black" />
                {loading ? "Please wait..." : mode === "login" ? "Sign In" : "Create Account"}
              </button>
            </form>

            {/* Switch mode */}
            <p className="text-center font-mono text-[11px] text-on-surface-variant mt-5">
              {mode === "login" ? "No account yet? " : "Already have an account? "}
              <button
                onClick={() => { setMode(mode === "login" ? "signup" : "login"); setError(null); }}
                className="text-tertiary hover:underline font-bold"
              >
                {mode === "login" ? "Create one" : "Sign in"}
              </button>
            </p>
          </div>

          {/* Skip auth */}
          <p className="text-center mt-5">
            <Link href="/home" className="font-mono text-[11px] text-on-surface-variant/60 hover:text-tertiary transition-colors uppercase tracking-wider">
              Continue without an account →
            </Link>
          </p>

          {/* Security badge */}
          <div className="mt-6 flex items-center justify-center gap-2 text-on-surface-variant/40 font-mono text-[10px] uppercase tracking-wider">
            <ShieldCheck size={12} />
            <span>Secured by Supabase Auth · TLS 1.3</span>
          </div>
          </div>
        </div>
      </main>
    </>
  );
}
