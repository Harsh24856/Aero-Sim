"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import { supabase } from "@/lib/supabase";
import { Eye, EyeOff, Lock, Mail, User, Zap } from "lucide-react";

function getAuthErrorMessage(message: string, mode: "login" | "signup") {
  if (mode === "signup" && message.toLowerCase().includes("rate limit")) {
    return "Email delivery is temporarily rate-limited. Please wait before trying again, or ask the project owner to configure a custom SMTP provider in Supabase.";
  }
  return message;
}

/** The switch mark: a cut-corner tile in two halves that fly in, lock together and flare. */
function SwitchMark() {
  return (
    <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center" aria-hidden>
      <div className="login-flare absolute h-40 w-40 rounded-full" />
      <svg width="64" height="64" viewBox="0 0 64 64" className="relative overflow-visible">
        <defs>
          <linearGradient id="markA" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#ffd3a7" /><stop offset="1" stopColor="#FF9100" />
          </linearGradient>
          <linearGradient id="markB" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#FF9100" /><stop offset="1" stopColor="#ed3919" />
          </linearGradient>
        </defs>
        {/* Split along the diagonal; each half keeps two cut corners. */}
        <polygon className="login-half-a" points="14,4 50,4 4,50 4,14" fill="url(#markA)" fillOpacity="0.9" stroke="#ffd3a7" strokeOpacity="0.5" />
        <polygon className="login-half-b" points="60,14 60,50 50,60 14,60" fill="url(#markB)" fillOpacity="0.9" stroke="#ffb347" strokeOpacity="0.5" />
      </svg>
    </div>
  );
}

const field = "login-field w-full rounded border border-outline-variant/50 bg-black/45 py-2.5 pl-10 pr-3 text-sm text-primary placeholder:text-on-surface-variant/30 outline-none transition-all duration-200 focus:border-tertiary";
const label = "mb-1.5 block font-mono text-[10px] uppercase tracking-[0.14em] text-on-surface-variant";
const icon = "pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-on-surface-variant/50 transition-colors peer-focus:text-tertiary";

// Switch choreography (ms), from the reference video: form out, mark in and flare, form in.
const OUT_MS = 220;
const MARK_MS = 700;

type Phase = "mark" | "form" | "out";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "signup">("login");
  // The page opens on the mark, as the reference does, then reveals the form.
  const [phase, setPhase] = useState<Phase>("mark");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [oauthLoading, setOauthLoading] = useState<"google" | "github" | null>(null);
  const timers = useRef<number[]>([]);

  // The card eases between the two forms' heights instead of jumping.
  const inner = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | null>(null);
  useLayoutEffect(() => {
    const el = inner.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setHeight(el.offsetHeight));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    timers.current.push(window.setTimeout(() => setPhase("form"), reduce ? 0 : MARK_MS));
    const t = timers.current;
    return () => t.forEach(clearTimeout);
  }, []);

  const switchMode = () => {
    if (phase !== "form") return;
    const next = mode === "login" ? "signup" : "login";
    setError(null);
    setNotice(null);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { setMode(next); return; }
    setPhase("out");
    timers.current.push(window.setTimeout(() => { setMode(next); setPhase("mark"); }, OUT_MS));
    timers.current.push(window.setTimeout(() => setPhase("form"), OUT_MS + MARK_MS));
  };

  const oauth = async (provider: "google" | "github") => {
    setError(null);
    setOauthLoading(provider);
    const { error } = await supabase.auth.signInWithOAuth({
      provider,
      options: { redirectTo: `${window.location.origin}/home` },
    });
    if (error) {
      setError(error.message);
      setOauthLoading(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setNotice(null);
    setLoading(true);
    const { error } = mode === "signup"
      ? await supabase.auth.signUp({ email, password, options: { data: { name } } })
      : await supabase.auth.signInWithPassword({ email, password });
    if (error) setError(getAuthErrorMessage(error.message, mode));
    else router.push("/home");
    setLoading(false);
  };

  // Sends Supabase's reset email; its link lands on /reset-password to choose a new one.
  const forgotPassword = async () => {
    setError(null);
    setNotice(null);
    if (!email) { setError("Enter your email address first, then press Forgot password."); return; }
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password`,
    });
    if (error) setError(error.message);
    else setNotice(`If an account exists for ${email}, a reset link is on its way.`);
  };

  const d = (ms: number) => ({ animationDelay: `${ms}ms` });

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden">
      {/* Grid + ambient glow (the original login background) */}
      <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-64 bg-tertiary/10 rounded-full blur-[80px]" />
      </div>

      <Link href="/home" className="absolute left-5 top-5 z-20 opacity-80 transition-opacity hover:opacity-100" aria-label="AERO-SIM home">
        <Image src="/aero-sim-logo-bar.svg" alt="AERO-SIM" width={120} height={28} priority />
      </Link>

      <div className="relative z-10 w-full max-w-[410px] px-4 py-12">
        {/* Octagonal glass card: the outer layer is the glowing edge, the inner the glass. */}
        <div className="login-card-edge">
          <div className="login-card relative">
            <span className="login-corner login-corner-tl" aria-hidden />
            <span className="login-corner login-corner-br" aria-hidden />
            {phase === "mark" && <SwitchMark />}

            <div className="overflow-hidden transition-[height] duration-500 ease-[cubic-bezier(.2,.8,.2,1)]"
                 style={{ height: height ?? undefined }}>
              <div ref={inner} className="px-7 py-8">
                {/* While the mark plays, the incoming form is laid out but invisible, so the card
                    eases to its height; the key remounts it so its entrance plays when revealed. */}
                <div key={`${mode}-${phase === "mark" ? "m" : "f"}`}
                     className={phase === "out" ? "login-out" : phase === "mark" ? "invisible" : ""}>
                  {(
                    <>
                      <h1 className="login-in font-headline-display text-[28px] font-bold uppercase tracking-tight text-primary" style={d(0)}>
                        {mode === "login" ? <>Welcome <span className="text-tertiary">Back</span></> : <>Create <span className="text-tertiary">Account</span></>}
                      </h1>
                      <p className="login-in mt-1 text-[13px] text-on-surface-variant" style={d(50)}>
                        {mode === "login"
                          ? "Sign in to fly your engines and see their history."
                          : "Keep your engines, flights and telemetry in one place."}
                      </p>

                      <form onSubmit={handleSubmit} className="mt-6 space-y-4">
                        {mode === "signup" && (
                          <div className="login-in" style={d(90)}>
                            <label htmlFor="name" className={label}>Pilot name</label>
                            <div className="relative">
                              <input id="name" type="text" required value={name} onChange={(e) => setName(e.target.value)}
                                placeholder="e.g. John Doe" className={`peer ${field}`} />
                              <User size={15} className={icon} />
                            </div>
                          </div>
                        )}

                        <div className="login-in" style={d(120)}>
                          <label htmlFor="email" className={label}>Email address</label>
                          <div className="relative">
                            <input id="email" type="email" required autoComplete="email" value={email}
                              onChange={(e) => setEmail(e.target.value)} placeholder="pilot@aero-sim.io" className={`peer ${field}`} />
                            <Mail size={15} className={icon} />
                          </div>
                        </div>

                        <div className="login-in" style={d(160)}>
                          <div className="mb-1.5 flex items-center justify-between">
                            <label htmlFor="password" className="font-mono text-[10px] uppercase tracking-[0.14em] text-on-surface-variant">Password</label>
                            {mode === "login" && (
                              <button type="button" onClick={forgotPassword}
                                className="text-[11px] text-on-surface-variant/70 transition-colors hover:text-tertiary">
                                Forgot password?
                              </button>
                            )}
                          </div>
                          <div className="relative">
                            <input id="password" type={showPassword ? "text" : "password"} required minLength={6}
                              autoComplete={mode === "login" ? "current-password" : "new-password"}
                              value={password} onChange={(e) => setPassword(e.target.value)}
                              placeholder={mode === "signup" ? "At least 6 characters" : "••••••••"}
                              className={`peer ${field} pr-10`} />
                            <Lock size={15} className={icon} />
                            <button type="button" onClick={() => setShowPassword((v) => !v)}
                              className="absolute right-3 top-1/2 -translate-y-1/2 text-on-surface-variant/50 transition-colors hover:text-primary"
                              aria-label={showPassword ? "Hide password" : "Show password"}>
                              {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                            </button>
                          </div>
                        </div>

                        {error && (
                          <div role="alert" className="rounded border border-red-500/30 bg-red-500/10 px-3 py-2 text-[12px] leading-relaxed text-red-300">
                            {error}
                          </div>
                        )}
                        {notice && (
                          <div role="status" className="rounded border border-tertiary/30 bg-tertiary/10 px-3 py-2 text-[12px] leading-relaxed text-[#ffd9a8]">
                            {notice}
                          </div>
                        )}

                        <button type="submit" disabled={loading}
                          className="login-in login-cta group relative flex w-full items-center justify-center gap-2 overflow-hidden rounded py-3 text-[11px] font-bold uppercase tracking-[0.12em] text-black transition-all duration-200 hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-60 disabled:hover:translate-y-0"
                          style={d(200)}>
                          <span className="login-shine" aria-hidden />
                          {loading
                            ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-black/30 border-t-black" aria-label="Please wait" />
                            : <><Zap size={14} className="fill-black" /> {mode === "login" ? "Sign in" : "Create account"}</>}
                        </button>
                      </form>

                      <div className="login-in my-5 flex items-center gap-3" style={d(240)}>
                        <div className="h-px flex-1 bg-outline-variant/40" />
                        <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-on-surface-variant/60">or</span>
                        <div className="h-px flex-1 bg-outline-variant/40" />
                      </div>

                      <div className="login-in grid grid-cols-2 gap-3" style={d(270)}>
                        <button type="button" onClick={() => oauth("google")} disabled={oauthLoading !== null}
                          className="flex items-center justify-center gap-2 rounded border border-outline-variant/50 bg-white/[0.03] py-2.5 text-[13px] font-medium text-primary transition-all duration-200 hover:-translate-y-0.5 hover:border-tertiary/60 hover:bg-white/[0.07] disabled:opacity-50">
                          <svg width="16" height="16" viewBox="0 0 18 18" aria-hidden="true">
                            <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.9c1.7-1.57 2.7-3.87 2.7-6.62z"/>
                            <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.9-2.26c-.8.54-1.84.86-3.06.86-2.35 0-4.34-1.59-5.05-3.72H.96v2.33A9 9 0 0 0 9 18z"/>
                            <path fill="#FBBC05" d="M3.95 10.7A5.4 5.4 0 0 1 3.67 9c0-.59.1-1.17.28-1.7V4.97H.96A9 9 0 0 0 0 9c0 1.45.35 2.83.96 4.03z"/>
                            <path fill="#EA4335" d="M9 3.58c1.32 0 2.51.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0A9 9 0 0 0 .96 4.97L3.95 7.3C4.66 5.17 6.65 3.58 9 3.58z"/>
                          </svg>
                          {oauthLoading === "google" ? "Redirecting..." : "Google"}
                        </button>
                        <button type="button" onClick={() => oauth("github")} disabled={oauthLoading !== null}
                          className="flex items-center justify-center gap-2 rounded border border-outline-variant/50 bg-white/[0.03] py-2.5 text-[13px] font-medium text-primary transition-all duration-200 hover:-translate-y-0.5 hover:border-tertiary/60 hover:bg-white/[0.07] disabled:opacity-50">
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                            <path d="M12 0C5.37 0 0 5.373 0 12c0 5.303 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222 0 1.606-.015 2.896-.015 3.286 0 .315.216.69.825.572C20.565 21.795 24 17.298 24 12c0-6.627-5.373-12-12-12z"/>
                          </svg>
                          {oauthLoading === "github" ? "Redirecting..." : "GitHub"}
                        </button>
                      </div>

                      <p className="login-in mt-6 text-center font-mono text-[11px] text-on-surface-variant" style={d(300)}>
                        {mode === "login" ? "No account yet? " : "Already have an account? "}
                        <button type="button" onClick={switchMode} className="font-bold text-tertiary hover:underline">
                          {mode === "login" ? "Create one" : "Sign in"}
                        </button>
                      </p>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        <p className="mt-6 text-center">
          <Link href="/home" className="font-mono text-[11px] uppercase tracking-wider text-on-surface-variant/60 transition-colors hover:text-tertiary">
            Continue without an account &rarr;
          </Link>
        </p>
      </div>

      <style>{`
        .login-card-edge { --cut: 26px; padding: 1px; animation: login-card-in .5s cubic-bezier(.2,.8,.2,1) both;
          clip-path: polygon(var(--cut) 0, 100% 0, 100% calc(100% - var(--cut)), calc(100% - var(--cut)) 100%, 0 100%, 0 var(--cut));
          background: linear-gradient(160deg, rgba(255,179,71,.55), rgba(255,255,255,.08) 35%, rgba(255,255,255,.05) 65%, rgba(255,97,27,.4)); }
        .login-card { clip-path: polygon(var(--cut) 0, 100% 0, 100% calc(100% - var(--cut)), calc(100% - var(--cut)) 100%, 0 100%, 0 var(--cut));
          background: linear-gradient(160deg, rgba(28,20,15,.72), rgba(10,9,8,.8)); backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px); }
        .login-corner { position: absolute; width: 18px; height: 18px; background: linear-gradient(135deg, rgba(255,179,71,.55), rgba(255,97,27,.15)); z-index: 1; }
        .login-corner-tl { left: 4px; top: 4px; clip-path: polygon(0 0, 100% 0, 0 100%); }
        .login-corner-br { right: 4px; bottom: 4px; clip-path: polygon(100% 0, 100% 100%, 0 100%); }
        @keyframes login-card-in { from { opacity: 0; transform: scale(.96); } to { opacity: 1; transform: none; } }

        .login-in { opacity: 0; animation: login-rise .5s cubic-bezier(.2,.8,.2,1) forwards; }
        @keyframes login-rise { from { opacity: 0; transform: translateY(12px); filter: blur(4px); } to { opacity: 1; transform: none; filter: none; } }
        .login-out { animation: login-fade-out ${OUT_MS}ms ease forwards; }
        @keyframes login-fade-out { to { opacity: 0; transform: translateY(-8px); filter: blur(6px); } }

        .login-half-a { animation: login-half-a ${MARK_MS}ms cubic-bezier(.2,.9,.2,1) both; }
        .login-half-b { animation: login-half-b ${MARK_MS}ms cubic-bezier(.2,.9,.2,1) both; }
        @keyframes login-half-a { 0% { opacity: 0; transform: translate(-46px,-46px) rotate(-25deg); } 45%, 80% { opacity: 1; transform: none; } 100% { opacity: 0; transform: scale(1.15); } }
        @keyframes login-half-b { 0% { opacity: 0; transform: translate(46px,46px) rotate(-25deg); } 45%, 80% { opacity: 1; transform: none; } 100% { opacity: 0; transform: scale(1.15); } }
        .login-flare { background: radial-gradient(circle, rgba(255,210,150,.95) 0%, rgba(255,145,0,.55) 25%, rgba(255,97,27,0) 65%);
          animation: login-flare ${MARK_MS}ms ease-out both; }
        @keyframes login-flare { 0%, 45% { opacity: 0; transform: scale(.2); } 65% { opacity: 1; transform: scale(.9); } 100% { opacity: 0; transform: scale(1.6); } }

        .login-field:focus { box-shadow: 0 0 0 3px rgba(255,145,0,.15), 0 0 18px rgba(255,145,0,.12); }
        .login-cta { background: linear-gradient(90deg, #ffb347, #FF9100 45%, #ff611b); box-shadow: 0 8px 24px -8px rgba(255,120,0,.6); }
        .login-cta:hover { box-shadow: 0 12px 32px -8px rgba(255,120,0,.85); }
        .login-shine { position: absolute; inset: 0; background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,.45) 50%, transparent 70%); transform: translateX(-120%); }
        .login-cta:hover .login-shine { transition: transform .8s ease; transform: translateX(120%); }
        @media (prefers-reduced-motion: reduce) {
          .login-in, .login-out, .login-card-edge { animation: none; opacity: 1; }
          .login-cta:hover .login-shine { transition: none; }
        }
      `}</style>
    </main>
  );
}
