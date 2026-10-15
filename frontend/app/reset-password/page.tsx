"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import { supabase } from "@/lib/supabase";
import { Lock } from "lucide-react";

/**
 * Where Supabase's password-reset email lands (login page, "Forgot password?"). The
 * Supabase client reads the recovery token from the link and opens a session; the user
 * then chooses a new password here.
 */
export default function ResetPasswordPage() {
  const router = useRouter();
  const [ready, setReady] = useState(false);
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => { if (data.session) setReady(true); });
    const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === "PASSWORD_RECOVERY" || session) setReady(true);
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    const { error } = await supabase.auth.updateUser({ password });
    setLoading(false);
    if (error) setError(error.message);
    else router.push("/home");
  };

  return (
    <>
      {/* same frame as /login: the app background, the brand mark home, the report type */}
      <main className="relative flex min-h-screen items-center justify-center px-4">
        <Link href="/home" className="absolute left-5 top-5 opacity-80 transition-opacity hover:opacity-100" aria-label="AERO-SIM home">
          <Image src="/aero-sim-logo-bar.svg" alt="AERO-SIM" width={120} height={28} priority />
        </Link>
        <div className="w-full max-w-[400px] rounded-lg border border-outline-variant/50 bg-surface/80 p-7 shadow-[2px_2px_0px_#000000] backdrop-blur-md">
          <p className="font-mono text-[11px] font-bold uppercase tracking-[0.16em] text-tertiary">Account // reset</p>
          <h1 className="mt-2 font-headline-display text-2xl font-bold uppercase tracking-tight text-primary">Choose a new password</h1>
          {!ready ? (
            <p className="mt-3 text-[13px] text-on-surface-variant">
              Open this page from the link in your reset email. If it has expired,{" "}
              <Link href="/login" className="text-tertiary hover:underline">request a new one</Link>.
            </p>
          ) : (
            <form onSubmit={submit} className="mt-5 space-y-4">
              <label htmlFor="new-password" className="block font-mono text-[11px] uppercase tracking-[0.14em] text-on-surface-variant">New password</label>
              <div className="relative !mt-1.5">
                <input id="new-password" type="password" required minLength={6} autoComplete="new-password" value={password}
                  onChange={(e) => setPassword(e.target.value)} placeholder="At least 6 characters"
                  className="w-full rounded border border-outline-variant/50 bg-black/45 py-2.5 pl-10 pr-3 text-sm text-primary outline-none focus:border-tertiary/70" />
                <Lock size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-on-surface-variant/80" />
              </div>
              {error && <div role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-[12px] text-red-300">{error}</div>}
              <button type="submit" disabled={loading}
                className="w-full rounded bg-gradient-to-r from-[#ffb347] via-[#FF9100] to-[#ff611b] py-3 font-mono text-[12px] font-bold uppercase tracking-wider text-black transition-transform hover:-translate-y-0.5 disabled:opacity-60">
                {loading ? "Saving..." : "Save password"}
              </button>
            </form>
          )}
        </div>
      </main>
    </>
  );
}
