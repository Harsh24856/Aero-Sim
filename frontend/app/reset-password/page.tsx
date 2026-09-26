"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
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
      <main className="flex min-h-screen items-center justify-center bg-[#050403] px-4">
        <div className="w-full max-w-[400px] rounded-2xl border border-white/10 bg-white/[0.04] p-7 backdrop-blur-2xl">
          <h1 className="text-[24px] font-bold tracking-tight text-primary">Choose a new password</h1>
          {!ready ? (
            <p className="mt-3 text-[13px] text-on-surface-variant">
              Open this page from the link in your reset email. If it has expired,{" "}
              <Link href="/login" className="text-tertiary hover:underline">request a new one</Link>.
            </p>
          ) : (
            <form onSubmit={submit} className="mt-5 space-y-4">
              <div className="relative">
                <input type="password" required minLength={6} autoComplete="new-password" value={password}
                  onChange={(e) => setPassword(e.target.value)} placeholder="At least 6 characters"
                  className="w-full rounded-lg border border-white/10 bg-black/40 py-2.5 pl-10 pr-3 text-sm text-primary outline-none focus:border-tertiary/70" />
                <Lock size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-on-surface-variant/50" />
              </div>
              {error && <div role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-[12px] text-red-300">{error}</div>}
              <button type="submit" disabled={loading}
                className="w-full rounded-lg bg-gradient-to-r from-[#ffb347] via-[#FF9100] to-[#ff611b] py-3 text-[13px] font-bold text-black disabled:opacity-60">
                {loading ? "Saving..." : "Save password"}
              </button>
            </form>
          )}
        </div>
      </main>
    </>
  );
}
