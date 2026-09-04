"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import { supabase } from "@/lib/supabase";

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
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

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
        <div className="absolute inset-0 grid-bg pointer-events-none opacity-50 z-0" />

        <div className="relative z-10 w-full max-w-md mx-auto px-4 py-16">
          <div className="bg-surface/80 backdrop-blur-xl border border-tertiary/30 rounded-lg p-8">
            <h1 className="font-headline-display text-2xl font-bold text-primary uppercase tracking-tight mb-1 text-center">
              {mode === "login" ? "Sign In" : "Create Account"}
            </h1>
            <p className="text-on-surface-variant text-[13px] text-center mb-6">
              {mode === "login" ? "Access your simulation history" : "Track your simulation runs"}
            </p>

            <form onSubmit={handleSubmit} className="space-y-4">
              {mode === "signup" && (
                <div>
                  <label className="block text-[11px] uppercase tracking-[0.1em] text-on-surface-variant mb-1.5">Name</label>
                  <input
                    type="text" required value={name} onChange={(e) => setName(e.target.value)}
                    className="w-full bg-surface-container-highest/60 border border-outline-variant/50 rounded px-3 py-2.5 text-primary text-sm focus:outline-none focus:border-tertiary"
                  />
                </div>
              )}
              <div>
                <label className="block text-[11px] uppercase tracking-[0.1em] text-on-surface-variant mb-1.5">Email</label>
                <input
                  type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-surface-container-highest/60 border border-outline-variant/50 rounded px-3 py-2.5 text-primary text-sm focus:outline-none focus:border-tertiary"
                />
              </div>
              <div>
                <label className="block text-[11px] uppercase tracking-[0.1em] text-on-surface-variant mb-1.5">Password</label>
                <input
                  type="password" required minLength={6} value={password} onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-surface-container-highest/60 border border-outline-variant/50 rounded px-3 py-2.5 text-primary text-sm focus:outline-none focus:border-tertiary"
                />
              </div>

              {error && (
                <div className="text-[13px] text-red-400 bg-red-500/10 border border-red-500/30 rounded px-3 py-2">
                  {error}
                </div>
              )}

              <button
                type="submit" disabled={loading}
                className="w-full py-3 bg-tertiary text-black text-[11px] tracking-[0.1em] font-bold uppercase rounded hover:brightness-110 transition-all disabled:opacity-50"
              >
                {loading ? "Please wait..." : mode === "login" ? "Sign In" : "Create Account"}
              </button>
            </form>

            <p className="text-center text-[13px] text-on-surface-variant mt-6">
              {mode === "login" ? "No account yet? " : "Already have an account? "}
              <button
                onClick={() => { setMode(mode === "login" ? "signup" : "login"); setError(null); }}
                className="text-tertiary hover:underline"
              >
                {mode === "login" ? "Create one" : "Sign in"}
              </button>
            </p>
          </div>

          <p className="text-center mt-4">
            <Link href="/home" className="text-[13px] text-on-surface-variant hover:text-tertiary transition-colors">
              Continue without an account
            </Link>
          </p>
        </div>
      </main>
    </>
  );
}
