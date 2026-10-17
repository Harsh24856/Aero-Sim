import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

// Without the env (CI, a fresh clone) the app still builds and serves every page;
// sign-in and run history fail until .env.local is filled in (see .env.example).
// createClient throws on an empty URL, which used to fail `next build` outright.
const supabaseConfigured = Boolean(supabaseUrl && supabaseAnonKey);
if (!supabaseConfigured) {
  console.warn("NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY not set - sign-in and run history are disabled.");
}

// Single shared client for the whole app - uses the anon/publishable key, which is
// safe to expose client-side by design (Row-Level Security on every table is the
// actual security boundary, not keeping this key secret).
export const supabase = createClient(
  supabaseUrl || "http://localhost:54321",
  supabaseAnonKey || "supabase-not-configured",
);
