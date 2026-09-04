import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

// Single shared client for the whole app - uses the anon/publishable key, which is
// safe to expose client-side by design (Row-Level Security on every table is the
// actual security boundary, not keeping this key secret).
export const supabase = createClient(supabaseUrl, supabaseAnonKey);
