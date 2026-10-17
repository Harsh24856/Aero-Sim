// Where the physics API lives. NEXT_PUBLIC_ vars are inlined at build time, so set
// NEXT_PUBLIC_API_URL before `next build` for any host other than this machine.
export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
export const WS_URL = `${API.replace(/^http/, "ws")}/ws`;
