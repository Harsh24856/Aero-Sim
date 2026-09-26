# AERO-SIM Frontend

Next.js 16 (App Router) + React 19 cockpit for the UAV engine digital twin: live telemetry, 3D engine models, AI diagnostics, mission replay and reports.

**Stack:** Next.js 16 · React 19 · TypeScript · Tailwind CSS 4 · Recharts · three.js / @react-three/fiber · Supabase JS · EmailJS

---

## Setup

Requires Node.js ≥ 20.9.

```bash
cd frontend
npm ci
cp .env.example .env.local   # then fill in values
npm run dev                  # http://localhost:3000
```

Start the backend (`:8000`) first — pages open their WebSocket on load.

| Script | Does |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Production build |
| `npm start` | Serve the production build |

### Environment (`.env.local`, never committed)

| Variable | Required for | Where to get it |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | Login, run history | Supabase → Settings → API |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Login, run history | Supabase → Settings → API (anon/publishable) |
| `NEXT_PUBLIC_EMAILJS_SERVICE_ID` | `/about_us` contact form | EmailJS → Email Services |
| `NEXT_PUBLIC_EMAILJS_TEMPLATE_ID` | `/about_us` contact form | EmailJS → Email Templates |
| `NEXT_PUBLIC_EMAILJS_PUBLIC_KEY` | `/about_us` contact form | EmailJS → Account |

EmailJS template variables: `{{title}}`, `{{name}}`, `{{email}}`, `{{message}}`, `{{time}}`; set *Reply To* to `{{email}}`. Restart `npm run dev` after editing `.env.local`.

---

## Routes (`app/`)

| Route | Page |
|---|---|
| `/` | Landing |
| `/login` | Supabase sign-in |
| `/home` | Home / engine picker |
| `/dashboard` | Run history |
| `/simulate?engine=<key>` | Live simulator: controls, telemetry, AI diagnostics |
| `/engine`, `/engine_info` | 3D engine viewer and engine specs |
| `/mission` | Mission presets |
| `/mission/replay/[id]` | Replay a recorded mission |
| `/mission/report/[id]` | Post-flight report + AI summary |
| `/telemetry`, `/telemetry/[id]` | Telemetry browser / single run |
| `/about_us` | Team + contact form |

## Code layout

| Path | Contents |
|---|---|
| `components/Simulator*.tsx` | Per-engine simulator panels (914 default, 912/915/916 variants) |
| `components/EngineViewer*.tsx` | three.js engine models (GLBs in `public/models/`) |
| `components/Diagnostics.tsx`, `Meters.tsx`, `Sensr.tsx`, `HealthVignette.tsx` | Instrument + AI panels |
| `components/MissionReplay.tsx` | Replay timeline |
| `components/*Plane.tsx`, `*Illustration.tsx` | Decorative / explanatory graphics |
| `lib/supabase.ts` | Supabase client |
| `lib/engineSound.ts` | Web-Audio engine sound |
| `lib/missionPresets.ts`, `timeScale.ts`, `units.ts` | Mission data, sim-time scaling, unit conversion |
| `public/` | Logos, favicon source, GLB models |

## Backend URLs

`http://localhost:8000` and `ws://localhost:8000/ws` are hard-coded as `API` / `WS_URL` constants at the top of the pages that use them (e.g. `app/simulate/page.tsx`). For a non-local deploy, change those and restrict the backend CORS policy to your origin.

## Troubleshooting

- **No telemetry** — check <http://localhost:8000/docs> opens, then reload.
- **WebSocket error** — backend must listen on `:8000`; otherwise update `WS_URL`.
- **Build errors** — `rm -rf node_modules .next && npm ci && npm run build`.
