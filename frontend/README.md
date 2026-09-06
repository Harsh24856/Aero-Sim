# UAV Digital Twin Frontend

Next.js interface for controlling the live UAV Digital Twin and viewing real-time telemetry and automatically detected fault risk.

## Requirements


Check your Node version with:

```bash
node --version
npm --version
```

## Install

From this directory:

```bash
cd /Users/harsh/Documents/UAV_Engine/frontend
npm ci
```

Use `npm install` if you intentionally need to regenerate or update the lockfile.

## Development

Start the Next.js development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in a browser. Start the backend first so the page can establish its WebSocket connection.

The page connects to:


## Production build

Build and serve the application locally:

```bash
npm run build
npm start
```

The production server is available at [http://localhost:3000](http://localhost:3000) by default.

## Using the interface


Faults are not manually selected in the UI. Hold a high throttle/RPM condition or reduce airspeed to observe stress and automatic fault detection change over time.

## Configuration and deployment

The application uses the App Router under `app/`. Supabase authentication and history require these variables in `frontend/.env.local`:

```dotenv
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-publishable-or-anon-key
```

### EmailJS contact form

The About Us contact form sends through EmailJS. Copy `.env.example` to `.env.local` and fill in the three values from your EmailJS dashboard:

```dotenv
NEXT_PUBLIC_EMAILJS_SERVICE_ID=service_xxxxxxx
NEXT_PUBLIC_EMAILJS_TEMPLATE_ID=template_xxxxxxx
NEXT_PUBLIC_EMAILJS_PUBLIC_KEY=your_public_key
```

In EmailJS, set the template recipient (`To Email`) to the address that should receive enquiries. The template variables used by the form are `{{title}}`, `{{name}}`, `{{email}}`, `{{message}}`, and `{{time}}`. Set `Reply To` to `{{email}}` so replies go directly to the sender. Restart `npm run dev` after changing `.env.local`; Next.js loads public environment variables at startup.

The backend and WebSocket currently use local URLs (`http://localhost:8000` and `ws://localhost:8000/ws`) in the simulation components. For a non-local deployment, update those client URLs and configure the backend CORS policy and network access for the frontend origin. The backend currently allows all origins for MVP use; restrict that policy before production deployment.

## Troubleshooting

### The page shows no telemetry

Confirm that the backend is running and that `http://localhost:8000/docs` opens. Then reload the frontend page. The browser opens the WebSocket as soon as the page loads.

### The browser reports a WebSocket error

Check that the backend is listening on port `8000` and that the URL is `ws://localhost:8000/ws`. If the backend is running on another host or port, update the API and WebSocket URLs in the simulation client components under `app/` and `components/`.

### Dependency or build errors

Verify Node.js is at least `20.9.0`, remove an incomplete installation if necessary, and reinstall from the lockfile:

```bash
rm -rf node_modules .next
npm ci
npm run build
```

## Project files

