# UAV Digital Twin Frontend

Next.js interface for controlling the live UAV Digital Twin and viewing real-time telemetry and automatically detected fault risk.

## Requirements

- Node.js `20.9.0` or newer
- npm
- The backend running locally on port `8000`

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

- REST API: `http://localhost:8000`
- Telemetry WebSocket: `ws://localhost:8000/ws`

## Production build

Build and serve the application locally:

```bash
npm run build
npm start
```

The production server is available at [http://localhost:3000](http://localhost:3000) by default.

## Using the interface

- **Start** begins the backend simulation loop.
- **Stop** pauses the loop.
- **Reset** creates a fresh physics twin.
- The altitude, throttle, airspeed, and angle-of-attack sliders send parameter updates to the backend.
- Live telemetry displays engine, propeller, aerodynamic, fuel, temperature, oil, and vibration values.
- Fault risk bars show automatically accumulated stress for eight monitored channels.
- Fault flags show the current detected fault type for each channel.

Faults are not manually selected in the UI. Hold a high throttle/RPM condition or reduce airspeed to observe stress and automatic fault detection change over time.

## Configuration and deployment

The API and WebSocket URLs are currently constants in `pages/index.js`:

```js
const API = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws";
```

There are currently no `NEXT_PUBLIC_*` environment variables or Next.js rewrites for changing these URLs. For a non-local deployment, update both constants and make sure the backend CORS policy and network access allow the frontend origin. The backend currently allows all origins for MVP use; restrict that policy before production deployment.

## Troubleshooting

### The page shows no telemetry

Confirm that the backend is running and that `http://localhost:8000/docs` opens. Then reload the frontend page. The browser opens the WebSocket as soon as the page loads.

### The browser reports a WebSocket error

Check that the backend is listening on port `8000` and that the URL is `ws://localhost:8000/ws`. If the backend is running on another host or port, update both constants in `pages/index.js`.

### Dependency or build errors

Verify Node.js is at least `20.9.0`, remove an incomplete installation if necessary, and reinstall from the lockfile:

```bash
rm -rf node_modules .next
npm ci
npm run build
```

## Project files

- `pages/index.js`: main control dashboard and API/WebSocket client
- `pages/_app.js`: Next.js application wrapper
- `package.json`: scripts and dependencies
- `styles/`: frontend styles directory
