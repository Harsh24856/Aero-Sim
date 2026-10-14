import { test as base, expect, type Page } from "@playwright/test";
import { MOCK_API } from "../playwright.config";

// What the physics API answers while idle. Tests override pieces of it.
export const IDLE_STATE = {
  running: false, engine_model: "Rotax_914_ULF", physics_version: "v5", ai_model_version: "v5",
  params: { altitude: 0, throttle: 0.35, airspeed: 0, aoa: 0, isa_dev_c: 0 },
  telemetry: null, ai: null, residuals: null, advisory: null,
  sim_status: { health: "idle", recoveries: 0, ai: "pending", ai_breaker: "closed" },
};

type Backend = {
  /** Overrides merged into GET /state. */
  state: Record<string, unknown>;
  /** Push one telemetry frame to every open /ws socket. */
  sendFrame: (frame: Record<string, unknown>) => void;
  /** The WebSocket opened by the page, once it connects. */
  wsConnected: Promise<void>;
};

type Fixtures = { backend: Backend; consoleErrors: string[] };

export const test = base.extend<Fixtures>({
  // Mocks the whole physics API (HTTP and WebSocket) for every test.
  backend: [async ({ page }, use) => {
    const backend: Backend = { state: {}, sendFrame: () => {}, wsConnected: Promise.resolve() };
    const sockets: { send: (m: string) => void }[] = [];
    let connected!: () => void;
    backend.wsConnected = new Promise((r) => { connected = r; });
    backend.sendFrame = (frame) => sockets.forEach((ws) => ws.send(JSON.stringify(frame)));

    await page.route(`${MOCK_API}/**`, (route) => {
      const path = new URL(route.request().url()).pathname;
      const body = path === "/state" ? { ...IDLE_STATE, ...backend.state }
        : path === "/engines" ? { engines: ["Rotax_914_ULF", "Rotax_912_ULS", "Rotax_915_iS", "Rotax_916_iS"], current: "Rotax_914_ULF" }
        : path === "/scenarios" ? { status: "ok", scenarios: [], applicable_faults: [], inputs: null }
        : { status: "ok" };
      return route.fulfill({ json: body });
    });
    await page.routeWebSocket(`${MOCK_API.replace("http", "ws")}/ws`, (ws) => {
      sockets.push(ws);
      connected();
    });
    await use(backend);
  }, { auto: true }],

  // Any uncaught exception or console error fails the test, after it has run.
  consoleErrors: [async ({ page }, use) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
    page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
    await use(errors);
    expect(errors, "console errors").toEqual([]);
  }, { auto: true }],
});

export { expect };
export type { Page };
