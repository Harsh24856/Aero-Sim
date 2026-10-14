import { defineConfig, devices } from "@playwright/test";

// The physics API is mocked in every test (e2e/fixtures.ts), so the app is pointed at
// a port nothing listens on: a request that escapes the mocks fails loudly instead of
// reaching a real backend. Port 3100 keeps clear of a dev server on 3000.
const PORT = 3100;
export const MOCK_API = "http://127.0.0.1:8999";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  // Headless Chromium renders the three.js scenes in software: more than two pages at
  // once starve each other of CPU (seven workers timed out half the suite).
  workers: 2,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: { baseURL: `http://localhost:${PORT}`, trace: "retain-on-failure" },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    // CI tests the production build; locally `next dev` is faster to start.
    command: process.env.CI ? `npx next build && npx next start -p ${PORT}` : `npx next dev -p ${PORT}`,
    url: `http://localhost:${PORT}/home`,
    reuseExistingServer: !process.env.CI,
    timeout: 240_000,
    env: { NEXT_PUBLIC_API_URL: MOCK_API },
  },
});
