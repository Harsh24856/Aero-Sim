import { test, expect } from "./fixtures";

// Every route renders, with no console error, on desktop and mobile (two projects).
const PUBLIC = ["/home", "/simulate", "/engine", "/engine_info", "/about_us", "/login", "/reset-password"];
const SIGNED_IN_ONLY = ["/mission", "/dashboard", "/telemetry"];

for (const path of PUBLIC) {
  test(`${path} renders`, async ({ page }) => {
    const res = await page.goto(path);
    expect(res?.status()).toBe(200);
    await expect(page.locator("body")).not.toBeEmpty();
    // No sideways scroll on any viewport.
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });
}

for (const path of SIGNED_IN_ONLY) {
  test(`${path} sends a guest to /login`, async ({ page }) => {
    await page.goto(path);
    await expect(page).toHaveURL(/\/login$/);
  });
}

test("/ redirects to /home", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/home$/);
});

test("unknown route is a 404", async ({ page, consoleErrors }) => {
  const res = await page.goto("/no-such-page");
  expect(res?.status()).toBe(404);
  // the themed page, readable on the dark body (Next's default was black on black)
  await expect(page.getByRole("heading", { name: "Page not found" })).toBeVisible();
  consoleErrors.length = 0;   // the browser logs the 404 itself
});

test("/engine shows each engine's real AI status", async ({ page, backend }) => {
  backend.engines = { ai_online: true, ai_valid_engines: ["Rotax_914_ULF", "Rotax_912_ULS", "Rotax_915_iS", "Rotax_916_iS"],
                      ai_placeholder_engines: ["Rotax_912_ULS", "Rotax_915_iS", "Rotax_916_iS"] };
  await page.goto("/engine");
  await expect(page.getByText("1 of 4 AI twins ready")).toBeVisible();
  await expect(page.getByText("AI stand-in")).toHaveCount(3);
});

test("/engine says so when the AI service is down", async ({ page, backend }) => {
  backend.engines = { ai_online: false };
  await page.goto("/engine");
  await expect(page.getByText("AI offline").first()).toBeVisible();
  await expect(page.getByText("AI ready")).toHaveCount(0);
});

// The 3D scenes' HDR lighting is served from public/hdri, not drei's CDN: offline (a
// demo venue, a blocked network) every 3D view renders with no error and no request out.
for (const path of ["/engine", "/simulate", "/engine_info"]) {
  test(`${path} renders its 3D views with no CDN`, async ({ page }) => {
    const outside: string[] = [];
    await page.route(/raw\.githack\.com|raw\.githubusercontent\.com/, (r) => { outside.push(r.request().url()); return r.abort(); });
    await page.goto(path);
    await expect(page.locator("canvas").first()).toBeVisible();
    await page.waitForTimeout(3000);   // the HDR loads a moment after first paint
    expect(outside, "3D assets fetched from a CDN").toEqual([]);
  });
}
