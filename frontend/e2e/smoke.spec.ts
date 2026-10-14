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
  consoleErrors.length = 0;   // the browser logs the 404 itself
});
