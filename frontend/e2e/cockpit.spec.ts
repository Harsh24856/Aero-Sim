import { test, expect } from "./fixtures";

test("live telemetry from the WebSocket reaches the sensor panel", async ({ page, backend }) => {
  await page.goto("/simulate");
  await backend.wsConnected;
  const sensors = page.locator("aside").filter({ hasText: "Sensors" }).first();
  await expect(sensors).toContainText("waiting for data");

  backend.sendFrame({ time: 1, physics_version: "v5", engine_rpm: 4321, egt: 777 });
  await expect(sensors).toContainText("4321");
  await expect(sensors).not.toContainText("waiting for data");

  backend.sendFrame({ time: 2, physics_version: "v5", engine_rpm: 4400, egt: 780 });
  await expect(sensors).toContainText("4400");
});

test("the flight display keeps a usable size on every viewport", async ({ page }, testInfo) => {
  await page.goto("/simulate");
  const display = page.getByTestId("flight-display");
  const box = await display.boundingBox();
  await testInfo.attach("cockpit", { body: await page.screenshot({ fullPage: true }), contentType: "image/png" });
  expect(box?.height ?? 0).toBeGreaterThanOrEqual(220);
  expect(box?.width ?? 0).toBeGreaterThanOrEqual(300);
});

test("no STOP button before a flight; a flight started elsewhere is announced", async ({ page, backend }) => {
  await page.goto("/simulate");
  await backend.wsConnected;
  await expect(page.getByRole("button", { name: "STOP SIMULATION" })).toHaveCount(0);
  await expect(page.getByText("A flight is already running")).toHaveCount(0);

  backend.sendFrame({ time: 50, physics_version: "v5", engine_rpm: 5100 });
  await expect(page.getByText("A flight is already running")).toBeVisible();
  // Frames stop (the other flight ended): the notice goes away.
  await expect(page.getByText("A flight is already running")).toHaveCount(0, { timeout: 10_000 });
});

// One Simulator, one landscape per engine (components/Simulator.tsx SCENES).
const SCENERY: Record<string, { hills: boolean; sky: string }> = {
  Rotax_914_ULF: { hills: true, sky: "rgb(178, 60, 34)" },     // sunset mountains
  Rotax_912_ULS: { hills: true, sky: "rgb(74, 144, 196)" },    // desert dunes
  Rotax_915_iS: { hills: true, sky: "rgb(47, 127, 201)" },     // ocean
  Rotax_916_iS: { hills: false, sky: "rgb(5, 8, 24)" },        // night: stars, moon, no hills
};
for (const [engine, want] of Object.entries(SCENERY)) {
  test(`${engine} flies over its own scenery`, async ({ page }) => {
    await page.goto(`/simulate?engine=${engine}`);
    const display = page.getByTestId("flight-display");
    await expect(display.locator("#farGrad")).toHaveCount(want.hills ? 1 : 0);
    await expect(display.locator("circle")).toHaveCount(want.hills ? 0 : 90);
    const skies = await display.locator("div").evaluateAll((els) => els.map((e) => getComputedStyle(e).backgroundImage));
    expect(skies.some((b) => b.startsWith(`linear-gradient(${want.sky}`))).toBe(true);
  });
}
