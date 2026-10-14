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
