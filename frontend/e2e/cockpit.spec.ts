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
