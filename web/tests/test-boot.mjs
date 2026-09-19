// Loads the real page in a real browser and reports what actually happens.
//
//     cd web && npm run dev          # in one terminal
//     node tests/test-boot.mjs       # in another
//
// Written after two failed guesses at the onnxruntime wasm path. Both failures
// were browser-only - the module graph resolved, the files served with the
// right MIME type, and the node-side parity tests passed throughout. Only a
// real browser executing the real page shows whether the runtime instantiates.
//
// Chromium is given a fake camera so the whole pipeline runs: fake video in,
// landmarks out, prediction on screen.
import { chromium } from "playwright";

const URL_BASE = process.env.SIGNSPEAK_URL ?? "http://localhost:5199/";
const failures = [];

function check(name, ok, detail = "") {
  if (!ok) failures.push(name);
  console.log(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(46)} ${detail}`);
}

const browser = await chromium.launch({
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
  ],
});
const context = await browser.newContext({ permissions: ["camera"] });
const page = await context.newPage();

const consoleErrors = [];
const pageErrors = [];
const failedRequests = [];
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
page.on("pageerror", (e) => pageErrors.push(e.message));
page.on("requestfailed", (r) =>
  failedRequests.push(`${r.url()} ${r.failure()?.errorText ?? ""}`));

await page.goto(URL_BASE, { waitUntil: "domcontentloaded" });
check("page loads", true, URL_BASE);

await page.click("#start");

// Model download plus wasm instantiation is slow on a cold cache.
const note = page.locator("#overlay-note");
const overlay = page.locator("#overlay");

let booted = false;
for (let i = 0; i < 90; i++) {
  if (await overlay.evaluate((el) => el.classList.contains("hidden"))) {
    booted = true;
    break;
  }
  const text = (await note.textContent())?.trim() ?? "";
  if (text && !text.includes("loading")) break;
  await page.waitForTimeout(1000);
}

const noteText = (await note.textContent())?.trim() ?? "";
check("model and camera start", booted, booted ? "" : `overlay says: ${noteText}`);

if (booted) {
  // Let the rolling buffer fill and a few predictions run.
  await page.waitForTimeout(6000);

  const fps = (await page.locator("#fps").textContent())?.trim();
  const status = (await page.locator("#status").textContent())?.trim();
  const prediction = (await page.locator("#prediction").textContent())?.trim();
  const confidence = (await page.locator("#confidence-text").textContent())?.trim();

  check("render loop runs", /\d/.test(fps ?? ""), `fps: ${fps}`);
  check("model reports its classes", /letters/.test(status ?? ""), status);
  check("prediction element updates", prediction !== undefined,
        `prediction "${prediction}", ${confidence}`);

  const canvasPainted = await page.evaluate(() => {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < data.length; i += 4000) {
      if (data[i] || data[i + 1] || data[i + 2]) return true;
    }
    return false;
  });
  check("camera frames reach the canvas", canvasPainted);

  await page.screenshot({ path: "tests/boot-screenshot.png" });
  console.log("      screenshot: tests/boot-screenshot.png");
}

const realErrors = consoleErrors.filter((e) =>
  !e.includes("favicon") && !e.toLowerCase().includes("autoplay"));
check("no console errors", realErrors.length === 0,
      realErrors.slice(0, 3).join(" | "));
check("no uncaught exceptions", pageErrors.length === 0,
      pageErrors.slice(0, 3).join(" | "));
check("no failed requests", failedRequests.length === 0,
      failedRequests.slice(0, 3).join(" | "));

await browser.close();

console.log();
console.log(failures.length ? `${failures.length} FAILURE(S)` : "ALL PASS");
process.exit(failures.length ? 1 : 0);
