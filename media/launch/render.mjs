#!/usr/bin/env node
// Deterministic renderer for the OpenChoice-9B launch video, poster, stills and social card.
//
//   node render.mjs                     # full video -> out/openchoice-launch.mp4
//   node render.mjs --assets            # every image below, in one run
//   node render.mjs --poster            # poster     -> ../../assets/openchoice-scoreboard.png
//   node render.mjs --still cpu         # still      -> ../../assets/openchoice-<name>.png (capabilities, cpu, finetune)
//   node render.mjs --social            # 1200x630   -> ../../assets/openchoice-social.png
//   node render.mjs --stills 2,8.5,40   # preview PNGs -> out/stills/
//
// Options: --fps 60  --crf 17  --workers 4  --out <file>  --to <seconds> (render a prefix)
//          --hf-user <name> --gh-user <name>  Hugging Face / GitHub accounts shown in the quickstart scene
// Env: CHROME_PATH (defaults to the Playwright Chromium build), FFMPEG (defaults to `ffmpeg`).
import { spawn } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { homedir } from "node:os";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const HERE = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  if (i < 0) return def;
  const v = args[i + 1];
  return v == null || v.startsWith("--") ? true : v;
};

const FPS = Number(opt("fps", 60));
const CRF = String(opt("crf", 17));
const WORKERS = Number(opt("workers", 4));
const CHROME = process.env.CHROME_PATH || join(homedir(), ".cache/ms-playwright/chromium-1208/chrome-linux64/chrome");
const FFMPEG = process.env.FFMPEG || "ffmpeg";

const MIME = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".woff2": "font/woff2", ".svg": "image/svg+xml", ".png": "image/png" };
const ASSETS = join(HERE, "../../assets");
const HF_USER = opt("hf-user", false);
const GH_USER = opt("gh-user", false);
const ACCOUNTS = [["hf-user", HF_USER], ["gh-user", GH_USER]].filter(([, v]) => typeof v === "string").map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");

function serve() {
  const server = createServer((req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^\/+/, "");
    const file = join(HERE, path || "index.html");
    if (!file.startsWith(HERE) || !existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream" });
    createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(0, "127.0.0.1", () => ok(server)));
}

async function openPage(browser, url, viewport = { width: 1920, height: 1080 }) {
  const page = await browser.newPage();
  await page.setViewport({ ...viewport, deviceScaleFactor: 1 });
  page.on("pageerror", (e) => console.error("page error:", e.message));
  await page.goto(url, { waitUntil: "load" });
  await page.evaluate(() => window.ready);
  return page;
}

const shot = (page) => page.screenshot({ type: "png", optimizeForSpeed: true, captureBeyondViewport: false });

async function main() {
  const server = await serve();
  const base = `http://127.0.0.1:${server.address().port}/index.html${ACCOUNTS ? `?${ACCOUNTS}` : ""}`;
  const withQuery = (qs) => `${base}${base.includes("?") ? "&" : "?"}${qs}`;
  // One browser per worker: headless Chrome stalls screenshots of background tabs.
  const browsers = [];
  const launch = async () => {
    const b = await puppeteer.launch({
      executablePath: CHROME,
      headless: true,
      args: ["--disable-gpu", "--font-render-hinting=none", "--force-color-profile=srgb", "--hide-scrollbars", "--no-first-run"],
    });
    browsers.push(b);
    return b;
  };
  const browser = await launch();

  try {
    const images = [];
    const all = opt("assets", false);
    if (all || opt("poster", false)) {
      const custom = typeof opt("poster") === "string" ? opt("poster") : null;
      images.push({ out: resolve(custom || join(ASSETS, "openchoice-scoreboard.png")), url: withQuery("still=scoreboard") });
    }
    const stillNames = all ? ["capabilities", "cpu", "finetune"] : typeof opt("still") === "string" ? [opt("still")] : [];
    for (const name of stillNames) images.push({ out: join(ASSETS, `openchoice-${name}.png`), url: withQuery(`still=${name}`) });
    if (all || opt("social", false)) {
      images.push({ out: join(ASSETS, "openchoice-social.png"), url: withQuery("social"), viewport: { width: 1200, height: 630 } });
    }
    if (images.length) {
      mkdirSync(ASSETS, { recursive: true });
      for (const { out, url, viewport } of images) {
        const page = await openPage(browser, url, viewport);
        await page.screenshot({ path: out, type: "png" });
        await page.close();
        console.log(`image -> ${out}`);
      }
      return;
    }

    const stills = opt("stills", false);
    if (stills) {
      const dir = join(HERE, "out/stills");
      mkdirSync(dir, { recursive: true });
      const page = await openPage(browser, base);
      for (const t of String(stills).split(",").map(Number)) {
        await page.evaluate((x) => window.renderFrame(x), t);
        const file = join(dir, `t-${t.toFixed(2).padStart(5, "0")}.png`);
        await page.screenshot({ path: file, type: "png" });
        console.log(file);
      }
      return;
    }

    const out = resolve(opt("out", join(HERE, "out/openchoice-launch.mp4")));
    mkdirSync(dirname(out), { recursive: true });
    const pages = await Promise.all(
      Array.from({ length: WORKERS }, async (_, w) => openPage(w === 0 ? browser : await launch(), base)),
    );
    const duration = await pages[0].evaluate(() => window.DURATION);
    const total = Math.round(Math.min(duration, Number(opt("to", duration))) * FPS);

    const ff = spawn(FFMPEG, [
      "-y", "-loglevel", "error",
      "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "png", "-i", "-",
      "-an",
      "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
      "-c:v", "libx264", "-preset", "slow", "-crf", CRF, "-profile:v", "high",
      "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
      "-r", String(FPS), "-movflags", "+faststart",
      out,
    ], { stdio: ["pipe", "inherit", "inherit"] });
    const done = new Promise((ok, fail) => ff.on("close", (c) => (c === 0 ? ok() : fail(new Error(`ffmpeg exited ${c}`)))));
    const write = (buf) => new Promise((ok) => (ff.stdin.write(buf) ? ok() : ff.stdin.once("drain", ok)));

    const started = Date.now();
    for (let f = 0; f < total; f += WORKERS) {
      const batch = await Promise.all(pages.map(async (page, w) => {
        const i = f + w;
        if (i >= total) return null;
        await page.evaluate((x) => window.renderFrame(x), i / FPS);
        return shot(page);
      }));
      for (const buf of batch) if (buf) await write(buf);
      if ((f / WORKERS) % 30 === 0) {
        const el = (Date.now() - started) / 1000;
        process.stdout.write(`\rframe ${Math.min(f + WORKERS, total)}/${total}  ${el.toFixed(0)}s`);
      }
    }
    ff.stdin.end();
    await done;
    console.log(`\nvideo -> ${out}`);
  } finally {
    await Promise.all(browsers.map((b) => b.close()));
    server.close();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
