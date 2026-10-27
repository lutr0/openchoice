# Launch video, stills and social card

`index.html` + `main.js` + `styles.css` draw every frame as a pure function of time (`window.renderFrame(t)`), so the output is deterministic. Fonts are self-hosted in `fonts/` under the SIL OFL. `img/chart-07.png` is project-owned synthetic chart artwork from the OpenChoice development set (used in the vision scene).

1. `npm install` (or `bun install`) here to get `puppeteer-core`. You also need Chromium or Chrome (set `CHROME_PATH`) and `ffmpeg` with libx264.
2. `node render.mjs` writes `out/openchoice-launch.mp4` (1920x1080, 60 fps, 54 s, H.264 High, no audio).
3. `node render.mjs --assets` writes every image to `../../assets/`: `openchoice-scoreboard.png` ("where it stands" feature table), `openchoice-capabilities.png`, `openchoice-cpu.png`, `openchoice-finetune.png` (1920x1080) and `openchoice-social.png` (1200x630 link preview). Single images: `--poster` (the feature table), `--still <capabilities|cpu|finetune|scoreboard>`, `--social`.
4. `node render.mjs --stills 8,17,25` writes preview PNGs of video frames to `out/stills/`. Other flags: `--crf 17`, `--workers 4`, `--out <file>`, `--to <seconds>`. Rendering uses the CPU only (`--disable-gpu`).

The quickstart scene shows the published accounts (`lutr0` on Hugging Face and GitHub). Pass `--hf-user <name>` and `--gh-user <name>` to any command to override them.

Scenes: hook (0–5 s), CPU footprint, optional GPU offload, vision on CPU, quickstart, fine-tune and export, where it stands (artifact feature table, not a ranking), close. Every number and comparator fact is a constant near the top of `main.js`; sources are listed in `../../LAUNCH.md`.
