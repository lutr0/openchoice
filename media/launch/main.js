/*
 * OpenChoice-9B launch video, poster, stills and social card.
 *
 * Every visual property is a pure function of time: `window.renderFrame(t)` (seconds)
 * sets all state, so the headless renderer can capture any frame deterministically.
 * No CSS transitions, CSS animations or requestAnimationFrame timing are used.
 *
 * Numbers: every figure below traces to a file listed in ../../LAUNCH.md ("Sources").
 * Keep them in sync with RESULTS.md.
 */
(() => {
  "use strict";

  const DURATION = 54;
  const q = new URLSearchParams(location.search);
  // Override the published accounts with `node render.mjs --hf-user <name> --gh-user <name>`.
  const HF_USER = q.get("hf-user") || "lutr0";
  const GH_USER = q.get("gh-user") || "lutr0";
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // ---------------------------------------------------------------- data
  const PUB = "published";
  const MEAS = "measured locally";

  // CPU runtime: results/runtime/gguf-q4-norepack.json (Q4_K_M, repacking off), vision-q4-q8_0.json,
  // http-cpu-resources.json, production-gpu-verification.json, gpu-measurement-4.json.
  // GPU auto policy: README.md / openchoice/llamacpp.py (admit ≥ allowance + 2 GiB free, release < 1 GiB).
  const RT = {
    weightsGiB: 5.24,
    projectorMiB: 595,
    rssText: 6.1,
    rssVision: 6.7,
    lat300: 1.7,
    lat600: 3.3,
    cpuCurve: [[150, 0.90], [300, 1.73], [600, 3.28], [2000, 11.05], [3700, 20.77]],
    cpu600: 3.28,
    cpu3700: 20.77,
    gpuLayers: 4,
    gpuVramGiB: 1.9,
    gpu600: 1.02,
    gpu3700: 4.71,
    admitGiB: 4,
    admitReserveGiB: 2,
    releaseGiB: 1,
    quietS: 62,
  };

  // Capability cards (still only): released GGUF on CPU, results/gguf-nimble-public.json by_family;
  // chart card: results/runtime/vision-q4-q8_0.json.

  // "Where it stands": released-artifact facts, not scores. Wording mirrors the README comparison
  // table (source of truth; evidence: results/artifact-comparison.json, snapshot 2026-09-29).
  const FEAT_COLS = ["/v1/systemone", "Vision", "Developer GGUF", "Documented artifact", "CPU documented", "Fine-tune path"];
  const FEATURES = [
    ["OpenChoice-9B", "SDK 0.6 tested", "CPU Python API", "Q4 + projector", "5.8 GiB serving / 6.0 GiB full", "6–7 GiB RAM measured", "LoRA → GGUF kit"],
    ["Winnow-12B Q8", "Documented", "Yes", "Q8 / BF16", "11.80 GiB + 0.163 GiB projector", "CPU/offload supported; GPU tested", "LoRA recipe; data private"],
    ["JevK5 v0.3", "Documented", "Text interface", "Q4 / Q5 / Q8", "2.71 GB Q4", "Yes", "LoRA recipe + trainer code"],
    ["Plumb-4B", "JevK5 runtime", "Text interface", "Q4 / Q8", "2.7 GB Q4", "Yes", "Training repo + data"],
    ["Imajev-4B", "Documented", "0–2 images", "Not established", "4B + 487.6 MB adapter", "Mac MLX / GPU", "Detailed LoRA trainer"],
    ["GLiNER2.5-Decide", "Not established", "Text-only", "Not core format", "340M parameters", "Yes", "Full-model + LoRA"],
    ["Laya (421M / 322M)", "Documented (laya-serve, Unsloth Desktop)", "Text-only", "Not core format", "~808 MB English / ~647 MB multilingual", "Yes; 4–5 GB RAM minimum per Unsloth", "Fine-tuning notebook (free 2×T4)"],
    ["Jev 1.13.0", "Native", "Text-only", "No", "Hosted", "Not local", "Not established"],
  ];
  const NOT_YES = new Set(["No", "Not established", "Not local", "Text-only", "Text interface", "Not core format"]);

  const CAPS = [
    { k: "Safety moderation", big: `<span data-count="81.2">0.0</span><span class="pct">%</span>`, src: "Aegis 2.0 · 250 rows<br>family seen in training" },
    { k: "Intent", two: true, big: `<span data-count="84.6">0.0</span><small>EN</small><span class="sep"></span><span data-count="83.1">0.0</span><small>DE</small>`, src: "MASSIVE en-US / de-DE<br>350 rows each" },
    { k: "Inference", big: `<span data-count="78.3">0.0</span><span class="pct">%</span>`, src: "MultiNLI · 299 rows" },
    { k: "Biomedical QA", big: `<span data-count="77.2">0.0</span><span class="pct">%</span>`, src: "PubMedQA · 250 rows" },
    { k: "Summary consistency", big: `<span data-count="83.3">0.0</span><span class="pct">%</span>`, src: "SummEval · 144 rows" },
    { k: "Toxicity", big: `<span data-count="81.3">0.0</span><span class="pct">%</span>`, src: "Civil Comments · 300 rows" },
    { k: "Paraphrase", big: `<span data-count="81.6">0.0</span><span class="pct">%</span>`, src: "PAWS · 250 rows" },
    { k: "Reads charts · CPU", big: `<span data-count="49" data-dec="0">0</span><span class="of">/60</span>`, src: "synthetic charts · dev set", warn: "charts: Python engine only" },
  ];
  const CAPS_FOOT = `All cards: released GGUF on CPU. Text cards: Nimble public suite subsets (one runner interruption counted wrong). Charts: 60-question synthetic development set used for checkpoint selection; HTTP API is text-only. Aegis 2.0, HelpSteer2 and BoolQ families were seen in training.`;

  // Real vision row: project-owned chart (img/chart-07.png), CPU probabilities from
  // results/runtime/production-vision-verification.json.
  const VISION = [
    { name: "A", v: 99.97, k: "oc" },
    { name: "B", v: 0.01, k: "slate" },
    { name: "C", v: 0.0, k: "slate", txt: "<0.01" },
    { name: "D", v: 0.0, k: "slate", txt: "<0.01" },
    { name: "E", v: 0.01, k: "slate" },
  ];

  // Quickstart commands, verbatim from README.md "Quickstart". The API line in the scene is the real
  // response in results/gguf-sdk-smoke.json, case alone_topic (CPU, 0.62 s, billing 99.93%).
  const QS_DOCKER = [
    `git clone https://github.com/${GH_USER}/openchoice.git`,
    "cd openchoice",
    "docker build -t openchoice:cpu .",
    "docker volume create openchoice-model",
    `docker run --rm --mount type=volume,source=openchoice-model,target=/models openchoice:cpu download ${HF_USER}/OpenChoice-9B`,
    `docker run --rm --entrypoint python openchoice:cpu -c "import secrets; print('OPENCHOICE_API_KEY=' + secrets.token_hex(32))" > openchoice.env`,
    "docker run -d --name openchoice --env-file openchoice.env -p 127.0.0.1:8768:8768 --mount type=volume,source=openchoice-model,target=/models,readonly openchoice:cpu",
    "docker logs -f openchoice",
  ];
  const QS_NATIVE = [
    "bash scripts/build_llamacpp.sh",
    `uv run hf download ${HF_USER}/OpenChoice-9B --local-dir ./OpenChoice-9B-GGUF`,
    "uv run python -m openchoice.server --model ./OpenChoice-9B-GGUF --port 8768",
  ];

  // Fine-tune CLI (FinetuneKit). Update together with FINETUNING.md. Every step also takes FT_DATA.
  // Tutorial evidence: examples/policy-results/final-summary.json (+ Policy-300-GGUF.parity.json).
  const FT = "uv run --group train python -m openchoice.finetune";
  const FT_DATA = "--data examples/support-tickets.jsonl";
  const FT_STEPS = [
    { cmd: "validate", arg: [], d: "Checks every row first" },
    { cmd: "train", arg: ["--init OpenChoice-9B-GGUF", "--output runs/policy", "--steps 300", "…"], d: "LoRA on a 24 GB GPU" },
    { cmd: "eval", arg: ["--before OpenChoice-9B-GGUF", "--after runs/policy/model", "--output runs/policy-eval.json"], d: "Before vs after, held out" },
    { cmd: "export", arg: ["--checkpoint runs/policy/model", "--projector-from OpenChoice-9B-GGUF", "--output Policy-GGUF"], d: "Q4_K_M + projector" },
  ];
  const FT_EXPORT_DIR = "./Policy-GGUF";
  // Held-out policy rows, released GGUF vs fine-tuned GGUF.
  const FT_BEFORE = "28/64";
  const FT_AFTER = "61/64";
  const PKG_FILES = ["manifest.json", "openchoice-Q4_K_M.gguf", "mmproj-q8_0.gguf", "processor/ · model.json", "adapter/ · heads.safetensors"];

  // ---------------------------------------------------------------- math
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const lerp = (a, b, p) => a + (b - a) * p;
  const easeOutCubic = (p) => 1 - Math.pow(1 - p, 3);
  const easeOutQuart = (p) => 1 - Math.pow(1 - p, 4);
  const easeInOutCubic = (p) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);
  const easeOutBack = (p) => {
    const c1 = 1.5;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(p - 1, 3) + c1 * Math.pow(p - 1, 2);
  };
  const prog = (t, start, dur) => clamp((t - start) / dur);

  // ---------------------------------------------------------------- html helpers
  // Count-up spans without their own start time follow their nearest revealed ancestor.
  const inheritCountStarts = (root) => {
    for (const el of root.querySelectorAll("[data-count]:not([data-in])")) {
      const host = el.parentElement.closest("[data-in]");
      el.dataset.in = (host ? parseFloat(host.dataset.in) + 0.15 : 0).toFixed(2);
    }
  };

  const pct = (v, dec = 1) => `<span class="num">${v.toFixed(dec)}</span><span class="pct">%</span>`;

  function hrow(r, at, opts = {}) {
    const dec = opts.dec ?? 1;
    const tag = opts.tags === false ? "" : `<div class="tag">${r.pub ? PUB : MEAS}</div>`;
    const anim = at == null ? "" : ` data-in="${at}" data-fx="left"`;
    const barAnim = at == null ? ` style="width:${r.v}%"` : ` data-bar="${r.v}" data-in="${at + 0.15}"`;
    const valAnim = at == null || r.txt ? "" : ` data-count="${r.v}" data-dec="${dec}" data-in="${at + 0.15}"`;
    return `
      <div class="hrow k-${r.k}${r.pub ? " pub" : ""}"${anim}>
        <div class="lbl"><div class="nm">${r.name}</div>${tag}</div>
        <div class="track"><div class="bar"${barAnim}></div></div>
        <div class="val"${valAnim}>${r.txt ? `<span class="num">${esc(r.txt)}</span><span class="pct">%</span>` : pct(at == null ? r.v : 0, dec)}</div>
      </div>`;
  }

  // `at == null`: glyphs are static (legible from the first frame).
  function wordmark(text, at, step) {
    return `<div class="wordmark" data-wm>${[...text]
      .map((c, i) =>
        at == null
          ? `<span class="ch">${c}</span>`
          : `<span class="ch" data-in="${(at + i * step).toFixed(3)}" data-dur="0.9" data-fx="rise">${c}</span>`)
      .join("")}</div>`;
  }

  // Terminal line: prompt + typed command; the cursor sits on the line while it is current.
  // cur = [typing start, typing end, cursor leaves].
  // Wrapped lines may only break at spaces: intra-word hyphens are shown as non-breaking hyphens.
  function termLine(cmd, at, dur, leave, wrap = false) {
    const shown = wrap ? cmd.replace(/(\w)-(?=\w)/g, "$1\u2011") : cmd;
    return `<div class="tl${wrap ? " wrap" : ""}"><span class="ps" data-in="${at - 0.25}" data-dur="0.3">$</span><span class="cmd" data-type="${esc(shown)}" data-in="${at}" data-dur="${dur}"></span><i class="cursor" data-cur="${at - 0.25},${at + dur},${leave}"></i></div>`;
  }

  const lmax = Math.max(...RT.cpuCurve.map(([, s]) => s));

  // Latency pair: horizontal seconds bars for CPU-only vs CPU + GPU offload.
  function secBars(title, cpu, gpu, at) {
    const max = Math.max(cpu, gpu);
    const row = (nm, v, k, a) => `
      <div class="sbr k-${k}" data-in="${a}" data-fx="left">
        <div class="nm">${nm}</div>
        <div class="track"><div class="bar" data-bar="${((v / max) * 100).toFixed(2)}" data-in="${a + 0.15}"></div></div>
        <div class="val"><span data-count="${v}" data-dec="2" data-in="${a + 0.15}">0.00</span><span class="u">s</span></div>
      </div>`;
    return `
      <div class="lat-group">
        <div class="lg-h" data-in="${at}">${title}</div>
        ${row("CPU only", cpu, "slate", at + 0.1)}
        ${row(`+ ${RT.gpuLayers} GPU layers`, gpu, "oc", at + 0.25)}
      </div>`;
  }

  // ---------------------------------------------------------------- scenes
  const SCENES = [
    {
      // The hook must already be legible on frame 0 (feed thumbnails, muted autoplay).
      id: "s1", start: 0, end: 5.2, hook: true,
      html: `
        <div class="intro">Introducing</div>
        ${wordmark("OpenChoice-9B", null)}
        <div class="subline">A <b>5.8 GiB</b> decision model that runs on a <b>plain CPU</b></div>
        <div class="chips">
          <div class="chip" data-in="0.25" data-fx="pop"><span class="dot"></span>No GPU required</div>
          <div class="chip" data-in="0.4" data-fx="pop"><span class="dot"></span>Optional GPU offload</div>
          <div class="chip" data-in="0.55" data-fx="pop"><span class="dot"></span>Reads charts on CPU</div>
          <div class="chip" data-in="0.7" data-fx="pop"><span class="dot"></span>Fine-tune → GGUF</div>
        </div>
        <div class="basis" data-in="0.9">Qwen3.5-9B + LoRA · GGUF Q4_K_M · Choice, Score and Yes/No probabilities, no text generation</div>`,
    },
    {
      id: "s-cpu", start: 4.7, end: 12.6,
      html: `
        <div class="eyebrow" data-in="0.2">Runs on a plain CPU · measured</div>
        <div class="title sm" data-in="0.3">No GPU required. <span class="muted">Under 7 GiB of RAM.</span></div>
        <div class="cards">
          <div class="card" data-in="0.6">
            <div class="k">Model file</div>
            <div class="big"><span data-count="${RT.weightsGiB}" data-dec="2" data-in="0.8" data-dur="1.2">0.00</span><span class="u">GiB</span></div>
            <div class="d">GGUF Q4_K_M text model<br>+ ${RT.projectorMiB} MiB vision projector</div>
          </div>
          <div class="card" data-in="0.75">
            <div class="k">Peak RAM</div>
            <div class="big"><span data-count="${RT.rssText}" data-dec="1" data-in="0.95" data-dur="1.2">0.0</span><span class="u">GiB</span></div>
            <div class="d">text decisions<br>${RT.rssVision} GiB with vision</div>
          </div>
          <div class="card" data-in="0.9">
            <div class="k">Per decision</div>
            <div class="big range"><span data-count="${RT.lat300}" data-dec="1" data-in="1.1" data-dur="1.2">0.0</span><span class="dash">–</span><span data-count="${RT.lat600}" data-dec="1" data-in="1.1" data-dur="1.2">0.0</span><span class="u">s</span></div>
            <div class="d">300–600-token prompts<br>12 CPU threads</div>
          </div>
          <div class="card" data-in="1.05">
            <div class="k">VRAM</div>
            <div class="big"><span data-count="0" data-dec="0" data-in="1.25" data-dur="1.2">0</span><span class="u">MiB</span></div>
            <div class="d">no GPU process<br>in nvidia-smi</div>
          </div>
        </div>
        <div class="curve" data-in="1.6">
          <div class="ck">CPU latency<br>by prompt length</div>
          <div class="cbars">
            ${RT.cpuCurve.map(([tok, s], i) => `
            <div class="cb">
              <div class="cv"><span data-count="${s}" data-dec="2" data-in="${(1.8 + i * 0.12).toFixed(2)}">0</span> s</div>
              <div class="ct"><i data-meter="${((s / lmax) * 100).toFixed(2)}" data-in="${(1.8 + i * 0.12).toFixed(2)}"></i></div>
              <div class="cl">${tok.toLocaleString("en-US")} tokens</div>
            </div>`).join("")}
          </div>
        </div>
        <div class="footnote" data-in="2.4">GGUF Q4_K_M, llama.cpp CPU repacking off; medians of warmed Choice decisions. Peak RSS includes memory-mapped weights. AMD Ryzen 9 7900X3D desktop, 12 of 24 threads, measured on a live shared desktop, not an isolated benchmark.</div>`,
    },
    {
      id: "s-gpu", start: 12.1, end: 18.6,
      html: `
        <div class="eyebrow" data-in="0.2">Optional GPU offload · measured</div>
        <div class="title sm" data-in="0.3">Borrows your GPU <span class="hl">only when it’s free</span></div>
        <div class="lanes" data-in="0.6">
          <div class="lane">
            <div class="ln">CPU instance</div>
            <div class="lt"><div class="seg cpu" data-meter="100" data-in="0.8" data-dur="1.4"><span>always loaded · serves every request when the GPU is busy</span></div></div>
          </div>
          <div class="lane">
            <div class="ln">GPU offload</div>
            <div class="lt">
              <div class="seg on" style="left:0;width:33%" data-in="1.3" data-fx="left"><span>${RT.gpuLayers} layers · ${RT.gpuVramGiB} GiB VRAM</span></div>
              <div class="seg off" style="left:34%;width:32%" data-in="2.0" data-fx="left"><span>released · CPU serves</span></div>
              <div class="seg on" style="left:67%;width:33%" data-in="2.7" data-fx="left"><span>re-admitted</span></div>
            </div>
          </div>
        </div>
        <div class="evs">
          <div class="ev" style="left:33.5%" data-in="1.9"><i></i><span>game starts</span></div>
          <div class="ev" style="left:66.5%" data-in="2.6"><i></i><span>${RT.quietS} s of free VRAM</span></div>
        </div>
        <div class="lat">
          ${secBars("600-token decision", RT.cpu600, RT.gpu600, 3.0)}
          ${secBars("3,700-token decision", RT.cpu3700, RT.gpu3700, 3.3)}
        </div>
        <div class="footnote" data-in="3.6">With <code>--gpu auto</code>, a GPU instance joins only with at least ${RT.admitGiB} GiB free VRAM (model allowance + ${RT.admitReserveGiB} GiB reserve) and is released below ${RT.releaseGiB} GiB free or when a game starts; the CPU instance never unloads. GPU and CPU decisions agreed on 49 of 50 rows. RTX 4060 8 GB; latencies from separate runs on the same machine.</div>`,
    },
    {
      id: "s-vis", start: 18.1, end: 24.6,
      html: `
        <div class="eyebrow" data-in="0.2">Charts · on CPU · Python engine</div>
        <div class="title sm" data-in="0.3">It reads charts. <span class="muted">On the CPU, too.</span></div>
        <div class="vis">
          <div class="vis-img" data-in="0.6"><img src="img/chart-07.png" width="560" height="420" alt=""></div>
          <div class="vis-r">
            <div class="vq" data-in="0.9"><span class="vk">Choice</span>Which category has the greatest output?</div>
            <div class="hbars">${VISION.map((r, i) => hrow(r, 1.2 + i * 0.1, { tags: false, dec: 2 })).join("")}</div>
          </div>
        </div>
        <div class="vstats" data-in="2.2">
          <span><b>49/60</b>synthetic charts on CPU</span><i></i>
          <span><b>52/60</b>NF4 on GPU</span><i></i>
          <span><b>~2.3 s</b>image encode</span>
          <span class="qual">charts: Python engine · HTTP API is text-only</span>
        </div>
        <div class="footnote" data-in="2.6">Project-owned synthetic chart from the 60-row development set used for checkpoint selection, not a sealed test. GGUF Q4_K_M + Q8_0 projector: Choice 30/30, Score 19/30. Only charts have been evaluated.</div>`,
    },
    {
      id: "s-quick", start: 24.1, end: 32.8,
      html: `
        <div class="eyebrow" data-in="0.2">Quickstart · Docker, Linux/amd64 verified</div>
        <div class="title sm" data-in="0.3">One container to a local API</div>
        <div class="term" data-in="0.6">
          <div class="term-bar"><i></i><i></i><i></i><span>docker · CPU only · no local Python or compiler</span></div>
          <div class="term-body qs">
            ${QS_DOCKER.map((c, i, a) => termLine(c, 0.9 + i * 0.55, 0.45, i === a.length - 1 ? 5.6 : 0.9 + (i + 1) * 0.55 - 0.25, true)).join("")}
            <div class="tc" data-in="5.6"># then POST /v1/systemone → topic: <span class="hit">billing 99.93%</span> · 0.62 s on CPU (real SDK 0.6.0 smoke response)</div>
          </div>
        </div>
        <div class="native" data-in="6.0">
          <div class="nk">Native Linux<br>no Docker</div>
          <div class="nc">${QS_NATIVE.map((c) => `<div><span class="ps">$</span>${esc(c)}</div>`).join("")}</div>
        </div>
        <div class="footnote" data-in="6.4">Linux/amd64 container built and verified here (healthcheck, 34/34 SDK smoke cases, 353 MiB image without weights); linux/arm64 built and smoke-run under QEMU emulation only. The server requires <code>OPENCHOICE_API_KEY</code>. Wire-format compatible only; not affiliated with TypeSafe.</div>`,
    },
    {
      id: "s-tune", start: 32.3, end: 40.6,
      html: `
        <div class="eyebrow" data-in="0.2">Fine-tune it</div>
        <div class="title sm" data-in="0.3">Teach it <span class="hl">your</span> decisions. <span class="muted">Export back to GGUF.</span></div>
        <div class="flow" data-in="0.6">
          <div class="fl-h"><span class="ps">$</span>${FT} <span class="st">&lt;step&gt;</span> ${FT_DATA}</div>
          ${FT_STEPS.map((s, i) => `
          <div class="step${i === FT_STEPS.length - 1 ? " last" : ""}" data-in="${(0.9 + i * 0.3).toFixed(2)}" data-fx="left">
            <div class="sn">0${i + 1}</div>
            <div class="sc">${s.cmd}</div>
            <div class="sa">${s.arg.length ? s.arg.map((a) => `<span class="a">${esc(a)}</span>`).join(" ") : `<span class="none">no other flags</span>`}</div>
            <div class="sd">${s.d}</div>
          </div>`).join("")}
        </div>
        <div class="ft-out">
          <div class="tree" data-in="2.3">
            <div class="tr-h">${FT_EXPORT_DIR}/</div>
            ${PKG_FILES.map((f, i) => `<div class="tr-f" data-in="${(2.5 + i * 0.1).toFixed(2)}">${i === PKG_FILES.length - 1 ? "└─" : "├─"} ${f}</div>`).join("")}
          </div>
          <div class="serve" data-in="3.2">
            <div class="sv-k">then serve it like the release</div>
            <div class="sv-c"><span class="ps">$</span><span data-type="${esc(`uv run python -m openchoice.server --model ${FT_EXPORT_DIR}`)}" data-in="3.5" data-dur="1.2"></span></div>
            <div class="sv-r" data-in="4.4"><b>${FT_BEFORE} → <span class="grad">${FT_AFTER}</span></b><span>held-out policy decisions, released GGUF → fine-tuned GGUF</span></div>
          </div>
        </div>
        <div class="footnote" data-in="3.6">Training needs a CUDA GPU: 300 LoRA updates took 35 min on a rented 24 GB RTX A5000 (18.6 GiB reserved VRAM); serving needs none. Exploratory tutorial, not a benchmark: a self-authored company policy, 192 training and 64 differently worded held-out rows.</div>`,
    },
    {
      id: "s-board", start: 40.1, end: 48.3,
      html: `
        <div class="eyebrow" data-in="0.2">Where it stands · artifacts, not a ranking</div>
        <div class="title sm" data-in="0.3">One package for CPU, charts and fine-tuning. <span class="muted">Not the smallest.</span></div>
        <table class="ftab" data-in="0.6">
          <thead><tr><th>System</th>${FEAT_COLS.map((c) => `<th>${c}</th>`).join("")}</tr></thead>
          <tbody>${FEATURES.map((r, i) => `<tr class="${i === 0 ? "oc" : ""}" data-in="${(0.8 + i * 0.12).toFixed(2)}" data-fx="left"><td>${r[0]}</td>${r.slice(1).map((v) => `<td class="${NOT_YES.has(v) ? "n" : ""}">${v}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>
        <div class="footnote" data-in="2.2">Snapshot 29 Sep 2026 from developer cards (evidence: results/artifact-comparison.json; full table in the README). Official JevBench v1.5.1 score for OpenChoice: pending (sealed split needs a maintainer submission). Not affiliated with TypeSafe or Jev.</div>`,
    },
    {
      id: "s-end", start: 47.8, end: Infinity,
      html: `
        ${wordmark("OpenChoice-9B", 0.2, 0.035)}
        <div class="summary">
          <div class="line" data-in="1.2" data-fx="left"><i class="mk g"></i><span><span class="n grad">5.2 GiB</span> GGUF + 595 MiB projector · plain CPU · optional GPU</span></div>
          <div class="line" data-in="1.55" data-fx="left"><i class="mk g"></i><span>Reads charts on CPU · TypeSafe SDK 0.6-compatible API</span></div>
          <div class="line" data-in="1.9" data-fx="left"><i class="mk g"></i><span>Fine-tune on your decisions → <span class="n grad">same GGUF runtime</span></span></div>
        </div>
        <div class="footnote" data-in="2.5">Built on Qwen3.5-9B. Official JevBench v1.5.1 score pending. Not affiliated with TypeSafe or Jev. Methodology and all measurements in the repo.</div>`,
    },
    {
      // Standalone still only (not on the video timeline).
      id: "s-cap", still: true,
      html: `
        <div class="eyebrow" data-in="0.2">Capabilities · measured locally</div>
        <div class="title" data-in="0.3">What it can do</div>
        <div class="caps">
          ${CAPS.map((c, i) => `
          <div class="cap${c.warn ? " warn" : ""}" data-in="${(0.6 + i * 0.12).toFixed(2)}">
            <div class="k">${c.k}</div>
            <div class="big${c.two ? " two" : ""}" data-in="${(0.8 + i * 0.12).toFixed(2)}">${c.big}</div>
            ${c.warn ? `<div class="qual">${c.warn}</div>` : ""}
            <div class="src">${c.src}</div>
          </div>`).join("")}
        </div>
        <div class="hard" data-in="2.2">
          <span class="hk">Still hard</span>
          <span>summary relevance <b>40.4%</b></span><i></i>
          <span>exact 5-level helpfulness <b>41.4%</b></span><i></i>
          <span>fact verification <b>75.3%</b></span>
        </div>
        <div class="footnote" data-in="2.6">${CAPS_FOOT}</div>`,
    },
  ];
  const TIMELINE = SCENES.filter((s) => !s.still);

  // ---------------------------------------------------------------- social card (1200x630)
  const socialHtml = () => `
    <div class="so-top"><span class="dot"></span>Qwen3.5-9B + LoRA · GGUF Q4_K_M · llama.cpp</div>
    <div class="wordmark grad">OpenChoice-9B</div>
    <div class="so-sub">A <b>5.8 GiB</b> decision model.<br>Runs on a <b>plain CPU</b>.</div>
    <div class="so-chips">
      <span>No GPU required</span><span>Optional GPU offload</span><span>Charts on CPU</span><span>Fine-tune → GGUF</span>
    </div>
    <div class="so-foot"><code>POST /v1/systemone</code><span>TypeSafe SDK 0.6-compatible · ${RT.lat300}–${RT.lat600} s per decision on CPU</span></div>`;

  // ---------------------------------------------------------------- build
  const scenesEl = document.getElementById("scenes");
  scenesEl.innerHTML = SCENES.map((s) => `<section class="scene" id="${s.id}">${s.html}</section>`).join("");
  document.getElementById("social").innerHTML = socialHtml();

  const brand = document.getElementById("brand");
  const glows = ["g1", "g2", "g3"].map((c) => document.querySelector(`.glow.${c}`));
  const grid = document.querySelector(".grid");

  inheritCountStarts(scenesEl);
  for (const s of SCENES) {
    s.el = document.getElementById(s.id);
    s.reveals = [...s.el.querySelectorAll("[data-in]")].map((el) => ({
      el,
      at: parseFloat(el.dataset.in),
      dur: parseFloat(el.dataset.dur || "0.7"),
      fx: el.dataset.fx || "up",
      bar: el.dataset.bar != null ? parseFloat(el.dataset.bar) : null,
      vbar: el.dataset.vbar != null ? parseFloat(el.dataset.vbar) : null,
      meter: el.dataset.meter != null ? parseFloat(el.dataset.meter) : null,
      count: el.dataset.count != null ? parseFloat(el.dataset.count) : null,
      dec: el.dataset.dec != null ? parseInt(el.dataset.dec, 10) : 1,
      num: el.querySelector(".num") || el,
      type: el.dataset.type ?? null,
      follow: el.hasAttribute("data-follow"),
    }));
    s.wordmarks = [...s.el.querySelectorAll("[data-wm]")];
    s.cursors = [...s.el.querySelectorAll("[data-cur]")].map((el) => ({ el, cur: el.dataset.cur.split(",").map(Number) }));
  }

  // Wordmark: each glyph animates independently but shares one gradient spanning the word.
  function layoutWordmarks() {
    for (const s of SCENES) {
      for (const wm of s.wordmarks) {
        const w = wm.scrollWidth;
        for (const ch of wm.children) {
          ch._x = ch.offsetLeft - wm.offsetLeft;
          ch._w = w;
        }
      }
    }
  }

  // ---------------------------------------------------------------- per-frame
  function applyReveal(r, lt) {
    const p = prog(lt, r.at, r.dur);
    const el = r.el;

    if (r.bar != null || r.vbar != null || r.meter != null || r.count != null || r.type != null) {
      const e = easeOutQuart(prog(lt, r.at, r.dur === 0.7 ? 1.5 : r.dur));
      if (r.bar != null) el.style.width = `${r.bar * e}%`;
      if (r.meter != null) el.style.width = `${r.meter * e}%`;
      if (r.vbar != null) el.style.height = `${r.vbar * e}%`;
      if (r.count != null) {
        r.num.textContent = (r.count * e).toFixed(r.dec);
        if (r.follow) {
          el.style.setProperty("--h", `${r.count * e}%`);
          el.style.opacity = clamp(e * 3);
        }
      }
      if (r.type != null) {
        const n = Math.round(r.type.length * prog(lt, r.at, r.dur));
        el.textContent = r.type.slice(0, n);
      }
      return;
    }

    let e;
    let tf = "";
    let blur = 0;
    switch (r.fx) {
      case "rise":
        e = easeOutQuart(p);
        tf = `translateY(${lerp(90, 0, e)}px)`;
        blur = lerp(16, 0, e);
        break;
      case "left":
        e = easeOutCubic(p);
        tf = `translateX(${lerp(-40, 0, e)}px)`;
        blur = lerp(6, 0, e);
        break;
      case "pop":
        e = easeOutBack(p);
        tf = `scale(${lerp(0.82, 1, e)})`;
        blur = lerp(6, 0, easeOutCubic(p));
        break;
      default:
        e = easeOutCubic(p);
        tf = `translateY(${lerp(28, 0, e)}px)`;
        blur = lerp(8, 0, e);
    }
    el.style.opacity = clamp(easeOutCubic(p) * 1.15);
    el.style.transform = p >= 1 ? "" : tf;
    el.style.filter = blur > 0.05 ? `blur(${blur.toFixed(2)}px)` : "";
  }

  function renderWordmarks(s, lt) {
    for (const wm of s.wordmarks) {
      // Static hook wordmark: a slight scale settle keeps frame 0 alive without hiding it.
      if (s.hook) wm.style.transform = `scale(${lerp(1.03, 1, easeOutCubic(prog(lt, 0, 2.4))).toFixed(4)})`;
      // A soft highlight sweeps across the word once the glyphs have landed.
      const sweep = lerp(-0.35, 1.35, easeInOutCubic(prog(lt, 1.1, 1.4)));
      for (const ch of wm.children) {
        const W = ch._w;
        const shineX = sweep * W - ch._x - W * 0.2;
        ch.style.backgroundImage =
          "linear-gradient(100deg, transparent 0%, rgba(255,255,255,0.75) 50%, transparent 100%), " +
          "linear-gradient(90deg, #c6ff3d 0%, #6ef7a4 48%, #2ee6e6 100%)";
        ch.style.backgroundSize = `${W * 0.4}px 100%, ${W}px 100%`;
        ch.style.backgroundRepeat = "no-repeat";
        ch.style.backgroundPosition = `${shineX}px 0, ${-ch._x}px 0`;
      }
    }
  }

  function renderBackground(t, W = 1920, H = 1080) {
    const pos = [
      [W * 0.22 + 180 * Math.sin(t * 0.19), H * 0.28 + 110 * Math.cos(t * 0.23)],
      [W * 0.82 + 160 * Math.cos(t * 0.17 + 1), H * 0.78 + 120 * Math.sin(t * 0.21 + 0.5)],
      [W * 0.55 + 220 * Math.sin(t * 0.11 + 2), H * 1.02 + 90 * Math.cos(t * 0.13)],
    ];
    const breathe = [0.85 + 0.15 * Math.sin(t * 0.5), 0.85 + 0.15 * Math.cos(t * 0.43), 1];
    glows.forEach((g, i) => {
      g.style.transform = `translate(${pos[i][0].toFixed(1)}px, ${pos[i][1].toFixed(1)}px)`;
      g.style.opacity = breathe[i].toFixed(3);
    });
    grid.style.transform = `translate(${((t * 10) % 80).toFixed(2)}px, ${((t * 6) % 80).toFixed(2)}px)`;
  }

  function hideScene(s) {
    s.el.style.visibility = "hidden";
  }

  // `settled`: render the scene fully revealed, without entry/exit motion (stills).
  function renderScene(s, t, settled = false) {
    const lt = settled ? 30 : t - s.start;
    const inP = settled || s.hook ? 1 : easeOutCubic(prog(lt, 0, 0.8));
    const outP = settled || s.end === Infinity ? 0 : easeInOutCubic(prog(t, s.end - 0.6, 0.6));
    const vis = inP * (1 - outP);
    if (vis <= 0.001) {
      hideScene(s);
      return;
    }
    s.el.style.visibility = "visible";
    s.el.style.opacity = vis.toFixed(4);
    const y = lerp(36, 0, inP) - 28 * outP;
    const sc = lerp(0.985, 1, inP) + 0.015 * outP;
    s.el.style.transform = settled ? "" : `translateY(${y.toFixed(2)}px) scale(${sc.toFixed(4)})`;
    const blur = 10 * (1 - inP) + 12 * outP;
    s.el.style.filter = blur > 0.05 ? `blur(${blur.toFixed(2)}px)` : "";

    for (const r of s.reveals) applyReveal(r, lt);
    renderWordmarks(s, settled ? 0 : lt);

    for (const { el, cur: [from, typed, leave] } of s.cursors) {
      const on = lt >= from && lt < leave;
      const blink = lt < typed ? 1 : Math.floor((lt - typed) / 0.5) % 2 === 0 ? 0 : 1;
      el.style.opacity = settled ? (leave > 30 ? 1 : 0) : on ? blink : 0;
    }
  }

  function renderFrame(t) {
    renderBackground(t);
    const b = easeOutCubic(prog(t, 4.8, 0.6)) * (1 - easeInOutCubic(prog(t, 47.5, 0.6)));
    brand.style.opacity = b.toFixed(3);
    for (const s of TIMELINE) renderScene(s, t);
    for (const s of SCENES) if (s.still) hideScene(s);
  }

  function renderStill(id) {
    renderBackground(7.5);
    brand.style.opacity = 1;
    for (const s of SCENES) {
      if (s.id === id) renderScene(s, 0, true);
      else hideScene(s);
    }
  }

  function renderSocial() {
    document.body.classList.add("social");
    brand.style.opacity = 0;
    renderBackground(7.5, 1200, 630);
  }

  window.DURATION = DURATION;
  // Standalone stills: asset name -> scene id (rendered settled).
  window.STILLS = { capabilities: "s-cap", cpu: "s-cpu", finetune: "s-tune", scoreboard: "s-board" };
  window.renderFrame = renderFrame;
  window.renderStill = renderStill;
  window.renderSocial = renderSocial;
  const fontLoads = [...document.fonts].map((f) => f.load());
  const images = [...document.images].map((img) => img.decode().catch(() => {}));
  window.ready = Promise.all([...fontLoads, ...images]).then(() => document.fonts.ready).then(() => {
    layoutWordmarks();
    if (q.has("social")) renderSocial();
    else if (q.has("still")) renderStill(window.STILLS[q.get("still")]);
    else renderFrame(parseFloat(q.get("t") || "0"));
    return true;
  });
})();
