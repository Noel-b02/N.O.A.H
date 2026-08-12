# N.O.A.H.

A personal, local-first AI assistant. Runs entirely on your own hardware —
local LLM (via [Ollama](https://ollama.com)), local web search (via
[SearXNG](https://github.com/searxng/searxng)), local speech-to-text and
text-to-speech (Whisper + Kokoro), and a CAD bridge into Autodesk Fusion 360.
Nothing is sent to a third-party API by default.

## What it does

- **Chat**, with a configurable personality (`personality.txt`) and
  persistent conversation memory/recall across sessions.
- **Web search**, via a self-hosted SearXNG instance — no external search
  API or key required.
- **Voice**: speech-to-text (Whisper) for input, text-to-speech (Kokoro) for
  spoken replies.
- **Fusion 360 integration**, two ways:
  - *Parametric*: describe simple geometry in words ("a 40mm cube with a
    10mm hole through it") and Noah generates and executes Fusion API code
    directly in your open document.
  - *Image-to-3D*: ask for a model of a real subject ("a model of a fox")
    and Noah finds or generates reference photos, runs them through a local
    image-to-3D model ([Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)),
    and imports the resulting mesh — including an experimental multiview mode
    that sources front/side/back reference photos for better geometry. See
    `IMAGE_GEN_ROADMAP.md` for where this is headed next.
- **Self-modification**: Noah can propose changes to its own source code,
  staged as a draft git branch/commit for you to review and approve or
  discard before anything touches `master`.

## Architecture

A single Express server (`server.ts`) fronts everything: it routes chat
requests, calls Ollama for generation, proxies SearXNG for search, talks to
the local speech service over HTTP, and dispatches Fusion 360 requests to a
bridge add-in running inside Fusion itself. The frontend (`public/`) is a
single HTML/JS page — no build step, no framework.

```
public/            chat UI (static HTML/JS)
server.ts          main server — routing, Ollama calls, all the pipelines
speech/            local Whisper (STT) + Kokoro (TTS) service
fusion-bridge/      Fusion 360 add-in (NoahFusionBridge) — executes CAD code
image23d/          Hunyuan3D-2, vendored, for image-to-3D generation
searxng/           SearXNG config for the local search container
scripts/           startup helpers (auto-launch Ollama/SearXNG if not running)
memories/          conversation history/recall (gitignored — personal data)
```

## Setup

Requires Node.js, Python, Docker (for SearXNG), and an NVIDIA GPU (for
Ollama/Whisper/image-to-3D — CPU fallback exists for some pieces but is much
slower). Each subsystem has its own one-time setup doc:

1. [`SPEECH_SETUP.md`](SPEECH_SETUP.md) — Whisper + Kokoro venv
2. [`SEARXNG_SETUP.md`](SEARXNG_SETUP.md) — local search container
3. [`FUSION_SETUP.md`](FUSION_SETUP.md) — the Fusion 360 add-in
4. [`HUNYUAN3D_SETUP.md`](HUNYUAN3D_SETUP.md) — the Hunyuan3D-2 venv

Then:

```bash
npm install
npm run dev
```

`predev` automatically starts Ollama and SearXNG if they aren't already
running. The server comes up at `http://localhost:3000`.

`npm run dev` uses `tsx watch` and auto-reloads on file changes. `npm start`
runs the compiled build (`npm run build` first) instead.

## Hardware notes

Image-to-3D generation and the full Ollama/Whisper stack are VRAM-hungry.
An 8GB card runs the core assistant and shape-only mesh generation fine;
Hunyuan3D's full texture+paint pipeline wants closer to 16GB. GPU-exclusive
locking keeps generation tasks from fighting Ollama for VRAM on smaller
cards — see the comments around `withGpuExclusive` in `server.ts`.
