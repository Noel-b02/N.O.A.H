# Getting started

A linear walkthrough from "just cloned this" to "chatting with Noah." Each
subsystem has its own detailed setup doc (linked below) — this page is the
order to do them in, plus the one step none of them cover: actually
installing Ollama and pulling the models Noah expects.

## What's required vs optional

Only **Ollama** is required to get a working chat assistant. Everything
else adds a capability on top:

| Subsystem | Required? | Adds |
|---|---|---|
| Ollama | **Yes** | Chat itself — nothing works without this |
| SearXNG | No | Web search, news, recall of past conversations |
| Speech service | No | Voice input/output |
| Fusion 360 bridge | No | CAD generation (parametric + image-to-3D) |
| Hunyuan3D-2 | No | The image-to-3D part of the above specifically |

If you just want to try the chat assistant, you can skip straight to
[Run it](#5-run-it) after step 2 below.

## 1. Prerequisites

- **Node.js** 20+ (built and tested on 24)
- **Python** 3.10+ (each subsystem below gets its own separate venv)
- **[Ollama](https://ollama.com/download)** — install it, then confirm it
  works: `ollama --version`
- An **NVIDIA GPU** if you want any of the optional subsystems at a usable
  speed — CPU fallback exists for some pieces but is much slower. See
  "Hardware notes" in `README.md` for what's actually needed for what.
- **Docker Desktop**, only if you want SearXNG (web search)

## 2. Clone and install

```bash
git clone https://github.com/Noel-b02/N.O.A.H.git
cd N.O.A.H
npm install
cp .env.example .env
```

`.env`'s defaults work as-is for a local single-machine setup — you only
need to edit it if you're changing ports or running a subsystem on a
different machine.

## 3. Pull the Ollama models

Noah expects two specific models by default (configurable via
`OLLAMA_CHAT_MODEL`/`OLLAMA_CODE_MODEL` in `.env` if you want different
ones):

```bash
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
```

The first is used for everyday chat, the second for more complex requests,
Fusion 360 code generation, and — if you set up Hunyuan3D-2 — verifying
image-to-3D reference photos. Both need to support vision input for the
image-related features to work; substitute accordingly if you use different
models.

## 4. Optional subsystems

Set up whichever of these you want, in any order:

- [`SEARXNG_SETUP.md`](SEARXNG_SETUP.md) — web search, news, recall
- [`SPEECH_SETUP.md`](SPEECH_SETUP.md) — voice input/output
- [`FUSION_SETUP.md`](FUSION_SETUP.md) — CAD generation in Fusion 360
  - [`HUNYUAN3D_SETUP.md`](HUNYUAN3D_SETUP.md) — needed on top of the above
    specifically for "make me a model of X" style requests

## 5. Run it

```bash
npm run dev
```

This automatically starts Ollama and SearXNG if they aren't already
running (see `scripts/ensure-ollama.ts`/`ensure-searxng.ts`), then starts
the server at `http://localhost:3000`. Open that in a browser and send a
message — if Ollama's up and the models are pulled, that's a fully working
chat assistant already, independent of whichever optional pieces you did
or didn't set up.

`npm run dev` uses `tsx watch` and auto-reloads on file changes, so it's
what you want for actually working on the project. `npm run build` then
`npm start` runs the compiled version instead.

## Troubleshooting

- **"fetch failed" errors in chat** — almost always Ollama not running or
  the model names in `.env` not matching what you pulled. Check
  `ollama list` against `OLLAMA_CHAT_MODEL`/`OLLAMA_CODE_MODEL`.
- **Ollama's tray app doesn't auto-start after a reboot** — this is a known
  bug in Ollama's own tray app (not this project), see the comment in
  `scripts/ensure-ollama.ts`. `npm run dev` already works around it by
  launching `ollama serve` directly.
- **First image-to-3D request takes forever** — first use of any local
  generative model downloads its weights from Hugging Face (several GB) —
  this is a one-time cost per model, see `HUNYUAN3D_SETUP.md`.
- **CUDA out-of-memory errors** — see "Hardware notes" in `README.md`.
  This project is built and tuned for 8GB VRAM (see the `noah-8gb-edition`
  tag) — some things genuinely need more, and are labeled as such in their
  setup docs.
