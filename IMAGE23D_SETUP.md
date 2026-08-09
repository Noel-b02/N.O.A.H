# Image-to-3D setup (one-time)

Lets Noah turn a described real-world subject (e.g. "a model of a fox") into
an actual mesh by finding a reference image and running it through
[TripoSR](https://github.com/VAST-AI-Research/TripoSR) (Stability AI / Tripo,
MIT license, free, runs fully locally). This is separate from the parametric
Fusion 360 code-generation pipeline (`FUSION_SETUP.md`) — that one is for
simple shapes described in words (boxes, cylinders); this one is for
recognizable objects/characters found via a reference photo.

The repo is already vendored at `image23d/TripoSR/`. It needs its own venv,
kept separate from `speech/`'s venv — the two pull in different, sometimes
conflicting native-dependency versions of `torch`.

## 1. Prerequisites (Windows)

`torchmcubes` (one of TripoSR's dependencies) compiles a native extension at
install time, so you need a C++ build toolchain first:

- Install **Visual Studio Build Tools** (the "Desktop development with C++"
  workload) if you don't already have it: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

## 2. Create the venv and install dependencies

```bash
cd image23d/TripoSR
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip setuptools
```

Install PyTorch with CUDA support first (same reasoning as the speech
service — the plain `torch` PyPI wheel is CPU-only). Match this to your
driver's CUDA support, same as `SPEECH_SETUP.md`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Then the rest of TripoSR's dependencies:

```bash
pip install -r requirements.txt
```

## 3. First run downloads the model weights

The first generation call downloads `stabilityai/TripoSR` from Hugging Face
(a few hundred MB) and caches it under your user profile — no separate
download step needed, but the first request after setup will be slower.

## 4. Verify it works standalone

```bash
venv\Scripts\python run.py examples/chair.png --output-dir output/
```

Should produce `output/0/mesh.obj` in a few seconds on the RTX 3060 Ti (this
model needs ~6GB VRAM, so close any other GPU-heavy app first while testing
standalone).

## How Noah uses it

`server.ts` spawns `run.py` as a one-off child process per request (not a
persistent service) — this keeps VRAM free for Ollama/Kokoro the rest of the
time, at the cost of a few seconds of model-loading latency per generation.
See `FUSION_SETUP.md` for the full request pipeline.

## Known limitations — read before relying on this

- **Whichever image SearXNG ranks first is what gets used**, with no human
  or model review in between — for an ambiguous or unusual subject, that can
  be the wrong picture entirely. If the result looks nothing like what you
  asked for, that's most likely why.
- **The output mesh has no real-world scale or unit information.** TripoSR
  normalizes geometry into its own coordinate space, so the imported model
  can come in absurdly tiny or huge — use Fusion's Scale tool after import.
- **Geometry and texture fidelity is rough** compared to a hand-modeled
  asset or a paid image-to-3D service — TripoSR trades quality for speed and
  being free/local. Fine detail, thin features, and accurate proportions on
  complex subjects (a specific character's face, for example) are not
  reliable.
- **VRAM is shared with Ollama and Kokoro on an 8GB card.** If a generation
  fails with a CUDA out-of-memory error, close/pause other GPU work
  (Ollama's model can be unloaded with `ollama stop <model>`) and retry.
- **No texture/color import is attempted.** `run.py` runs without
  `--bake-texture` for speed, so only the mesh geometry — no surface color —
  comes through into Fusion.
