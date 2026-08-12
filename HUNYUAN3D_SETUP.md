# Hunyuan3D-2 setup (one-time)

Lets Noah turn a described real-world subject (e.g. "a model of a fox") into
an actual mesh by finding a reference image and running it through
[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) (Tencent, real
generative model with texture-capable output). This is separate from the
parametric Fusion 360 code-generation pipeline (`FUSION_SETUP.md`) — that
one is for simple shapes described in words (boxes, cylinders); this one is
for recognizable objects/characters found via a reference photo.

The repo is already vendored at `image23d/Hunyuan3D-2/`. It needs its own
venv, kept separate from the speech service's — the two pull in different,
sometimes conflicting native-dependency versions of `torch`.

This covers the **shape-only** path (what `run_shape_only.py` and
`run_multiview.py` actually use today — no texture/color, just geometry).
The full texture+paint pipeline needs a separately-compiled CUDA rasterizer
and ~16GB VRAM; see `IMAGE_GEN_ROADMAP.md` for that.

## 1. Create the venv and install dependencies

```bash
cd image23d/Hunyuan3D-2
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip setuptools
```

Install PyTorch with CUDA support first (same reasoning as `SPEECH_SETUP.md`
— the plain `torch` PyPI wheel is CPU-only). Match this to your driver's
CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Then the rest of the dependencies (`requirements.txt` has `torch`/`torchvision`
commented out on purpose — see the comment there):

```bash
pip install -r requirements.txt
pip install -e .
```

The `pip install -e .` is what makes the `hy3dgen` package importable —
without it, `run_shape_only.py`/`run_multiview.py` fail on `import hy3dgen`.

## 2. First run downloads the model weights

The first shape-only generation call downloads `tencent/Hunyuan3D-2` from
Hugging Face — several GB, took roughly 30 minutes on a home connection when
this was set up. The multiview path (`run_multiview.py`) downloads a
*separate* multi-GB checkpoint from `tencent/Hunyuan3D-2mv` the first time
it's used, on top of that.

Hugging Face's newer "Xet" transfer backend has a well-documented stalling
bug on large files (confirmed directly — a multi-GB weight file hung
indefinitely at 0 bytes). `server.ts` already sets `HF_HUB_DISABLE_XET=1`
when launching these scripts to avoid it; if you ever run them standalone
outside of Noah, set that env var yourself too.

## 3. Verify it works standalone

```bash
venv\Scripts\python run_shape_only.py <path-to-an-image.jpg> output_test.obj
```

Should produce `output_test.obj`. Expect several minutes on the first run
(weight download + model load), much faster on subsequent runs once weights
are cached.

## How Noah uses it

`server.ts` spawns `run_shape_only.py` (single reference image) or
`run_multiview.py` (front/side/back, experimental — see
`IMAGE_GEN_ROADMAP.md`) as one-off child processes per request rather than as
a persistent service — this is a rarely-used, GPU-heavy step, and keeping it
out-of-process means it doesn't hold VRAM hostage from Ollama/Kokoro the
rest of the time. Runs under `withGpuExclusive` — see that function's
comment in `server.ts` for why Ollama gets paused/unloaded while this runs
on an 8GB card.

## Known limitations — read before relying on this

- **Whichever image SearXNG ranks first is what gets used** (single-image
  path) or **passes vision verification** (multiview path), with no human
  review in between — for an ambiguous or unusual subject, that can be the
  wrong picture entirely. If the result looks nothing like what you asked
  for, that's most likely why.
- **The output mesh has no real-world scale or unit information.** The
  model normalizes geometry into its own coordinate space, so the imported
  model can come in absurdly tiny or huge — use Fusion's Scale tool (or ask
  Noah for a specific size) after import.
- **Geometry and texture fidelity is rough** compared to a hand-modeled
  asset or a paid image-to-3D service — this trades quality for being
  free/local. Fine detail, thin features, and accurate proportions on
  complex subjects (a specific character's face, for example) are not
  reliable.
- **Shape-only means no color/texture** — the output is bare geometry. Fine
  for printing, not for anything needing surface color. See
  `IMAGE_GEN_ROADMAP.md` for the plan to change that.
- **Meaningfully slower than a single-pass reconstruction model** — this is
  a diffusion model doing iterative denoising, so expect noticeably longer
  generation time even once weights are cached.
- **VRAM is shared with Ollama and the speech service on an 8GB card.** If a
  generation fails with a CUDA out-of-memory error, close/pause other GPU
  work (Ollama's model can be unloaded with `ollama stop <model>`) and
  retry — `withGpuExclusive` handles this automatically for requests that
  go through Noah, but manual troubleshooting may still need it.
