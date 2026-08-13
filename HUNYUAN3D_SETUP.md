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
and ~16GB VRAM — not yet planned in detail, tracked as future work (see
"Known limitations" below; `IMAGE_GEN_ROADMAP.md` covers the other
16GB-unlocked features — novel-view synthesis and pose-guided generation —
not this one).

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

## 4. Novel-view synthesis (multiview reference generation)

`image23d/generate_novel_views.py` generates the "left" and "back" views
for the multiview pipeline from one seed photo, via
[Zero123++ v1.2](https://huggingface.co/sudo-ai/zero123plus-v1.2) — see
`IMAGE_GEN_ROADMAP.md` for the full design. No separate venv: it reuses
this one (diffusers/torch already installed).

First run downloads `sudo-ai/zero123plus-v1.2`'s weights plus the
`sudo-ai/zero123plus-pipeline` community pipeline code from Hugging Face —
loading it executes that code (`trust_remote_code=True`), same trust
category as any other model downloaded here, but worth knowing it's not
just weights. Verify it works standalone the same way as step 3:

```bash
venv\Scripts\python ..\generate_novel_views.py <path-to-an-image.jpg> left_test.jpg back_test.jpg
```

Should produce two output images. Inspect them by eye — a correct result
looks like plausible rotated views of the same subject, same art style and
pose family as the input.

## 5. Pose-guided seed generation (dynamic-pose fix)

`image23d/generate_pose_seed.py` replaces a search-sourced photo caught
mid-action (limbs overlapping/occluded/foreshortened — passes subject-match
verification but reliably wrecks the downstream mesh) with a cleanly
generated neutral-pose image, when the request reads as a dynamic/action
pose. See `IMAGE_GEN_ROADMAP.md` for the full design and why SDXL +
ControlNet-openpose was chosen over the originally-planned Z-Image-Turbo.
No separate venv — reuses this one (diffusers 0.39.0 already installed here
has everything needed).

First run downloads `stabilityai/stable-diffusion-xl-base-1.0` (~7GB) and
`thibaud/controlnet-openpose-sdxl-1.0` (~2.4GB, an older `.bin`-format
checkpoint — diffusers falls back to it automatically with a benign
warning, no `.safetensors` file exists in that repo). Verify it works
standalone:

```bash
venv\Scripts\python ..\generate_pose_seed.py "spiderman, red and blue spandex suit with black web pattern, standing in a neutral pose, isolated on a plain white background" pose_test.jpg
```

Should produce one output image: a full-body figure in a clean, neutral
standing pose matching `image23d/assets/canonical_pose.png`'s skeleton.
Confirmed live: ~8GB peak VRAM, 16-25s per generation once weights are
cached, with `enable_model_cpu_offload()` (already set up in the script).

### One-time canonical pose asset

`image23d/assets/canonical_pose.png` (committed to the repo) is a static
OpenPose-style skeleton image, used as fixed ControlNet conditioning for
every pose-guided generation — never extracted live from a search photo,
since that would reintroduce the exact pose-reliability problem this
feature exists to remove. It was made once via
`image23d/tools/make_canonical_pose_asset.py`, which:

1. Generates a source photo of a person in a clean, neutral, front-facing
   standing pose using plain SDXL text-to-image (no ControlNet) — avoids
   any licensing question around a sourced stock photo, and is fully
   reproducible from the script alone.
2. Runs [`rtmlib`](https://github.com/Tramac/rtmlib)'s whole-body pose
   estimator (RTMPose/DWPose-based, no `mmcv`/`mmdet`/`mmpose` dependency
   chain — just `numpy`/`opencv`/`onnxruntime`) against that photo and
   renders an OpenPose-style colored skeleton on a blank canvas.

`rtmlib` is **not** a dependency of `generate_pose_seed.py` or any other
live-runtime script — only needed if the canonical pose ever needs to
change. If you do need to regenerate it: `pip install rtmlib` into this
venv (or a disposable one) and run
`venv\Scripts\python tools\make_canonical_pose_asset.py` from
`image23d/`.

## How Noah uses it

`server.ts` spawns `run_shape_only.py` (single reference image),
`generate_pose_seed.py` (only when a dynamic pose is detected — see
`IMAGE_GEN_ROADMAP.md`), `generate_novel_views.py` + `run_multiview.py`
(single seed image, two angle views generated from it), as one-off child
processes per request rather than as a persistent service — this is a
rarely-used, GPU-heavy step, and keeping it out-of-process means it doesn't
hold VRAM hostage from Ollama/Kokoro the rest of the time. Runs under
`withGpuExclusive` — see that function's comment in `server.ts` for why
Ollama/speech get paused while these run.

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
  for printing, not for anything needing surface color. The full
  texture+paint pipeline is planned next but not yet designed in detail
  (see the note at the top of this file).
- **Meaningfully slower than a single-pass reconstruction model** — this is
  a diffusion model doing iterative denoising, so expect noticeably longer
  generation time even once weights are cached.
- **VRAM is shared with Ollama and the speech service.** If a generation
  fails with a CUDA out-of-memory error, close/pause other GPU work
  (Ollama's model can be unloaded with `ollama stop <model>`) and retry —
  `withGpuExclusive` handles this automatically for requests that go
  through Noah, but manual troubleshooting may still need it.
- **Pose-guided generation only helps humanoid subjects.** The fixed
  canonical pose skeleton is a human pose — a non-humanoid dynamic subject
  ("a dragon flying") isn't helped by this feature, and can still produce
  the original detached-limb failure. Attached images and pasted URLs are
  never pose-corrected either (they're the user's own real object) — a
  user-supplied photo in a dynamic pose can still hit the same failure
  mode this feature otherwise fixes for search-sourced photos.
