# Image-generation pipeline — single-seed novel-view synthesis (implemented)

Noah's multiview image-to-3D pipeline used to source front/side/back
reference photos from three independent web searches. Even with subject,
framing, and cross-image consistency verification, this was fundamentally
gambling that three unrelated photos happened to agree on pose, art style,
and appearance. Confirmed directly: a "spiderman swinging" request pulled
three photos in different poses, and Hunyuan3D-2mv — which assumes all
three inputs are one pose from three camera angles — produced a mesh with
two extra detached legs trying to reconcile the mismatch. Verification
caught *some* of this after the fact; it couldn't fix a scarcity problem
where no consistent set existed in the search results at all.

This was blocked on a GPU upgrade (the 3060 Ti's 8GB couldn't fit a local
novel-view model alongside Ollama/Hunyuan3D) — that upgrade (RTX 5060 Ti,
16GB) has happened, and the pipeline below is built and live-verified,
including a re-run of the exact "spiderman swinging" prompt that originally
caused the detached-legs failure (confirmed fixed: clean, coherent geometry,
no stray limbs).

## Pipeline

```
one reference image → vision verification → generate side/back views
(Zero123++) → Hunyuan3D-2mv (multiview) → mesh → slicer
```

1. **Reference image**: attached image, pasted URL, or the single best web
   search result (`"${subject} isolated on white background"`, falling back
   to a plain `subject` search).
2. **Vision verification**: `verifyImageMatchesSubject` in `server.ts` now
   runs on every search-sourced candidate (previously it only ran inside
   the old three-photo multiview attempt — a search-sourced single image
   could reach mesh generation completely unchecked, which is fixed now
   too). Attached images and pasted URLs skip verification, unchanged —
   it exists to filter untrustworthy search results, not to second-guess
   an image the user explicitly chose.
3. **Generate side/back views**: `image23d/generate_novel_views.py` runs
   the seed through [Zero123++ v1.2](https://huggingface.co/sudo-ai/zero123plus-v1.2)
   (`sudo-ai/zero123plus-v1.2`, community pipeline
   `sudo-ai/zero123plus-pipeline`), which generates 6 fixed-angle views in
   one diffusion pass — consistency across views is enforced by
   construction (one model, one pass), not checked after the fact. Output
   is a single 640×960 image: a 2-column × 3-row grid of 320×320 tiles.
   Confirmed empirically (not documented anywhere) that reading order
   top-left → top-right → mid-left → mid-right → bottom-left → bottom-right
   corresponds exactly to azimuths 30°/90°/150°/210°/270°/330° (relative to
   the seed as 0°/front). 90° is an exact match for "left"; there's no
   exact 180° for "back" — the script currently picks 210° (mid-right),
   30° off, as a first-pass choice. Both indices are named constants at the
   top of `generate_novel_views.py` — retune there alone if Hunyuan3D-2mv
   output looks better with 150° instead.

   Reuses the existing `image23d/Hunyuan3D-2/venv/` rather than a separate
   one (diffusers/torch already installed and compatible). Loading the
   community pipeline executes code from the `sudo-ai/zero123plus-pipeline`
   Hub repo at runtime (`trust_remote_code=True`) — same trust category as
   any other model this project downloads from Hugging Face, but worth
   knowing it's not just weights.
4. **Hunyuan3D-2mv**, unchanged — already expected exactly this
   `{front, left, back}` input shape; `generateMeshWithHunyuan3DMultiview`
   in `server.ts` didn't need to change at all.
5. **Slicer** (Bambu Studio) — `sliceModel()` in `server.ts` drives Bambu
   Studio's own CLI directly, chained after mesh repair and Fusion import;
   see `PRINTER_SETUP.md`.

`wantsSingleImageOnly` (opt-out phrasing like "one image only" / "no
multiview") skips steps 3-4 entirely and goes straight to shape-only
generation from the seed, unchanged from before. If novel-view generation
itself fails (script crash, timeout), the pipeline falls back to shape-only
rather than failing the request outright — same "still produce something
usable" reasoning the old search-based multiview attempt used.

## GPU usage

`withGpuExclusive` (see `server.ts`) is kept, not removed, despite the
16GB upgrade — this pipeline *adds* a new ~5GB generation step
(Zero123++) on top of the existing ~6GB one (Hunyuan3D-2 shape stage), and
Ollama's two resident models plus Whisper/Kokoro plus one 5-6GB generation
step is still a real way to approach 16GB. The two new steps don't contend
with each other (separate spawned processes — Zero123++'s VRAM is released
before Hunyuan3D-2mv's process starts); the risk `withGpuExclusive` guards
against is Ollama/speech contending with either one.

## What this fixes vs. doesn't

Fixes: pose/style inconsistency across the three views (the actual root
cause of the detached-legs failure) — by construction, not by catching it
after the fact. Live-verified against the original failure case.

Doesn't fix: the underlying shape-generation model still produces geometry
rougher than a hand-modeled asset — that's inherent to AI shape generation,
not a consistency problem. Bad proportions from monocular depth inference
struggling with foreshortened/dynamic poses are a separate issue this
pipeline doesn't touch either.

## Print-quality note (Bambu Lab P2S + AMS)

AMS's multi-color capability means Hunyuan3D's texture output isn't purely
cosmetic the way it would be on a single-color printer — Bambu Studio can
take a textured model and generate a multi-color AMS print plan
automatically. But it's a discretization, not photorealistic reproduction:
AMS prints a handful of distinct filament colors (4 per unit, more with
multiple units), not a smooth continuous texture gradient, so a generated
texture map gets quantized down to a limited palette with visible color-
transition seams — more "cartoon-shaded" than "painted." Worth knowing
before expecting a photorealistic result out of the printer.

## Next: texture+paint pipeline

The other 16GB-unlocked feature — Hunyuan3D-2's full texture+paint
pipeline (currently shape-only, bare geometry) — is planned next but not
yet designed in detail. It needs a separately-compiled CUDA rasterizer;
see `HUNYUAN3D_SETUP.md`'s "Known limitations" for what's documented about
it so far.
