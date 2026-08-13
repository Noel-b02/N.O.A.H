# Image-generation pipeline — single-seed novel-view synthesis + pose-guided generation (implemented)

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

A second, related bug remained even after single-seed novel-view synthesis
shipped: a single search-sourced photo can itself show the subject mid-action
(limbs overlapping/occluded/foreshortened), and Zero123++ faithfully
reproduces whatever pose the seed shows — so a bad single photo still
produced detached-limb geometry, just from a different root cause. This is
what pose-guided seed generation (step 3 below) fixes.

## Pipeline

```
one reference image → vision verification → [dynamic pose? generate a
clean neutral-pose replacement] → generate side/back views (Zero123++)
→ Hunyuan3D-2mv (multiview) → mesh → slicer
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
3. **Pose-guided seed generation (search-sourced images only)**: if the
   request reads as a dynamic/action pose (`DYNAMIC_POSE_PATTERN` in
   `server.ts` — swinging, jumping, fighting, etc.), the raw search photo's
   pose is never trusted. Instead: `describeImageAppearance` (an Ollama
   vision call, same pattern as `verifyImageMatchesSubject`) produces a
   short costume/appearance description from the verified photo, and
   `image23d/generate_pose_seed.py` generates a brand-new image combining
   that description with a **fixed canonical neutral pose** — SDXL
   (`stabilityai/stable-diffusion-xl-base-1.0`) + ControlNet-openpose
   (`thibaud/controlnet-openpose-sdxl-1.0`), conditioned on
   `image23d/assets/canonical_pose.png` (a static OpenPose-style skeleton,
   made once via `image23d/tools/make_canonical_pose_asset.py`, not
   regenerated per request). The pose is deliberately never extracted live
   from the search photo — doing that would just reintroduce the pose
   reliability problem this exists to remove. This generated image replaces
   the raw search photo as the seed for step 4; on any failure, the
   pipeline falls back to the raw search photo (same "still produce
   something usable" posture as the rest of this pipeline). Attached images
   and pasted URLs are never replaced — they're the user's own real object,
   not a generic subject.

   **Model note**: SDXL was chosen over the originally-planned
   Z-Image-Turbo after live testing. Z-Image-Turbo's 6B-parameter
   transformer measured ~15.5GB real VRAM on this 16GB card at both full
   bf16 and Q8_0 GGUF quantization alike (GGUF's dequantize-on-forward-pass
   approach didn't reduce PyTorch's peak allocator footprint the way its
   on-disk size would suggest), pushing generation past 10 minutes *per
   diffusion step*. `enable_model_cpu_offload()`, diffusers' standard
   low-VRAM fallback, crashed with a genuine device-mismatch bug in
   Z-Image's `adaLN_modulation` layer — confirmed as an upstream gap in how
   freshly Z-Image ControlNet support was added to diffusers as of the
   installed version, not fixable from this project's side. SDXL +
   `enable_model_cpu_offload()` measured a real ~8GB peak VRAM and
   16-25s per generation — confirmed directly, not assumed.
4. **Generate side/back views**: `image23d/generate_novel_views.py` runs
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
5. **Hunyuan3D-2mv**, unchanged — already expected exactly this
   `{front, left, back}` input shape; `generateMeshWithHunyuan3DMultiview`
   in `server.ts` didn't need to change at all.
6. **Slicer** (Bambu Studio) — `sliceModel()` in `server.ts` drives Bambu
   Studio's own CLI directly, chained after mesh repair and Fusion import;
   see `PRINTER_SETUP.md`.

`wantsSingleImageOnly` (opt-out phrasing like "one image only" / "no
multiview") skips steps 4-5 entirely and goes straight to shape-only
generation from the seed, unchanged from before — pose-guided generation
(step 3) still runs first if triggered, since a bad pose wrecks shape-only
geometry just as much as multiview. If novel-view generation itself fails
(script crash, timeout), the pipeline falls back to shape-only rather than
failing the request outright — same "still produce something usable"
reasoning the old search-based multiview attempt used.

## GPU usage

`withGpuExclusive` (see `server.ts`) is kept, not removed, despite the
16GB upgrade — this pipeline can now run up to three back-to-back
GPU-heavy steps in one request: pose-guided seed generation (~8GB, SDXL +
ControlNet, only when triggered), Zero123++ novel-view synthesis (~5GB),
and Hunyuan3D-2 shape generation (~6GB). Ollama's two resident models plus
Whisper/Kokoro plus any one of these generation steps is still a real way
to approach 16GB. The generation steps don't contend with each other
(separate spawned processes — each releases its VRAM before the next
starts); the risk `withGpuExclusive` guards against is Ollama/speech
contending with any one of them.

## What this fixes vs. doesn't

Fixes: pose/style inconsistency across the three views (the actual root
cause of the original detached-legs failure) — by construction, not by
catching it after the fact. Also fixes the related single-photo case: a
search-sourced photo caught mid-action producing detached-limb geometry
even after single-seed novel-view synthesis shipped — by generating a
clean, neutral-pose replacement before Zero123++ ever sees the seed. Both
live-verified against the original failure cases, including direct mesh
inspection (single connected component, no stray limb fragments) not just
visual/textual confirmation.

Doesn't fix: the underlying shape-generation model still produces geometry
rougher than a hand-modeled asset — that's inherent to AI shape generation,
not a consistency problem. The canonical pose skeleton is a human pose, so
pose-guided generation only meaningfully helps humanoid subjects — a
non-humanoid dynamic subject (e.g. "a dragon flying") isn't helped by it,
and remains an open gap. Attached images and pasted URLs are never
pose-corrected (they're the user's own real object) — a user-supplied photo
in a dynamic pose can still produce the original failure mode.

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
