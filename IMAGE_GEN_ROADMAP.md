# Image-generation pipeline — novel-view synthesis, pose-guided generation, texture+paint (implemented)

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
→ Hunyuan3D-2mv (multiview) → mesh → repair → texture+paint → slicer
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
6. **Texture+paint** (search-sourced *and* attached/URL images — unlike
   step 3, this isn't limited to search results): after mesh repair and
   Fusion import, `image23d/Hunyuan3D-2/run_texture_paint.py` runs
   Tencent's own `Hunyuan3DPaintPipeline` on the repaired mesh plus
   **every reference image available** — the front seed, and the left/back
   Zero123++ novel views from step 4 when they were generated — producing a
   textured `.glb` that becomes the browser preview directly (replacing the
   plain untextured GLB conversion, not running alongside it). Multiple
   images is deliberate, confirmed via a real controlled comparison: a
   front-only image leaves the mesh's back essentially hallucinated by the
   model's general priors (confirmed live as a near-blank/faint-lines
   result); feeding the left/back views the shape stage already generated
   produces a real, coherent back-of-costume texture instead, with no
   measured cost to front quality. **Default-on**, opt out with phrasing
   like "shape only" / "no texture" / "don't paint it" (`wantsShapeOnly` in
   `server.ts`). Runs on the *repaired* mesh specifically — repair drops
   stray disconnected components first, so texture generation never wastes
   time/VRAM painting geometry that would have been thrown away anyway. On
   any failure (native extension not compiled on a given machine, model
   download failure, CUDA OOM, timeout), falls back to the plain untextured
   GLB conversion — never blocks a request that would have succeeded
   before this feature existed. See `HUNYUAN3D_SETUP.md` section 6 for the
   native-extension compilation this needed (one CUDA extension, one plain
   C++ extension with a pure-Python fallback) and a real CUDA-13.x/MSVC
   preprocessor compatibility fix required along the way.

   **Face-count cap before texture painting** (`MAX_FACES`/
   `TARGET_FACES_AFTER_SIMPLIFY` in `run_texture_paint.py`): a real bug,
   confirmed live, not hypothetical — an unusually complex mesh (~1.7M
   faces) stalled the `mesh_uv_wrap` (xatlas UV-unwrap) step at 0% GPU
   utilization for 30+ minutes, a CPU-bound step that scales very poorly
   with face count. Meshes over 500K faces are now simplified to 300K via
   `trimesh.simplify_quadric_decimation` before texture painting — a
   reasoned first-pass cap using the one confirmed-safe reference point
   (407K faces, ~163s), not a full empirical bisection (each iteration
   costs 30+ minutes on a mesh this large). Confirmed live: decimating a
   1.63M-face mesh down to 300K took 1.5s, and the full texture-painting
   run that previously stalled indefinitely then completed in 129s with no
   visible quality loss.

   Confirmed live: ~7.2GB peak VRAM (`enable_model_cpu_offload()`, same
   pattern as pose-guided generation), ~130-165s per generation once
   weights are cached (varies with final face count after any capping),
   ~26 minutes combined for the first-run `hunyuan3d-paint-v2-0-turbo` +
   `hunyuan3d-delight-v2-0` weight download. Real texture quality confirmed
   by direct visual inspection of a live "spiderman standing" request's
   output: recognizable red/blue costume, visible spider emblem, muscular
   shading — a dramatic improvement over bare grey geometry. One known
   cosmetic gap: small isolated mesh regions (e.g. one foot) can
   occasionally come out untextured — a baking/inpainting edge case, not a
   functional failure.
7. **Slicer** (Bambu Studio) — `sliceModel()` in `server.ts` drives Bambu
   Studio's own CLI directly, chained after mesh repair and Fusion import;
   see `PRINTER_SETUP.md`. Slices the pre-texture `.obj`, unaffected by
   whether texture generation succeeded — texture only ever changes the
   browser preview `.glb`.

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
16GB upgrade — this pipeline can now run up to four back-to-back
GPU-heavy steps in one request, all confirmed live (not estimated):
pose-guided seed generation (~8GB, SDXL + ControlNet, only when
triggered), Zero123++ novel-view synthesis (~5GB), Hunyuan3D-2 shape
generation (~6GB), and texture+paint (~7.2GB, only when not opted out).
Texture generation runs in its *own* `withGpuExclusive` window, separate
from the one wrapping the first three steps — by the time repair/import
finish, that first window has already closed, and texture is a genuinely
new GPU-heavy step rather than a continuation of it. Ollama's two resident
models plus Whisper/Kokoro plus any one of these generation steps is still
a real way to approach 16GB. The generation steps don't contend with each
other (separate spawned processes — each releases its VRAM before the
next starts); the risk `withGpuExclusive` guards against is Ollama/speech
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
rougher than a hand-modeled asset — confirmed directly via live inspection
(blobby, fingerless hands; featureless faces). This is a real ceiling of
Hunyuan3D-2 (2.0) specifically, not a settings problem: Hunyuan3D-2.5
(which is specifically trained to fix this) has no public weights and
isn't usable locally, and Hunyuan3D-2.1 was evaluated and rejected as
needing ~21-29GB VRAM, more than this 16GB card has. Texture/color quality
is now genuinely good (see step 6 above); shape/geometry quality is not,
and closing that gap isn't possible with a model generation this project
can currently run. The canonical pose skeleton is a human pose, so
pose-guided generation only meaningfully helps humanoid subjects — a
non-humanoid dynamic subject (e.g. "a dragon flying") isn't helped by it,
and remains an open gap. Attached images and pasted URLs are never
pose-corrected (they're the user's own real object) — a user-supplied photo
in a dynamic pose can still produce the original failure mode.

## Print-quality note (Bambu Lab P2S + AMS)

Now live, not hypothetical: AMS's multi-color capability means
Hunyuan3D's texture output isn't purely cosmetic the way it would be on a
single-color printer — Bambu Studio can take a textured model and
generate a multi-color AMS print plan automatically. But it's a
discretization, not photorealistic reproduction: AMS prints a handful of
distinct filament colors (4 per unit, more with multiple units), not a
smooth continuous texture gradient, so a generated texture map gets
quantized down to a limited palette with visible color-transition seams —
more "cartoon-shaded" than "painted." Worth knowing before expecting a
photorealistic result out of the printer. (Note: `sliceModel()` currently
slices the pre-texture `.obj`, not the textured `.glb` — this AMS-plan
behavior applies if/when the slicing step is pointed at the textured
output in the future, not to today's actual print output.)

## Mesh editing (implemented)

Chat-driven editing of a model Noah already generated, scoped to Phase 1:
editing what's already in the registry, not uploading external files
(that's a deferred Phase 2) and not combining two models (deferred
Phase 3, blocked on reference-resolution design, not on tooling —
`manifold3d` boolean union/difference already confirmed working on test
geometry).

1. **Target resolution.** `openModelId` rides along on every `/api/chat`
   request from the frontend's already-tracked `currentModelId`.
   `resolveEditTargetModel()` prefers that explicit id (looked up in the
   file-backed model registry) and falls back to the most-recently-created
   registry entry when nothing is open — e.g. "make it bigger" said right
   after a generation, before the viewer panel is opened. Live-verified:
   with no `openModelId` in the request, a "mirror it" correctly resolved
   to whichever model was most recently created in the registry.

2. **Trigger phrasing.** Four independent patterns, checked before the
   older `wantsModification` (self-modification / personality-and-memory
   patch) regex, which would otherwise swallow phrasings like "edit the
   mesh" or "update the model" — `wantsModification` now explicitly gates
   on `!looksLikeMeshEditRequest`. Live-verified: a genuine
   "remember that I prefer metric units" request still routed to the
   self-modification path, not mesh editing, after this change.
   - scale-to-absolute-size ("make it 5cm") and by-factor ("2x bigger",
     "smaller", "half", "by 30%")
   - mirror ("mirror it", "flip it", axis inferred from wording —
     vertical/depth phrasing selects Y/Z, defaults to X)
   - simplify / reduce poly count ("simplify it", "fewer faces", "by
     half")
   - fill holes ("fill the holes", "patch the gaps")

3. **Reuse vs. new.** Scale-to-absolute-size reuses `scaleMeshToTargetSize()`
   / `scale_mesh.py` unchanged; fill-holes reuses `repairMesh()` /
   `repair_mesh.py` unchanged (it already handles this, more carefully
   than a reimplementation would). Scale-by-factor, mirror, and simplify
   are new, in `image23d/edit_mesh.py` — one script, an operation-name
   argv, same conventions as every other script here (plain `sys.argv`,
   `trimesh.load(..., process=False)` + Scene-flatten, no try/except,
   writes to a separate output file so a failed or partial edit can never
   corrupt the input).

4. **Texture and simplify don't mix.** Confirmed directly, not assumed:
   simplifying a textured `.glb` via `simplify_quadric_decimation`
   converts `TextureVisuals` to `ColorVisuals` — the model comes out
   visually blank/speckled, texture genuinely destroyed, not just
   degraded. Texture-aware decimation is out of scope for Phase 1, so the
   simplify reply text carries an explicit warning ("removes
   color/texture — the model will look bare afterward") instead. Scale
   and mirror are pure vertex transforms and confirmed safe on a textured
   `.glb` (`TextureVisuals` preserved through both, live-verified);
   fill-holes was also confirmed to preserve texture.

5. **Output handling.** Edits replace the registry entry's file in place
   (same `id`, new filename) rather than creating a new entry — matches
   "editing," not "generating another model," and needs zero frontend
   changes since `openModelViewerPanelById()` already refetches
   `/api/models/:id` instead of trusting a baked-in URL. `saved` status
   survives an edit. Failure leaves the original file completely
   untouched: `editMesh()` only resolves success after confirming a zero
   exit code *and* that the output file actually exists, so
   `replaceGeneratedModelFile()` never runs on a failed edit. Verified
   directly — a deliberately malformed parameter makes `edit_mesh.py`
   exit non-zero and write no output file at all, exactly the contract
   `editMesh()` depends on.

6. **Preview-only.** No auto re-slice or re-import after an edit — same
   posture as `scaleMeshToTargetSize`/`repairMesh` being pure
   mesh-to-mesh steps with no side effects. The reply says the edit
   applied and that printing again is a separate ask. There's currently
   no chat command meaning "(re-)import/print the model that's already
   open" outside a fresh generation reply — a reasonable small fast-follow
   later, reusing `importMeshToFusion`/`sliceModel` against
   `resolveEditTargetModel()`'s file.

## Custom pose reference generation (implemented)

Lets an actual dynamic pose reach the final model, instead of the
pose-guided-generation safety net (above) silently substituting a neutral
standing pose for every action-pose request. Attach a reference image
(a photo, or even a plain silhouette with no color/texture at all) plus
phrasing like "in this pose" / "use this pose" / "like this pose", and
the pipeline extracts a live pose skeleton from it instead of always using
the fixed canonical one.

1. **Trigger.** `POSE_REFERENCE_PATTERN` requires both an attached image
   *and* pose-reference phrasing — a flat silhouette obviously isn't a
   literal object to reconstruct, but not every attached image is a pose
   reference either (plain object-photo attachments are the existing,
   more common case), so this needs explicit wording rather than guessing
   from image content. Without that phrasing, an attached image still goes
   through the completely unchanged literal-reference path.

2. **Extraction.** New `image23d/extract_pose_skeleton.py` promotes the
   one-time canonical-pose-asset tool's `rtmlib.Wholebody` pose estimator
   into a live, per-request script (CPU-only, no GPU contention). Confirmed
   directly: it reliably extracts a usable skeleton even from a solid
   silhouette with zero internal visual detail — the same category of
   reference the pose-guided-seed system was built to avoid trusting.

3. **Safety check — the real root cause isn't "dynamic," it's overlap.**
   The original detached-limb failures came from limbs overlapping/
   occluding each other in the seed image, not from dynamic poses as a
   category. `extract_pose_skeleton.py` computes two signals per limb
   (mean keypoint confidence, and pairwise bounding-box overlap between
   the four limbs) and rejects the pose if either looks risky, falling
   back to the existing neutral canonical pose with a clear reason in the
   reply text rather than failing the request outright. Confirmed via
   direct unit testing of the overlap math against constructed keypoints
   (correctly flags a genuine crossing case, correctly passes a genuinely
   separated one) — a synthetic crossed-arms test image failed to actually
   fool the pose estimator into reading it as crossed, a known limitation
   of hand-drawn test images rather than a gap in the check itself.

4. **A real bug found and fixed during testing.** The first live generation
   from a silhouette-derived skeleton came back with the mask only
   covering the top half of the face, a real human chin/mouth visible
   underneath. Root cause: `openpose134`'s 68-point face contour mesh
   (indices 24-91) still emits *something* when run against a face with no
   real features to anchor to — low-confidence, noisy positions that
   apparently fight against SDXL rendering a full mask. Fixed by zeroing
   those keypoints' scores before rendering the skeleton (keeping only
   nose/eyes/ears for coarse head orientation), confirmed live: regenerating
   with the same skeleton produced a complete mask. The existing canonical
   pose asset never had this problem, since its source photo has a real
   face to detect.

5. **`generate_pose_seed.py` extension.** Optional 3rd arg swaps in the
   extracted skeleton instead of the fixed canonical one, and drops
   "action pose, dynamic pose" from the negative prompt (those exclusions
   only make sense against the neutral default). Absent, the call is
   byte-for-byte identical to before this feature existed — confirmed live
   by generating from both call shapes side by side.

6. **Live-verified, full pipeline.** A real end-to-end request (silhouette
   attachment + "in this pose", subject "spiderman") correctly extracted
   and accepted the skeleton, generated a genuinely dynamic-pose seed image
   (not the neutral default), and produced a textured, reasonably
   proportioned mesh through the unchanged downstream pipeline (novel-view
   synthesis, Hunyuan3D shape+multiview, texture paint, mesh repair,
   Fusion import) — reply text read "built from the pose from the image
   you attached, generated in the pose from the image you attached."
   Both regression paths were also confirmed unchanged in the same session:
   an attached image without pose phrasing still reads "built from the
   image you attached" (no extraction call at all), and a text-only
   dynamic-pose request with no attachment still produces the exact
   original wording, "replaced with a clean neutral-pose image generated
   to avoid an unreliable action-pose reference photo."

7. **Auto-search fallback — no attachment required.** A text-only
   dynamic-pose request (no attached image at all) now also tries to find
   a real pose first, before falling back to the neutral-pose
   substitution: a separate, pose-focused search query (`${subject} action
   pose` — deliberately different from the "isolated on white background"
   query used for the appearance reference, since that phrasing biases
   toward neutral studio shots) feeds candidates through the same
   `extractPoseSkeleton()` safety check as an attached image, reusing the
   existing `downloadFirstAvailableImage()` try-until-one-works helper.
   First candidate that passes wins; if none do (or the search finds
   nothing), it falls through to the exact same neutral-pose substitution
   as before this addition — unchanged wording, unchanged behavior.
   Live-verified: "spiderman swinging" with no attachment found and used a
   real action-pose search photo, reply text reading "generated in a
   dynamic pose found from a separate reference search," completing the
   full pipeline (textured, imported, sliced) successfully. Trade-off
   worth knowing: this reintroduces some of the uncertainty an attached
   image avoids — you get *a* pose that passes the safety check, not
   necessarily the specific pose you'd have picked yourself. Attaching
   your own reference is still the way to get an exact pose; this is a
   convenience fallback for when you don't have one handy.

**What this doesn't fix**: appearance/color accuracy (comic-correct suit
colors, logo placement) is explicitly out of scope here — the reference
image supplies pose only, and getting the exact costume right depends on
the base model's own prompting quality, a separate, already-known gap (see
"What this fixes vs. doesn't" above). One live test also showed
inconsistent per-leg coloring (one leg solid-filled, no suit pattern) on a
generation from a crude, non-anatomical test skeleton — not yet confirmed
whether this reproduces with an anatomically normal reference image (a
real photo or illustration), since it wasn't observed on the canonical
neutral-pose path with the same prompt. Worth rechecking with a real
reference photo before relying on this for consistently clean texture
output on dynamic poses.

## Text-to-image generation (implemented)

A standalone capability, separate from everything above: "generate an
image of X" produces a picture shown inline in the chat, with no 3D
pipeline involved at all. Scoped deliberately narrow at the time —
text-to-image only, no editing/img2img of an attached photo (that would
make attaching an image a 3-way ambiguity against the existing
"reconstruct as 3D model" / "use as pose reference" meanings; a fourth
meaning was added later, see "Image editing / restyling" below) — and
images are persisted (save/discard), mirroring the 3D model registry
rather than being ephemeral.

1. **Reuses proven infrastructure end to end.** `image23d/generate_image.py`
   is a plain `StableDiffusionXLPipeline` (no ControlNet) — the same SDXL
   base and Hunyuan3D-2 venv already used for pose-guided generation, just
   without the pose-conditioning machinery. No fixed seed (unlike the
   pose-seed script, which fixes one intentionally for a reproducible
   downstream base) — confirmed live that two runs of the identical
   prompt produce genuinely different images, which is the point here.

2. **A parallel registry, not a shared one.** `GeneratedImageEntry` /
   `IMAGES_DIR` / `loadImagesIndex()` / `saveImagesIndex()` /
   `registerGeneratedImage()` mirror the 3D model registry's shape
   exactly, including the same oldest-unsaved-first pruning — kept as its
   own separate structure rather than a generalized "asset registry",
   matching this codebase's existing preference for small parallel pieces
   over shared abstractions. Three REST endpoints
   (`GET`/`POST .../save`/`DELETE` on `/api/images/:id`) mirror
   `/api/models/:id` exactly, down to reusing the same `slugify()` for
   the save-rename.

3. **New chat surface, not a new panel.** Unlike 3D models (which get a
   dedicated viewer panel), a generated image renders directly inline in
   the assistant's chat bubble via a new optional `image` param on
   `appendMessage()`, with small SAVE/DISCARD buttons underneath —
   deliberately lightweight, since an image doesn't need the model
   viewer's dedicated UI surface.

4. **Trigger correctly excludes the existing 3D pipeline.**
   `IMAGE_GEN_TRIGGER_PATTERN` requires an image/picture/photo/
   illustration/artwork/drawing/painting/wallpaper noun, checked against
   every other trigger pattern in the file with no vocabulary overlap —
   verified directly (not just assumed) against realistic phrasings
   including the tricky edge case "make me a 3d model based on this
   picture", which contains both an image noun and a 3D-model phrase:
   the existing `!looksLikeFusionRequest` gate correctly keeps this
   routed to the 3D pipeline, not image generation.

5. **Live-verified, full path.** A real request ("a fox sitting in an
   autumn forest") generated a genuine SDXL image, rendered correctly
   inline in the chat bubble (confirmed via the actual running app, not
   just the API response), SAVE renamed the file to
   `fox-sitting-in-an-autumn-forest.png` and persisted it, and DISCARD on
   a second generated image removed it from both the DOM and disk.
   Confirmed regressions: a plain weather/chat question is entirely
   unaffected, and the 3D-model trigger patterns remain mutually
   exclusive with the new one under direct testing.

## Image editing / restyling (implemented)

The fourth meaning of "attach an image": broad restyling of an attached
photo ("make this look like a watercolor painting", "make it darker and
moodier"), via SDXL img2img rather than a dedicated instruction-based
edit model — a deliberate scope choice (precise, localized edits like
"change the car to red" would need the latter) that let this reuse the
existing SDXL infrastructure with zero new model download.

1. **`image23d/edit_image.py`, mirroring `generate_image.py`.** Same
   venv, same `stabilityai/stable-diffusion-xl-base-1.0` weights (already
   cached locally) — `StableDiffusionXLImg2ImgPipeline` shares the base
   pipeline's unet/vae/text-encoders, confirmed no new download needed.
   Input photos are fit within a 1024x1024 box preserving aspect ratio
   (never hard-cropped to a square) and rounded to a multiple of 8 for
   the VAE.

2. **A bare style name doesn't work — confirmed live, this took real
   iteration to find.** "a watercolor painting" alone, at any strength
   tested, at best did nothing and at worst still looked fully
   photorealistic. Neither a stronger negative prompt (explicitly
   discouraging "photograph, realistic") nor generic quality boosters
   ("highly detailed") fixed this. Only concrete medium/texture language
   did ("soft brushstrokes, visible paint texture, color bleeding").
   `IMAGE_EDIT_STYLE_EXPANSIONS` in `server.ts` maps every style/mood
   keyword `IMAGE_EDIT_TRIGGER_PATTERN` recognizes to language confirmed
   to actually work, via `buildImageEditPrompt()` — the SDXL prompt is
   never the raw user message.

3. **No fixed seed means outcome quality is genuinely random per run,
   independent of the prompt** — confirmed directly: the *identical*
   prompt and strength (0.7) produced a clear, convincing watercolor
   restyle on one run and a near-untouched photorealistic result on the
   next, repeatedly. This is a real, known limitation of this approach,
   not a bug to chase further right now: 0.6 was consistently too subtle
   regardless of seed; 0.8 (the current default) was clearly more
   consistent across many repeated runs, but not bulletproof — expect an
   occasional run that comes back only weakly restyled. Fixing this
   properly would mean either accepting heavier identity drift from a
   higher strength, or a more involved change (e.g. detecting a
   too-similar-to-source result and automatically retrying).

4. **Two required collision fixes**, not just a new trigger pattern.
   `wantsModification`'s regex already matches the literal word "edit" —
   "edit this photo to look like a painting" would otherwise misroute
   into the self-modification flow, the same failure class already fixed
   once for mesh editing and face enrollment. Separately,
   `IMAGE_GEN_TRIGGER_PATTERN`'s "make ... photo" genuinely matches "make
   this photo darker," so `looksLikeImageGenRequest` needed an explicit
   exclusion too, confirmed via direct testing, not just added
   defensively.

5. **Live-verified**, including the specific failure mode this was built
   to avoid: "can you edit this photo to look like a painting" correctly
   restyles the photo rather than falling into self-modification.
   Regressions confirmed unaffected: plain text-to-image generation, a
   genuine self-modification request, face enrollment with an attached
   photo, and "make a 3d model of this" (still routes to the image-to-3D
   pipeline, not restyling, via `IMAGE_EDIT_3D_EXCLUDE_PATTERN`).

## What's next

All three of this session's 16GB-unlocked features are now built:
novel-view synthesis, pose-guided generation, texture+paint. Remaining
open gaps, in rough priority order: shape/geometry fidelity (blocked on a
model generation this project can't currently run — see "What this fixes
vs. doesn't" above); pose-guided generation only covering humanoid
subjects; custom pose reference's unconfirmed per-leg coloring
inconsistency (see above — needs rechecking with a real reference photo,
not just a synthetic test skeleton); moving generation steps to a
dedicated second GPU if the 3060 Ti gets reinstalled (would remove the
need for `withGpuExclusive` to pause Ollama/speech at all); mesh editing
Phase 2 (upload and edit external `.stl`/`.obj` files) and Phase 3
(boolean combine of two generated
models, needs a reference-resolution design for picking between multiple
same-type entities by subject name).
