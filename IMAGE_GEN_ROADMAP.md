# Image-generation pipeline — next-phase design (not yet implemented)

Blocked on a GPU upgrade (planned: RTX 5060 Ti 16GB) — the 3060 Ti's 8GB
can't fit a local image-gen model alongside Ollama/Hunyuan3D. This is a
design note to pick back up once that VRAM headroom exists, not a plan to
implement now.

## Problem this solves

The current multiview path (see `server.ts`'s image-fusion branch,
`generateMeshWithHunyuan3DMultiview`) sources front/side/back reference
photos from three independent web searches. Even with subject, framing, and
cross-image consistency verification (all shipped), this is fundamentally
gambling that three unrelated photos happen to agree on pose, art style,
and appearance. Confirmed directly: a "spiderman swinging" request pulled
three photos in different poses, and Hunyuan3D-2mv — which assumes all
three inputs are one pose from three camera angles — produced a mesh with
two extra detached legs trying to reconcile the mismatch. Verification
catches *some* of this after the fact; it can't fix a scarcity problem
where no consistent set exists in the search results at all.

## Target pipeline

```
one reference image → vision verification → generate side/back views →
Hunyuan3D-2mv (multiview) → mesh → slicer
```

1. **Reference image**: attached image, pasted URL, or the single best web
   search result — whichever the existing single-image path already uses.
2. **Vision verification (moved earlier)**: verify this one seed image
   before spending compute generating from it — cheaper and more useful
   than verifying three outputs after the fact, since a correct seed
   guarantees on-subject generated views, and a wrong seed would only ever
   produce consistently-wrong outputs no matter how good the generation is.
3. **Generate side/back views from the verified reference**: needs a model
   built for *novel-view synthesis* specifically (e.g. Zero123-style /
   multi-view diffusion), not generic SDXL/Flux text-to-image — plain
   img2img preserves style/content but isn't trained to correctly rotate a
   subject to a specific camera angle. This is the part that actually
   needs the new GPU.
4. **Hunyuan3D-2mv**, unchanged — already expects exactly this
   {front, left, back} input shape.
5. **Slicer** (Bambu Studio) — no longer outside Noah. `sliceModel()` in
   `server.ts` drives Bambu Studio's own CLI directly, chained after mesh
   repair and Fusion import; see `PRINTER_SETUP.md`.

## What this fixes vs. doesn't

Fixes: pose/style inconsistency across the three views (the actual root
cause of the detached-legs failure) — by construction, not by catching it
after the fact.

Doesn't fix: the underlying shape-generation model still produces geometry
rougher than a hand-modeled asset — that's inherent to AI shape generation,
not a consistency problem, and no GPU upgrade changes it. Bad proportions
from the older single-image path are a separate issue (monocular depth
inference struggling with foreshortened/dynamic poses) that this pipeline
doesn't touch either.

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
