import sys

import trimesh
from PIL import Image
from hy3dgen.texgen import Hunyuan3DPaintPipeline

# Texture+paint stage — takes the already-generated, already-repaired mesh
# plus one or more reference images, and produces a textured .glb. Separate
# script/process from run_shape_only.py so its VRAM releases independently
# and shape-only requests never pay this pipeline's load cost at all. Runs
# on the repaired mesh deliberately (not the raw shape output) so texture
# generation isn't wasted painting stray disconnected geometry that repair
# would have dropped anyway.
#
# Multiple images (front seed + left/back novel views, when available) is
# deliberate, not incidental — confirmed directly: Hunyuan3DPaintPipeline's
# __call__ accepts a list and genuinely uses every image passed in (not
# just the first). A single front-only image leaves the mesh's back
# essentially hallucinated by the model's general priors rather than
# reconstructed from anything real — confirmed live as a near-blank/faint-
# lines result. Feeding the left/back Zero123++ views the shape stage
# already generated fixes this with no measured cost to front quality.
# Face-count cap before texture painting — confirmed directly as a real,
# necessary fix: a 69MB/~1.7M-face mesh (from a genuinely complex subject)
# stalled at the mesh_uv_wrap (xatlas UV-unwrap) step for 30+ minutes at
# 0% GPU utilization — a CPU-bound step that scales very poorly with face
# count, not a hang. A 407K-face mesh completed the full pipeline in ~163s
# with no issue, so that's the confirmed-safe upper reference point.
# MAX_FACES gives real headroom above it; TARGET_FACES_AFTER_SIMPLIFY sits
# comfortably below it. Not a full empirical bisection (each iteration
# costs 30+ minutes on a mesh this large) — a reasoned first-pass cap,
# same honesty as this project's other unmeasured-but-reasoned constants.
MAX_FACES = 500000
TARGET_FACES_AFTER_SIMPLIFY = 300000

mesh_path = sys.argv[1]
output_path = sys.argv[2]
image_paths = sys.argv[3:]

loaded = trimesh.load(mesh_path, process=False)
# Export can be a multi-geometry Scene rather than a single Trimesh
# depending on the mesh format — same handling as scale_mesh.py/convert_to_glb.py.
mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded

if len(mesh.faces) > MAX_FACES:
    print(f"[texture] mesh has {len(mesh.faces)} faces, simplifying to {TARGET_FACES_AFTER_SIMPLIFY} before texture painting", file=sys.stderr)
    mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES_AFTER_SIMPLIFY)

images = [Image.open(p).convert("RGB") for p in image_paths]

pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
# Tencent's own README documents ~16GB total for shape+texture combined —
# this card's entire VRAM with zero measured margin for Ollama/speech's
# "paused, not exited" overhead. Same reasoning that led to
# enable_model_cpu_offload() for SDXL's pose-guided seed generation.
pipeline.enable_model_cpu_offload()

textured_mesh = pipeline(mesh, image=images)
# .glb, not .obj — texture needs UV+material data .obj can't carry.
textured_mesh.export(output_path)
