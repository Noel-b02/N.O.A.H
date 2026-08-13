import sys
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# Shape-only generation — skips the texture/paint pipeline entirely (that
# needs its own compiled CUDA rasterizer and ~16GB VRAM; this path only
# needs ~6GB). Fixes a bug in Tencent's own minimal_demo.py, which converts
# the image to RGBA before checking `if image.mode == 'RGB'` — a check that
# can then never be true, so background removal silently never runs there.
image_path = sys.argv[1]
output_path = sys.argv[2]

image = Image.open(image_path).convert("RGB")
rembg = BackgroundRemover()
image = rembg(image)

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
# octree_resolution=512 (max supported, default 384) — the generative
# model's internal spatial resolution, the real lever for raw geometric
# detail (not post-processing). Left at the conservative default under the
# 8GB constraint; bumped now that there's real VRAM headroom. See
# run_multiview.py for the same change and num_chunks reasoning.
mesh = pipeline(image=image, num_chunks=20000, octree_resolution=512)[0]
mesh.export(output_path)
