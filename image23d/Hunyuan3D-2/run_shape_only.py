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
mesh = pipeline(image=image)[0]
mesh.export(output_path)
