import sys
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# Multiview variant of run_shape_only.py — takes three separately-sourced
# angle photos instead of one, per Tencent's own examples/shape_gen_multiview.py.
# Same background-removal fix applies here (checking image.mode == 'RGB' after
# an unconditional convert("RGBA") can never be true, so run rembg first).
front_path, left_path, back_path, output_path = sys.argv[1:5]

images = {"front": front_path, "left": left_path, "back": back_path}
rembg = BackgroundRemover()
for key, p in images.items():
    image = Image.open(p).convert("RGB")
    images[key] = rembg(image)

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv", variant="fp16"
)
# num_chunks=20000 instead of the pipeline's default 8000 — a finer
# marching-cubes extraction pass, per Tencent's own multiview example.
# Costs more extraction time, not more VRAM or generation time, so it's a
# free-ish quality knob.
mesh = pipeline(image=images, num_chunks=20000)[0]
mesh.export(output_path)
