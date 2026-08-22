import sys

import torch
from diffusers import StableDiffusionXLPipeline

# Plain text-to-image SDXL generation — no ControlNet, no pose guidance.
# Reuses generate_pose_seed.py's proven model choice (SDXL base) and venv
# (Hunyuan3D-2's), confirmed there at ~7.95GB peak VRAM / ~16-25s per
# 30-step generation with enable_model_cpu_offload() — this should be equal
# or lighter since there's no control-image conditioning overhead. See
# generate_pose_seed.py's own comment for why SDXL was chosen over
# Z-Image-Turbo.
NEGATIVE_PROMPT = "blurry, deformed, extra limbs, missing limbs, watermark, text, cropped, worst quality"
NUM_INFERENCE_STEPS = 30

prompt = sys.argv[1]
output_path = sys.argv[2]

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

# Deliberately no manual_seed() here, unlike generate_pose_seed.py — that
# script fixes the seed because it's producing a reproducible neutral-pose
# base for a downstream 3D pipeline; this script's whole point is a fresh,
# varied picture per request, so a repeat "generate an image of a cat"
# giving back the identical image would be the bug, not the feature.
image = pipe(
    prompt,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=NUM_INFERENCE_STEPS,
).images[0]

image.save(output_path)
