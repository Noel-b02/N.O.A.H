import sys

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

# Img2img restyling of an attached photo — reuses generate_image.py's exact
# model choice and venv (same stabilityai/stable-diffusion-xl-base-1.0
# weights, already cached locally from that script's own first run). The
# Img2Img pipeline class shares the base pipeline's unet/vae/text-encoders,
# so this needed zero new download.
NEGATIVE_PROMPT = "blurry, deformed, extra limbs, missing limbs, watermark, text, cropped, worst quality"
NUM_INFERENCE_STEPS = 30

input_image_path = sys.argv[1]
prompt = sys.argv[2]
output_path = sys.argv[3]
# Effective step count is strength * NUM_INFERENCE_STEPS. This script sets
# no manual seed (unlike generate_pose_seed.py, which fixes one on purpose
# for a reproducible downstream base) — confirmed live that this makes
# outcome quality genuinely random per run, not just prompt-dependent: the
# *identical* prompt and strength (0.7) produced a clear, convincing
# watercolor restyle on one run and a near-untouched photorealistic result
# on the next, repeatedly. 0.6 was consistently too subtle regardless of
# seed. 0.8 was clearly more consistent across many repeated runs (most
# produced an obvious style transfer, one came back only partially
# restyled) — not bulletproof, but a real improvement, and an occasional
# weak result is a better failure mode for "broad restyling" than the
# heavier identity drift higher strengths cause. See IMAGE_GEN_ROADMAP.md
# for this as a documented known limitation. Not a universal constant,
# hence the CLI override.
strength = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8

# A hard resize to a fixed square would distort any non-square photo before
# SDXL even sees it. Fit within a 1024x1024 box instead, preserving aspect
# ratio, and round down to a multiple of 8 (SDXL's VAE requires dimensions
# divisible by 8).
source_image = Image.open(input_image_path).convert("RGB")
width, height = source_image.size
scale = min(1024 / width, 1024 / height, 1.0)
target_width = max(8, int(width * scale) // 8 * 8)
target_height = max(8, int(height * scale) // 8 * 8)
source_image = source_image.resize((target_width, target_height))

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt,
    image=source_image,
    strength=strength,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=NUM_INFERENCE_STEPS,
).images[0]

image.save(output_path)
