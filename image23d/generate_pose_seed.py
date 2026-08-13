import sys
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image

# Generates a clean, neutral-pose seed image from a text description plus a
# FIXED canonical pose skeleton (image23d/assets/canonical_pose.png, made
# once offline — see image23d/tools/make_canonical_pose_asset.py and
# HUNYUAN3D_SETUP.md). Exists because a search-sourced photo caught
# mid-action (limbs overlapping/occluded/foreshortened) still passes
# subject-match verification but reliably produces detached-limb mesh
# geometry downstream, since generate_novel_views.py's Zero123++ step
# faithfully reproduces whatever pose the seed shows — confirmed directly
# with "spiderman swinging". The skeleton is deliberately NEVER extracted
# live from the search photo: doing that would just reintroduce the same
# reliability problem this exists to remove.
#
# Model is SDXL + thibaud/controlnet-openpose-sdxl-1.0, not the initially
# planned Z-Image-Turbo — Z-Image's 6B-param transformer measured ~15.5GB
# real VRAM on this 16GB card regardless of GGUF quantization (dequantized
# working-set size dominates, not on-disk size), pushing generation past 10
# minutes per diffusion step, and diffusers' enable_model_cpu_offload()
# crashed on Z-Image's ControlNet path with a genuine upstream device-
# mismatch bug. SDXL + enable_model_cpu_offload() measured a real 7.95GB
# peak VRAM and ~16s for a 30-step generation — confirmed directly, not
# assumed.

CANONICAL_POSE_PATH = Path(__file__).parent / "assets" / "canonical_pose.png"

NEGATIVE_PROMPT = "blurry, deformed, extra limbs, missing limbs, watermark, text, cropped, worst quality, action pose, dynamic pose"
NUM_INFERENCE_STEPS = 30

prompt = sys.argv[1]
output_path = sys.argv[2]

if not CANONICAL_POSE_PATH.exists():
    raise FileNotFoundError(
        f"Canonical pose asset missing at {CANONICAL_POSE_PATH} — see "
        "HUNYUAN3D_SETUP.md's one-time canonical-pose-asset step."
    )

control_image = Image.open(CANONICAL_POSE_PATH).convert("RGB")

controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt,
    image=control_image,
    negative_prompt=NEGATIVE_PROMPT,
    controlnet_conditioning_scale=1.0,
    num_inference_steps=NUM_INFERENCE_STEPS,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

image.save(output_path)
