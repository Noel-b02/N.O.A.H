import sys

import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageOps

# Generates the "left" and "back" reference views for the multiview
# image-to-3D pipeline from a single verified seed photo, replacing the
# old approach of sourcing all three views from independent web searches
# (which could disagree on pose/style — confirmed to produce a mesh with
# detached extra limbs when they did). Zero123++ generates all 6 fixed-
# angle views in one diffusion pass specifically to keep them consistent
# with each other and with the seed, so consistency is enforced by
# construction here rather than checked after the fact.
#
# Output is a single 640x960 image: a 2-column x 3-row grid of 320x320
# tiles. Confirmed empirically (not from documentation, which didn't state
# this clearly) that reading order top-left -> top-right -> mid-left ->
# mid-right -> bottom-left -> bottom-right corresponds exactly to azimuths
# [30, 90, 150, 210, 270, 330] (relative to the seed as 0/front) — visually
# validated by the two edge-on/thin-silhouette tiles landing at top-right
# and bottom-left, which are the 90/270 side-profile azimuths.
#
# 90 is an exact match for "left". There's no exact 180 for "back" — 150
# and 210 are both 30 off; this picks 210 (mid-right) as a first pass.
# Retune by changing these two indices alone if Hunyuan3D-2mv output looks
# better with the other one.
GRID_COLS = 2
TILE_SIZE = 320
LEFT_TILE_INDEX = 1   # top-right,  azimuth  90 (exact)
BACK_TILE_INDEX = 3   # mid-right,  azimuth 210 (approximation, tunable)

NUM_INFERENCE_STEPS = 36  # SUDO-AI's own guidance: 28 for general objects, 75-100 for fine detail — this is a middle-ground default, not tuned against real print-quality output yet.

seed_path = sys.argv[1]
left_output_path = sys.argv[2]
back_output_path = sys.argv[3]

image = Image.open(seed_path).convert("RGB")
# Zero123++ expects a square input; pad rather than crop so nothing of the
# subject is lost.
image = ImageOps.pad(image, (max(image.size),) * 2, color=(255, 255, 255))

pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16,
    # This pulls and executes pipeline.py from the sudo-ai/zero123plus-pipeline
    # Hub repo at load time — same trust category as any other model this
    # project downloads from Hugging Face, but worth knowing it's not just
    # weights.
    trust_remote_code=True,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing="trailing"
)
pipeline.to("cuda")

grid = pipeline(image, num_inference_steps=NUM_INFERENCE_STEPS).images[0]


def tile(index: int) -> Image.Image:
    row, col = divmod(index, GRID_COLS)
    left = col * TILE_SIZE
    top = row * TILE_SIZE
    return grid.crop((left, top, left + TILE_SIZE, top + TILE_SIZE))


tile(LEFT_TILE_INDEX).save(left_output_path)
tile(BACK_TILE_INDEX).save(back_output_path)
