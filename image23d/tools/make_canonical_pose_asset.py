import cv2
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from rtmlib import Wholebody, draw_skeleton

# ONE-TIME asset generation, not a live dependency of generate_pose_seed.py.
# Run manually in a disposable venv (pip install rtmlib diffusers torch) if
# the canonical pose ever needs to change — the output is a committed static
# asset, not something regenerated at request time.
#
# Step 1 generates a source photo (plain SDXL text-to-image, no ControlNet)
# of a person in a clean, front-facing, neutral standing pose. A generated
# photo was used instead of a sourced stock photo to avoid licensing
# questions entirely, and because it's fully reproducible from this one
# script.
#
# Step 2 runs rtmlib's whole-body pose estimator (RTMPose/DWPose-based,
# no mmcv/mmdet/mmpose dependency chain) against that photo and renders an
# OpenPose-style colored skeleton on a blank canvas — the actual ControlNet
# conditioning image, saved to image23d/assets/canonical_pose.png.

sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)
sdxl_pipe.enable_model_cpu_offload()

source_prompt = (
    "full body photo of a person standing in a neutral relaxed standing pose, "
    "facing forward, arms slightly away from sides, feet shoulder width apart, "
    "plain studio background, plain grey backdrop, professional reference photo, "
    "photorealistic, sharp focus, entire body visible from head to feet"
)
source_negative_prompt = "sitting, action pose, dynamic pose, cropped, close-up, multiple people, blurry"

source_image = sdxl_pipe(
    source_prompt,
    negative_prompt=source_negative_prompt,
    height=1024,
    width=768,
    num_inference_steps=30,
    generator=torch.Generator("cuda").manual_seed(7),
).images[0]

source_path = "canonical_pose_source.png"
source_image.save(source_path)

img = cv2.imread(source_path)
wholebody = Wholebody(mode="balanced", to_openpose=True, backend="onnxruntime", device="cpu")
keypoints, scores = wholebody(img)

# Blank canvas rather than drawing over the source photo, so the
# ControlNet preprocessor sees only the skeleton, not background content.
blank = np.zeros_like(img)
skeleton_img = draw_skeleton(blank, keypoints, scores, openpose_skeleton=True, kpt_thr=0.3)
cv2.imwrite("../assets/canonical_pose.png", skeleton_img)
print("saved to ../assets/canonical_pose.png")
