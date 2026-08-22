import json
import sys

import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton

# Extracts a live, per-request OpenPose-style pose skeleton from a
# reference image, for use as generate_pose_seed.py's ControlNet
# conditioning image instead of the fixed canonical_pose.png. Promotes
# image23d/tools/make_canonical_pose_asset.py's Step 2 (rtmlib.Wholebody +
# draw_skeleton) from a one-time offline asset-builder into a live script
# — identical model/params, so the two produce compatible skeleton images.
# device="cpu" (also unchanged from the offline tool): pose estimation is
# lightweight and this way it never contends with GPU-heavy steps under
# withGpuExclusive.
reference_path = sys.argv[1]
output_path = sys.argv[2]

# Below this confidence, a keypoint is too unreliable to trust — same
# number already used for skeleton rendering (kpt_thr below), not a new
# arbitrary threshold. A limb whose joints score low here is likely
# self-occluded in the reference image.
CONFIDENCE_THRESHOLD = 0.3

# A limb pair's bounding-box overlap above this fraction is treated as
# dangerously occluding/crossing — the actual root cause of the original
# detached-limb mesh failures (limbs overlapping/foreshortened in the seed
# image, which generate_novel_views.py's Zero123++ step then faithfully
# reproduces, and Hunyuan3D-2 can't cleanly separate back out). First-pass
# number, not yet empirically tuned — same posture as repair_mesh.py's
# ADJACENCY_MARGIN_FRACTION.
OVERLAP_THRESHOLD = 0.25

# openpose134 keypoint indices (rtmlib/visualization/skeleton/openpose134.py).
LIMBS = {
    "right arm": [2, 3, 4],   # shoulder, elbow, wrist
    "left arm": [5, 6, 7],
    "right leg": [8, 9, 10],  # hip, knee, ankle
    "left leg": [11, 12, 13],
}

img = cv2.imread(reference_path)

wholebody = Wholebody(mode="balanced", to_openpose=True, backend="onnxruntime", device="cpu")
keypoints, scores = wholebody(img)

# Most-confident detected person, in case the detector finds more than one
# (background noise, a second figure) — reference images are expected to
# show a single subject. If nobody is detected at all, scores is empty and
# argmax() raises naturally, which is the intended failure signal (non-zero
# exit, no output written) — no explicit guard needed.
person_idx = 0 if keypoints.shape[0] == 1 else int(scores.mean(axis=1).argmax())
person_keypoints = keypoints[person_idx]
person_scores = scores[person_idx]

# openpose134 indices 24-91 are a dense 68-point face contour mesh. On a
# reference with no real facial detail (e.g. a plain silhouette), the
# detector still emits something for these — low-confidence, noisy
# positions with no real face to anchor them. Confirmed live: rendering
# that noisy partial face contour into the conditioning image caused SDXL
# to draw an actual half-visible human face poking through a half-rendered
# mask, instead of a full mask, on a test generation — the canonical
# neutral-pose asset never has this problem because its source photo has a
# real, clean face for the detector to lock onto. Zeroing these scores
# before rendering drops them from the skeleton entirely (draw_openpose
# skips anything below kpt_thr), leaving nose/eyes/ears (part of the
# 18-point body set, not this face-mesh block) for head orientation
# without imposing a distorted face outline.
render_scores = scores.copy()
render_scores[person_idx, 24:92] = 0.0

blank = np.zeros_like(img)
skeleton_img = draw_skeleton(
    blank,
    keypoints[person_idx:person_idx + 1],
    render_scores[person_idx:person_idx + 1],
    openpose_skeleton=True,
    kpt_thr=CONFIDENCE_THRESHOLD,
)
cv2.imwrite(output_path, skeleton_img)

limb_confidences = {name: float(person_scores[idxs].mean()) for name, idxs in LIMBS.items()}
low_confidence_limbs = [name for name, conf in limb_confidences.items() if conf < CONFIDENCE_THRESHOLD]


def limb_bbox(idxs):
    pts = person_keypoints[idxs]
    conf = person_scores[idxs]
    visible = pts[conf >= CONFIDENCE_THRESHOLD]
    if len(visible) == 0:
        return None
    return visible[:, 0].min(), visible[:, 1].min(), visible[:, 0].max(), visible[:, 1].max()


def bbox_overlap_fraction(a, b):
    # Fraction of the SMALLER box's area, not IoU — a thin forearm's box
    # fully swallowed by a much larger torso/leg box should still trip
    # this, even though that reads as low IoU.
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller > 0 else 0.0


limb_names = list(LIMBS.keys())
bboxes = {name: limb_bbox(idxs) for name, idxs in LIMBS.items()}

max_pairwise_overlap = 0.0
overlapping_pair = None
for i in range(len(limb_names)):
    for j in range(i + 1, len(limb_names)):
        a, b = limb_names[i], limb_names[j]
        overlap = bbox_overlap_fraction(bboxes[a], bboxes[b])
        if overlap > max_pairwise_overlap:
            max_pairwise_overlap = overlap
            overlapping_pair = (a, b)

overlapping_limbs = overlapping_pair if max_pairwise_overlap > OVERLAP_THRESHOLD else None
usable = len(low_confidence_limbs) == 0 and overlapping_limbs is None

if low_confidence_limbs:
    reason = f"low-confidence/occluded {' and '.join(low_confidence_limbs)} (likely self-occluded)"
elif overlapping_limbs:
    reason = f"overlapping {overlapping_limbs[0]} and {overlapping_limbs[1]}"
else:
    reason = None

report = {
    "usable": usable,
    "reason": reason,
    "limb_confidences": limb_confidences,
    "max_pairwise_overlap": max_pairwise_overlap,
}
print(json.dumps(report))
