import json
import sys

import trimesh

# Validates and repairs a generated mesh before it's sliced for printing.
# Disconnected-component check MUST run before hole-filling: fill_holes()
# operates per-component, so filling holes first would silently "fix"
# watertightness on a mesh that still has a detached-limb-style stray
# blob attached to it, defeating the point of this check.
mesh_path = sys.argv[1]
output_path = sys.argv[2]
# Optional third arg. Voxel remeshing (see below) trades away real detail,
# so it's off by default — most meshes slice fine without it. Only pass
# "remesh" when the caller already knows this specific mesh needs it (i.e.
# a first slice attempt on the un-remeshed output actually failed).
force_remesh = len(sys.argv) > 3 and sys.argv[3] == "remesh"

# Fallback only — see the proximity check below, which is the primary
# signal now. Kept as a last resort for when NOTHING is spatially adjacent
# to the dominant piece (e.g. several truly separate fragments), in which
# case there's no anchor to judge adjacency against and volume share is
# the only signal left.
DOMINANT_COMPONENT_VOLUME_THRESHOLD = 0.85

# Confirmed live: judging components by volume share alone discarded a
# real, legitimate limb on a dynamic-pose figure — a "swinging" pose's
# extended arm/leg can render as topologically separate from the torso in
# raw diffusion mesh output even though it's clearly part of the same
# figure, and a thin limb is often a small fraction of total body volume
# (confirmed with a synthetic torso+limb test: the limb was ~1.6% of total
# volume, far below any volume threshold that would keep it). Proximity to
# the dominant piece is a better signal: a real disconnected limb sits
# touching or nearly touching the main body, while a genuinely unrelated
# stray blob (the original bug this whole check exists for — a mismatched
# multiview reference producing an extra, oddly-placed body part) sits
# well outside it. Margin is a fraction of the dominant piece's own size,
# not an absolute distance, so it scales with the mesh.
ADJACENCY_MARGIN_FRACTION = 0.05

# Components below this fraction of total volume are mesh-generation noise
# (degenerate slivers, near-zero-volume artifacts) and get dropped
# unconditionally, regardless of proximity — confirmed live: a real mesh
# had 17 near-zero-volume junk components (largest at ~0.04% of total
# volume) that the proximity check alone kept just because they sat within
# the main body's bounding box, showing up as visible stray geometry (a
# spurious extra head). A real limb was ~1.6% of total volume in earlier
# testing — this floor sits well below that and well above the noise
# ceiling seen here, so it only screens out genuine junk.
MIN_COMPONENT_VOLUME_FRACTION = 0.005

# Voxels along the longest axis when remeshing (see below) — balances
# print-relevant detail against face count/slicing speed. 128 (the first
# pass) was confirmed too coarse on real print-quality review. Tried 256
# next, which regenerated the ORIGINAL Bambu Studio CLI crash this whole
# voxel-remesh step exists to avoid (confirmed live, silent failure, no
# stderr — same symptom as the raw-mesh crash). Bisected empirically on a
# real dense mesh: 128/160/192 all sliced fine (60K/94K/134K faces), 224/256
# both crashed (181K/229K faces) — the real limit sits somewhere in that
# gap, face count not voxel resolution being the actual constraint. 192
# gives real headroom below the ~180K-face failure point (different
# subjects will produce different face counts at the same resolution,
# depending on shape complexity) while being a meaningful detail
# improvement over 128.
VOXEL_RESOLUTION = 192

loaded = trimesh.load(mesh_path, process=False)
# Same as scale_mesh.py — export can be a multi-geometry Scene rather than
# a single Trimesh depending on the mesh format.
mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded

watertight_before = bool(mesh.is_watertight)

components = mesh.split(only_watertight=False)
components_found = len(components)
component_dropped_count = 0
usable = True
reason = None

if components_found > 1:
    # abs() guards against a non-watertight component reporting negative
    # volume (trimesh sign convention depends on face winding).
    volumes = [abs(c.volume) for c in components]
    total_volume = sum(volumes)
    dominant_index = volumes.index(max(volumes))
    dominant = components[dominant_index]
    dominant_fraction = (volumes[dominant_index] / total_volume) if total_volume > 0 else 0

    margin = max(dominant.extents) * ADJACENCY_MARGIN_FRACTION if len(dominant.extents) else 0
    dom_min, dom_max = dominant.bounds[0] - margin, dominant.bounds[1] + margin

    kept = [dominant]
    for i, c in enumerate(components):
        if i == dominant_index:
            continue
        if total_volume > 0 and (volumes[i] / total_volume) < MIN_COMPONENT_VOLUME_FRACTION:
            continue  # noise floor — too small to be a real body part regardless of position
        c_min, c_max = c.bounds
        overlaps = all(c_min[ax] <= dom_max[ax] and c_max[ax] >= dom_min[ax] for ax in range(3))
        if overlaps:
            kept.append(c)

    component_dropped_count = components_found - len(kept)

    if len(kept) > 1:
        # At least one non-dominant piece is spatially part of the figure
        # — keep it regardless of its individual volume share, since a
        # thin limb legitimately has little volume next to a torso.
        mesh = trimesh.util.concatenate(kept)
    elif dominant_fraction >= DOMINANT_COMPONENT_VOLUME_THRESHOLD:
        # Nothing was adjacent to anchor against — fall back to the old
        # volume-share check as a last resort.
        mesh = dominant
    else:
        usable = False
        reason = (
            f"dominant component is only {dominant_fraction:.0%} of total volume "
            f"across {components_found} disconnected pieces, none adjacent to it "
            "— too fragmented to trust"
        )

repaired = False
watertight_after = watertight_before

remeshed = False

if usable:
    if force_remesh:
        # Confirmed live against a real Hunyuan3D-2 output: fix_normals() +
        # fill_holes() alone can produce a mesh that reports is_watertight,
        # is_winding_consistent, AND is_volume all True, yet still crashes
        # Bambu Studio's CLI slicer outright with zero error output. The
        # giveaway was euler_number being far from the expected 2 for a
        # simple closed shape — a sign of self-intersecting/tunneling
        # geometry those checks don't catch. Voxelizing and rebuilding via
        # marching cubes guarantees a clean, non-self-intersecting manifold
        # surface, which is what actually fixed it in testing. Real
        # trade-off (confirmed live: visibly blockier even at a resolution
        # tuned right up against the CLI's face-count crash threshold) —
        # only worth paying when a first slice attempt on the un-remeshed
        # mesh actually failed, which is why this is opt-in via force_remesh
        # rather than applied to every mesh unconditionally.
        extents = mesh.extents
        max_extent = max(extents) if len(extents) else 0
        if max_extent > 0:
            pitch = max_extent / VOXEL_RESOLUTION
            mesh = mesh.voxelized(pitch=pitch).fill().marching_cubes
        remeshed = True
    else:
        trimesh.repair.fill_holes(mesh)

    trimesh.repair.fix_normals(mesh)
    watertight_after = bool(mesh.is_watertight)
    repaired = True

    mesh.export(output_path)

report = {
    "watertight_before": watertight_before,
    "watertight_after": watertight_after,
    "components_found": components_found,
    "component_dropped_count": component_dropped_count,
    "repaired": repaired,
    "remeshed": remeshed,
    "usable": usable,
    "reason": reason,
}
print(json.dumps(report))
