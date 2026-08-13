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

# First-pass heuristic — not derived from real failure data yet. Tune once
# tested against an actual bad multiview output.
DOMINANT_COMPONENT_VOLUME_THRESHOLD = 0.85

# Voxels along the longest axis when remeshing (see below) — balances
# print-relevant detail against face count/slicing speed. Not tuned
# against real print-quality comparisons yet, just against "does the
# slicer accept it."
VOXEL_RESOLUTION = 128

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
    dominant_fraction = (volumes[dominant_index] / total_volume) if total_volume > 0 else 0

    if dominant_fraction >= DOMINANT_COMPONENT_VOLUME_THRESHOLD:
        mesh = components[dominant_index]
        component_dropped_count = components_found - 1
    else:
        usable = False
        reason = (
            f"dominant component is only {dominant_fraction:.0%} of total volume "
            f"across {components_found} disconnected pieces — too fragmented to trust"
        )

repaired = False
watertight_after = watertight_before

if usable:
    # Confirmed live against a real Hunyuan3D-2 output: fix_normals() +
    # fill_holes() alone can produce a mesh that reports is_watertight,
    # is_winding_consistent, AND is_volume all True, yet still crashes
    # Bambu Studio's CLI slicer outright with zero error output. The
    # giveaway was euler_number being far from the expected 2 for a
    # simple closed shape — a sign of self-intersecting/tunneling
    # geometry that those checks don't catch. Voxelizing and rebuilding
    # via marching cubes guarantees a clean, non-self-intersecting
    # manifold surface, which is what actually fixed it in testing.
    # Trade-off: this discards fine surface detail in favor of a
    # blocky/rounded approximation — an acceptable cost since that level
    # of detail usually wouldn't print reliably on FDM anyway.
    extents = mesh.extents
    max_extent = max(extents) if len(extents) else 0
    if max_extent > 0:
        pitch = max_extent / VOXEL_RESOLUTION
        mesh = mesh.voxelized(pitch=pitch).fill().marching_cubes

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
    "usable": usable,
    "reason": reason,
}
print(json.dumps(report))
