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
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    watertight_after = bool(mesh.is_watertight)
    repaired = watertight_after and not watertight_before

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
