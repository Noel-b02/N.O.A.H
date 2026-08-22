import sys

import trimesh

# Single-mesh editing operations (scale-factor, mirror, simplify) that
# aren't already covered by scale_mesh.py (absolute target size) or
# repair_mesh.py (fill-holes) — those are reused unchanged from server.ts
# rather than duplicated here. Writes to a SEPARATE output file, same
# reasoning as scale_mesh.py: never overwrite the input, so a failed or
# partial edit can't corrupt the caller's only copy.
mesh_path = sys.argv[1]
output_path = sys.argv[2]
operation = sys.argv[3]

loaded = trimesh.load(mesh_path, process=False)
# Same as every other script here — export can be a multi-geometry Scene
# rather than a single Trimesh depending on the mesh format.
mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded

if operation == "scale-factor":
    factor = float(sys.argv[4])
    mesh.apply_scale(factor)
elif operation == "mirror":
    axis = sys.argv[4] if len(sys.argv) > 4 else "x"
    normal = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    mesh.apply_transform(trimesh.transformations.reflection_matrix(mesh.bounding_box.centroid, normal))
elif operation == "simplify":
    mesh = mesh.simplify_quadric_decimation(face_count=int(sys.argv[4]))
elif operation == "simplify-percent":
    percent = float(sys.argv[4])
    target = max(4, int(len(mesh.faces) * percent / 100))
    mesh = mesh.simplify_quadric_decimation(face_count=target)
else:
    raise ValueError(f"unknown edit_mesh operation: {operation}")

mesh.export(output_path)
