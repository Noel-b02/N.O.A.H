import sys
import trimesh

# Scales a mesh so its largest bounding-box dimension matches the requested
# size, writing to a SEPARATE output file rather than overwriting the input
# — if this step fails partway through, the original mesh must stay intact
# so the caller can still fall back to importing it unscaled.
mesh_path = sys.argv[1]
target_size_cm = float(sys.argv[2])
output_path = sys.argv[3]

loaded = trimesh.load(mesh_path, process=False)
# TripoSR's own export can be a multi-geometry Scene rather than a single
# Trimesh depending on the mesh format — collapse it to one mesh either way.
mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded

current_max = max(mesh.bounding_box.extents)
if current_max > 0:
    mesh.apply_scale(target_size_cm / current_max)

mesh.export(output_path)
