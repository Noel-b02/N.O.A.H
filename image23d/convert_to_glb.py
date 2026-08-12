import sys
import trimesh

# Converts a generated mesh to .glb purely for browser preview via
# <model-viewer> — Fusion's meshBodies.add() needs .obj/.stl/.3mf, but
# <model-viewer> needs glTF, so this runs as a separate step alongside the
# Fusion import rather than replacing the .obj output.
mesh_path = sys.argv[1]
output_path = sys.argv[2]

loaded = trimesh.load(mesh_path, process=False)
# Same as scale_mesh.py — export can be a multi-geometry Scene rather than
# a single Trimesh depending on the mesh format.
mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
mesh.export(output_path)
