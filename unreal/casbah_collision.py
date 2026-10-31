"""Post-import collision setup for a Casbah GLB in Unreal Engine 5.4+.

Run from the editor (Tools > Execute Python Script) after importing the .glb.
Set IMPORT_FOLDER to the Content Browser folder that holds the imported static meshes.

- Bldg_###  : keep the imported UCX_ convex hulls as simple collision (falls back to
              complex-as-simple if no hulls came through).
- Sea_*     : no collision.
- everything else (terrain, streets, stairs, gutters, pads, walls, ramparts):
              per-triangle collision (complex as simple), so the auto-generated box
              hull does not cover the streets.
"""
import unreal

IMPORT_FOLDER = "/Game/Casbah"

sm_sub = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
COMPLEX = unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE
SIMPLE_ONLY = unreal.CollisionTraceFlag.CTF_USE_SIMPLE_AS_COMPLEX
DEFAULT = unreal.CollisionTraceFlag.CTF_USE_DEFAULT

stats = {"houses_ucx": 0, "houses_fallback": 0, "complex": 0, "none": 0}
for path in unreal.EditorAssetLibrary.list_assets(IMPORT_FOLDER, recursive=True):
    mesh = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(mesh, unreal.StaticMesh):
        continue
    name = mesh.get_name()
    body = mesh.get_editor_property("body_setup")
    if body is None:
        continue
    mesh.modify()
    if name.startswith("Bldg_"):
        if sm_sub.get_convex_collision_count(mesh) > 0:
            body.set_editor_property("collision_trace_flag", DEFAULT)
            stats["houses_ucx"] += 1
        else:
            body.set_editor_property("collision_trace_flag", COMPLEX)
            stats["houses_fallback"] += 1
    elif name.startswith("Sea"):
        sm_sub.remove_collisions(mesh)
        body.set_editor_property("collision_trace_flag", SIMPLE_ONLY)
        stats["none"] += 1
    else:
        sm_sub.remove_collisions(mesh)
        body.set_editor_property("collision_trace_flag", COMPLEX)
        stats["complex"] += 1
    unreal.EditorAssetLibrary.save_loaded_asset(mesh)

unreal.log("Casbah collision: {}".format(stats))
