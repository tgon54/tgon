"""Convert motion.bvh to FBX with Blender (no GUI):
    blender -b -P blender_bvh_to_fbx.py -- OUT_DIR/motion.bvh OUT_DIR/motion.fbx
"""
import sys
import bpy

argv = sys.argv[sys.argv.index('--') + 1:]
src, dst = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_anim.bvh(filepath=src, global_scale=0.01, rotate_mode='NATIVE',
                        axis_forward='-Z', axis_up='Y', update_scene_fps=True, update_scene_duration=True)
bpy.ops.export_scene.fbx(filepath=dst, use_selection=False, add_leaf_bones=False, bake_anim=True,
                         axis_forward='-Z', axis_up='Y')
print('FBX written:', dst)
