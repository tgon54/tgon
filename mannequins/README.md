# Mannequins folder

Drop reference characters here (binary `.fbx`, or a `.json` rig). Every file becomes an
export target named after the file, next to the built-in `ue4` / `ue5` rigs:

* the pipeline writes `motion_<FileName>.fbx` for each one,
* the Timeline Cleaner lists them under **Export to Unreal** (press *Rescan* after adding files),
* CLI: `python -m aimocap.unreal --list`, `python -m aimocap.unreal OUT --target <FileName>` or `--target all`.

Requirements: Unreal (Epic) bone naming — `pelvis`, `spine_0x`, `neck_0x`, `head`, `clavicle_l/r`,
`upperarm`, `lowerarm`, `hand`, `thigh`, `calf`, `foot`, `ball` (+ `index_01_l` … fingers, optional).
UE4 Mannequin, UE5 Manny/Quinn and MetaHuman bodies work; the character's own rest pose and
proportions are used, so the motion fits that body.
Import the result in Unreal on the matching skeleton and untick *Import Mesh*.

Epic mannequin files are not committed to the public repository (`*.fbx` here is git-ignored);
keep your copies locally.
