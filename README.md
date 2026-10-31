# AI-MoCap

Local tool that turns one AI-generated multi-view video (e.g. a synchronized 2×2 grid) into corrected 3D motion,
ready for games and Unreal Engine. Arabic guide: [README_AR.md](README_AR.md).

## Quick start (Windows)

| Launcher | What it does |
|---|---|
| `install.bat` | installs dependencies (GPU if CUDA 12 + cuDNN 9, otherwise CPU) |
| `start_ui.bat` | web UI at `http://127.0.0.1:8642` — upload video, process, view, download |
| `start_cleaner.bat` | **Timeline Cleaner** — clean the motion and export FBX to Unreal |
| `run.bat` | command-line processing of one video |
| `export_unreal.bat` | export an output folder to UE4 + UE5 FBX |

Command line:

```bash
python -m aimocap my_ai_video.mp4 out --height 1.75 [--grid 2x2] [--no-hands]
python -m aimocap.unreal out --target ue4 [--root-motion]      # or ue5, or a .fbx skeleton with UE bone names
python -m aimocap.timeline out --auto 0.5 --target ue4 --target ue5
python test_unreal.py                                          # self-check of the Unreal export
```

## Unreal Engine export (UE4 / UE5)

Every run writes `motion_UE4.fbx` (UE4 Mannequin skeleton) and `motion_UE5.fbx` (UE5 Manny skeleton).
The animation is baked directly on the Unreal bone names and rest pose (`root`, `pelvis`, `spine_0x`,
`clavicle`, `upperarm`, `lowerarm`, `hand`, all finger bones, `thigh`, `calf`, `foot`, `ball`, plus `ik_*`
bones following the hands/feet), so no Blender step and no retargeting is needed:

1. In Unreal: **Import** → select the FBX.
2. **Skeleton** = `UE4_Mannequin_Skeleton` (UE5: `SK_Mannequin`), untick **Import Mesh**.
3. Optional `--root-motion`: the `root` bone follows the pelvis on the ground (enable *Root Motion* in the
   animation asset).

The FBX writer is pure Python (`aimocap/fbx_io.py`); the bundled rigs in `aimocap/rigs/` contain only
bone names and rest transforms. Any skeleton using UE naming can be targeted with `--target path/to/rig.fbx`.

## Smart hand motion

* 21 keypoints per hand are detected (RTMPose hand model, downloaded on first run) in crops placed from each
  view's elbow→wrist, follow the body's left/right swap fixes, are triangulated with the self-calibrated
  cameras and attached to the corrected wrists.
* At export, the palm drives the `hand` bone (wrist range limited, half of the wrist twist goes into the forearm
  twist bone) and each finger is solved as anatomical angles (MCP flex/spread, PIP, DIP) with joint limits.
  The fingertip falls back to natural DIP–PIP coupling when uncertain.
* Hands in AI video are small: every finger is blended per frame with a natural relaxed hand according to its
  tracking confidence, then smoothed — no broken or jittering fingers. Without hand data you still get the
  relaxed hand instead of a stiff rest pose.

## Timeline Cleaner

`http://127.0.0.1:8642/cleaner` (or the button on a finished job). Timeline lanes show detected problems —
spikes/glitches, jitter, low tracking quality, foot skating, lost hand tracking — plus foot contacts, your edits
and a speed curve (original vs cleaned). Drag to select frames, then:

* **Fix glitch** (rebuild from neighbouring frames), **Smooth** (strength), **Hold pose**, **Foot lock**,
  **Relax hands**, **Trim to selection**, **Loop blend**, **Auto-clean**, one-click **Fix** per issue.
* Edits are non-destructive (re-applied to the original motion, undo/redo), bone lengths are re-enforced on
  edited frames, and the 3D preview shows the original as a ghost.
* **Export FBX** writes `<name>_UE4.fbx` / `<name>_UE5.fbx`, `motion_clean.json`, `motion_clean.bvh`
  and `clean_edits.json` in the job's output folder.
* A `motion.json` produced by the command line can be imported through **Open…**.

## Outputs

`motion_UE4.fbx`, `motion_UE5.fbx`, `motion.bvh`, `motion.json` (body + hands), `viewer.html`,
`report.html/json`, `body_3d_xyz_m.csv/npy`, `hand_left|right_3d_xyz_m.npy`, `angles_com_contacts.csv`,
`virtual_cameras_calibration.toml`, `debug_2d_overlay.mp4`, `detections_2d.npz` (cached detections incl. hands).
