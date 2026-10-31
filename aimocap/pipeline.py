"""End-to-end pipeline: AI multi-view video -> corrected 3D motion + exports + report."""
import os
import json
import time
import numpy as np

from . import cameras as C
from . import correct as K
from .skeleton import NAMES, BONES, J, FLIP, center_of_mass, joint_angles, IDX


DEFAULTS = dict(
    yaws=None,            # e.g. [0,90,180,270]; None = auto (tests both rotation directions)
    fov=40.0,             # initial guess, refined by bundle adjustment
    height=1.75,          # subject height in meters (sets absolute scale)
    cutoff=6.0,           # Butterworth cutoff Hz
    reproj_thr=0.03,      # outlier threshold (fraction of panel height)
    min_conf=0.3,
    ba_frames=80,
    swap_lambda=0.02,
)


def _sample_frames(conf, n):
    T = conf.shape[1]
    score = conf.mean((0, 2))
    idx = np.argsort(-score)[: max(n * 3, n)]
    idx = np.sort(idx)
    if len(idx) > n:
        idx = idx[np.linspace(0, len(idx) - 1, n).astype(int)]
    return idx


def calibrate(kp, conf, sizes, o, log=print):
    best = None
    for yaws in C.candidate_yaw_sets(len(sizes), o['yaws']):
        cams = C.nominal_cameras(sizes, yaws, kp, conf, fov_deg=o['fov'])
        log(f'  testing camera layout yaws={[round(y) for y in yaws]}')
        err = None
        for r in range(2):
            uv, cf, _ = C.resolve_swaps(cams, kp, conf, lam=o['swap_lambda'], min_conf=o['min_conf'])
            fr = _sample_frames(cf, o['ba_frames'])
            cams2, err2 = C.bundle_adjust(cams, uv, cf, fr, log=log)
            if err2 is None:
                break
            cams, err = cams2, err2
        if err is None:
            err = 1e9
        log(f'    -> score {err:.5f}')
        if best is None or err < best[0]:
            best = (err, cams, yaws)
    best[1].yaws = best[2]
    return best[1], best[0]


def reconstruct(kp, conf, sizes, fps, opts=None, log=print):
    o = dict(DEFAULTS, **(opts or {}))
    V, T = kp.shape[:2]
    rep = {'views': V, 'frames': T, 'fps': fps, 'options': {k: v for k, v in o.items()}}
    t0 = time.time()

    # 1) 2D AI-glitch screening
    conf_g, gflags = K.flag_2d_glitches(kp, conf)
    rep['glitch_view_frames'] = int(gflags.sum())
    rep['glitch_keypoints'] = int(((conf > o['min_conf']) & (conf_g == 0)).sum())
    log(f'  AI glitch screening: {rep["glitch_keypoints"]} keypoints, {rep["glitch_view_frames"]} whole view-frames rejected')

    # 2) self-calibration of virtual cameras
    cams, score = calibrate(kp, conf_g, sizes, o, log)
    rep['calibration_score'] = float(score)

    # 3) left/right swap resolution with refined cameras
    uv, cf, dec = C.resolve_swaps(cams, kp, conf_g, lam=o['swap_lambda'], min_conf=o['min_conf'])
    rep['swaps'] = {g: d.sum(0).tolist() for g, d in dec.items()}
    log(f'  L/R swaps fixed per view: {rep["swaps"]}')

    # 4) robust triangulation + inconsistent-view rejection
    X, err, used, q = C.robust_triangulate(cams, uv, cf, thr=o['reproj_thr'], min_conf=o['min_conf'])
    badv = K.flag_inconsistent_views(err, used, o['reproj_thr'])
    rep['inconsistent_view_frames'] = int(badv.sum())
    if badv.any():
        cf = np.where(badv[..., None], 0, cf)
        X, err, used, q = C.robust_triangulate(cams, uv, cf, thr=o['reproj_thr'], min_conf=o['min_conf'])
    e_used = np.where(used, err, np.nan)
    rep['reproj_err_median_per_view'] = [float(np.nanmedian(e_used[v])) if np.isfinite(e_used[v]).any() else None for v in range(V)]
    rep['views_used_mean'] = float(used.sum(0).mean())

    # baseline ("raw"): nominal cameras, no screening, no swap fix, plain DLT
    cam_nom = C.nominal_cameras(sizes, cams.yaws, kp, conf, fov_deg=o['fov'])
    Xraw = C.triangulate(cam_nom, kp, np.where(conf > o['min_conf'], conf, 0))

    # 5) temporal: gaps, spikes, zero-phase low-pass
    Xf, filled = K.fill_gaps(X, q)
    rep['filled_samples_pct'] = float(filled.mean() * 100)
    Xf, nsp = K.hampel(Xf)
    rep['spikes_removed'] = nsp
    Xf = K.butter_lowpass(Xf, fps, o['cutoff'])

    # 6) world: mirror check, ground plane, facing, metric scale
    Xf, mir = K.fix_global_mirror(Xf)
    if mir:
        q = q[:, FLIP]
    rep['global_mirror_fixed'] = bool(mir)
    Xw, scale = K.align_world(Xf, o['height'])
    rep['scale'] = float(scale)
    stage_tri = Xw.copy()

    # 7) anatomy
    L = K.bone_targets(Xw, q)
    Xa = K.enforce_bones(Xw, L, q)
    Xa, nk = K.fix_knee_hyperextension(Xa)
    rep['knee_hyperextension_fixed'] = nk

    # 8) physics
    cont = K.foot_contacts(Xa, fps)
    Xp, pinned = K.fix_foot_skating(Xa, cont)
    Xp = K.enforce_bones(Xp, L, q, iters=15, heavy=pinned)
    Xp, npen = K.clamp_ground(Xp)
    rep['ground_clamped_samples'] = npen

    # raw aligned for comparison
    Xr, _ = K.fill_gaps(Xraw, np.isfinite(Xraw).all(-1).astype(float))
    Xr, _ = K.fix_global_mirror(Xr)
    try:
        Xr, _ = K.align_world(Xr, o['height'])
    except Exception:
        pass

    rep['metrics'] = {
        'raw': K.metrics(Xr, fps, cont),
        'triangulated': K.metrics(stage_tri, fps, cont),
        'final': K.metrics(Xp, fps, cont),
    }
    rep['contact_pct'] = {s: float(m.mean() * 100) for s, m in cont.items()}
    rep['bone_lengths_m'] = {f'{a}-{b}': float(l) for (a, b), l in zip(BONES, L)}
    rep['seconds'] = round(time.time() - t0, 1)
    log(f'  metrics: {json.dumps(rep["metrics"])}')

    return dict(final=Xp, raw=Xr, triangulated=stage_tri, quality=q, contacts=cont,
                cameras=cams, err=err, used=used, gflags=gflags, swaps=dec, report=rep,
                com=center_of_mass(Xp), angles=joint_angles(Xp), fps=fps)


# ----------------------------------------------------------------- export --
def export_all(res, out, log=print):
    from .viewer import write_viewer
    from .bvh import write_bvh
    from .report import write_report
    os.makedirs(out, exist_ok=True)
    P, fps, T = res['final'], res['fps'], len(res['final'])
    t = np.arange(T) / fps
    # scientific arrays
    np.save(os.path.join(out, 'body_3d_xyz_m.npy'), P)
    np.save(os.path.join(out, 'body_3d_raw_xyz_m.npy'), res['raw'])
    np.save(os.path.join(out, 'body_3d_quality.npy'), res['quality'])
    np.save(os.path.join(out, 'center_of_mass_xyz_m.npy'), res['com'])
    hdr = ['frame', 'time_s'] + [f'{n}_{a}' for n in NAMES for a in 'xyz']
    rows = np.concatenate([np.arange(T)[:, None], t[:, None], P.reshape(T, -1)], 1)
    np.savetxt(os.path.join(out, 'body_3d_xyz_m.csv'), rows, delimiter=',', header=','.join(hdr), comments='', fmt='%.5f')
    ang = res['angles']
    hdr = ['frame', 'time_s'] + list(ang) + ['com_x', 'com_y', 'com_z', 'contact_left', 'contact_right']
    rows = np.column_stack([np.arange(T), t] + [ang[k] for k in ang] + [res['com'], res['contacts']['left'], res['contacts']['right']])
    np.savetxt(os.path.join(out, 'angles_com_contacts.csv'), rows, delimiter=',', header=','.join(hdr), comments='', fmt='%.4f')
    # game / animation
    write_bvh(P, fps, os.path.join(out, 'motion.bvh'))
    res['cameras'].to_anipose_toml(os.path.join(out, 'virtual_cameras_calibration.toml'))
    r = lambda a: np.round(a, 4).tolist()
    data = dict(fps=fps, joints=NAMES, bones=[list(b) for b in BONES], corrected=r(P), raw=r(res['raw']),
                quality=np.round(res['quality'], 2).tolist(), com=r(res['com']),
                contacts={k: v.astype(int).tolist() for k, v in res['contacts'].items()})
    json.dump(dict(data, raw=None), open(os.path.join(out, 'motion.json'), 'w'), separators=(',', ':'))
    write_viewer(data, os.path.join(out, 'viewer.html'))
    json.dump(res['report'], open(os.path.join(out, 'report.json'), 'w'), indent=2, default=float)
    write_report(res, os.path.join(out, 'report.html'))
    log('  exported: motion.bvh, motion.json, viewer.html, report.html, CSV/NPY, calibration TOML')


def run(video, out, grid='auto', panels=None, opts=None, mode='performance', device='auto',
        max_frames=None, log=print):
    from . import video as Vd
    import cv2
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, 'detections_2d.npz')
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        kp, conf, fps, sizes = d['kp'], d['conf'], float(d['fps']), [tuple(s) for s in d['sizes']]
        log(f'loaded cached 2D detections ({cache})')
    else:
        cap = cv2.VideoCapture(video); W = int(cap.get(3)); H = int(cap.get(4)); cap.release()
        if isinstance(grid, (tuple, list)):
            xb, yb = [int(v) for v in grid[0]], [int(v) for v in grid[1]]
        elif grid != 'auto':
            xb, yb = Vd.grid_from_spec(grid, W, H)
        else:
            xb, yb = Vd.detect_grid(video)
        pans = Vd.panels_from_grid(xb, yb)
        use = panels if panels is not None else list(range(len(pans)))
        log(f'grid columns={xb} rows={yb} -> {len(pans)} panels, using {use}')
        if len(use) < 2:
            raise SystemExit('Need at least 2 views. Use --grid RxC if auto-detection failed.')
        det = Vd.Detector(mode=mode, device=device)
        log(f'2D detector: RTMPose Halpe-26 ({mode}) on {det.device}')
        kp, conf, fps, sizes = Vd.detect_video(video, pans, use, det, out, max_frames, log=log)
        np.savez(cache, kp=kp, conf=conf, fps=fps, sizes=np.array(sizes))
    log('3D reconstruction + smart correction...')
    res = reconstruct(kp, conf, sizes, fps, opts, log)
    export_all(res, out, log)
    return res
