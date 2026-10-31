"""Single-camera 3D (any direction: front, side, back or 3/4 view).

RTMW3D (COCO-WholeBody 133 points with per-joint depth) runs on one view. The per-joint depth
is combined with the 2D image positions through a per-frame metric scale solved from bone-length
consistency (a single camera cannot see absolute size), then the same smart correction,
anthropometric priors and physics pass as the multi-camera pipeline are applied. Hands
(21 points each) come from the same model."""
import time

import numpy as np

from . import correct as K
from .skeleton import NAMES, BONES, BONE_IDX, IDX, J, FOOT, center_of_mass, joint_angles

# COCO-WholeBody indices
WB_BODY = list(range(17))                     # same order as Halpe 0..16
WB_FEET = [17, 20, 18, 21, 19, 22]            # -> Halpe 20..25 (big toes, small toes, heels)
WB_CHIN = 23 + 8
WB_LHAND, WB_RHAND = list(range(91, 112)), list(range(112, 133))
W3D_MODEL = ('https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/'
             'rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx')


class MonoDetector:
    """RTMW3D whole-body 3D (YOLOX person detector + RTMPose3d)."""

    def __init__(self, device='auto'):
        try:
            from rtmlib import Wholebody3d
        except ImportError as e:
            raise RuntimeError('rtmlib >= 0.0.13 is required for single-camera mode: run install.bat (or pip install -U --no-deps rtmlib)  [' + str(e) + ']')
        dev = 'cuda' if device in ('auto', 'cuda') else 'cpu'
        try:
            import onnxruntime as ort
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                dev = 'cpu'
        except Exception:
            dev = 'cpu'
        self.model = Wholebody3d(mode='balanced', backend='onnxruntime', device=dev)
        self.device = dev

    def __call__(self, img):
        k3, sc, _, k2 = self.model(img)
        k3 = np.asarray(k3, float).reshape(-1, 133, 3)
        k2 = np.asarray(k2, float).reshape(-1, 133, 2)
        sc = np.asarray(sc, float).reshape(-1, 133)
        return k2, k3[..., 2], sc


def detect_video_mono(video, panel, det, out_dir=None, max_frames=None, log=print):
    """-> uv (T,133,2) panel px, z (T,133), score (T,133), fps, (w,h)."""
    import cv2
    import os
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    x0, y0, x1, y1 = panel
    UV, Z, S = [], [], []
    prev = None
    writer = None
    t = 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and t >= max_frames):
            break
        crop = fr[y0:y1, x0:x1]
        k2, z, sc = det(crop)
        if len(k2):
            if prev is not None:
                d = [np.nanmedian(np.linalg.norm(k[:17] - prev, axis=1)) for k in k2]
                i = int(np.argmin(d))
            else:
                i = int(np.argmax([np.ptp(k[:17, 0]) * np.ptp(k[:17, 1]) * s[:17].mean() for k, s in zip(k2, sc)]))
            prev = k2[i, :17]
            UV.append(k2[i]); Z.append(z[i]); S.append(sc[i])
        else:
            UV.append(np.full((133, 2), np.nan)); Z.append(np.full(133, np.nan)); S.append(np.zeros(133))
        if out_dir is not None:
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(out_dir, 'debug_2d_overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (crop.shape[1], crop.shape[0]))
            vis = crop.copy()
            for (u, v), s in zip(UV[-1], S[-1]):
                if s > 0.3 and np.isfinite(u):
                    cv2.circle(vis, (int(u), int(v)), 2, (0, 255, 0), -1)
            writer.write(vis)
        t += 1
        if t % 10 == 0:
            log(f'  2D detection: frame {t}/{total}')
    cap.release()
    if writer is not None:
        writer.release()
    return np.array(UV), np.array(Z), np.array(S), fps, (x1 - x0, y1 - y0)


# ------------------------------------------------------------------ lifting --
def to_halpe(uv, z, sc):
    """WholeBody 133 -> Halpe-26 (+ derived head top / neck / hip). uv (T,133,2), z, sc (T,133)."""
    T = len(uv)
    U = np.zeros((T, J, 2)); Zh = np.zeros((T, J)); Sh = np.zeros((T, J))
    U[:, :17], Zh[:, :17], Sh[:, :17] = uv[:, WB_BODY], z[:, WB_BODY], sc[:, WB_BODY]
    U[:, 20:26], Zh[:, 20:26], Sh[:, 20:26] = uv[:, WB_FEET], z[:, WB_FEET], sc[:, WB_FEET]
    for n, (a, b) in (('neck', (5, 6)), ('hip', (11, 12))):
        U[:, IDX[n]] = 0.5 * (uv[:, a] + uv[:, b]); Zh[:, IDX[n]] = 0.5 * (z[:, a] + z[:, b])
        Sh[:, IDX[n]] = np.minimum(sc[:, a], sc[:, b])
    em_u, em_z = 0.5 * (uv[:, 3] + uv[:, 4]), 0.5 * (z[:, 3] + z[:, 4])
    U[:, IDX['head']] = em_u + (em_u - uv[:, WB_CHIN])
    Zh[:, IDX['head']] = em_z + (em_z - z[:, WB_CHIN])
    Sh[:, IDX['head']] = np.minimum(np.minimum(sc[:, 3], sc[:, 4]), sc[:, WB_CHIN])
    return U, Zh, Sh


def _lift(U, Zr, m, sz, f, c):
    """Camera-space (x right, y down, z forward) metres*k from image px + model depth."""
    hip = U[:, IDX['hip']]
    root = np.column_stack([(hip[:, 0] - c[0]) * m, (hip[:, 1] - c[1]) * m, f * m])
    rel = np.concatenate([(U - hip[:, None]) * m[:, None, None],
                          (sz * (Zr - Zr[:, IDX['hip']][:, None]))[..., None]], -1)
    return root[:, None] + rel


def solve_scale(U, S, fps, height=1.75):
    """Per-frame metres per image px. RTMW3D depth is metric (verified against 4-camera
    reconstructions: best depth gain ~1.0), so the in-plane scale comes from the person's
    apparent height (rolling 2 s maximum: bending/crouching does not shrink the person)."""
    from scipy.ndimage import maximum_filter1d
    T = len(U)
    y = np.where(S > 0.3, U[..., 1], np.nan)
    with np.errstate(all='ignore'):
        ext = np.nanmax(y, 1) - np.nanmin(y, 1)
    t = np.arange(T)
    ok = np.isfinite(ext) & (ext > 10)
    ext = np.interp(t, t[ok], ext[ok]) if ok.sum() >= 2 else np.full(T, 400.0)
    ext = maximum_filter1d(ext, max(3, int(round(2 * fps)) | 1), mode='nearest')
    lm = np.log(height / ext)
    if T > 15:
        lm = K.butter_lowpass(lm[:, None], fps, 0.5)[:, 0]
    return np.exp(lm)


# ------------------------------------------------------------ reconstruction --
def reconstruct_mono(uv133, z133, sc133, fps, size, opts=None, log=print):
    from .pipeline import DEFAULTS, _physics
    from .physics import anthropometric_lengths
    from .hands import finalize_hands, _similarity
    o = dict(DEFAULTS, **(opts or {}))
    t0 = time.time()
    U, Zr, S = to_halpe(uv133, z133, sc133)
    T = len(U)
    rep = {'views': 1, 'frames': T, 'fps': fps, 'mode': 'single_view',
           'options': {k: v for k, v in o.items()}}
    # 2D AI-glitch screening (same detector as multi-view, one view)
    S_g, gflags = K.flag_2d_glitches(U[None], S[None])
    S_g = S_g[0]
    rep['glitch_view_frames'] = int(gflags.sum())
    rep['glitch_keypoints'] = int(((S > o['min_conf']) & (S_g == 0)).sum())
    m = solve_scale(U, S_g, fps, o['height'])
    sz = 1.0                                   # RTMW3D: larger z = farther from the camera
    w, h = size
    f = 0.5 * h / np.tan(np.radians(o['fov']) / 2)
    Xc = _lift(U, Zr, m, sz, f, (w / 2, h / 2)) * np.array([1, -1, -1])       # camera, y up
    q = np.clip(S_g, 0, 1)
    q = np.where(np.isfinite(Xc).all(-1), q, 0)
    Xraw = Xc.copy()
    # temporal: gaps, spikes, zero-phase low-pass
    Xf, filled = K.fill_gaps(Xc, q)
    rep['filled_samples_pct'] = float(filled.mean() * 100)
    Xf, rep['spikes_removed'] = K.hampel(Xf)
    Xf = K.butter_lowpass(Xf, fps, o['cutoff'])
    Xf, mir = K.fix_global_mirror(Xf)
    if mir:
        from .skeleton import FLIP
        q = q[:, FLIP]
    rep['global_mirror_fixed'] = bool(mir)
    Xw, scale = K.align_world(Xf, o['height'])
    rep['scale'] = float(scale)
    stage = Xw.copy()
    L = K.bone_targets(Xw, q)
    L, rep['anthropometric_clamped_bones'] = anthropometric_lengths(L, BONES, o['height'], tol=o.get('anthro_tol', 0.12))
    Xa = K.enforce_bones(Xw, L, q)
    Xa, rep['knee_hyperextension_fixed'] = K.fix_knee_hyperextension(Xa)
    cont = K.foot_contacts(Xa, fps)
    Xp, pinned = K.fix_foot_skating(Xa, cont)
    Xp = K.enforce_bones(Xp, L, q, iters=15, heavy=pinned)
    Xp, rep['ground_clamped_samples'] = K.clamp_ground(Xp)
    Xp, F, air = _physics(Xp, cont, L, q, fps, o, rep, log)
    # hands: same lifting, carried to world by the body's camera->world similarity
    hands = None
    if o.get('hands', True):
        Hc = []
        for idx in (WB_LHAND, WB_RHAND):
            hu = uv133[:, idx]; hz = z133[:, idx]
            hip = U[:, IDX['hip']]
            root = np.column_stack([(hip[:, 0] - w / 2) * m, (hip[:, 1] - h / 2) * m, f * m])
            rel = np.concatenate([(hu - hip[:, None]) * m[:, None, None],
                                  (sz * (hz - Zr[:, IDX['hip']][:, None]))[..., None]], -1)
            Hc.append((root[:, None] + rel) * np.array([1, -1, -1]))
        Hc = np.stack(Hc, 1)                                                   # (T,2,21,3)
        if mir:
            Hc = Hc[:, ::-1]
        A, B = Xc.reshape(-1, 3), Xw.reshape(-1, 3)
        okp = np.isfinite(A).all(-1) & np.isfinite(B).all(-1)
        s_, R_, t_ = _similarity(A[okp], B[okp])
        Hw = s_ * Hc @ R_.T + t_
        hq = np.stack([sc133[:, WB_LHAND], sc133[:, WB_RHAND]], 1)
        if mir:
            hq = hq[:, ::-1]
        hands = finalize_hands(Hw, hq, fps, Xp)
        rep['hands_tracked_pct'] = {s: (float((np.median(hands['conf_' + s], 1) > 0.5).mean() * 100)
                                        if hands[s] is not None else 0.0) for s in ('left', 'right')}
    Xr, _ = K.fill_gaps(Xraw, np.isfinite(Xraw).all(-1).astype(float))
    try:
        Xr, _ = K.align_world(K.fix_global_mirror(Xr)[0], o['height'])
    except Exception:
        pass
    rep['metrics'] = {'raw': K.metrics(Xr, fps, cont), 'triangulated': K.metrics(stage, fps, cont),
                      'final': K.metrics(Xp, fps, cont)}
    rep['contact_pct'] = {s_: float(v.mean() * 100) for s_, v in cont.items()}
    rep['bone_lengths_m'] = {f'{a}-{b}': float(l) for (a, b), l in zip(BONES, L)}
    rep.update(swaps={}, inconsistent_view_frames=0, reproj_err_median_per_view=[None], views_used_mean=1.0,
               seconds=round(time.time() - t0, 1))
    log(f'  metrics: {rep["metrics"]["final"]}')
    return dict(final=Xp, raw=Xr, triangulated=stage, quality=q, contacts=cont, cameras=None,
                err=np.zeros((1, T, J)), used=np.ones((1, T, J), bool), gflags=gflags, swaps={}, report=rep,
                com=center_of_mass(Xp), angles=joint_angles(Xp), fps=fps, hands=hands, grf=F, airborne=air)
