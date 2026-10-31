"""Dogs and horses: AP-10K/APT-36K 17-point skeleton (ViTPose), single- or multi-camera 3D,
anatomical + physical correction, and an Unreal-ready FBX with a generated quadruped skeleton
and a skinned proxy body (Unreal creates the Skeleton/Skeletal Mesh on first import)."""
import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from . import correct as K
from . import cameras as C

NAMES = ['left_eye', 'right_eye', 'nose', 'neck', 'tail_root',
         'left_shoulder', 'left_elbow', 'left_front_paw', 'right_shoulder', 'right_elbow', 'right_front_paw',
         'left_hip', 'left_knee', 'left_back_paw', 'right_hip', 'right_knee', 'right_back_paw']
IDX = {n: i for i, n in enumerate(NAMES)}
J = len(NAMES)
FLIP = np.arange(J)
for _n, _i in IDX.items():
    if _n.startswith('left_'):
        FLIP[_i] = IDX['right_' + _n[5:]]
    elif _n.startswith('right_'):
        FLIP[_i] = IDX['left_' + _n[6:]]
_p = lambda *ns: sum([[IDX['left_' + n], IDX['right_' + n]] for n in ns], [])
SWAP_GROUPS = {'front': _p('shoulder', 'elbow', 'front_paw'), 'back': _p('hip', 'knee', 'back_paw'), 'eyes': _p('eye')}
BONES = [('neck', 'nose'), ('nose', 'left_eye'), ('nose', 'right_eye'), ('neck', 'tail_root'),
         ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
         ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_front_paw'),
         ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_front_paw'),
         ('tail_root', 'left_hip'), ('tail_root', 'right_hip'),
         ('left_hip', 'left_knee'), ('left_knee', 'left_back_paw'),
         ('right_hip', 'right_knee'), ('right_knee', 'right_back_paw'),
         ('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip')]
BONE_IDX = np.array([[IDX[a], IDX[b]] for a, b in BONES])
SYM = []
for _k, (_a, _b) in enumerate(BONES):
    if 'left_' in _a + _b:
        _m = [m for m, (c, d) in enumerate(BONES) if (c, d) == (_a.replace('left_', 'right_'), _b.replace('left_', 'right_'))]
        SYM += [(_k, _m[0])] if _m else []
PAWS = {'front_left': [IDX['left_front_paw']], 'front_right': [IDX['right_front_paw']],
        'back_left': [IDX['left_back_paw']], 'back_right': [IDX['right_back_paw']]}
LEGS = [('left_shoulder', 'left_elbow', 'left_front_paw'), ('right_shoulder', 'right_elbow', 'right_front_paw'),
        ('left_hip', 'left_knee', 'left_back_paw'), ('right_hip', 'right_knee', 'right_back_paw')]

# height = withers height (m). Widths / leg chain as a fraction of it; mass in kg.
SPECIES = {
    'dog': dict(label='Dog', height=0.55, mass=25.0, shoulder_w=0.30, hip_w=0.28, eye_w=0.10, leg=0.70,
                coco=(16,), contact_h=0.04, contact_v=0.35),
    'horse': dict(label='Horse', height=1.60, mass=500.0, shoulder_w=0.26, hip_w=0.30, eye_w=0.12, leg=0.62,
                  coco=(17,), contact_h=0.07, contact_v=0.6),
}
ANIMAL_COCO = (15, 16, 17, 18, 19, 20, 21, 22, 23)
# (proximal, distal, mass fraction, COM position from proximal): approximate canine/equine data
SEGMENTS = [('neck', 'nose', 0.09, 0.45), ('neck', 'tail_root', 0.58, 0.40),
            ('left_shoulder', 'left_front_paw', 0.065, 0.30), ('right_shoulder', 'right_front_paw', 0.065, 0.30),
            ('left_hip', 'left_back_paw', 0.09, 0.30), ('right_hip', 'right_back_paw', 0.09, 0.30)]


def center_of_mass(P):
    com = np.zeros(P.shape[:1] + (3,)); tot = 0.0
    for a, b, m, r in SEGMENTS:
        pa, pb = P[:, IDX[a]], P[:, IDX[b]]
        com += m * (pa + r * (pb - pa)); tot += m
    return com / tot


# --------------------------------------------------------------- detection --
class AnimalDetector:
    MODELS = {'lightweight': ('https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx', (416, 416),
                              'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-s-apt36k.onnx', (192, 256)),
              'balanced': ('https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx', (640, 640),
                           'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx', (192, 256))}

    def __init__(self, species='dog', mode='balanced', device='auto'):
        from rtmlib import YOLOX, ViTPose
        dev = 'cpu'
        if device in ('auto', 'cuda'):
            try:
                import onnxruntime as ort
                dev = 'cuda' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'cpu'
            except Exception:
                pass
        d, ds, p, ps = self.MODELS['lightweight' if mode == 'lightweight' else 'balanced']
        self.det = YOLOX(d, det_mode='multiclass', model_input_size=ds, backend='onnxruntime', device=dev)
        self.pose = ViTPose(p, model_input_size=ps, to_openpose=False, backend='onnxruntime', device=dev)
        self.want = SPECIES[species]['coco']
        self.device = dev

    def boxes(self, img):
        b, c = self.det(img)
        b = np.asarray(b, float).reshape(-1, 4); c = np.asarray(c).reshape(-1)
        for pref in (self.want, ANIMAL_COCO):
            sel = [bb for bb, cc in zip(b, c) if int(cc) in pref]
            if sel:
                return sel
        sel = [bb for bb, cc in zip(b, c) if int(cc) != 0]       # anything but people (riders)
        return sel or [np.array([0, 0, img.shape[1], img.shape[0]], float)]

    def __call__(self, img, prev=None):
        bx = self.boxes(img)
        if prev is not None and len(bx) > 1:
            ctr = np.nanmean(prev, 0)
            bx = [min(bx, key=lambda b: np.hypot((b[0] + b[2]) / 2 - ctr[0], (b[1] + b[3]) / 2 - ctr[1]))]
        elif len(bx) > 1:
            bx = [max(bx, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))]
        k, s = self.pose(img, bboxes=[list(bx[0])])
        return np.asarray(k, float).reshape(J, 2), np.asarray(s, float).reshape(J)


def detect_video(video, panels, use, det, out_dir=None, max_frames=None, log=print):
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps > 240 or fps < 1:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total = min(total, max_frames) if total else max_frames
    V = len(use)
    KP, SC = [[] for _ in range(V)], [[] for _ in range(V)]
    prev = [None] * V
    writer = None
    t = 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and t >= max_frames):
            break
        for v, pi in enumerate(use):
            x0, y0, x1, y1 = panels[pi]
            k, s = det(fr[y0:y1, x0:x1], prev[v])
            KP[v].append(k); SC[v].append(s)
            if s.mean() > 0.3:
                prev[v] = k
            for (a, b) in BONE_IDX:
                if s[a] > .3 and s[b] > .3:
                    cv2.line(fr, (int(x0 + k[a, 0]), int(y0 + k[a, 1])), (int(x0 + k[b, 0]), int(y0 + k[b, 1])), (0, 255, 0), 2)
        if out_dir is not None:
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(out_dir, 'debug_2d_overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (fr.shape[1], fr.shape[0]))
            writer.write(fr)
        t += 1
        if t % 10 == 0:
            log(f'  2D detection: frame {t}/{total}')
    cap.release()
    if writer is not None:
        writer.release()
    sizes = [(panels[pi][2] - panels[pi][0], panels[pi][3] - panels[pi][1]) for pi in use]
    return np.array(KP), np.array(SC), fps, sizes


# ------------------------------------------------------------ single view --
def _lowpass(x, fps, hz):
    return K.butter_lowpass(x.reshape(len(x), -1), fps, hz).reshape(x.shape) if len(x) > 15 else x


def lift_single(kp, conf, size, fps, sp, fov=40.0):
    """Quadruped from one camera. In-plane positions from the image; depth from the body's
    yaw (left/right shoulder & hip offsets vs. the species' body width) and each joint's
    known lateral side. Strongest for side / 3-4 views (typical animal footage)."""
    T = len(kp)
    w, h = size
    H = sp['height']
    q = np.where(np.isfinite(kp).all(-1), conf, 0.0)
    kp = np.where(np.isfinite(kp), kp, 0.0)
    # metric scale from the 2D length of the leg chains (bending does not change it)
    chains = []
    for a, b, c in LEGS:
        ia, ib, ic = IDX[a], IDX[b], IDX[c]
        Lc = np.linalg.norm(kp[:, ib] - kp[:, ia], axis=-1) + np.linalg.norm(kp[:, ic] - kp[:, ib], axis=-1)
        ok = np.minimum(np.minimum(q[:, ia], q[:, ib]), q[:, ic]) > 0.3
        chains.append(np.where(ok, Lc, np.nan))
    chains = np.array(chains).T
    cpx = np.nanmedian(chains, 1)
    t = np.arange(T)
    good = np.isfinite(cpx)
    if good.sum() < 2:
        cpx = np.full(T, 0.4 * h)
    else:
        cpx = np.interp(t, t[good], cpx[good])
    m = np.exp(_lowpass(np.log(sp['leg'] * H / np.maximum(cpx, 1.0)), fps, 1.0))      # m per px
    x = (kp[..., 0] - w / 2) * m[:, None]
    y = (h / 2 - kp[..., 1]) * m[:, None]
    # yaw: sin from left/right offsets, cos sign from facing direction
    ws, wh = sp['shoulder_w'] * H, sp['hip_w'] * H
    num = np.zeros(T); den = np.zeros(T)
    for (l, r), wd in (((IDX['left_shoulder'], IDX['right_shoulder']), ws), ((IDX['left_hip'], IDX['right_hip']), wh)):
        cw = np.minimum(q[:, l], q[:, r])
        num += cw * (-(x[:, l] - x[:, r]) / wd); den += cw
    s = np.clip(np.where(den > 0, num / np.maximum(den, 1e-6), 0.0), -0.95, 0.95)
    from scipy.ndimage import median_filter
    s = _lowpass(median_filter(s, 9, mode='nearest'), fps, 2.0)
    face = np.sign(median_filter(np.sign(x[:, IDX['neck']] - x[:, IDX['tail_root']]), 15, mode='nearest'))
    face[face == 0] = 1
    c = face * np.maximum(np.sqrt(np.clip(1 - s * s, 0, 1)), 0.35)
    lat = np.zeros(J)
    for n, i in IDX.items():
        half = sp['eye_w'] * H / 2 if 'eye' in n else (ws / 2 if ('shoulder' in n or 'elbow' in n or 'front' in n) else wh / 2)
        lat[i] = half if n.startswith('left_') else (-half if n.startswith('right_') else 0.0)
    xc = 0.5 * (x[:, IDX['neck']] + x[:, IDX['tail_root']])
    f_ = ((x - xc[:, None]) + lat[None] * s[:, None]) / c[:, None]
    z = -f_ * s[:, None] - lat[None] * c[:, None]
    fpx = 0.5 * h / np.tan(np.radians(fov) / 2)
    D = fpx * m
    X = np.stack([x, y, z - D[:, None]], -1)
    return X, q, dict(yaw_deg=np.degrees(np.arctan2(s, c)))


# ------------------------------------------------------------------ world --
def align_world(P, height):
    """Ground plane from the paws (RANSAC), +Y up, facing +Z, origin under the body, scaled
    so the withers (neck/shoulders) reach `height`."""
    paws = P[:, [i[0] for i in PAWS.values()]].reshape(-1, 3)
    paws = paws[np.isfinite(paws).all(-1)]
    body = 0.5 * (P[:, IDX['neck']] + P[:, IDX['tail_root']])
    pm = np.nanmean(P[:, [i[0] for i in PAWS.values()]], 1)
    up0 = np.nanmedian(body - pm, 0); up0 /= np.linalg.norm(up0)
    hgt = paws @ up0
    cand = paws[hgt < np.percentile(hgt, 40)]
    rng = np.random.default_rng(0)
    best, bn = None, -1
    tol = 0.05 * np.nanmedian(np.linalg.norm(body - pm, axis=-1))
    for _ in range(300):
        if len(cand) < 3:
            break
        smp = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(smp[1] - smp[0], smp[2] - smp[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        n = n if n @ up0 > 0 else -n
        if n @ up0 < np.cos(np.radians(35)):
            continue
        inl = np.abs((cand - smp[0]) @ n) < tol
        if inl.sum() > bn:
            bn, best = inl.sum(), inl
    pts = cand[best] if best is not None and best.sum() >= 3 else cand
    ctr = pts.mean(0)
    nn = np.linalg.svd(pts - ctr)[2][-1] if len(pts) >= 3 else up0
    nn = nn if nn @ up0 > 0 else -nn
    R = K._rot_between(nn, np.array([0, 1.0, 0]))
    Q = (P - ctr) @ R.T
    Q[..., 1] -= np.percentile(Q[:, [i[0] for i in PAWS.values()], 1], 2)
    fwd = np.nanmean((Q[:, IDX['neck']] - Q[:, IDX['tail_root']])[: max(5, len(Q) // 10)], 0)
    a = np.arctan2(fwd[0], fwd[2])
    ca, sa = np.cos(-a), np.sin(-a)
    Q = Q @ np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]]).T
    o = (0.5 * (Q[0, IDX['neck']] + Q[0, IDX['tail_root']])).copy(); o[1] = 0
    Q -= o
    wh = np.percentile(0.5 * (Q[:, IDX['left_shoulder'], 1] + Q[:, IDX['right_shoulder'], 1]), 75) / 0.82
    scale = height / max(wh, 1e-6)
    return Q * scale, scale


def fix_mirror(P):
    """Whole-body left/right label flip: the left side must point to the animal's left."""
    fwd = P[:, IDX['neck']] - P[:, IDX['tail_root']]
    lat = (P[:, IDX['left_shoulder']] - P[:, IDX['right_shoulder']]) + (P[:, IDX['left_hip']] - P[:, IDX['right_hip']])
    left = np.cross(np.array([0, 1.0, 0]), fwd)
    if np.nanmedian((lat * left).sum(-1)) < 0:
        return P[:, FLIP], True
    return P, False


# ------------------------------------------------------------ reconstruct --
def reconstruct(kp, conf, sizes, fps, species='dog', opts=None, log=print):
    from .pipeline import DEFAULTS, calibrate
    from .physics import simulate
    sp = SPECIES[species]
    o = dict(DEFAULTS, height=sp['height'], weight=sp['mass'])
    o.update({k: v for k, v in (opts or {}).items() if v is not None})
    t0 = time.time()
    V, T = kp.shape[:2]
    rep = {'subject': species, 'views': V, 'frames': T, 'fps': fps, 'mode': 'multi_view' if V > 1 else 'single_view'}
    conf_g, gflags = K.flag_2d_glitches(kp, conf)
    rep['glitch_keypoints'] = int(((conf > o['min_conf']) & (conf_g == 0)).sum())
    cams = None
    if V >= 2:
        o2 = dict(o, _swap=dict(groups=SWAP_GROUPS, flip=FLIP))
        cams, rep['calibration_score'] = calibrate(kp, conf_g, sizes, o2, log)
        uv, cf, dec = C.resolve_swaps(cams, kp, conf_g, lam=o['swap_lambda'], min_conf=o['min_conf'],
                                      groups=SWAP_GROUPS, flip=FLIP)
        rep['swaps'] = {g: d.sum(0).tolist() for g, d in dec.items()}
        X, err, used, q = C.robust_triangulate(cams, uv, cf, thr=o['reproj_thr'], min_conf=o['min_conf'])
    else:
        X, q, info = lift_single(kp[0], conf_g[0], sizes[0], fps, sp, o['fov'])
        rep['yaw_deg_median'] = float(np.median(info['yaw_deg']))
    Xraw = X.copy()
    Xf, filled = K.fill_gaps(X, q)
    rep['filled_samples_pct'] = float(filled.mean() * 100)
    Xf, rep['spikes_removed'] = K.hampel(Xf)
    Xf = K.butter_lowpass(Xf, fps, o['cutoff'])
    Xf, mir = fix_mirror(Xf)
    if mir:
        q = q[:, FLIP]
    rep['global_mirror_fixed'] = bool(mir)
    Xw, rep['scale'] = align_world(Xf, o['height'])
    stage = Xw.copy()
    L = K.bone_targets(Xw, q, BONE_IDX, SYM)
    Xa = K.enforce_bones(Xw, L, q, bone_idx=BONE_IDX)
    hs = o['height'] / sp['height']
    cont = K.foot_contacts(Xa, fps, h_thr=sp['contact_h'] * hs, v_thr=sp['contact_v'] * hs, feet=PAWS)
    Xp, pinned = K.fix_foot_skating(Xa, cont, max_shift=0.1 * o['height'], feet=PAWS)
    Xp = K.enforce_bones(Xp, L, q, iters=15, heavy=pinned, bone_idx=BONE_IDX)
    Xp, rep['ground_clamped_samples'] = K.clamp_ground(Xp)

    def repin(P):
        P, pn = K.fix_foot_skating(P, cont, max_shift=0.1 * o['height'], feet=PAWS)
        P = K.enforce_bones(P, L, q, iters=10, heavy=pn, bone_idx=BONE_IDX)
        return K.clamp_ground(P)[0]
    F = np.zeros((T, 3)); air = np.zeros(T, bool)
    if o.get('physics', True) and T >= 10:
        Xp, rep['physics'], F, air = simulate(Xp, fps, center_of_mass, PAWS, cont, o['weight'], repin, log)
    mk = lambda P: K.metrics(P, fps, cont, bone_idx=BONE_IDX, feet=PAWS)
    Xr, _ = K.fill_gaps(Xraw, np.isfinite(Xraw).all(-1).astype(float))
    try:
        Xr, _ = align_world(fix_mirror(Xr)[0], o['height'])
    except Exception:
        pass
    rep['metrics'] = {'raw': mk(Xr), 'triangulated': mk(stage), 'final': mk(Xp)}
    rep['contact_pct'] = {k: float(v.mean() * 100) for k, v in cont.items()}
    rep['bone_lengths_m'] = {f'{a}-{b}': float(l) for (a, b), l in zip(BONES, L)}
    rep['seconds'] = round(time.time() - t0, 1)
    log(f'  metrics: {json.dumps(rep["metrics"]["final"])}')
    return dict(final=Xp, raw=Xr, quality=q, contacts=cont, cameras=cams, report=rep, fps=fps,
                com=center_of_mass(Xp), grf=F, airborne=air, species=species)


# --------------------------------------------------------------- rig + FBX --
W2R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # world (Y up, +Z fwd) -> rig (Z up, -Y fwd)


def _body_frame(P):
    fwd = (0.5 * (P[:, IDX['left_shoulder']] + P[:, IDX['right_shoulder']])
           - 0.5 * (P[:, IDX['left_hip']] + P[:, IDX['right_hip']]))
    lat = (P[:, IDX['left_shoulder']] - P[:, IDX['right_shoulder']]) + (P[:, IDX['left_hip']] - P[:, IDX['right_hip']])
    f = fwd / (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9)
    l = lat - (lat * f).sum(-1, keepdims=True) * f
    l /= np.linalg.norm(l, axis=-1, keepdims=True) + 1e-9
    u = np.cross(f, l)
    return f, l, u


def _virtual(P):
    """Rig landmark positions (T,3) from the 17 keypoints."""
    g = lambda n: P[:, IDX[n]]
    hipm = 0.5 * (g('left_hip') + g('right_hip'))
    shm = 0.5 * (g('left_shoulder') + g('right_shoulder'))
    V = {'pelvis': hipm, 'chest': shm, 'neck': g('neck'), 'nose': g('nose'),
         'eyes': 0.5 * (g('left_eye') + g('right_eye')), 'tail': g('tail_root')}
    V['spine_01'] = hipm + (shm - hipm) / 3
    V['spine_02'] = hipm + 2 * (shm - hipm) / 3
    V['head'] = g('neck') + 0.6 * (V['eyes'] - g('neck'))
    for s in ('l', 'r'):
        side = 'left_' if s == 'l' else 'right_'
        for k, n in (('upperarm', 'shoulder'), ('lowerarm', 'elbow'), ('hand', 'front_paw'),
                     ('thigh', 'hip'), ('calf', 'knee'), ('foot', 'back_paw')):
            V[f'{k}_{s}'] = g(side + n)
    return V


# name, parent, head landmark, tail landmark (direction), radius (fraction of height)
RIG = [('pelvis', 'root', 'pelvis', 'spine_01', 0.12), ('spine_01', 'pelvis', 'spine_01', 'spine_02', 0.13),
       ('spine_02', 'spine_01', 'spine_02', 'chest', 0.13), ('spine_03', 'spine_02', 'chest', 'neck', 0.12),
       ('neck_01', 'spine_03', 'neck', 'head', 0.07), ('head', 'neck_01', 'head', 'nose', 0.07),
       ('tail_01', 'pelvis', 'pelvis', 'tail', 0.08), ('tail_02', 'tail_01', 'tail', None, 0.025),
       ('upperarm_l', 'spine_03', 'upperarm_l', 'lowerarm_l', 0.045), ('lowerarm_l', 'upperarm_l', 'lowerarm_l', 'hand_l', 0.03),
       ('hand_l', 'lowerarm_l', 'hand_l', None, 0.025),
       ('upperarm_r', 'spine_03', 'upperarm_r', 'lowerarm_r', 0.045), ('lowerarm_r', 'upperarm_r', 'lowerarm_r', 'hand_r', 0.03),
       ('hand_r', 'lowerarm_r', 'hand_r', None, 0.025),
       ('thigh_l', 'pelvis', 'thigh_l', 'calf_l', 0.055), ('calf_l', 'thigh_l', 'calf_l', 'foot_l', 0.03),
       ('foot_l', 'calf_l', 'foot_l', None, 0.025),
       ('thigh_r', 'pelvis', 'thigh_r', 'calf_r', 0.055), ('calf_r', 'thigh_r', 'calf_r', 'foot_r', 0.03),
       ('foot_r', 'calf_r', 'foot_r', None, 0.025)]


def build_rig(P, species):
    """Standing rest pose (cm, Z up, -Y forward) from the clip's median proportions."""
    sp = SPECIES[species]
    V = _virtual(P)
    f, l, u = _body_frame(P)
    body = lambda p: np.stack([(p * l).sum(-1), (p * u).sum(-1), (p * f).sum(-1)], -1)   # (left, up, fwd)
    med = lambda k, ref: np.median(body(V[k] - V[ref]), 0)
    ln = lambda a, b: float(np.median(np.linalg.norm(V[a] - V[b], axis=-1)))
    rest = {}
    front_len = [ln('upperarm_l', 'lowerarm_l') + ln('lowerarm_l', 'hand_l'), ln('upperarm_r', 'lowerarm_r') + ln('lowerarm_r', 'hand_r')]
    back_len = [ln('thigh_l', 'calf_l') + ln('calf_l', 'foot_l'), ln('thigh_r', 'calf_r') + ln('calf_r', 'foot_r')]
    hip_h, sh_h = float(np.mean(back_len)), float(np.mean(front_len))
    spine = ln('pelvis', 'chest')
    rest['pelvis'] = np.array([0, hip_h, -spine / 2])
    rest['chest'] = np.array([0, sh_h, spine / 2])
    rest['spine_01'] = rest['pelvis'] + (rest['chest'] - rest['pelvis']) / 3
    rest['spine_02'] = rest['pelvis'] + 2 * (rest['chest'] - rest['pelvis']) / 3
    for k in ('neck', 'head', 'nose'):
        rest[k] = rest['chest'] + med(k, 'chest') * np.array([0, 1, 1])        # keep on the midline
    rest['tail'] = rest['pelvis'] + med('tail', 'pelvis') * np.array([0, 1, 1])
    for s in ('l', 'r'):
        sg = 1 if s == 'l' else -1
        half_s = abs(med('upperarm_' + s, 'chest')[0]); half_h = abs(med('thigh_' + s, 'pelvis')[0])
        u1, l1 = ln('upperarm_' + s, 'lowerarm_' + s), ln('lowerarm_' + s, 'hand_' + s)
        t1, c1 = ln('thigh_' + s, 'calf_' + s), ln('calf_' + s, 'foot_' + s)
        rest['upperarm_' + s] = np.array([sg * half_s, sh_h, spine / 2])
        rest['lowerarm_' + s] = rest['upperarm_' + s] - [0, u1, 0]
        rest['hand_' + s] = np.array([sg * half_s, sh_h - u1 - l1, spine / 2])
        rest['thigh_' + s] = np.array([sg * half_h, hip_h, -spine / 2])
        rest['calf_' + s] = rest['thigh_' + s] - [0, t1, 0]
        rest['foot_' + s] = np.array([sg * half_h, hip_h - t1 - c1, -spine / 2])
    # body coords (left, up, fwd) metres -> rig (x=left, y=-fwd, z=up) cm
    to_rig = lambda p: np.array([p[0], -p[2], p[1]]) * 100
    pos = {'root': np.zeros(3)}
    for name, parent, head, tail, rad in RIG:
        pos[name] = to_rig(rest[head])
    tails = {}
    for name, parent, head, tail, rad in RIG:
        if tail is not None:
            tails[name] = to_rig(rest[tail])
        elif name == 'tail_02':
            tails[name] = pos[name] + to_rig(np.array([0, -0.15, -0.30]) * sp['height'])
        else:
            tails[name] = pos[name] + to_rig(np.array([0, -0.0, 0.08]) * sp['height'])
    bones = [dict(name=sp['label'], parent=None, kind='Null', t=[0, 0, 0], r=[0, 0, 0], s=[1, 1, 1], pre=[0, 0, 0], post=[0, 0, 0]),
             dict(name='root', parent=sp['label'], kind='Root', t=[0, 0, 0], r=[0, 0, 0], s=[1, 1, 1], pre=[0, 0, 0], post=[0, 0, 0])]
    for name, parent, head, tail, rad in RIG:
        bones.append(dict(name=name, parent=parent, kind='LimbNode', t=list(pos[name] - pos[parent]), r=[0, 0, 0],
                          s=[1, 1, 1], pre=[0, 0, 0], post=[0, 0, 0]))
    return dict(name=f'{sp["label"]}_AIMoCap', label=f'{sp["label"]} (generated rig)', species=species,
                axes=dict(UpAxis=2, UpAxisSign=1, FrontAxis=1, FrontAxisSign=-1, CoordAxis=0, CoordAxisSign=1,
                          OriginalUpAxis=2, OriginalUpAxisSign=1, UnitScaleFactor=1.0, OriginalUnitScaleFactor=1.0),
                bones=bones, rest_pos=pos, rest_tail=tails, radius={n: r * sp['height'] * 100 for n, _, _, _, r in RIG})


def retarget(P, rig, fps):
    """Animate the generated rig: every bone's frame = (bone direction, body left axis);
    rotation = delta from the rest frame; FK with the rig's fixed lengths."""
    T = len(P)
    V = {k: v @ W2R.T * 100 for k, v in _virtual(P).items()}                 # rig space, cm
    fW, lW, uW = _body_frame(P)
    left = lW @ W2R.T
    pos0, tail0 = rig['rest_pos'], rig['rest_tail']
    par = {b['name']: b['parent'] for b in rig['bones']}

    def frame(d, s):
        d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9)
        s = s - (s * d).sum(-1, keepdims=True) * d
        s /= np.linalg.norm(s, axis=-1, keepdims=True) + 1e-9
        return np.stack([d, s, np.cross(d, s)], -1)
    Rg = {}
    for name, parent, head, tail, rad in RIG:
        d0 = tail0[name] - pos0[name]
        if tail is not None:
            d = V[tail] - V[head]
        else:                                                                  # paws / tail tip follow the parent
            Rg[name] = Rg[par[name]]; continue
        F = frame(d, left)
        F0 = frame(d0[None], np.array([[1.0, 0, 0]]))
        Rg[name] = F @ np.swapaxes(F0, -1, -2)
    # smooth rotations (zero-phase on rotation vectors relative to the parent)
    G = {}
    for b in rig['bones']:
        n, p = b['name'], b['parent']
        if p is None or n == 'root':
            G[n] = np.tile(np.eye(4), (T, 1, 1)); continue
        g = np.tile(np.eye(4), (T, 1, 1))
        if n in Rg:
            g[:, :3, :3] = Rg[n]
        else:
            g[:, :3, :3] = G[p][:, :3, :3]
        if n == 'pelvis':
            g[:, :3, 3] = V['pelvis']
        else:
            g[:, :3, 3] = G[p][:, :3, 3] + np.einsum('tij,j->ti', G[p][:, :3, :3], np.asarray(b['t']))
        G[n] = g
    bones = {}
    for b in rig['bones']:
        n, p = b['name'], b['parent']
        Pg = G[p] if p is not None else np.tile(np.eye(4), (T, 1, 1))
        Lc = np.linalg.inv(Pg) @ G[n]
        bones[n] = (Lc[:, :3, :3], Lc[:, :3, 3])
    return dict(bones=bones, globals=G)


def proxy_mesh(rig):
    """Tapered boxes along each bone, rigidly skinned (Unreal needs a mesh to make a Skeleton)."""
    verts, polys, weights = [], [], {}
    for name, parent, head, tail, rad in RIG:
        a, b = rig['rest_pos'][name], rig['rest_tail'][name]
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        x = d / L
        y = np.cross(x, [0, 0, 1.0]) if abs(x[2]) < 0.9 else np.cross(x, [1.0, 0, 0])
        y /= np.linalg.norm(y); z = np.cross(x, y)
        r0, r1 = rig['radius'][name], rig['radius'][name] * 0.8
        base = len(verts)
        for p, r in ((a, r0), (b, r1)):
            for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                verts.append(p + r * (sy * y + sz * z))
        q = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
        polys += [[base + i for i in f] for f in q]
        weights[name] = (np.arange(base, base + 8), np.ones(8))
    return np.array(verts), polys, weights


def write_fbx(path, rig, res, fps, take='Take 001', with_mesh=True):
    from . import fbx_io as F
    from .fbx_io import P as Pp
    from .unreal import build_fbx
    from .retarget import bone_curves
    curves = bone_curves(rig, res)
    root = build_fbx(rig, curves, fps, take=take)
    if with_mesh:
        _add_mesh(root, rig, *proxy_mesh(rig))
    F.write(path, root, 7400)


def _add_mesh(root, rig, verts, polys, weights):
    """Geometry + Skin/Clusters + BindPose appended to a build_fbx() scene."""
    from .fbx_io import P as Pp, Node
    from .unreal import _name, _Ids
    objs, conns, defs = root.find('Objects'), root.find('Connections'), root.find('Definitions')
    ids = _Ids(); ids.n = 9_000_000
    model_id = {}
    for m in objs.findall('Model'):
        model_id[m.values[1].split(b'\x00')[0].decode()] = m.values[0]
    gid, mid, sid = ids(), ids(), ids()
    g = objs.add('Geometry', Pp(b'L', gid), _name('Proxy', 'Geometry'), 'Mesh')
    g.add('Vertices', Pp(b'd', np.asarray(verts, float).ravel()))
    pvi = []
    for p in polys:
        pvi += list(p[:-1]) + [-p[-1] - 1]
    g.add('PolygonVertexIndex', Pp(b'i', np.asarray(pvi, np.int32)))
    g.add('GeometryVersion', 124)
    # per-polygon-vertex normals (flat), so importers need no smoothing guess
    nrm = []
    for p in polys:
        v = np.asarray(verts)[p]
        n = np.cross(v[1] - v[0], v[2] - v[0]); n /= np.linalg.norm(n) + 1e-12
        nrm += list(n) * len(p)
    ln = g.add('LayerElementNormal', 0)
    ln.add('Version', 101); ln.add('Name', ''); ln.add('MappingInformationType', 'ByPolygonVertex')
    ln.add('ReferenceInformationType', 'Direct'); ln.add('Normals', Pp(b'd', np.asarray(nrm, float)))
    lm = g.add('LayerElementMaterial', 0)
    lm.add('Version', 101); lm.add('Name', ''); lm.add('MappingInformationType', 'AllSame')
    lm.add('ReferenceInformationType', 'IndexToDirect'); lm.add('Materials', Pp(b'i', np.zeros(1, np.int32)))
    lay = g.add('Layer', 0); lay.add('Version', 100)
    for t in ('LayerElementNormal', 'LayerElementMaterial'):
        le = lay.add('LayerElement'); le.add('Type', t); le.add('TypedIndex', 0)
    mo = objs.add('Model', Pp(b'L', mid), _name('Proxy', 'Model'), 'Mesh')
    mo.add('Version', 232)
    from .fbx_io import p70
    mp = mo.add('Properties70'); p70(mp, 'DefaultAttributeIndex', 'int', 'Integer', '', 0)
    mo.add('Shading', Pp(b'C', True)); mo.add('Culling', 'CullingOff')
    matid = ids()
    mat = objs.add('Material', Pp(b'L', matid), _name('ProxyMaterial', 'Material'), '')
    mat.add('Version', 102); mat.add('ShadingModel', 'lambert'); mat.add('MultiLayer', 0)
    mpp = mat.add('Properties70'); p70(mpp, 'DiffuseColor', 'Color', '', 'A', 0.7, 0.72, 0.76)
    conns.add('C', 'OO', Pp(b'L', gid), Pp(b'L', mid))
    conns.add('C', 'OO', Pp(b'L', mid), Pp(b'L', 0))
    conns.add('C', 'OO', Pp(b'L', matid), Pp(b'L', mid))
    sk = objs.add('Deformer', Pp(b'L', sid), _name('Skin', 'Deformer'), 'Skin')
    sk.add('Version', 101); sk.add('Link_DeformAcuracy', 50.0)
    conns.add('C', 'OO', Pp(b'L', sid), Pp(b'L', gid))
    from .rig import rest_globals
    G0 = rest_globals(rig)
    pose = objs.add('Pose', Pp(b'L', ids()), _name('BindPose', 'Pose'), 'BindPose')
    pose.add('Type', 'BindPose'); pose.add('Version', 100)
    nodes = [(mid, np.eye(4))] + [(model_id[b['name']], G0[b['name']]) for b in rig['bones'] if b['name'] in model_id]
    pose.add('NbPoseNodes', len(nodes))
    for nid, M in nodes:
        pn = pose.add('PoseNode'); pn.add('Node', Pp(b'L', nid)); pn.add('Matrix', Pp(b'd', np.asarray(M, float).T.ravel()))
    ncl = 0
    for bn, (idx, w) in weights.items():
        cid = ids(); ncl += 1
        cl = objs.add('Deformer', Pp(b'L', cid), _name(bn, 'SubDeformer'), 'Cluster')
        cl.add('Version', 100); cl.add('UserData', '', '')
        cl.add('Indexes', Pp(b'i', np.asarray(idx, np.int32))); cl.add('Weights', Pp(b'd', np.asarray(w, float)))
        L = G0[bn]
        cl.add('Transform', Pp(b'd', np.linalg.inv(L).T.ravel())); cl.add('TransformLink', Pp(b'd', np.asarray(L).T.ravel()))
        conns.add('C', 'OO', Pp(b'L', cid), Pp(b'L', sid))
        conns.add('C', 'OO', Pp(b'L', model_id[bn]), Pp(b'L', cid))
    # definitions
    cnt = defs.find('Count')
    cnt.props[0].v += 1 + 1 + 1 + 1 + ncl + 1
    for c in defs.findall('ObjectType'):
        if c.values[0] == b'Model':
            c.find('Count').props[0].v += 1
    for typ, n in (('Geometry', 1), ('Material', 1), ('Deformer', 1 + ncl), ('Pose', 1)):
        ot = defs.add('ObjectType', typ); ot.add('Count', n)


def write_bvh(path, rig, res, fps):
    """BVH (Y up, cm) of the generated rig."""
    from scipy.spatial.transform import Rotation as Rr
    M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)                   # rig (Z up) -> BVH (Y up)
    bones = [b for b in rig['bones'] if b['kind'] != 'Null']
    ch = {}
    for b in bones:
        ch.setdefault(b['parent'], []).append(b['name'])
    by = {b['name']: b for b in bones}
    lines = ['HIERARCHY']
    order = []

    def emit(n, d):
        ind = '  ' * d
        o = M @ np.asarray(by[n]['t'])
        lines.append(f'{ind}{"ROOT" if d == 0 else "JOINT"} {n}'); lines.append(ind + '{')
        lines.append(f'{ind}  OFFSET {o[0]:.4f} {o[1]:.4f} {o[2]:.4f}')
        lines.append(f'{ind}  CHANNELS {"6 Xposition Yposition Zposition " if d == 0 else "3 "}Zrotation Xrotation Yrotation')
        order.append(n)
        kids = ch.get(n, [])
        for c in kids:
            emit(c, d + 1)
        if not kids:
            e = M @ (rig['rest_tail'].get(n, rig['rest_pos'][n] + [0, 0, -5]) - rig['rest_pos'][n])
            lines.extend([f'{ind}  End Site', ind + '  {', f'{ind}    OFFSET {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}', ind + '  }'])
        lines.append(ind + '}')
    emit('root', 0)
    T = len(res['globals']['root'])
    eul = {}
    for n in order:
        R = M @ res['bones'][n][0] @ M.T
        eul[n] = np.degrees(np.unwrap(np.radians(Rr.from_matrix(R).as_euler('ZXY', degrees=True)), axis=0))
    rootp = (M @ res['bones']['root'][1].T).T
    lines += ['MOTION', f'Frames: {T}', f'Frame Time: {1.0 / fps:.6f}']
    # the root bone stays at the origin; pelvis translation is not a BVH channel -> bake into root
    pel = (M @ res['globals']['pelvis'][:, :3, 3].T).T - M @ rig['rest_pos']['pelvis']
    for t in range(T):
        vals = list(rootp[t] + pel[t])
        for n in order:
            vals += list(eul[n][t])
        lines.append(' '.join(f'{v:.4f}' for v in vals))
    open(path, 'w').write('\n'.join(lines) + '\n')


def export_all(res, out, log=print):
    from .viewer import write_viewer
    from .physics import write_physics_csv
    os.makedirs(out, exist_ok=True)
    P, fps, species = res['final'], res['fps'], res['species']
    sp = SPECIES[species]
    T = len(P)
    np.save(os.path.join(out, 'body_3d_xyz_m.npy'), P)
    hdr = ['frame', 'time_s'] + [f'{n}_{a}' for n in NAMES for a in 'xyz']
    np.savetxt(os.path.join(out, 'body_3d_xyz_m.csv'), np.concatenate([np.arange(T)[:, None], (np.arange(T) / fps)[:, None],
               P.reshape(T, -1)], 1), delimiter=',', header=','.join(hdr), comments='', fmt='%.5f')
    write_physics_csv(os.path.join(out, 'physics_com_grf.csv'), res['com'], res['grf'], res['airborne'], fps,
                      res['report'].get('physics', {}).get('mass_kg', sp['mass']))
    r = lambda a: np.round(a, 4).tolist()
    data = dict(fps=fps, subject=species, joints=NAMES, bones=[list(b) for b in BONES], corrected=r(P), raw=None,
                quality=np.round(res['quality'], 2).tolist(), com=r(res['com']),
                contacts={k: v.astype(int).tolist() for k, v in res['contacts'].items()},
                feet={k: NAMES[v[0]] for k, v in PAWS.items()}, root='neck')
    json.dump(data, open(os.path.join(out, 'motion.json'), 'w'), separators=(',', ':'))
    write_viewer(data, os.path.join(out, 'viewer.html'))
    json.dump(res['report'], open(os.path.join(out, 'report.json'), 'w'), indent=2, default=float)
    write_report(res, os.path.join(out, 'report.html'))
    rig = build_rig(P, species)
    rt = retarget(P, rig, fps)
    write_fbx(os.path.join(out, f'motion_{sp["label"]}.fbx'), rig, rt, fps, take=f'{sp["label"]}_motion')
    write_bvh(os.path.join(out, 'motion.bvh'), rig, rt, fps)
    if res.get('cameras') is not None:
        res['cameras'].to_anipose_toml(os.path.join(out, 'virtual_cameras_calibration.toml'))
    log(f'  exported: motion_{sp["label"]}.fbx (generated {sp["label"].lower()} rig + proxy body), motion.bvh, '
        'motion.json, viewer.html, report.html, physics CSV')


def write_report(res, path):
    r = res['report']; m = r['metrics']; ph = r.get('physics', {})
    sp = SPECIES[res['species']]
    row = lambda k, lab: (f'<tr><td>{lab}</td>' + ''.join(f'<td>{m[s].get(k, 0):.3f}</td>' for s in ('raw', 'triangulated', 'final')) + '</tr>')
    html = f'''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>{sp["label"]} motion report</title>
<style>body{{font:14px Tahoma,Segoe UI,Arial;max-width:900px;margin:24px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #e3e8ee;padding:6px 8px;text-align:center}}th{{background:#f4f7fa}}</style></head><body>
<h1>تقرير حركة {"الكلب" if res["species"] == "dog" else "الحصان"}</h1>
<p>الإطارات {r["frames"]} · الزوايا {r["views"]} ({"كاميرا واحدة" if r["views"] == 1 else "متعددة"}) · FPS {r["fps"]:.1f} · الوزن {ph.get("mass_kg", sp["mass"]):.0f} كغ</p>
<table><tr><th>المقياس</th><th>خام</th><th>قبل القيود</th><th>نهائي</th></tr>
{row("bone_length_cv_pct", "تذبذب أطوال العظام %")}{row("mean_accel_m_s2", "الارتعاش m/s²")}{row("foot_skate_m_s", "انزلاق الأقدام m/s")}</table>
<h2>الفيزياء</h2><table>{"".join(f"<tr><td>{k}</td><td>{v:.2f}</td></tr>" for k, v in ph.items() if isinstance(v, (int, float)))}</table>
<h2>التلامس (% من الإطارات)</h2><table>{"".join(f"<tr><td>{k}</td><td>{v:.0f}%</td></tr>" for k, v in r["contact_pct"].items())}</table>
</body></html>'''
    open(path, 'w', encoding='utf-8').write(html)


def run(video, out, species='dog', grid='auto', panels=None, opts=None, mode='balanced', device='auto',
        max_frames=None, log=print):
    from .pipeline import _panels
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, 'detections_2d.npz')
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        kp, conf, fps, sizes = d['kp'], d['conf'], float(d['fps']), [tuple(s) for s in d['sizes']]
        log(f'loaded cached 2D detections ({cache})')
    else:
        pans, use = _panels(video, grid, panels, log)
        det = AnimalDetector(species, mode=mode, device=device)
        log(f'2D detector: ViTPose APT-36K ({SPECIES[species]["label"]}) on {det.device}')
        kp, conf, fps, sizes = detect_video(video, pans, use, det, out, max_frames, log=log)
        np.savez(cache, kp=kp, conf=conf, fps=fps, sizes=np.array(sizes), subject=species)
    log(f'3D reconstruction ({"single view" if len(sizes) == 1 else f"{len(sizes)} views"}) + smart correction...')
    res = reconstruct(kp, conf, sizes, fps, species, opts, log)
    export_all(res, out, log)
    return res
