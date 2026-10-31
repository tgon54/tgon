"""Synthetic benchmark: known 3D motion -> imperfect 'AI' multi-view 2D detections
(wrong camera layout guesses, noise, dropouts, L/R swaps, morph glitches,
hallucinated views) -> pipeline -> error vs ground truth."""
import numpy as np

from .skeleton import IDX, J, FLIP, SWAP_GROUPS
from .cameras import Cameras, look_at, rot_to_rvec


def _rx(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _ik(hip, ankle, l1, l2, fwd):
    d = ankle - hip; L = np.linalg.norm(d)
    L = min(L, l1 + l2 - 1e-4)
    a = (l1 ** 2 - l2 ** 2 + L ** 2) / (2 * L)
    h = np.sqrt(max(l1 ** 2 - a ** 2, 0))
    u = d / np.linalg.norm(d)
    f = fwd - (fwd @ u) * u; f /= np.linalg.norm(f)
    return hip + a * u + h * f


def generate_motion(T=240, fps=30.0):
    P = np.zeros((T, J, 3))
    for t in range(T):
        s = t / fps
        squat = 0.30 * (0.5 - 0.5 * np.cos(2 * np.pi * s / 2.4)) if s < 4.8 else 0.04
        step = s >= 4.8
        ph = 0.94 - squat
        lean = np.radians(45) * squat / 0.30
        twist = np.radians(18) * np.sin(2 * np.pi * s / 3.0)
        R = _ry(twist) @ _rx(lean)
        pel = np.array([0, ph, 0.0])
        g = {}
        g['hip'] = pel
        g['left_hip'] = pel + _ry(twist * .3) @ np.array([0.1, 0, 0])
        g['right_hip'] = pel + _ry(twist * .3) @ np.array([-0.1, 0, 0])
        for side, sx in (('left', 1), ('right', -1)):
            lift = 0.0
            if step:
                ph2 = np.sin(2 * np.pi * (s - 4.8) / 1.2 + (0 if side == 'left' else np.pi))
                lift = 0.16 * max(0.0, ph2)
            an = np.array([0.12 * sx, 0.08 + lift, 0.02 + lift * 0.4])
            g[side + '_ankle'] = an
            g[side + '_knee'] = _ik(g[side + '_hip'], an, 0.45, 0.43, np.array([0, 0, 1.0]))
            g[side + '_heel'] = an + np.array([0, -0.06, -0.05])
            g[side + '_big_toe'] = an + np.array([-0.015 * sx, -0.07, 0.15])
            g[side + '_small_toe'] = an + np.array([0.04 * sx, -0.07, 0.12])
        neck = pel + R @ np.array([0, 0.52, 0])
        g['neck'] = neck
        g['head'] = neck + R @ np.array([0, 0.25, 0.01])
        g['nose'] = neck + R @ np.array([0, 0.15, 0.10])
        for side, sx in (('left', 1), ('right', -1)):
            g[side + '_eye'] = g['nose'] + R @ np.array([0.032 * sx, 0.035, -0.015])
            g[side + '_ear'] = neck + R @ np.array([0.075 * sx, 0.14, -0.02])
            sh = neck + R @ np.array([0.18 * sx, 0, 0])
            g[side + '_shoulder'] = sh
            phase = 0 if side == 'left' else np.pi * 0.7
            flex = np.radians(10 + 80 * (0.5 - 0.5 * np.cos(2 * np.pi * s / 2.0 + phase)))
            abd = np.radians(15 + 25 * (0.5 + 0.5 * np.sin(2 * np.pi * s / 3.3)))
            dvec = np.array([np.sin(abd) * sx, -np.cos(abd) * np.cos(flex), np.cos(abd) * np.sin(flex)])
            el = sh + 0.30 * (R @ dvec)
            ef = flex + np.radians(20 + 60 * (0.5 + 0.5 * np.sin(2 * np.pi * s / 1.7 + phase)))
            fvec = np.array([np.sin(abd) * sx * .6, -np.cos(ef), np.sin(ef)])
            fvec /= np.linalg.norm(fvec)
            g[side + '_elbow'] = el
            g[side + '_wrist'] = el + 0.26 * (R @ fvec)
        for n, v in g.items():
            P[t, IDX[n]] = v
    return P


def true_cameras(size=(480, 540)):
    yaws = [0, 98, 175, 276]
    dists = [3.2, 3.7, 2.9, 3.4]
    hts = [1.05, 1.3, 0.9, 1.15]
    fovs = [40, 34, 47, 42]
    rv, tv, fs = [], [], []
    for y, d, h, fv in zip(yaws, dists, hts, fovs):
        a = np.radians(y)
        Cc = np.array([d * np.sin(a), h, d * np.cos(a)])
        R, t = look_at(Cc, [0.03, 0.85, 0.02])
        rv.append(rot_to_rvec(R)); tv.append(t); fs.append(0.5 * size[1] / np.tan(np.radians(fv) / 2))
    return Cameras(rv, tv, fs, [size] * 4, yaws)


def corrupt(P, cams, seed=1, noise_px=1.5):
    rng = np.random.default_rng(seed)
    uv, _ = cams.project(P)            # (V,T,J,2)
    V, T = uv.shape[:2]
    uv = uv + rng.normal(0, noise_px, uv.shape)
    conf = rng.uniform(0.6, 0.95, (V, T, J))
    log = {}
    # dropouts / occlusion
    m = rng.random((V, T, J)) < 0.05
    conf[m] = rng.uniform(0.0, 0.25, m.sum()); uv[m] += rng.normal(0, 25, (m.sum(), 2))
    # back view: detector mirrors legs in runs of frames; side view swaps arms sometimes
    def swap_runs(v, group, frac):
        idx = np.array(SWAP_GROUPS[group]); n = 0
        t = 0
        while t < T:
            if rng.random() < frac / 6:
                L = rng.integers(3, 12)
                tt = np.arange(t, min(T, t + L))
                a = uv[v][tt][:, idx].copy(); uv[v, tt[:, None], idx[None]] = uv[v][tt][:, FLIP[idx]]
                uv[v, tt[:, None], FLIP[idx][None]] = a
                n += len(tt); t += L
            t += 1
        return n
    log['swapped_leg_frames_back'] = swap_runs(2, 'legs', 0.35)
    log['swapped_arm_frames_side'] = swap_runs(1, 'arms', 0.15)
    log['swapped_leg_frames_side'] = swap_runs(3, 'legs', 0.10)
    # single keypoint glitches (AI morph)
    g = rng.random((V, T, J)) < 0.01
    uv[g] += rng.normal(0, 40, (g.sum(), 2))
    # whole-view morph frames
    wv = rng.random((V, T)) < 0.03
    uv[wv] += rng.normal(0, 30, (wv.sum(), 1, 2)) + rng.normal(0, 12, (wv.sum(), J, 2))
    log['morph_view_frames'] = int(wv.sum())
    # hallucinated view: view 3 shows a different moment for 12 frames
    uv[3, 100:112] = uv[3, 80:92]
    log['hallucinated_frames'] = 12
    return uv, conf, log


def similarity_align(X, Y):
    """Best similarity transform mapping X to Y (Umeyama). Both (N,3)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (Xc ** 2).sum(1).mean()
    return lambda Z: s * (Z - mx) @ R.T + my


def mpjpe(X, GT):
    ok = np.isfinite(X).all(-1)
    f = similarity_align(X[ok], GT[ok])
    e = np.linalg.norm(f(np.nan_to_num(X)) - GT, axis=-1)
    e[~ok] = np.nan
    return float(np.nanmean(e) * 1000), float(np.nanpercentile(e, 95) * 1000)
