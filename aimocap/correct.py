"""Smart correction engine: AI-glitch detection, gap filling, filtering, anatomical
and physical constraints. Every stage is measurable (see metrics())."""
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import binary_opening, binary_closing

from .skeleton import (IDX, J, BONE_IDX, SYM_BONES, FOOT, FLIP, center_of_mass)


# ------------------------------------------------------- 2D glitch filter --
def flag_2d_glitches(kp, conf, k=6.0):
    """AI videos morph/flicker. A keypoint that jumps away from its own temporal
    median (window 5) by more than k * robust spread is a glitch -> confidence 0.
    Also flags frames where the whole figure jumps (identity/morph glitch)."""
    V, T = kp.shape[:2]
    out = conf.copy()
    flags = np.zeros((V, T), bool)
    if T < 7:
        return out, flags
    for v in range(V):
        x = kp[v]
        hgt = np.nanmedian(np.nanmax(x[..., 1], 1) - np.nanmin(x[..., 1], 1))
        if not np.isfinite(hgt) or hgt <= 0:
            continue
        xf = np.nan_to_num(x, nan=0.0)
        med = np.stack([np.stack([medfilt(xf[:, j, a], 5) for a in range(2)], -1) for j in range(J)], 1)
        d = np.linalg.norm(xf - med, axis=-1) / hgt
        s = np.nanmedian(d, 0) * 1.4826 + 0.004
        bad = d > k * s[None]
        out[v][bad] = 0
        frame_bad = bad.mean(1) > 0.5
        flags[v] = frame_bad
        out[v][frame_bad] = 0
    return out, flags


def flag_inconsistent_views(err, used, thr):
    """A whole view in a frame is rejected when its mean reprojection error is far
    above threshold: typical of AI views that hallucinate a different pose."""
    e = np.where(used, err, np.nan)
    with np.errstate(all='ignore'):
        m = np.nanmean(e, -1)
    return np.nan_to_num(m, nan=0) > 2.5 * thr


# ------------------------------------------------------------- temporal ----
def fill_gaps(P, q, qmin=0.15, max_gap=None):
    """PCHIP interpolation over low-quality samples. Returns P, filled mask."""
    T = P.shape[0]
    P = P.copy()
    filled = np.zeros(P.shape[:2], bool)
    t = np.arange(T)
    for j in range(P.shape[1]):
        ok = (q[:, j] >= qmin) & np.isfinite(P[:, j]).all(-1)
        if ok.sum() < 2:
            continue
        miss = ~ok
        if max_gap:
            # do not invent very long segments: they stay interpolated but flagged
            pass
        f = PchipInterpolator(t[ok], P[ok, j], axis=0, extrapolate=False)
        P[miss, j] = f(t[miss])
        # edges: hold nearest
        first, last = t[ok][0], t[ok][-1]
        P[:first, j] = P[first, j]; P[last + 1:, j] = P[last, j]
        filled[miss, j] = True
    return P, filled


def hampel(P, win=7, k=4.0):
    """Remove 3D spikes (residual outliers) per joint/axis."""
    P = P.copy()
    T = P.shape[0]
    if T < win:
        return P, 0
    h = win // 2
    n = 0
    pad = np.pad(P, ((h, h), (0, 0), (0, 0)), mode='edge')
    windows = np.stack([pad[i:i + T] for i in range(win)], 0)
    med = np.median(windows, 0)
    mad = np.median(np.abs(windows - med), 0) * 1.4826 + 1e-4
    bad = np.abs(P - med) > k * mad
    bad = bad.any(-1, keepdims=True).repeat(3, -1)
    P[bad] = med[bad]
    return P, int(bad[..., 0].sum())


def butter_lowpass(P, fps, cutoff=6.0, order=4):
    """Zero-phase Butterworth (standard in biomechanics; Winter 2009)."""
    cutoff = min(cutoff, 0.4 * fps)
    if P.shape[0] < 3 * (order + 1) + 1:
        return P
    b, a = butter(order // 2, cutoff / (fps / 2))  # filtfilt doubles order
    return filtfilt(b, a, P, axis=0, method='gust')


# ------------------------------------------------------------- geometry ----
def _rot_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def pelvis_forward(P):
    x = P[:, IDX['left_hip']] - P[:, IDX['right_hip']]
    y = P[:, IDX['neck']] - P[:, IDX['hip']]
    f = np.cross(x, y)
    return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9)


def feet_forward(P):
    f = 0
    for s in ('left', 'right'):
        f = f + P[:, IDX[s + '_big_toe']] - P[:, IDX[s + '_heel']]
    return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9)


def fix_global_mirror(P):
    """If the whole body is labelled mirrored (all views agreed on a wrong swap),
    the pelvis 'forward' points backwards relative to the feet. Detect & flip labels."""
    d = (pelvis_forward(P) * feet_forward(P)).sum(-1)
    if np.nanmedian(d) < 0:
        return P[:, FLIP], True
    return P, False


def align_world(P, height=None, face_z=True):
    """Find the ground plane from foot points (RANSAC), make it y=0 with +Y up,
    face the subject towards +Z at the start, center at origin, scale to height."""
    feet = np.concatenate([P[:, FOOT['left']], P[:, FOOT['right']]], 1).reshape(-1, 3)
    feet = feet[np.isfinite(feet).all(-1)]
    up0 = np.nanmedian(P[:, IDX['neck']] - P[:, IDX['hip']], 0)
    up0 /= np.linalg.norm(up0)
    # candidate points: lowest 30% along current up
    hgt = feet @ up0
    cand = feet[hgt < np.percentile(hgt, 30)]
    rng = np.random.default_rng(0)
    best, bn = None, -1
    scale0 = np.nanmedian(np.linalg.norm(P[:, IDX['neck']] - P[:, IDX['hip']], axis=-1))
    tol = 0.04 * scale0 / 0.5
    for _ in range(300):
        s = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ up0 < 0:
            n = -n
        if n @ up0 < np.cos(np.radians(35)):
            continue
        inl = np.abs((cand - s[0]) @ n) < tol
        if inl.sum() > bn:
            bn, best = inl.sum(), (n, inl)
    n = best[0] if best is not None else up0
    inl = best[1] if best is not None else np.ones(len(cand), bool)
    # refine with SVD on inliers
    pts = cand[inl]
    c = pts.mean(0)
    nn = np.linalg.svd(pts - c)[2][-1]
    n = nn if nn @ n > 0 else -nn
    R = _rot_between(n, np.array([0, 1.0, 0]))
    Q = (P - c) @ R.T
    floor = np.percentile(np.concatenate([Q[:, FOOT['left'], 1], Q[:, FOOT['right'], 1]], 1), 2)
    Q[..., 1] -= floor
    if face_z:
        f = pelvis_forward(Q[: max(5, len(Q) // 10)]).mean(0)
        a = np.arctan2(f[0], f[2])
        ca, sa = np.cos(-a), np.sin(-a)
        Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        Q = Q @ Ry.T
    o = Q[0, IDX['hip']].copy(); o[1] = 0
    Q -= o
    scale = 1.0
    if height:
        top = Q[:, IDX['head'], 1]
        est = np.percentile(top, 90) * 1.03  # 'head' keypoint sits slightly below vertex
        scale = height / est
        Q *= scale
    return Q, scale


# ----------------------------------------------------------- anatomical ----
def bone_targets(P, q=None):
    """Robust bone lengths (median over confident frames), L/R averaged."""
    a, b = BONE_IDX[:, 0], BONE_IDX[:, 1]
    L = np.linalg.norm(P[:, b] - P[:, a], axis=-1)
    if q is not None:
        w = np.minimum(q[:, a], q[:, b]) > 0.4
        L = np.where(w, L, np.nan)
    tgt = np.nanmedian(L, 0)
    for i, k in SYM_BONES:
        m = np.nanmean([tgt[i], tgt[k]])
        tgt[i] = tgt[k] = m
    return tgt


def enforce_bones(P, L, q=None, iters=25, heavy=None):
    """Position-based dynamics: enforce constant bone lengths. Low-confidence joints
    move more than confident ones. heavy: (T,J) bool joints pinned (e.g. feet in contact)."""
    P = P.copy()
    T = P.shape[0]
    inv = 1.0 / (np.clip(q, 0, 1) + 0.1) if q is not None else np.ones((T, J))
    if heavy is not None:
        inv = np.where(heavy, 1e-3, inv)
    for _ in range(iters):
        for (a, b), l in zip(BONE_IDX, L):
            d = P[:, b] - P[:, a]
            n = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
            corr = (n - l) / n * d
            wa, wb = inv[:, a:a + 1], inv[:, b:b + 1]
            s = wa + wb
            P[:, a] += wa / s * corr
            P[:, b] -= wb / s * corr
    return P


def fix_knee_hyperextension(P, max_back=0.03):
    """A knee can only flex forward. If the knee sits behind the hip-ankle line
    (relative to foot direction) beyond tolerance, reflect it forward."""
    P = P.copy(); n = 0
    for s in ('left', 'right'):
        h, k, a = IDX[s + '_hip'], IDX[s + '_knee'], IDX[s + '_ankle']
        fwd = P[:, IDX[s + '_big_toe']] - P[:, IDX[s + '_heel']]
        line = P[:, a] - P[:, h]
        ll = np.linalg.norm(line, axis=-1, keepdims=True) + 1e-9
        ld = line / ll
        rel = P[:, k] - P[:, h]
        perp = rel - (rel * ld).sum(-1, keepdims=True) * ld
        fwd = fwd - (fwd * ld).sum(-1, keepdims=True) * ld
        fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9
        comp = (perp * fwd).sum(-1)
        bad = comp < -max_back * ll[:, 0]
        P[bad, k] -= 2 * comp[bad, None] * fwd[bad]
        n += int(bad.sum())
    return P, n


# -------------------------------------------------------------- physics ----
def foot_contacts(P, fps, h_thr=0.05, v_thr=0.25, min_len=3):
    """Contact = foot near ground AND nearly stationary (hysteresis by morphology)."""
    out = {}
    for s, idx in FOOT.items():
        h = P[:, idx, 1].min(1)
        c = P[:, idx].mean(1)
        v = np.zeros(len(P))
        v[1:] = np.linalg.norm(np.diff(c, axis=0), axis=-1) * fps
        v = medfilt(v, 3) if len(v) >= 3 else v
        m = (h < h_thr) & (v < v_thr)
        m = binary_closing(m, np.ones(3)) if len(m) > 3 else m
        m = binary_opening(m, np.ones(min_len)) if len(m) > min_len else m
        out[s] = m.astype(bool)
    return out


def _segments(m):
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def fix_foot_skating(P, contacts, ramp=4, max_shift=0.05):
    """Pin each foot during contact (no sliding) and blend in/out smoothly.
    Frames needing a shift > max_shift are treated as false contacts (left untouched)."""
    P = P.copy()
    T = len(P)
    pinned = np.zeros((T, J), bool)
    for s, idx in FOOT.items():
        delta = np.zeros((T, 3))
        wgt = np.zeros(T)
        for a, b in _segments(contacts[s]):
            anchor = np.median(P[a:b, idx], 0)      # (4,3) robust anchor
            anchor_c = anchor.mean(0)
            cur_c = P[a:b, idx].mean(1)             # (L,3)
            dl = anchor_c - cur_c
            dl[:, 1] = np.maximum(0, anchor[:, 1].min()) - P[a:b, idx, 1].min(1)  # sit on ground
            ok = np.linalg.norm(dl, axis=-1) < max_shift
            delta[a:b] = dl; wgt[a:b] = ok
            for r in range(1, ramp + 1):            # blend outside segment
                al = 1 - r / (ramp + 1)
                if a - r >= 0 and wgt[a - r] < al:
                    delta[a - r] = dl[0]; wgt[a - r] = al
                if b - 1 + r < T and wgt[b - 1 + r] < al:
                    delta[b - 1 + r] = dl[-1]; wgt[b - 1 + r] = al
            pinned[np.arange(a, b)[ok][:, None], np.array(idx)[None]] = True
        P[:, idx] += (delta * wgt[:, None])[:, None]
    return P, pinned


def clamp_ground(P):
    n = int((P[..., 1] < 0).sum())
    P = P.copy()
    P[..., 1] = np.maximum(P[..., 1], 0)
    return P, n


# -------------------------------------------------------------- metrics ----
def metrics(P, fps, contacts=None, L=None):
    a, b = BONE_IDX[:, 0], BONE_IDX[:, 1]
    bl = np.linalg.norm(P[:, b] - P[:, a], axis=-1)
    cv = np.nanmean(np.nanstd(bl, 0) / (np.nanmean(bl, 0) + 1e-9))
    acc = np.diff(P, 2, axis=0) * fps * fps
    jitter = float(np.nanmean(np.linalg.norm(acc, axis=-1)))
    out = {'bone_length_cv_pct': float(cv * 100), 'mean_accel_m_s2': jitter,
           'ground_penetration_frames': int((np.concatenate([P[:, FOOT['left'], 1], P[:, FOOT['right'], 1]], 1) < -0.01).any(1).sum())}
    if contacts is not None:
        sk = []
        for s, idx in FOOT.items():
            c = P[:, idx].mean(1)
            v = np.linalg.norm(np.diff(c[:, [0, 2]], axis=0), axis=-1) * fps
            m = contacts[s][1:] & contacts[s][:-1]
            if m.any():
                sk.append(v[m].mean())
        out['foot_skate_m_s'] = float(np.mean(sk)) if sk else 0.0
    return out
