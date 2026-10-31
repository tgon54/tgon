"""Virtual cameras, self-calibration (bundle adjustment on the subject's own body),
robust weighted triangulation and Viterbi left/right swap resolution."""
import itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .skeleton import J, SWAP_GROUPS, FLIP, IDX


# -------------------------------------------------------------- geometry --
def rodrigues(r):
    """r: (...,3) -> (...,3,3)"""
    th = np.linalg.norm(r, axis=-1, keepdims=True)
    k = r / np.maximum(th, 1e-12)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    Z = np.zeros_like(kx)
    Kx = np.stack([np.stack([Z, -kz, ky], -1), np.stack([kz, Z, -kx], -1), np.stack([-ky, kx, Z], -1)], -2)
    th = th[..., None]
    I = np.broadcast_to(np.eye(3), Kx.shape)
    return I + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def rot_to_rvec(R):
    import cv2
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def look_at(C, T, up=(0, 1, 0)):
    """World Y-up. Returns R (world->cam, rows x,y,z: right, down, forward), t."""
    z = np.asarray(T, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


class Cameras:
    """V pinhole cameras. params per cam: rvec(3), tvec(3), log f. cx, cy fixed to panel center."""

    def __init__(self, rvec, tvec, f, sizes, yaws=None):
        self.rvec = np.asarray(rvec, float)
        self.tvec = np.asarray(tvec, float)
        self.f = np.asarray(f, float)
        self.sizes = [tuple(s) for s in sizes]
        self.c = np.array([[w / 2, h / 2] for w, h in self.sizes], float)
        self.yaws = yaws

    @property
    def V(self):
        return len(self.f)

    @property
    def R(self):
        return rodrigues(self.rvec)

    def centers(self):
        return -np.einsum('vji,vj->vi', self.R, self.tvec)

    def vec(self):
        return np.concatenate([self.rvec, self.tvec, np.log(self.f)[:, None]], 1).ravel()

    @classmethod
    def from_vec(cls, x, sizes, yaws=None):
        x = x.reshape(-1, 7)
        return cls(x[:, :3], x[:, 3:6], np.exp(x[:, 6]), sizes, yaws)

    def project(self, X):
        """X: (...,3) -> uv (V,...,2), depth (V,...)."""
        R = self.R
        Xc = np.einsum('vij,...j->v...i', R, X) + self.tvec.reshape((self.V,) + (1,) * (X.ndim - 1) + (3,))
        z = Xc[..., 2]
        sh = (self.V,) + (1,) * (X.ndim - 1)
        uv = Xc[..., :2] / np.maximum(z[..., None], 1e-6) * self.f.reshape(sh)[..., None] + self.c.reshape(sh + (2,))
        return uv, z

    def P_normalized(self):
        """(V,3,4) projection matrices for normalized image coords."""
        return np.concatenate([self.R, self.tvec[:, :, None]], 2)

    def to_anipose_toml(self, path):
        """FreeMoCap / anipose compatible calibration file."""
        lines = []
        for v in range(self.V):
            w, h = self.sizes[v]
            f = self.f[v]
            lines += [f'[cam_{v}]', f'name = "cam_{v}"', f'size = [ {w}, {h},]',
                      f'matrix = [ [ {f}, 0.0, {w/2},], [ 0.0, {f}, {h/2},], [ 0.0, 0.0, 1.0,],]',
                      'distortions = [ 0.0, 0.0, 0.0, 0.0, 0.0,]',
                      'rotation = [ ' + ', '.join(f'{a}' for a in self.rvec[v]) + ',]',
                      'translation = [ ' + ', '.join(f'{a * 1000}' for a in self.tvec[v]) + ',]', '']
        lines += ['[metadata]', 'adjusted = false', 'error = 0.0', 'source = "aimocap virtual cameras"', '']
        open(path, 'w').write('\n'.join(lines))


def nominal_cameras(sizes, yaws, kp, conf, fov_deg=40.0, subject_h=1.7, cam_h=1.0, target_h=0.9):
    """Initial virtual cameras on a circle around the subject; distance from apparent size."""
    rv, tv, fs = [], [], []
    for v, (w, h) in enumerate(sizes):
        f = 0.5 * h / np.tan(np.radians(fov_deg) / 2)
        ok = conf[v] > 0.4
        ys = np.where(ok, kp[v, ..., 1], np.nan)
        hp = np.nanmedian(np.nanmax(ys, 1) - np.nanmin(ys, 1)) if np.isfinite(ys).any() else 0.6 * h
        D = float(np.clip(f * subject_h / max(hp, 10.0), 1.0, 30.0))
        a = np.radians(yaws[v])
        C = np.array([D * np.sin(a), cam_h, D * np.cos(a)])
        R, t = look_at(C, [0, target_h, 0])
        rv.append(rot_to_rvec(R)); tv.append(t); fs.append(f)
    return Cameras(rv, tv, fs, sizes, list(yaws))


# --------------------------------------------------------- triangulation --
def triangulate(cams, uv, w):
    """Weighted DLT. uv (V,...,2) pixels, w (V,...) >=0. Returns X (...,3); NaN if <2 views."""
    V = cams.V
    shp = uv.shape[1:-1]
    xn = (uv - cams.c.reshape((V,) + (1,) * len(shp) + (2,))) / cams.f.reshape((V,) + (1,) * len(shp) + (1,))
    P = cams.P_normalized()
    xn = np.moveaxis(xn, 0, -2).reshape(-1, V, 2)
    ww = np.moveaxis(w, 0, -1).reshape(-1, V)
    ww = np.where(np.isfinite(xn).all(-1), ww, 0.0)
    xn = np.nan_to_num(xn)
    A1 = xn[..., 0:1] * P[None, :, 2] - P[None, :, 0]
    A2 = xn[..., 1:2] * P[None, :, 2] - P[None, :, 1]
    A = np.concatenate([A1 * ww[..., None], A2 * ww[..., None]], 1)  # (N,2V,4)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1]
    X = Xh[:, :3] / np.where(np.abs(Xh[:, 3:]) < 1e-12, 1e-12, Xh[:, 3:])
    X[(ww > 0).sum(1) < 2] = np.nan
    return X.reshape(shp + (3,))


def reproj_error(cams, X, uv):
    """Error normalized by panel height. (V,...)"""
    p, z = cams.project(X)
    H = np.array([s[1] for s in cams.sizes], float).reshape((cams.V,) + (1,) * (X.ndim - 1))
    e = np.linalg.norm(p - uv, axis=-1) / H
    return np.where(z > 0, e, 1.0)


def robust_triangulate(cams, uv, conf, thr=0.03, min_conf=0.3):
    """Iteratively drop the worst view where reprojection error > thr (while >=3 views remain)."""
    w = np.where(conf >= min_conf, conf, 0.0)
    w = np.where(np.isfinite(uv).all(-1), w, 0.0)
    for _ in range(cams.V - 2):
        X = triangulate(cams, uv, w)
        e = reproj_error(cams, np.nan_to_num(X), np.nan_to_num(uv))
        e = np.where(w > 0, e, -1)
        nv = (w > 0).sum(0)
        worst = e.argmax(0)
        bad = (e.max(0) > thr) & (nv > 2)
        if not bad.any():
            break
        m = np.zeros_like(w, bool)
        np.put_along_axis(m, worst[None], bad[None], 0)
        w = np.where(m, 0.0, w)
    X = triangulate(cams, uv, w)
    e = reproj_error(cams, np.nan_to_num(X), np.nan_to_num(uv))
    used = w > 0
    nv = used.sum(0)
    emean = np.where(used, e, 0).sum(0) / np.maximum(nv, 1)
    cmean = np.where(used, conf, 0).sum(0) / np.maximum(nv, 1)
    q = cmean * np.minimum(1.0, nv / 3.0) * np.exp(-emean / thr)
    q[~np.isfinite(X).all(-1)] = 0
    return X, e, used, q


# ---------------------------------------------------------- swap (Viterbi) --
def resolve_swaps(cams, uv, conf, ref=0, lam=0.02, min_conf=0.3):
    """For every symmetric joint group, decide per frame which views have left/right
    labels swapped. Hidden state = swap pattern of non-reference views; emission =
    reprojection error of triangulating the group; transition = lam * changes.
    Returns corrected uv, conf and decisions dict group->(T,V) bool."""
    V, T = uv.shape[:2]
    others = [v for v in range(V) if v != ref]
    states = list(itertools.product([0, 1], repeat=len(others)))
    uv2, cf2 = uv.copy(), conf.copy()
    decisions = {}
    for g, idx in SWAP_GROUPS.items():
        idx = np.array(idx)
        fl = FLIP[idx]
        cost = np.zeros((len(states), T))
        for si, st in enumerate(states):
            u = uv[:, :, idx].copy(); c = conf[:, :, idx].copy()
            for v, s in zip(others, st):
                if s:
                    u[v] = uv[v][:, fl]; c[v] = conf[v][:, fl]
            w = np.where(c >= min_conf, c, 0)
            X = triangulate(cams, u, w)
            e = reproj_error(cams, np.nan_to_num(X), np.nan_to_num(u))
            e = np.minimum(e, 0.15)
            cost[si] = (e * w).sum((0, 2)) / np.maximum(w.sum((0, 2)), 1e-6)
            cost[si][~np.isfinite(X).any(-1).any(-1)] = 0
        S = np.array(states)
        trans = lam * np.abs(S[:, None, :] - S[None, :, :]).sum(-1)
        D = cost[:, 0].copy(); back = np.zeros((len(states), T), int)
        for t in range(1, T):
            M = D[:, None] + trans
            back[:, t] = M.argmin(0)
            D = M.min(0) + cost[:, t]
        path = np.zeros(T, int); path[-1] = D.argmin()
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[path[t], t]
        dec = np.zeros((T, V), bool)
        for t in range(T):
            for v, s in zip(others, states[path[t]]):
                dec[t, v] = bool(s)
        for v in others:
            tt = np.where(dec[:, v])[0]
            if len(tt):
                a = uv[v][tt][:, idx]; b = uv[v][tt][:, fl]
                uv2[v, tt[:, None], idx[None]] = b
                uv2[v, tt[:, None], fl[None]] = a
                a = conf[v][tt][:, idx]; b = conf[v][tt][:, fl]
                cf2[v, tt[:, None], idx[None]] = b
                cf2[v, tt[:, None], fl[None]] = a
        decisions[g] = dec
    return uv2, cf2, decisions


# --------------------------------------------------- bundle adjustment ----
def bundle_adjust(cams, uv, conf, frames, prior=(1e3, 0.05), min_conf=0.5, max_nfev=200, log=None):
    """Self-calibration using the subject's own keypoints as the calibration target.
    Camera 0 is strongly anchored (gauge); others weakly regularized towards the
    nominal rig, which keeps AI-hallucinated geometry from drifting."""
    V = cams.V
    u = uv[:, frames].reshape(V, -1, 2)
    c = conf[:, frames].reshape(V, -1)
    w = np.where((c >= min_conf) & np.isfinite(u).all(-1), c, 0)
    X0 = triangulate(cams, u, w)
    good = np.isfinite(X0).all(-1) & ((w > 0).sum(0) >= 2)
    u, w, X0 = u[:, good], w[:, good], X0[good]
    M = len(X0)
    if M < 20:
        return cams, None
    obs_v, obs_m = np.nonzero(w > 0)
    H = np.array([s[1] for s in cams.sizes], float)
    x0c = cams.vec()
    pw = np.full(V * 7, prior[1]); pw[:7] = prior[0]

    def resid(x):
        cc = Cameras.from_vec(x[:V * 7], cams.sizes)
        X = x[V * 7:].reshape(M, 3)
        R = cc.R[obs_v]
        Xc = np.einsum('nij,nj->ni', R, X[obs_m]) + cc.tvec[obs_v]
        z = np.maximum(Xc[:, 2], 1e-3)
        p = Xc[:, :2] / z[:, None] * cc.f[obs_v, None] + cc.c[obs_v]
        r = (p - u[obs_v, obs_m]) / H[obs_v, None] * np.sqrt(w[obs_v, obs_m])[:, None]
        return np.concatenate([r.ravel(), pw * (x[:V * 7] - x0c)])

    n_obs = len(obs_v)
    S = lil_matrix((2 * n_obs + V * 7, V * 7 + M * 3), dtype=int)
    rows = np.arange(n_obs)
    for k in range(7):
        S[2 * rows, obs_v * 7 + k] = 1; S[2 * rows + 1, obs_v * 7 + k] = 1
    for k in range(3):
        S[2 * rows, V * 7 + obs_m * 3 + k] = 1; S[2 * rows + 1, V * 7 + obs_m * 3 + k] = 1
    for k in range(V * 7):
        S[2 * n_obs + k, k] = 1
    x0 = np.concatenate([x0c, X0.ravel()])
    r0 = resid(x0)[:2 * n_obs]
    res = least_squares(resid, x0, jac_sparsity=S, loss='soft_l1', f_scale=0.01,
                        x_scale='jac', max_nfev=max_nfev, method='trf')
    r1 = res.fun[:2 * n_obs]
    out = Cameras.from_vec(res.x[:V * 7], cams.sizes, cams.yaws)
    e0 = np.median(np.linalg.norm(r0.reshape(-1, 2), axis=1))
    e1 = np.median(np.linalg.norm(r1.reshape(-1, 2), axis=1))
    if log:
        log(f'    BA: {M} pts, {n_obs} obs, median weighted reproj {e0:.4f} -> {e1:.4f} (x panel height)')
    return out, e1


def candidate_yaw_sets(V, given=None):
    if given is not None:
        return [list(given)]
    step = 360.0 / V
    cw = [i * step for i in range(V)]
    ccw = [(-i * step) % 360 for i in range(V)]
    return [cw, ccw] if V > 2 else [cw]
