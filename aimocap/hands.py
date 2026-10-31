"""Smart hand motion: 21-point hand keypoints per view (RTMPose-hand on crops placed from the
body's elbow/wrist), lifted to 3D with the calibrated virtual cameras, cleaned, and attached
to the corrected body wrists. Finger angles, joint limits and the relaxed-pose fallback are
solved at export time in `retarget._fingers`."""
import numpy as np

from .skeleton import IDX
from . import cameras as C
from . import correct as K

HAND_MODEL = ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
              'rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip')
NH = 21
SIDES = ('left', 'right')


class HandDetector:
    """RTMPose hand (21 kp) run on wrist-centred crops. Same device logic as the body model."""

    def __init__(self, device='cpu'):
        from rtmlib import RTMPose
        dev = 'cuda' if str(device).startswith('cuda') else 'cpu'
        try:
            self.model = RTMPose(HAND_MODEL, model_input_size=(256, 256), backend='onnxruntime', device=dev)
        except Exception:
            if dev == 'cpu':
                raise
            self.model = RTMPose(HAND_MODEL, model_input_size=(256, 256), backend='onnxruntime', device='cpu')

    def __call__(self, img, boxes):
        """boxes: list of [x0,y0,x1,y1] or None -> kp (len,21,2), conf (len,21)."""
        kp = np.full((len(boxes), NH, 2), np.nan)
        sc = np.zeros((len(boxes), NH))
        valid = [i for i, b in enumerate(boxes) if b is not None]
        if valid:
            k, s = self.model(img, bboxes=[boxes[i] for i in valid])
            kp[valid] = np.asarray(k, float).reshape(-1, NH, 2)
            sc[valid] = np.asarray(s, float).reshape(-1, NH)
        return kp, sc


def hand_boxes(k, s, shape, min_conf=0.25):
    """Square crop per hand from the body keypoints of one panel: centred past the wrist
    along the forearm, sized from forearm length (works for any hand orientation)."""
    H, W = shape[:2]
    out = []
    for side in SIDES:
        el, wr = IDX[side + '_elbow'], IDX[side + '_wrist']
        if s[el] < min_conf or s[wr] < min_conf or not np.isfinite(k[[el, wr]]).all():
            out.append(None); continue
        f = k[wr] - k[el]
        L = max(np.linalg.norm(f), 8.0)
        c = k[wr] + 0.45 * f
        h = 0.75 * L
        x0, y0, x1, y1 = c[0] - h, c[1] - h, c[0] + h, c[1] + h
        if x1 < 0 or y1 < 0 or x0 > W or y0 > H:
            out.append(None); continue
        out.append([max(0.0, x0), max(0.0, y0), min(W - 1.0, x1), min(H - 1.0, y1)])
    return out


def detect_panel_hands(det, crop, k, s):
    """-> (2,21,2) kp, (2,21) conf for one panel image and its chosen body pose."""
    return det(crop, hand_boxes(k, s, crop.shape))


def detect_hands_video(video, panels, use, kp, conf, det, max_frames=None, log=print):
    """Second pass for cached body detections: (V,T,2,21,2), (V,T,2,21)."""
    import cv2
    V, T = kp.shape[:2]
    HK = np.full((V, T, 2, NH, 2), np.nan)
    HS = np.zeros((V, T, 2, NH))
    cap = cv2.VideoCapture(video)
    t = 0
    while t < T:
        ok, fr = cap.read()
        if not ok or (max_frames and t >= max_frames):
            break
        for v, pi in enumerate(use):
            x0, y0, x1, y1 = panels[pi]
            HK[v, t], HS[v, t] = detect_panel_hands(det, fr[y0:y1, x0:x1], kp[v, t], conf[v, t])
        t += 1
        if t % 10 == 0:
            log(f'  hand detection: frame {t}/{T}')
    cap.release()
    return HK, HS


# ------------------------------------------------------------------- 3D ----
def _similarity(A, B):
    """Least-squares s, R, t with B ~ s R A + t (Umeyama). A, B (N,3)."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum()
    return s, R, mb - s * R @ ma


def reconstruct_hands(cams, hkp, hconf, arm_swaps, body_tri, body_world, body_final, mirrored, fps,
                      thr=0.04, min_conf=0.3, cutoff=8.0):
    """hkp (V,T,2,21,2) panel px, hconf (V,T,2,21). arm_swaps (T,V) bool from resolve_swaps.
    body_tri: triangulated body (camera space, labels before mirror fix);
    body_world: same frames after world alignment; body_final: corrected body.
    Returns dict(left, right (T,21,3) m, conf_left, conf_right (T,21))."""
    V, T = hkp.shape[:2]
    uv = hkp.copy(); cf = hconf.copy()
    sw = arm_swaps.T                                      # (V,T)
    uv[sw] = uv[sw][:, ::-1]; cf[sw] = cf[sw][:, ::-1]    # hands follow the arm L/R decision
    uv = uv.reshape(V, T, 2 * NH, 2); cf = cf.reshape(V, T, 2 * NH)
    X, err, used, q = C.robust_triangulate(cams, uv, cf, thr=thr, min_conf=min_conf)
    X = X.reshape(T, 2, NH, 3); q = q.reshape(T, 2, NH)
    if mirrored:
        X = X[:, ::-1]; q = q[:, ::-1]
    # camera space -> world: the global similarity used by align_world
    A, B = body_tri.reshape(-1, 3), body_world.reshape(-1, 3)
    ok = np.isfinite(A).all(-1) & np.isfinite(B).all(-1)
    s, R, t = _similarity(A[ok], B[ok])
    X = s * X @ R.T + t
    out = {}
    for i, side in enumerate(SIDES):
        H, qh = X[:, i], q[:, i]
        if (qh > 0.15).sum() < 3:
            out[side] = None; out['conf_' + side] = None; continue
        qh = np.where(np.isfinite(H).all(-1), np.clip(qh / 0.6, 0, 1), 0.0)
        rel = H - H[:, :1]
        rel, cq, _ = sanitize_hand(rel, qh, fps)
        rel = K.butter_lowpass(rel, fps, cutoff)
        rel[:, 0] = 0
        # attach to the corrected wrist (body corrections move the arm slightly)
        out[side] = rel + body_final[:, IDX[side + '_wrist']][:, None]
        out['conf_' + side] = cq
    return out


# ------------------------------------------------------------- sanitize --
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
# Adult hand segment lengths (m) per chain: wrist/CMC->MCP, proximal, middle, distal.
HAND_DEFAULT = [[0.035, 0.040, 0.032, 0.027], [0.090, 0.045, 0.025, 0.020], [0.090, 0.045, 0.028, 0.020],
                [0.085, 0.042, 0.027, 0.020], [0.080, 0.035, 0.020, 0.018]]


def sanitize_hand(R, C, fps, strength=0.5, rng=None):
    """Remove anatomically impossible finger points (AI-video hands are tiny and the detector
    often lands on the other hand, the body or the background).

    R (T,21,3) wrist-relative points, C (T,21) confidence 0..1. A point is rejected when its
    bone length is off the hand's own (robust) length, it is too far from the wrist, it jumps
    away from its neighbours in time, or the palm itself is malformed. Rejected points get
    confidence 0 and are re-interpolated from good frames; with confidence 0 the exporter
    blends that finger to the natural relaxed hand. strength 0..1 tightens the limits.
    `rng` (a, b) limits the changes to those frames. Returns R, C, fraction rejected."""
    T = len(R)
    R = np.array(R, float); C = np.array(C, float)
    a, b = (0, T - 1) if rng is None else rng
    inside = np.zeros(T, bool); inside[a:b + 1] = True
    k = float(np.clip(strength, 0, 1))
    lo, hi = 0.62 + 0.13 * k, 1.6 - 0.3 * k            # allowed bone length ratio
    jump = 0.05 - 0.025 * k                            # m, out-and-back jump vs 5-frame median
    ok = np.isfinite(R).all(-1) & (C > 0.05)
    Rn = np.where(np.isfinite(R), R, 0.0)
    bad = ~ok
    # robust per-hand bone lengths, kept within a plausible range of an adult hand
    ref = np.array(HAND_DEFAULT)
    good = ok & (C > 0.5)
    for ci, ch in enumerate(CHAINS):
        for j in range(4):
            p, c = ch[j], ch[j + 1]
            m = good[:, p] & good[:, c]
            if m.sum() >= 8:
                L = np.median(np.linalg.norm(Rn[m, c] - Rn[m, p], axis=-1))
                ref[ci, j] = np.clip(L, 0.65 * HAND_DEFAULT[ci][j], 1.45 * HAND_DEFAULT[ci][j])
    # 1) bone lengths along each chain (a bad joint also invalidates everything after it)
    for ci, ch in enumerate(CHAINS):
        broken = np.zeros(T, bool)
        for j in range(4):
            p, c = ch[j], ch[j + 1]
            r = np.linalg.norm(Rn[:, c] - Rn[:, p], axis=-1) / ref[ci, j]
            broken |= (r < lo) | (r > hi) | bad[:, c]
            bad[:, c] |= broken
    # 2) reach: nothing further from the wrist than a stretched hand
    reach = ref[2].sum() * 1.25
    bad |= np.linalg.norm(Rn, axis=-1) > reach
    # 3) palm shape: index/pinky MCPs must span a sane width and angle
    w = np.linalg.norm(Rn[:, 5] - Rn[:, 17], axis=-1)
    v1, v2 = Rn[:, 5], Rn[:, 17]
    ang = np.degrees(np.arccos(np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9), -1, 1)))
    palm_bad = (w < 0.025) | (w > 0.13) | (ang < 8) | (ang > 70) | bad[:, [5, 9, 17]].any(1)
    bad |= palm_bad[:, None]
    # 4) temporal glitches
    if T > 6:
        from scipy.ndimage import median_filter
        med = median_filter(Rn, size=(5, 1, 1), mode='nearest')
        bad |= np.linalg.norm(Rn - med, axis=-1) > jump
    bad[:, 0] = False
    bad &= inside[:, None]
    C2 = np.where(bad, 0.0, C)
    # re-interpolate rejected points from good frames (inside the range only)
    R2, filled = K.fill_gaps(Rn, np.where(bad | ~ok, 0.0, 1.0), qmin=0.5)
    R2 = np.where(inside[:, None, None], R2, Rn)
    R2[:, 0] = 0
    # still-unknown points (no good frame at all): park them on the relaxed default hand
    nogood = ~(np.isfinite(R2).all(-1))
    if nogood.any():
        R2 = np.where(nogood[..., None], 0.0, R2)
    C2 = np.where(filled & inside[:, None], np.minimum(C2, 0.0), C2)
    return R2, C2, float(bad[inside].mean()) if inside.any() else 0.0
