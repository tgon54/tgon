"""Video input: multi-view grid detection, panel splitting and 2D keypoint detection."""
import os
import numpy as np
import cv2

from .skeleton import J, BONE_IDX


# ---------------------------------------------------------------- grid ----
def _separators(profile, length, min_gap):
    """Find separator lines in a 1-D 'edge-ness' profile. Candidate lines are
    matched against equally spaced grids (AI split-screens are near-uniform), so a
    floor line or horizon crossing all panels is not mistaken for a separator."""
    thr = profile.mean() + 2.5 * profile.std()
    idx = np.where(profile > thr)[0]
    groups, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > 2:
            groups.append(cur); cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    cuts = [int(np.mean(g)) for g in groups]
    cuts = [c for c in cuts if min_gap < c < length - min_gap]
    merged = []
    for c in cuts:
        if merged and c - merged[-1][-1] <= min_gap // 4:
            merged[-1].append(c)           # both edges of the same separator line
        else:
            merged.append([c])
    cand = [int(round(np.mean(g))) for g in merged]
    if not cand:
        return []
    strength = {c: float(profile[max(0, c - 3):c + 4].max()) for c in cand}
    best, best_score = None, -1.0
    for n in range(2, 7):
        tol = 0.04 * length
        picked = []
        for i in range(1, n):
            e = length * i / n
            near = [c for c in cand if abs(c - e) < tol]
            if not near:
                picked = None
                break
            picked.append(min(near, key=lambda c: abs(c - e)))
        if picked:
            score = float(np.mean([strength[c] for c in picked]))
            if score > best_score * 1.15 or (best is not None and score > 0.8 * best_score and n > len(best) + 1):
                best, best_score = picked, score
    if best is not None:
        return best
    return []          # no uniform grid found along this axis (adjust manually in the UI / --grid)


def detect_grid(video, samples=12):
    """Auto-detect a panel grid. Uses temporal-median frame: separator lines are
    static and straight, so they survive the median while the moving figure fades."""
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or samples
    frames = []
    for k in np.linspace(0, max(0, n - 1), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if not frames:
        raise RuntimeError('cannot read video: ' + video)
    med = np.median(np.stack(frames), 0)
    H, W = med.shape
    gx = np.abs(cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3))
    # a separator is a column/row whose gradient is consistent along its full length
    colp = np.median(gx, 0)
    rowp = np.median(gy, 1)
    xs = _separators(colp, W, W // 12)
    ys = _separators(rowp, H, H // 12)
    return [0] + xs + [W], [0] + ys + [H]


def grid_from_spec(spec, W, H):
    r, c = [int(v) for v in spec.lower().split('x')]
    return [round(W * i / c) for i in range(c + 1)], [round(H * i / r) for i in range(r + 1)]


def panels_from_grid(xb, yb, pad=3):
    out = []
    for r in range(len(yb) - 1):
        for c in range(len(xb) - 1):
            out.append((xb[c] + pad, yb[r] + pad, xb[c + 1] - pad, yb[r + 1] - pad))
    return out


# ------------------------------------------------------------ detector ----
class Detector:
    """RTMPose Halpe-26 (body + feet) through rtmlib/ONNX. GPU if available."""

    def __init__(self, mode='performance', device='auto'):
        try:
            from rtmlib import BodyWithFeet
        except ImportError as e:
            raise RuntimeError('Missing packages (rtmlib/onnxruntime). Run install.bat '
                               '(or: python -m aimocap.setup_env) then restart.  [' + str(e) + ']')
        try:
            import onnxruntime as ort
            if hasattr(ort, 'preload_dlls'):       # CUDA/cuDNN DLLs installed via pip (nvidia-*)
                try:
                    ort.preload_dlls()
                except Exception:
                    pass
            has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
        except Exception:
            has_cuda = False
        if device == 'auto':
            device = 'cuda' if has_cuda else 'cpu'
        try:
            self.model = BodyWithFeet(mode=mode, backend='onnxruntime', device=device)
            self.model(np.zeros((256, 192, 3), np.uint8))       # smoke test
        except Exception:
            if device == 'cpu':
                raise
            device = 'cpu'
            self.model = BodyWithFeet(mode=mode, backend='onnxruntime', device=device)
        # onnxruntime silently falls back to CPU when CUDA DLLs are missing: report the truth
        try:
            prov = self.model.pose_model.session.get_providers()
            device = 'cuda (GPU)' if 'CUDAExecutionProvider' in prov else 'cpu'
        except Exception:
            pass
        self.device = device

    def __call__(self, img):
        kp, sc = self.model(img)
        kp = np.asarray(kp, float).reshape(-1, J, 2)
        sc = np.asarray(sc, float).reshape(-1, J)
        return kp, sc


def _pick_person(kp, sc, prev):
    """Select the subject: continuity with previous frame, else biggest confident person."""
    if len(kp) == 0:
        return None
    if prev is not None:
        d = [np.nanmedian(np.linalg.norm(k - prev, axis=1)) for k in kp]
        i = int(np.argmin(d))
        return i
    area = [(np.ptp(k[:, 0]) * np.ptp(k[:, 1])) * s.mean() for k, s in zip(kp, sc)]
    return int(np.argmax(area))


def detect_video(video, panels, use, detector, out_dir, max_frames=None, overlay=True, log=print):
    """Run 2D detection per panel. Returns kp (V,T,J,2) panel pixels, conf (V,T,J), fps, sizes."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    V = len(use)
    KP, SC = [[] for _ in range(V)], [[] for _ in range(V)]
    prev = [None] * V
    writer = None
    t = 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and t >= max_frames):
            break
        if overlay and writer is None:
            writer = cv2.VideoWriter(os.path.join(out_dir, 'debug_2d_overlay.mp4'),
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps, (fr.shape[1], fr.shape[0]))
        for v, pi in enumerate(use):
            x0, y0, x1, y1 = panels[pi]
            kp, sc = detector(fr[y0:y1, x0:x1])
            i = _pick_person(kp, sc, prev[v])
            if i is None:
                k, s = np.full((J, 2), np.nan), np.zeros(J)
            else:
                k, s = kp[i], sc[i]
                prev[v] = k
            KP[v].append(k); SC[v].append(s)
            if writer is not None and i is not None:
                for a, b in BONE_IDX:
                    if s[a] > .3 and s[b] > .3:
                        cv2.line(fr, (int(x0 + k[a, 0]), int(y0 + k[a, 1])),
                                 (int(x0 + k[b, 0]), int(y0 + k[b, 1])), (0, 255, 0), 2)
                for j in range(J):
                    col = (0, 255, 0) if s[j] > .6 else (0, 165, 255) if s[j] > .3 else (0, 0, 255)
                    cv2.circle(fr, (int(x0 + k[j, 0]), int(y0 + k[j, 1])), 3, col, -1)
                cv2.putText(fr, f'view {v}', (x0 + 6, y0 + 18), 0, .5, (255, 255, 255), 1)
        if writer is not None:
            writer.write(fr)
        t += 1
        if t % 10 == 0:
            log(f'  2D detection: frame {t}/{total}')
    cap.release()
    if writer is not None:
        writer.release()
    sizes = [(panels[pi][2] - panels[pi][0], panels[pi][3] - panels[pi][1]) for pi in use]
    return np.array(KP), np.array(SC), fps, sizes
