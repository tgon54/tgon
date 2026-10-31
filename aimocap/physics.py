"""Physics & anthropometry for natural, physically plausible motion.

* Anthropometric priors (Drillis & Contini 1966): bone lengths kept within a natural range of
  the subject's height (critical for single-camera depth).
* Gravity: during flight (no foot contact) the centre of mass follows a ballistic parabola.
* Ground reaction force can only push: COM vertical acceleration >= -g, bounded above, and the
  horizontal acceleration stays inside the friction cone.
* Static balance: when (quasi-)standing, the COM projects inside the support polygon of the feet.
* Body weight: ground reaction force, peak loads and per-frame force export (needs the mass).

Works for any skeleton: pass the COM function and the foot (paw) joint groups."""
import numpy as np

from .correct import butter_lowpass

G = 9.81
MU = 0.8                         # foot/ground friction coefficient
A_UP_MAX = 4.0 * G               # max upward COM acceleration (landing/jump push-off)
A_SMOOTH_HZ = 4.0

# Drillis & Contini (1966) segment lengths as a fraction of body height.
ANTHRO = {
    ('left_hip', 'left_knee'): 0.245, ('right_hip', 'right_knee'): 0.245,
    ('left_knee', 'left_ankle'): 0.246, ('right_knee', 'right_ankle'): 0.246,
    ('left_shoulder', 'left_elbow'): 0.186, ('right_shoulder', 'right_elbow'): 0.186,
    ('left_elbow', 'left_wrist'): 0.146, ('right_elbow', 'right_wrist'): 0.146,
    ('hip', 'neck'): 0.288,
}
# Widths are left out on purpose: detectors mark joint centres, not body breadth.


def anthropometric_lengths(L, bones, height, tol=0.15):
    """Clamp estimated bone lengths L (per `bones` (a,b) pairs) to +-tol of the natural ratio."""
    L = np.array(L, float)
    n = 0
    for i, (a, b) in enumerate(bones):
        r = ANTHRO.get((a, b)) or ANTHRO.get((b, a))
        if r is None:
            continue
        lo, hi = r * height * (1 - tol), r * height * (1 + tol)
        v = float(np.clip(L[i], lo, hi))
        n += abs(v - L[i]) > 1e-6
        L[i] = v
    return L, n


def _segments(mask):
    out, t, T = [], 0, len(mask)
    while t < T:
        if mask[t]:
            e = t
            while e + 1 < T and mask[e + 1]:
                e += 1
            out.append((t, e))
            t = e + 1
        else:
            t += 1
    return out


def _acc(c, fps):
    a = np.zeros_like(c)
    if len(c) > 2:
        a[1:-1] = (c[2:] - 2 * c[1:-1] + c[:-2]) * fps * fps
        a[0], a[-1] = a[1], a[-2]
    return a


def feasible_mask(a, airborne):
    """Per-frame check of COM acceleration against contact physics."""
    ay, ah = a[:, 1], np.linalg.norm(a[:, [0, 2]], axis=-1)
    stance_ok = (ay >= -G - 0.5) & (ay <= A_UP_MAX) & (ah <= MU * np.maximum(ay + G, 0) + 1.0)
    flight_ok = (np.abs(ay + G) < 2.0) & (ah < 2.0)
    return np.where(airborne, flight_ok, stance_ok)


def flight_fix(P, com, fps, airborne, max_s=1.2):
    """Make the COM follow a ballistic parabola during each flight phase (translation only)."""
    T = len(P)
    delta = np.zeros((T, 3))
    n = 0
    accepted = np.zeros(T, bool)
    for a, b in _segments(airborne):
        if b - a + 1 < 3 or (b - a + 1) / fps > max_s or a == 0 or b == T - 1:
            continue
        i0, i1 = a - 1, b + 1
        dt = (i1 - i0) / fps
        t = (np.arange(i0, i1 + 1) - i0) / fps
        y0, y1 = com[i0, 1], com[i1, 1]
        v0 = (y1 - y0 + 0.5 * G * dt * dt) / dt
        y = y0 + v0 * t - 0.5 * G * t * t
        # a real flight of this duration lifts the COM by g*dt^2/8; if the measured COM does not
        # rise at all, the "flight" is a missed foot contact -> leave it alone
        rise = com[a:b + 1, 1].max() - max(y0, y1)
        if y.max() - max(y0, y1) > 0.8 or rise < 0.3 * G * dt * dt / 8:
            continue
        xz = com[i0, [0, 2]] + (com[i1, [0, 2]] - com[i0, [0, 2]]) * (t / dt)[:, None]
        tgt = np.column_stack([xz[:, 0], y, xz[:, 1]])
        delta[i0:i1 + 1] = tgt - com[i0:i1 + 1]
        accepted[a:b + 1] = True
        n += 1
    return P + delta[:, None], n, accepted


def project_com(com, fps, airborne, iters=60):
    """Smallest smooth change of the COM path that satisfies the contact physics.
    Confirmed flight frames are already exact parabolas and are not touched."""
    c = com.copy()
    for _ in range(iters):
        a = _acc(c, fps)
        bad = ~feasible_mask(a, airborne) & ~airborne
        bad[[0, -1]] = False
        if not bad.any():
            break
        idx = np.where(bad)[0]
        # move offending samples towards the neighbour average (reduces |a| locally)
        c[idx] = 0.5 * c[idx] + 0.25 * (c[idx - 1] + c[idx + 1])
    return c


def balance_fix(P, com, foot_pts, contact_any, fps, margin=0.03, max_shift=0.15, double=None):
    """Quasi-static frames: shift the body so the COM ground projection lies inside the support
    polygon (bounding region of the contacting foot points)."""
    T = len(P)
    a = _acc(com, fps)
    v = np.gradient(com, 1.0 / fps, axis=0) if T > 1 else np.zeros_like(com)
    # only genuinely still stances (walking/running COM is legitimately outside the feet);
    # need >= 0.5 s of stillness
    still = contact_any & (np.linalg.norm(v[:, [0, 2]], axis=-1) < 0.15) & (np.linalg.norm(a[:, [0, 2]], axis=-1) < 0.8)
    double = contact_any if double is None else double
    quasi = np.zeros(T, bool)
    for s0, s1 in _segments(still):
        dur = (s1 - s0 + 1) / fps
        # standing on both feet >= 0.5 s, or on one foot >= 1 s (stepping is dynamic, not static)
        if (dur >= 0.5 and double[s0:s1 + 1].all()) or dur >= 1.0:
            quasi[s0:s1 + 1] = True
    shift = np.zeros((T, 3))
    for t in np.where(quasi)[0]:
        pts = foot_pts[t]
        pts = pts[np.isfinite(pts).all(-1)]
        if len(pts) < 2:
            continue
        lo, hi = pts[:, [0, 2]].min(0) - margin, pts[:, [0, 2]].max(0) + margin
        p = com[t, [0, 2]]
        q = np.clip(p, lo, hi)
        d = q - p
        nd = np.linalg.norm(d)
        if nd > max_shift:
            d *= max_shift / nd
        shift[t, [0, 2]] = d
    if T > 15:
        shift = butter_lowpass(shift, fps, 2.0)
    return P + shift[:, None], float((np.abs(shift).sum(-1) > 0.005).mean() * 100)


def grf(com, fps, mass):
    """Ground reaction force (N) from COM acceleration: F = m (a + g)."""
    a = butter_lowpass(_acc(com, fps), fps, 6.0) if len(com) > 15 else _acc(com, fps)
    F = mass * (a + np.array([0, G, 0]))
    F[:, 1] = np.maximum(F[:, 1], 0)
    return F


def simulate(P, fps, com_fn, feet, contacts, mass=70.0, repin=None, log=print):
    """Physics pass. P (T,J,3) m Y-up. feet: {side: [joint idx]}; contacts: {side: (T,) bool}.
    repin(P) -> P re-applies foot pinning / bone lengths after whole-body shifts.
    Returns P, report dict."""
    rep = {}
    T = len(P)
    cont = np.zeros(T, bool)
    for s in feet:
        cont |= np.asarray(contacts.get(s, np.zeros(T, bool)), bool)
    low = np.min(np.stack([P[:, feet[s], 1].min(1) for s in feet], 1), 1)
    airborne = (~cont) & (low > 0.03)
    # flight phases need a few frames to be trusted
    air = np.zeros(T, bool)
    for a, b in _segments(airborne):
        if b - a + 1 >= 3:
            air[a:b + 1] = True
    if air.mean() > 0.5:          # contact detection unreliable (e.g. feet never reach the floor)
        log(f'  physics: feet rarely on the ground ({air.mean() * 100:.0f}% "airborne") -> flight/contact limits skipped')
        air[:] = False
    com0 = com_fn(P)
    P, nfl, air = flight_fix(P, com0, fps, air)          # only physically confirmed flights count
    rep['flight_phases'] = nfl
    rep['physics_violation_pct_before'] = float((~feasible_mask(_acc(com0, fps), air)).mean() * 100)
    com = com_fn(P)
    c2 = project_com(com, fps, air)
    P = P + (c2 - com)[:, None]
    foot_pts = np.concatenate([np.where(np.asarray(contacts[s], bool)[:, None, None], P[:, feet[s]], np.nan)
                               for s in feet], 1)
    nfeet = np.sum([np.asarray(contacts.get(s, np.zeros(T, bool)), bool) for s in feet], 0)
    P, rep['balance_fixed_pct'] = balance_fix(P, com_fn(P), foot_pts, cont, fps, double=nfeet >= 2)
    if repin is not None:
        P = repin(P)
    com = com_fn(P)
    rep['physics_violation_pct_after'] = float((~feasible_mask(_acc(com, fps), air)).mean() * 100)
    F = grf(com, fps, mass)
    bw = mass * G
    rep.update(mass_kg=float(mass), airborne_pct=float(air.mean() * 100),
               peak_grf_bw=float(F[:, 1].max() / bw), mean_grf_bw=float(F[:, 1].mean() / bw))
    log(f"  physics: violations {rep['physics_violation_pct_before']:.0f}% -> {rep['physics_violation_pct_after']:.0f}%, "
        f"flight phases {nfl}, balance fixes {rep['balance_fixed_pct']:.0f}%, peak GRF {rep['peak_grf_bw']:.1f} BW")
    return P, rep, F, air


def write_physics_csv(path, com, F, air, fps, mass):
    T = len(com)
    v = np.gradient(com, 1.0 / fps, axis=0) if T > 1 else np.zeros_like(com)
    rows = np.column_stack([np.arange(T), np.arange(T) / fps, com, v, F, F[:, 1] / (mass * G), air.astype(int)])
    hdr = 'frame,time_s,com_x,com_y,com_z,com_vx,com_vy,com_vz,grf_x_N,grf_y_N,grf_z_N,grf_y_bw,airborne'
    np.savetxt(path, rows, delimiter=',', header=hdr, comments='', fmt='%.4f')
