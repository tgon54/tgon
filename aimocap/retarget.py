"""Retarget AI-MoCap joints (Halpe-26 + optional 21-point hands) onto a game skeleton
(UE4 Mannequin / UE5 Manny naming).

Every driven bone gets an anatomical frame (primary direction + secondary axis) built
the same way from the mocap points and from the rig's rest pose; the bone's rotation is
the delta between the two frames applied to its rest orientation. That keeps the rig's
own rest pose (A-pose, bone rolls) intact, so the result imports onto the original
Unreal skeleton without any retargeting step."""
import re

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

from .skeleton import IDX
from .rig import rest_globals, local_matrix

# Hand keypoint layout (RTMPose hand / COCO-WholeBody hand, 21 points).
FINGERS = {'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8], 'middle': [9, 10, 11, 12],
           'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20]}
# Natural relaxed hand used when hand tracking is missing or unreliable:
# (MCP flex, PIP flex, DIP flex) in degrees.
RELAXED = {'index': (12, 22, 14), 'middle': (16, 28, 18), 'ring': (20, 32, 20), 'pinky': (24, 36, 22),
           'thumb': (10, 14, 12)}
# Anatomical limits (degrees): MCP flex, MCP abduction, PIP, DIP.
LIM_MCP, LIM_ABD, LIM_PIP, LIM_DIP = (-20, 90), (-22, 22), (0, 110), (-5, 85)
LIM_THUMB = (-15, 65)
IK_FOLLOW = {'ik_foot_l': 'foot_l', 'ik_foot_r': 'foot_r', 'ik_hand_gun': 'hand_r',
             'ik_hand_l': 'hand_l', 'ik_hand_r': 'hand_r'}


def _n(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _frame(d, s):
    d = _n(d)
    s = _n(s - (s * d).sum(-1, keepdims=True) * d)
    return np.stack([d, s, np.cross(d, s)], -1)


def _hinge(a, b, c, fallback, sign=1.0):
    n = np.cross(b - a, c - b) * sign
    mag = np.linalg.norm(n, axis=-1, keepdims=True) / (
        np.linalg.norm(b - a, axis=-1, keepdims=True) * np.linalg.norm(c - b, axis=-1, keepdims=True) + 1e-9)
    w = np.clip((mag - 0.08) / 0.25, 0, 1)
    return w * _n(n) + (1 - w) * _n(fallback)


def _slerp(Ra, Rb, w):
    """Per-frame slerp between rotation stacks (T,3,3) with weights w (T,) or scalar."""
    w = np.broadcast_to(np.asarray(w, float), (len(Ra),))
    rel = Rotation.from_matrix(np.swapaxes(Ra, -1, -2) @ Rb).as_rotvec()
    return Ra @ Rotation.from_rotvec(rel * w[:, None]).as_matrix()


def _swing(a, b):
    """Minimal rotation taking unit vectors a -> b, (T,3) -> (T,3,3)."""
    a, b = _n(a), _n(b)
    ax = np.cross(a, b)
    s = np.linalg.norm(ax, axis=-1, keepdims=True)
    ang = np.arctan2(s[..., 0], (a * b).sum(-1))
    fb = np.cross(a, np.where(np.abs(a[..., :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    ax = np.where(s > 1e-8, ax / np.maximum(s, 1e-12), _n(fb))
    return Rotation.from_rotvec(ax * ang[..., None]).as_matrix()


def _limit(Rrel, max_deg):
    rv = Rotation.from_matrix(Rrel).as_rotvec()
    a = np.linalg.norm(rv, axis=-1, keepdims=True)
    lim = np.radians(max_deg)
    return Rotation.from_rotvec(np.where(a > lim, rv * lim / np.maximum(a, 1e-9), rv)).as_matrix()


def _lowpass(x, fps, cutoff):
    if len(x) < 16 or cutoff <= 0 or cutoff >= fps / 2:
        return x
    b, a = butter(2, cutoff / (fps / 2))
    return filtfilt(b, a, x, axis=0)


def _twist_angle(R, axis):
    """Signed twist (radians) of R about unit axis (T,3) (swing-twist decomposition)."""
    q = Rotation.from_matrix(R).as_quat()          # x y z w
    p = (q[:, :3] * axis).sum(-1)
    return 2 * np.arctan2(p, q[:, 3])


# ---------------------------------------------------------------- rig map --
class RigMap:
    """Resolves semantic slots to the rig's bone names (UE naming; `prefix` handles namespaces)."""

    def __init__(self, rig):
        self.rig = rig
        self.bones = {b['name']: b for b in rig['bones']}
        names = list(self.bones)
        self.children = {}
        for b in rig['bones']:
            self.children.setdefault(b['parent'], []).append(b['name'])
        req = ['pelvis', 'thigh_l', 'calf_l', 'foot_l', 'thigh_r', 'calf_r', 'foot_r',
               'upperarm_l', 'lowerarm_l', 'hand_l', 'upperarm_r', 'lowerarm_r', 'hand_r', 'head']
        missing = [n for n in req if n not in self.bones]
        if missing:
            raise ValueError('target skeleton is missing UE bones: ' + ', '.join(missing))
        self.spine = sorted(n for n in names if re.fullmatch(r'spine_\d\d', n))
        self.neck = sorted(n for n in names if re.fullmatch(r'neck_\d\d', n))
        self.has = lambda n: n in self.bones

    def finger(self, side, f):
        out = [f'{f}_0{k}_{side}' for k in (1, 2, 3)]
        return out if all(n in self.bones for n in out) else None


# ------------------------------------------------------------ retargeting --
def _points(X, hands=None):
    """Mocap landmark dict (positions already in rig space)."""
    g = lambda n: X[:, IDX[n]]
    V = {
        'hip': g('hip'), 'neck': g('neck'), 'top': g('head'),
        'headmid': 0.5 * (g('left_ear') + g('right_ear')), 'lear': g('left_ear'), 'rear': g('right_ear'),
        'lhip': g('left_hip'), 'rhip': g('right_hip'), 'lkn': g('left_knee'), 'rkn': g('right_knee'),
        'lan': g('left_ankle'), 'ran': g('right_ankle'),
        'lto': 0.5 * (g('left_big_toe') + g('left_small_toe')), 'rto': 0.5 * (g('right_big_toe') + g('right_small_toe')),
        'lsh': g('left_shoulder'), 'rsh': g('right_shoulder'), 'lel': g('left_elbow'), 'rel': g('right_elbow'),
        'lwr': g('left_wrist'), 'rwr': g('right_wrist'),
    }
    V['spine'] = 0.5 * (V['hip'] + V['neck'])
    return V


def _rest_points(m, G):
    p = lambda n: G[n][None, :3, 3]
    V = {
        'hip': 0.5 * (p('thigh_l') + p('thigh_r')), 'neck': 0.5 * (p('upperarm_l') + p('upperarm_r')),
        'headmid': p('head'),
        'lhip': p('thigh_l'), 'rhip': p('thigh_r'), 'lkn': p('calf_l'), 'rkn': p('calf_r'),
        'lan': p('foot_l'), 'ran': p('foot_r'),
        'lto': p('ball_l') if m.has('ball_l') else p('foot_l') + [[0, -10.0, -8]],
        'rto': p('ball_r') if m.has('ball_r') else p('foot_r') + [[0, -10.0, -8]],
        'lsh': p('upperarm_l'), 'rsh': p('upperarm_r'), 'lel': p('lowerarm_l'), 'rel': p('lowerarm_r'),
        'lwr': p('hand_l'), 'rwr': p('hand_r'),
    }
    V['spine'] = 0.5 * (V['hip'] + V['neck'])
    return V


def _body_frames(V, up, left):
    """Anatomical frames (T,3,3) for body segments. `up`/`left` are fallback axes (T,3)."""
    pel_x = V['lhip'] - V['rhip']
    sh_x = V['lsh'] - V['rsh']
    chest_z = np.cross(sh_x, V['neck'] - V['spine'])
    legL = _hinge(V['lhip'], V['lkn'], V['lan'], pel_x)
    legR = _hinge(V['rhip'], V['rkn'], V['ran'], pel_x)
    armL = _hinge(V['lsh'], V['lel'], V['lwr'], np.cross(chest_z, V['lel'] - V['lsh']), sign=-1)
    armR = _hinge(V['rsh'], V['rel'], V['rwr'], np.cross(chest_z, V['rsh'] - V['rel']), sign=1)
    trunk_up = V['neck'] - V['hip']
    F = {
        'pelvis': _frame(trunk_up, pel_x),
        'chest': _frame(V['neck'] - V['spine'], sh_x),
        'clavicle_l': _frame(V['lsh'] - V['neck'], trunk_up),
        'clavicle_r': _frame(V['rsh'] - V['neck'], trunk_up),
        'upperarm_l': _frame(V['lel'] - V['lsh'], armL), 'lowerarm_l': _frame(V['lwr'] - V['lel'], armL),
        'upperarm_r': _frame(V['rel'] - V['rsh'], armR), 'lowerarm_r': _frame(V['rwr'] - V['rel'], armR),
        'thigh_l': _frame(V['lkn'] - V['lhip'], legL), 'calf_l': _frame(V['lan'] - V['lkn'], legL),
        'thigh_r': _frame(V['rkn'] - V['rhip'], legR), 'calf_r': _frame(V['ran'] - V['rkn'], legR),
        'foot_l': _frame(V['lto'] - V['lan'], legL), 'foot_r': _frame(V['rto'] - V['ran'], legR),
    }
    if 'lear' in V:
        # the 'head' keypoint sits forward of the skull top (constant forward pitch) and AI
        # face points are unreliable: the neutral is calibrated against the chest in retarget()
        F['head'] = _frame(V['top'] - V['headmid'], V['lear'] - V['rear'])
    else:
        F['head'] = _frame(up, left)
    return F


CLAVICLE_MAX, CLAVICLE_GAIN = 20.0, 0.6
ARM_TWIST_MAX, LEG_TWIST_MAX = 80.0, 40.0     # deg, humerus / femur axial rotation


def _neutralized(Fp, F0p, Fc, fps, max_deg, gain):
    """Child frame relative to parent with its clip-average as neutral; deviation scaled,
    limited and smoothed. Returns (live frame, rest frame)."""
    rel = np.swapaxes(Fp, -1, -2) @ Fc
    rn = Rotation.from_matrix(rel).mean().as_matrix()
    dev = Rotation.from_matrix(rn.T @ rel).as_rotvec() * gain
    a = np.linalg.norm(dev, axis=-1, keepdims=True)
    lim = np.radians(max_deg)
    dev = _lowpass(np.where(a > lim, dev * lim / np.maximum(a, 1e-9), dev), fps, 6.0)
    return Fp @ rn @ Rotation.from_rotvec(dev).as_matrix(), F0p @ rn


def _despike(P, thr=0.045):
    """Out-and-back keypoint glitches (a joint leaps away for 1-3 frames) -> 7-frame median,
    feathered into the clean signal. Real motion is preserved (median follows it)."""
    if len(P) < 9:
        return P
    from scipy.ndimage import median_filter
    M = median_filter(P, size=(7, 1, 1), mode='nearest')
    d = np.linalg.norm(P - M, axis=-1)
    w = np.clip((d - thr) / thr, 0, 1)
    w = np.maximum(w, median_filter(w, size=(3, 1), mode='nearest'))
    return P * (1 - w[..., None]) + M * w[..., None]


def _limb_secondary(Dpar, d, d0, s0, hinge, fps, lim_deg):
    """Twist-stable secondary axis for a limb segment.
    Reference = rest secondary carried by the parent delta and the minimal swing from the rest
    direction (never degenerate, unlike a cross product with a fixed axis). The elbow/knee
    hinge only adds a twist about the bone, weighted by how bent the joint is, limited to the
    anatomical range and smoothed -> no 180 deg flips when the limb straightens."""
    T = len(d)
    dn = _n(d)
    d0n = _n(d0)
    s0n = _n(s0 - (s0 @ d0n) * d0n)
    dl = np.einsum('tji,tj->ti', Dpar, dn)                      # direction in parent-rest space
    Rs = _swing(np.broadcast_to(d0n, (T, 3)), dl)
    ref = np.einsum('tij,tjk,k->ti', Dpar, Rs, s0n)
    ref = _n(ref - (ref * dn).sum(-1, keepdims=True) * dn)
    hn = hinge - (hinge * dn).sum(-1, keepdims=True) * dn
    bend = np.linalg.norm(hinge, axis=-1)
    hn = _n(hn)
    ang = np.arctan2((np.cross(ref, hn) * dn).sum(-1), (ref * hn).sum(-1))
    w = np.clip((bend - 0.15) / 0.35, 0, 1)                      # sin(bend angle): 9..30 deg
    w = np.where(np.abs(ang) > np.pi / 2, 0.0, w)                # bends backward: axis unreliable
    w = _lowpass(w, fps, 3.0).clip(0, 1)
    ang = _lowpass(np.clip(ang, -np.radians(lim_deg), np.radians(lim_deg)) * w, fps, 5.0)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return c * ref + s * np.cross(dn, ref)


WRIST_MAX = 70.0
HINGE_RANGE = (0.0, 150.0)   # deg, knee/elbow flexion


def _hinge_child(dp, dc, sec, sign):
    """Knee/elbow as a true 1-DOF hinge about `sec`: child direction = parent rotated by the
    flexion angle. Backward bends (AI depth errors) are reflected, range clamped."""
    x = _n(dp)
    y = sign * np.cross(sec, x)                                  # direction of normal flexion
    a = np.abs(np.arctan2((dc * y).sum(-1), (dc * x).sum(-1)))
    a = np.clip(a, *np.radians(HINGE_RANGE))[:, None]
    return np.cos(a) * x + np.sin(a) * y


def _stable_limbs(F, F0, V, V0, Dc, Dp, fps):
    """Rebuild arm/leg/foot frames (live and rest) with `_limb_secondary`."""
    def hinge(a, b, c, sign):
        u, v = _n(b - a), _n(c - b)
        return np.cross(u, v) * sign                             # |.| = sin(bend)
    up0 = _n(V0['neck'] - V0['hip'])[0]
    sh0 = (V0['lsh'] - V0['rsh'])[0]
    fwd0 = _n(np.cross(sh0, (V0['neck'] - V0['spine'])[0]))
    pel0 = (V0['lhip'] - V0['rhip'])[0]
    for k, sh, el, wr, sign in (('l', 'lsh', 'lel', 'lwr', -1.0), ('r', 'rsh', 'rel', 'rwr', 1.0)):
        d0u = (V0[el] - V0[sh])[0]
        s0 = np.cross(fwd0, d0u) * (1.0 if k == 'l' else -1.0)
        sec = _limb_secondary(Dc, V[el] - V[sh], d0u, s0, hinge(V[sh], V[el], V[wr], sign), fps, ARM_TWIST_MAX)
        F['upperarm_' + k] = _frame(V[el] - V[sh], sec)
        F['lowerarm_' + k] = _frame(_hinge_child(V[el] - V[sh], V[wr] - V[el], sec, sign), sec)
        F0['upperarm_' + k] = _frame(d0u[None], s0[None])
        F0['lowerarm_' + k] = _frame((V0[wr] - V0[el]), s0[None])
    for k, hp, kn, an, to in (('l', 'lhip', 'lkn', 'lan', 'lto'), ('r', 'rhip', 'rkn', 'ran', 'rto')):
        d0t = (V0[kn] - V0[hp])[0]
        sec = _limb_secondary(Dp, V[kn] - V[hp], d0t, pel0, hinge(V[hp], V[kn], V[an], 1.0), fps, LEG_TWIST_MAX)
        F['thigh_' + k] = _frame(V[kn] - V[hp], sec)
        shank = _hinge_child(V[kn] - V[hp], V[an] - V[kn], sec, 1.0)
        F['calf_' + k] = _frame(shank, sec)
        if k + 'foot_angles' in V:                                  # same ankle angles, corrected shank
            y, p, _ = V[k + 'foot_angles']
            fwd = _n(np.cross(V['lhip'] - V['rhip'], V['neck'] - V['hip']))
            toe = _foot_dir(_n(shank), fwd, y, p)
        else:
            toe = V[to] - V[an]
        F['foot_' + k] = _frame(toe, sec)
        F0['thigh_' + k] = _frame(d0t[None], pel0[None])
        F0['calf_' + k] = _frame(V0[an] - V0[kn], pel0[None])
        F0['foot_' + k] = _frame(V0[to] - V0[an], pel0[None])


NECK_MAX = 45.0             # deg, head motion around its neutral relative to the chest
FOOT_YAW_MAX = 30.0          # deg, ankle/subtalar rotation of the foot around the shank
FOOT_PITCH_RANGE = (-25.0, 45.0)   # deg around the clip's neutral (dorsi-, plantarflexion)


def _foot_dir(n, fwd, y, p):
    """Toe direction from ankle yaw/pitch relative to the shank `n` (pointing down)."""
    f = _n(fwd - (fwd * n).sum(-1, keepdims=True) * n)
    l = np.cross(n, f)
    return np.cos(p) * (np.cos(y) * f + np.sin(y) * l) + np.sin(p) * n


def _fix_feet(V, fps):
    """AI-video foot keypoints often flip heel/toe or swing sideways. Re-express each
    ankle->toe direction in the leg's own frame, reject impossible frames (toe behind the
    ankle, sideways > FOOT_YAW_MAX), clamp to the ankle range, fill and smooth. In place."""
    T = len(V['hip'])
    pel_x = V['lhip'] - V['rhip']
    up = _n(V['neck'] - V['hip'])
    fwd_p = _n(np.cross(pel_x, up))                          # pelvis forward
    for k, side in (('l', 1.0), ('r', -1.0)):
        an, kn, to = V[k + 'an'], V[k + 'kn'], V[k + 'to']
        n = _n(an - kn)                                       # shank, pointing down
        f = _n(fwd_p - (fwd_p * n).sum(-1, keepdims=True) * n)   # leg forward, perpendicular to shank
        l = np.cross(n, f)
        d = to - an
        L = np.linalg.norm(d, axis=-1)
        Lm = float(np.median(L)) if np.isfinite(L).any() else 0.12
        df, dl, dn = (d * f).sum(-1), (d * l).sum(-1), (d * n).sum(-1)
        yaw = np.degrees(np.arctan2(dl, df))
        pitch = np.degrees(np.arctan2(dn, np.hypot(df, dl)))
        good = (df > 0) & (np.abs(yaw) < 60) & (L > 0.4 * Lm) & (L < 2.0 * Lm)
        if good.sum() < 5:                                    # nothing usable: foot follows the shank
            yaw = np.zeros(T); pitch = np.full(T, 35.0); good = np.ones(T, bool)
        t = np.arange(T)
        yaw = np.interp(t, t[good], yaw[good]); pitch = np.interp(t, t[good], pitch[good])
        neutral = float(np.median(pitch[good]))
        yaw = np.clip(yaw, -FOOT_YAW_MAX, FOOT_YAW_MAX)
        pitch = np.clip(pitch, neutral + FOOT_PITCH_RANGE[0], neutral + FOOT_PITCH_RANGE[1])
        yaw, pitch = _lowpass(yaw, fps, 5.0), _lowpass(pitch, fps, 5.0)
        y, p = np.radians(yaw)[:, None], np.radians(pitch)[:, None]
        V[k + 'foot_angles'] = (y, p, Lm)                       # reused with the hinge-corrected shank
        V[k + 'to'] = an + Lm * _foot_dir(n, fwd_p, y, p)


def _euler_continuous(R):
    """Rotation stack -> FBX XYZ euler degrees, choosing the equivalent solution closest
    to the previous frame so curves never flip."""
    e = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    alt = np.stack([e[:, 0] + 180, 180 - e[:, 1], e[:, 2] + 180], -1)
    out = np.empty_like(e)
    prev = e[0]
    for t in range(len(e)):
        best = None
        for c in (e[t], alt[t]):
            c = prev + (c - prev + 180) % 360 - 180
            d = np.abs(c - prev).sum()
            if best is None or d < best[0]:
                best = (d, c)
        out[t] = prev = best[1]
    return out


def retarget(P, fps, rig, hands=None, contacts=None, root_motion=False, smooth_hands=True):
    """P (T,26,3) meters, Y-up, subject faces +Z.  hands: None or dict with
    'left'/'right' (T,21,3) and 'conf_left'/'conf_right' (T,21) in [0,1].
    Returns dict(bones={name: (R_local (T,3,3), t_local (T,3))}, globals={name: (T,4,4)}, info)."""
    m = RigMap(rig)
    G0 = rest_globals(rig)
    T = len(P)
    pos0 = {k: v[:3, 3] for k, v in G0.items()}
    R0 = {k: v[:3, :3] / np.linalg.norm(v[:3, :3], axis=0) for k, v in G0.items()}

    # mocap basis (x=left, y=up, z=forward) -> rig basis
    ax = rig.get('axes', {})
    if 'UpAxis' in ax:
        u_r = np.zeros(3); u_r[int(ax['UpAxis'])] = float(ax.get('UpAxisSign', 1))
    else:
        u_r = _n(pos0['head'] - pos0['pelvis'])
    l_r = pos0['upperarm_l'] - pos0['upperarm_r']
    l_r = _n(l_r - (l_r @ u_r) * u_r)
    B = np.stack([l_r, u_r, np.cross(l_r, u_r)], 1)
    legm = np.nanmedian(np.linalg.norm(P[:, IDX['left_knee']] - P[:, IDX['left_hip']], axis=-1)
                        + np.linalg.norm(P[:, IDX['left_ankle']] - P[:, IDX['left_knee']], axis=-1))
    legr = np.linalg.norm(pos0['calf_l'] - pos0['thigh_l']) + np.linalg.norm(pos0['foot_l'] - pos0['calf_l'])
    scale = legr / legm
    X = (_despike(P) @ B.T) * scale

    V = _points(X)
    _fix_feet(V, fps)
    V0 = _rest_points(m, G0)
    up_t = np.broadcast_to(u_r, (T, 3)); left_t = np.broadcast_to(l_r, (T, 3))
    F = _body_frames(V, up_t, left_t)
    F0 = _body_frames(V0, u_r[None], l_r[None])
    F0['head'] = _frame(u_r[None], l_r[None])


    # neutral head = the clip's average head pose relative to the chest (AI face keypoints
    # have no reliable absolute 'level'); motion around it is kept
    rel = np.swapaxes(F['chest'], -1, -2) @ F['head']
    rn = Rotation.from_matrix(rel).mean().as_matrix()
    dev = Rotation.from_matrix(rn.T @ rel).as_rotvec()                 # head motion around neutral
    a = np.linalg.norm(dev, axis=-1, keepdims=True)
    lim = np.radians(NECK_MAX)
    dev = np.where(a > lim, dev * lim / np.maximum(a, 1e-9), dev)
    dev = _lowpass(dev, fps, 6.0)
    F['head'] = F['chest'] @ rn @ Rotation.from_rotvec(dev).as_matrix()
    F0['head'] = F0['chest'] @ rn

    # clavicles: AI shoulder points sit off the rig's clavicle line (constant ~30 deg offset that
    # dropped the shoulders) -> neutral = clip average vs chest, keep a damped, limited shrug
    for k in ('clavicle_l', 'clavicle_r'):
        F[k], F0[k] = _neutralized(F['chest'], F0['chest'], F[k], fps, CLAVICLE_MAX, CLAVICLE_GAIN)
    Dc = F['chest'] @ np.swapaxes(F0['chest'], -1, -2)
    Dp = F['pelvis'] @ np.swapaxes(F0['pelvis'], -1, -2)
    _stable_limbs(F, F0, V, V0, Dc, Dp, fps)

    # neutral foot pitch: AI feet are toe-tip based, rig feet are ball based -> calibrate on contact frames
    for s, k in (('left', 'l'), ('right', 'r')):
        d = _n(V[k + 'to'] - V[k + 'an'])
        sel = contacts[s].astype(bool) if contacts is not None and contacts[s].any() else \
            V[k + 'an'] @ u_r <= np.percentile(V[k + 'an'] @ u_r, 30)
        pitch_m = np.median(np.arcsin(np.clip(d[sel] @ u_r, -1, 1)))
        d0 = _n(V0[k + 'to'] - V0[k + 'an'])[0]
        pitch_r = np.arcsin(np.clip(d0 @ u_r, -1, 1))
        ax = _n(np.cross(d0, u_r))
        Radj = Rotation.from_rotvec(ax * (pitch_m - pitch_r)).as_matrix()
        F0['foot_' + k] = Radj @ F0['foot_' + k]

    D = {k: F[k] @ np.swapaxes(F0[k], -1, -2) for k in F}        # global delta rotations (T,3,3)
    Rg = {}                                                       # driven global rotations

    def put(bone, Rd):
        if m.has(bone):
            Rg[bone] = Rd @ R0[bone]
    put('pelvis', D['pelvis'])
    ns = len(m.spine)
    for i, b in enumerate(m.spine):
        put(b, _slerp(D['pelvis'], D['chest'], (i + 1) / ns))
    nn = len(m.neck)
    for i, b in enumerate(m.neck):
        put(b, _slerp(D['chest'], D['head'], (i + 1) / (nn + 1)))
    put('head', D['head'])
    for k in ('clavicle_l', 'clavicle_r', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 'lowerarm_r',
              'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r'):
        put(k, D[k])

    info = {'scale_m_to_rig': float(scale), 'hands': {}}
    # ---- hands: palm orientation + smart fingers
    for side, k in (('left', 'l'), ('right', 'r')):
        Dfore = D['lowerarm_' + k]
        H = None if hands is None or hands.get(side) is None else np.asarray(hands[side], float) @ B.T * scale
        C = None if H is None else np.asarray(hands.get('conf_' + side), float)
        f_mcp = [m.finger(k, f) for f in ('index', 'middle', 'pinky')]
        Dhand, w_palm = Dfore, np.zeros(T)
        if H is not None and all(f_mcp):
            ok = np.isfinite(H).all(-1)
            w_palm = np.clip((np.where(ok[:, [0, 5, 9, 17]], C[:, [0, 5, 9, 17]], 0).min(1) - 0.25) / 0.35, 0, 1)
            w_palm = _lowpass(w_palm, fps, 2.0).clip(0, 1) if smooth_hands else w_palm
            Hn = np.where(ok[..., None], H, 0)
            Fp = _frame(Hn[:, 9] - Hn[:, 0], Hn[:, 5] - Hn[:, 17])
            Fp0 = _frame((pos0[f_mcp[1][0]] - pos0['hand_' + k])[None], (pos0[f_mcp[0][0]] - pos0[f_mcp[2][0]])[None])
            rel = Fp @ np.swapaxes(Fp0, -1, -2) @ np.swapaxes(Dfore, -1, -2)   # wrist motion vs forearm
            rv = Rotation.from_matrix(rel).as_rotvec()
            a = np.linalg.norm(rv, axis=-1, keepdims=True)
            lim = np.radians(WRIST_MAX)
            rv = np.where(a > lim, rv * lim / np.maximum(a, 1e-9), rv) * w_palm[:, None]
            if smooth_hands:
                rv = _lowpass(rv, fps, 4.0)
            Dhand = Rotation.from_rotvec(rv).as_matrix() @ Dfore
        put('hand_' + k, Dhand)
        # half the wrist twist goes into the forearm twist bone (avoids candy-wrapping)
        tw = 'lowerarm_twist_01_' + k
        if m.has(tw):
            axis = _n(V[k + 'wr'] - V[k + 'el'])
            ang = _twist_angle(Dhand @ np.swapaxes(Dfore, -1, -2), axis)
            Rg[tw] = Rotation.from_rotvec(axis * (0.5 * ang)[:, None]).as_matrix() @ Dfore @ R0[tw]
        info['hands'][side] = _fingers(m, k, side, H, C, Dhand, G0, R0, pos0, Rg, fps, smooth_hands)
        info['hands'][side]['palm_tracked_pct'] = float((w_palm > 0.5).mean() * 100)

    # ---- translations: pelvis (and root motion)
    hip0 = V0['hip'][0]
    pel = V['hip'] + np.einsum('tij,j->ti', D['pelvis'], pos0['pelvis'] - hip0)
    out = _assemble(m, rig, G0, Rg, pel, T, root_motion, u_r)
    # ground: put the lowest toe/ankle contact at the rig's rest height
    def h(n):
        return out['globals'][n][:, :3, 3] @ u_r
    low = np.minimum(h('ball_l') if m.has('ball_l') else h('foot_l'), h('ball_r') if m.has('ball_r') else h('foot_r'))
    rest_low = min(pos0['ball_l'] @ u_r, pos0['ball_r'] @ u_r) if m.has('ball_l') else pos0['foot_l'] @ u_r
    dz = rest_low - np.percentile(low, 5)
    pel = pel + dz * u_r
    out = _assemble(m, rig, G0, Rg, pel, T, root_motion, u_r)
    info['ground_offset'] = float(dz)
    out['info'] = info
    out['basis'] = B
    out['scale'] = scale
    return out


def _fingers(m, k, side, H, C, Dhand, G0, R0, pos0, Rg, fps, smooth):
    """Smart finger solver: anatomical angles from 3D hand points, clamped to joint limits,
    DIP coupled to PIP when the fingertip is unsure, blended per finger with a relaxed
    pose by tracking confidence, then zero-phase smoothed."""
    T = len(Dhand)
    names = {f: m.finger(k, f) for f in FINGERS}
    if not names['middle'] or not names['index'] or not names['pinky']:
        return {'fingers': False}
    hb = 'hand_' + k
    sgn = 1.0 if side == 'right' else -1.0
    # rest hand basis (y: wrist->middle MCP, lat: pinky->index, nv: palm/volar normal)
    y0 = _n(pos0[names['middle'][0]] - pos0[hb])
    lat0 = pos0[names['index'][0]] - pos0[names['pinky'][0]]
    lat0 = _n(lat0 - (lat0 @ y0) * y0)
    nv0 = sgn * np.cross(lat0, y0)
    Bh0 = np.stack([y0, lat0, nv0], 1)                       # hand-local -> rig rest
    if H is not None:
        y = _n(H[:, 9] - H[:, 0])
        lat = H[:, 5] - H[:, 17]
        lat = _n(lat - (lat * y).sum(-1, keepdims=True) * y)
        nv = sgn * np.cross(lat, y)
        Bh = np.stack([y, lat, nv], -1)                     # (T,3,3) hand-local -> world
    stats = {'fingers': True, 'tracked_pct': {}}
    Rhand = Rg[hb]
    for f, bones in names.items():
        if not bones:
            continue
        idx = FINGERS[f]
        # rest segment directions in hand-local coordinates
        p0 = [pos0[b] for b in bones]
        d0 = [p0[1] - p0[0], p0[2] - p0[1]]
        ax3 = R0[bones[2]][:, 0]
        d0.append(ax3 * np.sign(ax3 @ d0[1] + 1e-9))
        relax = np.radians(RELAXED[f])
        if f == 'thumb':
            u1_rest = _n(Bh0.T @ d0[0])
        if H is not None:
            u = [_n(np.einsum('tji,tj->ti', Bh, H[:, idx[j + 1]] - H[:, idx[j]])) for j in range(3)]
            cj = np.where(np.isfinite(H[:, idx]).all(-1), C[:, idx], 0)
            w = np.clip((np.median(cj, 1) - 0.3) / 0.4, 0, 1)          # AI video hands are tiny: trust only clear detections
            w_tip = np.clip((cj[:, 3] - 0.3) / 0.3, 0, 1)
            if smooth:
                w = _lowpass(w, fps, 2.0).clip(0, 1)
            u = [np.where(np.isfinite(x), x, 0) for x in u]
        else:
            w = np.zeros(T); w_tip = np.zeros(T)
        stats['tracked_pct'][f] = float((w > 0.5).mean() * 100)
        if f != 'thumb':
            if H is not None:
                th1 = np.arctan2(u[0][:, 2], u[0][:, 0])
                ph1 = np.arctan2(u[0][:, 1], np.hypot(u[0][:, 0], u[0][:, 2]))
                n1 = _n(np.array([0, 0, 1.0]) - u[0][:, 2:3] * u[0])
                th2 = np.arctan2((u[1] * n1).sum(-1), (u[1] * u[0]).sum(-1))
                n2 = _n(np.cos(th2)[:, None] * n1 - np.sin(th2)[:, None] * u[0])
                th3 = np.arctan2((u[2] * n2).sum(-1), (u[2] * u[1]).sum(-1))
                th3 = w_tip * th3 + (1 - w_tip) * th2 * 0.67          # DIP-PIP coupling
            else:
                th1 = th2 = th3 = ph1 = np.zeros(T)
            d0l = _n(Bh0.T @ d0[0])
            ph_rest = np.arctan2(d0l[1], np.hypot(d0l[0], d0l[2]))
            cl = lambda a, lim: np.clip(a, *np.radians(lim))
            th1 = w * cl(th1, LIM_MCP) + (1 - w) * relax[0]
            ph1 = w * cl(ph1, LIM_ABD) + (1 - w) * ph_rest
            th2 = w * cl(th2, LIM_PIP) + (1 - w) * relax[1]
            th3 = w * cl(th3, LIM_DIP) + (1 - w) * relax[2]
            if smooth:      # smooth, then re-apply limits (filter ringing must not break joints)
                th1, ph1, th2, th3 = (cl(_lowpass(a, fps, 6.0), lim) for a, lim in
                                      ((th1, LIM_MCP), (ph1, LIM_ABD), (th2, LIM_PIP), (th3, LIM_DIP)))
            c = lambda a: np.cos(a)[:, None]
            s = lambda a: np.sin(a)[:, None]
            Y, L, N = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
            u1 = c(ph1) * (c(th1) * Y + s(th1) * N) + s(ph1) * L
            n1 = _n(N - u1[:, 2:3] * u1)
            u2 = c(th2) * u1 + s(th2) * n1
            n2 = c(th2) * n1 - s(th2) * u1
            u3 = c(th3) * u2 + s(th3) * n2
            tgt = [u1, u2, u3]
        else:
            # thumb: CMC direction (cone-limited around rest), MCP/IP hinge-like flexion
            if H is not None:
                v1 = u[0]
                ang = np.arccos(np.clip(v1 @ u1_rest, -1, 1))
                lim = np.radians(55)
                k_ = np.where(ang > lim, lim / np.maximum(ang, 1e-6), 1.0)[:, None]
                v1 = _n(u1_rest + k_ * (v1 - u1_rest))
                t2 = np.arccos(np.clip((u[0] * u[1]).sum(-1), -1, 1))
                t3 = np.arccos(np.clip((u[1] * u[2]).sum(-1), -1, 1))
            else:
                v1 = np.broadcast_to(u1_rest, (T, 3)); t2 = t3 = np.zeros(T)
            v1 = _n(w[:, None] * v1 + (1 - w[:, None]) * u1_rest)
            t2 = w * np.clip(t2, *np.radians(LIM_THUMB)) + (1 - w) * relax[1]
            t3 = w * np.clip(t3, *np.radians(LIM_THUMB)) + (1 - w) * relax[2]
            if smooth:
                lim = np.radians(LIM_THUMB)
                v1 = _n(_lowpass(v1, fps, 6.0))
                t2 = np.clip(_lowpass(t2, fps, 6.0), *lim); t3 = np.clip(_lowpass(t3, fps, 6.0), *lim)
            # thumb flexes toward the palm and the little finger
            pv = _n(np.array([0.0, -0.5, 1.0]) - (v1 @ np.array([0.0, -0.5, 1.0]))[:, None] * v1)
            c = lambda a: np.cos(a)[:, None]
            s = lambda a: np.sin(a)[:, None]
            u2 = c(t2) * v1 + s(t2) * pv
            pv2 = c(t2) * pv - s(t2) * v1
            u3 = c(t3) * u2 + s(t3) * pv2
            tgt = [v1, u2, u3]
        # hand-local -> world through the animated hand, then swing each bone onto its target
        Rhd = Rhand @ R0[hb].T                                           # hand delta (T,3,3)
        parent_R, parent_R0 = Rhand, R0[hb]
        par = m.bones[bones[0]]['parent']
        if par != hb:                                                    # UE5 metacarpal: carried by the hand
            Rg[par] = Rhd @ R0[par]
            parent_R, parent_R0 = Rg[par], R0[par]
        for j, b in enumerate(bones):
            carried = parent_R @ parent_R0.T @ R0[b]
            dir_c = np.einsum('tij,j->ti', parent_R @ parent_R0.T, d0[j])
            dir_t = np.einsum('tij,tj->ti', Rhd, tgt[j] @ Bh0.T)
            Rg[b] = _swing(dir_c, dir_t) @ carried
            parent_R, parent_R0 = Rg[b], R0[b]
    return stats


def _assemble(m, rig, G0, Rg, pel, T, root_motion, up):
    """Driven global rotations + pelvis position -> local (R, t) per bone and global matrices."""
    G, bones = {}, {}
    for b in rig['bones']:
        n, p = b['name'], b['parent']
        Lrest = local_matrix(b)
        Pg = G[p] if p is not None else np.broadcast_to(np.eye(4), (T, 4, 4))
        if n == 'root' and root_motion:
            g = np.broadcast_to(G0[n], (T, 4, 4)).copy()
            g[:, :3, 3] = pel - (pel @ up)[:, None] * up[None] + (G0[n][:3, 3] @ up) * up
        elif n in Rg or n == 'pelvis' or n in IK_FOLLOW:
            g = np.empty((T, 4, 4)); g[:] = np.eye(4)
            if n in IK_FOLLOW and IK_FOLLOW[n] in G:
                g = G[IK_FOLLOW[n]].copy()
            else:
                g[:, :3, :3] = Rg[n] if n in Rg else G0[n][:3, :3]
                g[:, :3, 3] = pel if n == 'pelvis' else np.einsum('tij,j->ti', Pg, Lrest[:, 3])[:, :3]
        else:
            g = Pg @ Lrest
        G[n] = g
        Lc = np.linalg.inv(Pg) @ g
        bones[n] = (Lc[:, :3, :3], Lc[:, :3, 3])
    return dict(bones=bones, globals=G)


def bone_curves(rig, res):
    """Per-bone FBX curves: {name: dict(r=(T,3) Lcl Rotation deg, t=(T,3) Lcl Translation)}."""
    from .rig import euler_to_mat
    out = {}
    for b in rig['bones']:
        R, t = res['bones'][b['name']]
        Rl = euler_to_mat(b['pre']).T @ R @ euler_to_mat(b['post'])
        out[b['name']] = dict(r=_euler_continuous(Rl), t=np.asarray(t, float))
    return out
