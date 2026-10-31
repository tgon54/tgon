"""Positions -> BVH (rotations solved from full segment frames, not just directions,
so twist is defined). Includes forward kinematics for verification."""
import numpy as np
from scipy.spatial.transform import Rotation

from .skeleton import IDX


def _n(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _frame(d, s):
    """Orthonormal frame with columns built from primary d and secondary s."""
    d = _n(d)
    s = _n(s - (s * d).sum(-1, keepdims=True) * d)
    t = np.cross(d, s)
    return np.stack([d, s, t], -1)


def _P(P, n):
    return P[:, IDX[n]]


def virtual_points(P):
    g = lambda n: _P(P, n)
    return {
        'hip': g('hip'), 'spine': 0.5 * (g('hip') + g('neck')), 'neck': g('neck'),
        'headmid': 0.5 * (g('left_ear') + g('right_ear')), 'top': g('head'),
        'lsh': g('left_shoulder'), 'rsh': g('right_shoulder'), 'lel': g('left_elbow'), 'rel': g('right_elbow'),
        'lwr': g('left_wrist'), 'rwr': g('right_wrist'),
        'lhip': g('left_hip'), 'rhip': g('right_hip'), 'lkn': g('left_knee'), 'rkn': g('right_knee'),
        'lan': g('left_ankle'), 'ran': g('right_ankle'),
        'lto': 0.5 * (g('left_big_toe') + g('left_small_toe')), 'rto': 0.5 * (g('right_big_toe') + g('right_small_toe')),
    }


def _hinge(a, b, c, fallback, sign=1.0):
    n = np.cross(b - a, c - b) * sign
    mag = np.linalg.norm(n, axis=-1, keepdims=True) / (
        np.linalg.norm(b - a, axis=-1, keepdims=True) * np.linalg.norm(c - b, axis=-1, keepdims=True) + 1e-9)
    w = np.clip((mag - 0.08) / 0.25, 0, 1)
    fb = _n(fallback)
    s = w * _n(n) + (1 - w) * fb
    return s


# name, parent, position key, child key (for offset), (dir_fn, sec_fn) keys, rest dir, rest sec
def build_hierarchy(V, L):
    X, Y = np.array([1., 0, 0]), np.array([0., 1, 0])
    pel_x = V['lhip'] - V['rhip']
    sh_x = V['lsh'] - V['rsh']
    up = V['neck'] - V['hip']
    ear_x = None
    legL = _hinge(V['lhip'], V['lkn'], V['lan'], pel_x)
    legR = _hinge(V['rhip'], V['rkn'], V['ran'], pel_x)
    chest_z = np.cross(sh_x, V['neck'] - V['spine'])
    armL = _hinge(V['lsh'], V['lel'], V['lwr'], np.cross(chest_z, V['lel'] - V['lsh']), sign=-1)
    armR = _hinge(V['rsh'], V['rel'], V['rwr'], np.cross(chest_z, V['rsh'] - V['rel']), sign=1)
    toeL0 = L['toeL']; toeR0 = L['toeR']
    J = [
        ('Hips', None, 'hip', V['neck'] - V['hip'], pel_x, Y, X),
        ('Spine', 'Hips', 'spine', V['neck'] - V['spine'], sh_x, Y, X),
        ('Neck', 'Spine', 'neck', V['headmid'] - V['neck'], sh_x, Y, X),
        ('Head', 'Neck', 'headmid', V['top'] - V['headmid'], sh_x, Y, X),
        ('LeftShoulder', 'Spine', 'neck', V['lsh'] - V['neck'], up, X, Y),
        ('LeftArm', 'LeftShoulder', 'lsh', V['lel'] - V['lsh'], armL, X, Y),
        ('LeftForeArm', 'LeftArm', 'lel', V['lwr'] - V['lel'], armL, X, Y),
        ('LeftHand', 'LeftForeArm', 'lwr', V['lwr'] - V['lel'], armL, X, Y),
        ('RightShoulder', 'Spine', 'neck', V['rsh'] - V['neck'], up, -X, Y),
        ('RightArm', 'RightShoulder', 'rsh', V['rel'] - V['rsh'], armR, -X, Y),
        ('RightForeArm', 'RightArm', 'rel', V['rwr'] - V['rel'], armR, -X, Y),
        ('RightHand', 'RightForeArm', 'rwr', V['rwr'] - V['rel'], armR, -X, Y),
        ('LeftUpLeg', 'Hips', 'lhip', V['lkn'] - V['lhip'], legL, -Y, X),
        ('LeftLeg', 'LeftUpLeg', 'lkn', V['lan'] - V['lkn'], legL, -Y, X),
        ('LeftFoot', 'LeftLeg', 'lan', V['lto'] - V['lan'], legL, toeL0 / np.linalg.norm(toeL0), X),
        ('RightUpLeg', 'Hips', 'rhip', V['rkn'] - V['rhip'], legR, -Y, X),
        ('RightLeg', 'RightUpLeg', 'rkn', V['ran'] - V['rkn'], legR, -Y, X),
        ('RightFoot', 'RightLeg', 'ran', V['rto'] - V['ran'], legR, toeR0 / np.linalg.norm(toeR0), X),
    ]
    return J


def rest_offsets(V):
    med = lambda a, b: float(np.median(np.linalg.norm(V[b] - V[a], axis=-1)))
    hw = 0.5 * med('rhip', 'lhip')
    cl = 0.5 * med('rsh', 'lsh')
    thigh = 0.5 * (med('lhip', 'lkn') + med('rhip', 'rkn'))
    shank = 0.5 * (med('lkn', 'lan') + med('rkn', 'ran'))
    up = 0.5 * (med('lsh', 'lel') + med('rsh', 'rel'))
    fo = 0.5 * (med('lel', 'lwr') + med('rel', 'rwr'))
    s1 = med('hip', 'spine'); s2 = med('spine', 'neck'); s3 = med('neck', 'headmid'); s4 = med('headmid', 'top')
    def toe(an, to):
        d = V[to] - V[an]
        return np.array([0.0, np.median(d[:, 1]), np.median(np.linalg.norm(d[:, [0, 2]], axis=-1))])
    tl, tr = toe('lan', 'lto'), toe('ran', 'rto')
    tm = 0.5 * (tl + tr)
    off = {
        'Hips': [0, 0, 0], 'Spine': [0, s1, 0], 'Neck': [0, s2, 0], 'Head': [0, s3, 0], 'Head_end': [0, s4, 0],
        'LeftShoulder': [0, s2, 0], 'LeftArm': [cl, 0, 0], 'LeftForeArm': [up, 0, 0], 'LeftHand': [fo, 0, 0], 'LeftHand_end': [0.08, 0, 0],
        'RightShoulder': [0, s2, 0], 'RightArm': [-cl, 0, 0], 'RightForeArm': [-up, 0, 0], 'RightHand': [-fo, 0, 0], 'RightHand_end': [-0.08, 0, 0],
        'LeftUpLeg': [hw, 0, 0], 'LeftLeg': [0, -thigh, 0], 'LeftFoot': [0, -shank, 0], 'LeftFoot_end': list(tm),
        'RightUpLeg': [-hw, 0, 0], 'RightLeg': [0, -thigh, 0], 'RightFoot': [0, -shank, 0], 'RightFoot_end': list(tm),
    }
    return {k: np.array(v, float) for k, v in off.items()}, {'toeL': tm, 'toeR': tm}


def solve(P):
    V = virtual_points(P)
    off, L = rest_offsets(V)
    H = build_hierarchy(V, L)
    Rg = {}
    for name, parent, pos, d, s, r0, s0 in H:
        Fc = _frame(d, s)
        Fr = _frame(r0[None], s0[None])[0]
        Rg[name] = Fc @ Fr.T
    Rl = {}
    for name, parent, *_ in H:
        Rl[name] = Rg[name] if parent is None else np.swapaxes(Rg[parent], -1, -2) @ Rg[name]
    return H, off, Rl, V


def write_bvh(P, fps, path, unit=100.0):
    """P (T,26,3) meters, Y-up, +Z forward. unit=100 -> centimeters."""
    H, off, Rl, V = solve(P)
    children = {}
    for name, parent, *_ in H:
        children.setdefault(parent, []).append(name)
    lines = ['HIERARCHY']

    def emit(name, depth):
        ind = '  ' * depth
        o = off[name] * unit
        lines.append(f'{ind}{"ROOT" if depth == 0 else "JOINT"} {name}')
        lines.append(f'{ind}{{')
        lines.append(f'{ind}  OFFSET {o[0]:.4f} {o[1]:.4f} {o[2]:.4f}')
        if depth == 0:
            lines.append(f'{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation')
        else:
            lines.append(f'{ind}  CHANNELS 3 Zrotation Xrotation Yrotation')
        ch = children.get(name, [])
        if ch:
            for c in ch:
                emit(c, depth + 1)
        else:
            e = off[name + '_end'] * unit
            lines.extend([f'{ind}  End Site', f'{ind}  {{', f'{ind}    OFFSET {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}', f'{ind}  }}'])
        lines.append(f'{ind}}}')

    emit('Hips', 0)
    T = len(P)
    order = [h[0] for h in H]
    # emit order must follow depth-first traversal
    dfs = []
    def walk(n):
        dfs.append(n)
        for c in children.get(n, []):
            walk(c)
    walk('Hips')
    eul = {n: Rotation.from_matrix(Rl[n]).as_euler('ZXY', degrees=True) for n in dfs}
    for n in dfs:
        eul[n] = np.degrees(np.unwrap(np.radians(eul[n]), axis=0))
    lines += ['MOTION', f'Frames: {T}', f'Frame Time: {1.0 / fps:.6f}']
    root = V['hip'] * unit
    for t in range(T):
        vals = list(root[t])
        for n in dfs:
            vals += list(eul[n][t])
        lines.append(' '.join(f'{v:.4f}' for v in vals))
    open(path, 'w').write('\n'.join(lines) + '\n')
    return dfs


def fk(P):
    """Forward kinematics from the solved rotations; returns dict name->(T,3) positions."""
    H, off, Rl, V = solve(P)
    par = {h[0]: h[1] for h in H}
    G, pos = {}, {}
    for name, parent, *_ in H:
        if parent is None:
            G[name] = Rl[name]; pos[name] = V['hip']
        else:
            G[name] = G[parent] @ Rl[name]
            pos[name] = pos[parent] + np.einsum('tij,j->ti', G[parent], off[name])
    return pos, V
