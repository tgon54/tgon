"""Self-check for the Unreal export: retarget a synthetic motion onto the UE4/UE5 skeletons,
write the FBX, read it back with the bundled parser and compare every bone, plus the
smart-hand limits.   python test_unreal.py"""
import os
import sys
import tempfile

import numpy as np

from aimocap import synthetic as S, fbx_io as F
from aimocap.rig import load_rig, euler_to_mat
from aimocap.retarget import retarget, FINGERS
from aimocap.unreal import export_unreal
from aimocap.skeleton import IDX


def read_back(path):
    """FBX -> {bone: (T,4,4) global} by evaluating the written curves."""
    _, root = F.read(path)
    objs, conns = root.find('Objects'), root.find('Connections').findall('C')
    models = {m.values[0]: m for m in objs.findall('Model')}
    curves = {c.values[0]: c for c in objs.findall('AnimationCurve')}
    parent, cn_target, cv_target = {}, {}, {}
    for c in conns:
        v = c.values
        if v[0] == b'OO' and v[1] in models:
            parent[v[1]] = v[2]
        elif v[0] == b'OP' and v[2] in models:
            cn_target[v[1]] = (v[2], v[3].decode())
        elif v[0] == b'OP' and v[1] in curves:
            cv_target[v[1]] = (v[2], v[3].decode())
    T = max(len(c.find('KeyValueFloat').values[0]) for c in curves.values())
    chan = {}
    for cid, (cnid, axis) in cv_target.items():
        mid, prop = cn_target[cnid]
        chan.setdefault((mid, prop), np.zeros((T, 3)))[:, 'XYZ'.index(axis[-1])] = curves[cid].find('KeyValueFloat').values[0]
    G, name = {}, {}

    def glob(mid):
        if mid in G:
            return G[mid]
        pp = F.props70(models[mid])
        t = chan.get((mid, 'Lcl Translation'), np.tile(pp.get('Lcl Translation', [0, 0, 0])[:3], (T, 1)))
        r = chan.get((mid, 'Lcl Rotation'), np.tile(pp.get('Lcl Rotation', [0, 0, 0])[:3], (T, 1)))
        L = np.tile(np.eye(4), (T, 1, 1))
        L[:, :3, :3] = euler_to_mat(r); L[:, :3, 3] = t
        p = parent.get(mid, 0)
        G[mid] = L if p == 0 else glob(p) @ L
        return G[mid]
    return {models[k].values[1].split(b'\x00')[0].decode(): glob(k) for k in models}


def main():
    P = S.generate_motion()
    fps = 30.0
    T = len(P)
    rng = np.random.default_rng(0)
    # noisy synthetic hands around the wrists, full confidence -> exercises clamping
    hands = {}
    for s in ('left', 'right'):
        w = P[:, IDX[s + '_wrist']]
        e = P[:, IDX[s + '_elbow']]
        d = (w - e) / np.linalg.norm(w - e, axis=-1, keepdims=True)
        base = np.concatenate([np.zeros((1, 3)), np.cumsum(rng.normal(0, 0.01, (20, 3)), 0)])
        hands[s] = w[:, None] + base[None] + d[:, None] * np.linspace(0, 0.18, 21)[None, :, None] \
            + rng.normal(0, 0.006, (T, 21, 3))
        hands['conf_' + s] = np.ones((T, 21))
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for tgt in ('ue4', 'ue5'):
            rig = load_rig(tgt)
            for root_motion in (False, True):
                path = os.path.join(tmp, f'{tgt}_{root_motion}.fbx')
                res = export_unreal(P, fps, path, tgt, hands, root_motion=root_motion, log=lambda *_: None)
                back = read_back(path)
                err = max(np.abs(back[n][:, :3, 3] - res['globals'][n][:, :3, 3]).max() for n in res['globals'])
                rerr = max(np.abs(back[n][:, :3, :3] - res['globals'][n][:, :3, :3]).max() for n in res['globals'])
                # anatomical limit: PIP flexion (angle between proximal and middle phalanx) <= 110 deg
                worst = 0.0
                for side in ('l', 'r'):
                    for f in ('index', 'middle', 'ring', 'pinky'):
                        p1, p2, p3 = (res['globals'][f'{f}_0{k}_{side}'][:, :3, 3] for k in (1, 2, 3))
                        u, v = p2 - p1, p3 - p2
                        c = (u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1)
                        worst = max(worst, np.degrees(np.arccos(np.clip(c, -1, 1))).max())
                good = err < 0.01 and rerr < 1e-4 and worst <= 111 and len(back) == len(rig['bones'])
                ok &= good
                print(f'{tgt} root_motion={root_motion}: {len(back)} bones, read-back max pos err {err:.5f} cm, '
                      f'rot err {rerr:.1e}, max PIP flexion {worst:.0f} deg -> {"OK" if good else "FAIL"}')
        # no hand data -> relaxed, constant fingers
        res = retarget(P, fps, load_rig('ue4'), hands=None)
        R = res['bones']['index_02_l'][0]
        const = np.abs(R - R[0]).max() < 1e-9
        ok &= const
        print('no hand data -> constant relaxed fingers:', 'OK' if const else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
