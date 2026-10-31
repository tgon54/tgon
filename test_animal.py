"""Animal export self-check:  python test_animal.py
A synthetic trotting dog/horse is reconstructed through the generated rig: the FBX read back
must match the solver exactly, every limb bone must point where the motion's limb points,
and the proxy body must be skinned to every bone."""
import os
import sys
import tempfile

import numpy as np

from aimocap import animal as A
from aimocap import fbx_io as F
from test_unreal import read_back


def trot(species, T=90, fps=30.0):
    H = A.SPECIES[species]['height']
    P = np.zeros((T, A.J, 3))
    for t in range(T):
        s = t / fps
        x0 = 0.9 * H * s
        g = {}
        ws, wh = A.SPECIES[species]['shoulder_w'] * H / 2, A.SPECIES[species]['hip_w'] * H / 2
        g['neck'] = [0, 1.05 * H, x0 + 0.55 * H]; g['tail_root'] = [0, 0.95 * H, x0 - 0.55 * H]
        g['nose'] = [0, 1.25 * H, x0 + 0.95 * H]
        g['left_eye'] = [0.05 * H, 1.3 * H, x0 + 0.85 * H]; g['right_eye'] = [-0.05 * H, 1.3 * H, x0 + 0.85 * H]
        for side, sx, ph in (('left', 1, 0.0), ('right', -1, np.pi)):
            for front, zc, w, phs in ((True, 0.45 * H, ws, ph), (False, -0.45 * H, wh, ph + np.pi)):
                a = 0.35 * np.sin(2 * np.pi * 1.6 * s + phs)
                top = np.array([sx * w, 0.82 * H, x0 + zc])
                k = top + 0.42 * H * np.array([0, -np.cos(a), np.sin(a)])
                paw = k + 0.40 * H * np.array([0, -np.cos(a * 0.4), np.sin(a * 0.4) - (0.15 if front else -0.15)])
                n1, n2, n3 = (f'{side}_shoulder', f'{side}_elbow', f'{side}_front_paw') if front else \
                             (f'{side}_hip', f'{side}_knee', f'{side}_back_paw')
                g[n1], g[n2], g[n3] = top, k, paw
        for n, v in g.items():
            P[t, A.IDX[n]] = v
    P[..., 1] -= P[..., 1].min()
    return P, fps


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for sp in ('dog', 'horse'):
            P, fps = trot(sp)
            rig = A.build_rig(P, sp)
            rt = A.retarget(P, rig, fps)
            path = os.path.join(tmp, sp + '.fbx')
            A.write_fbx(path, rig, rt, fps)
            back = read_back(path)
            err = max(np.abs(back[n][:, :3, 3] - rt['globals'][n][:, :3, 3]).max() for n in rt['globals'])
            V = {k: v @ A.W2R.T for k, v in A._virtual(P).items()}
            worst = 0.0
            for name, parent, head, tail, rad in A.RIG:
                if tail is None or name.startswith(('spine', 'pelvis', 'tail', 'neck', 'head')):
                    continue
                d1 = rt['globals'][[n for n, p, *_ in A.RIG if p == name][0]][:, :3, 3] - rt['globals'][name][:, :3, 3]
                d2 = V[tail] - V[head]
                c = (d1 * d2).sum(-1) / np.linalg.norm(d1, axis=-1) / np.linalg.norm(d2, axis=-1)
                worst = max(worst, np.degrees(np.arccos(np.clip(c, -1, 1))).max())
            _, root = F.read(path)
            objs = root.find('Objects')
            ncl = sum(1 for d in objs.findall('Deformer') if d.values[2] == b'Cluster')
            good = err < 0.01 and worst < 1.0 and ncl == len(A.RIG) and objs.find('Geometry') is not None
            ok &= good
            print(f'{sp}: {len(back)} nodes, read-back err {err:.5f} cm, worst limb direction error {worst:.2f} deg, '
                  f'skin clusters {ncl}/{len(A.RIG)} -> {"OK" if good else "FAIL"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
