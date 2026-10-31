"""Physics self-check:  python test_physics.py
1) a jump whose flight phase was captured wrongly (COM floats at a constant height) must come
   out as a ballistic parabola (COM vertical acceleration = -g);
2) a still stance leaning outside the feet must be brought back over the support area;
3) a balanced stance with natural sway (COM inside the feet) must be left alone."""
import sys

import numpy as np

from aimocap import synthetic as S
from aimocap.physics import simulate, G
from aimocap.skeleton import FOOT, IDX, center_of_mass

FPS = 60.0


def stance(T):
    P = S.generate_motion(T=1, fps=FPS)[0]                     # one standing pose
    P = P - [0, P[:, 1].min(), 0]
    return np.repeat(P[None], T, 0)


def main():
    ok = True
    # 1) jump: 0.4 s flight, but captured as a flat plateau 25 cm up (AI video artefact)
    T = 120
    P = stance(T)
    a, b = 40, 64
    lift = np.zeros(T)
    lift[a:b] = 0.25
    lift[a - 4:a] = np.linspace(0, 0.25, 4)
    lift[b:b + 4] = np.linspace(0.25, 0, 4)
    P[:, :, 1] += lift[:, None]
    cont = {s: (lift < 0.01) for s in FOOT}
    out, rep, F, air = simulate(P, FPS, center_of_mass, FOOT, cont, 70.0, None, log=lambda *x: None)
    com = center_of_mass(out)
    acc = (com[2:, 1] - 2 * com[1:-1, 1] + com[:-2, 1]) * FPS * FPS
    inside = np.arange(1, T - 1)
    fl = air[1:-1] & air[2:] & air[:-2]
    err = np.abs(acc[fl[inside - 1]] + G).max() if fl.any() else np.inf
    good = rep['flight_phases'] == 1 and err < 0.05
    ok &= good
    print(f'jump: flight phases {rep["flight_phases"]}, max |a_y + g| in flight {err:.4f} m/s2 -> {"OK" if good else "FAIL"}')
    # 2) still stance leaning 12 cm forward outside the feet (upper body only)
    P = stance(90)
    up = [i for i in range(P.shape[1]) if i not in sum(FOOT.values(), []) and 'knee' not in str(i)]
    upper = [IDX[n] for n in ('hip', 'neck', 'head', 'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                              'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')]
    P[:, upper, 2] += 0.50
    cont = {s: np.ones(90, bool) for s in FOOT}
    c0 = center_of_mass(P)
    feet = np.concatenate([P[:, FOOT[s]] for s in FOOT], 1)
    out, rep, F, air = simulate(P, FPS, center_of_mass, FOOT, cont, 70.0, None, log=lambda *x: None)
    c1 = center_of_mass(out)
    zmax = feet[..., 2].max() + 0.03
    before, after = c0[45, 2] - zmax, c1[45, 2] - zmax
    good = before > 0.05 and after <= 0.005 and rep['balance_fixed_pct'] > 50
    ok &= good
    print(f'balance: COM ahead of the feet by {before * 100:.1f} cm -> {after * 100:.1f} cm '
          f'({rep["balance_fixed_pct"]:.0f}% frames fixed) -> {"OK" if good else "FAIL"}')
    # 3) balanced stance with a gentle 2 cm sway: nothing to fix
    P = stance(180)
    sway = 0.02 * np.sin(2 * np.pi * np.arange(180) / 120)
    P[:, upper, 0] += sway[:, None]
    cont = {s: np.ones(180, bool) for s in FOOT}
    out, rep, F, air = simulate(P, FPS, center_of_mass, FOOT, cont, 70.0, None, log=lambda *x: None)
    d = np.abs(out - P).max() * 1000
    good = d < 1 and rep['balance_fixed_pct'] == 0 and abs(rep['mean_grf_bw'] - 1) < 0.02
    ok &= good
    print(f'balanced stance: max change {d:.2f} mm, balance edits {rep["balance_fixed_pct"]:.0f}%, '
          f'mean GRF {rep["mean_grf_bw"]:.2f} BW -> {"OK" if good else "FAIL"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
