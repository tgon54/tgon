"""Scientific validation on synthetic data with simulated AI-video defects."""
import sys, json, numpy as np
from aimocap import synthetic as S
from aimocap.pipeline import reconstruct, export_all
from aimocap.bvh import fk, virtual_points

out = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_out'
GT = S.generate_motion()
cams = S.true_cameras()
uv, conf, log = S.corrupt(GT, cams)
print('injected defects:', log)
res = reconstruct(uv, conf, cams.sizes, 30.0, {'height': 1.75})
r = {k: S.mpjpe(res[k], GT) for k in ('raw', 'triangulated', 'final')}
print('MPJPE mean / p95 (mm):', json.dumps(r))
pos, V = fk(res['final'])
fkerr = np.mean([np.linalg.norm(pos[n] - V[k], axis=-1).mean() for n, k in
    [('LeftForeArm','lel'),('RightForeArm','rel'),('LeftLeg','lkn'),('RightLeg','rkn'),('LeftFoot','lan'),('RightFoot','ran'),('Neck','neck')]])
print(f'BVH forward-kinematics error vs corrected positions: {fkerr*1000:.1f} mm')
res['report']['synthetic_validation'] = {'mpjpe_mm': r, 'defects': log, 'bvh_fk_err_mm': fkerr * 1000}
export_all(res, out)
