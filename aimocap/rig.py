"""Target skeletons for Unreal export (UE4 Mannequin, UE5 Manny or any FBX skeleton).

A rig is a dict: name, axes (FBX GlobalSettings), bones = [dict(name, parent, kind, t, r, s, pre, post)]
with FBX local transforms (degrees, file units).  Bundled rigs store only the skeleton (no mesh)."""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from . import fbx_io as F

RIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rigs')
BUNDLED = {'ue4': 'ue4_mannequin.json', 'ue5': 'ue5_manny.json'}
AXIS_KEYS = ('UpAxis', 'UpAxisSign', 'FrontAxis', 'FrontAxisSign', 'CoordAxis', 'CoordAxisSign',
             'OriginalUpAxis', 'OriginalUpAxisSign', 'UnitScaleFactor', 'OriginalUnitScaleFactor')


def _name(v):
    return v.split(b'\x00')[0].decode('utf-8', 'replace')


def rig_from_fbx(path):
    """Extract the skeleton (Root/LimbNode/Null models above it) from a binary FBX."""
    _, root = F.read(path)
    gs = F.props70(root.find('GlobalSettings'))
    objs = root.find('Objects')
    models = {}
    for m in objs.findall('Model'):
        kind = m.values[2].decode()
        pp = F.props70(m)
        g = lambda k, d: [float(x) for x in pp.get(k, d)[:3]]
        models[m.values[0]] = dict(name=_name(m.values[1]), kind=kind, parent=None,
                                   t=g('Lcl Translation', [0, 0, 0]), r=g('Lcl Rotation', [0, 0, 0]),
                                   s=g('Lcl Scaling', [1, 1, 1]), pre=g('PreRotation', [0, 0, 0]),
                                   post=g('PostRotation', [0, 0, 0]),
                                   order=int(pp.get('RotationOrder', [0])[0]))
    for c in root.find('Connections').findall('C'):
        v = c.values
        if v[0] == b'OO' and v[1] in models and v[2] in models:
            models[v[1]]['parent'] = v[2]
    skel = {k for k, m in models.items() if m['kind'] in ('LimbNode', 'Root')}
    if not skel:
        raise ValueError('no skeleton (LimbNode) found in ' + str(path))
    keep = set(skel)
    for k in skel:                          # keep Null ancestors (e.g. the 'Mannequin' armature node)
        p = models[k]['parent']
        while p is not None and p not in keep:
            keep.add(p); p = models[p]['parent']
    order, seen = [], set()

    def visit(k):
        if k in seen:
            return
        p = models[k]['parent']
        if p in keep:
            visit(p)
        seen.add(k); order.append(k)
    for k in models:
        if k in keep:
            visit(k)
    bones = []
    for k in order:
        m = dict(models[k]); p = m.pop('parent')
        m['parent'] = models[p]['name'] if p in keep else None
        if m.pop('order') != 0:
            raise ValueError(f"bone {m['name']}: only XYZ rotation order is supported")
        bones.append(m)
    return dict(name=os.path.splitext(os.path.basename(path))[0],
                axes={k: gs[k][0] for k in AXIS_KEYS if k in gs}, bones=bones)


def load_rig(spec='ue4'):
    """'ue4' | 'ue5' | path to .json rig | path to .fbx with the target skeleton."""
    if spec in BUNDLED:
        spec = os.path.join(RIG_DIR, BUNDLED[spec])
    if spec.lower().endswith('.fbx'):
        return rig_from_fbx(spec)
    return json.load(open(spec, encoding='utf-8'))


# ----------------------------------------------------------- transforms --
def euler_to_mat(e):
    """FBX eEulerXYZ (X applied first) in degrees -> (...,3,3)."""
    return Rotation.from_euler('xyz', np.asarray(e, float), degrees=True).as_matrix()


def mat_to_euler(R):
    return Rotation.from_matrix(R).as_euler('xyz', degrees=True)


def local_matrix(b, R=None, t=None):
    """4x4 local transform. R overrides (Pre * Lcl * Post^-1) rotation."""
    M = np.eye(4)
    if R is None:
        R = euler_to_mat(b['pre']) @ euler_to_mat(b['r']) @ euler_to_mat(b['post']).T
    M[:3, :3] = R * np.asarray(b['s'])[None, :]
    M[:3, 3] = b['t'] if t is None else t
    return M


def rest_globals(rig):
    G = {}
    for b in rig['bones']:
        L = local_matrix(b)
        G[b['name']] = L if b['parent'] is None else G[b['parent']] @ L
    return G


def lcl_rotation(b, R_local):
    """Local rotation matrix -> Lcl Rotation euler (undoing Pre/PostRotation)."""
    return mat_to_euler(euler_to_mat(b['pre']).T @ R_local @ euler_to_mat(b['post']))
