"""Target skeletons for Unreal export (UE4 Mannequin, UE5 Manny or any FBX skeleton).

A rig is a dict: name, axes (FBX GlobalSettings), bones = [dict(name, parent, kind, t, r, s, pre, post)]
with FBX local transforms (degrees, file units).  Bundled rigs store only the skeleton (no mesh)."""
import json
import os
import re

import numpy as np
from scipy.spatial.transform import Rotation

from . import fbx_io as F

RIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rigs')
BUNDLED = {'ue4': 'ue4_mannequin.json', 'ue5': 'ue5_manny.json'}
# Drop-in folder: every .fbx / .json skeleton here becomes an export target named after the file.
MANNEQUIN_DIR = os.environ.get('AIMOCAP_MANNEQUINS',
                               os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mannequins'))
_CACHE = {}
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


def _folder_files():
    if not os.path.isdir(MANNEQUIN_DIR):
        return {}
    out = {}
    for f in sorted(os.listdir(MANNEQUIN_DIR)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in ('.fbx', '.json'):
            out.setdefault(stem, os.path.join(MANNEQUIN_DIR, f))
    return out


def list_targets():
    """Bundled UE4/UE5 rigs + every skeleton in the mannequins folder.
    Returns [dict(id, label, source, ok, error, bones)]; `ok` = usable (has the UE bone names)."""
    from .retarget import RigMap
    out = []
    for tid in BUNDLED:
        r = load_rig(tid)
        out.append(dict(id=tid, label=r.get('label', tid), source='bundled', ok=True, error=None, bones=len(r['bones'])))
    for stem, path in _folder_files().items():
        d = dict(id=stem, label=stem, source='mannequins/' + os.path.basename(path), ok=False, error=None, bones=0)
        try:
            r = load_rig(path)
            d['bones'] = len(r['bones'])
            RigMap(r)
            d['ok'] = True
        except Exception as e:
            d['error'] = str(e)
        out.append(d)
    return out


def load_rig(spec='ue4'):
    """'ue4' | 'ue5' | name of a file in the mannequins folder | path to a .json/.fbx skeleton."""
    if spec in BUNDLED:
        spec = os.path.join(RIG_DIR, BUNDLED[spec])
    elif not os.path.exists(spec):
        files = _folder_files()
        if spec not in files:
            raise ValueError(f"unknown target '{spec}' (use ue4, ue5, a file in {MANNEQUIN_DIR} or a path)")
        spec = files[spec]
    key = (os.path.abspath(spec), os.path.getmtime(spec))
    if key not in _CACHE:
        _CACHE[key] = rig_from_fbx(spec) if spec.lower().endswith('.fbx') else json.load(open(spec, encoding='utf-8'))
    return _CACHE[key]


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


# ----------------------------------------------------------------- meshes --
_MESH_CACHE = {}


def mesh_from_fbx(path, max_verts=25000):
    """Skinned body mesh for the preview: dict(verts (N,3) f32 in file space, faces (M,3) u32,
    bones [names], joints (N,4) u16, weights (N,4) f32, bind {bone: 4x4 TransformLink}).
    Picks the most detailed LOD with <= max_verts vertices (all meshes when there are no LODs)."""
    key = (os.path.abspath(path), os.path.getmtime(path), max_verts)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    _, root = F.read(path)
    objs = root.find('Objects')
    conns = [c.values for c in root.find('Connections').findall('C')]
    by_id = {o.values[0]: o for o in objs.children if o.props}
    name = lambda i: _name(by_id[i].values[1]) if i in by_id else ''
    child_of = {}
    for c in conns:
        if c[0] == b'OO':
            child_of.setdefault(c[2], []).append(c[1])
    geos = [g for g in objs.findall('Geometry') if g.values[2] == b'Mesh' and g.find('Vertices') is not None]
    if not geos:
        raise ValueError('no mesh in ' + str(path))
    nverts = lambda g: len(g.find('Vertices').values[0]) // 3
    lod = {}
    for g in geos:
        m = re.search(r'LOD(\d+)', name(g.values[0]))
        lod.setdefault(int(m.group(1)) if m else -1, []).append(g)
    if -1 in lod or len(lod) == 1:
        pick = sum(lod.values(), [])
    else:
        ok = [k for k in sorted(lod) if sum(nverts(g) for g in lod[k]) <= max_verts]
        pick = lod[ok[0] if ok else max(lod)]
    V, Fc, Jn, Wt, bones, bind = [], [], [], [], [], {}
    off = 0
    for g in pick:
        v = np.asarray(g.find('Vertices').values[0], float).reshape(-1, 3)
        pi = np.asarray(g.find('PolygonVertexIndex').values[0], np.int64)
        ends = np.where(pi < 0)[0]
        idx = np.where(pi < 0, -pi - 1, pi)
        start = 0
        tris = []
        for e in ends:                                   # fan-triangulate
            poly = idx[start:e + 1]
            for k in range(1, len(poly) - 1):
                tris.append((poly[0], poly[k], poly[k + 1]))
            start = e + 1
        n = len(v)
        jw = [[] for _ in range(n)]
        Tm = _model_global(g.values[0], conns, by_id)       # mesh node's own transform
        for skin in [s for s in child_of.get(g.values[0], []) if s in by_id and by_id[s].values[2] == b'Skin']:
            for cl in child_of.get(skin, []):
                c = by_id.get(cl)
                if c is None or c.values[2] != b'Cluster' or c.find('Indexes') is None:
                    continue
                links = [b for b in child_of.get(cl, []) if b in by_id and by_id[b].name == 'Model']
                if not links:
                    continue
                bn = name(links[0])
                if bn not in bind:
                    bind[bn] = np.asarray(c.find('TransformLink').values[0], float).reshape(4, 4).T
                    bones.append(bn)
                bi = bones.index(bn)
                for i, w in zip(c.find('Indexes').values[0], c.find('Weights').values[0]):
                    jw[i].append((w, bi))
        v = v @ Tm[:3, :3].T + Tm[:3, 3]
        J4 = np.zeros((n, 4), np.uint16); W4 = np.zeros((n, 4), np.float32)
        for i, lst in enumerate(jw):
            lst.sort(reverse=True)
            for k, (w, b) in enumerate(lst[:4]):
                J4[i, k], W4[i, k] = b, w
        s = W4.sum(1, keepdims=True)
        W4 = np.where(s > 0, W4 / np.maximum(s, 1e-9), 0)
        V.append(v); Fc.append(np.asarray(tris, np.int64) + off); Jn.append(J4); Wt.append(W4)
        off += n
    out = dict(verts=np.concatenate(V).astype(np.float32), faces=np.concatenate(Fc).astype(np.uint32),
               bones=bones, joints=np.concatenate(Jn), weights=np.concatenate(Wt).astype(np.float32), bind=bind)
    _MESH_CACHE[key] = out
    return out


def _model_global(geo_id, conns, by_id):
    """Global matrix of the Model that owns a Geometry (cluster 'Transform' is not used: UE
    exports store the inverse bind there, other exporters store the mesh matrix)."""
    parent = {c[1]: c[2] for c in conns if c[0] == b'OO'}
    mid = parent.get(geo_id)
    G = np.eye(4)
    while mid in by_id and by_id[mid].name == 'Model':
        pp = F.props70(by_id[mid])
        g = lambda k, d: [float(x) for x in pp.get(k, d)[:3]]
        L = local_matrix(dict(t=g('Lcl Translation', [0, 0, 0]), r=g('Lcl Rotation', [0, 0, 0]),
                              s=g('Lcl Scaling', [1, 1, 1]), pre=g('PreRotation', [0, 0, 0]),
                              post=g('PostRotation', [0, 0, 0])))
        G = L @ G
        mid = parent.get(mid)
    return G


def mesh_file_for(target):
    """FBX that provides the body mesh for a target: the mannequins/ file itself, or for the
    bundled ue4/ue5 rigs a mannequins/ file with the same skeleton."""
    files = _folder_files()
    if target in files:
        return files[target] if files[target].lower().endswith('.fbx') else None
    if target in BUNDLED:
        want = {b['name'] for b in load_rig(target)['bones'] if b['kind'] != 'Null'}
        for stem, p in files.items():
            if p.lower().endswith('.fbx'):
                try:
                    if {b['name'] for b in load_rig(p)['bones'] if b['kind'] != 'Null'} == want:
                        return p
                except Exception:
                    pass
    if os.path.exists(target) and target.lower().endswith('.fbx'):
        return target
    return None
