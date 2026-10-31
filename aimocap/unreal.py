"""Export AI-MoCap motion as an Unreal-ready FBX animation.

    python -m aimocap.unreal OUT_DIR [--target ue4|ue5|rig.fbx] [--root-motion] [--name Take]

The FBX contains the target skeleton (exact UE bone names and rest transforms, no mesh)
with one baked animation take. In Unreal: Import -> pick the FBX -> Skeleton =
UE4_Mannequin_Skeleton (or SK_Mannequin for UE5), untick 'Import Mesh'."""
import argparse
import json
import os
import time

import numpy as np

from . import fbx_io as F
from .fbx_io import Node, P, p70, KTIME
from .rig import load_rig
from .retarget import retarget, bone_curves

# The FBX SDK cross-checks these with the footer; same constants Blender uses.
FILE_ID = bytes.fromhex('28b32aebb624ccc2bfc8b02aa92bfcf1')
TIME_ID = b'1970-01-01 10:00:00:000'
TIME_MODES = {120: 1, 100: 2, 60: 3, 50: 4, 48: 5, 30: 6, 25: 10, 24: 11, 1000: 12, 96: 15, 72: 16}


class _Ids:
    def __init__(self):
        self.n = 1_000_000

    def __call__(self):
        self.n += 1
        return self.n


def _name(n, cls):
    return P(b'S', n.encode() + b'\x00\x01' + cls.encode())


def build_fbx(rig, curves, fps, take='Take 001', animated=None):
    """rig: rig dict; curves: {bone: dict(r=(T,3), t=(T,3))}. Returns root Node."""
    T = len(next(iter(curves.values()))['r'])
    stop = int(round((T - 1) * KTIME / fps))
    ktimes = np.round(np.arange(T) * KTIME / fps).astype(np.int64)
    ids = _Ids()
    root = Node('')
    now = time.localtime()

    hx = root.add('FBXHeaderExtension')
    hx.add('FBXHeaderVersion', 1003)
    hx.add('FBXVersion', 7400)
    ts = hx.add('CreationTimeStamp')
    for k, v in (('Version', 1000), ('Year', now.tm_year), ('Month', now.tm_mon), ('Day', now.tm_mday),
                 ('Hour', now.tm_hour), ('Minute', now.tm_min), ('Second', now.tm_sec), ('Millisecond', 0)):
        ts.add(k, v)
    hx.add('Creator', 'AI-MoCap Unreal exporter')
    root.add('FileId', P(b'R', FILE_ID))
    root.add('CreationTime', P(b'S', TIME_ID))
    root.add('Creator', 'AI-MoCap Unreal exporter')

    gs = root.add('GlobalSettings'); gs.add('Version', 1000)
    gp = gs.add('Properties70')
    ax = dict(UpAxis=2, UpAxisSign=1, FrontAxis=1, FrontAxisSign=-1, CoordAxis=0, CoordAxisSign=1,
              OriginalUpAxis=2, OriginalUpAxisSign=1, UnitScaleFactor=1.0, OriginalUnitScaleFactor=1.0)
    ax.update(rig.get('axes', {}))
    for k, v in ax.items():
        if 'Unit' in k:
            p70(gp, k, 'double', 'Number', '', v)
        else:
            p70(gp, k, 'int', 'Integer', '', v)
    p70(gp, 'AmbientColor', 'ColorRGB', 'Color', '', 0, 0, 0)
    p70(gp, 'DefaultCamera', 'KString', '', '', 'Producer Perspective')
    mode = TIME_MODES.get(int(round(fps)), 14) if abs(fps - round(fps)) < 1e-3 else 14
    p70(gp, 'TimeMode', 'enum', '', '', mode)
    p70(gp, 'TimeProtocol', 'enum', '', '', 2)
    p70(gp, 'SnapOnFrameMode', 'enum', '', '', 0)
    p70(gp, 'TimeSpanStart', 'KTime', 'Time', '', 0)
    p70(gp, 'TimeSpanStop', 'KTime', 'Time', '', stop)
    p70(gp, 'CustomFrameRate', 'double', 'Number', '', float(fps))
    p70(gp, 'CurrentTimeMarker', 'int', 'Integer', '', -1)

    docs = root.add('Documents'); docs.add('Count', 1)
    doc = docs.add('Document', ids(), '', 'Scene')
    dp = doc.add('Properties70')
    p70(dp, 'SourceObject', 'object', '', '')
    p70(dp, 'ActiveAnimStackName', 'KString', '', '', take)
    doc.add('RootNode', P(b'L', 0))
    root.add('References')

    objs = Node('Objects')
    conns = Node('Connections')
    model_id = {}
    nattr = 0
    for b in rig['bones']:
        mid = ids(); model_id[b['name']] = mid
        kind = b['kind'] if b['kind'] in ('LimbNode', 'Root', 'Null') else 'LimbNode'
        aid = ids(); nattr += 1
        a = objs.add('NodeAttribute', P(b'L', aid), _name(b['name'], 'NodeAttribute'), kind)
        if kind == 'LimbNode':
            ap = a.add('Properties70'); p70(ap, 'Size', 'double', 'Number', '', 3.0)
            a.add('TypeFlags', 'Skeleton')
        elif kind == 'Root':
            a.add('TypeFlags', 'Null', 'Skeleton', 'Root')
        else:
            a.add('TypeFlags', 'Null')
        mo = objs.add('Model', P(b'L', mid), _name(b['name'], 'Model'), kind)
        mo.add('Version', 232)
        mp = mo.add('Properties70')
        p70(mp, 'RotationActive', 'bool', '', '', 1)
        p70(mp, 'InheritType', 'enum', '', '', 1)
        p70(mp, 'ScalingMax', 'Vector3D', 'Vector', '', 0, 0, 0)
        p70(mp, 'DefaultAttributeIndex', 'int', 'Integer', '', 0)
        if any(b['pre']):
            p70(mp, 'PreRotation', 'Vector3D', 'Vector', '', *b['pre'])
        if any(b['post']):
            p70(mp, 'PostRotation', 'Vector3D', 'Vector', '', *b['post'])
        c = curves.get(b['name'])
        t0 = c['t'][0] if c is not None else b['t']
        r0 = c['r'][0] if c is not None else b['r']
        p70(mp, 'Lcl Translation', 'Lcl Translation', '', 'A', *t0)
        p70(mp, 'Lcl Rotation', 'Lcl Rotation', '', 'A', *r0)
        p70(mp, 'Lcl Scaling', 'Lcl Scaling', '', 'A', *b['s'])
        mo.add('Shading', P(b'C', True))
        mo.add('Culling', 'CullingOff')
        conns.add('C', 'OO', P(b'L', aid), P(b'L', mid))
    for b in rig['bones']:
        conns.add('C', 'OO', P(b'L', model_id[b['name']]),
                  P(b'L', model_id[b['parent']] if b['parent'] else 0))

    # ---- animation
    sid, lid = ids(), ids()
    st = objs.add('AnimationStack', P(b'L', sid), _name(take, 'AnimStack'), '')
    sp = st.add('Properties70')
    for k in ('LocalStart', 'ReferenceStart'):
        p70(sp, k, 'KTime', 'Time', '', 0)
    for k in ('LocalStop', 'ReferenceStop'):
        p70(sp, k, 'KTime', 'Time', '', stop)
    objs.add('AnimationLayer', P(b'L', lid), _name('BaseLayer', 'AnimLayer'), '')
    conns.add('C', 'OO', P(b'L', lid), P(b'L', sid))
    ncn = ncv = 0
    for b in rig['bones']:
        c = curves.get(b['name'])
        if c is None or (animated is not None and b['name'] not in animated):
            continue
        for ch, prop, vals in (('T', 'Lcl Translation', c['t']), ('R', 'Lcl Rotation', c['r'])):
            if np.ptp(vals, 0).max() < 1e-5 and ch == 'T' and b['kind'] != 'Root':
                continue                    # static translation: the bind value is enough
            cnid = ids(); ncn += 1
            cn = objs.add('AnimationCurveNode', P(b'L', cnid), _name(ch, 'AnimCurveNode'), '')
            cp = cn.add('Properties70')
            for i, a in enumerate('XYZ'):
                p70(cp, 'd|' + a, 'Number', '', 'A', float(vals[0, i]))
            conns.add('C', 'OO', P(b'L', cnid), P(b'L', lid))
            conns.add('C', 'OP', P(b'L', cnid), P(b'L', model_id[b['name']]), prop)
            for i, a in enumerate('XYZ'):
                cvid = ids(); ncv += 1
                cv = objs.add('AnimationCurve', P(b'L', cvid), _name('', 'AnimCurve'), '')
                cv.add('Default', P(b'D', float(vals[0, i])))
                cv.add('KeyVer', 4008)
                cv.add('KeyTime', P(b'l', ktimes))
                cv.add('KeyValueFloat', P(b'f', vals[:, i].astype(np.float32)))
                cv.add('KeyAttrFlags', P(b'i', np.array([24836], np.int32)))      # linear keys
                cv.add('KeyAttrDataFloat', P(b'f', np.array([0, 0, 9.419963346924634e-30, 0], np.float32)))
                cv.add('KeyAttrRefCount', P(b'i', np.array([T], np.int32)))
                conns.add('C', 'OP', P(b'L', cvid), P(b'L', cnid), 'd|' + a)

    nb = len(rig['bones'])
    defs = root.add('Definitions'); defs.add('Version', 100)
    defs.add('Count', 1 + nb + nattr + 2 + ncn + ncv)
    for typ, cnt, tmpl in (('GlobalSettings', 1, None), ('Model', nb, 'FbxNode'),
                           ('NodeAttribute', nattr, 'FbxSkeleton'), ('AnimationStack', 1, 'FbxAnimStack'),
                           ('AnimationLayer', 1, 'FbxAnimLayer'), ('AnimationCurveNode', ncn, 'FbxAnimCurveNode'),
                           ('AnimationCurve', ncv, None)):
        ot = defs.add('ObjectType', typ); ot.add('Count', cnt)
        if tmpl:
            pt = ot.add('PropertyTemplate', tmpl).add('Properties70')
            if tmpl == 'FbxNode':
                p70(pt, 'RotationOrder', 'enum', '', '', 0)
                p70(pt, 'InheritType', 'enum', '', '', 0)
                p70(pt, 'Lcl Translation', 'Lcl Translation', '', 'A', 0, 0, 0)
                p70(pt, 'Lcl Rotation', 'Lcl Rotation', '', 'A', 0, 0, 0)
                p70(pt, 'Lcl Scaling', 'Lcl Scaling', '', 'A', 1, 1, 1)
                p70(pt, 'Visibility', 'Visibility', '', 'A', 1.0)
            elif tmpl == 'FbxSkeleton':
                p70(pt, 'Size', 'double', 'Number', '', 100.0)
            elif tmpl == 'FbxAnimCurveNode':
                p70(pt, 'd', 'Compound', '', '')
            elif tmpl == 'FbxAnimLayer':
                p70(pt, 'Weight', 'Number', '', 'A', 100.0)
    root.children.append(objs)
    root.children.append(conns)
    takes = root.add('Takes'); takes.add('Current', take)
    tk = takes.add('Take', take)
    tk.add('FileName', take.replace(' ', '_') + '.tak')
    tk.add('LocalTime', P(b'L', 0), P(b'L', stop))
    tk.add('ReferenceTime', P(b'L', 0), P(b'L', stop))
    return root


def export_unreal(P_, fps, path, target='ue4', hands=None, contacts=None, root_motion=False,
                  take=None, log=print):
    """Retarget positions (+hands) onto the target skeleton and write an FBX for Unreal."""
    rig = load_rig(target)
    res = retarget(np.asarray(P_, float), float(fps), rig, hands=hands, contacts=contacts,
                   root_motion=root_motion)
    curves = bone_curves(rig, res)
    take = take or os.path.splitext(os.path.basename(path))[0]
    F.write(path, build_fbx(rig, curves, fps, take=take), 7400)
    h = res['info']['hands']
    log(f'  Unreal FBX ({rig.get("label", rig["name"])}): {os.path.basename(path)}  '
        f'{len(P_)} frames @ {fps:g} fps, hands: '
        + ', '.join(f'{s} palm {v.get("palm_tracked_pct", 0):.0f}% tracked' for s, v in h.items()))
    return res


def load_motion(out_dir, prefer_clean=True):
    """Load an AI-MoCap output folder -> (P, fps, hands, contacts)."""
    mj = None
    for n in (['motion_clean.json'] if prefer_clean else []) + ['motion.json']:
        p = os.path.join(out_dir, n)
        if os.path.exists(p):
            mj = json.load(open(p)); break
    if mj is None:
        raise FileNotFoundError('motion.json not found in ' + out_dir)
    P_ = np.array(mj['corrected'], float)
    hands = None
    if mj.get('hands'):
        hj = mj['hands']
        hands = {k: (np.array(hj[k], float) if hj.get(k) is not None else None)
                 for k in ('left', 'right', 'conf_left', 'conf_right')}
    cont = {k: np.array(v, bool) for k, v in mj.get('contacts', {}).items()} or None
    return P_, float(mj['fps']), hands, cont


def main():
    ap = argparse.ArgumentParser(description='Export AI-MoCap output to an Unreal Engine FBX animation')
    ap.add_argument('out_dir', help='AI-MoCap output folder (contains motion.json)')
    ap.add_argument('--target', default='ue4', help="ue4 (Mannequin), ue5 (Manny) or a .fbx/.json skeleton")
    ap.add_argument('--root-motion', action='store_true', help='move the root bone under the pelvis (UE root motion)')
    ap.add_argument('--output', help='FBX path (default OUT_DIR/motion_<target>.fbx)')
    ap.add_argument('--original', action='store_true', help='ignore motion_clean.json from the timeline cleaner')
    a = ap.parse_args()
    P_, fps, hands, cont = load_motion(a.out_dir, prefer_clean=not a.original)
    tag = a.target if a.target in ('ue4', 'ue5') else 'custom'
    path = a.output or os.path.join(a.out_dir, f'motion_{tag.upper()}.fbx')
    export_unreal(P_, fps, path, a.target, hands, cont, a.root_motion)
    print('written', path)


if __name__ == '__main__':
    main()
