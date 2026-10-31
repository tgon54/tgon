"""Timeline cleaner: issue detection + non-destructive edit list for AI-MoCap motion.

Edits use ORIGINAL frame numbers and are re-applied from the untouched motion every time,
so the edit list can be changed or undone freely. Trim and loop are applied last.

Edit ops:
  interpolate {start,end}            rebuild the range from the frames around it (glitch removal)
  smooth      {start,end,strength}   zero-phase low-pass with soft edges (strength 0..1)
  hold        {start,end}            freeze the pose of `start`
  foot_lock   {start,end}            pin planted feet (removes sliding) inside the range
  relax_hands {start,end,side}       ignore hand tracking -> natural relaxed fingers
  auto_clean  {strength}             interpolate detected spikes + smooth jitter everywhere
  trim        {start,end}            keep only [start, end]
  loop        {frames}               cross-fade the end into the start for a seamless cycle
"""
import json
import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter

from . import correct as K
from .skeleton import IDX, FOOT

OPS = ('interpolate', 'smooth', 'hold', 'foot_lock', 'relax_hands', 'auto_clean', 'trim', 'loop')
ISSUE_TYPES = ('spike', 'jitter', 'low_quality', 'foot_skate', 'hand_lost')


# ------------------------------------------------------------------ load --
def load(out_dir):
    mj = json.load(open(os.path.join(out_dir, 'motion.json')))
    m = dict(fps=float(mj['fps']), P=np.array(mj['corrected'], float),
             quality=np.array(mj['quality'], float) if mj.get('quality') else None,
             contacts={k: np.array(v, bool) for k, v in (mj.get('contacts') or {}).items()},
             hands=None)
    if mj.get('hands'):
        m['hands'] = {k: (np.array(v, float) if v is not None else None) for k, v in mj['hands'].items()}
    if not m['contacts']:
        m['contacts'] = K.foot_contacts(m['P'], m['fps'])
    return m


def load_edits(out_dir):
    p = os.path.join(out_dir, 'clean_edits.json')
    return json.load(open(p)).get('edits', []) if os.path.exists(p) else []


# ---------------------------------------------------------------- issues --
def analyze(m):
    """Per-frame issue severities (0..1) and merged segments for the timeline."""
    P, fps = m['P'], m['fps']
    T = len(P)
    sev = {}
    # spikes: joint acceleration far outside the clip's own distribution
    # spikes: a joint jumps away from its 5-frame running median and comes back (AI glitch);
    # unlike an acceleration test this ignores genuine impacts such as heel strikes
    sp = np.zeros(T)
    if T > 8:
        r = np.linalg.norm(P - median_filter(P, size=(5, 1, 1), mode='nearest'), axis=-1)   # (T,J) m
        base = np.median(r, 0) + 4 * np.median(np.abs(r - np.median(r, 0)), 0) * 1.4826
        sp = np.clip(((r - base).max(1) - 0.025) / 0.05, 0, 1)
    sev['spike'] = sp
    # jitter: energy above 8 Hz (human motion lives below it), relative to the clip's own level
    jit = np.zeros(T)
    if T > 20:
        r = np.linalg.norm(P - K.butter_lowpass(P, fps, 8.0), axis=-1).mean(1)
        thr = max(0.004, 3 * np.median(r))
        jit = np.clip((r - thr) / 0.006, 0, 1)
    sev['jitter'] = jit
    q = m.get('quality')
    sev['low_quality'] = np.clip((0.35 - np.mean(q, 1)) / 0.25, 0, 1) if q is not None else np.zeros(T)
    # foot skate: horizontal foot speed while in contact
    sk = np.zeros(T)
    for side, idx in FOOT.items():
        c = m['contacts'].get(side)
        if c is None or T < 2:
            continue
        f = P[:, idx].mean(1)
        v = np.zeros(T)
        v[1:] = np.linalg.norm((f[1:] - f[:-1])[:, [0, 2]], axis=-1) * fps
        sk = np.maximum(sk, np.where(c, np.clip((v - 0.12) / 0.3, 0, 1), 0))
    sev['foot_skate'] = sk
    hl = np.zeros(T)
    h = m.get('hands')
    if h:
        for s in ('left', 'right'):
            c = h.get('conf_' + s)
            if c is not None:
                hl = np.maximum(hl, np.clip((0.4 - np.median(c, 1)) / 0.3, 0, 1))
    sev['hand_lost'] = hl
    segs = []
    for typ, s in sev.items():
        on = s > 0.15
        t = 0
        while t < T:
            if on[t]:
                e = t
                while e + 1 < T and (on[e + 1] or (e + 2 < T and on[e + 2])):
                    e += 1
                segs.append(dict(type=typ, start=int(t), end=int(e), severity=round(float(s[t:e + 1].max()), 2)))
                t = e + 1
            else:
                t += 1
    return {k: np.round(v, 2).tolist() for k, v in sev.items()}, segs


# ------------------------------------------------------------------- ops --
def _rng(e, T):
    a = int(np.clip(e.get('start', 0), 0, T - 1))
    b = int(np.clip(e.get('end', T - 1), 0, T - 1))
    return min(a, b), max(a, b)


def _interp(X, a, b, ctx=4):
    """Rebuild frames a..b of X (T,...) with a cubic through `ctx` frames on each side."""
    T = len(X)
    left = np.arange(max(0, a - ctx), a)
    right = np.arange(b + 1, min(T, b + 1 + ctx))
    keys = np.concatenate([left, right])
    if len(keys) == 0:
        return X
    Y = X.copy()
    if len(keys) == 1:
        Y[a:b + 1] = X[keys[0]]
        return Y
    flat = X.reshape(T, -1)
    if len(left) == 0 or len(right) == 0:          # range touches the clip edge: hold the nearest pose
        Y[a:b + 1] = X[keys[-1] if len(left) == 0 else keys[0]]
        return Y
    cs = CubicSpline(keys, flat[keys], axis=0, bc_type='natural')
    Y.reshape(T, -1)[a:b + 1] = cs(np.arange(a, b + 1))
    return Y


def _smooth(X, fps, a, b, strength, ramp=5):
    T = len(X)
    if T < 16:
        return X
    cutoff = 8.0 - 6.5 * float(np.clip(strength, 0, 1))
    S = K.butter_lowpass(X.reshape(T, -1), fps, cutoff).reshape(X.shape)
    w = np.zeros(T)
    w[a:b + 1] = 1
    for i in range(1, ramp + 1):                    # soft edges, no pops
        if a - i >= 0:
            w[a - i] = max(w[a - i], 1 - i / (ramp + 1))
        if b + i < T:
            w[b + i] = max(w[b + i], 1 - i / (ramp + 1))
    w = w.reshape((T,) + (1,) * (X.ndim - 1))
    return X * (1 - w) + S * w


def _hand_rel(m):
    """Hands as wrist-relative offsets so body edits carry them correctly."""
    h = m.get('hands')
    if not h:
        return None
    out = {}
    for s in ('left', 'right'):
        H = h.get(s)
        out[s] = None if H is None else H - m['P'][:, IDX[s + '_wrist']][:, None]
        c = h.get('conf_' + s)
        out['conf_' + s] = None if c is None else c.copy()
    return out


def apply_edits(m, edits):
    """Returns a new motion dict with the edit list applied."""
    P = m['P'].copy()
    fps, T = m['fps'], len(m['P'])
    H = _hand_rel(m)
    cont = {k: v.copy() for k, v in m['contacts'].items()}
    hand_arrays = lambda: [H[s] for s in ('left', 'right') if H and H.get(s) is not None]
    trim, loop = None, None
    touched = np.zeros(T, bool)
    for e in edits:
        op = e.get('op')
        if op == 'trim':
            trim = _rng(e, T); continue
        if op == 'loop':
            loop = int(e.get('frames', 10)); continue
        if op == 'auto_clean':
            st = float(e.get('strength', 0.5))
            for _ in range(2):                      # second pass catches glitches next to fixed ones
                sev, segs = analyze(dict(m, P=P, contacts=cont))
                for sgm in segs:
                    if sgm['type'] == 'spike':
                        a, b = max(0, sgm['start'] - 1), min(T - 1, sgm['end'] + 1)
                        P = _interp(P, a, b)
                        for s in ('left', 'right'):
                            if H and H.get(s) is not None:
                                H[s] = _interp(H[s], a, b)
                        touched[a:b + 1] = True
            for sgm in segs:
                if sgm['type'] == 'jitter':
                    P = _smooth(P, fps, sgm['start'], sgm['end'], 0.3 + 0.6 * st)
                    touched[sgm['start']:sgm['end'] + 1] = True
            P = _smooth(P, fps, 0, T - 1, 0.15 * st)
            continue
        a, b = _rng(e, T)
        if op == 'interpolate':
            P = _interp(P, a, b)
            for s in ('left', 'right'):
                if H and H.get(s) is not None:
                    H[s] = _interp(H[s], a, b)
        elif op == 'smooth':
            st = float(e.get('strength', 0.5))
            P = _smooth(P, fps, a, b, st)
            for s in ('left', 'right'):
                if H and H.get(s) is not None:
                    H[s] = _smooth(H[s], fps, a, b, st)
        elif op == 'hold':
            P[a:b + 1] = P[a]
            for s in ('left', 'right'):
                if H and H.get(s) is not None:
                    H[s][a:b + 1] = H[s][a]
        elif op == 'foot_lock':
            sub = {k: np.zeros(T, bool) for k in cont}
            for k in cont:
                sub[k][a:b + 1] = cont[k][a:b + 1]
            P, _ = K.fix_foot_skating(P, sub)
        elif op == 'relax_hands':
            side = e.get('side', 'both')
            for s in ('left', 'right'):
                if H and H.get('conf_' + s) is not None and side in (s, 'both'):
                    H['conf_' + s][a:b + 1] = 0.0
        else:
            raise ValueError('unknown edit op: ' + str(op))
        touched[a:b + 1] = True
    if touched.any():                            # keep the skeleton anatomically valid
        L = K.bone_targets(m['P'])
        Pa = K.enforce_bones(P, L, iters=15)
        P = np.where(touched[:, None, None], Pa, P)
        P, _ = K.clamp_ground(P)
    keep = np.arange(T) if trim is None else np.arange(trim[0], trim[1] + 1)
    P = P[keep]
    cont = {k: v[keep] for k, v in cont.items()}
    if H:
        H = {k: (v[keep] if v is not None else None) for k, v in H.items()}
    if loop and len(P) > 2 * loop + 2:
        n = loop
        w = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n)))[:, None, None]
        root = P[:, IDX['hip']].copy(); root[:, 1] = 0
        rel = P - root[:, None]
        tgt = rel[:n]
        rel[-n:] = rel[-n:] * (1 - w) + tgt * w
        P = rel + root[:, None]
        if H:
            for s in ('left', 'right'):
                if H.get(s) is not None:
                    H[s][-n:] = H[s][-n:] * (1 - w) + H[s][:n] * w
    out = dict(m, P=P, contacts=cont, frames=len(P), trim=[int(keep[0]), int(keep[-1])])
    out['quality'] = m['quality'][keep] if m.get('quality') is not None else None
    if H:
        hands = {}
        for s in ('left', 'right'):
            hands[s] = None if H.get(s) is None else H[s] + P[:, IDX[s + '_wrist']][:, None]
            hands['conf_' + s] = H.get('conf_' + s)
        out['hands'] = hands
    return out


# ---------------------------------------------------------------- export --
def to_json(m):
    r = lambda a: np.round(a, 4).tolist()
    d = dict(fps=m['fps'], frames=len(m['P']), corrected=r(m['P']),
             contacts={k: v.astype(int).tolist() for k, v in m['contacts'].items()})
    if m.get('hands'):
        d['hands'] = {k: (None if v is None else (r(v) if k in ('left', 'right') else np.round(v, 2).tolist()))
                      for k, v in m['hands'].items()}
    return d


def save_clean(out_dir, edits, targets=('ue4',), root_motion=False, name=None, log=print):
    """Apply edits to OUT_DIR/motion.json and write motion_clean.json/.bvh + Unreal FBX files."""
    from .bvh import write_bvh
    from .skeleton import NAMES, BONES
    from .unreal import export_unreal, target_tag
    m = load(out_dir)
    c = apply_edits(m, edits)
    json.dump(dict(edits=edits), open(os.path.join(out_dir, 'clean_edits.json'), 'w'), indent=1)
    d = to_json(c)
    d.update(joints=NAMES, bones=[list(b) for b in BONES], source='timeline_cleaner', trim=c['trim'])
    json.dump(d, open(os.path.join(out_dir, 'motion_clean.json'), 'w'), separators=(',', ':'))
    files = ['motion_clean.json', 'motion_clean.bvh']
    write_bvh(c['P'], c['fps'], os.path.join(out_dir, 'motion_clean.bvh'))
    base = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in (name or 'motion_clean'))[:60] or 'motion_clean'
    for t in targets:
        fn = f'{base}_{target_tag(t)}.fbx'
        export_unreal(c['P'], c['fps'], os.path.join(out_dir, fn), t, c.get('hands'), c['contacts'],
                      root_motion=root_motion, take=base, log=log)
        files.append(fn)
    return files, c


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Apply timeline-cleaner edits and export to Unreal')
    ap.add_argument('out_dir')
    ap.add_argument('--edits', help='JSON file with {"edits": [...]} (default: OUT_DIR/clean_edits.json)')
    ap.add_argument('--auto', type=float, default=None, help='add an auto_clean pass with this strength (0..1)')
    ap.add_argument('--target', action='append', help='ue4, ue5 or a mannequins/ name; repeatable; default ue4')
    ap.add_argument('--root-motion', action='store_true')
    a = ap.parse_args()
    edits = json.load(open(a.edits))['edits'] if a.edits else load_edits(a.out_dir)
    if a.auto is not None:
        edits = [dict(op='auto_clean', strength=a.auto)] + edits
    files, _ = save_clean(a.out_dir, edits, a.target or ['ue4'], a.root_motion)
    print('written:', ', '.join(files))


if __name__ == '__main__':
    main()
