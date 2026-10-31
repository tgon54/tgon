"""Minimal binary FBX (7.x) reader/writer - pure Python, no Autodesk SDK / Blender needed.

A file is a tree of `Node(name, props, children)`. Property values keep their FBX
type code so a parsed file can be written back unchanged."""
import struct
import zlib

import numpy as np

MAGIC = b'Kaydara FBX Binary  \x00\x1a\x00'
FOOT_ID = bytes.fromhex('fabcab09d0c8d466b176fb831cf7267e')
FOOT_MAGIC = bytes.fromhex('f85a8c6adef5d97eece90ce3758f290b')
KTIME = 46186158000                     # FBX ticks per second

_SCALAR = {b'Y': '<h', b'C': '<?', b'I': '<i', b'F': '<f', b'D': '<d', b'L': '<q'}
_ARRAY = {b'f': ('<f', np.float32), b'd': ('<d', np.float64), b'l': ('<q', np.int64),
          b'i': ('<i', np.int32), b'b': ('<?', np.bool_)}


class P:
    """Typed property value."""
    __slots__ = ('t', 'v')

    def __init__(self, t, v):
        self.t, self.v = t, v

    def __repr__(self):
        return f'{self.t.decode()}:{self.v!r}'


class Node:
    __slots__ = ('name', 'props', 'children')

    def __init__(self, name, props=None, children=None):
        self.name = name
        self.props = props or []
        self.children = children or []

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def findall(self, name):
        return [c for c in self.children if c.name == name]

    def add(self, name, *props, children=None):
        n = Node(name, [p if isinstance(p, P) else auto(p) for p in props], children)
        self.children.append(n)
        return n

    @property
    def values(self):
        return [p.v for p in self.props]


def auto(v):
    if isinstance(v, P):
        return v
    if isinstance(v, bool):
        return P(b'C', v)
    if isinstance(v, int):
        return P(b'L', v) if abs(v) > 0x7fffffff else P(b'I', v)
    if isinstance(v, float):
        return P(b'D', v)
    if isinstance(v, str):
        return P(b'S', v.encode('utf-8'))
    if isinstance(v, bytes):
        return P(b'S', v)
    raise TypeError(type(v))


# ------------------------------------------------------------------ read --
def _read_node(b, o, wide):
    if wide:
        end, nprop, _plen = struct.unpack_from('<QQQ', b, o); o += 24
    else:
        end, nprop, _plen = struct.unpack_from('<III', b, o); o += 12
    nlen = b[o]; o += 1
    if end == 0:
        return None, o
    name = b[o:o + nlen].decode('ascii', 'replace'); o += nlen
    props = []
    for _ in range(nprop):
        t = b[o:o + 1]; o += 1
        if t in _SCALAR:
            f = _SCALAR[t]
            props.append(P(t, struct.unpack_from(f, b, o)[0])); o += struct.calcsize(f)
        elif t in _ARRAY:
            n, enc, clen = struct.unpack_from('<III', b, o); o += 12
            raw = b[o:o + clen]; o += clen
            if enc == 1:
                raw = zlib.decompress(raw)
            props.append(P(t, np.frombuffer(raw, _ARRAY[t][1], n).copy()))
        elif t in (b'S', b'R'):
            n = struct.unpack_from('<I', b, o)[0]; o += 4
            props.append(P(t, bytes(b[o:o + n]))); o += n
        else:
            raise ValueError(f'unknown FBX property type {t!r} at {o}')
    children = []
    while o < end:
        c, o = _read_node(b, o, wide)
        if c is None:
            break
        children.append(c)
    return Node(name, props, children), end


def read(path):
    """Returns (version, root Node). Binary FBX only."""
    b = open(path, 'rb').read()
    if not b.startswith(MAGIC):
        raise ValueError('not a binary FBX file (ASCII FBX is not supported): ' + str(path))
    ver = struct.unpack_from('<I', b, 23)[0]
    wide = ver >= 7500
    o, root = 27, Node('')
    while o < len(b):
        n, o = _read_node(b, o, wide)
        if n is None:
            break
        root.children.append(n)
    return ver, root


# ----------------------------------------------------------------- write --
def _enc_prop(p):
    t, v = p.t, p.v
    if t in _SCALAR:
        return t + struct.pack(_SCALAR[t], v)
    if t in _ARRAY:
        a = np.ascontiguousarray(np.asarray(v, _ARRAY[t][1]))
        raw = a.tobytes()
        if len(raw) > 128:
            z = zlib.compress(raw, 6)
            return t + struct.pack('<III', len(a), 1, len(z)) + z
        return t + struct.pack('<III', len(a), 0, len(raw)) + raw
    if t in (b'S', b'R'):
        return t + struct.pack('<I', len(v)) + v
    raise ValueError(t)


def _enc_node(n, off, wide):
    hdr = 25 if wide else 13
    name = n.name.encode('ascii')
    pb = b''.join(_enc_prop(p) for p in n.props)
    body_off = off + hdr + len(name) + len(pb)
    parts, cur = [], body_off
    for c in n.children:
        e = _enc_node(c, cur, wide)
        parts.append(e); cur += len(e)
    if n.children or not n.props or n.name in ('AnimationStack', 'AnimationLayer'):   # matches Blender/SDK output
        parts.append(b'\x00' * hdr); cur += hdr
    fmt = '<QQQ' if wide else '<III'
    return struct.pack(fmt, cur, len(n.props), len(pb)) + bytes([len(name)]) + name + pb + b''.join(parts)


def write(path, root, version=7400):
    wide = version >= 7500
    out = bytearray(MAGIC + struct.pack('<I', version))
    for c in root.children:
        out += _enc_node(c, len(out), wide)
    out += b'\x00' * (25 if wide else 13)
    out += FOOT_ID
    out += b'\x00' * ((16 - len(out) % 16) % 16 or 16)
    out += struct.pack('<I', version) + b'\x00' * 120 + FOOT_MAGIC
    open(path, 'wb').write(bytes(out))


# ------------------------------------------------------------- helpers --
def props70(node):
    """Properties70 -> {name: [values...]}."""
    out = {}
    p = node.find('Properties70') if node is not None else None
    if p is None:
        return out
    for c in p.children:
        v = c.values
        out[v[0].decode('utf-8', 'replace')] = v[4:]
    return out


def p70(parent, name, ptype, label, flags, *vals):
    """Append one `P:` entry; numeric types follow the FBX conventions used by Maya/Blender."""
    n = parent.add('P', name, ptype, label, flags)
    for v in vals:
        if ptype in ('int', 'enum', 'bool', 'Bool', 'Integer', 'Visibility Inheritance'):
            n.props.append(P(b'I', int(v)))
        elif ptype == 'KTime':
            n.props.append(P(b'L', int(v)))
        else:
            n.props.append(P(b'D', float(v)) if not isinstance(v, (str, bytes)) else auto(v))
    return n
