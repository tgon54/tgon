"""Halpe-26 skeleton definition (RTMPose body-with-feet), bones, symmetry and
segment inertial parameters (de Leva 1996, adapted from Zatsiorsky)."""
import numpy as np

NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
    'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip',
    'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe',
    'left_heel', 'right_heel']
IDX = {n: i for i, n in enumerate(NAMES)}
J = len(NAMES)

# Left/right index permutation (applying it swaps sides).
FLIP = np.arange(J)
for n, i in IDX.items():
    if n.startswith('left_'):
        FLIP[i] = IDX['right_' + n[5:]]
    elif n.startswith('right_'):
        FLIP[i] = IDX['left_' + n[6:]]

# Groups that are swap-tested independently (detectors often swap one limb pair).
def _pair(*names):
    out = []
    for n in names:
        out += [IDX['left_' + n], IDX['right_' + n]]
    return out

SWAP_GROUPS = {
    'legs': _pair('hip', 'knee', 'ankle', 'big_toe', 'small_toe', 'heel'),
    'arms': _pair('shoulder', 'elbow', 'wrist'),
    'face': _pair('eye', 'ear'),
}

# Bones used for length constraints & drawing (parent, child).
BONES = [
    ('hip', 'left_hip'), ('hip', 'right_hip'), ('hip', 'neck'), ('neck', 'head'),
    ('neck', 'nose'), ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ('left_ankle', 'left_heel'), ('left_ankle', 'left_big_toe'), ('left_ankle', 'left_small_toe'),
    ('right_ankle', 'right_heel'), ('right_ankle', 'right_big_toe'), ('right_ankle', 'right_small_toe'),
    ('left_heel', 'left_big_toe'), ('right_heel', 'right_big_toe'),
    ('left_hip', 'right_hip'), ('left_shoulder', 'right_shoulder'),
]
BONE_IDX = np.array([[IDX[a], IDX[b]] for a, b in BONES])

# Symmetric bone pairs: lengths are averaged L/R (human bodies are ~symmetric).
def _sym_pairs():
    out = []
    for k, (a, b) in enumerate(BONES):
        if 'left_' in a or 'left_' in b:
            ra, rb = a.replace('left_', 'right_'), b.replace('left_', 'right_')
            for m, (c, d) in enumerate(BONES):
                if (c, d) == (ra, rb):
                    out.append((k, m))
    return out

SYM_BONES = _sym_pairs()

FOOT = {
    'left': [IDX['left_ankle'], IDX['left_heel'], IDX['left_big_toe'], IDX['left_small_toe']],
    'right': [IDX['right_ankle'], IDX['right_heel'], IDX['right_big_toe'], IDX['right_small_toe']],
}

# Segment model: (name, proximal joint expr, distal joint expr, mass fraction, COM % from proximal)
# de Leva (1996) male values. Joint exprs are names or 'mid:a,b'.
SEGMENTS = [
    ('head_neck', 'neck', 'head', 0.0694, 0.50),
    ('trunk', 'hip', 'neck', 0.4346, 0.5138),
    ('upper_arm_L', 'left_shoulder', 'left_elbow', 0.0271, 0.5772),
    ('upper_arm_R', 'right_shoulder', 'right_elbow', 0.0271, 0.5772),
    ('forearm_L', 'left_elbow', 'left_wrist', 0.0162, 0.4574),
    ('forearm_R', 'right_elbow', 'right_wrist', 0.0162, 0.4574),
    ('hand_L', 'left_wrist', 'left_wrist', 0.0061, 0.0),
    ('hand_R', 'right_wrist', 'right_wrist', 0.0061, 0.0),
    ('thigh_L', 'left_hip', 'left_knee', 0.1416, 0.4095),
    ('thigh_R', 'right_hip', 'right_knee', 0.1416, 0.4095),
    ('shank_L', 'left_knee', 'left_ankle', 0.0433, 0.4459),
    ('shank_R', 'right_knee', 'right_ankle', 0.0433, 0.4459),
    ('foot_L', 'left_heel', 'left_big_toe', 0.0137, 0.4415),
    ('foot_R', 'right_heel', 'right_big_toe', 0.0137, 0.4415),
]


def center_of_mass(P):
    """P: (T,J,3) -> (T,3) whole-body COM."""
    com = np.zeros(P.shape[:1] + (3,))
    tot = 0.0
    for _, a, b, m, r in SEGMENTS:
        pa, pb = P[:, IDX[a]], P[:, IDX[b]]
        com += m * (pa + r * (pb - pa))
        tot += m
    return com / tot


# Joint angles reported in the scientific CSV: (name, a, b, c) -> angle at b in degrees.
ANGLES = [
    ('knee_L', 'left_hip', 'left_knee', 'left_ankle'),
    ('knee_R', 'right_hip', 'right_knee', 'right_ankle'),
    ('hip_L', 'neck', 'left_hip', 'left_knee'),
    ('hip_R', 'neck', 'right_hip', 'right_knee'),
    ('elbow_L', 'left_shoulder', 'left_elbow', 'left_wrist'),
    ('elbow_R', 'right_shoulder', 'right_elbow', 'right_wrist'),
    ('shoulder_L', 'hip', 'left_shoulder', 'left_elbow'),
    ('shoulder_R', 'hip', 'right_shoulder', 'right_elbow'),
    ('ankle_L', 'left_knee', 'left_ankle', 'left_big_toe'),
    ('ankle_R', 'right_knee', 'right_ankle', 'right_big_toe'),
]


def joint_angles(P):
    out = {}
    for n, a, b, c in ANGLES:
        u = P[:, IDX[a]] - P[:, IDX[b]]
        v = P[:, IDX[c]] - P[:, IDX[b]]
        cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
        out[n] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return out
