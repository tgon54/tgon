"""CLI:  python -m aimocap VIDEO.mp4 OUT_DIR [options]"""
import argparse


def main():
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(prog='aimocap', description='AI-video multi-view motion capture with smart correction')
    ap.add_argument('video')
    ap.add_argument('out')
    ap.add_argument('--grid', default='auto', help="'auto' or RxC, e.g. 2x2 or 2x5")
    ap.add_argument('--panels', default=None, help='panel indices to use (row-major), e.g. 0,2,5,8')
    ap.add_argument('--yaws', default=None, help='camera yaw per used panel in degrees, e.g. 0,90,180,270 (default: auto)')
    ap.add_argument('--height', type=float, default=1.75, help='subject height in meters')
    ap.add_argument('--fov', type=float, default=40.0, help='initial field-of-view guess (refined automatically)')
    ap.add_argument('--cutoff', type=float, default=6.0, help='low-pass cutoff Hz')
    ap.add_argument('--mode', default='performance', choices=['performance', 'balanced', 'lightweight'])
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--no-hands', action='store_true', help='skip smart hand/finger tracking')
    ap.add_argument('--subject', default='human', choices=['human', 'dog', 'horse'])
    ap.add_argument('--single-view', action='store_true',
                    help='one camera (any direction); with a grid, pick the view with --panels N')
    ap.add_argument('--weight', type=float, default=None, help='body mass in kg (default: 70 human, 25 dog, 500 horse)')
    ap.add_argument('--no-physics', action='store_true', help='skip the gravity/contact/balance pass')
    a = ap.parse_args()
    from .pipeline import run
    opts = dict(height=a.height, fov=a.fov, cutoff=a.cutoff, physics=not a.no_physics)
    if a.weight:
        opts['weight'] = a.weight
    if a.subject != 'human' and a.height == 1.75:
        opts.pop('height')                       # species default (withers height)
    if a.yaws:
        opts['yaws'] = [float(v) for v in a.yaws.split(',')]
    panels = [int(v) for v in a.panels.split(',')] if a.panels else None
    run(a.video, a.out, grid=a.grid, panels=panels, opts=opts, mode=a.mode, device=a.device, max_frames=a.max_frames, hands=not a.no_hands,
        subject=a.subject, single_view=a.single_view)
    print('done ->', a.out)


if __name__ == '__main__':
    main()
