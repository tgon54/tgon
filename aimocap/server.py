"""Local web UI server (standard library only).
    python -m aimocap.server            -> http://127.0.0.1:8765
"""
import os
import re
import sys
import json
import time
import uuid
import shutil
import threading
import traceback
import webbrowser
import mimetypes
from urllib.parse import urlparse, parse_qs, unquote
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(ROOT, 'ui', 'index.html')
CLEANER_UI = os.path.join(ROOT, 'ui', 'cleaner.html')
WS = os.environ.get('AIMOCAP_WORKSPACE', os.path.join(os.path.dirname(ROOT), 'aimocap_workspace'))
UP, JOBS = os.path.join(WS, 'videos'), os.path.join(WS, 'jobs')
os.makedirs(UP, exist_ok=True); os.makedirs(JOBS, exist_ok=True)

RUNNING = {}          # job id -> dict(thread, cancel flag)
LOCK = threading.Lock()

STAGES = [('detect', 'كشف ثنائي الأبعاد'), ('calib', 'المعايرة الذاتية'), ('correct', 'التصحيح الذكي'), ('export', 'التصدير')]


class Cancelled(Exception):
    pass


# ----------------------------------------------------------------- helpers --
def jread(p, default=None):
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception:
        return default


def jwrite(p, d):
    tmp = p + '.tmp'
    json.dump(d, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, p)


def safe_name(n):
    n = os.path.basename(n)
    n = re.sub(r'[^\w.\-]+', '_', n, flags=re.UNICODE).strip('._') or 'video.mp4'
    return n[:80]


def video_info(vid):
    d = os.path.join(UP, vid)
    meta = jread(os.path.join(d, 'info.json'))
    if meta:
        return meta
    path = os.path.join(d, jread(os.path.join(d, 'name.json'))['file'])
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError('لا يمكن قراءة الفيديو (صيغة غير مدعومة؟)')
    meta = dict(id=vid, file=os.path.basename(path), name=jread(os.path.join(d, 'name.json'))['name'],
                w=int(cap.get(3)), h=int(cap.get(4)), fps=float(cap.get(5) or 30),
                frames=int(cap.get(7)), created=os.path.getmtime(path))
    cap.release()
    from .video import detect_grid
    try:
        xs, ys = detect_grid(path)
    except Exception:
        xs, ys = [0, meta['w']], [0, meta['h']]
    meta['grid'] = dict(xs=xs, ys=ys)
    jwrite(os.path.join(d, 'info.json'), meta)
    return meta


def video_path(vid):
    d = os.path.join(UP, vid)
    return os.path.join(d, jread(os.path.join(d, 'name.json'))['file'])


def frame_jpeg(vid, i, width=960):
    cap = cv2.VideoCapture(video_path(vid))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(i)))
    ok, f = cap.read(); cap.release()
    if not ok:
        f = np.zeros((360, 640, 3), np.uint8)
    if width and f.shape[1] > width:
        f = cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def job_summary(jid):
    d = os.path.join(JOBS, jid)
    j = jread(os.path.join(d, 'job.json'), {})
    if not j:
        return None
    j['running'] = jid in RUNNING
    if j.get('status') == 'running' and jid not in RUNNING:
        j['status'] = 'stopped'
    rep = jread(os.path.join(d, 'out', 'report.json'))
    if rep:
        m = rep.get('metrics', {})
        j['summary'] = dict(frames=rep.get('frames'), views=rep.get('views'), fps=rep.get('fps'),
                            raw=m.get('raw'), final=m.get('final'), triangulated=m.get('triangulated'),
                            swaps=rep.get('swaps'), glitch=rep.get('glitch_keypoints'),
                            inconsistent=rep.get('inconsistent_view_frames'),
                            reproj=rep.get('reproj_err_median_per_view'),
                            filled=rep.get('filled_samples_pct'), seconds=rep.get('seconds'),
                            contact=rep.get('contact_pct'), validation=rep.get('synthetic_validation'))
        out = os.path.join(d, 'out')
        j['files'] = sorted(f for f in os.listdir(out) if not f.endswith('.tmp'))
    return j


# --------------------------------------------------------------------- jobs --
def run_job(jid, cancel):
    d = os.path.join(JOBS, jid)
    jp = os.path.join(d, 'job.json')
    job = jread(jp)
    out = os.path.join(d, 'out')
    os.makedirs(out, exist_ok=True)
    state = dict(last=0.0)

    def log(msg):
        if cancel.is_set():
            raise Cancelled()
        msg = str(msg)
        job['log'].append(time.strftime('%H:%M:%S ') + msg)
        job['log'] = job['log'][-400:]
        m = re.search(r'frame (\d+)/(\d+)', msg)
        if m:
            job['stage'] = 'detect'
            job['progress'] = 0.02 + 0.68 * int(m.group(1)) / max(1, int(m.group(2)))
        elif 'loaded cached' in msg:
            job['stage'] = 'calib'; job['progress'] = 0.70
        elif 'testing camera layout' in msg or 'BA:' in msg:
            job['stage'] = 'calib'; job['progress'] = max(job['progress'], 0.72) + 0.03
        elif 'L/R swaps' in msg or 'AI glitch' in msg:
            job['stage'] = 'correct' if 'L/R' in msg else job['stage']; job['progress'] = max(job['progress'], 0.84 if 'L/R' in msg else 0.71)
        elif 'metrics:' in msg:
            job['stage'] = 'export'; job['progress'] = 0.93
        elif 'exported' in msg:
            job['progress'] = 1.0
        now = time.time()
        if now - state['last'] > 0.4 or m is None:
            jwrite(jp, job); state['last'] = now

    try:
        import warnings
        warnings.filterwarnings('ignore')
        from .pipeline import run
        s = job['settings']
        src = s.get('reuse_from')
        if src:
            c = os.path.join(JOBS, src, 'out', 'detections_2d.npz')
            if os.path.exists(c):
                shutil.copy(c, os.path.join(out, 'detections_2d.npz'))
                if os.path.exists(os.path.join(JOBS, src, 'out', 'debug_2d_overlay.mp4')):
                    shutil.copy(os.path.join(JOBS, src, 'out', 'debug_2d_overlay.mp4'), out)
        opts = dict(height=float(s['height']), fov=float(s['fov']), cutoff=float(s['cutoff']))
        if s.get('yaws'):
            opts['yaws'] = [float(v) for v in s['yaws']]
        log('بدء المعالجة')
        run(video_path(job['video']), out, grid=(s['xs'], s['ys']), panels=s['panels'], opts=opts,
            mode=s['mode'], device=s['device'], max_frames=s.get('max_frames') or None, log=log)
        job['status'] = 'done'; job['progress'] = 1.0; job['finished'] = time.time()
        job['log'].append(time.strftime('%H:%M:%S ') + 'اكتملت المعالجة')
    except Cancelled:
        job['status'] = 'cancelled'; job['log'].append('أُلغيت المعالجة')
    except BaseException as e:
        job['status'] = 'error'
        job['error'] = f'{type(e).__name__}: {e}'
        job['log'] += traceback.format_exc().splitlines()[-12:]
    finally:
        jwrite(jp, job)
        with LOCK:
            RUNNING.pop(jid, None)


def start_job(body):
    with LOCK:
        if RUNNING:
            raise ValueError('توجد معالجة قيد التشغيل. انتظر انتهاءها أو ألغها.')
    vid = body['video']
    video_info(vid)
    s = dict(xs=[int(v) for v in body['xs']], ys=[int(v) for v in body['ys']],
             panels=[int(v) for v in body['panels']], yaws=body.get('yaws') or None,
             height=float(body.get('height', 1.75)), mode=body.get('mode', 'performance'),
             device=body.get('device', 'auto'), max_frames=int(body.get('max_frames') or 0),
             fov=float(body.get('fov', 40)), cutoff=float(body.get('cutoff', 6)),
             reuse_from=body.get('reuse_from'))
    if len(s['panels']) < 2:
        raise ValueError('اختر زاويتين على الأقل')
    if s['yaws'] and len(s['yaws']) != len(s['panels']):
        raise ValueError('عدد الاتجاهات لا يطابق عدد الزوايا المختارة')
    jid = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:4]
    d = os.path.join(JOBS, jid); os.makedirs(d)
    job = dict(id=jid, video=vid, video_name=video_info(vid)['name'], settings=s, status='running',
               stage='detect' if not s['reuse_from'] else 'calib', progress=0.0, created=time.time(), log=[])
    jwrite(os.path.join(d, 'job.json'), job)
    cancel = threading.Event()
    t = threading.Thread(target=run_job, args=(jid, cancel), daemon=True)
    with LOCK:
        RUNNING[jid] = dict(thread=t, cancel=cancel)
    t.start()
    return jid


# ------------------------------------------------------- timeline cleaner --
def _out(jid):
    d = os.path.join(JOBS, os.path.basename(jid), 'out')
    if not os.path.exists(os.path.join(d, 'motion.json')):
        raise ValueError('this job has no motion yet')
    return d


def _motion_payload(m, issues=True):
    from .timeline import analyze, to_json
    d = to_json(m)
    d['quality'] = np.round(np.mean(m['quality'], 1), 3).tolist() if m.get('quality') is not None else None
    if issues:
        d['severity'], d['issues'] = analyze(m)
    return d


def job_motion(jid):
    from .timeline import load, load_edits
    from .skeleton import NAMES, BONES
    out = _out(jid)
    d = _motion_payload(load(out))
    j = jread(os.path.join(JOBS, jid, 'job.json'), {})
    d.update(id=jid, name=j.get('video_name', jid), joints=NAMES, bones=[list(b) for b in BONES],
             edits=load_edits(out), files=sorted(os.listdir(out)))
    return d


def clean_preview(jid, body):
    from .timeline import load, apply_edits
    c = apply_edits(load(_out(jid)), body.get('edits', []))
    d = _motion_payload(c)
    d['trim'] = c['trim']
    return d


_SKIP = re.compile(r'^(root|ik_.*|attach|interaction|center_of_mass|.*twist.*)$')


def rig_preview(jid, body):
    """Bone positions of the chosen mannequin exactly as exported (solver output, finger limits,
    relaxed fallback), mapped back to the viewer's space (meters, Y-up)."""
    from .timeline import load, apply_edits
    from .rig import load_rig
    from .retarget import retarget
    c = apply_edits(load(_out(jid)), body.get('edits', []))
    tid = body.get('target', 'ue4')
    rig = load_rig(tid)
    res = retarget(c['P'], c['fps'], rig, hands=c.get('hands'), contacts=c['contacts'])
    names = [b['name'] for b in rig['bones'] if b['kind'] != 'Null' and not _SKIP.match(b['name'])]
    keep = set(names)
    par = {b['name']: b['parent'] for b in rig['bones']}
    links = []
    for n in names:
        p = par[n]
        while p is not None and p not in keep:
            p = par[p]
        if p is not None:
            links.append([names.index(p), names.index(n)])
    X = np.stack([res['globals'][n][:, :3, 3] for n in names], 1)          # (T,B,3) rig units
    X = (X / res['scale']) @ res['basis']                                  # back to mocap space
    tips = [names.index(n) for n in names if re.match(r'^(thumb|index|middle|ring|pinky)_03_[lr]$', n)]
    out = dict(target=tid, names=names, links=links, tips=tips, pos=np.round(X, 4).tolist(),
               fingers=res['info']['hands'], mesh=False)
    from .rig import mesh_file_for, mesh_from_fbx
    mf = mesh_file_for(tid) if body.get('body', True) else None
    if mf:
        try:
            mesh = mesh_from_fbx(mf)
            A = np.eye(4); A[:3, :3] = res['basis'].T / res['scale']          # rig -> viewer space
            K = np.stack([A @ res['globals'][b] @ np.linalg.inv(mesh['bind'][b]) if b in res['globals']
                          else np.tile(A @ np.eye(4), (len(X), 1, 1)) for b in mesh['bones']], 1)   # (T,B,4,4)
            out.update(mesh=os.path.basename(mf), mesh_target=mesh_key(mf), skin=_b64(K[:, :, :3, :]))
        except Exception as e:
            out['mesh_error'] = f'{type(e).__name__}: {e}'
    return out


def _b64(a, dt=np.float32):
    import base64
    return base64.b64encode(np.ascontiguousarray(a, dt).tobytes()).decode('ascii')


def mesh_key(path):
    return os.path.splitext(os.path.basename(path))[0]


def mesh_payload(stem):
    from .rig import _folder_files, mesh_from_fbx
    p = _folder_files().get(stem)
    if not p or not p.lower().endswith('.fbx'):
        raise ValueError('no mesh file ' + stem)
    m = mesh_from_fbx(p)
    return dict(name=stem, n=len(m['verts']), nf=len(m['faces']), bones=m['bones'],
                verts=_b64(m['verts']), faces=_b64(m['faces'], np.uint32),
                joints=_b64(m['joints'], np.uint16), weights=_b64(m['weights']))


def clean_export(jid, body):
    from .timeline import save_clean
    from .rig import list_targets
    ok = {t['id'] for t in list_targets() if t['ok']}
    targets = [t for t in body.get('targets', ['ue4']) if t in ok] or ['ue4']
    log = []
    files, c = save_clean(_out(jid), body.get('edits', []), targets, bool(body.get('root_motion')),
                          body.get('name') or None, log=log.append)
    return dict(files=files, frames=len(c['P']), log=log)


def import_motion(raw, name):
    """Register an AI-MoCap motion.json (e.g. from the CLI) as a finished job for the cleaner."""
    d = json.loads(raw)
    if 'corrected' not in d or 'fps' not in d:
        raise ValueError('not an AI-MoCap motion.json (needs "corrected" and "fps")')
    jid = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:4]
    out = os.path.join(JOBS, jid, 'out'); os.makedirs(out)
    T = len(d['corrected'])
    d.setdefault('quality', [[1.0] * len(d['corrected'][0])] * T)
    json.dump(d, open(os.path.join(out, 'motion.json'), 'w'), separators=(',', ':'))
    jwrite(os.path.join(JOBS, jid, 'job.json'),
           dict(id=jid, video=None, video_name=name, settings=dict(panels=[], height='-', yaws=None), status='done', stage='export',
                progress=1.0, created=time.time(), imported=True, log=['imported ' + name]))
    return jid


# ------------------------------------------------------------------ server --
class H(BaseHTTPRequestHandler):
    server_version = 'aimocap'

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json; charset=utf-8', extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, default=float).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg, code=400):
        self._send(code, {'error': msg})

    def _file(self, path, download=False):
        if not os.path.isfile(path):
            return self._err('not found', 404)
        ctype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
        if path.endswith('.bvh') or path.endswith('.npy') or path.endswith('.npz') or path.endswith('.toml'):
            ctype = 'application/octet-stream'
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(size))
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(path)}"')
        self.end_headers()
        with open(path, 'rb') as f:
            shutil.copyfileobj(f, self.wfile, 1 << 20)

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n) or b'{}')

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query); p = u.path
        try:
            if p in ('/', '/index.html'):
                return self._send(200, open(UI, 'rb').read(), 'text/html; charset=utf-8')
            if p in ('/cleaner', '/cleaner.html'):
                return self._send(200, open(CLEANER_UI, 'rb').read(), 'text/html; charset=utf-8')
            m = re.match(r'^/api/jobs/([\w-]+)/motion$', p)
            if m:
                return self._send(200, job_motion(m.group(1)))
            m = re.match(r'^/api/mesh/([^/]+)$', p)
            if m:
                return self._send(200, mesh_payload(unquote(m.group(1))))
            if p == '/api/targets':
                from .rig import list_targets, MANNEQUIN_DIR
                return self._send(200, dict(folder=MANNEQUIN_DIR, targets=list_targets()))
            if p == '/api/status':
                return self._send(200, dict(workspace=WS, running=list(RUNNING), gpu=_gpu()))
            if p == '/api/videos':
                vs = []
                for v in sorted(os.listdir(UP), reverse=True):
                    try:
                        vs.append(video_info(v))
                    except Exception:
                        pass
                return self._send(200, vs)
            m = re.match(r'^/api/videos/([\w-]+)$', p)
            if m:
                return self._send(200, video_info(m.group(1)))
            m = re.match(r'^/api/videos/([\w-]+)/frame$', p)
            if m:
                return self._send(200, frame_jpeg(m.group(1), int(q.get('i', ['0'])[0]), int(q.get('w', ['960'])[0])), 'image/jpeg')
            if p == '/api/jobs':
                out = [s for s in (job_summary(j) for j in sorted(os.listdir(JOBS), reverse=True)) if s]
                for s in out:
                    s.pop('log', None)
                return self._send(200, out)
            m = re.match(r'^/api/jobs/([\w-]+)$', p)
            if m:
                s = job_summary(m.group(1))
                return self._send(200, s) if s else self._err('not found', 404)
            m = re.match(r'^/files/([\w-]+)/([\w.\-]+)$', p)
            if m:
                return self._file(os.path.join(JOBS, m.group(1), 'out', m.group(2)), download='dl' in q)
            return self._err('not found', 404)
        except Exception as e:
            return self._err(str(e), 500)

    def do_POST(self):
        p = urlparse(self.path).path
        try:
            if p == '/api/upload':
                name = unquote(self.headers.get('X-Filename', 'video.mp4'))
                n = int(self.headers.get('Content-Length') or 0)
                if n <= 0:
                    return self._err('ملف فارغ')
                vid = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:4]
                d = os.path.join(UP, vid); os.makedirs(d)
                fn = safe_name(name)
                if not os.path.splitext(fn)[1]:
                    fn += '.mp4'
                with open(os.path.join(d, fn), 'wb') as f:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            break
                        f.write(chunk); left -= len(chunk)
                jwrite(os.path.join(d, 'name.json'), dict(name=name, file=fn))
                try:
                    return self._send(200, video_info(vid))
                except Exception as e:
                    shutil.rmtree(d, ignore_errors=True)
                    return self._err(str(e))
            if p == '/api/jobs':
                return self._send(200, {'id': start_job(self._body())})
            m = re.match(r'^/api/jobs/([\w-]+)/(clean|export|rig_preview)$', p)
            if m:
                if m.group(1) in RUNNING:
                    raise ValueError('job is still running')
                fn = {'clean': clean_preview, 'export': clean_export, 'rig_preview': rig_preview}[m.group(2)]
                return self._send(200, fn(m.group(1), self._body()))
            if p == '/api/import_motion':
                n = int(self.headers.get('Content-Length') or 0)
                return self._send(200, {'id': import_motion(self.rfile.read(n), unquote(self.headers.get('X-Filename', 'motion.json')))})
            m = re.match(r'^/api/jobs/([\w-]+)/cancel$', p)
            if m:
                r = RUNNING.get(m.group(1))
                if r:
                    r['cancel'].set()
                return self._send(200, {'ok': True})
            return self._err('not found', 404)
        except ValueError as e:
            return self._err(str(e))
        except Exception as e:
            return self._err(f'{type(e).__name__}: {e}', 500)

    def do_DELETE(self):
        p = urlparse(self.path).path
        m = re.match(r'^/api/(jobs|videos)/([\w-]+)$', p)
        if not m:
            return self._err('not found', 404)
        if m.group(2) in RUNNING:
            return self._err('لا يمكن حذف معالجة قيد التشغيل')
        shutil.rmtree(os.path.join(JOBS if m.group(1) == 'jobs' else UP, m.group(2)), ignore_errors=True)
        return self._send(200, {'ok': True})


def _gpu():
    try:
        import onnxruntime as ort
        return 'CUDAExecutionProvider' in ort.get_available_providers()
    except Exception:
        return None


class Server(ThreadingHTTPServer):
    # Never share a port with another program (on Windows SO_REUSEADDR lets two
    # servers bind the same port and the other one silently answers).
    allow_reuse_address = False

    def server_bind(self):
        import socket
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main():
    port = int(os.environ.get('AIMOCAP_PORT', '8642'))
    for p in range(port, port + 30):
        try:
            srv = Server(('127.0.0.1', p), H)
            break
        except OSError:
            continue
    url = f'http://127.0.0.1:{srv.server_address[1]}/'
    print('AI-MoCap UI ->', url)
    print('workspace   ->', WS)
    print('(Ctrl+C to stop)')
    if '--no-browser' not in sys.argv:
        page = url + ('cleaner' if '--cleaner' in sys.argv else '')
        threading.Timer(0.8, lambda: webbrowser.open(page)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
