"""GPU diagnosis / repair:  python -m aimocap.gpu_check [--fix]"""
import glob
import os
import shutil
import subprocess
import sys

PKGS = ['onnxruntime-gpu[cuda,cudnn]', 'nvidia-cublas', 'nvidia-cuda-runtime', 'nvidia-cufft',
        'nvidia-curand', 'nvidia-cuda-nvrtc', 'nvidia-cudnn-cu13']


def sh(cmd):
    print('>', ' '.join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    fix = '--fix' in sys.argv
    print('Python:', sys.executable)
    print('\n[1] NVIDIA driver')
    smi = shutil.which('nvidia-smi')
    if smi:
        out = subprocess.run([smi, '--query-gpu=name,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True).stdout.strip()
        print('   ', out or 'nvidia-smi returned nothing')
        try:
            drv = float(out.split(',')[-1].strip().split('.')[0])
            if drv < 580:
                print('    !! Driver too old for CUDA 13 (need 580+). Update the NVIDIA driver.')
        except Exception:
            pass
    else:
        print('    nvidia-smi not found -> no NVIDIA driver / GPU visible')

    if fix:
        print('\n[fix] installing GPU runtime packages (~1-2 GB, one time)')
        sh([sys.executable, '-m', 'pip', 'uninstall', '-y', 'onnxruntime'])
        if sh([sys.executable, '-m', 'pip', 'install', '--upgrade', *PKGS]) != 0:
            print('pip failed - send this window as a screenshot')

    print('\n[2] onnxruntime')
    try:
        import onnxruntime as ort
    except Exception as e:
        print('    cannot import onnxruntime:', e); return
    print('    version', ort.__version__, '| file', ort.__file__)
    try:
        import importlib.metadata as md
        names = [d.metadata['Name'] for d in md.distributions()]
        both = [n for n in names if n and n.lower() in ('onnxruntime', 'onnxruntime-gpu')]
        print('    installed builds:', both)
        if len(both) > 1:
            print('    !! Both CPU and GPU builds installed -> run: python -m aimocap.gpu_check --fix')
    except Exception:
        pass

    print('\n[3] CUDA DLLs from pip')
    sp = os.path.dirname(os.path.dirname(ort.__file__))
    for pat in ('nvidia/cu13/bin/*/cublasLt64_13.dll', 'nvidia/cu13/bin/*/cudart64_13.dll', 'nvidia/cudnn/bin/cudnn64_9.dll'):
        hits = glob.glob(os.path.join(sp, pat))
        print('    ', 'OK  ' if hits else 'MISSING', pat)
    if hasattr(ort, 'preload_dlls'):
        try:
            ort.preload_dlls()
        except Exception as e:
            print('    preload_dlls error:', e)

    print('\n[4] Real GPU session test')
    models = glob.glob(os.path.join(os.path.expanduser('~'), '.cache', 'rtmlib', 'hub', 'checkpoints', '*.onnx'))
    if not models:
        print('    no model cached yet (run one processing first)'); return
    try:
        s = ort.InferenceSession(models[0], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        prov = s.get_providers()
        print('    session providers:', prov)
        if 'CUDAExecutionProvider' in prov:
            print('\n    >>> GPU WORKS <<<')
        else:
            print('\n    >>> GPU NOT USED (CPU fallback). Run: python -m aimocap.gpu_check --fix')
    except Exception as e:
        print('    session error:', e)


if __name__ == '__main__':
    main()
