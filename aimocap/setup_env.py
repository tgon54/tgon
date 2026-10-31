"""One-shot dependency installer:  python -m aimocap.setup_env
Installs everything for the Python that runs it. rtmlib is installed WITHOUT its
dependencies because it pins the CPU 'onnxruntime', which would break the GPU build."""
import subprocess
import sys
import shutil


def pip(*args):
    cmd = [sys.executable, '-m', 'pip', *args]
    print('>', ' '.join(cmd[2:]), flush=True)
    return subprocess.call(cmd)


def has_nvidia():
    exe = shutil.which('nvidia-smi')
    if not exe:
        return False
    try:
        return subprocess.call([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15) == 0
    except Exception:
        return False


def main():
    print(f'Python: {sys.executable}  ({sys.version.split()[0]})')
    pip('install', '--upgrade', 'pip')
    if pip('install', 'numpy', 'scipy', 'opencv-python', 'matplotlib', 'tqdm') != 0:
        sys.exit('pip failed while installing base packages')
    if pip('install', '--no-deps', 'rtmlib') != 0:
        sys.exit('pip failed while installing rtmlib')
    gpu = has_nvidia()
    print('NVIDIA GPU detected' if gpu else 'No NVIDIA GPU detected -> CPU build')
    if gpu:
        pip('uninstall', '-y', 'onnxruntime')          # CPU build would shadow the GPU one
        # GPU build + CUDA/cuDNN runtime DLLs from pip (no CUDA Toolkit install needed)
        from aimocap.gpu_check import PKGS
        ok = pip('install', '--upgrade', *PKGS) == 0
        if not ok:
            print('onnxruntime-gpu failed, falling back to CPU build')
            pip('install', 'onnxruntime')
    else:
        pip('uninstall', '-y', 'onnxruntime-gpu')
        pip('install', '--upgrade', 'onnxruntime')
    # report
    print('\n================ CHECK ================')
    bad = []
    for m in ('numpy', 'scipy', 'cv2', 'matplotlib', 'rtmlib', 'onnxruntime'):
        try:
            __import__(m); print(f'  OK   {m}')
        except Exception as e:
            bad.append(m); print(f'  FAIL {m}: {e}')
    try:
        import onnxruntime as ort
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        prov = ort.get_available_providers()
        print('  onnxruntime providers:', prov)
        if gpu and 'CUDAExecutionProvider' not in prov:
            print('  NOTE: GPU build installed but CUDA provider missing -> will run on CPU.')
    except Exception:
        pass
    if bad:
        sys.exit('Some packages failed: ' + ', '.join(bad))
    print('\nAll good. Start the interface with start_ui.bat')


if __name__ == '__main__':
    main()
