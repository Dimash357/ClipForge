"""NVIDIA CUDA 12 libraries for faster-whisper on Windows (standard library only).

CTranslate2 (faster-whisper) needs cuBLAS 12 and cuDNN 9 DLLs. install_gpu.bat installs them as pip packages
(nvidia-cublas-cu12, nvidia-cudnn-cu12) into the project's env: they land in site-packages/nvidia/*/bin, a
folder Windows does not search by itself. register() adds those folders to the DLL search path.
"""
import os
import site
import sysconfig
from pathlib import Path

REQUIRED = ('cublas64_12.dll', 'cudnn64_9.dll')
_done = []


def nvidia_dirs():
    roots = set()
    for key in ('purelib', 'platlib'):
        try:
            roots.add(sysconfig.get_paths()[key])
        except Exception:
            pass
    try:
        roots.update(site.getsitepackages())
    except Exception:
        pass
    out = []
    for r in roots:
        base = Path(r) / 'nvidia'
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir()):
            for name in ('bin', 'lib'):
                d = sub / name
                if d.is_dir() and any(d.glob('*.dll')):
                    out.append(d)
    return out


def register():
    """Make the pip-installed NVIDIA DLLs loadable (Windows only; harmless elsewhere). Returns the folders."""
    if os.name != 'nt' or _done:
        return list(_done)
    for d in nvidia_dirs():
        try:
            os.add_dll_directory(str(d))
        except (OSError, AttributeError):
            pass
        os.environ['PATH'] = str(d) + os.pathsep + os.environ.get('PATH', '')
        _done.append(d)
    return list(_done)


def missing_dlls():
    """Names of the CUDA 12 / cuDNN 9 DLLs that cannot be loaded (Windows), [] when all load or not Windows."""
    if os.name != 'nt':
        return []
    import ctypes
    register()
    bad = []
    for name in REQUIRED:
        try:
            ctypes.WinDLL(name)
        except OSError:
            bad.append(name)
    return bad
