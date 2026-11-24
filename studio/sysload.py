"""CPU / GPU load for the progress UI. No extra packages: /proc/stat or GetSystemTimes for the CPU, nvidia-smi
(installed with the NVIDIA driver) for the GPU. Values are cached for 2 s, so polling pages cost nothing."""
import os
import subprocess
import threading
import time

_lock = threading.Lock()
_cache = {'at': 0.0, 'value': None}
_cpu_prev = [None]


def _cpu_times():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        v = lambda ft: (ft.dwHighDateTime << 32) | ft.dwLowDateTime        # noqa: E731
        return v(idle), v(kernel) + v(user)            # kernel time includes idle time
    try:
        with open('/proc/stat') as f:
            parts = [int(x) for x in f.readline().split()[1:]]
        return parts[3] + (parts[4] if len(parts) > 4 else 0), sum(parts)
    except (OSError, ValueError):
        return None


def cpu_percent():
    cur = _cpu_times()
    prev, _cpu_prev[0] = _cpu_prev[0], cur
    if not cur or not prev or cur[1] <= prev[1]:
        return None
    return round(100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1])), 1)


def gpu():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,utilization.decoder,'
                              'utilization.encoder,memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=3, stdin=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if out.returncode:
            return None
        f = [x.strip() for x in out.stdout.splitlines()[0].split(',')]
        num = lambda x: float(x) if x not in ('', '[N/A]', 'N/A') else None      # noqa: E731
        return {'name': f[0], 'util': num(f[1]), 'decoder': num(f[2]), 'encoder': num(f[3]),
                'mem_used': num(f[4]), 'mem_total': num(f[5])}
    except Exception:
        return None


def snapshot():
    with _lock:
        now = time.monotonic()
        if _cache['value'] is not None and now - _cache['at'] < 2.0:
            return _cache['value']
        value = {'cpu': cpu_percent(), 'cores': os.cpu_count(), 'gpu': gpu(), 'at': time.time()}
        _cache.update(at=now, value=value)
        return value
