"""Hardware / tooling diagnostics. Acceleration is used only after a real test passes."""
import importlib
import logging
import os
import shutil
import subprocess
import sys
import threading
import time

from django.conf import settings

log = logging.getLogger('studio.diag')
_cache = {}
_lock = threading.Lock()
TTL = 600


def _cached(key, fn, ttl=TTL):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    value = fn()
    with _lock:
        _cache[key] = (time.time(), value)
    return value


def reset_cache():
    with _lock:
        _cache.clear()


def _run(cmd, timeout=30):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                          errors='replace', timeout=timeout)


def ffmpeg_info():
    def probe():
        info = {'ffmpeg': False, 'ffprobe': False, 'version': '', 'libass': False, 'nvenc_listed': False,
                'libx264': False, 'filters_ok': {}}
        if shutil.which('ffmpeg'):
            info['ffmpeg'] = True
            try:
                info['version'] = _run(['ffmpeg', '-hide_banner', '-version']).stdout.splitlines()[0]
                flt = _run(['ffmpeg', '-hide_banner', '-filters']).stdout
                enc = _run(['ffmpeg', '-hide_banner', '-encoders']).stdout
                info['libass'] = ' ass ' in flt
                for name in ('xfade', 'boxblur', 'gblur', 'blackdetect', 'silencedetect', 'xstack'):
                    info['filters_ok'][name] = f' {name} ' in flt
                info['nvenc_listed'] = 'h264_nvenc' in enc
                info['libx264'] = 'libx264' in enc
            except Exception as exc:
                info['error'] = str(exc)
        info['ffprobe'] = bool(shutil.which('ffprobe'))
        return info
    return _cached('ffmpeg', probe)


def nvenc_works():
    """Real 0.2 s test encode. Listed-but-unusable NVENC (no GPU / old driver) returns False."""
    def probe():
        if not ffmpeg_info().get('nvenc_listed'):
            return {'ok': False, 'detail': 'h264_nvenc отсутствует в этой сборке ffmpeg'}
        try:
            p = _run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                      'color=c=black:s=640x360:d=0.3:r=30', '-c:v', 'h264_nvenc', '-f', 'null', '-'], timeout=40)
            if p.returncode == 0:
                return {'ok': True, 'detail': 'тестовое кодирование прошло'}
            return {'ok': False, 'detail': (p.stderr or '')[-300:].strip() or 'ошибка кодирования'}
        except Exception as exc:
            return {'ok': False, 'detail': str(exc)}
    return _cached('nvenc', probe)


_CUDA_SNIPPET = (
    "import json,sys\n"
    "out={'count':0,'types':[]}\n"
    "try:\n"
    "    import ctranslate2 as c\n"
    "    out['count']=c.get_cuda_device_count()\n"
    "    if out['count']>0: out['types']=sorted(c.get_supported_compute_types('cuda'))\n"
    "except Exception as e:\n"
    "    out['error']=str(e)[:300]\n"
    "print(json.dumps(out))\n"
)


def cuda_probe():
    """ctranslate2 CUDA check in a *child process* so a driver/library crash cannot kill the server."""
    def probe():
        try:
            p = _run([sys.executable, '-c', _CUDA_SNIPPET], timeout=60)
            import json
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as exc:
            return {'count': 0, 'types': [], 'error': str(exc)[:300]}
    return _cached('cuda', probe)


def gpu_info():
    def probe():
        if not shutil.which('nvidia-smi'):
            return {}
        try:
            out = _run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                        '--format=csv,noheader,nounits'], timeout=15).stdout.strip().splitlines()
            if not out:
                return {}
            name, mem, drv = [x.strip() for x in out[0].split(',')]
            return {'name': name, 'vram_mb': int(float(mem)), 'driver': drv}
        except Exception:
            return {}
    return _cached('gpu', probe)


_whisper_gpu_failed = False


def mark_whisper_gpu_failed(reason=''):
    global _whisper_gpu_failed
    _whisper_gpu_failed = True
    log.warning('Whisper CUDA disabled for this session: %s', reason)


def whisper_settings():
    """(device, compute_type, note). env WHISPER_DEVICE: cpu | cuda | auto (default auto)."""
    device = os.getenv('WHISPER_DEVICE', 'auto').lower()
    compute = os.getenv('WHISPER_COMPUTE', '')
    if device == 'cpu':
        return 'cpu', compute or 'int8', 'CPU (задано WHISPER_DEVICE=cpu)'
    if device == 'cuda':
        return 'cuda', compute or 'float16', 'CUDA (принудительно: WHISPER_DEVICE=cuda)'
    # auto
    if _whisper_gpu_failed:
        return 'cpu', compute or 'int8', 'CPU (CUDA не заработала в этой сессии)'
    probe = cuda_probe()
    if probe.get('count', 0) > 0:
        types = probe.get('types', [])
        ctype = compute or ('float16' if 'float16' in types else 'int8_float16' if 'int8_float16' in types else 'int8')
        return 'cuda', ctype, f'CUDA авто ({ctype}); при ошибке библиотек — автоматический откат на CPU'
    return 'cpu', compute or 'int8', 'CPU (CUDA недоступна: ' + (probe.get('error') or 'устройства не найдены') + ')'


def choose_encoder(preference='auto'):
    """-> ('h264_nvenc'|'libx264', note). NVENC only if listed AND a test encode succeeded."""
    if preference == 'cpu':
        return 'libx264', 'CPU (выбрано вручную)'
    t = nvenc_works()
    if t['ok']:
        return 'h264_nvenc', 'NVIDIA NVENC'
    return 'libx264', 'CPU libx264' + ('' if preference == 'auto' else f' (NVENC недоступен: {t["detail"][:80]})')


def _version(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, '__version__', 'установлен')
    except Exception as exc:
        return None if isinstance(exc, ImportError) else f'ошибка: {exc}'


def full_report():
    ff = ffmpeg_info()
    dev, ctype, note = whisper_settings()
    enc, enc_note = choose_encoder('auto')
    try:
        free = shutil.disk_usage(str(settings.MEDIA_ROOT if settings.MEDIA_ROOT.exists() else settings.BASE_DIR)).free
    except Exception:
        free = None
    cascades = None
    try:
        import cv2
        cascades = os.path.exists(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception:
        pass
    return {
        'python': sys.version.split()[0],
        'platform': sys.platform,
        'ffmpeg': ff,
        'nvenc': nvenc_works(),
        'cuda': cuda_probe(),
        'gpu': gpu_info(),
        'whisper': {'device': dev, 'compute': ctype, 'note': note, 'model': os.getenv('WHISPER_MODEL', 'small')},
        'encoder': {'name': enc, 'note': enc_note},
        'libs': {'faster_whisper': _version('faster_whisper'), 'av': _version('av'), 'cv2': _version('cv2'),
                 'numpy': _version('numpy'), 'yt_dlp': _version('yt_dlp'), 'PIL': _version('PIL'),
                 'django': _version('django'), 'haar_cascades': cascades},
        'disk_free_gb': round(free / 1024 ** 3, 1) if free is not None else None,
        'max_jobs': settings.MAX_CONCURRENT_JOBS,
        'ollama': os.getenv('USE_OLLAMA', '0') == '1',
        'ai': _ai_status(),
    }


def _ai_status():
    try:
        from . import ai_provider
        return ai_provider.status()
    except Exception as exc:
        return {'kind': None, 'reason': str(exc)[:100]}
