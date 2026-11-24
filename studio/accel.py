"""Hardware video decoding (NVIDIA NVDEC through ffmpeg `-hwaccel cuda`) with automatic CPU fallback.

Decoding is bit-exact on the GPU, so using NVDEC never changes the analysed or rendered pictures - it only moves
the heaviest part of reading long 1080p60 AV1/VP9/H.264 streams off the CPU. Support is checked per source (codec,
size, bit depth) with a 2-second test decode that *fails* when NVDEC cannot decode it (output format cuda), and the
answer is cached. CLIPFORGE_HWDECODE=off disables it.

This module only answers "can this file be decoded on the GPU"; which job uses the GPU when is the scheduler's
business (later), not this module's.
"""
import json
import logging
import os
import subprocess
import threading

from django.conf import settings

log = logging.getLogger('studio.accel')

_lock = threading.Lock()
_cache = {}
_listed = []


def enabled():
    return str(getattr(settings, 'HW_DECODE', os.getenv('CLIPFORGE_HWDECODE', 'auto'))).lower() not in (
        'off', '0', 'false', 'no', 'cpu')


def _cuda_listed():
    if not _listed:
        try:
            out = subprocess.run(['ffmpeg', '-hide_banner', '-hwaccels'], capture_output=True, text=True, timeout=20,
                                 stdin=subprocess.DEVNULL).stdout
            _listed.append('cuda' in out.split())
        except Exception:
            _listed.append(False)
    return _listed[0]


def video_info(source):
    """codec, pix_fmt of the first video stream (cheap ffprobe)."""
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                              'stream=codec_name,pix_fmt', '-of', 'json', str(source)], capture_output=True,
                             text=True, timeout=60, stdin=subprocess.DEVNULL).stdout
        st = (json.loads(out or '{}').get('streams') or [{}])[0]
        return st.get('codec_name') or '', st.get('pix_fmt') or ''
    except Exception:
        return '', ''


def download_format(source):
    """Pixel format of frames downloaded from the GPU (8-bit -> nv12, 10/12-bit -> p010le)."""
    _, pix = video_info(source)
    return 'p010le' if any(b in pix for b in ('10', '12')) else 'nv12'


def nvdec_ok(source):
    """True if NVDEC can decode this file (tested once per file version, then cached)."""
    if not enabled() or not _cuda_listed():
        return False
    try:
        st = os.stat(source)
    except OSError:
        return False
    key = (str(source), st.st_size, st.st_mtime_ns)
    with _lock:
        if key in _cache:
            return _cache[key]
    cmd = ['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
           '-t', '2', '-i', str(source), '-an', '-sn', '-dn', '-f', 'null', '-']
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
                           encoding='utf-8', errors='replace')
        ok = p.returncode == 0 and 'error' not in p.stderr.lower()
        if not ok:
            log.info('NVDEC not used for %s: %s', os.path.basename(str(source)), p.stderr.strip()[-200:])
    except Exception as exc:
        log.info('NVDEC test failed: %s', exc)
        ok = False
    with _lock:
        _cache[key] = ok
    if ok:
        log.info('NVDEC decoding available for %s', os.path.basename(str(source)))
    return ok


def decode_args(source):
    """Input options for GPU decoding with frames returned to system memory (CPU filters stay unchanged)."""
    return ['-hwaccel', 'cuda'] if nvdec_ok(source) else []


def status():
    return {'enabled': enabled(), 'cuda_hwaccel': _cuda_listed()}
