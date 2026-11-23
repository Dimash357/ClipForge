"""Asset URIs -> safe local paths, and ffprobe facts about them (for the Timeline renderer).

URI schemes
    media:<rel>   file under MEDIA_ROOT (uploaded sources, renders)
    cache:<rel>   file under MEDIA_ROOT/analysis (analysis cache)
    gen:<rel>     generated file under <base_dir>/gen
    file:<rel>    file under <base_dir> (default MEDIA_ROOT); an absolute path is accepted only inside an allowed root
Every resolved path must stay inside its root (no '..' escapes, no symlink tricks), must be a regular file and is
handed to ffmpeg as 'file:<absolute path>' so a name can never be read as an ffmpeg protocol or option.
"""
import json
import os
import subprocess
from pathlib import Path

from django.conf import settings

from .serialize import TimelineError

SCHEMES = ('media', 'cache', 'gen', 'file')


class AssetError(TimelineError):
    pass


def _inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class Resolver:
    def __init__(self, base_dir=None, roots=()):
        self.media = Path(settings.MEDIA_ROOT).resolve()
        self.base = Path(base_dir).resolve() if base_dir else self.media
        self.roots = [self.media, self.base] + [Path(r).resolve() for r in roots]

    def root_for(self, scheme):
        return {'media': self.media, 'cache': self.media / 'analysis', 'gen': self.base / 'gen',
                'file': self.base}[scheme]

    def resolve(self, uri, path='asset'):
        """-> absolute Path of an existing regular file, or AssetError."""
        if not isinstance(uri, str) or ':' not in uri:
            raise AssetError(f'{path}: bad uri {uri!r}', [{'path': path, 'message': f'bad uri {uri!r}'}])
        scheme, rel = uri.split(':', 1)
        if scheme not in SCHEMES:
            msg = f'unsupported uri scheme {scheme!r} (supported: {", ".join(SCHEMES)})'
            raise AssetError(f'{path}: {msg}', [{'path': path, 'message': msg}])
        if not rel or '\x00' in rel or any(c in rel for c in '\r\n'):
            raise AssetError(f'{path}: bad uri {uri!r}', [{'path': path, 'message': 'empty or unsafe path'}])
        root = self.root_for(scheme)
        p = Path(rel)
        if p.is_absolute():
            if scheme != 'file':
                raise AssetError(f'{path}: absolute path in {scheme}:', [{'path': path, 'message': 'must be relative'}])
            full = p.resolve()
            ok = any(_inside(full, r) for r in self.roots)
        else:
            full = (root / p).resolve()
            ok = _inside(full, root.resolve())
        if not ok:
            msg = f'{uri!r} points outside the allowed folders'
            raise AssetError(f'{path}: {msg}', [{'path': path, 'message': msg}])
        if not full.is_file():
            msg = f'file not found: {full}'
            raise AssetError(f'{path}: {msg}', [{'path': path, 'message': msg}])
        return full


def ffmpeg_input(path):
    """The argument for `-i`: the file protocol, never a protocol or option taken from the name."""
    return 'file:' + str(path)


def signature(path):
    """Cheap change detector for cache keys: size + modification time."""
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


_probe_cache = {}


def probe(path):
    """-> {'duration', 'width', 'height', 'has_video', 'has_audio'} (cached per path+size+mtime)."""
    key = (str(path), *signature(path))
    if key in _probe_cache:
        return _probe_cache[key]
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                        'format=duration:stream=codec_type,width,height,duration', '-of', 'json', ffmpeg_input(path)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
                       timeout=120)
    if p.returncode:
        raise AssetError(f'cannot read media file {path}: {p.stderr.strip()[-200:]}')
    d = json.loads(p.stdout or '{}')
    streams = d.get('streams') or []
    v = next((s for s in streams if s.get('codec_type') == 'video'), None)
    dur = d.get('format', {}).get('duration')
    info = {'duration': float(dur) if dur not in (None, 'N/A') else None,
            'width': int(v['width']) if v and v.get('width') else None,
            'height': int(v['height']) if v and v.get('height') else None,
            'has_video': v is not None, 'has_audio': any(s.get('codec_type') == 'audio' for s in streams)}
    _probe_cache[key] = info
    return info
