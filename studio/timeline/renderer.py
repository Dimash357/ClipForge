"""Timeline renderer (experimental): Timeline IR -> mp4, with a per-scene cache.

    from studio.timeline import renderer
    report = renderer.render(tl, 'out.mp4', renderer.RenderSettings(base_dir='my_project'))
    report = renderer.render(tl, 'out.mp4', settings, rerender={'sc_...'})      # redo one scene only

Steps: validate -> check_supported -> resolve + probe assets -> scene passes (cached) -> final pass.
A scene's cache key = scene fingerprint + its transitions + the size/mtime of every file it reads + the picture
settings + RENDERER_VERSION: change any of them and only that scene is rendered again.
NVENC is used when it works (diagnostics.choose_encoder); a failed NVENC run is retried on the CPU (libx264).
"""
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from django.conf import settings as dj

from .. import diagnostics, exports
from ..procutil import JobCancelled, run_ffmpeg
from . import compile as C
from . import media as M
from .serialize import TimelineError, content_hash, scene_fingerprint, to_dict
from .subtitles import write_ass
from .validate import ensure_valid

log = logging.getLogger('studio.timeline')

RENDERER_VERSION = 1
# intermediate scene clips: visually lossless enough that the final encode dominates the quality
INTERMEDIATE = {'h264_nvenc': ['-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr', '-cq', '16', '-b:v', '0',
                               '-pix_fmt', 'yuv420p'],
                'libx264': ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', '-pix_fmt', 'yuv420p']}
MAX_GRAPH = 24000


@dataclass
class RenderSettings:
    quality: str = 'balanced'           # exports.QUALITY key (final encode)
    encoder: str = 'auto'               # 'auto' (NVENC if it works) | 'cpu'
    cache_dir: object = None            # default MEDIA_ROOT/render_cache
    base_dir: object = None             # root of file:/gen: assets (default MEDIA_ROOT)
    roots: tuple = ()                   # extra folders absolute file: paths may point into
    workdir: object = None              # temp folder (default: a new temporary one)
    use_cache: bool = True
    audio_bitrate: str = '160k'
    loudness_measure: bool = True       # two-pass loudnorm for items with ext clipforge.loudness
    extra: dict = field(default_factory=dict)

    def cache_root(self):
        return Path(self.cache_dir or Path(dj.MEDIA_ROOT) / 'render_cache')


def _fail_unsupported(issues):
    lines = '; '.join(f'{i["path"]}: {i["message"]}' for i in issues[:6])
    raise C.UnsupportedFeature(f'the timeline renderer does not support: {lines}', issues)


def preflight(tl, settings):
    """Validate, check support, resolve and probe every asset an item uses. -> (paths, media)."""
    ensure_valid(tl)
    issues = C.check_supported(tl)
    if issues:
        _fail_unsupported(issues)
    res = M.Resolver(settings.base_dir, settings.roots)
    used = {}
    for ti, t in enumerate(tl.tracks):
        for ii, it in enumerate(t.items):
            if getattr(it, 'asset', None):
                used.setdefault(it.asset, []).append((f'tracks[{ti}].items[{ii}]', t, it))
    paths, media, problems = {}, {}, []
    for aid, refs in used.items():
        a = tl.asset(aid)
        try:
            paths[aid] = res.resolve(a.uri, f'assets[{tl.assets.index(a)}].uri')
            info = M.probe(paths[aid]) if a.type != 'image' else {'has_audio': False, 'has_video': True,
                                                                   'duration': None}
        except TimelineError as exc:
            problems += exc.issues or [{'path': aid, 'message': str(exc)}]
            continue
        media[aid] = dict(info, width=info.get('width') or a.width, height=info.get('height') or a.height)
        for path, t, it in refs:
            if it.type == 'video':
                if not info['has_video']:
                    problems.append({'path': f'{path}.asset', 'message': f'{a.uri} has no video stream'})
                elif not media[aid]['width']:
                    problems.append({'path': f'{path}.asset', 'message': f'{a.uri}: unknown frame size'})
                need = it.src_in + it.duration * it.speed
                if info['duration'] is not None and need > info['duration'] + 0.1:
                    problems.append({'path': f'{path}.duration', 'message':
                                     f'reads up to {need:.2f}s but {a.uri} is {info["duration"]:.2f}s long'})
            if it.type == 'audio':
                if not info['has_audio']:
                    problems.append({'path': f'{path}.asset', 'message': f'{a.uri} has no audio stream'})
                elif info['duration'] is not None and it.src_in + it.duration > info['duration'] + 0.1:
                    problems.append({'path': f'{path}.duration', 'message':
                                     f'reads up to {it.src_in + it.duration:.2f}s but {a.uri} is '
                                     f'{info["duration"]:.2f}s long'})
    for p in problems:
        p.setdefault('level', 'error')
    if problems:
        raise M.AssetError('; '.join(f'{p["path"]}: {p["message"]}' for p in problems[:6]), problems)
    return paths, media


def _scene_key(tl, win, paths):
    fp = scene_fingerprint(tl, win.id) if win.id else content_hash(tl)
    items = [it for _, it in tl.items(scene=win.id)] if win.id else [it for _, it in tl.items()]
    ids_ = {it.id for it in items}
    trans = [to_dict(it) for _, it in tl.items() if it.type == 'transition' and (it.from_item in ids_ or it.to_item in ids_)]
    files = {it.asset: M.signature(paths[it.asset]) for it in items if getattr(it, 'asset', None) in paths}
    payload = {'v': RENDERER_VERSION, 'fp': fp, 'window': [win.start, win.duration], 'transitions': trans,
               'files': files, 'canvas': to_dict(tl.canvas), 'intermediate': INTERMEDIATE['libx264'][4:6]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def _run(cmd_for, encoder, duration, on_progress, cancel, cwd, report):
    try:
        run_ffmpeg(cmd_for(encoder), duration=duration, on_progress=on_progress, cancel_check=cancel, timeout=3600,
                   cwd=cwd)
        return encoder
    except RuntimeError as exc:
        if encoder != 'h264_nvenc':
            raise
        log.warning('NVENC failed (%s); retrying on CPU', str(exc)[:200])
        report['encoder_fallback'] = True
        run_ffmpeg(cmd_for('libx264'), duration=duration, on_progress=on_progress, cancel_check=cancel, timeout=3600,
                   cwd=cwd)
        return 'libx264'


def _render_scene(tl, win, paths, media, dest, encoder, workdir, on_progress, cancel, report):
    for tol in (3.0, 8.0, 20.0):
        g = C.scene_graph(tl, win, media, paths, tol)
        if len(g.text()) < MAX_GRAPH:
            break
    tmp = dest.with_name(dest.stem + f'.part{os.getpid()}.mp4')

    def cmd_for(enc):
        return (['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y'] + g.cmd_inputs() +
                ['-filter_complex', g.text(), '-map', '[vout]', '-an', '-r', C.n(tl.canvas.fps), '-t', C.n(win.duration)]
                + INTERMEDIATE[enc] + ['-movflags', '+faststart', str(tmp)])
    used = _run(cmd_for, encoder, win.duration, on_progress, cancel, str(workdir), report)
    os.replace(tmp, dest)
    return used


def render(tl, dest, settings=None, rerender=(), on_progress=None, cancel=None):
    """Render a timeline to `dest` (mp4). -> report dict. Raises TimelineError / UnsupportedFeature / AssetError
    before any ffmpeg run, RuntimeError if ffmpeg fails, JobCancelled when cancelled."""
    settings = settings or RenderSettings()
    t_start = time.time()
    paths, media = preflight(tl, settings)
    encoder, enc_note = diagnostics.choose_encoder(settings.encoder)
    report = {'renderer': 'timeline', 'version': RENDERER_VERSION, 'encoder': encoder, 'encoder_note': enc_note,
              'size': f'{tl.canvas.width}x{tl.canvas.height}', 'fps': tl.canvas.fps, 'duration': tl.duration,
              'timeline_hash': content_hash(tl), 'scenes': []}
    windows = C.scene_windows(tl)
    cache = settings.cache_root() / 'scenes'
    cache.mkdir(parents=True, exist_ok=True)
    own_tmp = None
    workdir = settings.workdir
    if workdir is None:
        own_tmp = tempfile.mkdtemp(prefix='cf_tl_')
        workdir = own_tmp
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rerender = set(rerender or ())
    unknown = rerender - {w.id for w in windows}
    if unknown:
        raise TimelineError(f'unknown scene id(s) to re-render: {sorted(unknown)}')
    todo = []
    clips = {}
    for win in windows:
        key = _scene_key(tl, win, paths)
        clips[win.id] = cache / f'{key}.mp4'
        hit = settings.use_cache and clips[win.id].is_file() and win.id not in rerender
        report['scenes'].append({'id': win.id, 'key': key, 'start': win.start, 'duration': win.duration,
                                 'cached': bool(hit)})
        if not hit:
            todo.append(win)
    total = sum(w.duration for w in todo) + tl.duration * 1.5
    done = [0.0]

    def prog(weight):
        def f(frac):
            if on_progress:
                on_progress(min(1.0, (done[0] + frac * weight) / total))
        return f
    try:
        for win in todo:
            if cancel and cancel():
                raise JobCancelled()
            t0 = time.time()
            used = _render_scene(tl, win, paths, media, clips[win.id], encoder, workdir, prog(win.duration), cancel,
                                 report)
            done[0] += win.duration
            for s in report['scenes']:
                if s['id'] == win.id:
                    s.update(seconds=round(time.time() - t0, 2), encoder=used)
        for win in windows:                                     # cache hits: mark as recently used
            try:
                os.utime(clips[win.id])
            except OSError:
                pass
        _final(tl, dest, windows, clips, paths, media, encoder, workdir, settings, prog(tl.duration * 1.5), cancel,
               report)
    finally:
        if own_tmp:
            shutil.rmtree(own_tmp, ignore_errors=True)
    report['seconds'] = round(time.time() - t_start, 2)
    report['rendered_scenes'] = sum(1 for s in report['scenes'] if not s['cached'])
    prune_cache(settings.cache_root())
    return report


def _final(tl, dest, windows, clips, paths, media, encoder, workdir, settings, on_progress, cancel, report):
    from ..editor import _copy_fonts, measure_loudness
    g = C.Graph()
    clip_inputs = {}
    for win in windows:
        clip_inputs[win.id] = g.add_input('-i', M.ffmpeg_input(clips[win.id]))
    margins = []
    for win in windows:
        margins += C.scene_margins(tl, win, media)
    ass = workdir / 'timeline.ass'
    cap_report = write_ass(tl, ass, margins)
    if cap_report:
        _copy_fonts(workdir)
    C.final_video(tl, g, windows, clip_inputs, media, paths, 'timeline.ass' if cap_report else None)
    sources = C.audio_sources(tl, paths, media)
    items = {it.id: it for _, it in tl.items()}
    loud = []
    for s in sources:
        target = ((getattr(items[s.item_id], 'ext', None) or {}).get('clipforge.loudness') or {}).get('target')
        if target is None:
            continue
        spec = next((v for v in exports.LOUDNESS.values() if v['I'] == float(target)),
                    {'I': float(target), 'TP': -1.5, 'LRA': 11.0})
        base = f"loudnorm=I={spec['I']:g}:TP={spec['TP']:g}:LRA={spec['LRA']:g}"
        meas = measure_loudness(s.path, s.src_in, s.duration, cancel=cancel) if settings.loudness_measure else None
        if meas:
            s.loudnorm = (f"{base}:measured_I={meas['input_i']:.2f}:measured_TP={meas['input_tp']:.2f}:"
                          f"measured_LRA={meas['input_lra']:.2f}:measured_thresh={meas['input_thresh']:.2f}:"
                          f"offset={meas['target_offset']:.2f}:linear=true")
            loud.append({'item': s.item_id, 'target': spec['I'], 'mode': 'two-pass',
                         'measured_i': round(meas['input_i'], 1), 'measured_tp': round(meas['input_tp'], 1)})
        else:
            s.loudnorm = base
            loud.append({'item': s.item_id, 'target': spec['I'], 'mode': 'single-pass'})
    C.audio_graph(g, sources, tl.duration)
    report['audio'] = {'sources': len(sources), 'loudness': loud}
    report['captions'] = cap_report
    report['graph_chars'] = len(g.text())
    tmp = Path(dest).with_name(Path(dest).stem + '.part.mp4')

    def cmd_for(enc):
        return (['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y'] + g.cmd_inputs() +
                ['-filter_complex', g.text(), '-map', '[vout]', '-map', '[aout]', '-r', C.n(tl.canvas.fps),
                 '-t', C.n(tl.duration)] + exports.encoder_args(enc, {'quality': settings.quality}) +
                ['-c:a', 'aac', '-b:a', settings.audio_bitrate, '-ar', '48000', '-movflags', '+faststart', str(tmp)])
    report['encoder'] = _run(cmd_for, encoder, tl.duration, on_progress, cancel, str(workdir), report)
    if report.get('encoder_fallback'):
        report['encoder_note'] = 'CPU (откат: NVENC не сработал)'
    os.replace(tmp, dest)


def render_scene(tl, scene_id, settings=None, force=False, cancel=None):
    """Render (or take from the cache) the picture of one scene. -> Path of the cached scene clip (no sound)."""
    settings = settings or RenderSettings()
    paths, media = preflight(tl, settings)
    win = next((w for w in C.scene_windows(tl) if w.id == scene_id), None)
    if win is None:
        raise TimelineError(f'unknown scene {scene_id!r}')
    cache = settings.cache_root() / 'scenes'
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f'{_scene_key(tl, win, paths)}.mp4'
    if force or not dest.is_file():
        encoder, _ = diagnostics.choose_encoder(settings.encoder)
        with tempfile.TemporaryDirectory(prefix='cf_tl_') as td:
            _render_scene(tl, win, paths, media, dest, encoder, Path(td), None, cancel, {})
    return dest


def prune_cache(root, max_bytes=None):
    """Keep the scene cache under CLIPFORGE_TL_CACHE_GB (default 5 GB): least recently used clips go first.
    Only files this renderer wrote (<32 hex>.mp4 in render_cache/scenes) are ever removed."""
    if max_bytes is None:
        max_bytes = int(float(os.getenv('CLIPFORGE_TL_CACHE_GB', '5')) * 1024 ** 3)
    d = Path(root) / 'scenes'
    if not d.is_dir():
        return 0
    files = [f for f in d.glob('*.mp4') if len(f.stem) == 32 and all(c in '0123456789abcdef' for c in f.stem)]
    files.sort(key=lambda f: f.stat().st_mtime)
    size = sum(f.stat().st_size for f in files)
    removed = 0
    while files and size > max_bytes:
        f = files.pop(0)
        size -= f.stat().st_size
        f.unlink(missing_ok=True)
        removed += 1
    return removed
