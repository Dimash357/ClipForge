"""Clip Selection Studio (2D): one place to review, choose, edit and export Shorts.

Builds on what exists - nothing here re-implements analysis, editing or rendering:
* state()/apply(): one clip state = bounds + layout mode + caption/export overrides + Hook variant + Smart Editing
  mode/manual cuts. Applying stores the previous state in a short history (undo) and the very first state as the
  original (restore), then queues ONE render job for all changed clips (job queue + GPU scheduler as usual).
* preview(): a fast draft of the final vertical edit (360 px wide, 24 fps, small quality, single-pass loudness)
  through the same Timeline IR path - layout, captions, Hook teaser and Smart Editing included - cached by its
  settings, so the full render is never needed just to look.
* export_zip(): the chosen rendered clips + their titles, descriptions and hashtags (CSV for spreadsheets and one
  text file per clip, ready to paste into YouTube / TikTok).
"""
import csv
import hashlib
import io
import json
import logging
import re
import time
import zipfile
from pathlib import Path

from django.conf import settings

from . import exports
from .models import Clip

log = logging.getLogger('studio.clipstudio')

HISTORY = 10
PREVIEW_W = 360
PREVIEW_FPS = 24
OVERRIDE_KEYS = ('subtitle_style', 'caption_words', 'caption_highlight', 'captions', 'caption_font', 'caption_censor',
                 'caption_animate', 'preset', 'quality', 'bg_fill', 'transition', 'cam_position', 'cam_fit', 'cam_fill',
                 'hide_cam', 'gameplay_zoom')


# ------------------------------------------------------------------ clip state
def state(c):
    meta = c.meta or {}
    info = c.render_info or {}
    h = meta.get('hooks') or {}
    e = meta.get('edit') or {}
    return {'start': round(c.start, 3), 'end': round(c.end, 3), 'mode': info.get('mode'),
            'overrides': dict(info.get('opts') or {}), 'hook': h.get('selected'),
            'edit': {'mode': e.get('mode'), 'restore': list(e.get('restore') or [])}}


def original_state(c):
    """The clip as the pipeline first chose it (kept in meta['original'] once anything is changed)."""
    o = (c.meta or {}).get('original')
    if o:
        return o
    ab = (c.scores or {}).get('auto_bounds')
    s = state(c)
    if ab:
        s['start'], s['end'] = round(float(ab[0]), 3), round(float(ab[1]), 3)
    s.update(mode=None, overrides={}, hook=None, edit={'mode': None, 'restore': []})
    return s


def clean(draft, c):
    """Validated changes over the current state (unknown keys / values are dropped)."""
    from . import smartedit
    cur = state(c)
    out = dict(cur)
    try:
        s, e = float(draft.get('start', cur['start'])), float(draft.get('end', cur['end']))
        if 0 <= s < e and 3 <= e - s <= 180:
            out['start'], out['end'] = round(s, 3), round(e, 3)
    except (TypeError, ValueError):
        pass
    if draft.get('mode') in exports.MODE_KEYS:
        out['mode'] = draft['mode']
    if isinstance(draft.get('overrides'), dict):
        ov = {k: v for k, v in draft['overrides'].items() if k in OVERRIDE_KEYS}
        out['overrides'] = {k: v for k, v in exports.resolve(cur['overrides'], ov).items()
                            if k in OVERRIDE_KEYS and (k in ov or k in cur['overrides'])}
    if 'hook' in draft:
        h = (c.meta or {}).get('hooks') or {}
        vid = draft['hook']
        out['hook'] = None if vid in (None, '', 'orig') else vid if any(v['id'] == vid for v in h.get('variants', [])) \
            else cur['hook']
    if isinstance(draft.get('edit'), dict):
        m = draft['edit'].get('mode')
        if m in ('off', None) or m in smartedit.MODES:
            out['edit'] = {'mode': m, 'restore': [str(x) for x in (draft['edit'].get('restore') or [])][:200]}
    return out


def _put(c, st):
    """Write a state onto the clip record (the render job reads it from there)."""
    from . import hooks
    meta = dict(c.meta or {})
    c.start, c.end = st['start'], st['end']
    info = dict(c.render_info or {})
    info['opts'] = dict(st['overrides'] or {})
    if st.get('mode'):
        info['mode'] = st['mode']
    else:
        info.pop('mode', None)
    c.render_info = info
    h = meta.get('hooks')
    if h is not None:
        v = hooks.variant(h, st.get('hook'))
        h['selected'] = None if not v or v['kind'] == 'original' else v['id']
        meta['hooks'] = h
    meta['edit'] = {'mode': (st.get('edit') or {}).get('mode'), 'restore': list((st.get('edit') or {}).get('restore') or [])}
    c.meta = meta


def apply(p, items, segments=None, record=True):
    """items: [(clip, new_state)]. Saves history/original, writes the states and queues one render job.
    record=False: an undo (the current state is not pushed onto the history). -> (ok, clip ids)."""
    from . import jobs
    from .pipeline import clip_text
    ids, bounds = [], {}
    for c, st in items:
        meta = dict(c.meta or {})
        meta.setdefault('original', original_state(c))
        hist = list(meta.get('history') or [])
        prev = state(c)
        if record and prev != st:
            hist.append({'at': int(time.time()), 'state': prev})
        meta['history'] = hist[-HISTORY:]
        c.meta = meta
        _put(c, st)
        if segments:
            c.text = clip_text(segments, c.start, c.end)[:4000]
        c.save()
        ids.append(c.pk)
        bounds[str(c.pk)] = [st['start'], st['end']]
    ok = jobs.enqueue('render', p.pk, clip_ids=ids, bounds=bounds, hook=True)
    return ok, ids


def undo_target(c, to):
    """'previous' -> the state before the last applied change; 'original' -> the first state."""
    if to == 'original':
        return original_state(c)
    hist = list((c.meta or {}).get('history') or [])
    return hist[-1]['state'] if hist else None


def pop_history(c):
    meta = dict(c.meta or {})
    hist = list(meta.get('history') or [])
    if hist:
        hist.pop()
    meta['history'] = hist
    c.meta = meta


# ------------------------------------------------------------------ draft preview of the final vertical edit
def _preview_dir(c):
    d = Path(settings.MEDIA_ROOT) / 'previews' / str(c.pk)
    d.mkdir(parents=True, exist_ok=True)
    return d


def preview(p, c, st, segments=None, cancel=None):
    """Draft render of the clip in state `st` -> (relative media path, info). Cached by the state."""
    from . import analysis_cache as ac, editor, hooks, resources, smartedit
    from .pipeline import build_edits, clip_ir
    from .timeline import renderer
    opts = exports.resolve(p.options, st.get('overrides'))
    mode = st.get('mode') or p.edit_mode
    payload = json.dumps({'v': 1, 's': st, 'fp': p.source_fingerprint, 'o': opts, 'm': mode,
                          'h': ((c.meta or {}).get('hooks') or {}).get('version')}, sort_keys=True, default=str)
    key = hashlib.sha1(payload.encode()).hexdigest()[:16]
    out = _preview_dir(c) / f'{key}.mp4'
    rel = out.relative_to(settings.MEDIA_ROOT).as_posix()
    if out.is_file():
        return rel, {'cached': True}
    segments = segments if segments is not None else p.transcript
    source = Path(p.source.path)
    vinfo = editor.probe_video(source)
    a, b = st['start'], st['end']
    cache = ac.cache_dir(p.source_fingerprint) if p.source_fingerprint else None
    plan = editor.prepare(source, {'start': a, 'end': b, 'scores': c.scores or {}}, mode,
                          (c.render_info or {}).get('corner') or p.facecam_corner, opts, vinfo,
                          source_cam=(p.analysis or {}).get('cam'), cache=cache)
    tl, _ = clip_ir(p, c, plan, {'opts': opts, 'mode': mode}, vinfo, segments, start=a, end=b)
    em = (st.get('edit') or {}).get('mode')
    epl = None
    if em and em != 'off':
        epl = smartedit.plan(hooks.load_signals(p, segments), a, b, em, p.clip_min,
                             restore=set((st.get('edit') or {}).get('restore') or []))
    ed = build_edits(segments, a, b, epl, (c.meta or {}).get('hooks'), st.get('hook'))
    if ed:
        tl = ed['transform'](tl)
    scale = PREVIEW_W / tl.canvas.width
    tl.canvas.width = PREVIEW_W
    tl.canvas.height = int(round(tl.canvas.height * scale / 2)) * 2
    tl.canvas.fps = float(min(PREVIEW_FPS, tl.canvas.fps))
    tmp = out.with_suffix('.part.mp4')
    with resources.cpu_lease('Studio preview', 2):
        r = renderer.render(tl, tmp, renderer.RenderSettings(quality='small', loudness_measure=False,
                                                             workdir=_preview_dir(c) / 'work'), cancel=cancel)
    tmp.replace(out)
    for old in sorted(_preview_dir(c).glob('*.mp4'), key=lambda f: f.stat().st_mtime)[:-6]:
        old.unlink(missing_ok=True)                      # keep the last few previews of this clip
    return rel, {'cached': False, 'seconds': r.get('seconds'), 'duration': tl.duration,
                 'edit': (ed or {}).get('info', {}).get('edit'), 'hook': (ed or {}).get('info', {}).get('hook')}


# ------------------------------------------------------------------ export
def _slug(text, n=60):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', ' ', text or '').strip()
    return re.sub(r'\s+', ' ', s)[:n].strip() or 'clip'


def texts(c):
    """(title, description, hashtags) for publishing: the Shorts text if generated, else the clip metadata."""
    meta = c.meta or {}
    short = meta.get('short') or {}
    title = short.get('title') or c.title or (meta.get('titles') or [''])[0]
    desc = short.get('description') or meta.get('description') or ''
    tags = short.get('hashtags') or meta.get('hashtags') or []
    tags = ' '.join(t if str(t).startswith('#') else f'#{t}' for t in tags)
    return title, desc, tags


def export_zip(p, ids):
    """ZIP with the chosen ready clips (NN_title.mp4), one NN_title.txt each and shorts.csv. -> (path, skipped)."""
    clips = [c for c in Clip.objects.filter(project=p, pk__in=ids).order_by('position') if c.video]
    ready = [c for c in clips if c.status == 'ready' and Path(c.video.path).is_file()]
    skipped = sorted(set(ids) - {c.pk for c in ready})
    d = Path(settings.MEDIA_ROOT) / 'exports'
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(f'project{p.pk}_*.zip'):
        old.unlink(missing_ok=True)                      # only the latest export of a project is kept
    path = d / f'project{p.pk}_{int(time.time())}.zip'
    rows = io.StringIO()
    w = csv.writer(rows)
    w.writerow(['file', 'title', 'description', 'hashtags', 'duration_s', 'viral_score', 'hook_score'])
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as z:
        for n, c in enumerate(ready, 1):
            title, desc, tags = texts(c)
            base = f'{n:02d}_{_slug(title)}'
            z.write(c.video.path, f'{base}.mp4')
            z.writestr(f'{base}.txt', f'{title}\n\n{desc}\n\n{tags}\n'.encode('utf-8'),
                       compress_type=zipfile.ZIP_DEFLATED)
            v = (c.meta or {}).get('viral') or {}
            h = (c.meta or {}).get('hooks') or {}
            sel = next((x for x in h.get('variants', []) if x['id'] == (h.get('selected') or 'orig')), None)
            w.writerow([f'{base}.mp4', title, desc, tags, round((c.render_info or {}).get('output_duration') or
                                                                (c.end - c.start), 1),
                        v.get('score', ''), sel['score'] if sel else ''])
        z.writestr('shorts.csv', ('﻿' + rows.getvalue()).encode('utf-8'), compress_type=zipfile.ZIP_DEFLATED)
    return path, skipped
