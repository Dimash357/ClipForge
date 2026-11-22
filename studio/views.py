import json
import mimetypes
import os
import re
from pathlib import Path

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files import File
from django.db.models import Q
from django.http import FileResponse, Http404, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from . import captions, diagnostics, exports, jobs
from .models import Clip, Project
from .pipeline import process_project  # noqa: F401  (kept importable for v4 compatibility)
from .source_library import find_source
from .youtube import get_metadata, parse_youtube_url
from .youtube_download import retrieve_and_process  # noqa: F401

MAX_UPLOAD = settings.MAX_SOURCE_BYTES
CORNERS = ('auto', 'top_left', 'top_right', 'bottom_left', 'bottom_right')
LENGTHS = {'auto': (12, 60), 'short': (15, 32), 'standard': (20, 55), 'long': (45, 90)}
OPTION_FIELDS = ('preset', 'quality', 'fps', 'encoder', 'subtitle_style', 'caption_words', 'caption_animate', 'bg_fill',
                 'director_fallback', 'transition', 'smart_tracking', 'hide_cam', 'cam_position', 'highlight_engine',
                 'review_first', 'captions', 'caption_highlight', 'text_guard', 'cam_fit', 'cam_fill',
                 'highlight_audio', 'highlight_motion', 'loudness', 'streamer')
CHECKBOXES = ('captions', 'caption_animate', 'smart_tracking', 'hide_cam', 'review_first', 'text_guard', 'highlight_audio',
              'highlight_motion')


def _ctx_index(**extra):
    base = {'projects': Project.objects.order_by('-created_at')[:8], 'edit_modes': exports.EDIT_MODES,
            'styles': captions.STYLE_CHOICES, 'presets': exports.PRESETS, 'qualities': exports.QUALITY,
            'defaults': exports.DEFAULTS, 'max_source_gb': round(settings.MAX_SOURCE_BYTES / 1024 ** 3)}
    base.update(extra)
    return base


def _options_from_post(post):
    raw = {}
    for key in OPTION_FIELDS:
        if key in CHECKBOXES:
            if f'present_{key}' in post:            # checkbox groups send a hidden marker so "unchecked" is explicit
                raw[key] = key in post
        elif key in post:
            raw[key] = post.get(key)
    return exports.resolve(raw)


def _err(request, msg):
    return render(request, 'studio/index.html', _ctx_index(error=msg))


def index(request):
    if request.method == 'POST':
        jobs.recover_once()
        f = request.FILES.get('source')
        url = request.POST.get('youtube_url', '').strip()
        local_source = None
        if not f and url:
            try:
                local_source = find_source(url)
            except (ValidationError, ValueError) as e:
                return _err(request, str(e))
        auto_fetch = bool(url and not f and not local_source and request.POST.get('auto_fetch') == 'on')
        if not f and not local_source and not auto_fetch:
            return _err(request, 'Исходник не найден. Включите автоматическое получение разрешённого видео по ссылке либо загрузите файл.')
        if f and (f.size > MAX_UPLOAD or not f.name.lower().endswith(('.mp4', '.mov', '.mkv', '.webm'))):
            return _err(request, f'Допустимы MP4, MOV, MKV, WEBM до {MAX_UPLOAD / 1024 ** 3:.0f} ГБ. Большие файлы можно '
                                 'положить в папку source_library (имя файла = ID YouTube-видео) — без загрузки.')
        if not request.POST.get('rights'):
            return _err(request, 'Подтвердите право на обработку файла.')
        if url:
            try:
                parse_youtube_url(url)
            except ValidationError as e:
                return _err(request, str(e))
        lang = request.POST.get('language', 'auto')
        if lang not in ('auto', 'en', 'ru'):
            lang = 'auto'
        try:
            count = max(1, min(8, int(request.POST.get('clip_count', '3'))))
        except ValueError:
            count = 3
        mode = request.POST.get('edit_mode', 'smart')
        if mode not in exports.MODE_KEYS:
            mode = 'smart'
        corner = request.POST.get('facecam_corner', 'auto')
        if corner not in CORNERS:
            corner = 'auto'
        length_key = request.POST.get('clip_length', 'standard')
        cmin, cmax = LENGTHS.get(length_key, LENGTHS['standard'])
        opts = _options_from_post(request.POST)
        opts['auto_length'] = length_key == 'auto'
        title = (f.name if f else local_source.name if local_source else 'YouTube ' + parse_youtube_url(url))[:250]
        obj = Project(edit_mode=mode, facecam_corner=corner, title=title, youtube_url=url, language=lang,
                      clip_count=count, clip_min=cmin, clip_max=cmax, options=opts)
        if f:
            obj.source = f
            obj.save()
        elif local_source:
            with local_source.open('rb') as handle:
                obj.source.save(local_source.name, File(handle), save=False)
            obj.save()
        else:
            obj.save()
        jobs.enqueue('full', obj.pk)
        return redirect('project', pk=obj.pk)
    return render(request, 'studio/index.html', _ctx_index())


@ensure_csrf_cookie            # the page's buttons POST with the CSRF cookie; it must exist even when opened directly
@require_GET
def project(request, pk):
    p = get_object_or_404(Project, pk=pk)
    jobs.reap(pk)
    return render(request, 'studio/project.html', {
        'project': p, 'edit_modes': exports.EDIT_MODES, 'styles': captions.STYLE_CHOICES,
        'presets': exports.PRESETS, 'qualities': exports.QUALITY, 'stages': jobs.STAGES,
        'opts': exports.resolve(p.options)})


def _file_url(f):
    return f.url if f and f.name else None


def _clip_json(c, p):
    info = c.render_info or {}
    q = c.quality or {}
    return {
        'id': c.id, 'title': c.title, 'start': c.start, 'end': c.end, 'duration': round(c.end - c.start, 1),
        'score': round(c.score, 1), 'url': _file_url(c.video), 'poster': _file_url(c.poster), 'status': c.status,
        'error': c.error, 'reason': c.reason, 'scores': c.scores or {}, 'text': c.text[:600],
        'quality': {'score': q.get('score'), 'ok': q.get('ok'), 'checks': q.get('checks', [])} if q else None,
        'meta': c.meta or {}, 'mode': info.get('mode') or p.edit_mode, 'segments': info.get('segments', []), 'notes': info.get('notes', []),
        'encoder': info.get('encoder_note') or info.get('encoder'), 'fallback_renderer': info.get('fallback_renderer'),
        'opts': exports.resolve(p.options, info.get('opts')), 'position': c.position,
        'download': c.video.url + '?download=1' if c.video else None,
        'auto': info.get('auto'), 'short': (c.meta or {}).get('short'),
    }


def _stage_states(p):
    keys = [k for k, _, _ in jobs.STAGES]
    cur = p.stage if p.stage in keys else None
    out = []
    for i, (k, label, _) in enumerate(jobs.STAGES):
        if p.status in ('done', 'review'):
            state = 'done'
        elif cur is None:
            state = 'pending'
        elif i < keys.index(cur):
            state = 'done'
        elif k == cur:
            state = 'active' if p.status in jobs.ACTIVE_STATUSES else 'failed'
        else:
            state = 'pending'
        out.append({'key': k, 'label': label, 'state': state})
    return out


@require_GET
def status(request, pk):
    p = get_object_or_404(Project, pk=pk)
    jobs.reap(pk)
    p.refresh_from_db()
    clips = [_clip_json(c, p) for c in p.clips.order_by('position', '-score')]
    return JsonResponse({
        'status': p.status, 'progress': p.progress, 'message': p.message, 'error': p.error, 'stage': p.stage,
        'stage_progress': p.stage_progress, 'stages': _stage_states(p), 'active': jobs.is_active(pk),
        'cancel_requested': p.cancel_requested, 'mode': p.edit_mode, 'source_url': _file_url(p.source),
        'source_duration': p.source_duration, 'transcript_ready': bool(p.transcript),
        'transcript_meta': p.transcript_meta, 'analysis': _analysis_summary(p), 'clips': clips})


def _analysis_summary(p):
    a = p.analysis or {}
    if not a:
        return None
    cam = a.get('cam')
    return {'probe': a.get('probe'), 'language': a.get('language'), 'cam': cam.get('corner') if cam else None,
            'cam_checked': bool(a.get('cam_checked')), 'audio_wav': a.get('audio_wav')}


def _json_body(request):
    try:
        data = json.loads(request.body or b'{}')
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def _busy(pk):
    return JsonResponse({'error': 'Проект уже обрабатывается. Дождитесь завершения или остановите задачу.'}, status=409)


@require_POST
def cancel(request, pk):
    get_object_or_404(Project, pk=pk)
    ok = jobs.request_cancel(pk)
    return JsonResponse({'ok': ok})


@require_POST
def resume(request, pk):
    """Re-run a stopped/failed/interrupted project reusing the downloaded source and transcript."""
    p = get_object_or_404(Project, pk=pk)
    if p.status not in ('error', 'cancelled', 'interrupted', 'review', 'done'):
        return _busy(pk)
    data = _json_body(request)
    changed = False
    if data.get('edit_mode') in exports.MODE_KEYS:
        p.edit_mode = data['edit_mode']
        changed = True
    if data.get('facecam_corner') in CORNERS:
        p.facecam_corner = data['facecam_corner']
        changed = True
    if isinstance(data.get('options'), dict):
        p.options = exports.resolve(p.options, data['options'])
        changed = True
    if data.get('clip_count'):
        try:
            p.clip_count = max(1, min(8, int(data['clip_count'])))
            changed = True
        except (TypeError, ValueError):
            pass
    if changed:
        p.save()
    if not jobs.enqueue('reprocess' if data.get('reprocess') else 'resume', pk, skip_review=bool(data.get('skip_review'))):
        return _busy(pk)
    return JsonResponse({'ok': True})


def _clean_bounds(v):
    try:
        s, e = float(v[0]), float(v[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (0 <= s < e) or e - s < 3 or e - s > 180:
        return None
    return [s, e]


@require_POST
def render_clips(request, pk):
    """Render selected clips (review flow) or re-render with new settings / bounds."""
    p = get_object_or_404(Project, pk=pk)
    data = _json_body(request)
    items = data.get('clips') or []
    ids, bounds = [], {}
    valid = set(p.clips.values_list('id', flat=True))
    for it in items:
        try:
            cid = int(it.get('id'))
        except (TypeError, ValueError, AttributeError):
            continue
        if cid not in valid:
            continue
        ids.append(cid)
        b = _clean_bounds([it.get('start'), it.get('end')]) if 'start' in it else None
        if b:
            bounds[str(cid)] = b
    if not ids:
        return JsonResponse({'error': 'Не выбрано ни одного клипа.'}, status=400)
    payload = {'clip_ids': ids, 'bounds': bounds, 'discard_others': bool(data.get('discard_others')),
               'apply_to_project': bool(data.get('apply_to_project')), 'snap': bool(data.get('snap'))}
    if data.get('mode') in exports.MODE_KEYS:
        payload['mode'] = data['mode']
    if data.get('corner') in CORNERS:
        payload['corner'] = data['corner']
    if isinstance(data.get('overrides'), dict):
        payload['overrides'] = {k: v for k, v in data['overrides'].items() if k in exports.DEFAULTS}
    if jobs.is_active(pk):
        return _busy(pk)
    if not jobs.enqueue('render', pk, **payload):
        return _busy(pk)
    return JsonResponse({'ok': True})


@require_POST
def rerender_clip(request, pk):
    clip = get_object_or_404(Clip, pk=pk)
    data = _json_body(request)
    body = {'clips': [{'id': clip.pk, **({'start': data['start'], 'end': data['end']} if 'start' in data and 'end' in data else {})}],
            'mode': data.get('mode'), 'corner': data.get('corner'), 'overrides': data.get('overrides'),
            'snap': bool(data.get('snap'))}
    request._body = json.dumps(body).encode()
    return render_clips(request, clip.project_id)


@require_POST
def clip_short_meta(request, pk):
    """Generate (or regenerate) the Shorts title + description of one rendered clip. Stored in Clip.meta."""
    from . import shorts_meta
    clip = get_object_or_404(Clip, pk=pk)
    if jobs.is_active(clip.project_id):
        return _busy(clip.project_id)
    if clip.status != 'ready':
        return JsonResponse({'error': 'Клип ещё не готов.'}, status=400)
    data = _json_body(request)
    short = shorts_meta.ensure_for_clip(clip.project, clip, force=bool(data.get('regenerate', True)))
    if not short:
        return JsonResponse({'error': 'Не удалось сгенерировать название — подробности в logs/clipforge.log.'}, status=500)
    clip.title = short['title'][:240]
    clip.save(update_fields=['meta', 'title', 'updated_at'])
    return JsonResponse({'ok': True, 'short': short, 'title': clip.title})


@require_GET
def history(request):
    q = request.GET.get('q', '').strip()
    st = request.GET.get('status', '')
    mode = request.GET.get('mode', '')
    order = request.GET.get('order', 'new')
    qs = Project.objects.all()
    if q:
        qs = qs.filter(Q(title__icontains=q) | Q(youtube_url__icontains=q))
    if st:
        qs = qs.filter(status=st)
    if mode in exports.MODE_KEYS:
        qs = qs.filter(edit_mode=mode)
    qs = qs.order_by('created_at' if order == 'old' else '-created_at')[:200]
    items = []
    for p in qs:
        clips = [c for c in p.clips.order_by('position') if c.video]
        items.append({'p': p, 'clips': clips})
    return render(request, 'studio/history.html', {
        'items': items, 'q': q, 'status': st, 'mode': mode, 'order': order, 'edit_modes': exports.EDIT_MODES,
        'statuses': ['done', 'review', 'processing', 'queued', 'fetching', 'error', 'cancelled', 'interrupted']})


@ensure_csrf_cookie
@require_GET
def system(request):
    return render(request, 'studio/system.html', {'report': diagnostics.full_report()})


@require_POST
def system_recheck(request):
    diagnostics.reset_cache()
    return JsonResponse(diagnostics.full_report())


@require_POST
def metadata(request):
    try:
        data = json.loads(request.body)
        return JsonResponse(get_metadata(data.get('url', '')))
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@require_POST
def source_check(request):
    try:
        data = json.loads(request.body)
        url = data.get('url', '')
        path = find_source(url)
        return JsonResponse({'available': bool(path), 'filename': path.name if path else None, 'video_id': parse_youtube_url(url)})
    except (ValidationError, ValueError, TypeError) as e:
        return JsonResponse({'error': str(e)}, status=400)


# ----------------------------------------------------------------------------- media with HTTP Range
_RANGE = re.compile(r'bytes=(\d*)-(\d*)$')


def _iter_range(path, start, length, chunk=1 << 20):
    with open(path, 'rb') as fh:
        fh.seek(start)
        left = length
        while left > 0:
            data = fh.read(min(chunk, left))
            if not data:
                break
            left -= len(data)
            yield data


@require_http_methods(['GET', 'HEAD'])
def serve_media(request, path):
    """Serve MEDIA_ROOT files with Range support (needed for seeking in <video>). Nothing outside MEDIA_ROOT."""
    root = Path(settings.MEDIA_ROOT).resolve()
    full = (root / path).resolve()
    if root not in full.parents or not full.is_file():
        raise Http404()
    size = full.stat().st_size
    ctype = mimetypes.guess_type(str(full))[0] or 'application/octet-stream'
    rng = request.headers.get('Range')
    if rng:
        m = _RANGE.match(rng.strip())
        if not m or (m.group(1) == '' and m.group(2) == ''):
            return HttpResponse(status=416, headers={'Content-Range': f'bytes */{size}'})
        if m.group(1) == '':
            n = int(m.group(2))
            start, end = max(0, size - n), size - 1
        else:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start >= size or start > end:
            return HttpResponse(status=416, headers={'Content-Range': f'bytes */{size}'})
        length = end - start + 1
        resp = StreamingHttpResponse(_iter_range(full, start, length), status=206, content_type=ctype)
        resp['Content-Range'] = f'bytes {start}-{end}/{size}'
        resp['Content-Length'] = str(length)
    else:
        resp = FileResponse(open(full, 'rb'), content_type=ctype)
        resp['Content-Length'] = str(size)
    resp['Accept-Ranges'] = 'bytes'
    resp['Cache-Control'] = 'private, max-age=3600'
    if request.GET.get('download'):
        resp['Content-Disposition'] = f'attachment; filename="{full.name}"'
    return resp
