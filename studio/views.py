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
from .models import Clip, Job, Project, Script
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
                 'highlight_audio', 'highlight_motion', 'loudness', 'streamer', 'highlight_sound_ai', 'caption_font',
                 'caption_censor', 'hook_engine', 'smart_edit', 'viral_director')
CHECKBOXES = ('captions', 'caption_animate', 'smart_tracking', 'hide_cam', 'review_first', 'text_guard', 'highlight_audio',
              'highlight_motion', 'highlight_sound_ai', 'caption_censor')


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
        'output_duration': info.get('output_duration'), 'width': _clip_dims(info),
        'history': len((c.meta or {}).get('history') or []), 'edited': bool((c.meta or {}).get('original')),
        'edit_error': info.get('edit_error'), 'edit_mode': _smart_mode(p, c),
    }


def _smart_mode(p, c):
    from .pipeline import smart_edit_mode
    return smart_edit_mode(p, c)


def _clip_dims(info):
    try:
        w, h = (info.get('size') or '').split('x')
        return [int(w), int(h)]
    except ValueError:
        return None


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
    job = p.jobs.order_by('-created_at').first()
    from . import sysload
    return JsonResponse({
        'job': _live_info(job), 'load': sysload.snapshot() if job and job.status == Job.RUNNING else None,
        'status': p.status, 'progress': p.progress, 'message': p.message, 'error': p.error, 'stage': p.stage,
        'stage_progress': p.stage_progress, 'stages': _stage_states(p), 'active': jobs.is_active(pk),
        'cancel_requested': p.cancel_requested, 'mode': p.edit_mode, 'source_url': _file_url(p.source),
        'source_duration': p.source_duration, 'transcript_ready': bool(p.transcript),
        'transcript_meta': p.transcript_meta, 'analysis': _analysis_summary(p), 'clips': clips})


def _live_info(job):
    """Live operations of a job with elapsed / remaining time and how long ago it last showed signs of life."""
    if job is None:
        return None
    from django.utils import timezone
    act = dict(job.activity or {})
    beat = (timezone.now() - job.heartbeat_at).total_seconds() if job.heartbeat_at else None
    if job.status == Job.RUNNING and act.get('at') and act.get('ops'):
        lag = max(0.0, timezone.now().timestamp() - act['at'])     # snapshot age: advance the clocks for the UI
        for o in act['ops']:
            o['elapsed'] = round((o.get('elapsed') or 0) + lag, 1)
            if o.get('eta') is not None:
                o['eta'] = round(max(0.0, o['eta'] - lag), 1)
    return {'id': job.pk, 'status': job.status, 'ops': act.get('ops') or [], 'timings': act.get('timings') or {},
            'stalled_for': act.get('stalled_for'), 'heartbeat_age': None if beat is None else round(beat, 1),
            'alive': job.status == Job.RUNNING and beat is not None and beat < 3 * jobs.HEARTBEAT_SECONDS + 5,
            'run_seconds': round(job.run_seconds, 1)}


def system_load(request):
    from . import sysload
    return JsonResponse(sysload.snapshot())


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
def pause(request, pk):
    get_object_or_404(Project, pk=pk)
    job = jobs.live_job(pk)
    return JsonResponse({'ok': bool(job and jobs.pause_job(job.pk))})


@require_POST
def resume(request, pk):
    """Re-run a stopped/failed/interrupted project reusing the downloaded source and transcript."""
    p = get_object_or_404(Project, pk=pk)
    if p.status not in ('error', 'cancelled', 'interrupted', 'review', 'done', 'paused'):
        return _busy(pk)
    data = _json_body(request)
    if not data and p.status in ('paused', 'interrupted', 'error', 'cancelled'):
        # continue the project's last unfinished job from its checkpoints (finished steps are skipped)
        last = p.jobs.order_by('-created_at').first()
        if last and last.status in (Job.PAUSED, Job.INTERRUPTED, Job.FAILED, Job.CANCELLED) and last.checkpoints \
                and jobs.resume_job(last.pk):
            return JsonResponse({'ok': True, 'job': last.pk})
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


# ---------------------------------------------------------------- job queue page
def _job_json(j, now):
    elapsed = j.run_seconds + ((now - j.started_at).total_seconds() if j.status == Job.RUNNING and j.started_at else 0)
    try:
        label = jobs.handler(j.kind).label
    except ValueError:
        label = j.kind
    return {'id': j.pk, 'kind': j.kind, 'label': label, 'title': j.title, 'project': j.project_id,
            'status': j.status, 'status_label': dict(Job.STATUSES).get(j.status, j.status), 'priority': j.priority,
            'stage': j.stage, 'stage_label': dict((k, l) for k, l, _ in jobs.STAGES).get(j.stage, j.stage),
            'stage_progress': j.stage_progress, 'progress': j.progress, 'message': j.message, 'error': j.error[-600:],
            'attempts': j.attempts, 'max_attempts': j.max_attempts, 'checkpoints': sorted(j.checkpoints or {}),
            'pause_requested': j.pause_requested, 'cancel_requested': j.cancel_requested, 'live': _live_info(j),
            'created_at': j.created_at.isoformat(), 'finished_at': j.finished_at and j.finished_at.isoformat(),
            'elapsed': round(elapsed, 1)}


@ensure_csrf_cookie
def queue(request):
    jobs.recover_once()
    return render(request, 'studio/queue.html', {'max_jobs': jobs.max_heavy()})


@require_GET
def jobs_list(request):
    from django.utils import timezone
    now = timezone.now()
    live = list(Job.objects.filter(status__in=Job.LIVE).order_by('-priority', 'created_at'))
    done = list(Job.objects.exclude(status__in=Job.LIVE).order_by('-updated_at')[:50])
    from . import sysload
    return JsonResponse({'live': [_job_json(j, now) for j in live], 'recent': [_job_json(j, now) for j in done],
                         'max_jobs': jobs.max_heavy(),
                         'load': sysload.snapshot() if any(j.status == Job.RUNNING for j in live) else None})


@require_POST
def job_action(request, pk, action):
    job = get_object_or_404(Job, pk=pk)
    if action == 'pause':
        ok = jobs.pause_job(job.pk)
    elif action == 'resume':
        ok = jobs.resume_job(job.pk)
    elif action == 'cancel':
        ok = jobs.cancel_job(job.pk)
    elif action == 'retry':
        ok = jobs.retry_job(job.pk) is not None
    elif action == 'priority':
        try:
            ok = jobs.set_priority(job.pk, int(_json_body(request).get('priority')))
        except (TypeError, ValueError):
            return JsonResponse({'error': 'priority must be an integer'}, status=400)
    else:
        raise Http404
    if not ok:
        return JsonResponse({'ok': False, 'error': 'Действие сейчас недоступно для этой задачи.'}, status=409)
    return JsonResponse({'ok': True})


# ---------------------------------------------------------------- resource manager (1D)
@require_GET
def resources_state(request):
    from . import resources, sysload
    return JsonResponse(dict(resources.manager().snapshot(), load=sysload.snapshot()))


@require_POST
def resources_mode(request):
    from . import resources
    mgr = resources.manager()
    if mgr.mode_locked():
        return JsonResponse({'error': 'Режим задан переменной CLIPFORGE_RESOURCE_MODE.'}, status=409)
    try:
        mgr.set_mode(_json_body(request).get('mode'))
    except ValueError:
        return JsonResponse({'error': 'Неизвестный режим.'}, status=400)
    jobs.wake()                     # more heavy slots may be free now
    return JsonResponse({'ok': True, 'mode': mgr.mode()})


@require_POST
def resources_unload(request):
    from . import resources
    return JsonResponse({'ok': True, 'unloaded': resources.manager().unload_models(why='из интерфейса')})


# ---------------------------------------------------------------- Hook Engine (2A)
@require_http_methods(['GET', 'POST'])
def clip_hooks(request, pk):
    """GET: the clip's hook variants (computed from the cached analysis if missing, ?refresh=1 to redo).
    POST {variant}: choose one -> the clip is re-rendered in the queue with that opening."""
    from . import hooks
    clip = get_object_or_404(Clip, pk=pk)
    p = clip.project
    h = (clip.meta or {}).get('hooks')
    if request.method == 'GET':
        if not h or request.GET.get('refresh') == '1':
            if jobs.is_active(p.pk):
                return _busy(p.pk)
            if not p.transcript:
                return JsonResponse({'error': 'Нет транскрипта — обработайте проект.'}, status=400)
            a0, b0 = (h or {}).get('clip') or (clip.start, clip.end)
            h = hooks.analyze(hooks.load_signals(p), a0, b0, p.clip_min, p.clip_max, p.source_duration)
            clip.meta = dict(clip.meta or {}, hooks=h)
            clip.save(update_fields=['meta', 'updated_at'])
        return JsonResponse({'hooks': h, 'source_url': _file_url(p.source)})
    if not h:
        return JsonResponse({'error': 'Сначала найдите варианты начала.'}, status=400)
    v = hooks.variant(h, _json_body(request).get('variant'))
    if v is None:
        return JsonResponse({'error': 'Нет такого варианта.'}, status=400)
    if jobs.is_active(p.pk):
        return _busy(p.pk)
    h['selected'] = None if v['kind'] == 'original' else v['id']
    clip.meta = dict(clip.meta or {}, hooks=h)
    clip.save(update_fields=['meta', 'updated_at'])
    if not jobs.enqueue('render', p.pk, clip_ids=[clip.pk], bounds={str(clip.pk): [v['start'], v['end']]}, hook=True):
        return _busy(p.pk)
    return JsonResponse({'ok': True, 'selected': h['selected']})


# ---------------------------------------------------------------- Smart Editing (2B)
@require_http_methods(['GET', 'POST'])
def clip_edit(request, pk):
    """GET ?mode=: the cut plan for the clip (preview, nothing changes). POST {mode, restore: [cut ids]}: apply
    (or mode 'off' = back to the original) -> the clip is re-rendered in the queue."""
    from . import hooks, pipeline as pl, smartedit
    clip = get_object_or_404(Clip, pk=pk)
    p = clip.project
    if not p.transcript:
        return JsonResponse({'error': 'Нет транскрипта — обработайте проект.'}, status=400)
    if request.method == 'GET':
        mode = request.GET.get('mode') or pl.smart_edit_mode(p, clip)
        e = (clip.meta or {}).get('edit') or {}
        if mode == 'off':
            return JsonResponse({'mode': 'off', 'plan': None, 'modes': _edit_modes(), 'source_url': _file_url(p.source)})
        if mode not in smartedit.MODES:
            return JsonResponse({'error': 'Неизвестный режим.'}, status=400)
        rq = request.GET.get('restore')
        restore = set(filter(None, rq.split(','))) if rq is not None else set(e.get('restore') or [])
        plan = smartedit.plan(hooks.load_signals(p), clip.start, clip.end, mode, p.clip_min, restore=restore)
        return JsonResponse({'mode': mode, 'plan': plan, 'modes': _edit_modes(), 'applied': e.get('mode'),
                             'source_url': _file_url(p.source)})
    data = _json_body(request)
    mode = data.get('mode')
    if mode != 'off' and mode not in smartedit.MODES:
        return JsonResponse({'error': 'Неизвестный режим.'}, status=400)
    if jobs.is_active(p.pk):
        return _busy(p.pk)
    restore = [str(x) for x in (data.get('restore') or [])][:200]
    clip.meta = dict(clip.meta or {}, edit={'mode': mode, 'restore': restore})
    clip.save(update_fields=['meta', 'updated_at'])
    if not jobs.enqueue('render', p.pk, clip_ids=[clip.pk], hook=True):
        return _busy(p.pk)
    return JsonResponse({'ok': True, 'mode': mode})


def _edit_modes():
    from . import smartedit
    return {'off': {'label': 'Оригинал', 'note': 'без вырезок'},
            **{k: {'label': v['label'], 'note': v['note']} for k, v in smartedit.MODES.items()}}


# ---------------------------------------------------------------- Viral Director (2C)
class _QuietRep:
    def frac(self, *a, **k):
        pass

    def cancelled(self):
        return False


def _ranking(p):
    out = []
    for c in p.clips.exclude(status='candidate').order_by('position'):
        v = (c.meta or {}).get('viral')
        if v:
            out.append({'id': c.pk, 'title': c.title, 'score': v['score'], 'rank': v.get('rank'),
                        'duplicate_of': v.get('duplicate_of'), 'combo': v.get('combo'), 'gain': v.get('gain'),
                        'applied': v.get('applied')})
    return sorted(out, key=lambda x: (x['rank'] is None, x['rank'] or 0, -x['score']))


@require_http_methods(['GET', 'POST'])
def project_viral(request, pk):
    """GET: the project's ranking. POST: re-score every clip (plans only, nothing is rendered)."""
    from . import pipeline as pl
    from .viral import DISCLAIMER
    p = get_object_or_404(Project, pk=pk)
    if request.method == 'POST':
        if jobs.is_active(p.pk):
            return _busy(p.pk)
        if not p.transcript:
            return JsonResponse({'error': 'Нет транскрипта — обработайте проект.'}, status=400)
        clips = list(p.clips.exclude(status='candidate'))
        opts = dict(exports.resolve(p.options), viral_director='rank')
        pl._viral_director(p, clips, p.transcript, opts, _QuietRep(), apply=False)
    return JsonResponse({'ranking': _ranking(p), 'disclaimer': DISCLAIMER})


@require_POST
def clip_viral_apply(request, pk):
    """Apply the clip's best Hook x Smart Editing combination -> re-render in the queue."""
    from . import hooks
    clip = get_object_or_404(Clip, pk=pk)
    p = clip.project
    v = (clip.meta or {}).get('viral')
    if not v:
        return JsonResponse({'error': 'Сначала оцените клипы (Viral Director).'}, status=400)
    if jobs.is_active(p.pk):
        return _busy(p.pk)
    combo = v['combo']
    meta = dict(clip.meta)
    h = meta.get('hooks')
    var = hooks.variant(h, combo.get('hook')) if h else None
    if h is not None:
        h['selected'] = None if not var or var['kind'] == 'original' else var['id']
        meta['hooks'] = h
    meta['edit'] = {'mode': combo['edit'], 'restore': []}
    meta['viral'] = dict(v, applied=True)
    clip.meta = meta
    clip.save(update_fields=['meta', 'updated_at'])
    a, b = (var['start'], var['end']) if var and var['kind'] == 'start' else tuple((h or {}).get('clip') or
                                                                                   (clip.start, clip.end))
    if not jobs.enqueue('render', p.pk, clip_ids=[clip.pk], bounds={str(clip.pk): [a, b]}, hook=True):
        return _busy(p.pk)
    return JsonResponse({'ok': True, 'combo': combo})


# ---------------------------------------------------------------- Clip Selection Studio (2D)
@ensure_csrf_cookie
def clip_studio(request, pk):
    p = get_object_or_404(Project, pk=pk)
    jobs.reap(pk)
    return render(request, 'studio/clipstudio.html', {
        'project': p, 'edit_modes': exports.EDIT_MODES, 'styles': captions.STYLE_CHOICES,
        'fonts': sorted(exports.CHOICES['caption_font']), 'opts': exports.resolve(p.options)})


@require_POST
def clip_preview(request, pk):
    """Draft of the final vertical edit for the given (unsaved) changes -> {'url', 'info'}."""
    from . import clipstudio
    c = get_object_or_404(Clip, pk=pk)
    p = c.project
    if not p.source or not p.transcript:
        return JsonResponse({'error': 'Нет исходного видео или транскрипта.'}, status=400)
    st = clipstudio.clean(_json_body(request), c)
    try:
        rel, info = clipstudio.preview(p, c, st)
    except Exception as exc:
        import logging
        logging.getLogger('studio.views').exception('preview failed for clip %s', c.pk)
        return JsonResponse({'error': f'Не удалось собрать предпросмотр: {str(exc)[:200]}'}, status=500)
    return JsonResponse({'url': settings.MEDIA_URL + rel, 'info': info, 'state': st})


@require_POST
def project_apply(request, pk):
    """{clips: [{id, start, end, mode, overrides, hook, edit}]} -> one render job for all of them."""
    from . import clipstudio
    p = get_object_or_404(Project, pk=pk)
    if jobs.is_active(p.pk):
        return _busy(p.pk)
    items = []
    for d in (_json_body(request).get('clips') or [])[:50]:
        c = Clip.objects.filter(project=p, pk=d.get('id') if isinstance(d, dict) else None).first()
        if c is not None:
            items.append((c, clipstudio.clean(d, c)))
    if not items:
        return JsonResponse({'error': 'Не выбрано ни одного клипа.'}, status=400)
    ok, ids = clipstudio.apply(p, items, p.transcript)
    if not ok:
        return _busy(p.pk)
    return JsonResponse({'ok': True, 'clips': ids})


@require_POST
def clip_restore(request, pk):
    """{to: 'previous' | 'original'}: undo the last applied change, or go back to the original clip."""
    from . import clipstudio
    c = get_object_or_404(Clip, pk=pk)
    p = c.project
    to = _json_body(request).get('to', 'previous')
    if to not in ('previous', 'original'):
        return JsonResponse({'error': 'to: previous | original'}, status=400)
    if jobs.is_active(p.pk):
        return _busy(p.pk)
    target = clipstudio.undo_target(c, to)
    if target is None:
        return JsonResponse({'error': 'Нечего отменять.'}, status=400)
    if to == 'previous':
        clipstudio.pop_history(c)
    ok, _ = clipstudio.apply(p, [(c, target)], p.transcript, record=(to == 'original'))
    if not ok:
        return _busy(p.pk)
    return JsonResponse({'ok': True, 'state': target})


@require_GET
def project_export(request, pk):
    """ZIP of the chosen ready clips with titles, descriptions and hashtags (?ids=1,2,3; empty = all ready)."""
    from . import clipstudio
    p = get_object_or_404(Project, pk=pk)
    raw = request.GET.get('ids', '')
    ids = [int(x) for x in raw.split(',') if x.strip().isdigit()] or list(
        p.clips.filter(status='ready').values_list('pk', flat=True))
    path, skipped = clipstudio.export_zip(p, ids)
    if not zipfile_has_clips(path):
        path.unlink(missing_ok=True)
        return JsonResponse({'error': 'Среди выбранных нет готовых клипов.', 'skipped': skipped}, status=400)
    resp = FileResponse(open(path, 'rb'), as_attachment=True,
                        filename=f'{clipstudio._slug(p.title, 40)}_shorts.zip')
    resp['X-Skipped-Clips'] = ','.join(map(str, skipped))
    return resp


def zipfile_has_clips(path):
    import zipfile
    with zipfile.ZipFile(path) as z:
        return any(n.endswith('.mp4') for n in z.namelist())


# ---------------------------------------------------------------- AI Script Generator (3A)
def scripts_page(request):
    from . import ai_provider, scripts as sg
    if request.method == 'POST':
        idea = request.POST.get('idea', '').strip()
        if not idea:
            return redirect('scripts')
        sc = Script.objects.create(idea=idea[:2000], title=request.POST.get('title', '')[:250],
                                   **{k: v for k, v in sg.settings_of(request.POST).items()
                                      if k in ('format', 'language', 'duration', 'genre', 'style', 'audience')},
                                   status='generating')
        jobs.submit('script', title=f'Сценарий: {sc.title or idea[:60]}', script_id=sc.pk, action='full')
        return redirect('script_detail', pk=sc.pk)
    return render(request, 'studio/scripts.html', {
        'scripts': Script.objects.order_by('-created_at')[:100], 'formats': sg.FORMATS.items(),
        'langs': sg.LANGS.items(), 'ai': ai_provider.status()})


@ensure_csrf_cookie
def script_detail(request, pk):
    from . import scripts as sg
    sc = get_object_or_404(Script, pk=pk)
    return render(request, 'studio/script.html', {'script': sc, 'parts': sg.PART_RU, 'formats': dict(sg.FORMATS)})


def _script_json(sc):
    job = Job.objects.filter(kind='script', payload__script_id=sc.pk).order_by('-created_at').first()
    return {'id': sc.pk, 'title': sc.title, 'idea': sc.idea, 'format': sc.format, 'language': sc.language,
            'duration': sc.duration, 'genre': sc.genre, 'style': sc.style, 'audience': sc.audience, 'status': sc.status,
            'error': sc.error, 'data': sc.data, 'job': _live_info(job)}


@require_GET
def script_api(request, pk):
    return JsonResponse(_script_json(get_object_or_404(Script, pk=pk)))


def _script_busy():
    return JsonResponse({'error': 'Сценарий уже генерируется — дождитесь окончания.'}, status=409)


@require_POST
def script_generate(request, pk):
    """Regenerate the whole script (optionally with new settings)."""
    from . import scripts as sg
    sc = get_object_or_404(Script, pk=pk)
    if sc.status == 'generating':
        return _script_busy()
    data = _json_body(request)
    st = sg.settings_of(dict(sg.settings_of(sc), **{k: v for k, v in data.items() if k in
                                                   ('idea', 'format', 'language', 'duration', 'genre', 'style',
                                                    'audience', 'title')}))
    for k in ('idea', 'format', 'language', 'duration', 'genre', 'style', 'audience', 'title'):
        setattr(sc, k, st[k])
    sc.status = 'generating'
    sc.save()
    jobs.submit('script', title=f'Сценарий: {sc.title or sc.idea[:60]}', script_id=sc.pk, action='full')
    return JsonResponse({'ok': True})


@require_POST
def script_scene(request, pk, sid, action=None):
    """POST .../scenes/<sid>/ {changes}: manual edit. POST .../scenes/<sid>/regenerate/ {instruction}: new version."""
    from . import scripts as sg
    sc = get_object_or_404(Script, pk=pk)
    if sc.status == 'generating':
        return _script_busy()
    if not any(s['id'] == sid for s in (sc.data or {}).get('scenes', [])):
        raise Http404
    data = _json_body(request)
    if action == 'regenerate':
        sc.status = 'generating'
        sc.save(update_fields=['status', 'updated_at'])
        jobs.submit('script', title=f'Сцена сценария #{sc.pk}', script_id=sc.pk, action='scene', scene_id=sid,
                    instruction=str(data.get('instruction', ''))[:500])
        return JsonResponse({'ok': True})
    sc.data = sg.edit_scene(sc.data, sid, data)
    sc.save(update_fields=['data', 'updated_at'])
    return JsonResponse({'ok': True, 'data': sc.data})


@require_GET
def script_export(request, pk, what):
    """JSON of the script document ('script') or its Timeline IR ('timeline') - the input of the next stages."""
    from . import scripts as sg
    from .timeline import to_json
    sc = get_object_or_404(Script, pk=pk)
    if not (sc.data or {}).get('scenes'):
        return JsonResponse({'error': 'Сценарий ещё не готов.'}, status=400)
    if what == 'timeline':
        body = to_json(sg.to_timeline(sc.data, sc.pk), indent=2)
    else:
        body = json.dumps(sc.data, ensure_ascii=False, indent=2)
    resp = HttpResponse(body, content_type='application/json; charset=utf-8')
    resp['Content-Disposition'] = f'attachment; filename="script_{sc.pk}_{what}.json"'
    return resp


# ---------------------------------------------------------------- AI Storyboard (3B)
@ensure_csrf_cookie
def storyboard_page(request, pk):
    from . import storyboard as sb
    sc = get_object_or_404(Script, pk=pk)
    return render(request, 'studio/storyboard.html', {'script': sc, 'shots': sb.SHOTS, 'moves': sb.MOVES,
                                                      'comps': sb.COMPOSITIONS})


@require_http_methods(['GET', 'POST'])
def storyboard_api(request, pk):
    """GET: the storyboard + stale scenes. POST {scenes?: [ids]}: (re)build it in the queue (locked frames kept)."""
    from . import storyboard as sb
    sc = get_object_or_404(Script, pk=pk)
    if request.method == 'GET':
        from . import genmedia
        d = _script_json(sc)
        d['stale'] = sb.mark_stale(sc.data) if (sc.data or {}).get('scenes') else []
        d['media'] = genmedia.summary(sc.data) if (sc.data or {}).get('storyboard') else None
        d['media_url'] = settings.MEDIA_URL
        mj = Job.objects.filter(kind='media', payload__script_id=sc.pk).order_by('-created_at').first()
        d['media_job'] = _live_info(mj)
        return JsonResponse(d)
    if sc.status == 'generating':
        return _script_busy()
    if not (sc.data or {}).get('scenes'):
        return JsonResponse({'error': 'Сначала нужен готовый сценарий.'}, status=400)
    scenes = [x for x in (_json_body(request).get('scenes') or []) if isinstance(x, str)]
    sc.status = 'generating'
    sc.save(update_fields=['status', 'updated_at'])
    jobs.submit('script', title=f'Раскадровка: {sc.title or sc.idea[:60]}', script_id=sc.pk, action='storyboard',
                scenes=scenes)
    return JsonResponse({'ok': True})


@require_POST
def storyboard_frame(request, pk, fid, action=None):
    """.../frames/<fid>/ {changes}: edit; .../move/ {to}: reorder; .../regenerate/ {instruction}: new version."""
    from . import storyboard as sb
    sc = get_object_or_404(Script, pk=pk)
    if sc.status == 'generating':
        return _script_busy()
    if not (sc.data or {}).get('storyboard'):
        return JsonResponse({'error': 'Раскадровки ещё нет.'}, status=400)
    data = _json_body(request)
    try:
        if action == 'regenerate':
            sb._find(sc.data, fid)
            sc.status = 'generating'
            sc.save(update_fields=['status', 'updated_at'])
            jobs.submit('script', title=f'Кадр раскадровки #{sc.pk}', script_id=sc.pk, action='frame', frame_id=fid,
                        instruction=str(data.get('instruction', ''))[:500])
            return JsonResponse({'ok': True})
        sc.data = sb.move_frame(sc.data, fid, data.get('to', 0)) if action == 'move' else \
            sb.edit_frame(sc.data, fid, data)
    except KeyError:
        raise Http404
    sc.save(update_fields=['data', 'updated_at'])
    return JsonResponse({'ok': True})


@require_POST
def storyboard_bible(request, pk):
    """{style: {...}, characters: {id: {...}}} -> prompts of all frames re-composed (hand-written ones kept)."""
    from . import storyboard as sb
    sc = get_object_or_404(Script, pk=pk)
    if sc.status == 'generating':
        return _script_busy()
    if not (sc.data or {}).get('storyboard'):
        return JsonResponse({'error': 'Раскадровки ещё нет.'}, status=400)
    data = _json_body(request)
    sc.data = sb.edit_bibles(sc.data, data.get('style'), data.get('characters'))
    sc.save(update_fields=['data', 'updated_at'])
    return JsonResponse({'ok': True})


# ---------------------------------------------------------------- AI Visual & Voice Engine (3C)
@ensure_csrf_cookie
def generation_page(request):
    return render(request, 'studio/generation.html', {})


@require_GET
def media_status(request):
    from . import genmedia
    return JsonResponse(genmedia.status())


@require_POST
def media_config(request):
    from . import genmedia
    return JsonResponse({'ok': True, 'config': genmedia.save_config(_json_body(request))})


def _media_ready(sc):
    if sc.status == 'generating':
        return _script_busy()
    if not (sc.data or {}).get('storyboard'):
        return JsonResponse({'error': 'Сначала нужна раскадровка (3B).'}, status=400)
    return None


@require_POST
def script_media(request, pk):
    """{what: all|images|voice|frames|characters|lines, ids?, force?} -> a 'media' job (heavy, GPU)."""
    sc = get_object_or_404(Script, pk=pk)
    bad = _media_ready(sc)
    if bad:
        return bad
    data = _json_body(request)
    what = data.get('what', 'all')
    if what not in ('all', 'images', 'voice', 'frames', 'characters', 'lines'):
        return JsonResponse({'error': 'what: all | images | voice | frames | characters | lines'}, status=400)
    sc.status = 'generating'
    sc.save(update_fields=['status', 'updated_at'])
    jobs.submit('media', title=f'Медиа: {sc.title or sc.idea[:60]}', script_id=sc.pk, what=what,
                ids=data.get('ids') or None, force=bool(data.get('force')))
    return JsonResponse({'ok': True})


@require_POST
def media_upload_frame(request, pk, fid):
    """Replace a frame's image with the user's own picture."""
    from . import genmedia
    sc = get_object_or_404(Script, pk=pk)
    bad = _media_ready(sc)
    if bad:
        return bad
    f = request.FILES.get('file')
    if not f:
        return JsonResponse({'error': 'Нет файла.'}, status=400)
    try:
        sc.data = genmedia.replace_image(sc.data, sc.pk, fid, f)
    except KeyError:
        raise Http404
    except Exception as exc:
        return JsonResponse({'error': f'Не удалось прочитать изображение: {str(exc)[:150]}'}, status=400)
    sc.save(update_fields=['data', 'updated_at'])
    return JsonResponse({'ok': True})


@require_POST
def media_upload_voice(request, pk, sid, li):
    """Replace a line's audio with the user's own recording (timings are measured again)."""
    import tempfile
    from . import genmedia
    sc = get_object_or_404(Script, pk=pk)
    bad = _media_ready(sc)
    if bad:
        return bad
    f = request.FILES.get('file')
    if not f:
        return JsonResponse({'error': 'Нет файла.'}, status=400)
    with tempfile.NamedTemporaryFile(suffix=Path(f.name).suffix or '.wav', delete=False) as tmp:
        for chunk in f.chunks():
            tmp.write(chunk)
    try:
        sc.data = genmedia.replace_voice(sc.data, sc.pk, sid, int(li), tmp.name)
    except KeyError:
        raise Http404
    except Exception as exc:
        return JsonResponse({'error': f'Не удалось прочитать аудио: {str(exc)[:150]}'}, status=400)
    finally:
        os.unlink(tmp.name)
    sc.save(update_fields=['data', 'updated_at'])
    return JsonResponse({'ok': True})


@require_POST
def storyboard_voice(request, pk, cid):
    """{tts_voice}: the TTS voice of a character (from the voices the provider offers)."""
    sc = get_object_or_404(Script, pk=pk)
    bad = _media_ready(sc)
    if bad:
        return bad
    ch = sc.data['storyboard']['characters'].get(cid)
    if ch is None:
        raise Http404
    ch['tts_voice'] = str(_json_body(request).get('tts_voice') or '')[:120]
    sc.save(update_fields=['data', 'updated_at'])
    return JsonResponse({'ok': True})
