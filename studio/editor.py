"""Clip editing engine: scene plan -> filtergraph -> captions -> ffmpeg (with NVENC->CPU fallback)."""
import json
import logging
from pathlib import Path

from . import captions as cap
from . import diagnostics, exports, layouts
from . import scenes as scn
from .procutil import run_capture, run_ffmpeg

log = logging.getLogger('studio.editor')


def probe_video(path):
    d = json.loads(run_capture(['ffprobe', '-v', 'error', '-show_entries',
                                'format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate',
                                '-of', 'json', str(path)], timeout=30))
    v = next((s for s in d.get('streams', []) if s.get('codec_type') == 'video'), None)
    if not v:
        raise ValueError('Видео-поток не найден.')

    def rate(s):
        try:
            a, b = s.split('/')
            return float(a) / float(b) if float(b) else 0.0
        except Exception:
            return 0.0

    fps = rate(v.get('avg_frame_rate', '0/1')) or rate(v.get('r_frame_rate', '0/1')) or 30.0
    return {'width': int(v['width']), 'height': int(v['height']), 'fps': fps, 'codec': v.get('codec_name'),
            'has_audio': any(s.get('codec_type') == 'audio' for s in d['streams']),
            'duration': float(d['format'].get('duration') or 0)}


def prepare(source, clip, mode, corner, opts, vinfo, rep=None, progress=None):
    """Analyse the clip's time range and decide the scene plan. Never raises for analysis problems:
    on any failure the plan degrades to a safe single-layout fallback."""
    L = clip['end'] - clip['start']
    w, h = vinfo['width'], vinfo['height']
    cancel = rep.cancelled if rep else None
    info = {'mode': mode, 'analysis': 'skipped'}
    analysis = None
    notes = []
    if mode in ('director', 'smart', 'gameplay_facecam', 'fullframe'):
        try:
            analysis = scn.analyze(source, clip['start'], clip['end'], w, h, corner_hint=corner, cancel=cancel,
                                   progress=progress, detect_faces=(mode != 'fullframe'))
            info['analysis'] = f'{len(analysis["samples"])} кадров'
        except Exception as exc:
            from .procutil import JobCancelled
            if isinstance(exc, JobCancelled):
                raise
            log.warning('scene analysis failed (%s); using safe fallback', exc)
            info['analysis'] = f'ошибка: {str(exc)[:120]}'
    cam = (analysis or {}).get('cam')
    forced = scn.manual_cam(corner) if corner != 'auto' else None
    cw, ch = layouts.crop_dims(w, h, *exports.dims(opts))
    hs = scn.hardsub_summary(analysis['samples']) if analysis else {'present': False}
    if hs.get('present'):
        notes.append('В исходнике есть вшитые титры/субтитры — собственные субтитры ClipForge можно отключить '
                     '(кнопка «Субтитры» → «Без субтитров»), чтобы текст не дублировался.')

    def text_cut(seg_start, seg_end, path):
        """Would a 9:16 crop following `path` slice through the source's own burned-in text in this span?"""
        if not (opts['text_guard'] and hs.get('present')):
            return False
        sub = [dict(s, t=s['t'] - seg_start) for s in analysis['samples'] if seg_start <= s['t'] < seg_end]
        return scn.crop_cut_fraction(sub, path, w, cw) >= 0.25

    def tracked(seg_start, seg_end, avoid):
        samples = analysis['samples'] if analysis else []
        if opts['smart_tracking'] and samples:
            path, found = scn.track_path(samples, seg_start, seg_end, w, h, cw, ch, avoid_cam=avoid)
            return path, found
        faces = [f for s in samples for f in s['faces'] if f['area'] >= 0.002]
        if faces:
            xs = sorted(f['cx'] for f in faces)
            cx = xs[len(xs) // 2]
            x = max(0, min(w - cw, cx * w - cw / 2))
            return [(0.0, x, (h - ch) / 2)], True
        return None, False

    segs = []
    if mode == 'director' and analysis:
        plan = scn.plan_scenes(analysis, L, fallback=opts['director_fallback'],
                               forced_cam=(forced if forced else None))
        use_cam = cam or forced
        for sg in plan:
            seg = dict(sg)
            if sg['type'] in ('talking', 'smart'):
                path, found = tracked(sg['start'], sg['end'], use_cam)
                seg['path'] = path
                seg['face_found'] = found
                if text_cut(sg['start'], sg['end'], path):
                    seg.update(type='fullframe', fallback=True, reason='hardsub')
                    notes.append('Часть клипа показана целиком: кроп 9:16 разрезал бы вшитые титры исходника.')
            if sg['type'] == 'game_cam':
                seg['face_rel'] = (use_cam or {}).get('face_rel', (0.5, 0.5))
            segs.append(seg)
        info['cam'] = use_cam
    elif mode == 'director':
        segs = [{'start': 0.0, 'end': L, 'type': opts['director_fallback'], 'confidence': 0.0, 'fallback': True,
                 'detected': 'unknown'}]
    elif mode == 'smart':
        path, found = tracked(0.0, L, cam)
        if text_cut(0.0, L, path):
            segs = [{'start': 0.0, 'end': L, 'type': 'fullframe', 'confidence': 1.0, 'fallback': True,
                     'reason': 'hardsub', 'detected': 'smart'}]
            notes.append('Smart Reframe показал кадр целиком: в исходнике вшитые титры, кроп 9:16 их обрезал бы. '
                         'Чтобы всегда показывать весь кадр, выберите режим «Без обрезки».')
        else:
            segs = [{'start': 0.0, 'end': L, 'type': 'smart', 'path': path, 'face_found': found, 'confidence': 1.0,
                     'fallback': False}]
    elif mode == 'gameplay_facecam':
        use_cam = cam or forced
        if not use_cam:
            use_cam = scn.manual_cam('top_right')
            use_cam['source'] = 'default'
        info['cam'] = use_cam
        segs = [{'start': 0.0, 'end': L, 'type': 'game_cam', 'face_rel': use_cam.get('face_rel', (0.5, 0.5)),
                 'confidence': use_cam['confidence'], 'fallback': False}]
    elif mode == 'fullframe':
        segs = [{'start': 0.0, 'end': L, 'type': 'fullframe', 'confidence': 1.0, 'fallback': False}]
    elif mode == 'gameplay':
        segs = [{'start': 0.0, 'end': L, 'type': 'gameplay', 'confidence': 1.0, 'fallback': False}]
    else:
        segs = [{'start': 0.0, 'end': L, 'type': 'center', 'confidence': 1.0, 'fallback': False}]
    # a director segment that wants a cam layout without a known cam must not crash the graph
    cam_known = info.get('cam')
    for sg in segs:
        if sg['type'] == 'game_cam' and not cam_known:
            sg['type'] = opts['director_fallback']
            sg['fallback'] = True
    return {'segments': segs, 'cam': info.get('cam'), 'info': info, 'notes': notes}


def _public_segments(segs):
    return [{'start': round(s['start'], 2), 'end': round(s['end'], 2), 'type': s['type'],
             'confidence': s.get('confidence'), 'fallback': bool(s.get('fallback')),
             'detected': s.get('detected'), 'reason': s.get('reason')} for s in segs]


def render(source, clip, transcript, dest, temp, opts, plan, vinfo, mode, rep=None, on_progress=None):
    """Render one clip. Returns an info dict. Raises on failure (caller may fall back to legacy render)."""
    from . import pipeline
    temp = Path(temp)
    L = clip['end'] - clip['start']
    W, H = exports.dims(opts)
    segs = plan['segments']
    ctx = {'W': W, 'H': H, 'src_w': vinfo['width'], 'src_h': vinfo['height'], 'bg': opts['bg_fill'],
           'zoom': opts['gameplay_zoom'], 'cam': plan.get('cam'), 'cam_position': opts['cam_position'],
           'hide_cam': opts['hide_cam'], 'path_tol': 3.0}
    fps = None
    if opts['fps'] != 'source' and int(opts['fps']) < vinfo['fps'] - 0.5:
        fps = int(opts['fps'])
    transition = opts['transition'] if len(segs) > 1 else 'none'
    for tol in (3.0, 8.0, 20.0):
        ctx['path_tol'] = tol
        graph, margin_for = layouts.build_graph(segs, ctx, 'captions.ass', transition=transition, fps=fps)
        if len(graph) < 24000:
            break
    ass = temp / 'captions.ass'
    cap_report = {'style': opts['subtitle_style'], 'enabled': opts['captions']}
    if not opts['captions']:
        cap.build_ass([], clip['start'], clip['end'], ass, W, H, 'clean')
    elif opts['subtitle_style'] == 'classic':
        pipeline.captions_ass(transcript, clip['start'], clip['end'], ass, margin_v=margin_for(L / 2))
    else:
        cap_report = cap.build_ass(transcript, clip['start'], clip['end'], ass, W, H, opts['subtitle_style'],
                                   max_words=opts['caption_words'] or None, margin_for=margin_for,
                                   animate=opts['caption_animate'], highlight=opts['caption_highlight'])
    encoder, enc_note = diagnostics.choose_encoder(opts['encoder'])
    info = {'encoder': encoder, 'encoder_note': enc_note, 'graph_chars': len(graph), 'size': f'{W}x{H}',
            'segments': _public_segments(segs), 'captions': cap_report, 'mode': mode, 'notes': plan.get('notes', []), 'fps': fps or 'source',
            'transition': transition, 'cam': plan.get('cam') and {k: (round(v, 3) if isinstance(v, float) else v)
                                                                  for k, v in plan['cam'].items()}}

    def cmd_for(enc):
        c = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{clip["start"]:.3f}', '-i', str(source),
             '-t', f'{L:.3f}', '-filter_complex', graph, '-map', '[vout]', '-map', '0:a:0?']
        c += exports.encoder_args(enc, opts)
        c += ['-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', str(dest)]
        return c

    cancel = rep.cancelled if rep else None
    try:
        run_ffmpeg(cmd_for(encoder), duration=L, on_progress=on_progress, cancel_check=cancel, timeout=3600, cwd=str(temp))
    except RuntimeError as exc:
        if encoder == 'h264_nvenc':
            log.warning('NVENC failed (%s); retrying on CPU', str(exc)[:200])
            info['encoder'], info['encoder_note'] = 'libx264', 'CPU (откат: NVENC не сработал)'
            info['encoder_fallback'] = True
            run_ffmpeg(cmd_for('libx264'), duration=L, on_progress=on_progress, cancel_check=cancel, timeout=3600,
                       cwd=str(temp))
        else:
            raise
    return info
