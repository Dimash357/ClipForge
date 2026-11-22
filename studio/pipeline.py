import json
import math
import os
import re
import subprocess
import tempfile
import shutil
import traceback
from pathlib import Path
from django.conf import settings
from django.core.files import File
from django.db import close_old_connections
from .models import Project, Clip


def run(args, timeout=1800, cwd=None):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, encoding='utf-8', errors='replace', timeout=timeout, cwd=cwd)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-1700:])
    return p.stdout


def update(p, progress, message):
    Project.objects.filter(pk=p.pk).update(progress=progress, message=message)


def probe(path):
    d = json.loads(run(['ffprobe', '-v', 'error', '-show_entries',
                   'format=duration:stream=codec_type,width,height', '-of', 'json', str(path)], timeout=30))
    duration = float(d['format']['duration'])
    if not math.isfinite(duration) or duration < 5 or duration > 4*3600:
        raise ValueError('Поддерживаются видео от 5 секунд до 4 часов.')
    if not any(s.get('codec_type') == 'video' for s in d.get('streams', [])):
        kinds = ', '.join(sorted({s.get('codec_type', '?') for s in d.get('streams', [])})) or 'нет дорожек'
        raise ValueError(f'Видео-поток не найден: в файле только {kinds}. Если видео скачивалось по ссылке, '
                         'нажмите «Продолжить» — ClipForge скачает его заново (подробности в logs\\ytdlp-*.log).')
    return duration


_model_cache = {}


def _get_model(name, device, compute):
    from faster_whisper import WhisperModel
    key = (name, device, compute)
    if key not in _model_cache:
        _model_cache.clear()  # keep a single model resident (VRAM / RAM)
        _model_cache[key] = WhisperModel(name, device=device, compute_type=compute)
    return _model_cache[key]


def transcribe(path, lang, rep=None, duration=None):
    """faster-whisper. Device comes from diagnostics (auto: CUDA only if the probe passed, else CPU).
    Any CUDA failure (missing cuDNN/cuBLAS, OOM...) transparently restarts on CPU. Reports real progress."""
    from . import diagnostics
    from .procutil import JobCancelled
    model_name = os.getenv('WHISPER_MODEL', 'small')
    device, compute, note = diagnostics.whisper_settings()
    attempts = [(device, compute)] + ([('cpu', 'int8')] if device == 'cuda' else [])
    for dev, comp in attempts:
        try:
            model = _get_model(model_name, dev, comp)
            batch = whisper_batch_size(dev)
            out, info, used = None, None, 0
            if batch:
                try:
                    out, info = _run_whisper(model, path, lang, rep, duration, dev, batch)
                    used = batch
                except JobCancelled:
                    raise
                except Exception as exc:          # OOM / old faster-whisper: the sequential path still works
                    log.warning('batched Whisper failed on %s (%s); sequential decoding', dev, str(exc)[:200])
            if out is None:
                out, info = _run_whisper(model, path, lang, rep, duration, dev, 0)
            transcribe.last_info = {'model': model_name, 'device': dev, 'compute': comp, 'batch': used,
                                    'detected_language': getattr(info, 'language', None), 'requested_lang': lang}
            return out
        except JobCancelled:
            raise
        except Exception as exc:
            if dev == 'cuda':
                diagnostics.mark_whisper_gpu_failed(str(exc)[:200])
                _model_cache.clear()
                continue
            raise
    return []


transcribe.last_info = {}


def whisper_batch_size(device):
    """WHISPER_BATCH: 'auto' (default) = 16 on CUDA, off on CPU; '0' = off; a number = that batch size."""
    v = os.getenv('WHISPER_BATCH', 'auto').strip().lower()
    if v in ('auto', ''):
        return 16 if device == 'cuda' else 0
    try:
        return max(0, min(64, int(v)))
    except ValueError:
        return 0


def _run_whisper(model, path, lang, rep, duration, dev, batch):
    """One decoding pass. batch > 0: faster-whisper's BatchedInferencePipeline (VAD-cut chunks decoded in
    parallel: 2-4x faster on a GPU); 0: the classic sequential decoder."""
    kw = dict(language=None if lang == 'auto' else lang, beam_size=5, word_timestamps=True, vad_filter=True)
    if batch:
        from faster_whisper import BatchedInferencePipeline
        segments, info = BatchedInferencePipeline(model=model).transcribe(str(path), batch_size=batch, **kw)
    else:
        segments, info = model.transcribe(str(path), **kw)
    out = []
    label = dev.upper() + (f', пакеты по {batch}' if batch else '')
    for s in segments:
        if rep:
            rep.check()
            if duration:
                rep.frac(s.end / duration, f'Распознаём речь: {int(min(100, 100 * s.end / duration))}% ({label})')
        words = [{'start': round(w.start, 3), 'end': round(w.end, 3), 'text': w.word.strip()}
                 for w in (s.words or []) if w.word.strip()]
        out.append({'start': round(s.start, 3), 'end': round(s.end, 3), 'text': s.text.strip(), 'words': words})
    out.sort(key=lambda x: x['start'])           # batched output is chronological, but never trust it blindly
    return out, info


def candidates(segments, duration, minlen=20, maxlen=55):
    if not segments:
        return []
    out = []
    for i in range(len(segments)):
        start = segments[i]['start']
        for j in range(i, len(segments)):
            end = segments[j]['end']
            length = end-start
            if length > maxlen:
                break
            if length < minlen:
                continue
            group = segments[i:j+1]
            txt = ' '.join(s['text'] for s in group)
            words = re.findall(r'\w+', txt, re.UNICODE)
            if len(words) < 15:
                continue
            excitement = min(txt.count('!')*1.2+txt.count('?')*1.5, 6)
            hook = 3 if re.search(
                r'\b(why|how|secret|never|imagine|crazy|shocking|biggest|first|last|what if|почему|как|секрет|никогда|представь|самый|впервые)\b', txt, re.I) else 0
            pace = min(len(words)/max(length, 1)*1.2, 5)
            completeness = 3 if txt.rstrip().endswith(('.', '!', '?', '…')) else 0
            score = 50+excitement+hook+pace+completeness-abs(length-35)*.08
            out.append({'start': start, 'end': end,
                       'text': txt, 'score': score})
    return sorted(out, key=lambda x: x['score'], reverse=True)


def ollama_rank(items):
    # Optional local semantic reranker; falls back to transparent heuristics.
    from . import ai_provider
    if os.getenv('USE_OLLAMA', '0') != '1' or ai_provider.paused():
        return items
    try:
        import requests
        model = os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')
        subset = items[:18]
        prompt = 'Score each numbered transcript excerpt for short-form video: hook, clarity, interestingness, self-contained story. Return ONLY JSON array of objects with id (integer) and score (0-100).\n' + '\n'.join(
            f'{i}: {x["text"][:900]}' for i, x in enumerate(subset))
        r = requests.post(os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/') + '/api/generate', json={
                          'model': model, 'prompt': prompt, 'stream': False, 'format': 'json'}, timeout=(3.05, 120))
        r.raise_for_status()
        raw = json.loads(r.json()['response'])
        scores = raw if isinstance(raw, list) else raw.get('scores', [])
        for item in scores:
            idx = int(item['id'])
            value = float(item['score'])
            if 0 <= idx < len(subset):
                subset[idx]['score'] = 0.4*subset[idx]['score'] + \
                    0.6*max(0, min(100, value))
        return sorted(items, key=lambda x: x['score'], reverse=True)
    except Exception as exc:
        if not isinstance(exc, (ValueError, KeyError, TypeError)):
            ai_provider.trip(exc)
        return items


def choose(items, count):
    result = []
    for x in items:
        if any(max(0, min(x['end'], y['end'])-max(x['start'], y['start'])) > min(x['end']-x['start'], y['end']-y['start'])*.20 for y in result):
            continue
        result.append(x)
        if len(result) >= count:
            break
    return result


def ass_time(sec):
    cs = max(0, round(sec*100))
    return f'{cs//360000}:{(cs//6000)%60:02}:{(cs//100)%60:02}.{cs%100:02}'


def ass_escape(text): return text.replace('\\', '').replace(
    '{', '(').replace('}', ')').replace('\n', ' ')


def caption_groups(words, max_words=5, max_chars=36, max_gap=0.36):
    """Split on pauses, punctuation and line width rather than fixed word count."""
    groups, group = [], []
    for w in words:
        text = w['text'].strip()
        if not text:
            continue
        if group and (w['start']-group[-1]['end'] > max_gap or
                      len(group) >= max_words or
                      sum(len(x['text'])+1 for x in group)+len(text) > max_chars):
            groups.append(group)
            group=[]
        group.append(w)
        if re.search(r'[.!?…]$',text):
            groups.append(group)
            group=[]
    if group:groups.append(group)
    return groups


def captions_ass(segments, start, end, dest, margin_v=285):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,64,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,1,0,0,0,100,100,0,0,1,5,2,2,70,70,%MV%,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines=[header.replace('%MV%',str(int(margin_v)))]
    words=[]
    for seg in segments:
        if seg['end']<=start or seg['start']>=end:continue
        words.extend(w for w in seg.get('words',[]) if w['end']>start and w['start']<end)
    if words:
        for group in caption_groups(words):
            a=max(start,group[0]['start']); b=min(end,group[-1]['end'])
            if b<=a:continue
            # Word highlighting, respecting pause before the next word.
            payload=''.join(r'{\k'+str(max(1,round((min(end,w['end'])-max(start,w['start']))*100)))+'}'+ass_escape(w['text'])+(' ' if idx<len(group)-1 else '') for idx,w in enumerate(group))
            lines.append(f'Dialogue: 0,{ass_time(a-start)},{ass_time(b-start)},Default,,0,0,0,,{payload}\n')
    else:
        for seg in segments:
            a=max(start,seg['start']); b=min(end,seg['end'])
            if b>a:lines.append(f'Dialogue: 0,{ass_time(a-start)},{ass_time(b-start)},Default,,0,0,0,,{ass_escape(seg["text"])}\n')
    dest.write_text(''.join(lines),encoding='utf-8')


def video_dimensions(path):
    d=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','json',str(path)],timeout=30))
    stream=d['streams'][0]
    return int(stream['width']),int(stream['height'])


def detect_faces(source, seconds, corner='auto'):
    """Sample video with OpenCV Haar face detector; no network/model download."""
    try:
        import cv2
        cap=cv2.VideoCapture(str(source))
        detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        if detector.empty():return []
        faces=[]
        for t in seconds:
            cap.set(cv2.CAP_PROP_POS_MSEC,max(0,t)*1000)
            ok,frame=cap.read()
            if not ok:continue
            h,w=frame.shape[:2]
            small=cv2.resize(frame,(640,max(1,int(640*h/w))))
            gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
            boxes=detector.detectMultiScale(gray,scaleFactor=1.12,minNeighbors=5,minSize=(28,28))
            for x,y,bw,bh in boxes:
                cx=(x+bw/2)/640; cy=(y+bh/2)/small.shape[0]
                if corner!='auto':
                    is_left='left' in corner; is_top='top' in corner
                    if (cx<.5)!=is_left or (cy<.5)!=is_top:continue
                faces.append((t,cx,cy,bw*bh/(640*small.shape[0])))
        cap.release()
        return faces
    except Exception:
        return []


def facecam_crop(faces, corner):
    """Normalized square around a detected face or conventional overlay corner."""
    if faces:
        best=max(faces,key=lambda f:f[3]);cx,cy=best[1:3]
    else:
        cx,cy={'top_left':(.16,.19),'top_right':(.84,.19),'bottom_left':(.16,.81),'bottom_right':(.84,.81)}.get(corner,(.84,.19))
    # Crop 25% of source width and 40% height; clamp to source bounds.
    cw,ch=.30,.42
    return max(0,min(1-cw,cx-cw/2)),max(0,min(1-ch,cy-ch/2)),cw,ch


def render(source, clip, segments, dest, temp, mode='smart', corner='auto'):
    ass=temp/'captions.ass'
    captions_ass(segments,clip['start'],clip['end'],ass)
    w,h=video_dimensions(source)
    if mode=='gameplay_facecam':
        # Gameplay uses full source scaled to fit upper panel, facecam lower panel.
        # A local face detector identifies an existing facecam overlay; manual corner fallback.
        times=[clip['start']+(clip['end']-clip['start'])*x/5 for x in range(1,5)]
        faces=detect_faces(source,times,corner)
        x,y,cw,ch=facecam_crop(faces,corner)
        fw=max(2,int(w*cw)//2*2);fh=max(2,int(h*ch)//2*2)
        fx=max(0,min(w-fw,int(w*x)//2*2));fy=max(0,min(h-fh,int(h*y)//2*2))
        vf=(f'[0:v]split=2[game][cam];'
            f'[game]scale=1080:1240:force_original_aspect_ratio=decrease,'
            f'pad=1080:1240:(ow-iw)/2:(oh-ih)/2:black[gp];'
            f'[cam]crop={fw}:{fh}:{fx}:{fy},scale=1080:680:force_original_aspect_ratio=decrease,'
            f'pad=1080:680:(ow-iw)/2:(oh-ih)/2:black[fc];'
            f'[gp][fc]vstack=inputs=2,setsar=1,ass=captions.ass[vout]')
    elif mode=='gameplay':
        # Full game field-of-view retained, with soft blurred background.
        vf=('[0:v]split=2[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,'
            'crop=1080:1920,boxblur=20:2[b];[fg]scale=1080:1920:force_original_aspect_ratio=decrease[f];'
            '[b][f]overlay=(W-w)/2:(H-h)/2,setsar=1,ass=captions.ass[vout]')
    else:
        # Smart mode: fixed subject-aware crop based on sampled face positions.
        # Avoids jarring motion; a future version can track frame-by-frame.
        crop_w=min(w,max(2,int(h*9/16)//2*2))
        crop_h=min(h,max(2,int(w*16/9)//2*2))
        if mode=='smart' and crop_w<w:
            ts=[clip['start']+(clip['end']-clip['start'])*x/5 for x in range(1,5)]
            faces=detect_faces(source,ts)
            if faces:
                center=sorted(f[1] for f in faces)[len(faces)//2]
                x=max(0,min(w-crop_w,int(w*center-crop_w/2)))
            else:x=(w-crop_w)//2
        else:x=(w-crop_w)//2
        y=(h-crop_h)//2
        vf=f'[0:v]crop={crop_w}:{crop_h}:{x}:{y},scale=1080:1920,setsar=1,ass=captions.ass[vout]'
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(clip['start']),'-i',str(source),'-t',str(clip['end']-clip['start']),'-filter_complex',vf,'-map','[vout]','-map','0:a:0?','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-movflags','+faststart',str(dest)],timeout=1200,cwd=str(temp))

# =====================================================================================================
# v5 orchestration. Everything above this line is the proven v4 pipeline and is kept as fallback.
# =====================================================================================================
import hashlib
import logging
import time

from . import (audio_analysis, diagnostics, editor, exports, highlights, metadata, motion_analysis, quality, scenes,
               textutil as tx)
from .jobs import set_state
from .procutil import JobCancelled

log = logging.getLogger('studio.pipeline')
LEGACY_MODES = ('smart', 'center', 'gameplay', 'gameplay_facecam')
NEEDS_ANALYSIS = ('director', 'smart', 'gameplay_facecam', 'reaction', 'auto')
CAM_MODES = ('director', 'gameplay_facecam', 'reaction', 'auto')


def fingerprint(path):
    """Cheap identity of a big file: size + first and last MB."""
    path = Path(path)
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open('rb') as f:
        h.update(f.read(1 << 20))
        if size > (2 << 20):
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return h.hexdigest()


def clip_text(segments, start, end):
    words = [w['text'] for s in segments if s['end'] > start and s['start'] < end
             for w in (s.get('words') or []) if w['end'] > start and w['start'] < end]
    if words:
        return ' '.join(words)
    return ' '.join(s['text'] for s in segments if s['end'] > start and s['start'] < end)


def highlight_signals(source, duration, opts, vinfo=None, rep=None, cache=None, wav=None):
    """Audio profile, keyframe screen scan and a lazy gameplay-motion probe for the multimodal ranker.

    Results are stored in / loaded from the source's analysis cache (`cache` dir), so re-picking moments or
    re-editing never decodes the audio again. Never raises (except cancel): a broken/absent audio track or a
    decoder problem just means "rank on what is available"."""
    from . import analysis_cache as ac
    audio = motion = screen = None
    notes = []
    if source is None:
        return audio, motion, notes
    cancel = rep.cancelled if rep else None
    if opts.get('highlight_audio', True):
        try:
            hit = ac.load_npz(Path(cache) / 'audio.npz') if cache else None
            audio = audio_analysis.from_arrays(hit) if hit else None
            if audio is not None:
                notes.append('audio_cached')
            else:
                audio = audio_analysis.analyze_file(
                    wav or source, duration, cancel=cancel,
                    progress=(lambda f: rep.frac(0.6 * f, f'Слушаем звук: смех, крики, всплески... {int(f * 100)}%'))
                    if rep else None)
                if audio is None:
                    notes.append('no_audio')
                elif cache:
                    ac.save_npz(Path(cache) / 'audio.npz', **audio_analysis.to_arrays(audio))
            if audio is not None:
                log.info('audio profile: %s', audio.summary())
        except JobCancelled:
            raise
        except Exception as exc:
            log.warning('audio analysis failed (%s); ranking on the transcript only', str(exc)[:300])
            notes.append('audio_failed')
            audio = None
        if audio is not None and opts.get('highlight_sound_ai', True):
            tl = _sound_timeline(source, wav, duration, cache, rep, cancel, notes)
            if tl is not None:
                audio.attach_sound(tl)
                log.info('sound classifier events: %s', tl.summary())
    if opts.get('highlight_motion', True) and vinfo:
        try:
            hit = ac.load_npz(Path(cache) / 'screen.npz') if cache else None
            screen = motion_analysis.screen_from_arrays(hit) if hit else None
            if screen is None:
                if rep:
                    rep.frac(0.65, 'Смотрим на экран: смерти, провалы, вспышки, смены сцен...')
                screen = motion_analysis.scan_keyframes(source, duration, cancel=cancel)
                if screen is not None and cache:
                    ac.save_npz(Path(cache) / 'screen.npz', **motion_analysis.screen_to_arrays(screen))
        except JobCancelled:
            raise
        except Exception as exc:
            log.warning('screen scan failed (%s)', str(exc)[:300])
            notes.append('screen_failed')
            screen = None

        def motion(spans):
            if rep:
                rep.frac(0.85, 'Оцениваем динамику геймплея у лучших кандидатов...')
            return motion_analysis.analyze_spans(source, spans, vinfo['width'], vinfo['height'], cancel=cancel)
    highlight_signals.last_screen = screen
    return audio, motion, notes


highlight_signals.last_screen = None


def _sound_timeline(source, wav, duration, cache, rep, cancel, notes):
    """YAMNet sound classes for the whole track (cached). None -> keep the loudness heuristics."""
    from . import analysis_cache as ac, sound_events
    try:
        hit = ac.load_npz(Path(cache) / 'sound.npz') if cache else None
        tl = sound_events.from_arrays(hit) if hit else None
        if tl is not None:
            notes.append('sound_cached')
            return tl
        if rep:
            rep.frac(0.6, 'Загружаем классификатор звуков (один раз, 16 МБ)...')
        if sound_events.load(download=settings.SOUND_MODEL_DOWNLOAD,
                             progress=(lambda f: rep.frac(0.6, f'Загружаем классификатор звуков: {int(f * 100)}%'))
                             if rep else None) is None:
            notes.append('sound_model_unavailable')
            return None
        tl = sound_events.classify_file(
            wav or source, duration, cancel=cancel,
            progress=(lambda f: rep.frac(0.6 + 0.05 * f, f'Распознаём звуки: смех, крики, взрывы... {int(f * 100)}%'))
            if rep else None)
        if tl is not None and cache:
            ac.save_npz(Path(cache) / 'sound.npz', **sound_events.to_arrays(tl))
        return tl
    except JobCancelled:
        raise
    except Exception as exc:
        log.warning('sound classifier failed (%s); loudness heuristics only', str(exc)[:300])
        notes.append('sound_failed')
        return None


def pick_highlights(segments, duration, p, opts, source=None, vinfo=None, rep=None, cache=None, wav=None):
    """New ranker first; the proven v4 algorithm is the fallback when it errors or finds nothing."""
    if opts['highlight_engine'] == 'smart':
        screen = None
        try:
            audio, motion, _ = highlight_signals(source, duration, opts, vinfo, rep, cache=cache, wav=wav)
            screen = highlight_signals.last_screen
        except JobCancelled:
            raise
        except Exception:
            log.exception('highlight signals failed')
            audio = motion = None
        try:
            sel = highlights.find_highlights(segments, duration, p.clip_min, p.clip_max, p.clip_count,
                                             rerank=ollama_rank if os.getenv('USE_OLLAMA', '0') == '1' else None,
                                             audio=audio, motion=motion, screen=screen,
                                             auto_length=bool(opts.get('auto_length')))
            if sel:
                for c in sel:
                    c['engine'] = 'smart'
                return sel
            log.info('smart highlights found nothing; using classic ranker')
        except JobCancelled:
            raise
        except Exception:
            log.exception('smart highlights failed; using classic ranker')
    ranked = ollama_rank(candidates(segments, duration, p.clip_min, p.clip_max))
    sel = choose(ranked, p.clip_count)
    for c in sel:
        c.update({'engine': 'classic', 'scores': {}, 'reason': 'Выбран классическим алгоритмом (резервный режим).'})
    return sel


def delete_clip_files(clip):
    """Remove generated renders only. Never touches project sources."""
    for f in (clip.video, clip.poster):
        try:
            if f and f.name:
                f.storage.delete(f.name)
        except Exception:
            log.warning('could not delete %s', f.name)


def after_cancel(pk):
    for c in Clip.objects.filter(project_id=pk, status__in=('queued', 'rendering')):
        c.status = 'ready' if c.video else 'candidate'
        c.save(update_fields=['status', 'updated_at'])


def _preflight():
    ff = diagnostics.ffmpeg_info()
    if not ff['ffmpeg'] or not ff['ffprobe']:
        raise RuntimeError('FFmpeg/ffprobe не найдены в PATH. Установите FFmpeg (см. README) и перезапустите ClipForge.')
    try:
        free = shutil.disk_usage(str(settings.MEDIA_ROOT if Path(settings.MEDIA_ROOT).exists() else settings.BASE_DIR)).free
        if free < 1.5 * 1024 ** 3:
            raise RuntimeError(f'Мало места на диске: свободно {free / 1024 ** 3:.1f} ГБ (нужно минимум 1.5 ГБ).')
    except OSError:
        pass


def _ensure_source(p, rep):
    from .procutil import describe_streams, has_video
    if p.source and Path(p.source.path).is_file():
        path = Path(p.source.path)
        if has_video(path):
            rep.skip('fetch')
            return path
        if not p.youtube_url:
            raise RuntimeError(f'В загруженном файле нет видеодорожки ({describe_streams(path)}). '
                               'Загрузите файл с видео.')
        # an earlier download kept only the audio: fetch the video again (the old file is left untouched)
        log.warning('project %s: source %s has no video stream (%s); downloading again', p.pk, path.name,
                    describe_streams(path))
        Project.objects.filter(pk=p.pk).update(source='')
        p.refresh_from_db()
    if not p.youtube_url:
        raise RuntimeError('Исходный файл не найден на диске. Загрузите видео заново.')
    from .youtube_download import fetch_source
    rep.stage('fetch', 'Получаем видео...', status='fetching')
    fetch_source(p.pk, rep)
    p.refresh_from_db()
    return Path(p.source.path)


def _get_transcript(p, source, duration, fp, rep, wav=None):
    meta = p.transcript_meta or {}
    if p.transcript and meta.get('requested_lang', p.language) == p.language:
        rep.skip('transcribe')
        rep.stage('transcribe', 'Транскрипт уже есть — повторное распознавание не нужно', frac=1.0)
        return p.transcript
    if fp:
        for other in Project.objects.filter(source_fingerprint=fp, language=p.language).exclude(pk=p.pk):
            if other.transcript:
                set_state(p.pk, transcript=other.transcript, transcript_meta=dict(other.transcript_meta or {},
                                                                                   reused_from=other.pk))
                rep.skip('transcribe')
                rep.stage('transcribe', f'Транскрипт взят из проекта #{other.pk} (то же видео)', frac=1.0)
                return other.transcript
    rep.stage('transcribe', 'Распознаём речь (первый запуск скачает модель Whisper)...')
    try:
        segments = transcribe(wav or source, p.language, rep=rep, duration=duration)
    except JobCancelled:
        raise
    except Exception as exc:
        msg = str(exc)
        if any(k in msg for k in ('ConnectError', 'Hub', 'snapshot', 'Name or service', 'getaddrinfo')):
            raise RuntimeError('Не удалось загрузить модель Whisper. При первом запуске нужен интернет '
                               '(модель скачивается один раз и затем хранится локально). Проверьте соединение и '
                               'нажмите «Продолжить».\n\nТехнические детали: ' + msg[-300:])
        raise
    if not segments:
        raise ValueError('Речь не найдена. Для этой версии нужен ролик с разговорной речью.')
    info = dict(transcribe.last_info or {})
    set_state(p.pk, transcript=segments, transcript_meta=info)
    return segments


def snap_bounds(segments, start, end):
    """Manual bounds never cut a word: a start inside a word moves to the word's start, an end inside a word moves
    to the word's end (+ a short breath)."""
    for w in tx.flat_words(segments):
        if w['start'] < start < w['end']:
            start = max(0.0, w['start'] - 0.02)
        if w['start'] < end < w['end']:
            end = w['end'] + 0.08
    return round(start, 2), round(end, 2)


def _source_analysis(p, source, duration, vinfo, fp, rep):
    """Steps 1-2: probe facts, webcam presence/position for the whole stream, mono 16 kHz audio.
    Everything is cached per source fingerprint and re-used by later runs / other projects of the same video."""
    from . import analysis_cache as ac
    cache = ac.cache_dir(fp)
    old = dict(p.analysis or {})
    analysis = {'version': ac.VERSION, 'fingerprint': fp,
                'probe': {'duration': round(duration, 2), 'width': vinfo['width'], 'height': vinfo['height'],
                          'fps': round(vinfo['fps'], 2), 'has_audio': vinfo['has_audio']}}
    cancel = rep.cancelled
    wav = None
    if vinfo['has_audio']:
        try:
            wav = ac.extract_audio(source, cache / 'audio16k.wav', duration=duration, cancel=cancel,
                                   on_progress=lambda f: rep.frac(0.7 * f, f'Извлекаем звук (mono 16 kHz): {int(f * 100)}%'))
        except JobCancelled:
            raise
        except Exception as exc:
            log.warning('audio extraction failed (%s); Whisper will read the source directly', str(exc)[:300])
            wav = None
    analysis['audio_wav'] = bool(wav)
    if old.get('fingerprint') == fp and old.get('cam_checked') and old.get('cam_corner') == p.facecam_corner:
        analysis['cam'] = old.get('cam')
        analysis['cam_checked'] = True
    elif p.edit_mode in CAM_MODES:
        rep.frac(0.75, 'Ищем вебку стримера по всему видео...')
        try:
            analysis['cam'] = scenes.detect_source_cam(source, duration, vinfo['width'], vinfo['height'],
                                                       corner_hint=p.facecam_corner, cancel=cancel)
            analysis['cam_checked'] = True
        except JobCancelled:
            raise
        except Exception as exc:
            log.warning('source webcam detection failed: %s', exc)
            analysis['cam'] = None
    else:
        analysis['cam'] = old.get('cam') if old.get('fingerprint') == fp else None
    analysis['cam_corner'] = p.facecam_corner
    if old.get('source_meta') and old.get('fingerprint') == fp:
        analysis['source_meta'] = old['source_meta']
    elif p.youtube_url:
        try:
            from .youtube import get_metadata
            m = get_metadata(p.youtube_url)
            analysis['source_meta'] = {'title': m.get('title', '')[:300], 'author': m.get('author', '')[:120]}
        except Exception as exc:
            log.info('YouTube metadata unavailable: %s', exc)
    rep.frac(1.0)
    set_state(p.pk, analysis=analysis)
    return cache, wav, analysis


def run_job(kind, pk, payload, rep):
    if kind in ('full', 'resume', 'reprocess'):
        _run_full(pk, rep, payload)
    elif kind == 'render':
        _run_render(pk, rep, payload)
    else:
        raise ValueError(f'unknown job kind {kind}')


def _run_full(pk, rep, payload):
    p = Project.objects.get(pk=pk)
    opts = exports.resolve(p.options)
    rep.stage('audio', 'Проверяем систему и видео...', status='processing')
    _preflight()
    source = _ensure_source(p, rep)
    p.refresh_from_db()
    rep.stage('audio', 'Анализируем видео и аудио...')
    duration = probe(source)
    vinfo = editor.probe_video(source)
    fp = p.source_fingerprint or fingerprint(source)
    set_state(pk, source_duration=duration, source_fingerprint=fp, status='processing')
    p.refresh_from_db()
    cache, wav, analysis = _source_analysis(p, source, duration, vinfo, fp, rep)
    segments = _get_transcript(p, source, duration, fp, rep, wav=wav)
    p.refresh_from_db()
    analysis['language'] = (p.transcript_meta or {}).get('detected_language') or p.language
    rep.stage('highlights', 'Ищем интересные моменты...', frac=None)
    selected = pick_highlights(segments, duration, p, opts, source=source, vinfo=vinfo, rep=rep, cache=cache, wav=wav)
    analysis['candidates'] = [{'start': c['start'], 'end': c['end'], 'interest': (c.get('scores') or {}).get('interest'),
                               'engine': c.get('engine')} for c in selected]
    set_state(pk, analysis=analysis)
    if not selected:
        raise ValueError('Не удалось найти подходящие фрагменты. Попробуйте видео длиннее или с более плотной речью.')
    # Replace previous results only now that a new selection exists (a failed rerun never wipes old clips).
    for old in list(p.clips.all()):
        delete_clip_files(old)
        old.delete()
    review = opts['review_first'] and not payload.get('skip_review')
    clips = []
    for i, c in enumerate(selected):
        clips.append(Clip.objects.create(
            project=p, start=c['start'], end=c['end'], score=c['score'], position=i, text=c['text'][:4000],
            reason=c.get('reason', ''), scores=dict(c.get('scores') or {}, engine=c.get('engine'),
                                                   auto_bounds=c.get('auto_bounds') or [c['start'], c['end']]),
            title=f'Highlight {i + 1}', status='candidate' if review else 'queued'))
    if review:
        set_state(pk, status='review', stage='', stage_progress=None,
                  message='Фрагменты найдены. Просмотрите, при необходимости поправьте границы и запустите рендер.')
        return
    _render_clips(p, clips, opts, rep, segments, source, vinfo, mode=None, corner=None)


def _run_render(pk, rep, payload):
    p = Project.objects.get(pk=pk)
    _preflight()
    set_state(pk, status='processing')
    rep.stage('audio', 'Подготовка к повторному монтажу...')
    source = _ensure_source(p, rep)
    vinfo = editor.probe_video(source)
    segments = p.transcript
    if not segments:
        raise RuntimeError('Транскрипт отсутствует — запустите обработку проекта заново.')
    rep.skip('fetch', 'transcribe', 'highlights')
    clips = []
    ids = payload.get('clip_ids') or []
    for c in Clip.objects.filter(project=p, pk__in=ids):
        b = (payload.get('bounds') or {}).get(str(c.pk))
        if b:
            s, e = float(b[0]), float(b[1])
            s, e = max(0.0, s), min(float(p.source_duration or vinfo['duration']), e)
            if payload.get('snap'):
                s, e = snap_bounds(segments, s, e)
            if e - s >= 3:
                c.start, c.end = s, e
                c.text = clip_text(segments, s, e)[:4000]
        c.status = 'queued'
        c.error = ''
        c.save()
        clips.append(c)
    if payload.get('discard_others'):
        for c in Clip.objects.filter(project=p).exclude(pk__in=ids):
            if c.status == 'candidate':
                c.delete()
    if not clips:
        raise ValueError('Не выбрано ни одного клипа.')
    overrides = payload.get('overrides') or {}
    base = exports.resolve(p.options, overrides if payload.get('apply_to_project') else None)
    for c in clips:
        info = c.render_info or {}
        info['opts'] = dict(info.get('opts') or {}, **overrides) if overrides else info.get('opts') or {}
        if payload.get('mode'):
            info['mode'] = payload['mode']
        if payload.get('corner'):
            info['corner'] = payload['corner']
        c.render_info = info
        c.save(update_fields=['render_info'])
    _render_clips(p, clips, base, rep, segments, source, vinfo, mode=None, corner=None)


def _clip_opts(p, clip):
    info = clip.render_info or {}
    return exports.resolve(p.options, info.get('opts'))


def _render_clips(p, clips, base_opts, rep, segments, source, vinfo, mode, corner):
    n = len(clips)
    pk = p.pk
    per = {}
    for c in clips:
        info = c.render_info or {}
        per[c.pk] = {'opts': _clip_opts(p, c), 'mode': info.get('mode') or p.edit_mode,
                     'corner': info.get('corner') or p.facecam_corner}
    plans = {}
    from . import analysis_cache as ac
    p.refresh_from_db()
    src_cam = (p.analysis or {}).get('cam')
    cache = ac.cache_dir(p.source_fingerprint) if p.source_fingerprint else None
    any_analysis = any(v['mode'] in NEEDS_ANALYSIS for v in per.values())
    if any_analysis:
        rep.stage('scenes', 'Анализируем сцены (лица, facecam, тип кадра)...', frac=0.0)
    else:
        rep.skip('scenes')
    for i, c in enumerate(clips):
        rep.check()
        cfg = per[c.pk]
        clip = {'start': c.start, 'end': c.end, 'scores': c.scores or {}}
        if cfg['mode'] in NEEDS_ANALYSIS:
            plans[c.pk] = editor.prepare(source, clip, cfg['mode'], cfg['corner'], cfg['opts'], vinfo, rep=rep,
                                         progress=lambda f, i=i: rep.frac((i + f) / n, f'Анализ сцен: клип {i + 1} из {n}'),
                                         source_cam=src_cam, cache=cache)
        else:
            plans[c.pk] = editor.prepare(source, clip, cfg['mode'], cfg['corner'], cfg['opts'], vinfo, rep=rep)
        rep.frac((i + 1) / n)
    rep.stage('edit', 'Планируем монтаж, субтитры и названия...', frac=0.0)
    for i, c in enumerate(clips):
        rep.check()
        types = [s['type'] for s in plans[c.pk]['segments']]
        try:
            keep = (c.meta or {}).get('short')               # the Shorts title/description is kept across re-edits
            c.meta = metadata.generate(c.text or clip_text(segments, c.start, c.end), scene_types=types)
            if keep:
                c.meta['short'] = keep
            if c.meta.get('titles'):
                c.title = c.meta['titles'][0][:240]
        except Exception:
            log.exception('metadata failed for clip %s', c.pk)
        c.save(update_fields=['meta', 'title', 'updated_at'])
        rep.frac((i + 1) / n)
    rep.stage('render', 'Рендерим клипы...', frac=0.0)
    ok_ids = []
    with tempfile.TemporaryDirectory(prefix='clipforge_') as td:
        for i, c in enumerate(clips):
            rep.check()
            cfg = per[c.pk]
            c.status = 'rendering'
            c.save(update_fields=['status', 'updated_at'])
            temp = Path(td) / f'c{c.pk}'
            temp.mkdir()
            dest = temp / 'out.mp4'
            clip = {'start': c.start, 'end': c.end}
            try:
                info = editor.render(source, clip, segments, dest, temp, cfg['opts'], plans[c.pk], vinfo, cfg['mode'], rep=rep,
                                     on_progress=lambda f, i=i: rep.frac((i + f) / n, f'Рендер клипа {i + 1} из {n}: {int(f * 100)}%'))
            except JobCancelled:
                raise
            except Exception as exc:
                log.warning('new renderer failed for clip %s (%s); falling back to v4 renderer', c.pk, str(exc)[:300])
                try:
                    eff = plans[c.pk].get('mode') or cfg['mode']
                    legacy_mode = eff if eff in LEGACY_MODES else 'gameplay'
                    render(source, clip, segments, dest, temp, legacy_mode, cfg['corner'])
                    info = {'fallback_renderer': 'v4', 'fallback_reason': str(exc)[:300], 'mode': legacy_mode,
                            'segments': [], 'captions': {'style': 'classic', 'enabled': True}}
                except Exception as exc2:
                    c.status, c.error = 'error', str(exc2)[-1500:]
                    c.save(update_fields=['status', 'error', 'updated_at'])
                    continue
            old_video, old_poster = c.video.name, c.poster.name
            stamp = int(time.time())
            with dest.open('rb') as f:
                c.video.save(f'clip_{pk}_{c.position + 1}_{stamp}.mp4', File(f), save=False)
            prev = c.render_info or {}
            info['opts'] = prev.get('opts') or {}
            info['mode'] = cfg['mode']
            c.render_info = info
            c.status, c.error = 'ready', ''
            c.save()
            for old in (old_video, old_poster):  # replaced renders only
                if old and old != c.video.name:
                    try:
                        c.video.storage.delete(old)
                    except Exception:
                        pass
            ok_ids.append(c.pk)
            rep.frac((i + 1) / n)
        rep.stage('validate', 'Проверяем результат...', frac=0.0)
        for i, c in enumerate(Clip.objects.filter(pk__in=ok_ids)):
            rep.check()
            sibs = [(o.text, (o.start, o.end)) for o in Clip.objects.filter(project=p).exclude(pk=c.pk) if o.text]
            try:
                cr = (c.render_info or {}).get('captions')
                copts = per[c.pk]['opts'] if c.pk in per else base_opts
                expect = {'size': exports.dims(copts), 'fps': int(copts['fps']) if copts['fps'] != 'source' else None,
                          'lufs': (exports.LOUDNESS.get(copts.get('loudness')) or {}).get('I')}
                c.quality = quality.validate_clip(c.video.path, c.end - c.start, source_has_audio=vinfo['has_audio'],
                                                  caption_report=cr, text=c.text, siblings=sibs,
                                                  clip_min=p.clip_min, clip_max=p.clip_max, span=(c.start, c.end),
                                                  expect=None if (c.render_info or {}).get('fallback_renderer') else expect)
                pt = Path(td) / f'poster_{c.pk}.jpg'
                if quality.make_poster(c.video.path, pt, at=min(1.0, (c.end - c.start) / 3)):
                    with pt.open('rb') as f:
                        c.poster.save(f'poster_{pk}_{c.position + 1}_{int(time.time())}.jpg', File(f), save=False)
            except JobCancelled:
                raise
            except Exception as exc:
                log.warning('validation failed for clip %s: %s', c.pk, exc)
                c.quality = {'score': None, 'ok': True, 'checks': [{'id': 'validator', 'status': 'warn',
                                                                    'label': 'Автопроверка', 'detail': 'не удалось выполнить'}]}
            c.save()
            rep.frac((i + 1) / max(1, len(ok_ids)))
    _short_meta(p, ok_ids, rep)
    failed = [c for c in clips if c.status == 'error']
    if not ok_ids:
        raise RuntimeError('Ни один клип не удалось отрендерить: ' + (failed[0].error[-400:] if failed else ''))
    msg = 'Готово!' if not failed else f'Готово: {len(ok_ids)} из {n} клипов, у остальных ошибка рендера.'
    set_state(pk, status='done', progress=100, stage='', stage_progress=None, message=msg, error='')


def _short_meta(p, ok_ids, rep):
    """Shorts title + description for every freshly rendered clip. Reuses stored results when the clip's content did
    not change; the AI provider is time-limited and falls back locally, so this step can never stall the job."""
    from . import shorts_meta
    ids = list(ok_ids)
    if not ids:
        return
    rep.frac(0.0, 'Пишем названия и описания для Shorts...')
    p.refresh_from_db()
    for i, c in enumerate(Clip.objects.filter(pk__in=ids)):
        rep.check()
        short = shorts_meta.ensure_for_clip(p, c)
        if short:
            c.title = short['title'][:240]
            c.save(update_fields=['meta', 'title', 'updated_at'])
        rep.frac((i + 1) / len(ids), f'Названия и описания: {i + 1} из {len(ids)}')


def process_project(pk):
    """v4-compatible entry point: run the whole job synchronously in the calling thread."""
    from .jobs import _run_job
    close_old_connections()
    try:
        _run_job('full', pk, {})
    finally:
        close_old_connections()
