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


class _WhisperCache:
    """Compatibility handle: Whisper models now live in the resource manager's model registry."""

    def clear(self):
        from . import resources
        resources.manager().unload_models(group='whisper', why='cleared')


_model_cache = _WhisperCache()

# resident memory of a loaded model (MB, float16 on CUDA / int8 on CPU) and extra working memory per batch item
WHISPER_MB = {'tiny': 250, 'base': 350, 'small': 900, 'medium': 2100, 'large-v1': 3900, 'large-v2': 3900,
              'large-v3': 3900, 'large-v3-turbo': 1900, 'turbo': 1900, 'distil-large-v3': 1700}


def whisper_memory(name, device, compute, batch):
    """(model MB, working MB) estimates for the resource manager."""
    base = WHISPER_MB.get(name, 2100 if 'medium' in name else 3900 if 'large' in name else 900)
    if device == 'cuda':
        model = int(base * (0.6 if 'int8' in compute else 1.0))
        return model, 300 + int(base * 0.06) * max(1, batch or 1)
    return int(base * 0.6), 400 + int(base * 0.1)


def _get_model(name, device, compute):
    """Create a WhisperModel (the registry in resources.py caches it, see _whisper_model)."""
    from . import gpu_libs
    gpu_libs.register()              # pip-installed CUDA 12 DLLs (install_gpu.bat) on Windows
    from faster_whisper import WhisperModel
    return WhisperModel(name, device=device, compute_type=compute)


def _whisper_model(name, device, compute):
    """Lazy, shared, pinned while in use; one Whisper size resident at a time; unloaded when memory is needed."""
    from . import resources
    mb, _ = whisper_memory(name, device, compute, 0)
    gpu_known = resources.manager().facts()['vram_total'] > 0
    return resources.manager().model(
        ('whisper', name, device, compute), lambda: _get_model(name, device, compute), device=device,
        vram_mb=mb if (device == 'cuda' and gpu_known) else 0, ram_mb=0 if device == 'cuda' else mb,
        group='whisper', group_max=1)


def _whisper_lease(name, dev, comp, batch, cancel):
    """Working memory + CPU for one transcription. Waits (cancellable) while another job uses the GPU;
    Unavailable / timeout -> the caller decodes on the CPU instead."""
    from . import resources
    mgr = resources.manager()
    _, work = whisper_memory(name, dev, comp, batch)
    gpu_known = mgr.facts()['vram_total'] > 0
    cpu = 1 if dev == 'cuda' else min(4, mgr.capacity()['cpu'])
    return mgr.acquire(label=f'Whisper {name} ({dev})', cpu=cpu, vram_mb=work if dev == 'cuda' and gpu_known else 0,
                       ram_mb=0 if dev == 'cuda' else work, timeout=float(os.getenv('CLIPFORGE_GPU_WAIT', '900')),
                       cancel=cancel)


def transcribe(path, lang, rep=None, duration=None):
    """faster-whisper. Device comes from diagnostics (auto: CUDA only if the probe passed, else CPU).
    Any CUDA failure (missing cuDNN/cuBLAS, OOM...) transparently restarts on CPU. Reports real progress."""
    from . import diagnostics
    from .procutil import JobCancelled
    from . import resources
    model_name = os.getenv('WHISPER_MODEL', 'small')
    device, compute, note = diagnostics.whisper_settings()
    attempts = [(device, compute)] + ([('cpu', 'int8')] if device == 'cuda' else [])
    mgr = resources.manager()
    oom_retry = [True]
    for dev, comp in attempts:
        hb = _Heartbeat(rep)
        try:
            batch = whisper_batch_size(dev)
            try:
                lease = _whisper_lease(model_name, dev, comp, batch, getattr(rep, 'cancelled', None))
            except resources.Unavailable as exc:
                if dev != 'cuda':
                    raise
                log.warning('Whisper: GPU busy or too small (%s); transcribing on the CPU', exc)
                mgr.note('cpu_fallback', 'Whisper', str(exc))
                continue
            with lease:
                if not _whisper_cached(model_name):
                    hb.phase(f'Скачиваем модель Whisper «{model_name}» (один раз, ~0.5 ГБ; прогресс в окне сервера)')
                else:
                    hb.phase(f'Загружаем модель Whisper «{model_name}» ({dev.upper()})')
                t0 = time.monotonic()
                with _whisper_model(model_name, dev, comp) as model:
                    log.info('Whisper %s ready on %s/%s in %.1f s', model_name, dev, comp, time.monotonic() - t0)
                    out, info, used = _transcribe_with(model, path, lang, rep, duration, dev, batch, hb, mgr,
                                                       oom_retry)
            took = time.monotonic() - t0
            log.info('Whisper done: %d segments, %.0f s of audio in %.0f s (%.1fx realtime, %s, batch %s)', len(out),
                     duration or 0, took, (duration or 0) / max(took, 1), dev, used)
            transcribe.last_info = {'model': model_name, 'device': dev, 'compute': comp, 'batch': used,
                                    'detected_language': getattr(info, 'language', None), 'requested_lang': lang,
                                    'seconds': round(took)}
            return out
        except JobCancelled:
            raise
        except Exception as exc:
            log.warning('Whisper failed on %s: %s', dev, str(exc)[:300])
            if dev == 'cuda':
                if resources.is_oom(exc):            # out of memory is not a broken GPU: only this run goes to CPU
                    mgr.on_oom('Whisper', exc)
                else:
                    diagnostics.mark_whisper_gpu_failed(str(exc)[:200])
                _model_cache.clear()
                continue
            raise
        finally:
            hb.stop()
    return []


transcribe.last_info = {}


CUDA_LIB_ERROR = re.compile(r'(cublas|cudnn|cudart|nvrtc)\w*\.(dll|so)|is not found or cannot be loaded|'
                            r'Could not (locate|load) (library )?\S*cu', re.I)


def _whisper_cached(name):
    """Is the faster-whisper model already in the Hugging Face cache? (unknown -> True: no scary message)"""
    try:
        from faster_whisper.utils import _MODELS
        from huggingface_hub import try_to_load_from_cache
        repo = _MODELS.get(name)
        if not repo:
            return True
        return isinstance(try_to_load_from_cache(repo, 'model.bin'), str)
    except Exception:
        return True


class _Heartbeat:
    """Until Whisper yields its first segment the stage has no measurable progress: keep telling the user what is
    happening and for how long, so a long model download / VAD pass never looks like a hang."""

    def __init__(self, rep, every=5.0):
        import threading
        self.rep, self.every = rep, every
        self.started = time.monotonic()
        self.text = ''
        self.done = threading.Event()
        self.thread = None
        if rep is not None:
            self.thread = threading.Thread(target=self._loop, name='clipforge-whisper-heartbeat', daemon=True)
            self.thread.start()

    def phase(self, text):
        self.text = text
        log.info('Whisper: %s', text)
        self._say()

    def first_segment(self):
        if not self.done.is_set():
            log.info('Whisper: first segment after %.0f s', time.monotonic() - self.started)
        self.done.set()

    def _say(self):
        if self.rep is None or self.done.is_set() or not self.text:
            return
        el = int(time.monotonic() - self.started)
        try:
            self.rep.note(f'{self.text}… прошло {el // 60}:{el % 60:02d}')
        except Exception:
            pass

    def _loop(self):
        from django.db import connection
        try:
            while not self.done.wait(self.every):
                self._say()
        finally:
            connection.close()

    def stop(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


def _transcribe_with(model, path, lang, rep, duration, dev, batch, hb, mgr, oom_retry):
    """Batched decoding first; a CUDA out-of-memory frees idle models and retries once with half the batch;
    anything else (old faster-whisper...) falls back to sequential decoding."""
    from . import resources
    from .procutil import JobCancelled
    while batch:
        try:
            hb.phase(f'Ищем речь во всём файле и распознаём пакетами по {batch} ({dev.upper()}); '
                     'первые проценты появятся после первого пакета')
            out, info = _run_whisper(model, path, lang, rep, duration, dev, batch, hb)
            return out, info, batch
        except JobCancelled:
            raise
        except Exception as exc:
            if CUDA_LIB_ERROR.search(str(exc)):
                raise                     # missing cuBLAS/cuDNN: the sequential path would fail too -> CPU
            if dev == 'cuda' and resources.is_oom(exc) and oom_retry[0] and batch > 1:
                oom_retry[0] = False
                mgr.on_oom('Whisper batch', exc)
                batch = max(1, batch // 2)
                log.warning('Whisper out of GPU memory; retrying with batch %d', batch)
                continue
            log.warning('batched Whisper failed on %s (%s); sequential decoding', dev, str(exc)[:200])
            break
    hb.phase(f'Распознаём речь ({dev.upper()}); первые проценты появятся через минуту-две')
    out, info = _run_whisper(model, path, lang, rep, duration, dev, 0, hb)
    return out, info, 0


def whisper_batch_size(device):
    """WHISPER_BATCH: 'auto' (default) = 16 on CUDA, off on CPU; '0' = off; a number = that batch size."""
    v = os.getenv('WHISPER_BATCH', 'auto').strip().lower()
    if v in ('auto', ''):
        return 16 if device == 'cuda' else 0
    try:
        return max(0, min(64, int(v)))
    except ValueError:
        return 0


def _run_whisper(model, path, lang, rep, duration, dev, batch, hb=None):
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
        if hb is not None:
            hb.first_segment()
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


class BackgroundScreenScan:
    """The whole-video screen scan runs in a thread from the start of the job, next to audio extraction, Whisper
    and the sound classifier (it only needs the source file). Progress shows up as its own operation; a partial
    result is cached, so a pause, crash or cancel continues where it stopped."""

    def __init__(self, source, duration, cache, rep):
        import threading
        self.result = None
        self.error = None
        self.info = None
        self.done = threading.Event()
        self._stop = threading.Event()
        self._rep = rep
        self._args = (source, duration, Path(cache))
        from . import resources
        self._job = resources.manager().current_job()     # leases of the scan belong to this job
        rep.op('screen', 'Анализ экрана: подготовка...', frac=0.0)
        threading.Thread(target=self._run, daemon=True, name='screen-scan').start()

    def _progress(self, frac, info):
        self.info = info
        how = 'NVDEC' if info.get('hw') else ('ключевые кадры' if info.get('mode') == 'key' else 'CPU, 1 кадр/с')
        self._rep.op('screen', f'Анализ экрана ({how}): {int(frac * 100)}%', frac=frac)

    def _run(self):
        from . import analysis_cache as ac
        from . import resources
        source, duration, cache = self._args
        try:
            with resources.manager().job_context(self._job):
                sp = self._scan(source, duration, cache)
            if sp is not None:
                ac.save_npz(cache / 'screen.npz', **motion_analysis.screen_to_arrays(sp))
            self.result = sp
        except BaseException as exc:          # reported to the main thread in wait()
            self.error = exc
        finally:
            self._rep.op('screen', done=True)
            self.done.set()

    def _scan(self, source, duration, cache):
        return motion_analysis.scan_screen(source, duration, cancel=lambda: self._stop.is_set() or
                                           self._rep.cancelled(), progress=self._progress,
                                           part_path=str(cache / 'screen.part.npz'))

    def stop(self):
        self._stop.set()
        self.done.wait(20)

    def wait(self, rep):
        while not self.done.wait(0.5):
            rep.check()
        if self.error is not None:
            raise self.error
        return self.result


def highlight_signals(source, duration, opts, vinfo=None, rep=None, cache=None, wav=None, screen_job=None):
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
            if screen is None and screen_job is not None:
                if rep and not screen_job.done.is_set():
                    rep.frac(0.65, 'Смотрим на экран: смерти, провалы, вспышки, смены сцен (идёт параллельно)...')
                screen = screen_job.wait(rep)
            elif screen is None:
                if rep:
                    rep.frac(0.65, 'Смотрим на экран: смерти, провалы, вспышки, смены сцен...')
                screen = motion_analysis.scan_screen(
                    source, duration, cancel=cancel,
                    progress=(lambda f, i: rep.frac(0.65 + 0.2 * f, f'Смотрим на экран: {int(f * 100)}%'))
                    if rep else None, part_path=str(Path(cache) / 'screen.part.npz') if cache else None)
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
            mp = motion_analysis.analyze_spans(
                source, spans, vinfo['width'], vinfo['height'], cancel=cancel,
                progress=(lambda f: rep.frac(0.85 + 0.15 * f, f'Оцениваем динамику геймплея у лучших кандидатов: '
                                                              f'{int(f * 100)}%')) if rep else None)
            highlight_signals.last_motion[str(source)] = mp     # reused by the Hook Engine (no second decode)
            return mp
    highlight_signals.last_screen = screen
    return audio, motion, notes


highlight_signals.last_screen = None
highlight_signals.last_motion = {}           # source path -> MotionProfile (kept per source: jobs may overlap)


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


def pick_highlights(segments, duration, p, opts, source=None, vinfo=None, rep=None, cache=None, wav=None,
                    screen_job=None, count=None):
    """New ranker first; the proven v4 algorithm is the fallback when it errors or finds nothing."""
    if opts['highlight_engine'] == 'smart':
        screen = None
        try:
            audio, motion, _ = highlight_signals(source, duration, opts, vinfo, rep, cache=cache, wav=wav,
                                                 screen_job=screen_job)
            screen = highlight_signals.last_screen
        except JobCancelled:
            raise
        except Exception:
            log.exception('highlight signals failed')
            audio = motion = None
        try:
            sel = highlights.find_highlights(segments, duration, p.clip_min, p.clip_max, count or p.clip_count,
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
    sel = choose(ranked, count or p.clip_count)
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
    rep.checkpoint('source', name=p.source.name)
    rep.stage('audio', 'Анализируем видео и аудио...')
    duration = probe(source)
    vinfo = editor.probe_video(source)
    fp = p.source_fingerprint or fingerprint(source)
    set_state(pk, source_duration=duration, source_fingerprint=fp, status='processing')
    p.refresh_from_db()
    cache, wav, analysis = _source_analysis(p, source, duration, vinfo, fp, rep)
    screen_job = None
    if (opts['highlight_engine'] == 'smart' and opts.get('highlight_motion', True) and cache
            and not (Path(cache) / 'screen.npz').is_file() and not _resumed_clips(p, rep)):
        screen_job = BackgroundScreenScan(source, duration, cache, rep)
    try:
        _run_full_rest(p, pk, rep, payload, opts, source, duration, vinfo, fp, cache, wav, analysis, screen_job)
    finally:
        if screen_job is not None:
            screen_job.stop()


def _run_full_rest(p, pk, rep, payload, opts, source, duration, vinfo, fp, cache, wav, analysis, screen_job):
    segments = _get_transcript(p, source, duration, fp, rep, wav=wav)
    p.refresh_from_db()
    rep.checkpoint('transcript', segments=len(segments or []))
    resumed = _resumed_clips(p, rep)
    if resumed:                       # this job already chose and created its clips before it was interrupted
        rep.skip('highlights')
        _render_clips(p, resumed, opts, rep, segments, source, vinfo, mode=None, corner=None)
        return
    analysis['language'] = (p.transcript_meta or {}).get('detected_language') or p.language
    rep.stage('highlights', 'Ищем интересные моменты...', frac=None)
    pool = min(8, p.clip_count * 2) if opts.get('viral_director') == 'auto' else None   # choose the best of more
    selected = pick_highlights(segments, duration, p, opts, source=source, vinfo=vinfo, rep=rep, cache=cache, wav=wav,
                               screen_job=screen_job, count=pool)
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
    _find_hooks(p, clips, segments, source, vinfo, opts, rep)
    clips = _viral_director(p, clips, segments, opts, rep, review=review)
    rep.checkpoint('clips', ids=[c.pk for c in clips], review=bool(review))
    if review:
        set_state(pk, status='review', stage='', stage_progress=None,
                  message='Фрагменты найдены. Просмотрите, при необходимости поправьте границы и запустите рендер.')
        return
    _render_clips(p, clips, opts, rep, segments, source, vinfo, mode=None, corner=None)


def _find_hooks(p, clips, segments, source, vinfo, opts, rep):
    """Hook Engine (2A): variants of the first seconds for every new clip, from the transcript and the cached
    audio / sound / screen analysis plus one short motion probe. 'auto' applies the recommended variant."""
    from . import hooks, resources
    mode = opts.get('hook_engine', 'suggest')
    if mode == 'off' or not clips:
        return
    rep.frac(0.95, 'Ищем самое цепляющее начало (hook) для каждого клипа...')
    try:
        with resources.cpu_lease('Hook Engine', 1):
            wins = hooks.candidate_windows(segments, [(c.start, c.end) for c in clips], p.clip_min, p.clip_max)
            sig = hooks.load_signals(p, segments, source, vinfo, wins, cancel=rep.cancelled,
                                     motion=highlight_signals.last_motion.pop(str(source), None))
            for c in clips:
                h = hooks.analyze(sig, c.start, c.end, p.clip_min, p.clip_max, p.source_duration)
                rec = hooks.variant(h, h['recommended'])
                if mode == 'auto' and rec and rec['kind'] != 'original':
                    h['selected'] = rec['id']
                    if rec['kind'] == 'start':
                        c.start, c.end = rec['start'], rec['end']
                        c.text = clip_text(segments, c.start, c.end)[:4000]
                c.meta = dict(c.meta or {}, hooks=h)
                c.save()
    except JobCancelled:
        raise
    except Exception:
        log.exception('hook engine failed; clips keep their start')


def smart_edit_mode(p, c):
    """Smart Editing mode of a clip: its own choice (manual control) or the project option."""
    e = (c.meta or {}).get('edit') or {}
    return e.get('mode') or exports.resolve(p.options, (c.render_info or {}).get('opts')).get('smart_edit', 'off')


def _edit_plan(p, c, segments, sig=None):
    """Smart Editing plan for the clip's current range (stored in meta['edit'] with the manual choices)."""
    from . import hooks, smartedit
    mode = smart_edit_mode(p, c)
    e = dict((c.meta or {}).get('edit') or {})
    if mode == 'off' or mode not in smartedit.MODES:
        return None
    sig = sig or hooks.load_signals(p, segments)
    opts = exports.resolve(p.options, (c.render_info or {}).get('opts'))
    pl = smartedit.plan(sig, c.start, c.end, mode, p.clip_min, restore=set(e.get('restore') or []),
                        **smartedit.edit_args(e, opts))
    e.update(plan=pl)
    c.meta = dict(c.meta or {}, edit=e)
    Clip.objects.filter(pk=c.pk).update(meta=c.meta)
    return pl


def build_edits(segments, start, end, plan=None, hooks_data=None, selected=None):
    """Smart Editing plan + Hook teaser for a clip range -> {'transform', 'info'} or None (nothing to change).
    Shared by the render path and the Studio preview."""
    from . import hooks, smartedit, textutil
    pl = plan if plan and (len(plan['keep']) > 1 or plan['keep'][0] != list(plan['clip']) or plan.get('zooms')) \
        else None
    v = hooks.variant(hooks_data or {}, selected)
    teaser = v['teaser'] if v and v['kind'] == 'teaser' and start - 0.05 <= v['teaser'][0] and \
        v['teaser'][1] <= end + 0.05 else None
    if not pl and not teaser:
        return None
    info = {}
    if pl:
        info['edit'] = {'mode': pl['mode'], 'removed': pl['stats']['removed'], 'cuts': pl['stats']['cuts'],
                        'by_reason': pl['stats']['by_reason'], 'original': pl['stats']['original'],
                        'result': pl['stats']['result'], 'pauses': pl['stats'].get('pauses', 0),
                        'zooms': pl['stats'].get('zooms', 0)}
    if teaser:
        info['hook'] = {'kind': 'teaser', 'variant': v['id'], 'teaser': teaser, 'score': v['score'],
                        'renderer': 'timeline'}
    words = [w for w in textutil.flat_words(segments or []) if teaser and w['end'] > teaser[0] and w['start'] < teaser[1]]

    def transform(tl):
        if pl:
            tl = smartedit.to_timeline(tl, pl, start, pl['mode'])
        if teaser:
            tl = hooks.prepend_teaser(tl, teaser, start, src_words=words if pl else None)
        return tl
    return {'transform': transform, 'info': info}


def _clip_edits(p, c, cfg, segments):
    """What the timeline path must change in this clip: Smart Editing cuts and/or the chosen Hook teaser.
    -> {'transform': fn(Timeline) -> Timeline, 'info': {...}} or None (classic render, as before)."""
    h = (c.meta or {}).get('hooks') or {}
    return build_edits(segments, c.start, c.end, _edit_plan(p, c, segments), h, h.get('selected'))


def _viral_director(p, clips, segments, opts, rep, review=False, apply=None):
    """Viral Director (2C): score Hook x Smart Editing combinations of every clip (plans only, no renders), mark
    duplicates, rank. 'auto': keep the best clip_count clips and apply each one's best combination. -> clips."""
    from . import hooks, resources, viral
    mode = opts.get('viral_director', 'rank')
    if mode == 'off' or not clips:
        return clips
    apply = (mode == 'auto') if apply is None else apply
    rep.frac(0.97, 'Viral Director: сравниваем варианты начала и монтажа, отбираем лучшие клипы...')
    try:
        with resources.cpu_lease('Viral Director', 1):
            sig = hooks.load_signals(p, segments)
            texts = {c.pk: c.text or clip_text(segments, c.start, c.end) for c in clips}
            reports = {}
            for c in clips:
                h = (c.meta or {}).get('hooks')
                clip = tuple(h['clip']) if h and h.get('clip') else (c.start, c.end)
                others = [t for k, t in texts.items() if k != c.pk]
                r = viral.direct(sig, clip, h, c.scores or {}, others, p.clip_min, p.clip_max)
                auto = (c.render_info or {}).get('auto')
                if auto:
                    r['reasons'].append({'points': 0, 'text': f'AI Auto: {auto.get("label") or auto.get("mode")}'})
                reports[c.pk] = r
            ranked = viral.rank([{'id': c.pk, 'range': (c.start, c.end), 'text': texts[c.pk],
                                  'score': reports[c.pk]['score']} for c in clips])
            keep = set(viral.select(ranked, p.clip_count)) if apply and not review else {c.pk for c in clips}
            by_id = {c.pk: c for c in clips}
            out = []
            for pos, it in enumerate(ranked):
                c = by_id[it['id']]
                if c.pk not in keep:
                    c.delete()                                   # not rendered yet: nothing on disk
                    continue
                r = dict(reports[c.pk], rank=it['rank'], duplicate_of=it['duplicate_of'], applied=False)
                meta = dict(c.meta or {})
                if apply and not review:
                    combo = r['combo']
                    h = meta.get('hooks')
                    v = hooks.variant(h, combo['hook']) if h else None
                    if h is not None:
                        h['selected'] = None if not v or v['kind'] == 'original' else v['id']
                        meta['hooks'] = h
                    if v and v['kind'] == 'start':
                        c.start, c.end = v['start'], v['end']
                        c.text = clip_text(segments, c.start, c.end)[:4000]
                    meta['edit'] = {'mode': combo['edit'], 'restore': []}
                    r['applied'] = True
                meta['viral'] = r
                c.meta = meta
                c.position = len(out)
                c.save()
                out.append(c)
            return out
    except JobCancelled:
        raise
    except Exception:
        log.exception('viral director failed; clips are kept as chosen by the ranker')
        return clips


def _resumed_clips(p, rep):
    """Clips this job created before an interruption (checkpoint 'clips'), if they all still exist."""
    cp = rep.get('clips')
    if not cp or cp.get('review'):
        return None
    clips = list(Clip.objects.filter(project=p, pk__in=cp.get('ids') or []).order_by('position'))
    return clips if clips and len(clips) == len(cp['ids']) else None


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
    rehook = []
    ids = payload.get('clip_ids') or []
    for c in Clip.objects.filter(project=p, pk__in=ids):
        b = (payload.get('bounds') or {}).get(str(c.pk))
        if b and not payload.get('hook'):              # bounds edited by hand: new hook variants, no old choice
            rehook.append(c)
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
    if rehook:
        opts = exports.resolve(p.options)
        if opts.get('hook_engine') != 'off':
            _find_hooks(p, rehook, segments, source, vinfo, dict(opts, hook_engine='suggest'), rep)
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


KEEP_META = ('short', 'hooks', 'edit', 'viral', 'history', 'original')


def _render_clips(p, clips, base_opts, rep, segments, source, vinfo, mode, corner):
    n = len(clips)
    rendered = set((rep.get('rendered') or {}).get('ids', []))       # finished before an interruption: keep them
    kept = {c.pk for c in clips if c.pk in rendered and c.video and c.status == 'ready'}
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
        if c.pk in kept:
            continue
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
        if c.pk in kept:
            continue
        types = [s['type'] for s in plans[c.pk]['segments']]
        try:
            # kept across re-edits: Shorts text, Hook variants + choice, Smart Editing, Viral report, Studio history
            kept_meta = {k: v for k, v in (c.meta or {}).items() if k in KEEP_META and v is not None}
            c.meta = metadata.generate(c.text or clip_text(segments, c.start, c.end), scene_types=types)
            c.meta.update(kept_meta)
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
            if c.pk in kept:
                ok_ids.append(c.pk)
                continue
            cfg = per[c.pk]
            c.status = 'rendering'
            c.save(update_fields=['status', 'updated_at'])
            temp = Path(td) / f'c{c.pk}'
            temp.mkdir()
            dest = temp / 'out.mp4'
            clip = {'start': c.start, 'end': c.end}
            progress = (lambda f, i=i: rep.frac((i + f) / n, f'Рендер клипа {i + 1} из {n}: {int(f * 100)}%'))
            info = None
            edits = _clip_edits(p, c, cfg, segments)               # Smart Editing (2B) / Hook teaser (2A)
            if edits or getattr(settings, 'TIMELINE_RENDERER', False):  # timeline path: edits, or the experiment
                info = _timeline_render(p, c, plans[c.pk], cfg, vinfo, segments, dest, temp, rep, progress,
                                        transform=edits and edits['transform'])
                if info is not None and edits:
                    info.update(edits['info'])
            try:
                if info is None:
                    info = editor.render(source, clip, segments, dest, temp, cfg['opts'], plans[c.pk], vinfo, cfg['mode'],
                                         rep=rep, on_progress=progress)
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
            if edits and info.get('renderer') != 'timeline':           # edits failed: plain clip, say why
                info['edit_error'] = 'монтаж/тизер не применён (ошибка рендера, подробности в логе); клип без изменений'
            old_video, old_poster = c.video.name, c.poster.name
            stamp = int(time.time())
            with dest.open('rb') as f:
                c.video.save(f'clip_{pk}_{c.position + 1}_{stamp}.mp4', File(f), save=False)
            prev = c.render_info or {}
            info['opts'] = prev.get('opts') or {}
            info['mode'] = cfg['mode']
            if not info.get('fallback_renderer') and 'timeline' not in info:
                info['timeline'] = _timeline_snapshot(p, c, plans[c.pk], cfg['opts'], vinfo, info, segments)
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
            rep.checkpoint_add('rendered', c.pk)          # safe point: a pause here loses no rendered clip
        rep.stage('validate', 'Проверяем результат...', frac=0.0)
        for i, c in enumerate(Clip.objects.filter(pk__in=ok_ids)):
            rep.check()
            sibs = [(o.text, (o.start, o.end)) for o in Clip.objects.filter(project=p).exclude(pk=c.pk) if o.text]
            try:
                cr = (c.render_info or {}).get('captions')
                copts = per[c.pk]['opts'] if c.pk in per else base_opts
                expect = {'size': exports.dims(copts), 'fps': int(copts['fps']) if copts['fps'] != 'source' else None,
                          'lufs': (exports.LOUDNESS.get(copts.get('loudness')) or {}).get('I')}
                c.quality = quality.validate_clip(c.video.path, (c.render_info or {}).get('output_duration') or c.end - c.start,
                                                  source_has_audio=vinfo['has_audio'],
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


def _timeline_snapshot(p, c, plan, opts, vinfo, info, segments):
    """Timeline IR of the clip exactly as it was rendered (stored in render_info['timeline']).
    Describes, never drives, the render: any problem here is logged and the clip is kept as it is."""
    try:
        from .timeline import adapters
        return adapters.clip_timeline_dict(
            project_id=p.pk, clip_id=c.pk, source_name=p.source.name, fingerprint=p.source_fingerprint,
            clip_start=c.start, clip_end=c.end, plan=plan, opts=opts, vinfo=vinfo, render_info=info,
            transcript=segments, title=c.title)
    except Exception:
        log.exception('timeline snapshot failed for clip %s (the render itself is fine)', c.pk)
        return None


def clip_ir(p, c, plan, cfg, vinfo, segments, start=None, end=None):
    """Timeline IR of a clip as the renderer would build it (also used for Studio previews). -> (tl, pre_info)."""
    from .timeline import adapters
    opts = cfg['opts']
    segs = plan['segments']
    fps = None
    if opts['fps'] != 'source' and int(opts['fps']) < vinfo['fps'] - 0.5:
        fps = int(opts['fps'])
    target = exports.LOUDNESS.get(opts.get('loudness', 'stream')) if vinfo.get('has_audio', True) else None
    pre = {'fps': fps or 'source', 'transition': opts['transition'] if len(segs) > 1 else 'none',
           'mode': cfg['mode'], 'auto': plan.get('auto'), 'notes': plan.get('notes', []),
           'loudness': {'target': target['I'], 'mode': 'two-pass'} if target else {'target': None}}
    tl = adapters.clip_timeline(project_id=p.pk, clip_id=c.pk, source_name=p.source.name,
                                fingerprint=p.source_fingerprint, clip_start=c.start if start is None else start,
                                clip_end=c.end if end is None else end, plan=plan, opts=opts, vinfo=vinfo,
                                render_info=pre, transcript=segments, title=c.title)
    return tl, pre


def _timeline_render(p, c, plan, cfg, vinfo, segments, dest, temp, rep, on_progress, transform=None):
    """Experimental path (settings.TIMELINE_RENDERER): plan -> Timeline IR -> timeline renderer.
    -> render_info dict, or None when anything goes wrong (the caller then renders with the classic renderer)."""
    opts = cfg['opts']
    try:
        from .timeline import renderer
        from .timeline.serialize import to_dict
        segs = plan['segments']
        tl, pre = clip_ir(p, c, plan, cfg, vinfo, segments)
        if transform is not None:
            tl = transform(tl)
        r = renderer.render(tl, dest, renderer.RenderSettings(quality=opts['quality'], encoder=opts['encoder'],
                                                              workdir=Path(temp) / 'tl'),
                            on_progress=on_progress, cancel=rep.cancelled)
    except JobCancelled:
        raise
    except Exception as exc:
        log.warning('timeline renderer failed for clip %s (%s); using the classic renderer', c.pk, str(exc)[:300])
        return None
    loud = (r.get('audio') or {}).get('loudness') or []
    caps = (r.get('captions') or {}).get('items') or []
    return {'renderer': 'timeline', 'encoder': r['encoder'], 'encoder_note': r['encoder_note'],
            'encoder_fallback': bool(r.get('encoder_fallback')), 'graph_chars': r.get('graph_chars'),
            'size': r['size'], 'segments': editor._public_segments(segs),
            'captions': dict(caps[0] if caps else {}, enabled=bool(opts['captions']), style=opts['subtitle_style'],
                             censor=bool(opts.get('caption_censor'))),
            'notes': plan.get('notes', []), 'fps': pre['fps'], 'transition': pre['transition'],
            'loudness': ({k: v for k, v in loud[0].items() if k != 'item'} if loud else pre['loudness']),
            'auto': plan.get('auto'), 'cam': plan.get('cam') and {k: (round(v, 3) if isinstance(v, float) else v)
                                                                 for k, v in plan['cam'].items()},
            'timeline': to_dict(tl), 'output_duration': tl.duration,
            'timeline_render': {k: r[k] for k in ('scenes', 'seconds', 'rendered_scenes', 'timeline_hash', 'version')}}


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
