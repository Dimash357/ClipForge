"""Local text-to-speech and exact line timings.

Providers (none requires a paid API):
* piper      - Piper TTS (pip install piper-tts; voices = <name>.onnx + <name>.onnx.json in the voices folder,
               e.g. ru_RU-dmitri-medium, ru_RU-irina-medium, en_US-ryan-high). CPU, fast, ~60 MB per voice.
* openai_tts - any local server with the OpenAI speech API (/v1/audio/speech), e.g. Kokoro-FastAPI,
               openedai-speech, AllTalk; URL + model + voice names in the settings.
If no provider works, synthesis fails with a clear error - nothing is faked.

align(): word timings of a synthesized line - Whisper (already installed for Stream Clipper) on the short clip,
mapped onto the script's words; without Whisper, words are spread over the speech span measured from the audio
(leading / trailing silence trimmed).
"""
import json
import logging
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

log = logging.getLogger('studio.tts')


class TTSError(RuntimeError):
    pass


def wav_duration(path):
    with wave.open(str(path), 'rb') as w:
        return w.getnframes() / float(w.getframerate())


# ------------------------------------------------------------------ Piper
def piper_voices(folder):
    """{voice id: {'language', 'path', 'quality', 'speakers'}} of the voices installed in `folder`."""
    out = {}
    d = Path(folder) if folder else None
    if not d or not d.is_dir():
        return out
    for onnx in sorted(d.glob('*.onnx')):
        meta = onnx.with_suffix('.onnx.json')
        lang, quality, speakers = onnx.stem.split('-')[0], '', 1
        if meta.is_file():
            try:
                m = json.loads(meta.read_text(encoding='utf-8'))
                lang = (m.get('language') or {}).get('code') or lang
                quality = (m.get('audio') or {}).get('quality') or ''
                speakers = int(m.get('num_speakers') or 1)
            except (ValueError, OSError):
                pass
        out[onnx.stem] = {'language': lang, 'path': str(onnx), 'quality': quality, 'speakers': speakers,
                          'has_config': meta.is_file()}
    return out


def piper_available():
    try:
        import piper  # noqa: F401
        return True, ''
    except Exception as exc:
        return False, f'piper-tts не установлен ({str(exc)[:80]})'


def _piper(text, model_path, out, speed=1.0):
    """Synthesize with the installed piper package (1.2 or 1.3 API), CLI as a last resort."""
    try:
        from piper import PiperVoice
        voice = PiperVoice.load(model_path)
        with wave.open(str(out), 'wb') as w:
            if hasattr(voice, 'synthesize_wav'):                   # piper-tts >= 1.3
                try:
                    from piper import SynthesisConfig
                    voice.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=1.0 / max(0.5, speed)))
                except ImportError:
                    voice.synthesize_wav(text, w)
            else:                                                  # piper-tts 1.2
                voice.synthesize(text, w, length_scale=1.0 / max(0.5, speed))
        return
    except ImportError:
        pass
    except Exception as exc:
        log.warning('piper API failed (%s); trying the CLI', str(exc)[:200])
    p = subprocess.run([sys.executable, '-m', 'piper', '-m', str(model_path), '-f', str(out)], input=text,
                       capture_output=True, text=True, encoding='utf-8', timeout=300)
    if p.returncode or not Path(out).is_file():
        raise TTSError(f'Piper: {(p.stderr or p.stdout)[-300:]}')


# ------------------------------------------------------------------ OpenAI-compatible local server
def _openai(text, voice, out, cfg, post=None):
    if post is None:
        import requests
        post = requests.post
    url = cfg.get('tts_url', '').rstrip('/') + '/audio/speech'
    r = post(url, timeout=300, json={'model': cfg.get('tts_model') or 'tts-1', 'input': text, 'voice': voice,
                                     'response_format': 'wav', 'speed': float(cfg.get('tts_speed') or 1.0)})
    if r.status_code >= 400:
        raise TTSError(f'TTS server {r.status_code}: {r.text[:200]}')
    raw = Path(out).with_suffix('.raw')
    raw.write_bytes(r.content)
    _to_wav(raw, out)
    raw.unlink(missing_ok=True)


def _to_wav(src, dst):
    """Any audio ffmpeg reads -> 24 kHz mono PCM WAV."""
    p = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src), '-ac', '1', '-ar',
                        '24000', '-c:a', 'pcm_s16le', str(dst)], capture_output=True, text=True, timeout=120,
                       stdin=subprocess.DEVNULL)
    if p.returncode:
        raise TTSError(f'не удалось прочитать аудио: {p.stderr[-200:]}')


def import_audio(src, dst):
    """A user's own recording (wav/mp3/ogg/m4a...) -> the engine's WAV format."""
    _to_wav(src, dst)
    return wav_duration(dst)


# ------------------------------------------------------------------ public
def status(cfg):
    prov = cfg.get('tts_provider', 'piper')
    out = {'provider': prov, 'ok': False, 'voices': {}}
    if prov == 'piper':
        ok, why = piper_available()
        out['voices'] = piper_voices(cfg.get('piper_dir'))
        out['ok'] = ok and bool(out['voices'])
        out['error'] = why or ('' if out['voices'] else f'в папке {cfg.get("piper_dir")} нет голосов (*.onnx)')
    elif prov == 'openai_tts':
        out['ok'] = bool(cfg.get('tts_url'))
        out['voices'] = {v.strip(): {'language': 'any'} for v in (cfg.get('tts_voices') or '').split(',') if v.strip()}
        out['error'] = '' if out['ok'] else 'не задан адрес TTS-сервера'
    else:
        out['error'] = 'озвучка выключена'
    return out


def synthesize(text, voice, out, cfg, post=None):
    """text -> WAV at `out` with `voice` (piper voice id or server voice name). -> duration (s)."""
    text = re.sub(r'\s+', ' ', text or '').strip()
    if not text:
        raise TTSError('пустая реплика')
    prov = cfg.get('tts_provider', 'piper')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(out).with_name(Path(out).stem + '.part.wav')
    if prov == 'piper':
        voices = piper_voices(cfg.get('piper_dir'))
        if voice not in voices:
            raise TTSError(f'голос Piper «{voice}» не найден в {cfg.get("piper_dir")}')
        ok, why = piper_available()
        if not ok:
            raise TTSError(why)
        _piper(text, voices[voice]['path'], tmp, float(cfg.get('tts_speed') or 1.0))
    elif prov == 'openai_tts':
        if not cfg.get('tts_url'):
            raise TTSError('не задан адрес TTS-сервера')
        _openai(text, voice, tmp, cfg, post=post)
    else:
        raise TTSError('озвучка выключена в настройках')
    os.replace(tmp, out)
    d = wav_duration(out)
    if d < 0.1:
        raise TTSError('TTS вернул пустое аудио')
    return d


def _speech_span(path):
    """(first, last) second of audible speech (simple energy gate)."""
    import numpy as np
    with wave.open(str(path), 'rb') as w:
        sr, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    if not len(x):
        return 0.0, 0.0
    hop = max(1, sr // 100)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode='same'))[::hop]
    on = np.where(rms > max(200.0, 0.08 * rms.max()))[0]
    if not len(on):
        return 0.0, len(x) / sr
    return on[0] * hop / sr, min(len(x) / sr, (on[-1] + 1) * hop / sr)


def align(path, text, language='ru', use_whisper=True):
    """Word timings [{t0, t1, text}] of the script's words in the synthesized clip."""
    words = text.split()
    if not words:
        return []
    if use_whisper:
        try:
            from . import pipeline
            segs = pipeline.transcribe(str(path), language if language in ('ru', 'en', 'uk', 'kk', 'de', 'es') else 'auto')
            heard = [w for s in segs for w in (s.get('words') or [])]
            if heard and abs(len(heard) - len(words)) <= max(1, len(words) // 5):
                out = []
                for k, w in enumerate(words):                  # script words, Whisper's clock
                    h = heard[min(len(heard) - 1, round(k * (len(heard) - 1) / max(1, len(words) - 1)))]
                    out.append({'t0': round(h['start'], 3), 't1': round(h['end'], 3), 'text': w})
                for a, b in zip(out, out[1:]):
                    b['t0'] = max(b['t0'], a['t0'] + 0.01)
                    a['t1'] = min(max(a['t1'], a['t0'] + 0.05), b['t0'])
                return out
        except Exception as exc:
            log.info('whisper alignment unavailable (%s); proportional timings', str(exc)[:120])
    a, b = _speech_span(path)
    weights = [max(1, len(re.sub(r'\W', '', w))) for w in words]
    total = sum(weights)
    t, out = a, []
    for w, k in zip(words, weights):
        d = (b - a) * k / total
        out.append({'t0': round(t, 3), 't1': round(t + d - 0.02, 3), 'text': w})
        t += d
    return out
