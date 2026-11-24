"""Optional text-generation provider with hard time limits and a circuit breaker.

Configuration (environment variables, all optional):
    CLIPFORGE_AI           auto (default) | off | ollama | openai
    CLIPFORGE_AI_URL       OpenAI-compatible base URL, e.g. https://api.openai.com/v1, http://127.0.0.1:1234/v1 (LM Studio),
                           https://openrouter.ai/api/v1, http://127.0.0.1:11434/v1 (Ollama)
    CLIPFORGE_AI_KEY       API key for that URL (not needed for local servers)
    CLIPFORGE_AI_MODEL     model name for that URL
    USE_OLLAMA=1, OLLAMA_MODEL (default qwen2.5:3b), OLLAMA_HOST (default http://127.0.0.1:11434)
    CLIPFORGE_AI_TIMEOUT   seconds per request (default 25)

`auto` uses the OpenAI-compatible URL when CLIPFORGE_AI_URL and CLIPFORGE_AI_MODEL are set, else Ollama when
USE_OLLAMA=1, else nothing (callers then use their local, non-AI fallback). A failed or slow request disables the
provider for a few minutes, so a dead API can never stall a render job: every caller just falls back.
"""
import json
import logging
import os
import re
import threading
import time

log = logging.getLogger('studio.ai')

BREAKER_SECONDS = 300
_lock = threading.Lock()
_down_until = 0.0
_last_error = ''


def _timeout():
    try:
        return max(3.0, min(120.0, float(os.getenv('CLIPFORGE_AI_TIMEOUT', '25'))))
    except ValueError:
        return 25.0


def config():
    """-> {'kind': 'openai'|'ollama'|None, 'model', 'url', 'reason'}"""
    mode = os.getenv('CLIPFORGE_AI', 'auto').strip().lower()
    url, model = os.getenv('CLIPFORGE_AI_URL', '').strip(), os.getenv('CLIPFORGE_AI_MODEL', '').strip()
    oll = os.getenv('USE_OLLAMA', '0') == '1'
    if mode == 'off':
        return {'kind': None, 'reason': 'CLIPFORGE_AI=off'}
    if mode in ('auto', 'openai') and url and model:
        return {'kind': 'openai', 'url': url.rstrip('/'), 'model': model}
    if mode in ('auto', 'ollama') and (oll or mode == 'ollama'):
        return {'kind': 'ollama', 'url': os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/'),
                'model': os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')}
    return {'kind': None, 'reason': 'AI-провайдер не настроен'}


def available():
    return config()['kind'] is not None and time.monotonic() >= _down_until


def status():
    c = config()
    left = max(0, int(_down_until - time.monotonic()))
    return {'kind': c['kind'], 'model': c.get('model'), 'url': c.get('url'), 'paused_s': left,
            'last_error': _last_error, 'reason': c.get('reason')}


def paused():
    return time.monotonic() < _down_until


def trip(err):
    global _down_until, _last_error
    with _lock:
        _down_until = time.monotonic() + BREAKER_SECONDS
        _last_error = str(err)[:200]
    log.warning('AI provider unavailable (%s); local fallback for %d s', _last_error, BREAKER_SECONDS)


def reset():
    global _down_until, _last_error
    with _lock:
        _down_until, _last_error = 0.0, ''


def extract_json(text):
    """First JSON object in a model answer (models like to wrap it in prose or ``` fences)."""
    if isinstance(text, dict):
        return text
    m = re.search(r'\{.*\}', text or '', re.S)
    if not m:
        raise ValueError('no JSON object in the answer')
    return json.loads(m.group(0))


def unload_local(post=None):
    """Ask a local Ollama to drop its model from VRAM now (keep_alive 0), before ComfyUI needs the card."""
    c = config()
    if c['kind'] != 'ollama':
        return False
    try:
        if post is None:
            import requests
            post = requests.post
        post(c['url'] + '/api/generate', json={'model': c['model'], 'keep_alive': 0}, timeout=(3.05, 20))
        return True
    except Exception as exc:
        log.info('could not unload the Ollama model: %s', exc)
        return False


def is_local():
    """True when the configured model runs on this computer (Ollama, LM Studio...): it uses our GPU."""
    c = config()
    return c['kind'] == 'ollama' or bool(c['kind'] and re.search(r'//(127\.0\.0\.1|localhost|\[::1\])', c['url'] or ''))


def complete_json(system, user, temperature=0.8, post=None, timeout=None):
    """Ask the configured model for a JSON object. Returns (dict, 'kind:model') or (None, reason). Never raises.
    timeout: seconds for this request (long answers such as scripts), default CLIPFORGE_AI_TIMEOUT."""
    c = config()
    if not c['kind']:
        return None, c.get('reason', 'off')
    if time.monotonic() < _down_until:
        return None, 'paused after an error'
    if post is None:
        import requests
        post = requests.post
    to = (3.05, float(timeout) if timeout else _timeout())
    try:
        if c['kind'] == 'openai':
            headers = {'Content-Type': 'application/json'}
            key = os.getenv('CLIPFORGE_AI_KEY', '').strip()
            if key:
                headers['Authorization'] = f'Bearer {key}'
            r = post(c['url'] + '/chat/completions', headers=headers, timeout=to, json={
                'model': c['model'], 'temperature': temperature,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]})
            r.raise_for_status()
            text = r.json()['choices'][0]['message']['content']
        else:
            r = post(c['url'] + '/api/chat', timeout=to, json={
                'model': c['model'], 'stream': False, 'format': 'json', 'options': {'temperature': temperature},
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]})
            r.raise_for_status()
            text = r.json()['message']['content']
        return extract_json(text), f'{c["kind"]}:{c["model"]}'
    except Exception as exc:
        trip(exc)
        return None, f'error: {str(exc)[:120]}'
