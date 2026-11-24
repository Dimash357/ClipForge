"""ComfyUI client: text-to-image with SDXL or SD 1.5 checkpoints, optional IP-Adapter character references.

ComfyUI runs as its own local server (default http://127.0.0.1:8188); ClipForge builds the workflow in API
format, queues it (/prompt), waits for /history and downloads the image (/view). Nothing is generated here
without a real ComfyUI answering: an unreachable server or a missing checkpoint is an error, never a fake image.

* SDXL: 768x1344 (9:16 bucket), ~6.5 GB VRAM with the refiner off; SD 1.5: 512x912, ~3.5 GB.
* IP-Adapter: needs the ComfyUI_IPAdapter_plus nodes (+ CLIP vision and IP-Adapter weights). Each character in a
  frame adds an IPAdapter node fed with that character's reference sheet; without the nodes the Character Bible
  text + seed keep the look and the result says so.
* /free unloads ComfyUI's models after a batch, so Whisper / the LLM get the VRAM back; CUDA OOM -> /free and one
  retry at a smaller size.
"""
import json
import logging
import time
import uuid
from urllib.parse import urlencode

log = logging.getLogger('studio.comfy')

SIZES = {'sdxl': (768, 1344), 'sd15': (512, 912)}
SMALLER = {'sdxl': (640, 1136), 'sd15': (448, 800)}
VRAM_MB = {'sdxl': 6500, 'sd15': 3500}
IPA_NODES = ('IPAdapterUnifiedLoader', 'IPAdapter')


class ComfyError(RuntimeError):
    pass


class ComfyOOM(ComfyError):
    pass


class Client:
    def __init__(self, url='http://127.0.0.1:8188', timeout=600, session=None):
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex
        if session is None:
            import requests
            session = requests.Session()
        self.http = session

    # ------------------------------------------------------------ plumbing
    def _get(self, path, **kw):
        r = self.http.get(self.url + path, timeout=kw.pop('timeout', 15), **kw)
        r.raise_for_status()
        return r

    def _post(self, path, **kw):
        r = self.http.post(self.url + path, timeout=kw.pop('timeout', 30), **kw)
        if r.status_code >= 400:
            raise ComfyError(f'ComfyUI {path}: {r.status_code} {r.text[:300]}')
        return r

    # ------------------------------------------------------------ checks
    def status(self):
        """-> {'ok', 'version', 'vram_total', 'vram_free', 'checkpoints', 'ip_adapter', 'error'}."""
        out = {'ok': False, 'url': self.url, 'checkpoints': [], 'ip_adapter': False}
        try:
            st = self._get('/system_stats', timeout=5).json()
            dev = (st.get('devices') or [{}])[0]
            out.update(ok=True, version=(st.get('system') or {}).get('comfyui_version'),
                       device=dev.get('name'), vram_total=dev.get('vram_total'), vram_free=dev.get('vram_free'))
            info = self._get('/object_info/CheckpointLoaderSimple', timeout=10).json()
            out['checkpoints'] = info['CheckpointLoaderSimple']['input']['required']['ckpt_name'][0]
            out['ip_adapter'] = all(self._has_node(n) for n in IPA_NODES)
            if out['ip_adapter']:
                ipa = self._get('/object_info/IPAdapterUnifiedLoader', timeout=10).json()
                out['ip_presets'] = ipa['IPAdapterUnifiedLoader']['input']['required']['preset'][0]
        except Exception as exc:
            out['error'] = str(exc)[:300]
        return out

    def _has_node(self, name):
        try:
            return bool(self._get(f'/object_info/{name}', timeout=10).json().get(name))
        except Exception:
            return False

    # ------------------------------------------------------------ workflow
    @staticmethod
    def workflow(*, checkpoint, prompt, negative, seed, width, height, steps=28, cfg=6.0, sampler='dpmpp_2m',
                 scheduler='karras', prefix='clipforge', refs=(), ip_preset='PLUS (high strength)', ip_weight=0.7):
        """API-format graph. refs: uploaded image names (one per character) -> IP-Adapter nodes."""
        g = {
            '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': checkpoint}},
            '2': {'class_type': 'CLIPTextEncode', 'inputs': {'text': prompt, 'clip': ['1', 1]}},
            '3': {'class_type': 'CLIPTextEncode', 'inputs': {'text': negative, 'clip': ['1', 1]}},
            '4': {'class_type': 'EmptyLatentImage', 'inputs': {'width': width, 'height': height, 'batch_size': 1}},
            '6': {'class_type': 'VAEDecode', 'inputs': {'samples': ['5', 0], 'vae': ['1', 2]}},
            '7': {'class_type': 'SaveImage', 'inputs': {'images': ['6', 0], 'filename_prefix': prefix}},
        }
        model = ['1', 0]
        if refs:
            g['10'] = {'class_type': 'IPAdapterUnifiedLoader', 'inputs': {'model': model, 'preset': ip_preset}}
            model, ipa = ['10', 0], ['10', 1]
            w = round(ip_weight / max(1, len(refs)) ** 0.5, 3)         # several characters share the influence
            for k, name in enumerate(refs):
                li, ni = str(20 + 2 * k), str(21 + 2 * k)
                g[li] = {'class_type': 'LoadImage', 'inputs': {'image': name}}
                g[ni] = {'class_type': 'IPAdapter', 'inputs': {'model': model, 'ipadapter': ipa, 'image': [li, 0],
                                                               'weight': w, 'start_at': 0.0, 'end_at': 0.85,
                                                               'weight_type': 'standard'}}
                model = [ni, 0]
        g['5'] = {'class_type': 'KSampler', 'inputs': {
            'model': model, 'positive': ['2', 0], 'negative': ['3', 0], 'latent_image': ['4', 0],
            'seed': int(seed) % (2 ** 63), 'steps': int(steps), 'cfg': float(cfg), 'sampler_name': sampler,
            'scheduler': scheduler, 'denoise': 1.0}}
        return g

    def upload(self, path, name=None):
        """Upload a reference image -> the name ComfyUI knows it by."""
        with open(path, 'rb') as f:
            r = self._post('/upload/image', files={'image': (name or path.name, f, 'image/png')},
                           data={'overwrite': 'true'})
        return r.json().get('name') or (name or path.name)

    def run(self, graph, cancel=None, progress=None):
        """Queue the graph, wait for it, return the PNG bytes of the first saved image."""
        r = self._post('/prompt', json={'prompt': graph, 'client_id': self.client_id})
        pid = r.json().get('prompt_id')
        if not pid:
            raise ComfyError(f'ComfyUI did not accept the workflow: {r.text[:300]}')
        t0 = time.monotonic()
        while True:
            if cancel and cancel():
                try:
                    self._post('/interrupt')
                finally:
                    from .procutil import JobCancelled
                    raise JobCancelled()
            if time.monotonic() - t0 > self.timeout:
                raise ComfyError(f'ComfyUI did not finish in {self.timeout} s')
            h = self._get(f'/history/{pid}').json().get(pid)
            if h:
                st = h.get('status') or {}
                if st.get('status_str') == 'error' or not st.get('completed', True):
                    msg = json.dumps(st.get('messages') or st)[:600]
                    if 'out of memory' in msg.lower() or 'outofmemory' in msg.lower():
                        raise ComfyOOM(msg)
                    raise ComfyError(f'ComfyUI error: {msg}')
                for node in (h.get('outputs') or {}).values():
                    for img in node.get('images') or []:
                        q = urlencode({'filename': img['filename'], 'subfolder': img.get('subfolder', ''),
                                       'type': img.get('type', 'output')})
                        return self._get(f'/view?{q}', timeout=60).content
                if st.get('completed'):
                    raise ComfyError('ComfyUI finished without an image')
            if progress:
                progress(min(0.95, (time.monotonic() - t0) / 60))
            time.sleep(0.8)

    def free(self):
        """Unload models and free VRAM inside ComfyUI (other jobs can use the GPU again)."""
        try:
            self._post('/free', json={'unload_models': True, 'free_memory': True}, timeout=15)
            return True
        except Exception as exc:
            log.info('ComfyUI /free failed: %s', exc)
            return False
