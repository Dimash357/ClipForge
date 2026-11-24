"""3C AI Visual & Voice Engine against a local stand-in of ComfyUI and of an OpenAI-compatible TTS server (a tiny
HTTP server in this file: real PNG / WAV bytes, made here). Workflow building, checks, IP-Adapter references,
cache / resume, OOM retry, VRAM lease and /free, voices, exact timings, manual replacements, Timeline IR, the
media job, the API and the pages. No real model is involved and none is pretended."""
import io
import json
import math
import shutil
import struct
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TransactionTestCase, override_settings

from studio import ai_provider, comfy, genmedia, jobs, resources, scripts, storyboard as sb, tts
from studio import timeline as T
from studio.models import Script
from .test_jobs_views import wait_for
from .test_resources import GPU, manager


def png(w, h, seed=0):
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', (w, h), ((seed * 37) % 255, 90, 160)).save(buf, 'PNG')
    return buf.getvalue()


def speech_wav(text, sr=24000):
    """One 0.25 s tone per word with 0.1 s pauses, 0.15 s silence around: alignable 'speech'."""
    frames = [0] * int(0.15 * sr)
    for _ in text.split():
        frames += [int(9000 * math.sin(2 * math.pi * 220 * k / sr)) for k in range(int(0.25 * sr))]
        frames += [0] * int(0.1 * sr)
    frames += [0] * int(0.15 * sr)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f'<{len(frames)}h', *frames))
    return buf.getvalue()


class FakeServer:
    """ComfyUI (/system_stats, /object_info, /prompt, /history, /view, /upload/image, /free, /interrupt) and
    /v1/audio/speech. oom=N: the first N prompts fail with CUDA out of memory."""

    def __init__(self, checkpoints=('juggernautXL.safetensors',), ip_adapter=True, oom=0):
        self.graphs, self.uploads, self.frees, self.speech = [], [], 0, []
        self.checkpoints, self.ip, self.oom = list(checkpoints), ip_adapter, oom
        self.on_prompt = None
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, body, ctype='application/json', code=200):
                if isinstance(body, (dict, list)):
                    body = json.dumps(body).encode()
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def body(self):
                n = int(self.headers.get('Content-Length') or 0)
                return self.rfile.read(n) if n else b''

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == '/system_stats':
                    return self.send({'system': {'comfyui_version': 'test'},
                                      'devices': [{'name': 'cuda:0 NVIDIA GeForce RTX 4060 Laptop GPU',
                                                   'vram_total': 8 * 2 ** 30, 'vram_free': 7 * 2 ** 30}]})
                if u.path.startswith('/object_info/'):
                    node = u.path.rsplit('/', 1)[1]
                    if node == 'CheckpointLoaderSimple':
                        return self.send({node: {'input': {'required': {'ckpt_name': [srv.checkpoints]}}}})
                    if node in comfy.IPA_NODES and srv.ip:
                        return self.send({node: {'input': {'required': {'preset': [['PLUS (high strength)']]}}}})
                    return self.send({})
                if u.path.startswith('/history/'):
                    pid = u.path.rsplit('/', 1)[1]
                    i = int(pid)
                    g = srv.graphs[i]
                    if g.get('_oom'):
                        return self.send({pid: {'status': {'status_str': 'error', 'completed': False, 'messages': [
                            ['execution_error', {'exception_message': 'CUDA error: out of memory'}]]}}})
                    return self.send({pid: {'status': {'status_str': 'success', 'completed': True},
                                            'outputs': {'7': {'images': [{'filename': f'{pid}.png', 'subfolder': '',
                                                                          'type': 'output'}]}}}})
                if u.path == '/view':
                    g = srv.graphs[int(parse_qs(u.query)['filename'][0].split('.')[0])]
                    lat = g['4']['inputs']
                    return self.send(png(lat['width'], lat['height'], g['5']['inputs']['seed']), 'image/png')
                self.send({}, code=404)

            def do_POST(self):
                u = urlparse(self.path)
                data = self.body()
                if u.path == '/prompt':
                    g = json.loads(data)['prompt']
                    if srv.on_prompt:
                        srv.on_prompt(g)
                    if srv.oom > 0:
                        srv.oom -= 1
                        g['_oom'] = True
                    srv.graphs.append(g)
                    return self.send({'prompt_id': str(len(srv.graphs) - 1)})
                if u.path == '/upload/image':
                    name = data.split(b'filename="')[1].split(b'"')[0].decode()
                    srv.uploads.append(name)
                    return self.send({'name': name})
                if u.path == '/free':
                    srv.frees += 1
                    return self.send({})
                if u.path == '/interrupt':
                    return self.send({})
                if u.path == '/v1/audio/speech':
                    req = json.loads(data)
                    srv.speech.append(req)
                    return self.send(speech_wav(req['input']), 'audio/wav')
                self.send({}, code=404)

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


ST = scripts.settings_of({'idea': 'лис и сова', 'format': 'cartoon', 'language': 'ru', 'duration': 16, 'id': 9})


def doc():
    with mock.patch.object(ai_provider, 'available', return_value=False):
        return sb.build(scripts.generate(ST))


class Base(SimpleTestCase):
    def setUp(self):
        self.srv = FakeServer()
        self.addCleanup(self.srv.close)
        self.media = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.media, True)
        ov = override_settings(MEDIA_ROOT=self.media)
        ov.enable()
        self.addCleanup(ov.disable)
        self.cfg = dict(genmedia.DEFAULTS, comfy_url=self.srv.url, checkpoint='juggernautXL.safetensors',
                        tts_provider='openai_tts', tts_url=self.srv.url + '/v1', tts_voices='alloy,echo',
                        voice_ru='alloy', align='simple', piper_dir='')
        m = manager(GPU)
        p = mock.patch.object(resources, '_manager', m)
        p.start()
        self.addCleanup(p.stop)
        self.mgr = m
        self.d = doc()


class Workflow(SimpleTestCase):
    def test_graph_and_ip_adapter_chain(self):
        g = comfy.Client.workflow(checkpoint='x.safetensors', prompt='p', negative='n', seed=5, width=768, height=1344)
        self.assertEqual(g['5']['inputs']['model'], ['1', 0])
        self.assertEqual((g['4']['inputs']['width'], g['4']['inputs']['height']), (768, 1344))
        g2 = comfy.Client.workflow(checkpoint='x', prompt='p', negative='n', seed=5, width=8, height=8,
                                   refs=['a.png', 'b.png'], ip_weight=0.8)
        ipa = [k for k, v in g2.items() if v['class_type'] == 'IPAdapter']
        self.assertEqual(len(ipa), 2)
        self.assertEqual(g2['5']['inputs']['model'], [ipa[-1], 0])        # chained into the sampler
        self.assertAlmostEqual(g2[ipa[0]]['inputs']['weight'], round(0.8 / 2 ** 0.5, 3))


class Images(Base):
    def test_status_and_clear_errors(self):
        st = genmedia.status(self.cfg)
        self.assertTrue(st['comfy']['ok'] and st['comfy']['checkpoint_ok'] and st['comfy']['ip_adapter'])
        self.assertTrue(st['tts']['ok'])
        with self.assertRaisesRegex(RuntimeError, 'не найдена'):
            genmedia.gen_images(self.d, 1, dict(self.cfg, checkpoint='missing.safetensors'))
        with self.assertRaisesRegex(RuntimeError, 'ComfyUI недоступен'):
            genmedia.gen_images(self.d, 1, dict(self.cfg, comfy_url='http://127.0.0.1:9'))
        self.assertEqual(self.srv.graphs, [])                               # nothing was faked

    def test_generate_cache_resume_and_references(self):
        seen = []
        self.srv.on_prompt = lambda g: seen.append(self.mgr._used()['vram_mb'])
        with mock.patch.object(ai_provider, 'unload_local') as unload:
            st = genmedia.gen_images(self.d, 1, self.cfg)
        unload.assert_called_once()
        sheets = [c['image'] for c in self.d['storyboard']['characters'].values()]
        frames = [f for s in self.d['storyboard']['scenes'].values() for f in s['frames']]
        self.assertEqual(st['done'], len(sheets) + len(frames))
        self.assertTrue(all(genmedia.valid(a) for a in sheets))
        img = frames[0]['assets']['image']
        self.assertEqual((img['width'], img['height']), comfy.SIZES['sdxl'])
        self.assertGreater(min(seen), 0)                                     # VRAM lease held while ComfyUI works
        self.assertEqual(self.mgr._used()['vram_mb'], 0)
        self.assertGreaterEqual(self.srv.frees, 1)                           # ComfyUI gives the VRAM back
        with_chars = [g for g in self.srv.graphs if any(v['class_type'] == 'IPAdapter' for v in g.values())]
        self.assertTrue(with_chars)
        self.assertEqual(len(set(self.srv.uploads)), len(sheets))            # each sheet uploaded once
        n = len(self.srv.graphs)
        again = genmedia.gen_images(self.d, 1, self.cfg)                      # resume / cache: nothing to do
        self.assertEqual((again['done'], len(self.srv.graphs)), (0, n))
        f0 = frames[0]['id']
        self.d = sb.edit_frame(self.d, f0, {'action': 'другое действие'})
        third = genmedia.gen_images(self.d, 1, self.cfg)
        self.assertEqual(third['done'], 1)                                   # only the changed frame
        forced = genmedia.gen_images(self.d, 1, self.cfg, frame_ids={f0}, force=True)
        self.assertEqual(forced['done'], 1)                                   # sheets are not redone

    def test_oom_retry_smaller(self):
        self.srv.oom = 1
        st = genmedia.gen_images(self.d, 1, self.cfg, characters={next(iter(self.d['storyboard']['characters']))})
        self.assertEqual(st['done'], 1)
        ch = next(c for c in self.d['storyboard']['characters'].values() if c.get('image'))
        self.assertEqual((ch['image']['width'], ch['image']['height']), comfy.SMALLER['sdxl'])
        self.assertEqual(self.mgr.stats['oom'], 1)


class Voices(Base):
    def test_voices_timings_and_cache(self):
        st = genmedia.gen_voices(self.d, 1, self.cfg)
        lines = sum(len(s['lines']) for s in self.d['scenes'])
        self.assertEqual(st['done'], lines)
        sb_ = self.d['storyboard']
        for s in self.d['scenes']:
            slot = sb_['voice'][s['id']]
            end = 0.0
            for li, line in enumerate(s['lines']):
                v = slot[str(li)]
                self.assertAlmostEqual(v['duration'], tts.wav_duration(Path(self.media) / v['path']), places=3)
                self.assertEqual([w['text'] for w in v['words']], line['text'].split())
                self.assertGreaterEqual(v['start'], end)
                end = v['start'] + v['duration']
            self.assertGreaterEqual(s['duration'], end)                       # the scene fits its real voice
            self.assertAlmostEqual(sum(f['duration'] for f in sb_['scenes'][s['id']]['frames']), s['duration'],
                                   delta=0.02)
        n = len(self.srv.speech)
        self.assertEqual(genmedia.gen_voices(self.d, 1, self.cfg)['cached'], lines)
        self.assertEqual(len(self.srv.speech), n)

    def test_no_provider_is_an_error_not_a_fake(self):
        st = genmedia.gen_voices(self.d, 1, dict(self.cfg, tts_provider='piper', piper_dir=str(Path(self.media) / 'none')))
        self.assertEqual(st['done'], 0)
        self.assertTrue(st['errors'])
        self.assertIn('не найден', st['errors'][0]['error'])
        self.assertFalse(list(Path(self.media).rglob('*.wav')))

    def test_manual_replacements_are_kept(self):
        s0 = self.d['scenes'][0]
        f0 = self.d['storyboard']['scenes'][s0['id']]['frames'][0]
        self.d = genmedia.replace_image(self.d, 1, f0['id'], io.BytesIO(png(300, 500)))
        src = Path(self.media) / 'my.wav'
        src.write_bytes(speech_wav('раз два три'))
        self.d = genmedia.replace_voice(self.d, 1, s0['id'], 0, src, self.cfg)
        genmedia.gen_images(self.d, 1, self.cfg, frame_ids={f0['id']})
        genmedia.gen_voices(self.d, 1, self.cfg, lines=[[s0['id'], 0]])
        self.assertTrue(self.d['storyboard']['scenes'][s0['id']]['frames'][0]['assets']['image']['manual'])
        self.assertTrue(self.d['storyboard']['voice'][s0['id']]['0']['manual'])


class IR(Base):
    def test_timeline_points_at_real_files_and_real_timings(self):
        genmedia.gen_images(self.d, 1, self.cfg)
        genmedia.gen_voices(self.d, 1, self.cfg)
        tl = scripts.to_timeline(self.d, 1)
        self.assertEqual(T.validate(tl), [])
        imgs = [tl.asset(i.asset) for i in tl.tracks[0].items]
        self.assertTrue(all(a.uri.startswith('media:generated/scripts/1/frames/') for a in imgs))
        voice = next(t for t in tl.tracks if t.type == 'audio')
        s0 = self.d['scenes'][0]
        v0 = self.d['storyboard']['voice'][s0['id']]['0']
        self.assertAlmostEqual(voice.items[0].start, s0['start'] + v0['start'], places=2)
        self.assertAlmostEqual(voice.items[0].duration, v0['duration'], places=2)
        words = next(t for t in tl.tracks if t.type == 'caption').items[0].words
        self.assertAlmostEqual(words[0].t0, voice.items[0].start + v0['words'][0]['t0'], places=2)
        from studio.timeline import renderer
        paths, media = renderer.preflight(tl, renderer.RenderSettings())        # every file exists and probes
        used = {i.asset for t in tl.tracks for i in t.items if getattr(i, 'asset', None)}
        self.assertEqual(set(paths), used)


class ApiAndJob(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()
        self.srv = FakeServer()
        self.addCleanup(self.srv.close)
        self.media = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.media, True)
        ov = override_settings(MEDIA_ROOT=self.media, CLIPFORGE_SETTINGS_FILE=Path(self.media) / 's.json')
        ov.enable()
        self.addCleanup(ov.disable)
        p = mock.patch.object(ai_provider, 'available', return_value=False)
        p.start()
        self.addCleanup(p.stop)
        self.sc = Script.objects.create(idea='лис и сова', format='cartoon', status='ready', data=doc())

    def test_settings_status_job_upload(self):
        self.assertContains(self.client.get('/generation/'), 'generation.js')
        r = self.client.post('/api/media/config/', {'comfy_url': self.srv.url, 'checkpoint': 'juggernautXL.safetensors',
                                                    'tts_provider': 'openai_tts', 'tts_url': self.srv.url + '/v1',
                                                    'tts_voices': 'alloy', 'voice_ru': 'alloy', 'align': 'simple',
                                                    'steps': '20', 'evil': 1}, content_type='application/json')
        self.assertEqual(r.json()['config']['steps'], 20)
        st = self.client.get('/api/media/status/').json()
        self.assertTrue(st['comfy']['checkpoint_ok'] and st['tts']['ok'])
        self.assertEqual(self.client.post(f'/api/scripts/{self.sc.pk}/media/', {'what': 'bad'},
                                          content_type='application/json').status_code, 400)
        self.assertEqual(self.client.post(f'/api/scripts/{self.sc.pk}/media/', {'what': 'all'},
                                          content_type='application/json').status_code, 200)
        self.assertTrue(wait_for(lambda: Script.objects.get(pk=self.sc.pk).status != 'generating', timeout=120))
        sc = Script.objects.get(pk=self.sc.pk)
        self.assertEqual((sc.status, sc.error), ('ready', ''))
        summ = genmedia.summary(sc.data)
        self.assertEqual((summ['images'], summ['voiced'], summ['sheets']),
                         (summ['frames'], summ['lines'], summ['characters']))
        d = self.client.get(f'/api/scripts/{sc.pk}/storyboard/').json()
        self.assertEqual(d['media']['images'], summ['frames'])
        fid = next(iter(sc.data['storyboard']['scenes'].values()))['frames'][0]['id']
        up = SimpleUploadedFile('mine.png', png(200, 300), content_type='image/png')
        self.assertEqual(self.client.post(f'/api/scripts/{sc.pk}/media/frames/{fid}/upload/', {'file': up}).status_code,
                         200)
        sc.refresh_from_db()
        self.assertTrue(next(f for s in sc.data['storyboard']['scenes'].values() for f in s['frames']
                             if f['id'] == fid)['assets']['image']['manual'])
        bad = SimpleUploadedFile('x.png', b'not an image', content_type='image/png')
        self.assertEqual(self.client.post(f'/api/scripts/{sc.pk}/media/frames/{fid}/upload/', {'file': bad}).status_code,
                         400)
        cid = next(iter(sc.data['storyboard']['characters']))
        self.assertEqual(self.client.post(f'/api/scripts/{sc.pk}/storyboard/characters/{cid}/voice/',
                                          {'tts_voice': 'echo'}, content_type='application/json').status_code, 200)
        self.assertContains(self.client.get(f'/scripts/{sc.pk}/storyboard/'), 'storyboard.js')

    def test_unreachable_comfyui_fails_the_job_honestly(self):
        self.client.post('/api/media/config/', {'comfy_url': 'http://127.0.0.1:9', 'checkpoint': 'x'},
                         content_type='application/json')
        self.client.post(f'/api/scripts/{self.sc.pk}/media/', {'what': 'images'}, content_type='application/json')
        self.assertTrue(wait_for(lambda: Script.objects.get(pk=self.sc.pk).status != 'generating', timeout=60))
        sc = Script.objects.get(pk=self.sc.pk)
        self.assertEqual(sc.status, 'error')
        self.assertIn('ComfyUI недоступен', sc.error)
        self.assertEqual(genmedia.summary(sc.data)['images'], 0)
