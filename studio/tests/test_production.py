"""3D Cartoon Studio: idea -> MP4 through the real pipeline (template script, rules storyboard, images and voices
from the local ComfyUI / TTS stand-in of test_genmedia, real ffmpeg assembly + render). Checks the output file,
per-stage states, resume after cancel, partial redo (one frame -> one scene re-rendered), formats, music / sfx,
metadata, honest failures and the API / pages."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

from django.test import TransactionTestCase, override_settings

from studio import ai_provider, genmedia, jobs, production as P
from studio.models import Script
from .test_genmedia import FakeServer
from .test_jobs_views import wait_for


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,width,height:format=duration',
                          '-of', 'json', str(path)], capture_output=True, text=True).stdout
    d = json.loads(out)
    v = next(s for s in d['streams'] if s['codec_type'] == 'video')
    return {'w': v['width'], 'h': v['height'], 'audio': any(s['codec_type'] == 'audio' for s in d['streams']),
            'duration': float(d['format']['duration'])}


class Cancel:
    """Reporter stand-in: cancels after `after` progress calls."""
    def __init__(self, after):
        self.after, self.n = after, 0
        self.overall_fn = None

    def cancelled(self):
        return self.n >= self.after

    def stage(self, *a, **k):
        pass

    def frac(self, *a, **k):
        self.n += 1

    def checkpoint(self, *a, **k):
        pass


class Base(TransactionTestCase):
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
        genmedia.save_config({'comfy_url': self.srv.url, 'checkpoint': 'juggernautXL.safetensors',
                              'tts_provider': 'openai_tts', 'tts_url': self.srv.url + '/v1', 'tts_voices': 'alloy,echo',
                              'voice_ru': 'alloy', 'align': 'simple', 'steps': 4})

    def make(self, **st):
        st = P.clean_settings(dict({'resolution': '720', 'quality': 'small', 'fps': 24}, **st))
        return Script.objects.create(idea='почему кошки приземляются на лапы', format='cartoon', language='ru',
                                     duration=12, status='draft', data={'production': {'settings': st, 'stages': {}}})


class Pipeline(Base):
    def test_idea_to_mp4_then_partial_redo(self):
        sc = self.make()
        P.run(sc, None)
        sc.refresh_from_db()
        states = P.stage_states(sc.data)
        self.assertEqual({k: v['status'] for k, v in states.items()}, {k: 'done' for k in P.STAGE_KEYS})
        self.assertEqual(sc.status, 'ready')
        out = Path(self.media) / sc.data['production']['output']['path']
        info = probe(out)
        tl = json.loads((Path(self.media) / sc.data['production']['timeline_path']).read_text())
        self.assertEqual((info['w'], info['h']), (720, 1280))
        self.assertTrue(info['audio'])
        self.assertAlmostEqual(info['duration'], tl['duration'], delta=0.3)
        roles = {i.get('role') for t in tl['tracks'] for i in t['items'] if i['type'] == 'audio'}
        self.assertEqual(roles, {'voice', 'music', 'sfx'})
        self.assertTrue(any(i['type'] == 'transition' for t in tl['tracks'] for i in t['items']))
        self.assertTrue(any(t['type'] == 'caption' for t in tl['tracks']))
        meta = sc.data['production']['meta']
        self.assertTrue(meta['title'] and meta['hashtags'][0].startswith('#'))
        self.assertTrue(out.with_suffix('.txt').is_file())
        prompts = len(self.srv.graphs)
        # second run: everything done -> nothing regenerated
        P.run(sc, None)
        self.assertEqual(len(self.srv.graphs), prompts)
        # redo one frame: only it is generated, assembly / render go stale, only its scene re-renders
        sc.refresh_from_db()
        doc = sc.data
        f0 = next(iter(doc['storyboard']['scenes'].values()))['frames'][0]
        genmedia.gen_images(doc, sc.pk, P.media_cfg(doc), frame_ids={f0['id']}, force=True)
        Script.objects.filter(pk=sc.pk).update(data=doc)
        sc.refresh_from_db()
        st = P.stage_states(sc.data)
        self.assertEqual((st['images']['status'], st['assemble']['status'], st['render']['status']),
                         ('done', 'stale', 'stale'))
        P.run(sc, None)
        sc.refresh_from_db()
        o = sc.data['production']['output']
        self.assertEqual(len(self.srv.graphs), prompts + 1)
        self.assertEqual(o['rendered_scenes'], 1)
        self.assertGreater(o['scenes'], 1)

    def test_resume_after_cancel_and_formats(self):
        sc = self.make(aspect='16:9', music='synth', transition='dynamic', motion='dynamic')
        from studio.procutil import JobCancelled
        with self.assertRaises(JobCancelled):
            P.run(sc, Cancel(after=4))
        sc.refresh_from_db()
        st = P.stage_states(sc.data)
        self.assertEqual(st['script']['status'], 'done')
        self.assertNotEqual(st['render']['status'], 'done')
        made = genmedia.summary(sc.data)['images']
        P.run(sc, None, until='render')
        sc.refresh_from_db()
        st = P.stage_states(sc.data)
        self.assertEqual((st['render']['status'], st['meta']['status']), ('done', 'pending'))
        total = genmedia.summary(sc.data)['frames'] + genmedia.summary(sc.data)['characters']
        self.assertGreater(made, 0)
        # finished items are kept; at most the one interrupted mid-generation is made again
        self.assertLessEqual(len(self.srv.graphs), total + 1)
        frame_graph = [g for g in self.srv.graphs if 'clipforge_fr' in json.dumps(g)][0]
        self.assertEqual((frame_graph['4']['inputs']['width'], frame_graph['4']['inputs']['height']), (1344, 768))
        info = probe(Path(self.media) / sc.data['production']['output']['path'])
        self.assertEqual((info['w'], info['h']), (1280, 720))
        self.assertIn('horizontal 16:9', frame_graph['2']['inputs']['text'])

    def test_square_without_voice_music_off_and_library(self):
        lib = Path(self.media) / 'lib'
        lib.mkdir()
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=f=330:d=3', str(lib / 'calm_tune.mp3')],
                       check=True)
        sc = self.make(aspect='1:1', voice=False, music='library', music_dir=str(lib), sfx=False, captions=False)
        P.run(sc, None, until='assemble')
        sc.refresh_from_db()
        self.assertEqual(P.stage_states(sc.data)['voice']['status'], 'skipped')
        self.assertEqual(self.srv.speech, [])
        tl = json.loads((Path(self.media) / sc.data['production']['timeline_path']).read_text())
        self.assertEqual((tl['canvas']['width'], tl['canvas']['height']), (720, 720))
        roles = [i.get('role') for t in tl['tracks'] for i in t['items'] if i['type'] == 'audio']
        self.assertEqual(roles, ['music'])
        self.assertFalse(any(t['type'] == 'caption' for t in tl['tracks']))
        self.assertEqual(sc.data['production']['stages']['assemble']['info']['music']['source'], 'library')

    def test_failures_are_honest(self):
        genmedia.save_config({'comfy_url': 'http://127.0.0.1:9'})
        sc = self.make()
        with self.assertRaises(Exception):
            P.run(sc, None)
        sc.refresh_from_db()
        st = P.stage_states(sc.data)
        self.assertEqual(st['images']['status'], 'error')
        self.assertIn('ComfyUI недоступен', st['images']['message'])
        self.assertEqual((st['render']['status'], sc.status), ('pending', 'error'))
        self.assertFalse((Path(self.media) / 'generated' / 'scripts' / str(sc.pk) / 'export').exists())
        with self.assertRaises(P.StageError):              # assembly refuses missing media instead of faking it
            P.assemble(sc.data, sc.pk)


class Audio(Base):
    def test_synth_music_and_sfx_are_real_audio(self):
        dest = Path(self.media) / 'm.wav'
        P.synth_music('playful', 5.0, dest)
        import wave
        import numpy as np
        with wave.open(str(dest)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            self.assertAlmostEqual(w.getnframes() / w.getframerate(), 5.0, delta=0.05)
        self.assertGreater(np.abs(x).max(), 2000)
        for name in P.SFX:
            self.assertGreater(P.sfx_file(name, self.media).stat().st_size, 1000)
        for mood in P.MOODS:
            for ch in P.MOODS[mood][0]:
                self.assertEqual(len(P._chord(ch)), 3)

    def test_meta_local_and_settings_clean(self):
        d = {'title': 'Кошки и гравитация', 'logline': 'Почему кошки всегда приземляются на лапы', 'format': 'cartoon',
             'language': 'ru', 'scenes': [{'lines': [{'text': 'Кошки умеют поворачиваться в воздухе.'}]}]}
        m = P.gen_meta(d)
        self.assertEqual(m['provider'], 'local')
        self.assertIn('#мультфильм', m['hashtags'])
        self.assertTrue(all(t.startswith('#') for t in m['hashtags']))
        st = P.clean_settings({'aspect': '4:3', 'fps': '24', 'music_volume': 5, 'sfx': 'false', 'evil': 1})
        self.assertEqual((st['aspect'], st['fps'], st['music_volume'], st['sfx']), ('9:16', 24, 1.0, False))
        self.assertNotIn('evil', st)


class Api(Base):
    def test_one_click_via_queue_and_pages(self):
        self.assertContains(self.client.get('/studio/'), 'cartoon.js')
        self.assertEqual(self.client.post('/api/studio/create/', {'idea': ''}, content_type='application/json')
                         .status_code, 400)
        r = self.client.post('/api/studio/create/', {'idea': 'лис и сова', 'format': 'cartoon', 'duration': 12,
                                                     'mode': 'oneclick', 'settings': {'resolution': '720',
                                                                                      'quality': 'small', 'fps': 24}},
                             content_type='application/json').json()
        pk = r['id']
        self.assertContains(self.client.get(r['url']), 'cartoon.js')
        self.assertTrue(wait_for(lambda: Script.objects.get(pk=pk).status not in ('generating', 'draft'), timeout=240))
        d = self.client.get(f'/api/studio/{pk}/').json()
        self.assertEqual(d['status'], 'ready', d.get('error'))
        self.assertEqual({k: v['status'] for k, v in d['stages'].items()}, {k: 'done' for k in P.STAGE_KEYS})
        self.assertTrue(d['output_url'].endswith('.mp4') and d['frames'] and d['lines'])
        self.assertEqual(d['job']['status'], 'completed')
        # edit metadata and settings; a format change makes only assembly / render (and storyboard prompts) stale
        m = self.client.post(f'/api/studio/{pk}/meta/', {'title': 'Мой заголовок', 'hashtags': 'лис #сова'},
                             content_type='application/json').json()['meta']
        self.assertEqual(m['hashtags'], ['#лис', '#сова'])
        self.client.post(f'/api/studio/{pk}/settings/', {'settings': {'music': 'off'}}, content_type='application/json')
        d = self.client.get(f'/api/studio/{pk}/').json()
        self.assertEqual((d['stages']['images']['status'], d['stages']['assemble']['status']), ('done', 'stale'))
        self.assertEqual(self.client.post(f'/api/studio/{pk}/run/', {'only': ['assemble', 'render']},
                                          content_type='application/json').status_code, 200)
        self.assertTrue(wait_for(lambda: self.client.get(f'/api/studio/{pk}/').json()['stages']['render']['status']
                                 == 'done', timeout=120))
        sc = Script.objects.get(pk=pk)
        self.assertEqual(sc.data['production']['meta']['title'], 'Мой заголовок')
        self.assertEqual(sc.data['production']['output']['rendered_scenes'], 0)     # audio-only change: scene cache
