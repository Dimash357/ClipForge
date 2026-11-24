"""3A AI Script Generator: normalisation of model answers, template drafts (all formats / languages), LLM path
with a VRAM lease, one-scene regeneration and edits, Timeline IR, the job queue, the API and the pages."""
import json
from unittest import mock

from django.test import SimpleTestCase, TransactionTestCase

from studio import ai_provider, jobs, resources, scripts
from studio import timeline as T
from studio.models import Job, Script
from .test_jobs_views import wait_for
from .test_resources import GPU, manager

ST = scripts.settings_of({'idea': 'почему кошки приземляются на лапы', 'format': 'faceless', 'language': 'ru',
                          'duration': 40, 'genre': 'научпоп', 'style': 'динамичный', 'audience': 'школьники', 'id': 7})

MESSY = {'title': 'Кошки', 'characters': [{'name': 'Кот Барсик', 'description': 'рыжий', 'voice': 'бас'}],
         'scenes': [{'part': 'development', 'title': 'Начало', 'lines': ['Вы знали, что кошки умеют это?'],
                     'visual': 'cat falls in slow motion'},
                    {'lines': [{'speaker': 'Кот Барсик', 'text': 'Мяу, смотрите!'}, {'speaker': 'Призрак', 'text': 'Бу'}],
                     'visual': 'cat on a sofa', 'duration': 'abc'},
                    {'text': 'Всё дело в вестибулярном аппарате и гибком позвоночнике.', 'visual': 'x-ray spine'},
                    {'part': 'weird', 'lines': [{'text': 'Подпишитесь!'}], 'visual': 'logo', 'duration': 3}]}


class Document(SimpleTestCase):
    def test_messy_answer_becomes_a_valid_script(self):
        d = scripts.normalize(MESSY, ST, provider='ollama:qwen')
        parts = [s['part'] for s in d['scenes']]
        self.assertEqual((parts[0], parts[-1]), ('hook', 'ending'))
        self.assertIn('climax', parts)
        self.assertAlmostEqual(d['total'], 40.0, delta=1.0)
        barsik = d['characters'][0]['id']
        speakers = [l['speaker'] for l in d['scenes'][1]['lines']]
        self.assertEqual(speakers, [barsik, 'narrator'])                    # unknown speaker -> narrator
        for s in d['scenes']:
            self.assertGreaterEqual(s['duration'], scripts.spoken_seconds(s, 'ru') - 0.01)   # speech always fits
            self.assertEqual(s['storyboard']['prompt'], s['visual'])                       # ready for 3B
        again = scripts.normalize(MESSY, ST)
        self.assertEqual([s['id'] for s in again['scenes']], [s['id'] for s in d['scenes']])   # stable ids
        with self.assertRaises(ValueError):
            scripts.normalize({'scenes': [{}]}, ST)

    def test_template_for_every_format_and_language(self):
        for fmt in scripts.FORMATS:
            for lang in ('ru', 'en', 'de'):
                st = dict(ST, format=fmt, language=lang, duration=60)
                raw, note = scripts.template(st)
                d = scripts.normalize(raw, st, provider='template', note=note)
                self.assertEqual(len(d['scenes']), scripts.scene_count(60))
                self.assertEqual([p for p in scripts.PARTS if any(s['part'] == p for s in d['scenes'])],
                                 list(scripts.PARTS))
                if fmt == 'cartoon':
                    self.assertEqual(len(d['characters']), 2)
                    self.assertTrue(all(l['speaker'] != 'narrator' for s in d['scenes'] for l in s['lines']))

    def test_prompt_carries_the_settings(self):
        p = scripts.prompt_full(ST)
        for x in ('40 seconds', 'русский', 'научпоп', 'динамичный', 'школьники', 'кошки'):
            self.assertIn(x, p)


class Generation(SimpleTestCase):
    def test_llm_answer_is_used_with_a_vram_lease(self):
        m = manager(GPU)
        seen = {}

        def answer(system, user, temperature=0.8, post=None, timeout=None):
            seen['vram'] = m._used()['vram_mb']
            seen['timeout'] = timeout
            return MESSY, 'ollama:qwen2.5:7b'
        with mock.patch.object(resources, '_manager', m), mock.patch.object(ai_provider, 'available', return_value=True), \
                mock.patch.object(ai_provider, 'is_local', return_value=True), \
                mock.patch.object(ai_provider, 'complete_json', side_effect=answer):
            d = scripts.generate(ST)
        self.assertEqual(d['provider'], 'ollama:qwen2.5:7b')
        self.assertGreater(seen['vram'], 0)                                 # the local model's VRAM is reserved
        self.assertEqual(seen['timeout'], scripts.LLM_TIMEOUT)
        self.assertEqual(m._used()['vram_mb'], 0)                           # and released afterwards

    def test_no_model_or_bad_answer_gives_an_honest_template(self):
        with mock.patch.object(ai_provider, 'available', return_value=False):
            d = scripts.generate(ST)
        self.assertEqual(d['provider'], 'template')
        self.assertIn('Шаблонный черновик', d['note'])
        with mock.patch.object(ai_provider, 'available', return_value=True), \
                mock.patch.object(ai_provider, 'is_local', return_value=False), \
                mock.patch.object(ai_provider, 'complete_json', return_value=({'scenes': 'nope'}, 'openai:x')):
            d = scripts.generate(ST)
        self.assertEqual(d['provider'], 'template')
        self.assertIn('unusable', d['note'])


class SceneEdits(SimpleTestCase):
    def setUp(self):
        self.doc = scripts.normalize(MESSY, ST)

    def test_regenerate_only_one_scene(self):
        sid = self.doc['scenes'][2]['id']
        new_scene = {'title': 'Наука', 'lines': [{'speaker': 'narrator', 'text': 'Секрет — в рефлексе переворота.'}],
                     'visual': 'slow motion cat twist'}
        with mock.patch.object(ai_provider, 'available', return_value=True), \
                mock.patch.object(ai_provider, 'is_local', return_value=False), \
                mock.patch.object(ai_provider, 'complete_json', return_value=(new_scene, 'openai:m')):
            d = scripts.regenerate_scene(self.doc, sid, 'проще')
        s = d['scenes'][2]
        self.assertEqual((s['id'], s['part'], s['revision']), (sid, self.doc['scenes'][2]['part'], 2))
        self.assertEqual(s['lines'][0]['text'], 'Секрет — в рефлексе переворота.')
        self.assertEqual(s['history'][-1]['lines'], self.doc['scenes'][2]['lines'])
        for k in (0, 1, 3):
            self.assertEqual(d['scenes'][k]['lines'], self.doc['scenes'][k]['lines'])
        with mock.patch.object(ai_provider, 'available', return_value=False):
            t = scripts.regenerate_scene(self.doc, sid)
        self.assertEqual(t['provider_last'], 'template')
        self.assertNotEqual(t['scenes'][2]['lines'], self.doc['scenes'][2]['lines'])
        with self.assertRaises(KeyError):
            scripts.regenerate_scene(self.doc, 'sc_nope00')

    def test_manual_edit(self):
        sid = self.doc['scenes'][1]['id']
        barsik = self.doc['characters'][0]['id']
        d = scripts.edit_scene(self.doc, sid, {'lines': [{'speaker': 'Кот Барсик', 'text': 'Новая реплика'},
                                                         {'speaker': barsik, 'text': 'И ещё'}, {'text': ''}],
                                               'visual': 'new visual', 'duration': 99, 'locked': True, 'evil': 1})
        s = d['scenes'][1]
        self.assertEqual([l['speaker'] for l in s['lines']], [barsik, barsik])
        self.assertEqual((s['visual'], s['storyboard']['prompt'], s['locked']), ('new visual', 'new visual', True))
        self.assertLessEqual(s['duration'], scripts.MAX_SCENE)
        self.assertNotIn('evil', s)


class Timeline(SimpleTestCase):
    def test_script_as_timeline_ir(self):
        d = scripts.normalize(MESSY, dict(ST, format='cartoon'))
        tl = scripts.to_timeline(d, 7)
        self.assertEqual(T.validate(tl), [])
        self.assertEqual(tl.kind, 'cartoon')
        self.assertEqual([s.id for s in tl.scenes], [s['id'] for s in d['scenes']])
        self.assertEqual(tl.scenes[0].generation.prompt, d['scenes'][0]['storyboard']['prompt'])
        self.assertEqual(tl.scenes[1].characters, [d['characters'][0]['id']])
        words = next(t for t in tl.tracks if t.type == 'caption').items[0].words
        self.assertEqual(len(words), sum(len(l['text'].split()) for s in d['scenes'] for l in s['lines']))
        self.assertAlmostEqual(tl.duration, d['total'], delta=0.01)
        self.assertEqual(T.from_json(T.to_json(tl)), tl)


class ApiAndQueue(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()
        p = mock.patch.object(ai_provider, 'available', return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def wait_ready(self, pk):
        self.assertTrue(wait_for(lambda: Script.objects.get(pk=pk).status in ('ready', 'error'), timeout=30))
        sc = Script.objects.get(pk=pk)
        self.assertEqual(sc.status, 'ready', sc.error)
        return sc

    def test_create_generate_edit_regenerate_export(self):
        self.assertEqual(self.client.get('/scripts/').status_code, 200)
        r = self.client.post('/scripts/', {'idea': 'как работают чёрные дыры', 'format': 'education',
                                           'language': 'ru', 'duration': '30', 'genre': 'научпоп'})
        sc = Script.objects.get()
        self.assertRedirects(r, f'/scripts/{sc.pk}/', fetch_redirect_response=False)
        self.assertTrue(Job.objects.filter(kind='script', payload__script_id=sc.pk).exists())
        sc = self.wait_ready(sc.pk)
        self.assertEqual(sc.data['format'], 'education')
        self.assertContains(self.client.get(f'/scripts/{sc.pk}/'), 'script.js')
        api = self.client.get(f'/api/scripts/{sc.pk}/').json()
        self.assertEqual(api['data']['scenes'][0]['part'], 'hook')
        sid = api['data']['scenes'][1]['id']
        r = self.client.post(f'/api/scripts/{sc.pk}/scenes/{sid}/', {'visual': 'black hole close-up', 'locked': True},
                             content_type='application/json')
        self.assertEqual(r.json()['data']['scenes'][1]['visual'], 'black hole close-up')
        r = self.client.post(f'/api/scripts/{sc.pk}/scenes/{sid}/regenerate/', {'instruction': 'короче'},
                             content_type='application/json')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.post(f'/api/scripts/{sc.pk}/scenes/{sid}/', {'visual': 'x'},
                                          content_type='application/json').status_code, 409)   # busy while generating
        sc = self.wait_ready(sc.pk)
        self.assertEqual(sc.data['scenes'][1]['revision'], 3)
        self.assertEqual(self.client.post(f'/api/scripts/{sc.pk}/scenes/sc_nope00/', {},
                                          content_type='application/json').status_code, 404)
        tl = T.from_json(b''.join(self.client.get(f'/api/scripts/{sc.pk}/export/timeline/')).decode())
        self.assertEqual(T.validate(tl), [])
        doc = json.loads(self.client.get(f'/api/scripts/{sc.pk}/export/script/').content)
        self.assertEqual(doc['scenes'][1]['id'], sid)

    def test_full_regeneration_with_new_settings(self):
        sc = Script.objects.create(idea='тест', status='ready', data={})
        r = self.client.post(f'/api/scripts/{sc.pk}/generate/', {'duration': 90, 'format': 'cartoon', 'genre': 'юмор'},
                             content_type='application/json')
        self.assertEqual(r.status_code, 200)
        sc = self.wait_ready(sc.pk)
        self.assertEqual((sc.duration, sc.format, sc.genre), (90, 'cartoon', 'юмор'))
        self.assertEqual(len(sc.data['scenes']), scripts.scene_count(90))
        self.assertEqual(self.client.get(f'/api/scripts/{Script.objects.create(idea="x").pk}/export/script/').status_code,
                         400)
