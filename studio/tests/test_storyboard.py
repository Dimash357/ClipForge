"""3B AI Storyboard: bibles, frames and timing, prompts that keep characters / style consistent, continuity, locked
frames, stale scenes, edits / reorder / one-frame regeneration, LLM path, Timeline IR for 3C, queue and API."""
import json
from unittest import mock

from django.test import SimpleTestCase, TransactionTestCase

from studio import ai_provider, jobs, scripts, storyboard as sb
from studio import timeline as T
from studio.models import Script
from .test_jobs_views import wait_for

ST = scripts.settings_of({'idea': 'лис встречает сову в ночном лесу', 'format': 'cartoon', 'language': 'ru',
                          'duration': 40, 'style': 'тёплый', 'id': 3})


def script_doc():
    with mock.patch.object(ai_provider, 'available', return_value=False):
        return scripts.generate(ST)


def built(doc=None):
    with mock.patch.object(ai_provider, 'available', return_value=False):
        return sb.build(doc or script_doc())


def frames(d):
    return [f for s in d['scenes'] for f in d['storyboard']['scenes'][s['id']]['frames']]


class Build(SimpleTestCase):
    def test_rules_storyboard(self):
        d = built()
        s = d['storyboard']
        self.assertEqual(s['provider'], 'rules')
        self.assertTrue(s['note'])
        self.assertEqual(set(s['characters']), {c['id'] for c in d['characters']})
        self.assertIn('тёплый', s['style']['look'])
        for scene in d['scenes']:
            fr = s['scenes'][scene['id']]['frames']
            self.assertEqual(len(fr), max(1, min(sb.MAX_FRAMES, round(scene['duration'] / sb.FRAME_SECONDS))))
            self.assertAlmostEqual(sum(f['duration'] for f in fr), scene['duration'], delta=0.02)
        ids = [f['id'] for f in frames(d)]
        self.assertEqual(len(ids), len(set(ids)))

    def test_prompts_carry_the_bibles(self):
        d = built()
        s = d['storyboard']
        for f in frames(d):
            self.assertIn(s['style']['look'], f['prompt'])
            self.assertIn('9:16', f['prompt'])
            self.assertIn('deformed', f['negative'])
            for cid in f['characters']:
                self.assertIn(sb.char_text(s['characters'][cid]), f['prompt'])        # same look every time
        ch = next(iter(s['characters'].values()))
        self.assertIn('character sheet', ch['ref_prompt'])

    def test_continuity_goes_into_the_next_scene_prompt(self):
        d = built()
        first = d['storyboard']['scenes'][d['scenes'][0]['id']]['frames'][-1]['action']
        p = sb.prompt_frames(d, d['scenes'][1], d['storyboard'], first)
        self.assertIn(first, p)

    def test_llm_bibles_and_frames(self):
        doc = script_doc()
        fox = doc['characters'][0]['name']
        answers = [({'style': {'look': 'watercolor storybook', 'palette': ['#112233', 'red', '#AABBCC'],
                               'lighting': 'moonlight'},
                     'characters': [{'name': fox, 'appearance': 'red fox, green eyes', 'clothing': 'blue scarf'}]},
                    'ollama:qwen')]
        answers += [({'location': 'night forest', 'frames': [
            {'shot': 'wide shot', 'action': 'fox walks among trees', 'camera': 'slow push-in', 'characters': [fox, 'Ghost'],
             'duration': 2}, {'shot': 'close-up', 'action': 'owl blinks', 'camera': 'static'}]}, 'ollama:qwen')] * 10
        with mock.patch.object(scripts, '_ask', side_effect=answers):
            d = sb.build(doc)
        s = d['storyboard']
        self.assertEqual(s['provider'], 'ollama:qwen')
        self.assertEqual(s['style']['palette'], ['#112233', '#AABBCC'])                 # invalid colour dropped
        fox_id = doc['characters'][0]['id']
        self.assertEqual(s['characters'][fox_id]['clothing'], 'blue scarf')
        f = s['scenes'][doc['scenes'][0]['id']]['frames'][0]
        self.assertEqual(f['characters'], [fox_id])                                    # unknown name dropped
        self.assertIn('blue scarf', f['prompt'])
        self.assertIn('night forest', f['prompt'])


class Editing(SimpleTestCase):
    def setUp(self):
        self.d = built()
        self.s0 = self.d['scenes'][0]['id']
        self.f0 = self.d['storyboard']['scenes'][self.s0]['frames'][0]

    def test_locked_frames_survive_rebuilds_and_stale_scenes(self):
        d = sb.edit_frame(self.d, self.f0['id'], {'locked': True, 'action': 'мой кадр'})
        again = built(d)
        kept = again['storyboard']['scenes'][self.s0]['frames']
        self.assertEqual(sum(f['id'] == self.f0['id'] for f in kept), 1)
        self.assertEqual(next(f for f in kept if f['id'] == self.f0['id'])['action'], 'мой кадр')
        changed = scripts.edit_scene(again, self.d['scenes'][2]['id'], {'visual': 'new place'})
        self.assertEqual(sb.mark_stale(changed), [self.d['scenes'][2]['id']])
        other = self.d['scenes'][3]['id']
        with mock.patch.object(ai_provider, 'available', return_value=False):
            upd = sb.build(changed, only={self.d['scenes'][2]['id']})
        self.assertEqual(sb.mark_stale(upd), [])
        self.assertEqual(upd['storyboard']['scenes'][other], changed['storyboard']['scenes'][other])

    def test_edit_move_and_manual_prompt(self):
        d = sb.edit_frame(self.d, self.f0['id'], {'shot': 'extreme close-up', 'duration': 4})
        f = d['storyboard']['scenes'][self.s0]['frames'][0]
        self.assertTrue(f['prompt'].startswith('extreme close-up'))
        self.assertEqual(f['revision'], 2)
        d = sb.edit_frame(d, self.f0['id'], {'prompt': 'my own prompt'})
        d = sb.edit_bibles(d, style={'look': 'pixel art'})
        fr = d['storyboard']['scenes'][self.s0]['frames']
        self.assertEqual(fr[0]['prompt'], 'my own prompt')                             # hand-written stays
        self.assertIn('pixel art', fr[1]['prompt'])                                     # the rest follows the bible
        moved = sb.move_frame(d, self.f0['id'], 1)
        mf = moved['storyboard']['scenes'][self.s0]['frames']
        self.assertEqual(mf[1]['id'], self.f0['id'])
        self.assertEqual(mf[0]['start'], 0.0)
        with self.assertRaises(KeyError):
            sb.edit_frame(d, 'fr_nope00', {})

    def test_regenerate_one_frame(self):
        with mock.patch.object(ai_provider, 'available', return_value=False):
            d = sb.regenerate_frame(self.d, self.f0['id'], 'крупнее')
        fr = d['storyboard']['scenes'][self.s0]['frames']
        self.assertEqual((fr[0]['id'], fr[0]['revision']), (self.f0['id'], 2))
        self.assertNotEqual((fr[0]['shot'], fr[0]['seed']), (self.f0['shot'], self.f0['seed']))
        self.assertEqual(fr[0]['history'][-1]['shot'], self.f0['shot'])
        self.assertEqual(fr[1:], self.d['storyboard']['scenes'][self.s0]['frames'][1:])
        self.assertEqual(d['storyboard']['provider_last'], 'rules')


class Timeline(SimpleTestCase):
    def test_frames_become_image_items_for_3c(self):
        d = built()
        tl = scripts.to_timeline(d, 3)
        self.assertEqual(T.validate(tl), [])
        imgs = [i for i in tl.tracks[0].items if i.type == 'image']
        self.assertEqual(len(imgs), len(frames(d)))
        a = tl.asset(imgs[0].asset)
        f0 = frames(d)[0]
        self.assertEqual((a.uri, a.generation.prompt, a.generation.seed), (f'gen:frames/{f0["id"]}.png', f0['prompt'],
                                                                           f0['seed']))
        self.assertTrue(all(c.refs and tl.asset(c.refs[0]).uri.startswith('gen:characters/') for c in tl.characters))
        voice = next(t for t in tl.tracks if t.name.startswith('Озвучка'))
        self.assertTrue(voice.items)
        self.assertEqual(tl.asset(voice.items[0].asset).generation.provider, 'tts')
        pushes = [i for i in imgs if any(x.property == 'zoom' and len(x.keyframes) == 2 for x in i.animations)]
        self.assertTrue(pushes)
        self.assertEqual(T.validate(scripts.to_timeline(d, 3, storyboard=False)), [])


class Api(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()
        p = mock.patch.object(ai_provider, 'available', return_value=False)
        p.start()
        self.addCleanup(p.stop)
        self.sc = Script.objects.create(idea='лис и сова', format='cartoon', status='ready', data=script_doc())

    def wait(self):
        self.assertTrue(wait_for(lambda: Script.objects.get(pk=self.sc.pk).status != 'generating', timeout=30))
        sc = Script.objects.get(pk=self.sc.pk)
        self.assertEqual(sc.status, 'ready', sc.error)
        return sc

    def test_build_edit_reorder_regenerate_export(self):
        self.assertContains(self.client.get(f'/scripts/{self.sc.pk}/storyboard/'), 'storyboard.js')
        self.assertEqual(self.client.post(f'/api/scripts/{self.sc.pk}/storyboard/').status_code, 200)
        self.assertEqual(self.client.post(f'/api/scripts/{self.sc.pk}/storyboard/').status_code, 409)   # running
        sc = self.wait()
        d = self.client.get(f'/api/scripts/{sc.pk}/storyboard/').json()
        self.assertEqual(d['stale'], [])
        s0 = sc.data['scenes'][0]['id']
        fr = sc.data['storyboard']['scenes'][s0]['frames']
        url = f'/api/scripts/{sc.pk}/storyboard/frames/{fr[0]["id"]}/'
        self.assertEqual(self.client.post(url, {'action': 'fox jumps', 'locked': True},
                                          content_type='application/json').status_code, 200)
        self.assertEqual(self.client.post(url + 'move/', {'to': 1}, content_type='application/json').status_code, 200)
        sc.refresh_from_db()
        moved = sc.data['storyboard']['scenes'][s0]['frames']
        self.assertEqual((moved[1]['id'], moved[1]['action'], moved[1]['locked']), (fr[0]['id'], 'fox jumps', True))
        self.assertEqual(self.client.post(f'/api/scripts/{sc.pk}/storyboard/frames/{fr[1]["id"]}/regenerate/',
                                          {'instruction': 'ближе'}, content_type='application/json').status_code, 200)
        sc = self.wait()
        self.assertEqual(next(f for f in sc.data['storyboard']['scenes'][s0]['frames']
                              if f['id'] == fr[1]['id'])['revision'], 2)
        r = self.client.post(f'/api/scripts/{sc.pk}/storyboard/bible/', {'style': {'look': 'pixel art'}},
                             content_type='application/json')
        self.assertEqual(r.status_code, 200)
        tl = T.from_json(self.client.get(f'/api/scripts/{sc.pk}/export/timeline/').content.decode())
        self.assertEqual(T.validate(tl), [])
        self.assertTrue(any(i.type == 'image' for i in tl.tracks[0].items))
        self.assertIn('pixel art', json.dumps(T.to_dict(tl)))
        self.assertEqual(self.client.post(f'/api/scripts/{sc.pk}/storyboard/frames/fr_nope00/', {},
                                          content_type='application/json').status_code, 404)

    def test_needs_a_script_first(self):
        empty = Script.objects.create(idea='x', data={})
        self.assertEqual(self.client.post(f'/api/scripts/{empty.pk}/storyboard/').status_code, 400)
        self.assertEqual(self.client.post(f'/api/scripts/{empty.pk}/storyboard/bible/', {},
                                          content_type='application/json').status_code, 400)
