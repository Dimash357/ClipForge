"""v5.3: AI Auto edit mode and Shorts titles/descriptions (AI provider + local fallback)."""
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from unittest import mock, skipUnless

from django.core.files import File
from django.test import Client, SimpleTestCase, TestCase, TransactionTestCase, override_settings

from studio import ai_provider, automode, editor, exports, jobs, pipeline, scenes, shorts_meta as sm
from studio.models import Clip, Project
from .helpers import HAVE_FFMPEG, fake_transcript, sample_video
from .test_jobs_views import wait_for

SLOW = os.getenv('CLIPFORGE_SKIP_SLOW') == '1'
CAM = {'x': .78, 'y': 0, 'w': .22, 'h': .29, 'corner': 'top_right', 'confidence': .8, 'face_rel': (.5, .5)}


# =====================================================================================================
# AI Auto decisions
# =====================================================================================================
def _samples(labels, act=None):
    out = []
    for i, lab in enumerate(labels):
        s = {'t': i * 0.5, 'faces': [], 'motion': 8.0, 'uniform': 0.1, 'edge': 0.05, 'text': [], 'lab': lab}
        if act:
            s['act'] = act(i)
        out.append(s)
    return out


def _decide(labels, scores=None, cam=None, w=1920, h=1080, act=None, hardsub=False):
    samples = _samples(labels, act)
    with mock.patch.object(automode.scn, 'classify_sample', lambda s, c: (s['lab'], 0.9)), \
            mock.patch.object(automode.scn, 'hardsub_summary', lambda s: {'present': hardsub}):
        return automode.choose({'samples': samples, 'cam': cam}, w, h, scores, None, 0.32)


class AutoModeRules(SimpleTestCase):
    def test_vertical_source(self):
        self.assertEqual(automode.choose({'samples': []}, 1080, 1920)[0], 'smart')

    def test_no_analysis_is_safe(self):
        mode, reason, _ = automode.choose(None, 1920, 1080)
        self.assertEqual(mode, 'smart')
        self.assertIn('недоступен', reason)

    def test_webcam_with_emotional_reaction(self):
        mode, reason, _ = _decide(['game_cam'] * 20, {'reaction': 78, 'audio': 70, 'action': 40}, CAM)
        self.assertEqual(mode, 'reaction')
        self.assertIn('реакция', reason)

    def test_webcam_with_intense_gameplay(self):
        mode, reason, _ = _decide(['game_cam'] * 20, {'reaction': 40, 'audio': 45, 'action': 80}, CAM)
        self.assertEqual(mode, 'gameplay_facecam')
        self.assertIn('без дубля', reason)

    def test_action_beats_moderate_reaction(self):
        self.assertEqual(_decide(['game_cam'] * 20, {'reaction': 62, 'audio': 50, 'action': 85}, CAM)[0],
                         'gameplay_facecam')

    def test_talking_screen_hardsub(self):
        self.assertEqual(_decide(['talking'] * 20)[0], 'smart')
        self.assertEqual(_decide(['screen'] * 20)[0], 'fullframe')
        self.assertEqual(_decide(['talking'] * 20, hardsub=True)[0], 'fullframe')

    def test_mixed_scenes_use_director(self):
        mode, reason, det = _decide(['game_cam'] * 10 + ['talking'] * 8 + ['gameplay'] * 2, cam=CAM)
        self.assertEqual(mode, 'director')
        self.assertIn('%', reason)
        self.assertIn('shares', det)

    def test_gameplay_without_webcam(self):
        spread = _decide(['gameplay'] * 20, act=lambda i: {'cx': 0.5, 'cy': 0.5, 'share': 0.2, 'sx': 0.3})
        self.assertEqual(spread[0], 'gameplay')                  # action all over the screen: nothing is cropped
        focused = _decide(['gameplay'] * 20, act=lambda i: {'cx': 0.4 + 0.01 * (i % 3), 'cy': 0.5, 'share': 0.05, 'sx': 0.05})
        self.assertEqual(focused[0], 'gameplay_facecam')         # one region: full-frame game following the action
        self.assertIn('слежением', focused[1])

    def test_registered_and_default_in_form(self):
        self.assertEqual(exports.MODE_KEYS[0], 'auto')
        self.assertIn('auto', pipeline.NEEDS_ANALYSIS)
        self.assertIn('auto', pipeline.CAM_MODES)


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class AutoModeOnVideo(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.src = str(sample_video())       # 0-4 game+cam, 4-8 talking, 8-12 gameplay, 12-16 screen
        cls.vinfo = editor.probe_video(cls.src)
        cls.opts = exports.resolve({'preset': '720p', 'quality': 'small'})

    def _plan(self, a, b, scores=None, cache=None):
        return editor.prepare(self.src, {'start': a, 'end': b, 'scores': scores or {}}, 'auto', 'auto', self.opts,
                              self.vinfo, cache=cache)

    def test_per_clip_choices(self):
        cases = [((0, 4, {'reaction': 80, 'audio': 70}), 'reaction', 'reaction_cam'),
                 ((0, 4, {'reaction': 30, 'action': 80}), 'gameplay_facecam', 'game_cam'),
                 ((4, 8, {}), 'smart', 'smart'),
                 ((12, 16, {}), 'fullframe', 'fullframe'),
                 ((0, 16, {}), 'director', None)]
        for (a, b, sc), mode, seg in cases:
            plan = self._plan(a, b, sc)
            self.assertEqual(plan['mode'], mode, (a, b, sc, plan['auto']))
            self.assertEqual(plan['auto']['mode'], mode)
            self.assertTrue(plan['auto']['reason'])
            self.assertTrue(any(n.startswith('AI Auto →') for n in plan['notes']))
            if seg:
                self.assertEqual(plan['segments'][0]['type'], seg)
        director = self._plan(0, 16)
        self.assertGreaterEqual(len({s['type'] for s in director['segments']}), 3)

    def test_decision_reuses_cached_analysis(self):
        cache = Path(tempfile.mkdtemp())
        self._plan(0, 4, cache=cache)
        with mock.patch.object(scenes, 'analyze', side_effect=AssertionError('must come from the cache')):
            plan = self._plan(0, 4, {'reaction': 80, 'audio': 70}, cache=cache)
        self.assertEqual(plan['mode'], 'reaction')

    def test_auto_render_has_no_duplicate_webcam(self):
        import cv2
        plan = self._plan(0, 4, {'reaction': 80, 'audio': 70})
        td = Path(tempfile.mkdtemp())
        info = editor.render(self.src, {'start': 0, 'end': 4}, fake_transcript(), td / 'o.mp4', td, self.opts, plan,
                             self.vinfo, 'auto')
        self.assertEqual(info['auto']['mode'], 'reaction')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '2', '-i', str(td / 'o.mp4'), '-frames:v', '1',
                        str(td / 'f.png')], check=True)
        im = cv2.imread(str(td / 'f.png'))
        Hc = int(0.4 * 1280)
        self.assertGreaterEqual(len(scenes.detect_faces_gray(cv2.cvtColor(im[:Hc], cv2.COLOR_BGR2GRAY))), 1)
        self.assertEqual(len(scenes.detect_faces_gray(cv2.cvtColor(im[Hc:], cv2.COLOR_BGR2GRAY))), 0)


# =====================================================================================================
# Shorts title / description
# =====================================================================================================
RU = 'Ну всё, иду на мост. ЧТО?! Меня опять убили! Да ладно, чат, вы это видели? Ахахаха.'


def ru_ctx(**kw):
    base = dict(lang='ru', audio_events=[{'kind': 'scream'}, {'kind': 'laughter'}],
                screen_events=[{'kind': 'fail_screen', 't': 3}], scene_types=['reaction_cam'], duration=18)
    base.update(kw)
    return sm.build_context(RU, **base)


class LocalTitles(SimpleTestCase):
    def setUp(self):
        ai_provider.reset()
        self.addCleanup(ai_provider.reset)

    def test_caps_emoji_truthful(self):
        with mock.patch.dict(os.environ, {'CLIPFORGE_AI': 'off'}):
            r = sm.generate(ru_ctx())
        letters = [ch for ch in sm.EMOJI_RX.sub('', r['title']) if ch.isalpha()]
        self.assertTrue(letters and all(ch.isupper() for ch in letters), r['title'])
        self.assertTrue(1 <= sm.emoji_count(r['title']) <= 3, r['title'])
        self.assertLessEqual(len(r['title']), 100)
        self.assertIn('МЕНЯ ОПЯТЬ УБИЛИ', r['title'])            # quotes what was actually said
        self.assertIn('💀', r['title'])
        self.assertEqual(r['provider'], 'local')
        self.assertEqual(r['hashtags'][0], '#shorts')
        self.assertLessEqual(len(r['hashtags']), 8)
        self.assertIn('#фейл', r['hashtags'])
        self.assertTrue(r['text'].endswith(' '.join(r['hashtags'])))

    def test_regenerate_gives_a_different_title(self):
        with mock.patch.dict(os.environ, {'CLIPFORGE_AI': 'off'}):
            a = sm.generate(ru_ctx())
            b = sm.generate(ru_ctx(), variant=1, avoid=a['history'])
        self.assertNotEqual(a['title'], b['title'])

    def test_english(self):
        ctx = sm.build_context("Okay let's go. No way! Did you see that? I died again lol.", None, scene_types=['game_cam'])
        with mock.patch.dict(os.environ, {'CLIPFORGE_AI': 'off'}):
            r = sm.generate(ctx)
        self.assertEqual(r['lang'], 'en')
        self.assertIn('#gaming', r['hashtags'])
        self.assertIsNone(__import__('re').search('[А-Яа-я]', r['title'] + r['text']))

    def test_streamer_only_when_verified(self):
        self.assertEqual(sm.verified_streamer('Kuplinov', None), 'Kuplinov')
        self.assertEqual(sm.verified_streamer('', {'title': 'Куплинов играет в хоррор', 'author': 'Нарезки'}), 'Kuplinov')
        self.assertEqual(sm.verified_streamer('', {'title': 'xQc reacts', 'author': 'xQcOW'}), 'xQc')
        self.assertIsNone(sm.verified_streamer('', {'title': 'Смешные моменты недели', 'author': 'Clips'}))
        self.assertIsNone(sm.verified_streamer('', None))
        with mock.patch.dict(os.environ, {'CLIPFORGE_AI': 'off'}):
            plain = sm.generate(ru_ctx())
            named = sm.generate(ru_ctx(streamer='Kuplinov'))
        self.assertNotIn('KUPLINOV', plain['title'])
        self.assertIn('KUPLINOV', named['title'])
        self.assertIn('#kuplinov', named['hashtags'])

    def test_quiet_clip_without_cues_still_gets_a_title(self):
        ctx = sm.build_context('Сегодня мы снова заходим на карту возле старого моста и проверяем углы.', 'ru')
        with mock.patch.dict(os.environ, {'CLIPFORGE_AI': 'off'}):
            r = sm.generate(ctx)
        self.assertTrue(r['title'])
        self.assertTrue(1 <= sm.emoji_count(r['title']) <= 3)


class FakeResp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f'HTTP {self.status_code}')

    def json(self):
        return self.payload


def openai_answer(obj):
    return FakeResp({'choices': [{'message': {'content': 'Here you go:\n```json\n' + json.dumps(obj, ensure_ascii=False) + '\n```'}}]})


AI_ENV = {'CLIPFORGE_AI': 'auto', 'CLIPFORGE_AI_URL': 'http://ai.local/v1', 'CLIPFORGE_AI_MODEL': 'm', 'CLIPFORGE_AI_KEY': 'k'}


class AiTitles(SimpleTestCase):
    def setUp(self):
        ai_provider.reset()
        self.addCleanup(ai_provider.reset)

    def _gen(self, answer, ctx=None, **kw):
        calls = []

        def post(url, **k):
            calls.append((url, k))
            if isinstance(answer, Exception):
                raise answer
            return answer
        with mock.patch.dict(os.environ, AI_ENV), mock.patch('requests.post', post):
            return sm.generate(ctx or ru_ctx(), **kw), calls

    def test_valid_answer_is_used_and_normalised(self):
        r, calls = self._gen(openai_answer({'title': 'Меня опять убили на мосту 💀😂🔥🎮😱',
                                            'description': 'Стример снова падает с моста и не верит своим глазам.',
                                            'hashtags': ['shorts', '#фейл', '#игры', 'смешно']}))
        self.assertEqual(r['provider'], 'openai:m')
        self.assertTrue(r['title'].startswith('МЕНЯ ОПЯТЬ УБИЛИ НА МОСТУ'))
        self.assertEqual(sm.emoji_count(r['title']), 3)
        self.assertEqual(r['hashtags'][0], '#shorts')
        url, k = calls[0]
        self.assertEqual(url, 'http://ai.local/v1/chat/completions')
        self.assertEqual(k['headers']['Authorization'], 'Bearer k')
        self.assertEqual(k['timeout'][0], 3.05)
        self.assertIn('Меня опять убили', k['json']['messages'][1]['content'])

    def test_invented_person_rejected(self):
        r, _ = self._gen(openai_answer({'title': 'xQc В ШОКЕ ОТ МОСТА 💀', 'description': 'xQc снова падает с моста.',
                                        'hashtags': ['#shorts']}))
        self.assertEqual(r['provider'], 'local')
        self.assertNotIn('XQC', r['title'])

    def test_invented_number_rejected(self):
        r, _ = self._gen(openai_answer({'title': 'УМЕР 100 РАЗ ПОДРЯД 💀', 'description': 'Сто смертей подряд на мосту.',
                                        'hashtags': ['#shorts']}))
        self.assertEqual(r['provider'], 'local')

    def test_wrong_language_rejected(self):
        r, _ = self._gen(openai_answer({'title': 'KILLED AGAIN ON THE BRIDGE 💀', 'description': 'He falls again.',
                                        'hashtags': ['#shorts']}))
        self.assertEqual(r['provider'], 'local')

    def test_dead_api_trips_the_breaker_and_never_blocks(self):
        r, calls = self._gen(ConnectionError('refused'))
        self.assertEqual(r['provider'], 'local')
        self.assertEqual(len(calls), 1)
        self.assertTrue(ai_provider.paused())
        r2, calls2 = self._gen(openai_answer({'title': 'X', 'description': 'y'}))
        self.assertEqual(calls2, [], 'paused provider must not be called again')
        self.assertEqual(r2['provider'], 'local')

    def test_real_unreachable_endpoint_is_fast(self):
        env = dict(AI_ENV, CLIPFORGE_AI_URL='http://127.0.0.1:9/v1', CLIPFORGE_AI_TIMEOUT='5')
        t0 = time.monotonic()
        with mock.patch.dict(os.environ, env):
            r = sm.generate(ru_ctx())
        self.assertLess(time.monotonic() - t0, 8)
        self.assertEqual(r['provider'], 'local')

    def test_ollama_and_off(self):
        with mock.patch.dict(os.environ, {'CLIPFORGE_AI': 'auto', 'USE_OLLAMA': '1', 'CLIPFORGE_AI_URL': ''}):
            self.assertEqual(ai_provider.config()['kind'], 'ollama')
        with mock.patch.dict(os.environ, dict(AI_ENV, CLIPFORGE_AI='off')):
            self.assertIsNone(ai_provider.config()['kind'])
        with mock.patch.dict(os.environ, {'CLIPFORGE_AI': 'auto', 'USE_OLLAMA': '0', 'CLIPFORGE_AI_URL': ''}):
            self.assertIsNone(ai_provider.config()['kind'])


# =====================================================================================================
# storage, endpoint, pipeline
# =====================================================================================================
class ShortMetaEndpoint(TestCase):
    def setUp(self):
        ai_provider.reset()
        self.addCleanup(ai_provider.reset)
        self.media = tempfile.mkdtemp()
        self.ov = override_settings(MEDIA_ROOT=self.media)
        self.ov.enable()
        self.addCleanup(self.ov.disable)
        self.p = Project.objects.create(title='t', status='done', options={'streamer': 'Kuplinov'},
                                        transcript_meta={'detected_language': 'ru'})
        self.c = Clip.objects.create(project=self.p, start=10, end=28, score=70, status='ready', text=RU,
                                     render_info={'segments': [{'type': 'reaction_cam'}],
                                                  'auto': {'mode': 'reaction', 'label': 'Reaction Priority', 'reason': 'r'}})

    def test_stored_once_and_regenerated_on_request(self):
        with mock.patch.dict(os.environ, {'CLIPFORGE_AI': 'off'}):
            a = sm.ensure_for_clip(self.p, self.c)
            self.c.save()
            with mock.patch.object(sm, 'generate', side_effect=AssertionError('must reuse the stored result')):
                self.assertEqual(sm.ensure_for_clip(self.p, self.c), a)
            r = Client().post(f'/api/clips/{self.c.pk}/short-meta/', '{"regenerate": true}', content_type='application/json')
        self.assertEqual(r.status_code, 200)
        b = r.json()['short']
        self.assertNotEqual(a['title'], b['title'])
        self.c.refresh_from_db()
        self.assertEqual(self.c.meta['short']['title'], b['title'])
        self.assertEqual(self.c.title, b['title'])
        self.assertIn('KUPLINOV', ' '.join(self.c.meta['short']['history']))
        j = Client().get(f'/api/projects/{self.p.pk}/status/').json()['clips'][0]
        self.assertEqual(j['short']['title'], b['title'])
        self.assertEqual(j['auto']['label'], 'Reaction Priority')

    def test_refused_while_busy_or_not_ready(self):
        with mock.patch.object(jobs, 'is_active', return_value=True):
            self.assertEqual(Client().post(f'/api/clips/{self.c.pk}/short-meta/').status_code, 409)
        self.c.status = 'candidate'
        self.c.save()
        self.assertEqual(Client().post(f'/api/clips/{self.c.pk}/short-meta/').status_code, 400)

    def test_streamer_option_sanitised(self):
        self.assertEqual(exports.resolve({'streamer': '<b>Kuplinov</b>!!'})['streamer'], 'bKuplinovb')
        self.assertEqual(len(exports.resolve({'streamer': 'x' * 200})['streamer']), 40)

    def test_form_default_mode_is_auto(self):
        html = Client().get('/').content.decode()
        self.assertIn('<option value="auto" selected>', html)
        self.assertIn('name="streamer"', html)


@skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')
class PipelineAutoAndTitles(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()
        ai_provider.reset()
        self.addCleanup(ai_provider.reset)     # this test trips the breaker on purpose (dead API)
        self.media = tempfile.mkdtemp()
        self.ov = override_settings(MEDIA_ROOT=self.media)
        self.ov.enable()
        self.addCleanup(self.ov.disable)
        self.addCleanup(shutil.rmtree, self.media, True)

    def run_job(self, kind, pk, **payload):
        self.assertTrue(jobs.enqueue(kind, pk, **payload))
        self.assertTrue(wait_for(lambda: not jobs.is_active(pk), timeout=300))

    def test_auto_mode_and_titles_end_to_end(self):
        p = Project(title='t', clip_count=2, clip_min=5, clip_max=9, edit_mode='auto',
                    options=exports.resolve({'preset': '720p', 'quality': 'small'}))
        with open(sample_video(), 'rb') as f:
            p.source.save('mix.mp4', File(f), save=True)
        env = dict(AI_ENV, CLIPFORGE_AI_URL='http://127.0.0.1:9/v1', CLIPFORGE_AI_TIMEOUT='4')   # dead API
        with mock.patch.object(pipeline, 'transcribe', return_value=fake_transcript()), mock.patch.dict(os.environ, env):
            t0 = time.monotonic()
            self.run_job('full', p.pk)
            p.refresh_from_db()
            self.assertEqual(p.status, 'done', p.error)
            self.assertLess(time.monotonic() - t0, 280)
            clips = list(p.clips.all())
            for c in clips:
                self.assertEqual(c.render_info['mode'], 'auto')
                self.assertIn(c.render_info['auto']['mode'], exports.MODE_KEYS)
                self.assertTrue(c.render_info['auto']['reason'])
                s = c.meta['short']
                self.assertEqual(s['provider'], 'local')            # dead API -> local fallback, job not stuck
                self.assertEqual(c.title, s['title'][:240])
                self.assertTrue(1 <= sm.emoji_count(s['title']) <= 3)
                self.assertTrue(c.meta['titles'])                   # the older title list is still there
            first = clips[0]
            stamp = first.meta['short']['generated_at']
            # re-edit with another subtitle style: same content -> the stored title is reused, not regenerated
            self.run_job('render', p.pk, clip_ids=[first.pk], bounds={}, overrides={'subtitle_style': 'karaoke'})
            first.refresh_from_db()
            self.assertEqual(first.meta['short']['generated_at'], stamp)
            # manual mode still possible
            self.run_job('render', p.pk, clip_ids=[first.pk], bounds={}, mode='gameplay_facecam')
            first.refresh_from_db()
            self.assertEqual(first.render_info['mode'], 'gameplay_facecam')
            self.assertIsNone(first.render_info.get('auto'))
