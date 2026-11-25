"""2B Smart Editing: cut plan per mode (pauses, emotional pauses, protected reactions, repeats, hesitations,
dull sentences, length floor, manual restore), Timeline IR (pieces, layouts, zooms, audio, captions in sync),
Hook teaser on an edited clip, real renders through the queue and the API (preview, apply, back to original)."""
import shutil
import subprocess
import tempfile
from unittest import mock, skipUnless

from django.core.files import File
from django.test import SimpleTestCase, TransactionTestCase, override_settings

from studio import exports, hooks, jobs, pipeline, smartedit
from studio import timeline as T
from studio.models import Project
from studio.timeline import adapters
from .helpers import HAVE_FFMPEG, sample_video
from .test_hooks import FakeAudio, seg
from .test_jobs_views import wait_for


def sig(segments, audio=None, screen=None):
    return hooks.Signals(segments, audio, screen)


def inside_word(t, words):
    return any(w['start'] + 0.01 < t < w['end'] - 0.01 for w in words)


TR = [seg(0.2, 'Сегодня я покажу одну интересную штуку.'),
      seg(4.5, 'Я я я думаю что это работает.'),                 # stutter after a 1.6 s pause
      seg(7.5, 'Э'),                                             # isolated hesitation
      seg(8.6, 'Вот смотрите как это происходит у нас в игре.'),
      seg(12.4, 'Потом мы немного походили по карте туда сюда.'),   # dull
      seg(17.6, 'И тут меня убили! Что?! Ничего себе!'),          # payoff after an emotional pause
      seg(20.5, 'Вот так бывает.')]


class Plan(SimpleTestCase):
    def test_natural_only_shortens_long_pauses(self):
        pl = smartedit.plan(sig(TR), 0.0, 23.0, 'natural', clip_min=10)
        self.assertEqual({c['reason'] for c in pl['cuts'] if c['applied']} - {'pause', 'silence'}, set())
        self.assertGreater(pl['stats']['removed'], 1.0)
        words = sig(TR).words
        for a, b in pl['keep']:
            self.assertFalse(inside_word(a, words) or inside_word(b, words))
        for x, y in zip(pl['keep'], pl['keep'][1:]):                     # what is left of a pause: never zero
            gap_words = [w for w in words if x[1] - 0.6 <= w['end'] <= x[1] + 0.01]
            self.assertTrue(gap_words)

    def test_dynamic_cuts_stutters_and_hesitations(self):
        pl = smartedit.plan(sig(TR), 0.0, 23.0, 'dynamic', clip_min=10)
        reasons = {c['reason'] for c in pl['cuts'] if c['applied']}
        self.assertTrue({'pause', 'repeat', 'filler'} <= reasons)
        kept = ' '.join(w['text'] for w in sig(TR).words
                        if any(a <= w['start'] and w['end'] <= b + 0.01 for a, b in pl['keep']))
        self.assertIn('я думаю что', kept)                                 # one take of the restart stays
        self.assertIsNone(__import__('re').search(r'(?i)(?<!\w)я я(?!\w)', kept))
        self.assertNotIn(' Э ', f' {kept} ')
        self.assertIn('убили', kept)

    def test_emotional_pause_is_kept_longer_and_reactions_are_never_cut(self):
        pl = smartedit.plan(sig(TR), 0.0, 23.0, 'dynamic', clip_min=10)
        before_payoff = [c for c in pl['cuts'] if c['start'] < 17.6 <= c['end'] + 0.7 and c['reason'] == 'pause']
        for c in before_payoff:                                          # 0.6 s of the pause is left before it
            self.assertGreaterEqual(17.6 - c['end'], 0.29)
        audio = FakeAudio(30, events=[{'start': 15.5, 'end': 17.0, 'kind': 'laughter', 'strength': 1.0}])
        pl2 = smartedit.plan(sig(TR, audio), 0.0, 23.0, 'aggressive', clip_min=5)
        for c in pl2['cuts']:
            if c['applied']:
                self.assertFalse(c['start'] < 17.0 and c['end'] > 15.5, c)  # the laughter stays

    def test_aggressive_drops_dull_sentences_but_keeps_setup_and_payoff(self):
        pl = smartedit.plan(sig(TR), 0.0, 23.0, 'aggressive', clip_min=8)
        boring = [c for c in pl['cuts'] if c['reason'] == 'boring' and c['applied']]
        self.assertTrue(any(c['start'] <= 4.6 for c in boring) or any(c['start'] >= 4.0 for c in boring))
        for c in boring:
            self.assertFalse(c['start'] < 17.6 < c['end'])
        self.assertLess(pl['stats']['result'], smartedit.plan(sig(TR), 0.0, 23.0, 'dynamic', clip_min=8)['stats']['result'])

    def test_length_floor_and_manual_restore(self):
        pl = smartedit.plan(sig(TR), 0.0, 23.0, 'aggressive', clip_min=30)
        self.assertGreaterEqual(pl['stats']['result'], 30 * smartedit.MODES['aggressive']['floor'] - 0.01)
        self.assertTrue(any(c.get('skipped') for c in pl['cuts']))
        full = smartedit.plan(sig(TR), 0.0, 23.0, 'dynamic', clip_min=10)
        rid = next(c['id'] for c in full['cuts'] if c['reason'] == 'repeat')
        back = smartedit.plan(sig(TR), 0.0, 23.0, 'dynamic', clip_min=10, restore={rid})
        self.assertFalse(next(c for c in back['cuts'] if c['id'] == rid)['applied'])
        self.assertGreater(back['stats']['result'], full['stats']['result'])

    def test_time_mapping(self):
        pl = {'keep': [[0.0, 2.0], [5.0, 6.0]]}
        self.assertEqual(smartedit.source_to_output(pl, 5.5), 2.5)
        self.assertIsNone(smartedit.source_to_output(pl, 3.0))


PLAN = {'segments': [{'start': 0.0, 'end': 12.0, 'type': 'action', 'path': [(0.0, 100.0, 0.0), (12.0, 700.0, 0.0)]},
                     {'start': 12.0, 'end': 23.0, 'type': 'game_cam', 'face_rel': (0.5, 0.5)}],
        'cam': {'x': .7, 'y': 0.0, 'w': .3, 'h': .3, 'corner': 'top_right', 'face_rel': (0.5, 0.5)}}


class IR(SimpleTestCase):
    def build(self, transition='none'):
        opts = exports.resolve({'preset': '720p', 'transition': transition})
        vinfo = {'width': 1280, 'height': 720, 'fps': 30.0, 'duration': 100.0, 'has_audio': True}
        info = {'fps': 30, 'transition': transition, 'loudness': {'target': -14.0, 'mode': 'two-pass'}}
        return adapters.clip_timeline(project_id=1, clip_id=3, source_name='sources/x.mp4', fingerprint='',
                                      clip_start=0.0, clip_end=23.0, plan=PLAN, opts=opts, vinfo=vinfo,
                                      render_info=info, transcript=TR)

    def test_pieces_audio_captions_in_sync(self):
        tl = self.build()
        pl = smartedit.plan(sig(TR), 0.0, 23.0, 'dynamic', clip_min=10)
        out = smartedit.to_timeline(tl, pl, 0.0, 'dynamic')
        self.assertEqual(T.validate(out), [])
        self.assertAlmostEqual(out.duration, pl['stats']['result'], delta=0.05)
        audio = out.tracks[1].items
        self.assertAlmostEqual(sum(a.duration for a in audio), out.duration, places=6)
        self.assertTrue(all(a.fade_in == 0.012 and a.ext['clipforge.loudness']['measure'] == [0.0, 23.0]
                            for a in audio))
        vids = [i for i in out.tracks[0].items if i.type == 'video']
        for v, a in zip(vids, audio):
            self.assertEqual((v.start, v.src_in), (a.start, a.src_in))   # picture and sound from the same moment
        words_src = [x for x in sig(TR).words if smartedit.source_to_output(pl, x['start']) is not None
                     and smartedit.source_to_output(pl, x['end']) is not None]
        mapped = [round(smartedit.source_to_output(pl, x['start']), 2) for x in words_src]
        got = [round(w.t0, 2) for w in out.tracks[2].items[0].words]
        self.assertEqual(len(got), len(mapped))
        self.assertTrue(all(abs(g - x) <= 0.011 for g, x in zip(got, mapped)), (got, mapped))

    def test_layouts_paths_and_zooms(self):
        tl = self.build()
        pl = smartedit.plan(sig(TR), 0.0, 23.0, 'dynamic', clip_min=10)
        out = smartedit.to_timeline(tl, pl, 0.0, 'dynamic')
        vids = [i for i in out.tracks[0].items if i.type == 'video']
        action = [v for v in vids if v.layout.type == 'action']
        cam = [v for v in vids if v.layout.type == 'game_cam']
        self.assertTrue(action and cam)
        for v in action:                                                # crop path re-timed to the piece
            x0 = v.layout.params['path'][0]
            self.assertEqual(x0[0], 0.0)
            self.assertAlmostEqual(x0[1], 100 + 600 * v.src_in / 12.0, delta=1.0)
        self.assertTrue(any(a.property == 'zoom' for v in action[1:] for a in v.animations))   # punch-ins hide cuts
        for v in cam:                     # webcam panel: no jump-cut zoom; a punch-in only aims at the face panel
            props = {a.property for a in v.animations}
            if props:
                self.assertIn('pan_y', props)
                self.assertTrue(all(k.value <= smartedit.MAX_ZOOM for a in v.animations if a.property == 'zoom'
                                    for k in a.keyframes))
                self.assertEqual(next(a for a in v.animations if a.property == 'zoom').keyframes[0].value, 1.0)
        nat = smartedit.to_timeline(tl, smartedit.plan(sig(TR), 0.0, 23.0, 'natural', clip_min=10), 0.0, 'natural')
        self.assertFalse(any(a.keyframes[0].value > 1.0 and len(a.keyframes) == 1
                             for v in nat.tracks[0].items if v.type == 'video' for a in v.animations))

    def test_scene_fade_kept_and_teaser_on_an_edited_clip(self):
        tl = self.build('fade')
        pl = smartedit.plan(sig(TR), 0.0, 23.0, 'natural', clip_min=10)        # the scene change is not cut
        out = smartedit.to_timeline(tl, pl, 0.0, 'natural')
        self.assertEqual(T.validate(out), [])
        self.assertEqual([i.kind for i in out.tracks[0].items if i.type == 'transition'], ['fade'])
        words = [w for w in sig(TR).words if 17.6 <= w['start'] and w['end'] <= 19.6]
        hooked = hooks.prepend_teaser(out, [17.6, 19.6], 0.0, src_words=words)
        self.assertEqual(T.validate(hooked), [])
        self.assertAlmostEqual(hooked.duration, out.duration + 2.0, delta=0.01)
        self.assertEqual(hooked.tracks[2].items[0].words[0].text, 'И')


# =====================================================================================================
def probe_duration(path):
    return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0',
                                 str(path)], capture_output=True, text=True).stdout)


def paused_transcript():
    """Sentences with 1.6 s pauses and a stutter, inside the 16 s sample video."""
    return [seg(0.3, 'Сегодня покажу одну очень интересную штуку.'), seg(4.0, 'Я я думаю это сработает.'),
            seg(7.6, 'Смотрите внимательно что сейчас будет.'), seg(11.6, 'Вот и всё на сегодня.')]


@skipUnless(HAVE_FFMPEG, 'needs ffmpeg')
class RenderAndApi(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()
        self.media = tempfile.mkdtemp()
        ov = override_settings(MEDIA_ROOT=self.media)
        ov.enable()
        self.addCleanup(ov.disable)
        self.addCleanup(shutil.rmtree, self.media, True)

    def run_full(self, **opts):
        p = Project.objects.create(title='t', clip_count=1, clip_min=8, clip_max=15, edit_mode='center',
                                   options=exports.resolve(dict({'preset': '720p', 'quality': 'small',
                                                                 'hook_engine': 'off'}, **opts)))
        with open(sample_video(), 'rb') as f:
            p.source.save('mix.mp4', File(f), save=True)
        with mock.patch.object(pipeline, 'transcribe', return_value=paused_transcript()):
            self.assertTrue(jobs.enqueue('full', p.pk))
            self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk), timeout=300))
        p.refresh_from_db()
        self.assertEqual(p.status, 'done', p.error)
        return p

    def wait(self, p):
        self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk), timeout=300))

    def test_dynamic_edit_renders_shorter_and_in_sync(self):
        p = self.run_full(smart_edit='dynamic')
        c = p.clips.get()
        pl = c.meta['edit']['plan']
        self.assertGreater(pl['stats']['removed'], 1.0)
        info = c.render_info
        self.assertEqual((info['renderer'], info['edit']['mode']), ('timeline', 'dynamic'))
        self.assertAlmostEqual(probe_duration(c.video.path), pl['stats']['result'], delta=0.15)
        tl = T.from_dict(info['timeline'])
        self.assertEqual(T.validate(tl), [])
        for w in tl.tracks[-1].items[0].words:                          # captions follow the cut sound
            self.assertLessEqual(w.t1, tl.duration + 0.01)

    def test_preview_apply_and_back_to_original(self):
        p = self.run_full()                                            # smart_edit off: as before
        c = p.clips.get()
        L = c.end - c.start
        self.assertNotIn('edit', c.render_info)
        self.assertAlmostEqual(probe_duration(c.video.path), L, delta=0.15)
        d = self.client.get(f'/api/clips/{c.pk}/edit/?mode=aggressive').json()
        self.assertEqual(d['mode'], 'aggressive')
        self.assertTrue(d['plan']['keep'])
        self.assertEqual(self.client.post(f'/api/clips/{c.pk}/edit/', {'mode': 'wild'},
                                          content_type='application/json').status_code, 400)
        off = self.client.get(f'/api/clips/{c.pk}/edit/?mode=dynamic&silence=0&zoom=0&max=2.5').json()
        self.assertEqual((off['choice']['silence'], off['choice']['zoom'], off['plan']['zooms']), (False, False, []))
        self.assertFalse([x for x in off['plan']['cuts'] if x['applied'] and x['reason'] in smartedit.SILENT])
        self.assertEqual(off['plan']['settings']['max'], 2.5)
        self.assertTrue({'pauses', 'zooms', 'original', 'result'} <= set(off['plan']['stats']))
        r = self.client.post(f'/api/clips/{c.pk}/edit/', {'mode': 'dynamic', 'zoom': False, 'overrides': {'max': 0.5}},
                             content_type='application/json')
        self.assertEqual(r.status_code, 200)
        self.wait(p)
        c.refresh_from_db()
        self.assertEqual((c.meta['edit']['zoom'], c.meta['edit']['overrides']), (False, {'max': 0.5}))
        self.assertEqual(c.meta['edit']['plan']['zooms'], [])
        r = self.client.post(f'/api/clips/{c.pk}/edit/', {'mode': 'dynamic'}, content_type='application/json')
        self.assertEqual(r.status_code, 200)
        self.wait(p)
        c.refresh_from_db()
        self.assertEqual(c.status, 'ready', c.error)
        c_plan = c.meta['edit']['plan']['stats']
        self.assertGreater(c_plan['removed'], 0.2)
        self.assertAlmostEqual(probe_duration(c.video.path), c_plan['result'], delta=0.15)
        r = self.client.post(f'/api/clips/{c.pk}/edit/', {'mode': 'off'}, content_type='application/json')
        self.wait(p)
        c.refresh_from_db()
        self.assertNotIn('edit', c.render_info)
        self.assertAlmostEqual(probe_duration(c.video.path), L, delta=0.15)
        self.assertEqual(c.meta['edit']['mode'], 'off')
