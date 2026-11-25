"""Smart Editing v2: silence / dead-air removal with audio evidence, protected non-speech moments (laughter, game
events, screen events, unrecognised voice), word boundaries, Smart Auto Zoom (peaks, cooldown, rate, 1.3x cap, face
focus, smooth return), switches / overrides, caption sync, and a real FFmpeg end-to-end render (duration, picture /
sound sync measured on the output audio, zoom visible in the frames)."""
import json
import subprocess
import tempfile
from pathlib import Path
from unittest import skipUnless

import numpy as np
from django.test import SimpleTestCase

from studio import audio_analysis, exports, hooks, smartedit
from studio import timeline as T
from studio.timeline import adapters, renderer
from .helpers import HAVE_FFMPEG
from .test_hooks import FakeAudio, FakeScreen

SR = 16000


def tone(dur, f0=150.0, amp=0.25):
    t = np.arange(int(SR * dur)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 5)) * (0.65 + 0.35 * np.sin(2 * np.pi * 4 * t))
    return (amp * x).astype(np.float32)


def quiet(dur):
    return (np.random.RandomState(1).randn(int(SR * dur)) * 0.0005).astype(np.float32)


def noise(dur, amp=0.5):
    return (np.random.RandomState(2).randn(int(SR * dur)) * amp).astype(np.float32)


def audio(parts):
    """parts: [(kind, seconds)] with kind 's' speech, 'q' silence, 'n' loud noise (game event)."""
    x = np.concatenate([{'s': tone, 'q': quiet, 'n': noise}[k](d) for k, d in parts])
    return audio_analysis.profile_from_samples(x, SR)


def words(*spans):
    return [{'start': a, 'end': b, 'text': t} for a, b, t in spans]


def segs(ws):
    return [{'start': ws[0]['start'], 'end': ws[-1]['end'], 'text': ' '.join(w['text'] for w in ws), 'words': ws}]


def applied(pl, reasons=None):
    return [c for c in pl['cuts'] if c['applied'] and (reasons is None or c['reason'] in reasons)]


def overlaps(c, a, b):
    return c['start'] < b and c['end'] > a


class Silence(SimpleTestCase):
    def test_whisper_stretched_word_over_silence_is_trimmed_never_the_speech(self):
        # speech 0.5-1.5, silence 1.5-3.6, speech 3.6-4.6; Whisper stretched the first word to 3.3
        prof = audio([('q', 0.5), ('s', 1.0), ('q', 2.1), ('s', 1.0), ('q', 0.4)])
        ws = words((0.5, 3.3, 'Смотрите'), (3.4, 4.6, 'внимательно'))
        old = smartedit.plan(hooks.Signals(segs(ws)), 0.0, 5.0, 'dynamic', clip_min=3)
        self.assertEqual(applied(old, smartedit.SILENT), [])            # words alone: no gap to cut
        pl = smartedit.plan(hooks.Signals(segs(ws), prof), 0.0, 5.0, 'dynamic', clip_min=3)
        cut = applied(pl, smartedit.SILENT)
        self.assertTrue(cut and cut[0]['reason'] == 'silence', pl['cuts'])
        for c in cut:                                                   # nothing of the speech is removed
            self.assertFalse(overlaps(c, 0.5, 1.5) or overlaps(c, 3.6, 4.6), c)
            self.assertGreaterEqual(smartedit.quiet_share(hooks.Signals(segs(ws), prof), c['start'], c['end']), 0.85)
        self.assertGreater(pl['stats']['removed'], 1.0)
        self.assertEqual(pl['stats']['pauses'], len(cut))

    def test_word_minimum_length_and_padding(self):
        prof = audio([('q', 0.5), ('s', 0.1), ('q', 3.0), ('s', 1.0)])      # a very short sound for a long word
        ws = words((0.5, 2.5, 'Невероятно'), (3.6, 4.6, 'да'))
        sig = hooks.Signals(segs(ws), prof)
        s, e = smartedit.speech_bounds(sig, ws[0])
        self.assertGreaterEqual(e - s, smartedit.min_word(ws[0]) - 1e-6)
        pl = smartedit.plan(sig, 0.0, 4.6, 'aggressive', clip_min=2)
        for c in applied(pl):
            self.assertFalse(overlaps(c, s, s + smartedit.min_word(ws[0])), c)

    def test_laughter_without_speech_is_kept_with_its_context(self):
        ws = words((0.3, 0.8, 'Это'), (0.9, 1.5, 'шутка.'), (7.0, 7.5, 'Ладно.'))
        aud = FakeAudio(10, events=[{'start': 3.0, 'end': 4.2, 'kind': 'laughter', 'strength': 0.9}])
        pl = smartedit.plan(hooks.Signals(segs(ws), aud), 0.0, 8.0, 'aggressive', clip_min=2)
        self.assertTrue(applied(pl, smartedit.SILENT))                  # the dead air around it goes
        for c in applied(pl):
            self.assertFalse(overlaps(c, 2.6, 4.9), c)                  # laughter + the beat before + reaction tail

    def test_game_event_and_screen_event_without_speech_are_kept(self):
        prof = audio([('s', 1.0), ('q', 2.0), ('n', 0.6), ('q', 3.4), ('s', 1.0)])
        ws = words((0.0, 1.0, 'Смотри'), (7.0, 8.0, 'вот'))
        screen = FakeScreen([{'t': 5.5, 'kind': 'flash'}])
        sig = hooks.Signals(segs(ws), prof, screen)
        self.assertTrue(any(e['kind'] in smartedit.KEEP_EVENTS for e in prof.events_in(2.8, 3.8)), prof.events)
        pl = smartedit.plan(sig, 0.0, 8.0, 'aggressive', clip_min=2)
        self.assertTrue(applied(pl, smartedit.SILENT))
        for c in applied(pl):
            self.assertFalse(overlaps(c, 3.0, 3.6), c)                  # the loud game event
            self.assertFalse(overlaps(c, 5.2, 6.2), c)                  # the screen event

    def test_unrecognised_voice_is_kept_when_unsure(self):
        prof = audio([('s', 1.0), ('q', 1.5), ('s', 0.8), ('q', 1.7), ('s', 1.0)])   # 2.5-3.3: no word covers it
        ws = words((0.0, 1.0, 'Раз'), (5.0, 6.0, 'два'))
        pl = smartedit.plan(hooks.Signals(segs(ws), prof), 0.0, 6.0, 'aggressive', clip_min=2)
        for c in applied(pl):
            self.assertFalse(overlaps(c, 2.6, 3.2), c)

    def test_dead_air_label_switch_off_and_overrides(self):
        ws = words((0.0, 0.5, 'Старт.'), (8.5, 9.0, 'Финиш.'))
        sig = hooks.Signals(segs(ws))
        pl = smartedit.plan(sig, 0.0, 9.5, 'dynamic', clip_min=2)
        self.assertEqual({c['reason'] for c in applied(pl)}, {'dead_air'})
        off = smartedit.plan(sig, 0.0, 9.5, 'dynamic', clip_min=2, silence=False)
        self.assertEqual(applied(off, smartedit.SILENT), [])
        self.assertEqual(off['stats']['result'], 9.5)
        tr = [{'start': 0, 'end': 3, 'text': 'я я думаю так', 'words': words((0, .3, 'я'), (.4, .7, 'я'), (.8, 1.2, 'думаю'),
                                                                            (1.3, 1.6, 'так'))}]
        rep = smartedit.plan(hooks.Signals(tr), 0.0, 4.5, 'dynamic', clip_min=1, silence=False)
        self.assertTrue(applied(rep, {'repeat'}))                      # the switch only covers silence
        many = hooks.Signals(segs(words(*[(k * 2.0, k * 2.0 + 0.6, f'слово{k}.') for k in range(10)])))
        loose = smartedit.plan(many, 0.0, 19.0, 'dynamic', clip_min=2, overrides={'max': 2.0})
        strict = smartedit.plan(many, 0.0, 19.0, 'dynamic', clip_min=2, overrides={'max': 0.3})
        self.assertEqual(applied(loose, smartedit.SILENT), [])         # 1.4 s pauses are under a 2 s threshold
        self.assertLess(loose['stats']['removed'], strict['stats']['removed'])
        self.assertEqual(smartedit.settings('dynamic', {'max': 99, 'evil': 1})['max'], smartedit.OVERRIDES['max'][1])

    def test_presets_scale(self):
        ws = words((0.0, 0.5, 'Раз.'), (1.6, 2.0, 'два.'), (4.5, 5.0, 'три.'), (9.0, 9.5, 'четыре.'))
        sig = hooks.Signals(segs(ws))
        r = [smartedit.plan(sig, 0.0, 10.0, m, clip_min=2)['stats']['result'] for m in ('natural', 'dynamic',
                                                                                         'aggressive')]
        self.assertTrue(r[0] > r[1] > r[2], r)
        self.assertTrue(all(smartedit.MODES[m]['floor'] <= 0.85 for m in smartedit.MODES))


class Zoom(SimpleTestCase):
    TR = [{'start': 0.2, 'end': 13.5, 'text': 'x', 'words': words(
        (0.2, 0.6, 'Смотри.'), (2.0, 2.4, 'Что?!'), (4.0, 4.5, 'Ничего'), (4.6, 5.0, 'себе!'), (9.0, 9.4, 'Ладно.'),
        (13.0, 13.5, 'Всё.'))}]

    def sig(self, **kw):
        ev = [{'start': 2.0, 'end': 3.0, 'kind': 'laughter', 'strength': 1.0},
              {'start': 2.8, 'end': 3.4, 'kind': 'scream', 'strength': 0.9},
              {'start': 10.5, 'end': 11.2, 'kind': 'explosion', 'strength': 0.8}]
        return hooks.Signals(self.TR, FakeAudio(15, events=ev, **kw))

    def test_peaks_levels_cooldown_rate_and_switch(self):
        pl = smartedit.plan(self.sig(), 0.0, 14.0, 'dynamic', clip_min=5, silence=False)
        zs = pl['zooms']
        self.assertTrue(zs)
        self.assertEqual(pl['stats']['zooms'], len(zs))
        for z in zs:
            self.assertTrue(1.1 <= z['level'] <= smartedit.MAX_ZOOM, z)
        outs = sorted(z['out'] for z in zs)
        self.assertTrue(all(b - a >= smartedit.MODES['dynamic']['cooldown'] - 1e-6 for a, b in zip(outs, outs[1:])))
        self.assertLessEqual(len(zs), max(1, round(14 / 10 * smartedit.MODES['dynamic']['rate'])))
        self.assertTrue(any(z['kind'] in ('laughter', 'scream') for z in zs))
        self.assertEqual(smartedit.plan(self.sig(), 0.0, 14.0, 'dynamic', clip_min=5, zoom=False)['zooms'], [])
        strong = smartedit.plan(self.sig(), 0.0, 14.0, 'aggressive', clip_min=5, overrides={'zoom_strength': 1.5})
        self.assertTrue(all(z['level'] <= smartedit.MAX_ZOOM for z in strong['zooms']))
        weak = smartedit.plan(self.sig(), 0.0, 14.0, 'dynamic', clip_min=5, overrides={'zoom_strength': 0.0})
        self.assertEqual(weak['zooms'], [])
        few = smartedit.plan(self.sig(), 0.0, 14.0, 'aggressive', clip_min=5, overrides={'zoom_rate': 0.2})
        self.assertLessEqual(len(few['zooms']), 1)

    def build(self, plan_segments, src=(1280, 720)):
        opts = exports.resolve({'preset': '720p'})
        vinfo = {'width': src[0], 'height': src[1], 'fps': 30.0, 'duration': 100.0, 'has_audio': True}
        return adapters.clip_timeline(project_id=1, clip_id=3, source_name='sources/x.mp4', fingerprint='',
                                      clip_start=0.0, clip_end=14.0, plan=plan_segments, opts=opts, vinfo=vinfo,
                                      render_info={'fps': 30}, transcript=self.TR)

    def test_timeline_punch_in_returns_and_aims_at_the_face(self):
        cam = {'x': .7, 'y': 0.0, 'w': .3, 'h': .3, 'corner': 'top_right', 'face_rel': (0.5, 0.5)}
        for seg_type, want_y in (('game_cam', lambda y: y > 0.5), ('talking', lambda y: y < 0.5),
                                 ('gameplay', lambda y: y == 0.5)):
            tl = self.build({'segments': [{'start': 0.0, 'end': 14.0, 'type': seg_type, 'face_rel': (0.5, 0.5),
                                           'path': [(0.0, 640.0, 0.0)]}], 'cam': cam})
            pl = smartedit.plan(self.sig(), 0.0, 14.0, 'dynamic', clip_min=5)
            out = smartedit.to_timeline(tl, pl, 0.0, 'dynamic')
            self.assertEqual(T.validate(out), [])
            vids = [i for i in out.tracks[0].items if i.type == 'video']
            zoomed = [v for v in vids if any(a.property == 'zoom' and max(k.value for k in a.keyframes) > 1.09
                                             for a in v.animations)]
            self.assertTrue(zoomed, seg_type)
            self.assertEqual(out.meta['edit']['zooms'], sum(1 for z in pl['zooms']))
            for v in zoomed:
                kf = next(a for a in v.animations if a.property == 'zoom').keyframes
                self.assertTrue(all(b.t > a.t for a, b in zip(kf, kf[1:])))
                self.assertLessEqual(max(k.value for k in kf), smartedit.MAX_ZOOM)
                self.assertLessEqual(kf[-1].t, v.duration + 1e-6)
                pan = {a.property: a.keyframes[0].value for a in v.animations if a.property != 'zoom'}
                self.assertTrue(want_y(0.5 + pan.get('pan_y', 0.0)), (seg_type, pan))
            # smooth return: a zoom that fits in its piece ends back at the base level
            if seg_type == 'gameplay':
                v = zoomed[0]
                kf = next(a for a in v.animations if a.property == 'zoom').keyframes
                if kf[-1].t < v.duration - 0.1:
                    self.assertEqual(kf[-1].value, kf[0].value)
                    self.assertEqual(kf[-1].easing, 'ease_in_out')


def _onsets(path, thr_db=-30.0):
    """Starts of tone bursts in a file's audio (20 ms frames)."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', '16000', '-f', 's16le', '-'],
                         capture_output=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    hop = 320
    db = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, 'same')[::hop]) + 1e-9)
    on = db > thr_db
    return [k * 0.02 for k in range(1, len(on)) if on[k] and not on[k - 1]]


@skipUnless(HAVE_FFMPEG, 'needs ffmpeg')
class EndToEnd(SimpleTestCase):
    """A real source (moving test picture + speech-like bursts, silences, one loud game event) through the real
    audio analysis, the plan, Timeline IR and the FFmpeg renderer."""
    databases = {'default'}           # the queue worker thread of earlier tests may still poll the database

    def test_render_shorter_in_sync_event_kept_zoom_visible(self):
        d = Path(tempfile.mkdtemp())
        # words (bursts) at 0.5, 1.5, 5.5, 6.5 (3 s of silence between), a loud event 9.0-9.6, last word 12.5
        bursts = [(0.5, 0.6), (1.5, 0.6), (5.5, 0.6), (6.5, 0.6), (12.5, 0.6)]
        expr = '+'.join(f'between(t,{a},{a + l})*0.3*sin(2*PI*220*t)' for a, l in bursts)
        expr += '+between(t,9.0,9.6)*0.6*(random(0)-0.5)'
        src = d / 'src.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=s=1280x720:r=30:d=14',
                        '-f', 'lavfi', '-i', f"aevalsrc=exprs='{expr}':s=48000:d=14", '-c:v', 'libx264', '-preset',
                        'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(src)], check=True)
        prof = audio_analysis.analyze_file(src, 14.0)
        names = ['Смотри!', 'сюда.', 'Что', 'случилось?!', 'Всё.']
        tr = [{'start': a, 'end': a + l, 'text': n, 'words': [{'start': a, 'end': a + l, 'text': n}]}
              for (a, l), n in zip(bursts, names)]
        sig = hooks.Signals(tr, prof)
        pl = smartedit.plan(sig, 0.0, 14.0, 'dynamic', clip_min=4)
        self.assertGreater(pl['stats']['removed'], 3.0, pl['cuts'])
        for c in applied(pl):
            self.assertFalse(overlaps(c, 9.0, 9.6), c)                  # the game event stays
            for a, l in bursts:
                self.assertFalse(overlaps(c, a, a + l), c)              # no word is cut
        opts = exports.resolve({'preset': '720p', 'quality': 'small'})
        vinfo = {'width': 1280, 'height': 720, 'fps': 30.0, 'duration': 14.0, 'has_audio': True}
        base = adapters.clip_timeline(project_id=1, clip_id=1, source_name=str(src), fingerprint='', clip_start=0.0,
                                      clip_end=14.0, plan={'segments': [{'start': 0.0, 'end': 14.0, 'type': 'gameplay'}]},
                                      opts=opts, vinfo=vinfo, render_info={'fps': 30}, transcript=tr)
        rs = renderer.RenderSettings(quality='small', loudness_measure=False, base_dir=str(d), roots=(str(d),),
                                     cache_dir=str(d / 'cache'))
        for a in base.assets:                                      # absolute source path for this test
            a.uri = f'file:{src}'
        out_tl = smartedit.to_timeline(base, pl, 0.0, 'dynamic')
        self.assertEqual(T.validate(out_tl), [])
        dest = d / 'out.mp4'
        renderer.render(out_tl, dest, rs)
        info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,duration',
                                          '-of', 'json', str(dest)], capture_output=True, text=True).stdout)
        durs = {s['codec_type']: float(s['duration']) for s in info['streams']}
        self.assertAlmostEqual(durs['video'], pl['stats']['result'], delta=0.12)
        self.assertAlmostEqual(durs['audio'], durs['video'], delta=0.1)          # picture and sound end together
        # every word starts in the output exactly where the plan maps it (sound follows the cut picture)
        got = _onsets(dest)
        for a, _ in bursts:
            want = smartedit.source_to_output(pl, a)
            self.assertTrue(any(abs(g - want) <= 0.08 for g in got), (a, want, got))
        # captions use the same clock
        cap = next(t for t in out_tl.tracks if t.type == 'caption').items[0].words
        for w, (a, _) in zip(cap, bursts):
            self.assertAlmostEqual(w.t0, smartedit.source_to_output(pl, a), delta=0.02)
        # zoom is really in the picture: the same output frame with and without the punch-in differs
        self.assertTrue(any(z['kind'] == 'loud' and abs(z['t'] - 9.0) < 0.3 for z in pl['zooms']), pl['zooms'])
        z = pl['zooms'][0]
        t = z['out'] + z['ease'] + 0.2
        flat = smartedit.to_timeline(base, dict(pl, zooms=[], switches={'zoom': False}), 0.0, 'dynamic')
        dest2 = d / 'flat.mp4'
        renderer.render(flat, dest2, rs)
        fa, fb = (subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', str(p), '-frames:v', '1',
                                  '-vf', 'scale=180:320', '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
                                 capture_output=True).stdout for p in (dest, dest2))
        diff = np.abs(np.frombuffer(fa, np.uint8).astype(int) - np.frombuffer(fb, np.uint8).astype(int)).mean()
        self.assertGreater(diff, 8.0)
