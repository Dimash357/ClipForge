"""Timeline renderer (1B): compile rules, asset safety, real ffmpeg renders, sync, scene cache, NVENC fallback,
classic-vs-timeline comparison and the experimental Stream Clipper flag."""
import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from unittest import mock, skipUnless

import numpy as np
from django.core.files import File
from django.test import SimpleTestCase, TransactionTestCase, override_settings

from studio import editor, exports, jobs, pipeline
from studio import timeline as T
from studio.timeline import adapters, examples, renderer, subtitles
from studio.timeline import compile as C
from studio.timeline import media as M
from studio.timeline import model as m
from .helpers import HAVE_FFMPEG, fake_transcript, sample_video
from .test_jobs_views import wait_for

SLOW = os.getenv('CLIPFORGE_SKIP_SLOW') == '1'
needs_ffmpeg = skipUnless(HAVE_FFMPEG and not SLOW, 'needs ffmpeg')


def probe(path):
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                        'format=duration:stream=codec_type,width,height,r_frame_rate,duration', '-of', 'json', str(path)],
                       capture_output=True, text=True, check=True)
    d = json.loads(p.stdout)
    v = next(s for s in d['streams'] if s['codec_type'] == 'video')
    a = [s for s in d['streams'] if s['codec_type'] == 'audio']
    return {'duration': float(d['format']['duration']), 'w': v['width'], 'h': v['height'], 'fps': v['r_frame_rate'],
            'audio': bool(a), 'audio_duration': float(a[0]['duration']) if a else None}


def ff(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args], check=True, timeout=300)


def py_expr(expr, **var):
    """Evaluate the ffmpeg expressions anim_expr() writes (if/lt/pow) in Python."""
    e = expr.replace('if(', 'IF(').replace('lt(', 'LT(')
    return eval(e, {'IF': lambda c, a, b: a if c else b, 'LT': lambda a, b: a < b, 'pow': pow}, var)


def tl_dict(kind='cartoon'):
    return T.to_dict(getattr(examples, kind)())


# =====================================================================================================
class Expressions(SimpleTestCase):
    def test_numbers_never_use_exponents(self):
        self.assertEqual([C.n(v) for v in (0.0, -0.0, 1e-7, 2.5, 1080, 1 / 3)], ['0', '0', '0', '2.5', '1080', '0.333333'])

    def test_keyframe_curves(self):
        ks = [m.Keyframe(t=0.0, value=1.0), m.Keyframe(t=2.0, value=2.0),
              m.Keyframe(t=4.0, value=0.0, easing='ease_in'), m.Keyframe(t=5.0, value=9.0, easing='hold')]
        e = C.anim_expr(ks, 't')
        for t, want in [(-1, 1.0), (0, 1.0), (1, 1.5), (2, 2.0), (3, 2.0 - 2.0 * 0.25), (4.5, 0.0), (5, 9.0), (9, 9.0)]:
            self.assertAlmostEqual(py_expr(e, t=t), want, places=6, msg=t)
        self.assertEqual(C.anim_expr([], 't', 1.0), '1')
        out = [m.Keyframe(t=0, value=0), m.Keyframe(t=1, value=1, easing='ease_out')]
        self.assertAlmostEqual(py_expr(C.anim_expr(out, 't'), t=0.5), 0.75)
        io = [m.Keyframe(t=0, value=0), m.Keyframe(t=1, value=1, easing='ease_in_out')]
        self.assertAlmostEqual(py_expr(C.anim_expr(io, 't'), t=0.25), 0.125)


class Support(SimpleTestCase):
    def issues(self, d):
        return C.check_supported(T.from_dict(d))

    def test_examples_are_supported(self):
        for k in ('cartoon', 'faceless'):
            self.assertEqual(self.issues(tl_dict(k)), [])

    def test_unsupported_features_are_named_explicitly(self):
        cases = []
        d = tl_dict()
        d['tracks'][0]['items'][2]['kind'] = 'wipe'
        cases.append((d, 'tracks[0].items[2].kind', 'wipe'))
        d = tl_dict()
        d['tracks'][0]['items'][0]['animations'].append({'property': 'rotation', 'keyframes': [{'t': 0, 'value': 5}]})
        cases.append((d, 'animations[1].property', 'rotation'))
        d = tl_dict()
        d['tracks'][0]['items'][0]['animations'][0]['keyframes'][1]['value'] = 0.5
        cases.append((d, 'tracks[0].items[0].animations[0]', 'zoom below'))
        d = tl_dict('faceless')
        d['tracks'][1]['items'][0]['animations'].append({'property': 'x', 'keyframes': [
            {'t': 0, 'value': .1}, {'t': .5, 'value': .5}, {'t': 1, 'value': .9}]})
        cases.append((d, 'tracks[1].items[0]', 'at most 2'))
        d = tl_dict('faceless')
        d['tracks'][1]['items'][0]['animations'][0]['keyframes'][1]['easing'] = 'ease_in_out'
        cases.append((d, 'tracks[1].items[0].animations[0]', 'easing'))
        d = tl_dict('faceless')
        d['tracks'][0]['items'][1]['layout']['params']['wobble'] = 1
        cases.append((d, 'layout.params', 'wobble'))
        d = tl_dict('faceless')
        d['styles'][0]['font'] = 'Evil,Font'
        cases.append((d, 'styles[0].font', 'commas'))
        for d, path, msg in cases:
            iss = self.issues(d)
            self.assertTrue(any(path in i['path'] and msg in i['message'] for i in iss), (path, iss))

    def test_render_refuses_before_running_ffmpeg(self):
        d = tl_dict()
        d['tracks'][0]['items'][2]['kind'] = 'whoosh'
        with mock.patch.object(renderer, 'run_ffmpeg') as run, self.assertRaises(C.UnsupportedFeature) as cm:
            renderer.render(T.from_dict(d), '/nonexistent/out.mp4')
        run.assert_not_called()
        self.assertIn('whoosh', str(cm.exception))
        bad = tl_dict()
        bad['tracks'][0]['items'][0]['asset'] = 'as_000000000000'          # invalid IR: validation first
        with self.assertRaises(T.TimelineError):
            renderer.render(T.from_dict(bad), '/nonexistent/out.mp4')


class AssetSafety(SimpleTestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.media = self.root / 'media'
        self.base = self.root / 'proj'
        (self.media / 'sources').mkdir(parents=True)
        (self.base / 'gen').mkdir(parents=True)
        (self.media / 'sources' / 'a.mp4').write_bytes(b'x')
        (self.base / 'gen' / 'img.png').write_bytes(b'x')
        (self.root / 'secret.txt').write_bytes(b'x')
        ov = override_settings(MEDIA_ROOT=self.media)
        ov.enable()
        self.addCleanup(ov.disable)
        self.r = M.Resolver(self.base)

    def test_resolves_inside_roots(self):
        self.assertEqual(self.r.resolve('media:sources/a.mp4'), (self.media / 'sources' / 'a.mp4').resolve())
        self.assertEqual(self.r.resolve('gen:img.png'), (self.base / 'gen' / 'img.png').resolve())
        self.assertEqual(self.r.resolve('file:gen/img.png'), (self.base / 'gen' / 'img.png').resolve())
        self.assertEqual(self.r.resolve('file:' + str(self.media / 'sources' / 'a.mp4')),
                         (self.media / 'sources' / 'a.mp4').resolve())
        self.assertTrue(M.ffmpeg_input(Path('/x/-y.mp4')).startswith('file:'))

    def test_rejects_escapes_and_unknown(self):
        bad = ['media:../secret.txt', 'gen:../../secret.txt', 'file:' + str(self.root / 'secret.txt'),
               'http://evil/x.mp4', 'concat:a|b', 'media:/etc/passwd', 'media:sources/missing.mp4', 'media:',
               'nocolon', 'media:a\x00b']
        for uri in bad:
            with self.assertRaises(M.AssetError, msg=uri):
                self.r.resolve(uri)

    @skipUnless(hasattr(os, 'symlink') and os.name != 'nt', 'symlinks')
    def test_symlink_out_of_root_is_rejected(self):
        os.symlink(self.root / 'secret.txt', self.media / 'sources' / 'link.mp4')
        with self.assertRaises(M.AssetError):
            self.r.resolve('media:sources/link.mp4')


class Subtitles(SimpleTestCase):
    def test_timeline_times_and_text_animation(self):
        tl = examples.faceless()
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, td, True)
        rep = subtitles.write_ass(tl, td / 'a.ass', [])
        text = (td / 'a.ass').read_text(encoding='utf-8')
        self.assertEqual(rep['events'], 3)
        self.assertIn('Dialogue: 0,0:00:06.20,', text)                                # word 'океан' at 6.2 s
        self.assertIn(r'\fscx80\fscy80\t(0,200,0.5,\fscx100\fscy100)', text)           # hook pop, ease_out
        self.assertIn('ТЫ ЭТОГО НЕ ЗНАЛ', text)
        self.assertRegex(text, r'Космос \{\\1c&H[0-9A-F]+\}пахнет')                    # emphasis colour

    def test_no_text_no_pass(self):
        d = tl_dict('faceless')
        d['tracks'] = [t for t in d['tracks'] if t['type'] in ('video', 'audio')]
        self.assertIsNone(subtitles.write_ass(T.from_dict(d), Path(tempfile.mkdtemp()) / 'a.ass', []))


# =====================================================================================================
@needs_ffmpeg
class ExampleRenders(SimpleTestCase):
    """Real renders of the reference cartoon and faceless timelines with local placeholder assets."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.base = Path(tempfile.mkdtemp(prefix='cf_tlr_'))
        cls.settings = renderer.RenderSettings(base_dir=cls.base, cache_dir=cls.base / 'cache', encoder='cpu',
                                               quality='small')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base, True)
        super().tearDownClass()

    def render(self, tl, name, **kw):
        examples.placeholder_assets(tl, self.base)
        out = self.base / f'{name}.mp4'
        return out, renderer.render(tl, out, self.settings, **kw)

    def test_cartoon_and_faceless(self):
        for kind, dur in (('cartoon', 8.0), ('faceless', 12.0)):
            out, rep = self.render(getattr(examples, kind)(), kind)
            info = probe(out)
            self.assertEqual((info['w'], info['h'], info['fps']), (1080, 1920, '30/1'), kind)
            self.assertAlmostEqual(info['duration'], dur, delta=0.05)
            self.assertTrue(info['audio'])
            self.assertAlmostEqual(info['audio_duration'], dur, delta=0.05)
            self.assertEqual(rep['encoder'], 'libx264')
            self.assertEqual(len(rep['scenes']), 2)
            self.assertGreater(rep['captions']['events'], 0)

    def test_scene_cache_and_single_scene_rerender(self):
        tl = examples.cartoon('cache')
        s1, s2 = tl.scenes[0].id, tl.scenes[1].id
        _, r1 = self.render(tl, 'c1')
        self.assertEqual(r1['rendered_scenes'], 2)
        _, r2 = self.render(tl, 'c2')
        self.assertEqual(r2['rendered_scenes'], 0)                         # everything from the cache
        tl.tracks[0].items[1].fit = 'contain'                               # edit scene 2's picture only
        _, r3 = self.render(tl, 'c3')
        self.assertEqual({s['id']: s['cached'] for s in r3['scenes']}, {s1: True, s2: False})
        _, r4 = self.render(tl, 'c4', rerender={s1})                        # forced re-render of one scene
        self.assertEqual({s['id']: s['cached'] for s in r4['scenes']}, {s1: False, s2: True})
        img = M.Resolver(self.base).resolve(tl.asset(tl.tracks[0].items[0].asset).uri)
        st = img.stat()
        os.utime(img, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # the source file changed on disk
        _, r5 = self.render(tl, 'c5')
        self.assertEqual({s['id']: s['cached'] for s in r5['scenes']}, {s1: False, s2: True})
        self.assertTrue(renderer.render_scene(tl, s2, self.settings).is_file())
        with self.assertRaises(T.TimelineError):
            self.render(tl, 'c6', rerender={'sc_000000000000'})

    def test_missing_asset_is_reported_with_its_path(self):
        tl = examples.faceless('missing')
        tl.assets[0].uri = 'file:stock/nope.mp4'
        with self.assertRaises(M.AssetError) as cm:
            renderer.render(tl, self.base / 'x.mp4', self.settings)
        self.assertTrue(any('assets[0].uri' in i['path'] for i in cm.exception.issues))

    def test_reading_past_the_end_of_a_file_is_refused(self):
        tl = examples.faceless('short')
        examples.placeholder_assets(tl, self.base)
        tl.assets[1].duration = None                                        # the IR does not know; the file does
        tl.tracks[0].items[1].src_in = 7.0
        with self.assertRaises(M.AssetError) as cm:
            renderer.render(tl, self.base / 'x.mp4', self.settings)
        self.assertIn('long', str(cm.exception))

    def test_nvenc_failure_falls_back_to_cpu(self):
        tl = examples.faceless('nvenc')
        examples.placeholder_assets(tl, self.base)
        st = copy.copy(self.settings)
        st.encoder = 'auto'
        real = renderer.run_ffmpeg

        def flaky(cmd, **kw):
            if 'h264_nvenc' in cmd:
                raise RuntimeError('No NVENC capable devices found')
            return real(cmd, **kw)
        with mock.patch.object(renderer.diagnostics, 'choose_encoder', return_value=('h264_nvenc', 'NVIDIA NVENC')), \
                mock.patch.object(renderer, 'run_ffmpeg', side_effect=flaky):
            rep = renderer.render(tl, self.base / 'nv.mp4', st)
        self.assertTrue(rep['encoder_fallback'])
        self.assertEqual(rep['encoder'], 'libx264')
        self.assertTrue(probe(self.base / 'nv.mp4')['audio'])


# =====================================================================================================
@needs_ffmpeg
class Sync(SimpleTestCase):
    """A flash + a beep at the same source moment must land on the same timeline moment (+ the caption word)."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix='cf_sync_'))
        self.addCleanup(shutil.rmtree, self.base, True)
        (self.base / 'gen').mkdir()
        ff('-f', 'lavfi', '-i', 'color=c=black:s=320x240:r=30:d=4', '-f', 'lavfi',
           '-i', 'sine=frequency=1000:sample_rate=48000:duration=4',
           '-vf', "drawbox=c=white:t=fill:enable='between(t,2,2.1)'",
           '-af', "volume=0:enable='lt(t,2)+gte(t,2.1)'", '-c:v', 'libx264', '-preset', 'ultrafast',
           '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(self.base / 'gen' / 'flash.mp4'))

    def timeline(self, W, H, speed=1.0):
        a = m.Asset(id='as_flash01', type='video', uri='gen:flash.mp4', duration=4.0, width=320, height=240,
                    has_audio=True)
        st = m.Style(id='st_cap001', max_words=1, size=60)
        return m.Timeline(id='tl_sync01', duration=5.0, canvas=m.Canvas(width=W, height=H, fps=25.0),
                          assets=[a], styles=[st], tracks=[
            m.Track(id='tr_video01', type='video', items=[
                m.VideoItem(id='it_video01', start=1.0, duration=3.0, asset=a.id, src_in=0.5, volume=0.0,
                            layout=m.Layout(type='fit', params={'bg': 'solid'}))]),
            m.Track(id='tr_audio01', type='audio', items=[
                m.AudioItem(id='it_audio01', start=1.0, duration=3.0, asset=a.id, src_in=0.5)]),
            m.Track(id='tr_capt01', type='caption', items=[
                m.CaptionItem(id='it_capt01', start=0.0, duration=5.0, style=st.id,
                              words=[m.Word(t0=2.5, t1=2.9, text='БАХ')])])])

    def onsets(self, out, fps):
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(out), '-vf', 'crop=iw/2:ih/8:iw/4:ih*7/16,scale=16:16,format=gray',
                              '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, 256).mean(axis=1)
        flash = int(np.argmax(frames > 128)) / fps
        pcm = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(out), '-ac', '1', '-ar', '48000', '-f', 's16le', '-'],
                             capture_output=True, check=True).stdout
        a = np.abs(np.frombuffer(pcm, np.int16).astype(np.float32))
        beep = int(np.argmax(a > 0.2 * a.max())) / 48000
        return flash, beep

    def test_picture_sound_and_captions_share_one_clock(self):
        for W, H in ((640, 360), (540, 960)):                              # landscape + vertical canvases
            out = self.base / f's{W}.mp4'
            tl = self.timeline(W, H)
            rep = renderer.render(tl, out, renderer.RenderSettings(base_dir=self.base, cache_dir=self.base / 'c',
                                                                   encoder='cpu'))
            info = probe(out)
            self.assertEqual((info['w'], info['h']), (W, H))
            self.assertAlmostEqual(info['duration'], 5.0, delta=0.05)
            flash, beep = self.onsets(out, 25.0)
            self.assertAlmostEqual(flash, 2.5, delta=0.041, msg=f'{W}x{H} flash')     # one frame at 25 fps
            self.assertAlmostEqual(beep, 2.5, delta=0.03, msg=f'{W}x{H} beep')
            self.assertEqual(rep['captions']['events'], 1)

    def test_mixed_and_ducked_sources_stay_in_place(self):
        tl = self.timeline(540, 960)
        aid = tl.assets[0].id
        tl.tracks[1].items.append(m.AudioItem(id='it_audio02', start=2.0, duration=3.0, asset=aid, src_in=0.5,
                                              role='voice'))                        # second beep at 3.5 s
        tl.tracks.append(m.Track(id='tr_music01', type='audio', items=[            # ducked bed: silent but mixed
            m.AudioItem(id='it_music01', start=0.0, duration=4.0, asset=aid, src_in=0.0, role='music', duck=True,
                        volume=0.5, fade_in=0.2, animations=[m.Animation(property='volume', keyframes=[
                            m.Keyframe(t=0, value=1.0), m.Keyframe(t=4, value=0.5)])])]))
        out = self.base / 'mix.mp4'
        renderer.render(tl, out, renderer.RenderSettings(base_dir=self.base, cache_dir=self.base / 'c', encoder='cpu'))
        pcm = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(out), '-ac', '1', '-ar', '48000', '-f', 's16le', '-'],
                             capture_output=True, check=True).stdout
        a = np.abs(np.frombuffer(pcm, np.int16).astype(np.float32))
        self.assertAlmostEqual(len(a) / 48000, 5.0, delta=0.03)
        win = lambda t0, t1: a[int(t0 * 48000):int(t1 * 48000)].max()        # noqa: E731
        loud = 0.2 * a.max()
        # music beep (src 2.0 -> 2.0), first voice (2.5), second voice (3.5); quiet in between
        for t in (2.0, 2.5, 3.5):
            self.assertGreater(win(t + 0.02, t + 0.08), loud, t)
        for t0, t1 in ((0.3, 1.95), (2.15, 2.45), (2.65, 3.45), (3.65, 5.0)):
            self.assertLess(win(t0, t1), loud, (t0, t1))


# =====================================================================================================
def _plan():
    cam = {'x': .7, 'y': 0.0, 'w': .3, 'h': .35, 'corner': 'top_right', 'confidence': .9, 'face_rel': (0.5, 0.45),
           'face_h': 0.5, 'eye_rel': 0.4, 'source': 'detected'}
    return {'segments': [{'start': 0.0, 'end': 3.0, 'type': 'game_cam', 'face_rel': (0.5, 0.45), 'confidence': .9},
                         {'start': 3.0, 'end': 6.0, 'type': 'action', 'path': [(0.0, 100.0, 0.0), (3.0, 400.0, 0.0)]},
                         {'start': 6.0, 'end': 8.0, 'type': 'gameplay'}],
            'cam': cam, 'mode': 'director', 'notes': []}


def ssim(a, b):
    p = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(a), '-i', str(b), '-lavfi', '[0:v][1:v]ssim', '-f', 'null',
                        '-'], capture_output=True, text=True)
    return float(re.search(r'All:([0-9.]+)', p.stderr).group(1))


@needs_ffmpeg
class ClassicVsTimeline(SimpleTestCase):
    """The same Stream Clipper clip through editor.render and through Timeline IR -> timeline renderer."""

    def setUp(self):
        self.media = Path(tempfile.mkdtemp(prefix='cf_cmp_'))
        self.addCleanup(shutil.rmtree, self.media, True)
        ov = override_settings(MEDIA_ROOT=self.media)
        ov.enable()
        self.addCleanup(ov.disable)
        (self.media / 'sources').mkdir()
        self.src = self.media / 'sources' / 'mix.mp4'
        shutil.copy(sample_video(), self.src)
        self.vinfo = editor.probe_video(self.src)

    def both(self, **o):
        opts = exports.resolve(dict({'preset': '720p', 'quality': 'high', 'transition': 'fade', 'encoder': 'cpu'}, **o))
        clip = {'start': 4.0, 'end': 12.0}
        tr = fake_transcript()
        old_dir = self.media / 'old'
        old_dir.mkdir(exist_ok=True)
        info = editor.render(self.src, clip, tr, old_dir / 'out.mp4', old_dir, opts, _plan(), self.vinfo, 'director')
        tl = adapters.clip_timeline(project_id=1, clip_id=1, source_name='sources/mix.mp4', fingerprint='',
                                    clip_start=4.0, clip_end=12.0, plan=_plan(), opts=opts, vinfo=self.vinfo,
                                    render_info=info, transcript=tr)
        new = self.media / 'new.mp4'
        work = self.media / 'work'
        rep = renderer.render(tl, new, renderer.RenderSettings(encoder='cpu', quality='high', workdir=work))
        return old_dir, new, work, info, rep

    def test_same_picture_sound_and_length(self):
        old_dir, new, _, info, rep = self.both(captions=False)
        a, b = probe(old_dir / 'out.mp4'), probe(new)
        self.assertEqual((a['w'], a['h']), (b['w'], b['h']))
        self.assertAlmostEqual(a['duration'], b['duration'], delta=0.06)
        self.assertTrue(a['audio'] and b['audio'])
        self.assertAlmostEqual(a['audio_duration'], b['audio_duration'], delta=0.06)
        self.assertEqual(rep['audio']['loudness'][0]['mode'], info['loudness']['mode'])
        score = ssim(old_dir / 'out.mp4', new)
        self.assertGreater(score, 0.95, score)

    def test_same_captions(self):
        old_dir, new, work, info, rep = self.both(subtitle_style='stream', caption_censor=True)
        dl = lambda p: [ln.split(',', 4)[1:3] + [ln.split(',', 9)[-1]] for ln in       # noqa: E731
                        p.read_text(encoding='utf-8').splitlines() if ln.startswith('Dialogue:')]
        old, new_ = dl(old_dir / 'captions.ass'), dl(work / 'timeline.ass')
        self.assertGreater(len(old), 5)
        self.assertEqual(old, new_)                                     # same times, same text and tags
        self.assertGreater(ssim(old_dir / 'out.mp4', new), 0.93)


# =====================================================================================================
@needs_ffmpeg
class StreamClipperFlag(TransactionTestCase):
    def setUp(self):
        jobs._active.clear()
        jobs._cancel.clear()
        self.media = tempfile.mkdtemp()
        self.ov = override_settings(MEDIA_ROOT=self.media)
        self.ov.enable()
        self.addCleanup(self.ov.disable)
        self.addCleanup(shutil.rmtree, self.media, True)

    def run_full(self):
        p = pipeline.Project(title='t', clip_count=2, clip_min=5, clip_max=9, edit_mode='director',
                             options=exports.resolve({'preset': '720p', 'quality': 'small', 'transition': 'fade'}))
        with open(sample_video(), 'rb') as f:
            p.source.save('mix.mp4', File(f), save=True)
        with mock.patch.object(pipeline, 'transcribe', return_value=fake_transcript()):
            self.assertTrue(jobs.enqueue('full', p.pk))
            self.assertTrue(wait_for(lambda: not jobs.is_active(p.pk), timeout=300))
        p.refresh_from_db()
        self.assertEqual(p.status, 'done', p.error)
        return p

    def test_flag_off_keeps_the_classic_renderer(self):
        p = self.run_full()
        for c in p.clips.all():
            self.assertNotIn('renderer', c.render_info)
            self.assertIsNotNone(c.render_info['timeline'])

    @override_settings(TIMELINE_RENDERER=True)
    def test_flag_on_renders_through_the_timeline(self):
        p = self.run_full()
        for c in p.clips.all():
            info = c.render_info
            self.assertEqual((c.status, info['renderer']), ('ready', 'timeline'))
            tl = T.from_dict(info['timeline'])
            self.assertEqual(T.validate(tl), [])
            got = probe(c.video.path)
            self.assertAlmostEqual(got['duration'], c.end - c.start, delta=0.1)
            self.assertEqual((got['w'], got['h']), (720, 1280))
            self.assertTrue(got['audio'])

    @override_settings(TIMELINE_RENDERER=True)
    def test_flag_on_falls_back_to_classic_on_errors(self):
        with mock.patch.object(renderer, 'render', side_effect=C.UnsupportedFeature('nope')):
            p = self.run_full()
        for c in p.clips.all():
            self.assertEqual(c.status, 'ready')
            self.assertNotIn('renderer', c.render_info)
