"""Long transcriptions never look frozen: phase + elapsed time until the first segment; quiet console."""
import logging
import time
from unittest import mock

from django.test import SimpleTestCase

from studio import pipeline
from clipforge.logfilters import QuietPolling


class Seg:
    def __init__(self, a, b, text):
        self.start, self.end, self.text, self.words = a, b, text, []


class Rep:
    def __init__(self):
        self.notes, self.fracs = [], []

    def note(self, m):
        self.notes.append(m)

    def frac(self, v, m=None):
        self.fracs.append((v, m))

    def check(self):
        pass


class Heartbeat(SimpleTestCase):
    def setUp(self):
        pipeline._model_cache.clear()

    def _run(self, cached, first_delay=0.35):
        class Model:
            def transcribe(self, *a, **k):
                def gen():
                    time.sleep(first_delay)          # VAD over a long file / slow first batch
                    yield Seg(0, 1, 'раз')
                return gen(), mock.Mock(language='ru')
        rep = Rep()
        with mock.patch('studio.diagnostics.whisper_settings', return_value=('cpu', 'int8', 'n')), \
                mock.patch.object(pipeline, '_get_model', return_value=Model()), \
                mock.patch.object(pipeline, '_whisper_cached', return_value=cached), \
                mock.patch.object(pipeline._Heartbeat, '__init__', lambda s, r, every=0.1: _init(s, r, 0.1)):
            out = pipeline.transcribe('a.wav', 'auto', rep=rep, duration=100)
        return out, rep

    def test_first_run_says_the_model_is_downloading(self):
        out, rep = self._run(cached=False)
        self.assertEqual(out[0]['text'], 'раз')
        self.assertTrue(any('Скачиваем модель Whisper' in n for n in rep.notes), rep.notes)

    def test_elapsed_time_is_shown_until_the_first_segment(self):
        _, rep = self._run(cached=True)
        ticks = [n for n in rep.notes if n.startswith('Распознаём речь (CPU)')]
        self.assertGreaterEqual(len(ticks), 2, rep.notes)
        self.assertIn('прошло 0:0', ticks[-1])
        self.assertTrue(rep.fracs and 'Распознаём речь: 1%' in rep.fracs[-1][1])
        self.assertIn('seconds', pipeline.transcribe.last_info)

    def test_cache_check_never_raises(self):
        with mock.patch('huggingface_hub.try_to_load_from_cache', side_effect=RuntimeError('x')):
            self.assertTrue(pipeline._whisper_cached('small'))


_real_init = pipeline._Heartbeat.__init__


def _init(self, rep, every):
    _real_init(self, rep, every)


class ConsoleFilter(SimpleTestCase):
    def _rec(self, msg):
        return logging.LogRecord('django.server', logging.INFO, __file__, 1, msg, None, None)

    def test_status_polls_hidden_errors_kept(self):
        f = QuietPolling()
        self.assertFalse(f.filter(self._rec('"GET /api/projects/1/status/ HTTP/1.1" 200 1803')))
        self.assertTrue(f.filter(self._rec('"GET /api/projects/1/status/ HTTP/1.1" 500 120')))
        self.assertTrue(f.filter(self._rec('"POST /api/projects/1/render/ HTTP/1.1" 200 11')))
        self.assertTrue(f.filter(self._rec('"GET /projects/1/ HTTP/1.1" 200 5000')))


class GpuLibraries(SimpleTestCase):
    """Missing CUDA 12 DLLs: decided once, CPU straight away, no minutes lost on doomed GPU attempts."""

    def setUp(self):
        from studio import diagnostics
        diagnostics._whisper_gpu_failed = False
        self.addCleanup(setattr, diagnostics, '_whisper_gpu_failed', False)
        pipeline._model_cache.clear()

    def test_probe_with_missing_dlls_picks_cpu(self):
        from studio import diagnostics
        with mock.patch.object(diagnostics, 'cuda_probe', return_value={'count': 1, 'types': ['float16'],
                                                                         'missing': ['cublas64_12.dll']}), \
                mock.patch.dict('os.environ', {'WHISPER_DEVICE': 'auto'}):
            dev, comp, note = diagnostics.whisper_settings()
        self.assertEqual(dev, 'cpu')
        self.assertIn('install_gpu.bat', note)

    def test_library_error_in_batched_mode_goes_straight_to_cpu(self):
        calls = []

        class Model:
            def __init__(self, dev):
                self.dev = dev

            def transcribe(self, *a, **k):
                calls.append(('seq', self.dev))
                return iter([Seg(0, 1, 'раз')]), mock.Mock(language='ru')

        class Batched:
            def __init__(self, model):
                self.dev = model.dev

            def transcribe(self, *a, **k):
                calls.append(('batched', self.dev))
                raise RuntimeError('Library cublas64_12.dll is not found or cannot be loaded')
        import faster_whisper
        from studio import diagnostics
        with mock.patch.object(diagnostics, 'whisper_settings', side_effect=lambda: ('cpu', 'int8', 'x')
                               if diagnostics._whisper_gpu_failed else ('cuda', 'float16', 'x')), \
                mock.patch.object(pipeline, '_get_model', side_effect=lambda n, d, c: Model(d)), \
                mock.patch.object(pipeline, '_whisper_cached', return_value=True), \
                mock.patch.object(faster_whisper, 'BatchedInferencePipeline', Batched):
            out = pipeline.transcribe('a.wav', 'auto')
        self.assertEqual(calls, [('batched', 'cuda'), ('seq', 'cpu')], 'no sequential retry on the broken GPU')
        self.assertEqual(out[0]['text'], 'раз')
        self.assertTrue(diagnostics._whisper_gpu_failed)

    def test_dll_folders_found_in_site_packages(self):
        import tempfile
        from pathlib import Path
        from studio import gpu_libs
        root = Path(tempfile.mkdtemp())
        for pkg in ('cublas', 'cudnn'):
            (root / 'nvidia' / pkg / 'bin').mkdir(parents=True)
            (root / 'nvidia' / pkg / 'bin' / f'{pkg}64.dll').write_bytes(b'x')
        (root / 'nvidia' / 'empty' / 'bin').mkdir(parents=True)
        with mock.patch('sysconfig.get_paths', return_value={'purelib': str(root), 'platlib': str(root)}), \
                mock.patch('site.getsitepackages', return_value=[]):
            dirs = gpu_libs.nvidia_dirs()
        self.assertEqual(sorted(d.parent.name for d in dirs), ['cublas', 'cudnn'])

    def test_non_windows_is_a_no_op(self):
        from studio import gpu_libs
        with mock.patch.object(gpu_libs.os, 'name', 'posix'):
            self.assertEqual(gpu_libs.missing_dlls(), [])
