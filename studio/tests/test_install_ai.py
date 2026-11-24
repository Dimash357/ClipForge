"""tools/install_ai.py: resumable downloads without duplicates, file checks, no download without an answer,
configuration of ClipForge and the provider check (against the local ComfyUI / TTS stand-in)."""
import importlib.util
import json
import shutil
import struct
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase, override_settings

from studio import genmedia
from .test_genmedia import FakeServer

spec = importlib.util.spec_from_file_location('install_ai', Path(__file__).resolve().parents[2] / 'tools' / 'install_ai.py')
IA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IA)


def safetensors(n=300_000):
    head = json.dumps({'w': {'dtype': 'F16', 'shape': [n // 2], 'data_offsets': [0, n]}}).encode()
    return struct.pack('<Q', len(head)) + head + b'\0' * n


class Files:
    """HTTP server with Range support; `cut` = bytes sent before the first connection drops."""
    def __init__(self, files):
        self.files, self.gets, self.cut = files, [], None
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _body(self):
                data = srv.files.get(self.path.lstrip('/'))
                if data is None:
                    self.send_response(404)
                    self.end_headers()
                    return None, 0
                start = int((self.headers.get('Range') or 'bytes=0-')[6:].split('-')[0])
                self.send_response(206 if start else 200)
                self.send_header('Content-Length', str(len(data) - start))
                self.end_headers()
                return data, start

            def do_HEAD(self):
                self._body()

            def do_GET(self):
                srv.gets.append((self.path, self.headers.get('Range')))
                data, start = self._body()
                if data is None:
                    return
                if srv.cut and not start:
                    self.wfile.write(data[:srv.cut])
                    srv.cut = None
                    self.connection.close()
                    return
                self.wfile.write(data[start:])

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


class Downloads(SimpleTestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.st = safetensors()
        self.srv = Files({'m.safetensors': self.st, 'bad.safetensors': b'garbage' * 1000})
        self.addCleanup(self.srv.httpd.shutdown)
        p = mock.patch.object(IA.time, 'sleep')
        p.start()
        self.addCleanup(p.stop)

    def test_resume_skip_and_integrity(self):
        dest = self.dir / 'm.safetensors'
        self.srv.cut = 100_000                              # first connection drops -> resumes with Range
        self.assertTrue(IA.download(f'{self.srv.url}/m.safetensors', dest, 'm'))
        self.assertEqual(dest.read_bytes(), self.st)
        self.assertTrue(any(r and r.startswith('bytes=') for _, r in self.srv.gets))
        n = len(self.srv.gets)
        self.assertTrue(IA.download(f'{self.srv.url}/m.safetensors', dest, 'm'))      # second run: no download
        self.assertEqual(len(self.srv.gets), n)
        self.assertFalse(IA.download(f'{self.srv.url}/bad.safetensors', self.dir / 'bad.safetensors', 'bad'))
        self.assertFalse((self.dir / 'bad.safetensors').exists())          # a broken file is not accepted
        self.assertFalse(IA.download(f'{self.srv.url}/missing.safetensors', self.dir / 'x.safetensors', 'x'))
        other = self.dir / 'other.safetensors'                              # a different existing file is kept
        other.write_bytes(b'mine')
        self.assertFalse(IA.download(f'{self.srv.url}/m.safetensors', other, 'o'))
        self.assertEqual(other.read_bytes(), b'mine')

    def test_space_and_questions(self):
        with mock.patch.object(IA, 'free_gb', return_value=3.0):
            self.assertFalse(IA.need_space(self.dir, 5, 'big'))
            self.assertFalse(IA.download(f'{self.srv.url}/m.safetensors', self.dir / 'y.safetensors', 'y', 5) and False)
        with mock.patch.object(IA.sys, 'stdin', None):      # no console: nothing is downloaded or installed
            self.assertFalse(IA.ask('download 7 GB?', True))
            self.assertEqual(IA.choose('which?', ['a', 'b'], default=1), [])
        with mock.patch.object(IA, 'YES', True):
            self.assertTrue(IA.ask('download?'))
            self.assertEqual(IA.choose('which?', ['a', 'b'], default=2), [1])


class ConfigureAndCheck(SimpleTestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.srv = FakeServer()
        self.addCleanup(self.srv.close)
        ov = override_settings(CLIPFORGE_SETTINGS_FILE=self.dir / 's.json')
        ov.enable()
        self.addCleanup(ov.disable)
        for k, v in {'ENV_BAT': self.dir / 'ai_env.bat', 'PIPER_DIR': self.dir / 'piper', 'STATE': self.dir / 'st.json',
                     'AI': self.dir}.items():
            p = mock.patch.object(IA, k, v)
            p.start()
            self.addCleanup(p.stop)

    def test_configure_then_check(self):
        st = {'ollama_model': 'qwen2.5:7b', 'checkpoint': 'juggernautXL.safetensors', 'image_model': 'sdxl',
              'comfy_dir': str(self.dir / 'Comfy')}
        IA.configure(st)
        env = (self.dir / 'ai_env.bat').read_text()
        self.assertIn('set "USE_OLLAMA=1"', env)
        self.assertIn('set "OLLAMA_MODEL=qwen2.5:7b"', env)
        cfg = genmedia.config()
        self.assertEqual((cfg['checkpoint'], cfg['image_model']), ('juggernautXL.safetensors', 'sdxl'))
        IA.save_state(st)
        self.assertEqual(IA.load_state(), st)
        genmedia.save_config({'comfy_url': self.srv.url, 'tts_provider': 'openai_tts', 'tts_url': self.srv.url + '/v1',
                              'tts_voices': 'alloy'})
        with mock.patch.object(IA, 'ollama_up', return_value=['qwen2.5:7b']):
            self.assertEqual(IA.check(st), 0)
        with mock.patch.object(IA, 'ollama_up', return_value=['llama3:8b']):
            self.assertEqual(IA.check(st), 1)                  # configured model missing -> reported
        genmedia.save_config({'comfy_url': 'http://127.0.0.1:9'})
        with mock.patch.object(IA, 'ollama_up', return_value=None):
            self.assertEqual(IA.check(st), 1)                  # ComfyUI installed but not running -> reported
