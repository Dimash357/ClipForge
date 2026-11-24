"""ClipForge AI providers: install, configure and check Ollama (LLM), ComfyUI (images) and Piper (voice).

Run through install_ai.bat (uses the project's env). Commands:
    menu (default)   step-by-step install with a question before every download / install
    check            check every provider (nothing is installed or downloaded)
    start-services   start Ollama / ComfyUI in the background if installed and not running (used by start.bat)
Options: --yes (answer yes to every question), --comfy-dir PATH (use an existing ComfyUI portable folder).

Safe to run again: finished downloads are recognised by size (and file header), interrupted ones resume from the
.part file, existing installs and models are never deleted or overwritten. Everything goes into the project's
ai\\ and models\\ folders or the ComfyUI folder; the system Python is never touched.
"""
import json
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AI = ROOT / 'ai'
DL = AI / 'downloads'
STATE = AI / 'install_ai.json'
ENV_BAT = ROOT / 'ai_env.bat'
PIPER_DIR = ROOT / 'models' / 'piper'
GB = 1024 ** 3
YES = '--yes' in sys.argv

HF = 'https://huggingface.co'
COMFY_URL = 'https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z'
IPA_NODES_URL = 'https://codeload.github.com/cubiq/ComfyUI_IPAdapter_plus/zip/refs/heads/main'
OLLAMA_SETUP = 'https://ollama.com/download/OllamaSetup.exe'
LLMS = [('qwen2.5:7b', 4.7, 'лучше пишет по-русски; рекомендуется (8 ГБ VRAM — по очереди с ComfyUI)'),
        ('qwen2.5:3b', 1.9, 'легче и быстрее, проще тексты')]
CHECKPOINTS = [   # key, name, file, url, GB, model type
    ('juggernaut', 'Juggernaut XL v9 (SDXL, фотореализм; рекомендуется)', 'Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors',
     f'{HF}/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors', 7.1, 'sdxl'),
    ('dreamshaper', 'DreamShaper 8 (SD 1.5, мультфильмы/иллюстрации, быстрее, ≈3.5 ГБ VRAM)', 'DreamShaper_8_pruned.safetensors',
     f'{HF}/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors', 2.1, 'sd15'),
    ('sdxl_base', 'SDXL base 1.0 (универсальная)', 'sd_xl_base_1.0.safetensors',
     f'{HF}/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors', 6.9, 'sdxl'),
]
IPA = {  # weights per model type + the shared CLIP vision encoder
    'sdxl': ('ipadapter', 'ip-adapter-plus_sdxl_vit-h.safetensors',
             f'{HF}/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors', 0.85),
    'sd15': ('ipadapter', 'ip-adapter-plus_sd15.safetensors',
             f'{HF}/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors', 0.1),
    'clip': ('clip_vision', 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors',
             f'{HF}/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors', 2.5),
}
VOICES = [('ru_RU-dmitri-medium', 'ru/ru_RU/dmitri/medium', 'русский, мужской (рассказчик)'),
          ('ru_RU-irina-medium', 'ru/ru_RU/irina/medium', 'русский, женский'),
          ('ru_RU-denis-medium', 'ru/ru_RU/denis/medium', 'русский, мужской (персонаж)'),
          ('en_US-ryan-high', 'en/en_US/ryan/high', 'English, male'),
          ('en_US-amy-medium', 'en/en_US/amy/medium', 'English, female')]


# ------------------------------------------------------------------ output / questions
def ok(m):
    print(f'[OK] {m}')


def warn(m):
    print(f'[!]  {m}')


def bad(m):
    print(f'[X]  {m}')


def head(m):
    print(f'\n=== {m} ===')


def ask(q, default=False):
    if YES:
        print(f'{q} [авто: да]')
        return True
    if not sys.stdin or not sys.stdin.isatty():
        print(f'{q} [нет ответа — пропускаю; --yes отвечает «да»]')
        return False
    a = input(f'{q} [{"Д/н" if default else "д/Н"}]: ').strip().lower()
    return default if not a else a[0] in 'дy'


def choose(q, options, default=1):
    """options: list of labels -> list of chosen indexes (comma-separated input, 0 = none)."""
    for i, o in enumerate(options, 1):
        print(f'  {i}. {o}')
    if YES:
        return [default - 1] if default else []
    if not sys.stdin or not sys.stdin.isatty():
        return []
    a = input(f'{q} (номера через запятую, 0 — ничего, Enter — {default}): ').strip()
    if not a:
        return [default - 1] if default else []
    out = []
    for x in re.split(r'[,\s]+', a):
        if x.isdigit() and 1 <= int(x) <= len(options):
            out.append(int(x) - 1)
    return out


# ------------------------------------------------------------------ state
def load_state():
    try:
        return json.loads(STATE.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_state(st):
    AI.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding='utf-8')


# ------------------------------------------------------------------ system checks
def free_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / GB


def need_space(path, gb, what):
    free = free_gb(path)
    if free < gb + 2:
        bad(f'Мало места для «{what}»: нужно ≈{gb:.1f} ГБ + 2 ГБ запаса, свободно {free:.1f} ГБ на диске {Path(path).anchor}. '
            f'Освободите место и запустите install_ai.bat снова (скачанное продолжится).')
        return False
    return True


def gpu():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        name, mem, drv = [x.strip() for x in out[0].split(',')]
        full = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=20).stdout
        cuda = re.search(r'CUDA Version:\s*([\d.]+)', full)
        return {'name': name, 'vram_gb': int(mem) / 1024, 'driver': drv, 'cuda': cuda.group(1) if cuda else ''}
    except Exception:
        return None


def system_checks():
    head('Проверка системы')
    v = sys.version_info
    (ok if (3, 10) <= v[:2] <= (3, 12) else warn)(f'Python {platform.python_version()} в окружении {sys.prefix}'
                                                   + ('' if (3, 10) <= v[:2] <= (3, 12) else ' — проверено на 3.10–3.12'))
    if os.name != 'nt':
        warn('Скрипт рассчитан на Windows; на этой системе установите ComfyUI/Ollama вручную, проверка работает.')
    g = gpu()
    if not g:
        warn('Видеокарта NVIDIA не найдена (nvidia-smi). ComfyUI на CPU непригоден — изображения будут недоступны.')
    else:
        ok(f'{g["name"]}, {g["vram_gb"]:.1f} ГБ VRAM, драйвер {g["driver"]}, CUDA {g["cuda"] or "?"}')
        try:
            if g['cuda'] and tuple(int(x) for x in g['cuda'].split('.')[:2]) < (12, 8):
                warn('Портативный ComfyUI собран под CUDA 12.8+: обновите драйвер NVIDIA (Game Ready / Studio 570+).')
        except ValueError:
            pass
        if g['vram_gb'] < 7.5:
            warn('Меньше 8 ГБ VRAM: выбирайте SD 1.5 (DreamShaper 8) вместо SDXL.')
    (ok if shutil.which('ffmpeg') or (ROOT / 'ffmpeg' / 'bin' / 'ffmpeg.exe').is_file() else bad)(
        'FFmpeg ' + ('найден' if shutil.which('ffmpeg') or (ROOT / 'ffmpeg' / 'bin' / 'ffmpeg.exe').is_file()
                     else 'не найден — сначала setup.bat'))
    ok(f'Свободно {free_gb(ROOT):.1f} ГБ на диске проекта {ROOT.anchor}')
    return g


# ------------------------------------------------------------------ downloads (resume, no duplicates)
def remote_size(url):
    import requests
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        if r.status_code < 400:
            return int(r.headers.get('x-linked-size') or r.headers.get('content-length') or 0) or None
    except Exception:
        pass
    return None


def file_ok(path, kind=None):
    """Cheap integrity check of a finished file: safetensors / json headers parse."""
    p = Path(path)
    kind = kind or p.suffix
    try:
        if kind == '.safetensors':
            with open(p, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                if not 2 <= n < 200_000_000:
                    return False
                json.loads(f.read(n))
        elif kind == '.json':
            json.loads(p.read_text(encoding='utf-8'))
        return p.stat().st_size > 0
    except Exception:
        return False


def download(url, dest, label, expect_gb=None):
    """-> True when dest is complete. Skips a complete file, resumes a .part, never re-downloads twice."""
    import requests
    dest = Path(dest)
    size = remote_size(url)
    if dest.is_file():
        if (size is None or dest.stat().st_size == size) and file_ok(dest):
            ok(f'{label}: уже есть ({dest.stat().st_size / GB:.2f} ГБ)')
            return True
        warn(f'{label}: файл {dest.name} есть, но размер/заголовок не совпадает — оставляю его как есть '
             f'(переименуйте или удалите сами, если хотите скачать заново)')
        return False
    if size is None and expect_gb is None:
        bad(f'{label}: нет доступа к {url.split("/")[2]} — проверьте интернет и запустите снова')
        return False
    total = size or int(expect_gb * GB)
    if not need_space(dest.parent, total / GB, label):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    for attempt in range(1, 6):
        have = part.stat().st_size if part.is_file() else 0
        if size and have >= size:
            break
        try:
            hdr = {'Range': f'bytes={have}-'} if have else {}
            with requests.get(url, headers=hdr, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                if have and r.status_code != 206:          # server ignored Range: start again
                    have = 0
                mode = 'ab' if have else 'wb'
                t0, last, got = time.time(), 0, have
                with open(part, mode) as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - last > 1:
                            last = time.time()
                            speed = (got - have) / max(1e-3, last - t0) / 1024 / 1024
                            print(f'\r     {label}: {got / GB:.2f} / {total / GB:.2f} ГБ  {speed:.1f} МБ/с   ',
                                  end='', flush=True)
            print()
            break
        except Exception as exc:
            print()
            warn(f'{label}: обрыв ({str(exc)[:120]}), попытка {attempt}/5 — продолжаю с места остановки')
            time.sleep(min(30, 3 * attempt))
    else:
        bad(f'{label}: не удалось скачать. Запустите install_ai.bat снова — загрузка продолжится.')
        return False
    if size and part.stat().st_size != size:
        bad(f'{label}: скачано {part.stat().st_size} из {size} байт — запустите снова, загрузка продолжится')
        return False
    if not file_ok(part, dest.suffix):
        part.replace(dest.with_name(dest.name + '.bad'))
        bad(f'{label}: скачанный файл повреждён (заголовок не читается), отложен как {dest.name}.bad. '
            f'Запустите снова — скачается заново.')
        return False
    part.replace(dest)
    ok(f'{label}: скачано → {dest}')
    return True


# ------------------------------------------------------------------ Ollama
def ollama_exe():
    p = shutil.which('ollama')
    if p:
        return p
    cand = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Ollama' / 'ollama.exe'
    return str(cand) if cand.is_file() else None


def ollama_up(host='http://127.0.0.1:11434'):
    import requests
    try:
        r = requests.get(host + '/api/tags', timeout=3)
        return [m['name'] for m in r.json().get('models', [])]
    except Exception:
        return None


def start_ollama():
    exe = ollama_exe()
    if not exe or ollama_up() is not None:
        return ollama_up() is not None
    flags = 0x00000008 | 0x00000200 if os.name == 'nt' else 0          # DETACHED_PROCESS | NEW_PROCESS_GROUP
    subprocess.Popen([exe, 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    for _ in range(20):
        time.sleep(1)
        if ollama_up() is not None:
            return True
    return False


def setup_ollama(st):
    head('Ollama — локальная LLM для сценариев, раскадровки и описаний')
    exe = ollama_exe()
    if exe:
        ok(f'Ollama установлена: {exe}')
    elif ask('Ollama не найдена. Установить (≈1 ГБ, установщик Ollama)?'):
        if shutil.which('winget'):
            subprocess.run(['winget', 'install', '-e', '--id', 'Ollama.Ollama', '--accept-package-agreements',
                            '--accept-source-agreements'])
        else:
            setup = DL / 'OllamaSetup.exe'
            if download(OLLAMA_SETUP, setup, 'Установщик Ollama', expect_gb=1.0):
                print('     Откроется установщик Ollama — завершите его и вернитесь сюда.')
                subprocess.run([str(setup)])
        exe = ollama_exe()
        if not exe:
            bad('Ollama не установилась. Установите вручную с ollama.com и запустите install_ai.bat снова.')
            return
    else:
        warn('Пропущено: без LLM сценарии и описания строятся по шаблонам.')
        return
    if not start_ollama():
        bad('Сервер Ollama не отвечает на 127.0.0.1:11434. Запустите приложение Ollama и повторите.')
        return
    have = ollama_up() or []
    ok('Модели Ollama: ' + (', '.join(have) or 'нет'))
    labels = [f'{m} — {gb} ГБ, {d}' + (' [уже есть]' if any(h.startswith(m) for h in have) else '')
              for m, gb, d in LLMS]
    picks = choose('Какие модели скачать?', labels, default=1)
    chosen = None
    for i in picks:
        m, gb, _ = LLMS[i]
        chosen = chosen or m
        if any(h.startswith(m) for h in have):
            ok(f'{m}: уже скачана')
            continue
        if not need_space(Path.home(), gb, m) or not ask(f'Скачать {m} (≈{gb} ГБ)?', True):
            continue
        if subprocess.run([exe, 'pull', m]).returncode:
            bad(f'{m}: ошибка загрузки (см. выше). Запустите install_ai.bat снова — Ollama продолжит.')
            chosen = None if chosen == m else chosen
    have = ollama_up() or []
    model = chosen or st.get('ollama_model') or next((m for m, _, _ in LLMS if any(h.startswith(m) for h in have)), None)
    if model and any(h.startswith(model) for h in have):
        st['ollama_model'] = model
        ok(f'ClipForge будет использовать {model}')


# ------------------------------------------------------------------ ComfyUI
def comfy_dir(st):
    for c in [st.get('comfy_dir'), AI / 'ComfyUI_windows_portable']:
        if c and (Path(c) / 'ComfyUI' / 'main.py').is_file():
            return Path(c)
    return None


def find_7z():
    for c in [shutil.which('7z'), r'C:\Program Files\7-Zip\7z.exe', r'C:\Program Files (x86)\7-Zip\7z.exe']:
        if c and Path(c).is_file():
            return c
    return None


def setup_comfy(st, g):
    head('ComfyUI — генерация изображений (SDXL / SD 1.5, IP-Adapter)')
    if '--comfy-dir' in sys.argv:
        p = Path(sys.argv[sys.argv.index('--comfy-dir') + 1])
        if (p / 'ComfyUI' / 'main.py').is_file():
            st['comfy_dir'] = str(p)
        else:
            bad(f'{p}: это не папка портативного ComfyUI (нет ComfyUI\\main.py)')
    d = comfy_dir(st)
    if d:
        ok(f'ComfyUI: {d}')
    else:
        if not g:
            warn('Без видеокарты NVIDIA ComfyUI ставить не стоит — пропускаю.')
            return
        if not ask('ComfyUI не найден. Скачать портативную сборку для NVIDIA (≈2.1 ГБ архив, ≈7 ГБ после распаковки)?'):
            warn('Пропущено. Уже есть ComfyUI? Запустите: install_ai.bat --comfy-dir "D:\\путь\\ComfyUI_windows_portable"')
            return
        arc = DL / 'ComfyUI_windows_portable_nvidia.7z'
        if not need_space(AI, 2.1 + 7, 'ComfyUI') or not download(COMFY_URL, arc, 'ComfyUI portable', 2.1):
            return
        seven = find_7z()
        print('     Распаковка (несколько минут)...')
        cmd = [seven, 'x', '-y', f'-o{AI}', str(arc)] if seven else ['tar', '-xf', str(arc), '-C', str(AI)]
        r = subprocess.run(cmd)
        d = comfy_dir(st)
        if r.returncode or not d:
            bad('Не удалось распаковать архив ComfyUI. Установите 7-Zip (7-zip.org) и запустите install_ai.bat снова — '
                'архив уже скачан, повторно качать не придётся.')
            return
        ok(f'ComfyUI распакован: {d}')
        if ask(f'Удалить архив {arc.name} (2.1 ГБ), он больше не нужен?', True):
            arc.unlink(missing_ok=True)
    st['comfy_dir'] = str(d)
    nodes = d / 'ComfyUI' / 'custom_nodes' / 'ComfyUI_IPAdapter_plus'
    if nodes.is_dir():
        ok('Узлы IP-Adapter (ComfyUI_IPAdapter_plus) установлены')
    elif ask('Установить узлы IP-Adapter (ComfyUI_IPAdapter_plus, 0.1 МБ) — постоянная внешность персонажей?', True):
        z = DL / 'ComfyUI_IPAdapter_plus.zip'
        try:
            import requests
            z.parent.mkdir(parents=True, exist_ok=True)
            z.write_bytes(requests.get(IPA_NODES_URL, timeout=60).content)
            with zipfile.ZipFile(z) as zf:
                top = zf.namelist()[0].split('/')[0]
                zf.extractall(nodes.parent)
            (nodes.parent / top).rename(nodes)
            z.unlink(missing_ok=True)
            ok('Узлы IP-Adapter установлены (перезапустите ComfyUI)')
        except Exception as exc:
            bad(f'IP-Adapter: {str(exc)[:150]}. Установите через ComfyUI Manager или распакуйте репозиторий '
                f'cubiq/ComfyUI_IPAdapter_plus в {nodes.parent}')
    models = d / 'ComfyUI' / 'models'
    head('Модели изображений (ComfyUI\\models\\checkpoints)')
    if g and g['vram_gb'] < 7.5:
        warn('Мало VRAM: рекомендую DreamShaper 8 (SD 1.5).')
    have = [c for c in CHECKPOINTS if (models / 'checkpoints' / c[2]).is_file()]
    labels = [f'{c[1]} — {c[4]} ГБ' + (' [уже есть]' if c in have else '') for c in CHECKPOINTS]
    picks = choose('Какие модели скачать?', labels, default=0 if have else (2 if g and g['vram_gb'] < 7.5 else 1))
    for i in picks:
        key, name, fname, url, gb, mtype = CHECKPOINTS[i]
        dest = models / 'checkpoints' / fname
        if dest.is_file() or ask(f'Скачать {name} (≈{gb} ГБ)?', True):
            download(url, dest, name, gb)
    found = sorted(p.name for p in (models / 'checkpoints').glob('*.safetensors')) if (models / 'checkpoints').is_dir() else []
    known = {c[2]: c[5] for c in CHECKPOINTS}
    if st.get('checkpoint') not in found:
        st['checkpoint'] = next((c[2] for c in CHECKPOINTS if c[2] in found), found[0] if found else None)
    if st.get('checkpoint'):
        st['image_model'] = known.get(st['checkpoint']) or ('sdxl' if 'xl' in st['checkpoint'].lower() else 'sd15')
        ok(f'ClipForge будет использовать {st["checkpoint"]} ({st["image_model"].upper()})')
    else:
        warn('Нет ни одной модели изображений — генерация кадров будет недоступна.')
    types = {known.get(f) or ('sdxl' if 'xl' in f.lower() else 'sd15') for f in found}
    if types and nodes.is_dir():
        need = [IPA[t] for t in sorted(types)] + [IPA['clip']]
        missing = [x for x in need if not (models / x[0] / x[1]).is_file()]
        if missing and ask('Скачать веса IP-Adapter (' + ' + '.join(f'{x[1]} {x[3]} ГБ' for x in missing) + ')?', True):
            for sub, fname, url, gb in missing:
                download(url, models / sub / fname, fname, gb)
        elif not missing:
            ok('Веса IP-Adapter на месте')


def comfy_running(port=8188):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0


def start_comfy(st):
    d = comfy_dir(st)
    if not d or comfy_running():
        return comfy_running()
    py = d / 'python_embeded' / 'python.exe'
    if os.name != 'nt' or not py.is_file():
        return False
    subprocess.Popen(f'start "ComfyUI" /min /D "{d}" "{py}" -s ComfyUI\\main.py --windows-standalone-build '
                     f'--disable-auto-launch', shell=True)
    return True


# ------------------------------------------------------------------ Piper
def piper_installed():
    try:
        import piper  # noqa: F401
        return True
    except Exception:
        return False


def setup_piper(st):
    head('Piper — локальная озвучка (CPU)')
    if piper_installed():
        ok('piper-tts установлен в окружение')
    elif ask('Установить piper-tts в окружение env (≈30 МБ)?', True):
        cons = Path(sys.prefix) / 'clipforge-constraints.txt'
        cmd = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', 'piper-tts>=1.3,<2']
        if cons.is_file():
            cmd += ['-c', str(cons)]
        if subprocess.run(cmd).returncode or not piper_installed():
            bad('piper-tts не установился (см. выше). Альтернатива: локальный TTS-сервер с OpenAI API '
                '(страница «Генерация»).')
            return
        ok('piper-tts установлен')
    else:
        return
    have = {p.stem for p in PIPER_DIR.glob('*.onnx')} if PIPER_DIR.is_dir() else set()
    labels = [f'{v} — {d}' + (' [уже есть]' if v in have else '') for v, _, d in VOICES]
    for i in choose('Какие голоса скачать (по 60–120 МБ)?', labels, default=1):
        v, path, _ = VOICES[i]
        for ext in ('.onnx', '.onnx.json'):
            download(f'{HF}/rhasspy/piper-voices/resolve/main/{path}/{v}{ext}', PIPER_DIR / f'{v}{ext}', v + ext, 0.12)
    have = {p.stem for p in PIPER_DIR.glob('*.onnx')} if PIPER_DIR.is_dir() else set()
    st['voice_ru'] = st.get('voice_ru') if st.get('voice_ru') in have else next(
        (v for v, _, _ in VOICES if v.startswith('ru') and v in have), None)
    st['voice_en'] = st.get('voice_en') if st.get('voice_en') in have else next(
        (v for v, _, _ in VOICES if v.startswith('en') and v in have), None)


# ------------------------------------------------------------------ configure ClipForge
def django_setup():
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'clipforge.settings')
    import django
    django.setup()


def configure(st):
    head('Настройка ClipForge')
    django_setup()
    from studio import genmedia
    ch = {}
    if st.get('checkpoint'):
        ch.update(checkpoint=st['checkpoint'], image_model=st.get('image_model', 'sdxl'), comfy_url='http://127.0.0.1:8188')
    if piper_installed() and (st.get('voice_ru') or st.get('voice_en')):
        ch.update(tts_provider='piper', piper_dir=str(PIPER_DIR))
        ch.update({k: st[k] for k in ('voice_ru', 'voice_en') if st.get(k)})
    genmedia.save_config(ch)
    ok('Настройки изображений и озвучки сохранены (страница «Генерация» показывает их же)')
    lines = ['@echo off', 'rem Created by install_ai.bat (run it again to change). Loaded by start.bat.']
    if st.get('ollama_model'):
        lines += ['set "USE_OLLAMA=1"', f'set "OLLAMA_MODEL={st["ollama_model"]}"']
    ENV_BAT.write_text('\r\n'.join(lines) + '\r\n', encoding='utf-8')
    ok(f'{ENV_BAT.name}: LLM {"= " + st["ollama_model"] if st.get("ollama_model") else "выключена"}'
       + (', ComfyUI запускается вместе с start.bat' if st.get('comfy_dir') else ''))


# ------------------------------------------------------------------ check
def check(st=None):
    st = st or load_state()
    head('Проверка провайдеров')
    problems = 0
    models = ollama_up()
    want = os.environ.get('OLLAMA_MODEL') or st.get('ollama_model')
    if models is None:
        warn('Ollama не отвечает (127.0.0.1:11434) — сценарии по шаблонам. Запустите Ollama или install_ai.bat.')
    elif want and not any(m.startswith(want) for m in models):
        bad(f'Ollama работает, но модели {want} нет. install_ai.bat → скачать модель.')
        problems += 1
    else:
        ok(f'Ollama: {", ".join(models) or "нет моделей"}' + (f' (используется {want})' if want else ''))
    django_setup()
    from studio import genmedia
    s = genmedia.status()
    c, t = s.get('comfy') or {}, s.get('tts') or {}
    if not c.get('ok'):
        bad(f'ComfyUI недоступен ({str(c.get("error", ""))[:100]}). '
            + ('Запустите start.bat — он поднимет ComfyUI.' if st.get('comfy_dir') else 'install_ai.bat → установить ComfyUI.'))
        problems += 1
    else:
        ok(f'ComfyUI {c.get("version") or ""}: {c.get("device") or ""}')
        (ok if c.get('checkpoint_ok') else bad)(
            f'Модель изображений: {genmedia.config().get("checkpoint") or "не выбрана"}'
            + ('' if c.get('checkpoint_ok') else f' — нет в ComfyUI (есть: {", ".join(c.get("checkpoints", [])[:4]) or "ничего"})'))
        problems += not c.get('checkpoint_ok')
        (ok if c.get('ip_adapter') else warn)('IP-Adapter ' + ('установлен' if c.get('ip_adapter')
                                                                 else 'не установлен — внешность героев только по описанию'))
    (ok if t.get('ok') else bad)(f'Озвучка ({t.get("provider")}): ' + (f'голосов {len(t.get("voices", {}))}'
                                                                       if t.get('ok') else str(t.get('error', ''))))
    problems += not t.get('ok')
    print()
    (ok if not problems else warn)('Всё готово к Cartoon Studio.' if not problems else
                                   f'Не готово пунктов: {problems} (см. [X] выше). Cartoon Studio остановится на этих этапах '
                                   f'с той же причиной; остальной ClipForge работает. Настройки — страница «Генерация».')
    return 1 if problems else 0


def main():
    cmd = next((a for a in sys.argv[1:] if not a.startswith('--') and a in ('menu', 'check', 'start-services')), 'menu')
    st = load_state()
    if cmd == 'check':
        return check(st)
    if cmd == 'start-services':
        if os.environ.get('USE_OLLAMA') == '1' and ollama_exe() and ollama_up() is None:
            print('     Запускаю Ollama...' if start_ollama() else '[!]  Ollama не запустилась')
        if st.get('comfy_dir') and not comfy_running():
            print('     Запускаю ComfyUI в отдельном окне (свёрнуто)...' if start_comfy(st) else '[!]  ComfyUI не запустился')
        return 0
    print(__doc__.split('\n\n')[0])
    g = system_checks()
    for step in (setup_ollama, lambda s: setup_comfy(s, g), setup_piper):
        try:
            step(st)
        except KeyboardInterrupt:
            print()
            warn('Прервано. Запустите install_ai.bat снова — готовое не пропадёт, загрузки продолжатся.')
            save_state(st)
            return 1
        save_state(st)
    configure(st)
    if st.get('comfy_dir') and not comfy_running() and ask('Запустить ComfyUI сейчас для проверки?', True):
        start_comfy(st)
        print('     Жду ComfyUI (первый запуск до пары минут)...')
        for _ in range(150):
            if comfy_running():
                break
            time.sleep(2)
    return check(st)


if __name__ == '__main__':
    sys.exit(main())
