"""ClipForge Studio: environment checks used by setup.bat / start.bat / run_tests.bat / diagnose.bat.

Standard library only (it must run before the dependencies are installed). Every check prints a clear message and
returns a non-zero exit code only for problems that would make installation or start fail.

    python tools/preflight.py venv                 must run inside a virtual environment (never the system Python)
    python tools/preflight.py ffmpeg               ffmpeg/ffprobe in PATH, libass ("ass" filter), loudnorm
    python tools/preflight.py constraints FILE     pip constraints that keep the env's current NumPy version
    python tools/preflight.py deps [--pip-check]   key imports work; optional `pip check` (conflicts are warnings)
    python tools/preflight.py backup-db            copy db.sqlite3 to backups/ before pending migrations
    python tools/preflight.py all                  venv + ffmpeg + deps (used by start.bat)
"""
import datetime as _dt
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_PY = (3, 10)
KEY_IMPORTS = [('django', 'Django'), ('numpy', 'numpy'), ('cv2', 'opencv-python-headless'), ('av', 'av'),
               ('faster_whisper', 'faster-whisper'), ('ctranslate2', 'ctranslate2'), ('PIL', 'Pillow'),
               ('requests', 'requests'), ('yt_dlp', 'yt-dlp')]


def ok(msg):
    print(f'[OK] {msg}')


def warn(msg):
    print(f'[!]  {msg}')


def fail(msg):
    print(f'[X]  {msg}')


# ------------------------------------------------------------------ virtual environment
def in_venv():
    return sys.prefix != getattr(sys, 'base_prefix', sys.prefix) or hasattr(sys, 'real_prefix')


def check_venv():
    if sys.version_info < MIN_PY:
        fail(f'Нужен Python {MIN_PY[0]}.{MIN_PY[1]}+; сейчас {sys.version.split()[0]} ({sys.executable}).')
        return 1
    if not in_venv():
        fail(f'Это системный Python ({sys.executable}), а не виртуальное окружение. '
             'Библиотеки в него ставиться не будут. Запустите setup.bat (он использует папку env).')
        return 1
    ok(f'Python {sys.version.split()[0]} из окружения {sys.prefix}')
    return 0


# ------------------------------------------------------------------ ffmpeg
def _run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                           errors='replace', timeout=timeout)
        return p.returncode, p.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, str(exc)


def ffmpeg_report(which=shutil.which, run=_run):
    """-> dict with paths, version, libass/loudnorm availability and a list of problems."""
    rep = {'ffmpeg': which('ffmpeg'), 'ffprobe': which('ffprobe'), 'version': None, 'ass': False, 'loudnorm': False,
           'problems': [], 'warnings': []}
    if not rep['ffmpeg']:
        rep['problems'].append('ffmpeg не найден в PATH.')
    if not rep['ffprobe']:
        rep['problems'].append('ffprobe не найден в PATH.')
    if rep['ffmpeg']:
        code, out = run([rep['ffmpeg'], '-hide_banner', '-version'])
        m = re.search(r'ffmpeg version (\S+)', out or '')
        rep['version'] = m.group(1) if m else None
        if code != 0:
            rep['problems'].append(f'ffmpeg найден ({rep["ffmpeg"]}), но не запускается: {(out or "")[-200:]}')
        else:
            code, out = run([rep['ffmpeg'], '-hide_banner', '-filters'])
            names = {ln.split()[1] for ln in (out or '').splitlines() if len(ln.split()) >= 2}
            rep['ass'] = 'ass' in names
            rep['loudnorm'] = 'loudnorm' in names
            if not rep['ass']:
                rep['problems'].append('В этой сборке FFmpeg нет фильтра "ass" (libass) — субтитры не будут вшиваться. '
                                       'Нужна полная сборка, например "ffmpeg-release-full" с gyan.dev.')
            if not rep['loudnorm']:
                rep['warnings'].append('Нет фильтра loudnorm — громкость −14 LUFS не будет выравниваться.')
    if rep['ffmpeg'] and rep['ffprobe']:
        d1, d2 = Path(rep['ffmpeg']).parent, Path(rep['ffprobe']).parent
        if d1 != d2:
            rep['warnings'].append(f'ffmpeg ({d1}) и ffprobe ({d2}) из разных папок — лучше одна сборка.')
    return rep


def check_ffmpeg():
    rep = ffmpeg_report()
    for w in rep['warnings']:
        warn(w)
    if rep['problems']:
        for p in rep['problems']:
            fail(p)
        print('     Как исправить: скачайте FFmpeg (gyan.dev → ffmpeg-release-full.7z), распакуйте и либо добавьте папку bin\n'
              '     в PATH (Параметры → Система → О программе → Дополнительные параметры → Переменные среды → Path),\n'
              f'     либо положите её как {ROOT / "ffmpeg" / "bin"} — bat-файлы подхватят её сами.\n'
              '     После изменения PATH откройте НОВОЕ окно cmd.')
        return 1                                  # every clip (even without captions) goes through the ass filter
    ok(f'FFmpeg {rep["version"]} ({rep["ffmpeg"]}), libass и loudnorm есть')
    return 0


# ------------------------------------------------------------------ YouTube downloads
def check_js_runtime(which=shutil.which):
    """yt-dlp needs deno (or Node.js) to see all YouTube formats; without it a download may come without video."""
    if which('deno'):
        ok('JavaScript для yt-dlp: deno')
    elif which('node'):
        ok('JavaScript для yt-dlp: Node.js')
    else:
        warn('Для скачивания с YouTube нужен deno или Node.js (иначе YouTube может отдать только звук).\n'
             '     Установка: winget install DenoLand.Deno  — затем откройте новое окно cmd.\n'
             '     Загрузка файлов и папка source_library работают и без этого.')
    return 0


# ------------------------------------------------------------------ models
def fetch_models():
    """Download the sound classifier once (16 MB, SHA-256 checked). Never fatal: without it the loudness
    heuristics are used and the download is retried on first use."""
    sys.path.insert(0, str(ROOT))
    try:
        from studio import model_files
    except Exception as exc:
        warn(f'Не удалось проверить модели: {exc}')
        return 0
    yunet = ROOT / 'studio' / 'models' / 'face_detection_yunet_2023mar.onnx'
    (ok if yunet.is_file() else warn)('Детектор лиц YuNet: ' + ('в комплекте' if yunet.is_file() else 'файл не найден — будет Haar'))
    if model_files.is_ready():
        ok('Классификатор звуков YAMNet уже скачан')
        return 0
    print('     Скачиваю классификатор звуков YAMNet (16 МБ, один раз)...')
    try:
        model_files.download()
        ok(f'YAMNet: {model_files.model_path()}')
    except Exception as exc:
        warn(f'YAMNet не скачан ({str(exc)[:150]}). Ничего страшного: скачается при первой обработке, '
             'а без интернета смех и крики определяются по громкости.')
    return 0


# ------------------------------------------------------------------ dependencies
def installed_version(dist):
    try:
        from importlib import metadata
        return metadata.version(dist)
    except Exception:
        return None


def write_constraints(path):
    """Keep the env's current NumPy: other packages in the same env (e.g. ultralytics needs numpy<2) must not break.
    pip then picks versions of opencv/onnxruntime/... that work with it. Set CLIPFORGE_ALLOW_NUMPY_UPGRADE=1 to
    let pip choose freely."""
    lines = []
    np_ver = installed_version('numpy')
    if np_ver and os.environ.get('CLIPFORGE_ALLOW_NUMPY_UPGRADE') != '1':
        lines.append(f'numpy=={np_ver}')
        ok(f'NumPy {np_ver} уже стоит в окружении — оставляю эту версию (без обновления).')
    elif np_ver:
        warn(f'CLIPFORGE_ALLOW_NUMPY_UPGRADE=1: NumPy {np_ver} может быть обновлена.')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    return 0


def check_deps(pip_check=False):
    bad = 0
    for mod, dist in KEY_IMPORTS:
        try:
            __import__(mod)
        except Exception as exc:
            fail(f'Не импортируется {mod} ({dist}): {str(exc)[:200]}')
            bad += 1
    if bad:
        print('     Запустите setup.bat ещё раз; если ошибка про NumPy/"compiled using NumPy 1.x" — см. README, раздел Windows.')
        return 1
    import numpy
    import cv2
    ok(f'Библиотеки на месте: numpy {numpy.__version__}, opencv {cv2.__version__}, '
       f'Django {installed_version("Django")}, faster-whisper {installed_version("faster-whisper")}')
    if pip_check:
        code, out = _run([sys.executable, '-m', 'pip', 'check'], timeout=120)
        if code == 0:
            ok('pip check: конфликтов зависимостей нет')
        else:
            warn('pip check нашёл конфликты в окружении (ClipForge работает, но проверьте другие пакеты):')
            for ln in (out or '').strip().splitlines()[:15]:
                print('       ' + ln)
            if 'numpy' in (out or '').lower():
                print('     Подсказка: конфликт по NumPy. setup.bat не меняет уже установленную NumPy; если её\n'
                      '     обновило что-то другое, верните нужную версию в env, например:\n'
                      '       env\\Scripts\\python.exe -m pip install "numpy<2"')
    return 0


# ------------------------------------------------------------------ database backup
def pending_migrations():
    """Names of unapplied migrations (needs Django + settings)."""
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'clipforge.settings')
    import django
    django.setup()
    from django.db import connection
    from django.db.migrations.executor import MigrationExecutor
    ex = MigrationExecutor(connection)
    plan = ex.migration_plan(ex.loader.graph.leaf_nodes())
    return [f'{m.app_label}.{m.name}' for m, _ in plan]


def backup_db(db=None, dest_dir=None, force=False, pending=None):
    """Consistent copy (sqlite backup API, includes the WAL) of db.sqlite3 before migrations change it."""
    db = Path(db or ROOT / 'db.sqlite3')
    dest_dir = Path(dest_dir or ROOT / 'backups')
    if not db.is_file():
        ok('База данных ещё не создана — резервная копия не нужна.')
        return None
    if pending is None and not force:
        try:
            pending = pending_migrations()
        except Exception as exc:
            warn(f'Не удалось проверить миграции ({str(exc)[:150]}) — делаю резервную копию на всякий случай.')
            pending = ['?']
    if not force and not pending:
        ok('База данных актуальна, миграции не нужны.')
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    dest = dest_dir / f'db-{stamp}.sqlite3'
    src = sqlite3.connect(str(db))
    try:
        out = sqlite3.connect(str(dest))
        try:
            src.backup(out)
        finally:
            out.close()
    finally:
        src.close()
    what = ''
    if pending:
        what = f' (ожидают миграции: {", ".join(pending)})' if len(pending) <= 4 else f' (ожидают {len(pending)} миграций)'
    ok(f'Резервная копия базы перед миграцией: {dest}{what}')
    return dest


# ------------------------------------------------------------------ CLI
def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == 'venv':
        return check_venv()
    if cmd == 'ffmpeg':
        return check_ffmpeg()
    if cmd == 'constraints':
        return write_constraints(rest[0] if rest else ROOT / 'env' / 'clipforge-constraints.txt')
    if cmd == 'deps':
        return check_deps(pip_check='--pip-check' in rest)
    if cmd == 'backup-db':
        try:
            backup_db()
            return 0
        except Exception as exc:
            fail(f'Не удалось сделать резервную копию базы: {exc}. Миграции не запускаю — данные в безопасности.')
            return 1
    if cmd == 'js':
        return check_js_runtime()
    if cmd == 'models':
        return fetch_models()
    if cmd == 'all':
        return check_venv() or check_ffmpeg() or check_deps() or check_js_runtime()
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
