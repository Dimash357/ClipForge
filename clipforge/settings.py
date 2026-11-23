from pathlib import Path
import os
BASE_DIR=Path(__file__).resolve().parent.parent
SECRET_KEY=os.getenv('DJANGO_SECRET_KEY','local-only-development-key-change-before-sharing')
DEBUG=os.getenv('DJANGO_DEBUG','1')=='1'
ALLOWED_HOSTS=['127.0.0.1','localhost']
INSTALLED_APPS=['django.contrib.admin','django.contrib.auth','django.contrib.contenttypes','django.contrib.sessions','django.contrib.messages','django.contrib.staticfiles','studio']
MIDDLEWARE=['django.middleware.security.SecurityMiddleware','django.contrib.sessions.middleware.SessionMiddleware','django.middleware.common.CommonMiddleware','django.middleware.csrf.CsrfViewMiddleware','django.contrib.auth.middleware.AuthenticationMiddleware','django.contrib.messages.middleware.MessageMiddleware','django.middleware.clickjacking.XFrameOptionsMiddleware']
ROOT_URLCONF='clipforge.urls'
TEMPLATES=[{'BACKEND':'django.template.backends.django.DjangoTemplates','DIRS':[],'APP_DIRS':True,'OPTIONS':{'context_processors':['django.template.context_processors.debug','django.template.context_processors.request','django.contrib.auth.context_processors.auth','django.contrib.messages.context_processors.messages']}}]
WSGI_APPLICATION='clipforge.wsgi.application'
DATABASES={'default':{'ENGINE':'django.db.backends.sqlite3','NAME':BASE_DIR/'db.sqlite3','OPTIONS':{'timeout':30},
                       # tests use a temporary *file* database: the default in-memory shared-cache DB fails with
                       # "database table is locked" when a worker thread writes while the test thread reads
                       # (WAL + busy_timeout, used by the real db.sqlite3, only work for file databases).
                       'TEST':{'NAME':str(Path(__import__('tempfile').gettempdir())/f'clipforge_test_{os.getpid()}.sqlite3')}}}
STATIC_URL='static/'
MEDIA_URL='media/'
MEDIA_ROOT=BASE_DIR/'media'
DEFAULT_AUTO_FIELD='django.db.models.BigAutoField'
DATA_UPLOAD_MAX_MEMORY_SIZE=104857600
FILE_UPLOAD_MAX_MEMORY_SIZE=2621440

SOURCE_LIBRARY_ROOT=BASE_DIR / "source_library"

# --- ClipForge v5 ---
# Heavy jobs (whisper / ffmpeg) run through one in-process queue. 1 is right for an 8 GB GPU.
MAX_CONCURRENT_JOBS = max(1, int(os.getenv('CLIPFORGE_MAX_JOBS', '1')))
# Tests never download models (they use a model that is already there, or the heuristic fallback)
import sys as _sys
SOUND_MODEL_DOWNLOAD = not (len(_sys.argv) > 1 and _sys.argv[1] == 'test') and os.getenv('CLIPFORGE_SOUND_DOWNLOAD', '1') == '1'
# Largest source video (upload, YouTube download, source_library). Long streams in 1080p easily exceed 1 GB.
# experimental: render Stream Clipper clips through the Timeline IR renderer (falls back to the classic one on error)
TIMELINE_RENDERER = os.getenv('CLIPFORGE_TIMELINE_RENDERER', '0').lower() in ('1', 'true', 'on', 'yes')
MAX_SOURCE_BYTES = int(float(os.getenv('CLIPFORGE_MAX_SOURCE_GB', '8')) * 1024 ** 3)
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOGGING = {
    'version': 1, 'disable_existing_loggers': False,
    'formatters': {'std': {'format': '%(asctime)s %(levelname)s %(name)s: %(message)s'}},
    'handlers': {
        'console': {'class': 'logging.StreamHandler', 'formatter': 'std'},
        'file': {'class': 'logging.handlers.RotatingFileHandler', 'filename': str(LOG_DIR / 'clipforge.log'),
                 'maxBytes': 2_000_000, 'backupCount': 3, 'encoding': 'utf-8', 'formatter': 'std'},
    },
    'filters': {'quiet_polling': {'()': 'clipforge.logfilters.QuietPolling'}},
    'loggers': {'studio': {'handlers': ['console', 'file'], 'level': os.getenv('CLIPFORGE_LOG_LEVEL', 'INFO')},
                # runserver request lines without the status polling every 1-2 s (it hid ClipForge's own messages)
                'django.server': {'handlers': ['server_console'], 'level': 'INFO', 'propagate': False}},
}
LOGGING['formatters']['server'] = {'format': '[%(asctime)s] %(message)s', 'datefmt': '%d/%b/%Y %H:%M:%S'}
LOGGING['handlers']['server_console'] = {'class': 'logging.StreamHandler', 'filters': ['quiet_polling'],
                                         'formatter': 'server'}
