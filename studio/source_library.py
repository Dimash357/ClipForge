"""Local, rights-cleared source lookup by canonical YouTube video ID."""
from pathlib import Path
from django.conf import settings
from .youtube import parse_youtube_url

EXTENSIONS = ('.mp4', '.mov', '.mkv', '.webm')
MAX_BYTES = 1024 * 1024 * 1024

def find_source(url):
    video_id = parse_youtube_url(url)
    library = Path(settings.SOURCE_LIBRARY_ROOT).resolve()
    library.mkdir(parents=True, exist_ok=True)
    for ext in EXTENSIONS:
        candidate = library / (video_id + ext)
        if candidate.is_file() and not candidate.is_symlink():
            if candidate.stat().st_size > MAX_BYTES:
                raise ValueError('Исходник превышает лимит 1 ГБ.')
            return candidate
    return None
