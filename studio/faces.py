"""Face detection: YuNet (OpenCV DNN, bundled 232 KB ONNX model) with the Haar cascade as fallback.

YuNet finds faces that Haar misses: turned heads, glasses, dark scenes, small webcams (down to ~10 px) and it
returns eye landmarks (used to put the eyes on the upper third in Reaction Priority).
Set CLIPFORGE_FACE_DETECTOR=haar to force the old detector.
"""
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger('studio.faces')

MODEL = Path(__file__).resolve().parent / 'models' / 'face_detection_yunet_2023mar.onnx'
SCORE = float(os.getenv('CLIPFORGE_FACE_SCORE', '0.6'))
GRAY_SCORE = 0.75
_local = threading.local()          # cv2.FaceDetectorYN is not thread-safe: one instance per thread
_state = {'backend': None, 'error': None}


def _yunet():
    det = getattr(_local, 'yunet', None)
    if det is None:
        import cv2
        det = cv2.FaceDetectorYN.create(str(MODEL), '', (320, 320), SCORE, 0.3, 50)
        _local.yunet = det
    return det


def backend():
    """'yunet' or 'haar' (decided once per process)."""
    if _state['backend'] is None:
        if os.getenv('CLIPFORGE_FACE_DETECTOR', 'auto').lower() == 'haar':
            _state['backend'] = 'haar'
        elif not MODEL.is_file():
            _state['backend'], _state['error'] = 'haar', 'модель YuNet не найдена'
        else:
            try:
                import cv2
                if not hasattr(cv2, 'FaceDetectorYN'):
                    raise RuntimeError(f'OpenCV {cv2.__version__} без FaceDetectorYN')
                _yunet()
                _state['backend'] = 'yunet'
            except Exception as exc:
                _state['backend'], _state['error'] = 'haar', str(exc)[:200]
                log.warning('YuNet unavailable (%s); using Haar cascade', exc)
    return _state['backend']


def status():
    return {'backend': backend(), 'error': _state['error'], 'model': MODEL.name if MODEL.is_file() else None}


def detect(img, min_size=24):
    """-> [{'box': (x, y, w, h), 'score': s, 'eyes': ((xr, yr), (xl, yl)) | None}] in pixels of `img`.
    `img` may be BGR (best) or grayscale."""
    import cv2
    if backend() == 'yunet':
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            det = _yunet()
            det.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, faces = det.detect(bgr)
        except Exception as exc:                     # never let detection break an analysis
            log.warning('YuNet failed on a frame (%s); Haar for this frame', exc)
            faces = None
            return _haar(img, min_size)
        out = []
        if faces is not None:
            min_side = max(8, min_size // 3)        # YuNet is reliable on much smaller faces than Haar
            # YuNet is trained on colour images: on a grey frame it needs a higher score to avoid false hits
            min_score = max(SCORE, GRAY_SCORE) if img.ndim == 2 else SCORE
            for f in faces:
                x, y, w, h = (int(round(v)) for v in f[:4])
                if w < min_side or h < min_side or float(f[14]) < min_score:
                    continue
                out.append({'box': (max(0, x), max(0, y), w, h), 'score': float(f[14]),
                            'eyes': ((float(f[4]), float(f[5])), (float(f[6]), float(f[7])))})
        return out
    return _haar(img, min_size)


def _haar(img, min_size):
    import cv2
    from . import scenes
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return [{'box': b, 'score': 1.0, 'eyes': None} for b in scenes.haar_faces(gray, min_size)]
