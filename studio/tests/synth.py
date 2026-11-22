"""Synthetic test videos (no downloads): game-like motion, webcam overlay with a real face, talking head, UI screen."""
import subprocess
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1920, 1080, 24


def _astro():
    from skimage import data
    return data.astronaut()[:, :, ::-1].copy()


def _face():
    """Tight face crop for a webcam overlay (4:3), face ~40% of the width."""
    return np.ascontiguousarray(_astro()[20:200, 100:340])


def _bust():
    """Head-and-shoulders crop for a full-frame talking head, face ~31% of the height."""
    return np.ascontiguousarray(_astro()[0:320, 60:380])


def game_frame(t, rng_seed=3):
    rng = np.random.default_rng(rng_seed)
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    base = np.zeros((H, W, 3), np.float32)
    base[..., 0] = 60 + 80 * np.sin(6 * x + t)
    base[..., 1] = 90 + 70 * np.cos(5 * y - t * 1.3)
    base[..., 2] = 70 + 60 * np.sin(4 * (x + y) + t * 0.7)
    img = np.clip(base, 0, 255).astype(np.uint8)
    for k in range(40):
        cx = int((rng.random() * W + t * (60 + 30 * (k % 5)) * (1 if k % 2 else -1)) % W)
        cy = int((rng.random() * H + t * 40 * (1 if k % 3 else -1)) % H)
        col = tuple(int(c) for c in rng.integers(40, 255, 3))
        cv2.circle(img, (cx, cy), int(20 + rng.random() * 50), col, -1)
        cv2.rectangle(img, (cx, cy), (cx + 60, cy + 30), col[::-1], 3)
    noise = rng.integers(0, 40, (H, W, 1), dtype=np.uint8)
    cv2.putText(img, 'HP 100  AMMO 30', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def cam_overlay(img, corner='top_right', w=420, h=315, margin=0):
    face = cv2.resize(_face(), (w, h))
    x = W - w - margin if 'right' in corner else margin
    y = margin if 'top' in corner else H - h - margin
    img[y:y + h, x:x + w] = face
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 4)
    return img


def talking_frame(t):
    big = cv2.resize(_bust(), (1080, 1080))
    bg = np.full((H, W, 3), (70, 55, 45), np.uint8)
    shift = int(250 * np.sin(t * 0.5))          # head drifts left/right
    x0 = W // 2 - 540 + shift
    bg[:, x0:x0 + 1080] = big
    return bg


def screen_frame(t):
    img = np.full((H, W, 3), 245, np.uint8)
    cv2.rectangle(img, (0, 0), (W, 70), (230, 230, 235), -1)
    for i in range(18):
        cv2.putText(img, f'Settings option {i} : value {(i * 7) % 11}', (80, 130 + i * 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (20, 20, 20), 2)
    cv2.rectangle(img, (1500, 150), (1800, 230), (200, 120, 30), -1)
    cv2.putText(img, 'SAVE', (1560, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    return img


def film_frame(t, subs=True):
    """Dark movie-like shot: face left of centre, burned-in title + yellow subtitle line, edge markers."""
    img = np.full((H, W, 3), (28, 24, 22), np.uint8)
    size = 1100                                           # close-up: face ~ 30% of the frame height
    face = cv2.resize(_bust(), (size, size)).astype(np.float32)
    fade = np.ones(size, np.float32) * 0.9
    k = int(size * 0.4)
    fade[-k:] = np.linspace(0.9, 0.05, k)                 # the shot falls off into darkness on the right (text sits there)
    face = (face * fade[None, :, None]).astype(np.uint8)
    x0 = -230 + int(15 * np.sin(t))
    y0 = -10
    xs = max(0, -x0)
    img[max(0, y0):min(H, y0 + size), max(0, x0):x0 + size] = face[max(0, -y0):min(size, H - y0), xs:]
    img[:, :24] = (0, 0, 255)          # red marker on the far left edge
    img[:, -24:] = (0, 255, 0)         # green marker on the far right edge
    if subs:
        (tw, th), _ = cv2.getTextSize('GENIUS', cv2.FONT_HERSHEY_SIMPLEX, 3.6, 6)
        cv2.putText(img, 'GENIUS', ((W - tw) // 2, 560), cv2.FONT_HERSHEY_SIMPLEX, 3.6, (0, 0, 0), 14)
        cv2.putText(img, 'GENIUS', ((W - tw) // 2, 560), cv2.FONT_HERSHEY_SIMPLEX, 3.6, (255, 255, 255), 6)
        (tw, th), _ = cv2.getTextSize('chess players in order', cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(img, 'chess players in order', ((W - tw) // 2, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8)
        cv2.putText(img, 'chess players in order', ((W - tw) // 2, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (40, 220, 255), 3)
    return img


# (duration, kind)
TIMELINE = [(8, 'game_cam'), (6, 'talking'), (6, 'gameplay'), (6, 'screen')]


def make_video(path, timeline=TIMELINE, audio=True, corner='top_right'):
    path = Path(path)
    total = sum(d for d, _ in timeline)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', 'pipe:0']
    if audio:
        cmd += ['-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=44100:duration={total}']
    cmd += ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20', '-pix_fmt', 'yuv420p']
    if audio:
        cmd += ['-c:a', 'aac', '-shortest']
    cmd += [str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for dur, kind in timeline:
        for _ in range(int(dur * FPS)):
            t = n / FPS
            if kind == 'game_cam':
                f = cam_overlay(game_frame(t), corner)
            elif kind == 'gameplay':
                f = game_frame(t)
            elif kind == 'talking':
                f = talking_frame(t)
            elif kind == 'film':
                f = film_frame(t)
            elif kind == 'film_plain':
                f = film_frame(t, subs=False)
            else:
                f = screen_frame(t)
            proc.stdin.write(f.tobytes())
            n += 1
    proc.stdin.close()
    proc.wait()
    return path
