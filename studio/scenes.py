"""Scene Analyzer + Director planner.

Lightweight and local: ffmpeg decodes ~2 frames/s at 640 px, OpenCV Haar finds faces, a few cheap image
statistics separate gameplay from screen/UI content. No neural network, no downloads.

Scene types: gameplay | talking | game_cam | screen.  Output of `plan_scenes` is a list of segments with
confidence; low-confidence segments fall back to a user-chosen layout instead of guessing.
"""
import logging
import subprocess
import tempfile
import os

import numpy as np

from .procutil import JobCancelled, kill_tree, _flags

log = logging.getLogger('studio.scenes')

SAMPLE_FPS = 2.0
SAMPLE_W = 640
FACE_EVERY = 2          # run face detection on every 2nd sample (1 per second)
MIN_SEG = 3.0          # seconds: shorter scenes are merged into a neighbour (no flicker)
MIN_CONF = 0.55        # below this a segment uses the fallback layout
BIG_FACE_AREA = 0.030  # fraction of frame area -> "talking head"
CAM_FACE_MAX = 0.045
CAM_FACE_MIN = 0.0006

_cascades = None


def _load_cascades():
    global _cascades
    if _cascades is None:
        import cv2
        base = cv2.data.haarcascades
        loaded = []
        for name in ('haarcascade_frontalface_alt2.xml', 'haarcascade_frontalface_default.xml'):
            if loaded:
                break  # alt2 only; 'default' is a fallback if alt2 is missing in a stripped OpenCV build
            c = cv2.CascadeClassifier(base + name)
            if not c.empty():
                loaded.append(c)
        _cascades = loaded
    return _cascades


def detect_faces_gray(gray, min_size=24):
    """-> list of (x, y, w, h) in pixels (frontal Haar, alt2)."""
    for c in _load_cascades():
        boxes = c.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(min_size, min_size))
        return [tuple(int(v) for v in b) for b in boxes]
    return []


# ------------------------------------------------------------------ frame sampling
def sample_frames(source, start, end, src_w, src_h, fps=SAMPLE_FPS, width=SAMPLE_W, cancel=None):
    """Yield (t_abs, bgr ndarray) using an ffmpeg raw pipe. Works for any codec ffmpeg can decode."""
    h = max(2, int(round(width * src_h / src_w / 2)) * 2)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', f'{start:.3f}', '-t', f'{end - start:.3f}', '-i',
           str(source), '-an', '-sn', '-vf', f'fps={fps},scale={width}:{h}:flags=area', '-pix_fmt', 'bgr24',
           '-f', 'rawvideo', 'pipe:1']
    size = width * h * 3
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, creationflags=_flags())
        try:
            k = 0
            while True:
                if cancel and cancel():
                    raise JobCancelled()
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                yield start + k / fps, np.frombuffer(buf, np.uint8).reshape(h, width, 3)
                k += 1
        finally:
            kill_tree(proc)


# ------------------------------------------------------------------ per-frame features
def find_text_boxes(bgr):
    """Burned-in captions / titles: wide, thin, bright, gappy blobs in the lower part of the frame.
    Returns normalized boxes [(x0, x1, y0, y1)]. Bright windows/lamps are rejected by their fill ratio."""
    import cv2
    H, W = bgr.shape[:2]
    y_lo = int(H * 0.45)
    roi = bgr[y_lo:int(H * 0.97)]
    bright = np.minimum(roi[:, :, 2], roi[:, :, 1]) >= 185      # min(R,G): catches white AND yellow captions
    gray = bright.astype(np.uint8) * 255
    mask = gray.copy()
    if mask.mean() > 90:            # mostly white picture (documents, snow): not caption-like
        return []
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, W // 45), 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    raw = bright
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < W * 0.10 or not (H * 0.015 <= h <= H * 0.14) or w / max(h, 1) < 3.5:
            continue
        fill = float(raw[y:y + h, x:x + w].mean())
        if fill > 0.62 or fill < 0.04:      # solid bright blob = window/lamp; almost empty = noise
            continue
        # glyph rhythm: a text line flips bright/dark many times along its height-centre row band
        band = raw[y + h // 4:y + 3 * h // 4 + 1, x:x + w].any(axis=0).astype(np.int8)
        flips = int(np.abs(np.diff(band)).sum())
        if flips < max(6, w // 30):
            continue
        # a caption line is made of several separate glyphs of similar height (teeth, lamps, reflections are not)
        sub = raw[y:y + h, x:x + w].astype(np.uint8)
        n2, _, st2, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        glyphs = sum(1 for k in range(1, n2) if st2[k, 3] >= 0.4 * h and st2[k, 2] <= 1.6 * h and st2[k, 4] >= 4)
        if glyphs < 4:
            continue
        boxes.append((x / W, (x + w) / W, (y + y_lo) / H, (y + h + y_lo) / H))
    return boxes


def frame_features(bgr, prev_small=None, detect=True, last_faces=None):
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    faces = []
    if detect:
        for (x, y, w, h) in detect_faces_gray(gray):
            faces.append({'cx': (x + w / 2) / W, 'cy': (y + h / 2) / H, 'w': w / W, 'h': h / H, 'area': w * h / (W * H)})
    else:
        faces = [dict(f) for f in (last_faces or [])]
    small = cv2.resize(gray, (160, max(2, int(160 * H / W))), interpolation=cv2.INTER_AREA)
    motion = None
    act = None
    if prev_small is not None:
        diff = np.abs(small.astype(np.int16) - prev_small.astype(np.int16))
        motion = float(diff.mean())
        act = motion_centroid(diff)
    g320 = cv2.resize(gray, (320, max(2, int(320 * H / W))), interpolation=cv2.INTER_AREA)
    bh, bw = g320.shape[0] // 16, g320.shape[1] // 16
    blocks = g320[:bh * 16, :bw * 16].reshape(bh, 16, bw, 16).swapaxes(1, 2).reshape(bh, bw, 256)
    uniform = float((blocks.std(axis=2) < 5.0).mean())
    edges = cv2.Canny(g320, 80, 160)
    edge_density = float((edges > 0).mean())
    sob = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return {'faces': faces, 'text': find_text_boxes(bgr), 'motion': motion, 'uniform': uniform, 'edge': edge_density, 'luma': float(gray.mean()),
            'dark': float((gray < 20).mean()), 'act': act}, small, sob


def motion_centroid(diff, thr=18):
    """Where is the action? Centre of the strongly changing pixels (static HUD never changes, so it is ignored).
    -> {'cx', 'cy', 'share'} normalized, or None when (almost) nothing moves."""
    m = diff >= thr
    share = float(m.mean())
    if share < 0.004:
        return None
    h, w = diff.shape
    ys, xs = np.nonzero(m)
    wts = diff[ys, xs].astype(np.float32)
    cx = float((xs * wts).sum() / wts.sum()) / w
    cy = float((ys * wts).sum() / wts.sum()) / h
    # whole-screen change (camera pan, scene cut) has no single centre -> weak target
    return {'cx': round(cx, 3), 'cy': round(cy, 3), 'share': round(share, 3)}


def analyze(source, start, end, src_w, src_h, corner_hint='auto', cancel=None, progress=None, detect_faces=True):
    """Sample [start,end] and return {'samples': [...], 'cam': {...}|None, 'fps':..., 't0':...}."""
    samples, prev, edge_acc, n = [], None, None, 0
    total = max(1, int((end - start) * SAMPLE_FPS))
    last_faces = []
    grays = []
    for t, bgr in sample_frames(source, start, end, src_w, src_h, cancel=cancel):
        # faces are the expensive part: detect once per second, reuse for the in-between sample
        feat, prev, sob = frame_features(bgr, prev, detect=(detect_faces and n % FACE_EVERY == 0), last_faces=last_faces)
        last_faces = feat['faces']
        feat['t'] = t - start
        samples.append(feat)
        edge_acc = sob if edge_acc is None else edge_acc + sob
        grays.append(_gray(bgr))
        n += 1
        if progress:
            progress(min(1.0, n / total))
    edge_mean = (edge_acc / n) if n else None
    cam = estimate_cam(samples, edge_mean, corner_hint) if (samples and detect_faces) else None
    if cam:
        mark_cam_presence(samples, grays, cam)
    return {'samples': samples, 'cam': cam, 'fps': SAMPLE_FPS, 't0': start, 'frame': (src_w, src_h),
            'duration': end - start}


def _gray(bgr):
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _face_in_cam(f, cam, pad=0.02):
    return cam['x'] - pad <= f['cx'] <= cam['x'] + cam['w'] + pad and cam['y'] - pad <= f['cy'] <= cam['y'] + cam['h'] + pad


def mark_cam_presence(samples, grays, cam):
    """Per-sample: is the webcam overlay on screen *right now*?

    1. a face inside the overlay rectangle -> yes
    2. else the overlay's border (inner edges) is as sharp as it was in the face frames -> yes
    3. no usable border: keep 'present' for 2.5 s around face frames
    A game scene without the overlay in the same clip is therefore not mistaken for a split screen.
    """
    import cv2
    n = len(samples)
    face_hit = [any(_face_in_cam(f, cam) for f in s['faces']) for s in samples]
    H, W = grays[0].shape
    left = 'left' in cam['corner']
    top = 'top' in cam['corner']
    x0, x1 = int(cam['x'] * W), int((cam['x'] + cam['w']) * W)
    y0, y1 = int(cam['y'] * H), int((cam['y'] + cam['h']) * H)
    ix = x1 if left else x0
    iy = y1 if top else y0

    def border_strength(g):
        pad = 4
        ys0, ys1 = max(0, y0 + (y1 - y0) // 6), min(H, y1 - (y1 - y0) // 6)
        xs0, xs1 = max(0, x0 + (x1 - x0) // 6), min(W, x1 - (x1 - x0) // 6)
        a = max(0, ix - pad), min(W, ix + pad)
        b = max(0, iy - pad), min(H, iy + pad)
        vs = hs = 0.0
        if a[1] - a[0] > 2 and ys1 > ys0:
            roi = g[ys0:ys1, a[0]:a[1]].astype(np.float32)
            vs = float(np.abs(np.diff(roi, axis=1)).max(axis=1).mean()) if roi.shape[1] > 1 else 0.0
        if b[1] - b[0] > 2 and xs1 > xs0:
            roi = g[b[0]:b[1], xs0:xs1].astype(np.float32)
            hs = float(np.abs(np.diff(roi, axis=0)).max(axis=0).mean()) if roi.shape[0] > 1 else 0.0
        return max(vs, hs)

    strengths = [border_strength(g) for g in grays]
    ref = [st for st, hit in zip(strengths, face_hit) if hit]
    ref_med = float(np.median(ref)) if ref else 0.0
    # compare with the strongest edge the *background* ever shows next to the border to be sure it is a real border
    present = [False] * n
    if cam.get('snapped') and ref_med >= 25:
        for i in range(n):
            present[i] = face_hit[i] or strengths[i] >= 0.6 * ref_med
        cam['presence'] = 'border'
    else:
        hold = int(2.5 * SAMPLE_FPS)
        for i in range(n):
            if face_hit[i]:
                for k in range(max(0, i - hold), min(n, i + hold + 1)):
                    present[k] = True
        cam['presence'] = 'face-hold'
    for s, flag in zip(samples, present):
        s['cam_present'] = flag


# ------------------------------------------------------------------ facecam estimation
def _corner_of(cx, cy):
    return ('top' if cy < 0.5 else 'bottom') + '_' + ('left' if cx < 0.5 else 'right')


def _in_corner_zone(cx, cy):
    return (cx < 0.42 or cx > 0.58) and (cy < 0.45 or cy > 0.55)


def estimate_cam(samples, edge_mean, corner_hint='auto'):
    """Find a stationary small face in one corner. Returns normalized rect + confidence, or None."""
    n = len(samples)
    if n == 0:
        return None
    by_corner = {}
    for s in samples:
        best = None
        for f in s['faces']:
            if CAM_FACE_MIN <= f['area'] <= CAM_FACE_MAX and _in_corner_zone(f['cx'], f['cy']):
                if best is None or f['area'] > best['area']:
                    best = f
        if best:
            by_corner.setdefault(_corner_of(best['cx'], best['cy']), []).append(best)
    manual = corner_hint != 'auto'
    if manual:
        by_corner = {k: v for k, v in by_corner.items() if k == corner_hint}
    if not by_corner:
        return None
    corner, hits = max(by_corner.items(), key=lambda kv: len(kv[1]))
    frac = len(hits) / n
    xs = np.array([h['cx'] for h in hits])
    ys = np.array([h['cy'] for h in hits])
    spread = max(float(xs.std()), float(ys.std()))
    need = 0.10 if manual else 0.25
    if frac < need or spread > 0.04:
        return None
    cx, cy = float(np.median(xs)), float(np.median(ys))
    fw = float(np.median([h['w'] for h in hits]))
    fh = float(np.median([h['h'] for h in hits]))
    return _make_cam_rect(cx, cy, fw, fh, corner, edge_mean, frac)


def _make_cam_rect(cx, cy, fw, fh, corner, edge_mean, frac, aspect=1.3):
    """Normalized (x,y,w,h) of the webcam overlay around a face; optionally snapped to a visible border."""
    # aspect here is normalized-frame ratio; face boxes are roughly square in pixels, frame is 16:9.
    cw = min(0.6, fw * 2.6)     # overlay width  ~ 2.6 x face width  (fw is a fraction of frame width)
    ch = min(0.6, fh * 1.95)    # overlay height ~ 1.95 x face height (fh is a fraction of frame height)
    x = cx - cw / 2
    y = cy - ch / 2
    x = max(0.0, min(1 - cw, x))
    y = max(0.0, min(1 - ch, y))
    snapped = False
    if edge_mean is not None:
        x, y, cw, ch, snapped = _snap_to_edges(edge_mean, x, y, cw, ch, corner)
    conf = min(1.0, 0.4 + 0.8 * frac + (0.1 if snapped else 0.0))
    face_rel = (round(max(0.0, min(1.0, (cx - x) / cw)), 3), round(max(0.0, min(1.0, (cy - y) / ch)), 3))
    return {'x': x, 'y': y, 'w': cw, 'h': ch, 'corner': corner, 'confidence': round(conf, 2), 'frac': round(frac, 2),
            'snapped': snapped, 'source': 'detected', 'face_rel': face_rel,
            'face_h': round(max(0.05, min(1.0, fh / ch)), 3)}


def _snap_to_edges(edge_mean, x, y, w, h, corner):
    """Refine the inner (toward frame centre) edges of the overlay using the time-averaged edge map.
    A webcam border is static, game edges average out, so the border shows up as a ridge."""
    H, W = edge_mean.shape
    left = 'left' in corner
    top = 'top' in corner
    snapped = False
    y0, y1 = int((y + h * 0.15) * H), int((y + h * 0.85) * H)
    x0, x1 = int((x + w * 0.15) * W), int((x + w * 0.85) * W)
    # inner vertical edge
    inner_x = int((x + w) * W) if left else int(x * W)
    span = int(w * W * 0.3)
    lo, hi = max(2, inner_x - span), min(W - 3, inner_x + span)
    if hi - lo > 6 and y1 > y0:
        prof = edge_mean[y0:y1, lo:hi].mean(axis=0)
        k = int(prof.argmax())
        if prof[k] > 1.8 * (np.median(prof) + 1e-3) and prof[k] > 12:
            edge = (lo + k) / W
            if left:
                w = max(0.08, edge - x)
            else:
                w = max(0.08, x + w - edge)
                x = edge
            snapped = True
    inner_y = int((y + h) * H) if top else int(y * H)
    span = int(h * H * 0.3)
    lo, hi = max(2, inner_y - span), min(H - 3, inner_y + span)
    x0, x1 = int((x + w * 0.15) * W), int((x + w * 0.85) * W)
    if hi - lo > 6 and x1 > x0:
        prof = edge_mean[lo:hi, x0:x1].mean(axis=1)
        k = int(prof.argmax())
        if prof[k] > 1.8 * (np.median(prof) + 1e-3) and prof[k] > 12:
            edge = (lo + k) / H
            if top:
                h = max(0.08, edge - y)
            else:
                h = max(0.08, y + h - edge)
                y = edge
            snapped = True
    return x, y, w, h, snapped


def manual_cam(corner):
    """User picked a corner but no face was detected: conventional overlay position (legacy defaults)."""
    cx, cy = {'top_left': (.16, .19), 'top_right': (.84, .19), 'bottom_left': (.16, .81),
              'bottom_right': (.84, .81)}.get(corner, (.84, .19))
    w, h = .30, .42
    return {'x': max(0, min(1 - w, cx - w / 2)), 'y': max(0, min(1 - h, cy - h / 2)), 'w': w, 'h': h,
            'corner': corner, 'confidence': 0.6, 'frac': 0.0, 'snapped': False, 'source': 'manual',
            'face_rel': (0.5, 0.5), 'face_h': 0.45}


# ------------------------------------------------------------------ classification
def _is_talking_face(f):
    """A clearly large face, or a prominent one that is not sitting in a corner (corner faces are webcams)."""
    if f['area'] >= BIG_FACE_AREA:
        return True
    prominent = f['h'] >= 0.15 and f['area'] >= 0.012
    return prominent and not _in_corner_zone(f['cx'], f['cy'])


def classify_sample(s, cam):
    """-> (scene_type, confidence). Rules are ordered and explainable."""
    faces = s['faces']
    big = [f for f in faces if _is_talking_face(f)]
    if cam:
        # a face inside the overlay rectangle is the facecam, not a talking-head scene
        def in_cam(f):
            return cam['x'] - 0.02 <= f['cx'] <= cam['x'] + cam['w'] + 0.02 and \
                   cam['y'] - 0.02 <= f['cy'] <= cam['y'] + cam['h'] + 0.02
        big = [f for f in big if not in_cam(f)]
    cam_here = bool(cam) and s.get('cam_present', True)
    if big:
        a = max(f['area'] for f in big)
        return 'talking', min(1.0, 0.62 + a * 6)
    if cam_here and cam['confidence'] >= 0.5:
        return 'game_cam', min(1.0, 0.45 + 0.55 * cam['confidence'])
    motion = s['motion'] if s['motion'] is not None else 5.0
    if s['uniform'] >= 0.45 and motion <= 6.0 and s['edge'] >= 0.01:
        margin = (s['uniform'] - 0.45) / 0.3 + (6.0 - motion) / 12.0
        return 'screen', min(1.0, 0.55 + 0.25 * margin)
    conf = 0.78 if motion >= 6 else 0.64
    return 'gameplay', conf


def smooth_labels(labels, confs, fps=SAMPLE_FPS, min_seg=MIN_SEG, window=5):
    """Majority filter + minimum-duration merge. Returns runs [(label, i0, i1_excl, mean_conf)]."""
    n = len(labels)
    if n == 0:
        return []
    half = window // 2
    sm = []
    for i in range(n):
        votes = {}
        for k in range(max(0, i - half), min(n, i + half + 1)):
            votes[labels[k]] = votes.get(labels[k], 0.0) + confs[k] * (1.5 if k == i else 1.0)
        sm.append(max(votes, key=votes.get))
    runs = []
    for i, lab in enumerate(sm):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i + 1
        else:
            runs.append([lab, i, i + 1])
    min_n = max(1, int(round(min_seg * fps)))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for idx, r in enumerate(runs):
            if r[2] - r[1] < min_n:
                left = runs[idx - 1] if idx > 0 else None
                right = runs[idx + 1] if idx + 1 < len(runs) else None
                tgt = left if (right is None or (left is not None and (left[2] - left[1]) >= (right[2] - right[1]))) else right
                tgt_i = runs.index(tgt)
                if tgt_i < idx:
                    tgt[2] = r[2]
                else:
                    tgt[1] = r[1]
                runs.pop(idx)
                changed = True
                break
        # merge neighbours that became identical
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2] = r[2]
            else:
                merged.append(r)
        if len(merged) != len(runs):
            changed = True
        runs = merged
    out = []
    for lab, a, b in runs:
        c = float(np.mean(confs[a:b])) if b > a else 0.0
        out.append((lab, a, b, c))
    return out


def plan_scenes(analysis, clip_len, fallback='gameplay', min_conf=MIN_CONF, min_seg=MIN_SEG, forced_cam=None):
    """Turn per-sample labels into stable segments. `fallback` is 'gameplay' or 'smart'."""
    samples = analysis['samples']
    cam = analysis.get('cam') or forced_cam
    fps = analysis['fps']
    if not samples:
        return [{'start': 0.0, 'end': clip_len, 'type': fallback, 'confidence': 0.0, 'fallback': True,
                 'reason': 'no_samples'}]
    labels, confs = [], []
    for s in samples:
        lab, c = classify_sample(s, cam)
        labels.append(lab)
        confs.append(c)
    runs = smooth_labels(labels, confs, fps, min_seg)
    segs = []
    for lab, a, b, conf in runs:
        t0 = a / fps
        t1 = min(clip_len, b / fps)
        fb = conf < min_conf
        segs.append({'start': t0, 'end': t1, 'type': fallback if fb else lab, 'detected': lab,
                     'confidence': round(conf, 2), 'fallback': fb})
    segs[0]['start'] = 0.0
    segs[-1]['end'] = clip_len
    # collapse neighbours that ended up with the same effective type
    out = []
    for sg in segs:
        if out and out[-1]['type'] == sg['type']:
            out[-1]['end'] = sg['end']
            out[-1]['confidence'] = round(min(out[-1]['confidence'], sg['confidence']), 2)
            out[-1]['fallback'] = out[-1]['fallback'] and sg['fallback']
        else:
            out.append(sg)
    return out


# ------------------------------------------------------------------ burned-in text guard
def hardsub_summary(samples, min_share=0.25):
    """Does the source carry burned-in captions/titles? -> {'present', 'share', 'x0', 'x1'} (normalized)."""
    hits = [s for s in samples if s.get('text')]
    share = len(hits) / max(1, len(samples))
    if share < min_share:
        return {'present': False, 'share': round(share, 2)}
    x0 = float(np.percentile([min(b[0] for b in s['text']) for s in hits], 10))
    x1 = float(np.percentile([max(b[1] for b in s['text']) for s in hits], 90))
    return {'present': True, 'share': round(share, 2), 'x0': round(x0, 3), 'x1': round(x1, 3)}


def crop_cut_fraction(samples, path, src_w, crop_w, margin=0.006):
    """Share of caption-bearing samples where the crop window (following `path`) slices through a text box."""
    if not path:
        path = [(0.0, (src_w - crop_w) / 2, 0.0)]
    pts = sorted(path)
    cut = total = 0
    for s in samples:
        if not s.get('text'):
            continue
        t = s['t']
        x = pts[-1][1]
        for (t0, x0, _), (t1, x1, _) in zip(pts, pts[1:]):
            if t0 <= t <= t1:
                x = x0 + (x1 - x0) * (t - t0) / max(1e-6, t1 - t0)
                break
        else:
            if t < pts[0][0]:
                x = pts[0][1]
        left, right = x / src_w, (x + crop_w) / src_w
        total += 1
        if any(b[0] < left - margin or b[1] > right + margin for b in s['text']):
            cut += 1
    return cut / total if total else 0.0


# ------------------------------------------------------------------ face tracking ("virtual camera")
def _choose_targets(samples, t_lo, t_hi, crop_w_frac, avoid_cam=None):
    """Per-sample subject position (cx, cy) in normalized source coords, or None if no face."""
    targets, prev = [], None
    for s in samples:
        if not (t_lo - 1e-6 <= s['t'] < t_hi):
            continue
        faces = s['faces']
        if avoid_cam:
            faces = [f for f in faces if not (avoid_cam['x'] - 0.02 <= f['cx'] <= avoid_cam['x'] + avoid_cam['w'] + 0.02 and
                                              avoid_cam['y'] - 0.02 <= f['cy'] <= avoid_cam['y'] + avoid_cam['h'] + 0.02)]
        faces = [f for f in faces if f['area'] >= 0.002]
        tgt = None
        if faces:
            top = max(f['area'] for f in faces)
            sig = [f for f in faces if f['area'] >= 0.35 * top]
            if len(sig) >= 2:
                x0 = min(f['cx'] - f['w'] / 2 for f in sig)
                x1 = max(f['cx'] + f['w'] / 2 for f in sig)
                if (x1 - x0) <= 0.9 * crop_w_frac:      # both fit -> frame them together
                    tgt = ((x0 + x1) / 2, float(np.mean([f['cy'] for f in sig])))
            if tgt is None:
                cand = max(faces, key=lambda f: f['area'])
                if prev is not None:                    # keep the current subject unless another is clearly bigger
                    near = [f for f in faces if abs(f['cx'] - prev[0]) < 0.12 and f['area'] >= 0.6 * top]
                    if near:
                        cand = max(near, key=lambda f: f['area'])
                tgt = (cand['cx'], cand['cy'])
        targets.append((s['t'], tgt))
        if tgt:
            prev = tgt
    return targets


def action_targets(samples, t_lo, t_hi, avoid_cam=None, smooth=5):
    """Per-sample 'main action' position: motion centroid, median-smoothed; None where nothing moves."""
    raw = []
    for s in samples:
        if not (t_lo - 1e-6 <= s['t'] < t_hi):
            continue
        a = s.get('act')
        if a and a['share'] < 0.6 and not (avoid_cam and _face_in_cam({'cx': a['cx'], 'cy': a['cy']}, avoid_cam, 0.0)):
            raw.append((s['t'], (a['cx'], a['cy'])))
        else:
            raw.append((s['t'], None))
    out = []
    half = smooth // 2
    for k, (t, tgt) in enumerate(raw):
        near = [p for _, p in raw[max(0, k - half):k + half + 1] if p]
        if tgt is None or len(near) < 2:
            out.append((t, tgt if near else None))
            continue
        out.append((t, (float(np.median([p[0] for p in near])), float(np.median([p[1] for p in near])))))
    return out


def track_action(samples, t_lo, t_hi, src_w, src_h, crop_w, crop_h, avoid_cam=None):
    """Virtual camera that follows the main on-screen action (no webcam / no face). Slower than face tracking."""
    tg = action_targets(samples, t_lo, t_hi, avoid_cam)
    return track_path(samples, t_lo, t_hi, src_w, src_h, crop_w, crop_h, targets=tg, vmax_frac=0.25, dead_zone=0.15)


def track_path(samples, t_lo, t_hi, src_w, src_h, crop_w, crop_h, avoid_cam=None, vmax_frac=0.35, targets=None,
               dead_zone=0.08):
    """Smooth virtual-camera path for a crop window. Returns [(t_rel, x_px, y_px)] relative to t_lo.

    Dead-zone + eased follow + speed limit => no jitter, no sudden jumps.
    targets: precomputed [(t, (cx, cy) | None)] (e.g. action_targets); default = faces.
    """
    fps = SAMPLE_FPS
    dt = 1.0 / fps
    cw_frac = crop_w / src_w
    if targets is None:
        targets = _choose_targets(samples, t_lo, t_hi, cw_frac, avoid_cam)
    cx0, cy0 = 0.5, 0.5
    first = next((tg for _, tg in targets if tg), None)
    cam_x, cam_y = ((first[0], first[1]) if first else (cx0, cy0))
    cam_x_px = cam_x * src_w - crop_w / 2
    cam_y_px = cam_y * src_h - crop_h / 2
    dz = dead_zone * crop_w
    vmax = vmax_frac * crop_w
    path = []
    last_tgt = (cam_x, cam_y)
    for t, tgt in targets:
        if tgt:
            last_tgt = tgt
        tx_px = last_tgt[0] * src_w - crop_w / 2
        ty_px = last_tgt[1] * src_h - crop_h / 2
        for attr in ('x', 'y'):
            cur, want = (cam_x_px, tx_px) if attr == 'x' else (cam_y_px, ty_px)
            lim = dz if attr == 'x' else 0.08 * crop_h
            diff = want - cur
            if abs(diff) <= lim:
                step = 0.0
            else:
                step = (abs(diff) - lim * 0.5) * 0.6
                step = min(step, vmax * dt) * (1 if diff > 0 else -1)
            if attr == 'x':
                cam_x_px += step
            else:
                cam_y_px += step
        px = max(0.0, min(src_w - crop_w, cam_x_px))
        py = max(0.0, min(src_h - crop_h, cam_y_px))
        path.append((t - t_lo, px, py))
    if not path:
        path = [(0.0, (src_w - crop_w) / 2, (src_h - crop_h) / 2)]
    return path, bool(first)


def simplify_path(path, tol=3.0):
    """Ramer-Douglas-Peucker on x(t) and y(t) jointly; keeps expressions short."""
    if len(path) <= 2:
        return path
    keep = [0, len(path) - 1]

    def rec(lo, hi):
        if hi - lo < 2:
            return
        (t0, x0, y0), (t1, x1, y1) = path[lo], path[hi]
        worst, idx = 0.0, None
        for k in range(lo + 1, hi):
            t, x, y = path[k]
            u = (t - t0) / (t1 - t0) if t1 > t0 else 0
            d = max(abs(x - (x0 + u * (x1 - x0))), abs(y - (y0 + u * (y1 - y0))))
            if d > worst:
                worst, idx = d, k
        if idx is not None and worst > tol:
            keep.append(idx)
            rec(lo, idx)
            rec(idx, hi)

    rec(0, len(path) - 1)
    return [path[i] for i in sorted(set(keep))]


# ------------------------------------------------------------------ whole-source webcam detection
def detect_source_cam(source, duration, src_w, src_h, corner_hint='auto', windows=12, win_len=1.5, cancel=None):
    """Step 1 of the pipeline: is there a webcam overlay in this stream and where?

    Samples `windows` short spans spread over the whole video (faces + time-averaged edges) and runs the same
    stationary-corner-face estimator as the per-clip analysis. Returns the cam dict or None."""
    if duration <= 0:
        return None
    samples, edge_acc, n = [], None, 0
    step = duration / (windows + 1)
    for k in range(windows):
        a = max(0.0, step * (k + 1) - win_len / 2)
        b = min(duration, a + win_len)
        prev = None
        for t, bgr in sample_frames(source, a, b, src_w, src_h, fps=2.0, cancel=cancel):
            feat, prev, sob = frame_features(bgr, prev, detect=True)
            samples.append(feat)
            edge_acc = sob if edge_acc is None else edge_acc + sob
            n += 1
    if not samples:
        return None
    cam = estimate_cam(samples, edge_acc / n, corner_hint)
    if cam:
        cam['source'] = 'detected_source'
        cam['face_rel'] = list(cam['face_rel'])
    return cam
