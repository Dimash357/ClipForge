"""Gameplay dynamics for highlight scoring: motion intensity, sudden motion bursts and scene cuts.

Decoding a whole multi-hour stream only for this would cost minutes, so the analyzer runs on the *best candidate
spans only* (a capped total duration), at 4 frames/s and 96 px width. Values are normalized against the median
motion of everything analyzed in the same video.
"""
import logging
import os

import numpy as np

from . import scenes as scn

log = logging.getLogger('studio.motion')

FPS = 4.0
WIDTH = 96
MAX_TOTAL_S = 900.0


def merge_spans(spans, gap=2.0):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def color_stats(bgr):
    """(luma 0..255, saturation 0..1, red dominance) of a small BGR frame."""
    f = bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    sat = float(np.mean((mx - mn) / np.maximum(mx, 1.0)))
    luma = float(np.mean(0.114 * b + 0.587 * g + 0.299 * r))
    red = float(np.mean(r - (g + b) / 2.0)) / 255.0
    return luma, sat, red


def screen_events(times, luma, sat, red, cuts, min_gap=1.5):
    """Explainable on-screen 'something happened' markers from per-frame colour statistics.

    fail_screen: picture drops to near-black / loses most of its colour / floods red (death, crash, 'mission
                 failed' screens in most games), flash: sudden bright burst (explosion), cut: hard scene change.
    """
    ev = []
    n = len(times)
    last_t = -1e9
    for k in range(1, n):
        lo = max(0, k - 3)
        ref_l = float(np.median(luma[lo:k]))
        ref_s = float(np.median(sat[lo:k]))
        ref_r = float(np.median(red[lo:k]))
        kind = None
        if ref_l > 40 and luma[k] < 0.35 * ref_l:
            kind = 'fail_screen'          # fade/cut to black
        elif ref_s > 0.25 and sat[k] < 0.35 * ref_s and 25 < luma[k] < 1.3 * ref_l + 10:
            kind = 'fail_screen'          # desaturated "wasted" style screen
        elif red[k] - ref_r > 0.18 and red[k] > 0.15:
            kind = 'fail_screen'          # red damage/death overlay
        elif luma[k] - ref_l > 70:
            kind = 'flash'
        elif cuts[k]:
            kind = 'cut'
        if kind and times[k] - last_t >= min_gap:
            ev.append({'t': round(float(times[k]), 2), 'kind': kind})
            last_t = times[k]
    return ev


def frame_stats(prev, cur):
    """(motion, cut) between two small gray frames."""
    diff = np.abs(cur.astype(np.int16) - prev.astype(np.int16))
    motion = float(diff.mean())
    h1 = np.histogram(prev, bins=16, range=(0, 256))[0].astype(np.float32)
    h2 = np.histogram(cur, bins=16, range=(0, 256))[0].astype(np.float32)
    hist = float(np.abs(h1 - h2).sum() / (2 * max(1.0, h1.sum())))
    cut = hist >= 0.45 or (motion >= 45 and hist >= 0.25)
    return motion, cut


class MotionProfile:
    def __init__(self, times, motion, cuts, luma=None, sat=None, red=None):
        self.t = np.asarray(times, np.float64)
        self.motion = np.asarray(motion, np.float32)
        self.cuts = np.asarray(cuts, bool)
        n = len(self.t)
        self.luma = np.asarray(luma if luma is not None else np.full(n, 100.0), np.float32)
        self.sat = np.asarray(sat if sat is not None else np.full(n, 0.4), np.float32)
        self.red = np.asarray(red if red is not None else np.zeros(n), np.float32)
        self.events = screen_events(self.t, self.luma, self.sat, self.red, self.cuts) if n else []
        self.ref = float(np.median(self.motion)) if len(self.motion) else 1.0
        self.ref = max(self.ref, 1.0)

    def covers(self, start, end, min_share=0.6):
        if not len(self.t):
            return False
        inside = ((self.t >= start) & (self.t < end)).sum()
        return inside >= min_share * (end - start) * FPS

    def span(self, start, end):
        m = (self.t >= start) & (self.t < end)
        if m.sum() < 4:
            return None
        mo = self.motion[m] / self.ref
        top = float(np.mean(np.sort(mo)[-max(1, len(mo) // 4):]))
        # bursts: motion jumps to >= 2x the running level
        lvl = np.convolve(mo, np.ones(8) / 8, mode='same')
        bursts = int(((mo[1:] >= 2.0 * np.maximum(lvl[:-1], 0.25)) & (mo[1:] >= 1.5)).sum())
        evs = [e for e in self.events if start <= e['t'] < end]
        return {'mean': float(mo.mean()), 'top': top, 'cuts': int(self.cuts[m].sum()), 'bursts': bursts,
                'static': float((mo < 0.2).mean()), 'screen_events': evs,
                'fail_screen': sum(1 for e in evs if e['kind'] == 'fail_screen'),
                'flash': sum(1 for e in evs if e['kind'] == 'flash')}


def analyze_spans(source, spans, src_w, src_h, cancel=None, max_total=MAX_TOTAL_S, progress=None):
    """Sample the given absolute spans [(start, end)] -> MotionProfile. progress(frac) over the sampled seconds."""
    import cv2
    times, motion, cuts, lum, sat, red = [], [], [], [], [], []
    budget = max_total
    merged = merge_spans(spans)
    total = min(max_total, sum(b - a for a, b in merged)) or 1.0
    for a, b in merged:
        if budget <= 0:
            break
        b = min(b, a + budget)
        budget -= b - a
        prev = None
        for t, bgr in scn.sample_frames(source, a, b, src_w, src_h, fps=FPS, width=WIDTH, cancel=cancel):
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                mo, cut = frame_stats(prev, g)
                times.append(t)
                motion.append(mo)
                cuts.append(cut)
                lu, sa, rd = color_stats(bgr)
                lum.append(lu)
                sat.append(sa)
                red.append(rd)
            prev = g
            if progress:
                progress(min(1.0, (max_total - budget - (b - t)) / total))
    return MotionProfile(times, motion, cuts, lum, sat, red)


def action_score(f):
    """0..100 'something is happening on screen' score + facts."""
    if not f:
        return None, []
    facts = []
    s = 30.0 + min(30.0, 18.0 * f['top']) + min(12.0, 10.0 * f['mean'])
    if f['bursts']:
        s += min(15.0, 5.0 * f['bursts'])
        facts.append(('action_burst', f['bursts']))
    if f['cuts']:
        s += min(10.0, 4.0 * f['cuts'])
        facts.append(('scene_change', f['cuts']))
    if f['static'] > 0.6:
        s -= 15
        facts.append(('static_picture', None))
    if f['top'] >= 1.6:
        facts.append(('fast_action', None))
    if f.get('fail_screen'):
        s += 14
        facts.append(('fail_screen', f['fail_screen']))
    if f.get('flash'):
        s += 5
        facts.append(('flash', f['flash']))
    return max(0.0, min(100.0, s)), facts


# ------------------------------------------------------------------ whole-source keyframe scan
class ScreenProfile:
    """Coarse whole-video screen timeline from keyframes only (cheap: no full decode)."""

    def __init__(self, times, luma, sat, red, hist_diff):
        self.t = np.asarray(times, np.float64)
        self.luma = np.asarray(luma, np.float32)
        self.sat = np.asarray(sat, np.float32)
        self.red = np.asarray(red, np.float32)
        self.hist = np.asarray(hist_diff, np.float32)
        cuts = self.hist >= 0.45
        self.events = screen_events(self.t, self.luma, self.sat, self.red, cuts, min_gap=2.0) if len(self.t) else []

    def events_in(self, a, b):
        return [e for e in self.events if a <= e['t'] < b]


KEY_DENSE = 1 / 8.0          # keyframes per second needed for the cheap keyframe-only scan


def keyframe_rate(source, probe_seconds=300):
    """Keyframes per second, measured on the first minutes from packet flags (demux only, no decoding)."""
    import subprocess
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-read_intervals',
                              f'%+{int(probe_seconds)}', '-show_entries', 'packet=pts_time,flags', '-of', 'csv=p=0',
                              str(source)], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL).stdout
    except Exception:
        return None
    ts, keys = [], 0
    for line in out.splitlines():
        parts = line.split(',')
        try:
            ts.append(float(parts[0]))
        except (ValueError, IndexError):
            continue
        if len(parts) > 1 and 'K' in parts[1]:
            keys += 1
    if len(ts) < 2:
        return None
    span = max(ts) - min(ts)
    return keys / span if span > 0 else None


def _start_time(source):
    import subprocess
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=start_time', '-of', 'csv=p=0',
                              str(source)], capture_output=True, text=True, timeout=60,
                             stdin=subprocess.DEVNULL).stdout.strip()
        return float(out) if out not in ('', 'N/A') else 0.0
    except (ValueError, Exception):
        return 0.0


def _frame_row(frame):
    """luma, saturation, red share and the 16-bin grey histogram of one tiny frame."""
    lu, sa, rd = color_stats(frame)
    hist = np.histogram(frame.mean(axis=2), bins=16, range=(0, 256))[0].astype(np.float32)
    return lu, sa, rd, hist


def _profile(rows):
    """rows: [(t, luma, sat, red, hist16)] in any order -> ScreenProfile (hist diffs in time order)."""
    rows = sorted(rows, key=lambda r: r[0])
    if len(rows) < 3:
        return None
    hd, prev = [], None
    for r in rows:
        hd.append(0.0 if prev is None else float(np.abs(r[4] - prev).sum() / (2 * max(1.0, r[4].sum()))))
        prev = r[4]
    return ScreenProfile([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], hd)


PART_VERSION = 2
CHUNK_MIN, CHUNK_MAX = 60.0, 300.0     # dense scan: the video is cut into chunks decoded in parallel


def chunk_seconds(duration, workers):
    """~4 chunks per worker keeps every decoder busy until the end; 60..300 s keeps seek pre-roll small."""
    return float(min(CHUNK_MAX, max(CHUNK_MIN, round(duration / (4 * max(1, workers))))))


def _load_part(path, key):
    try:
        d = np.load(path, allow_pickle=False)
        if int(d['v']) != PART_VERSION or str(d['key']) != key:
            return None, set()
        rows = [(float(t), float(lu), float(sa), float(rd), h.astype(np.float32))
                for t, lu, sa, rd, h in zip(d['t'], d['luma'], d['sat'], d['red'], d['hist'])]
        return rows, set(int(x) for x in d['done'])
    except Exception:
        return None, set()


def _save_part(path, key, rows, done):
    from pathlib import Path
    path = Path(path)
    tmp = path.with_name(path.stem + '.tmp.npz')
    with open(tmp, 'wb') as f:
        np.savez(f, v=PART_VERSION, key=key, t=np.asarray([r[0] for r in rows], np.float64),
                 luma=np.asarray([r[1] for r in rows], np.float32), sat=np.asarray([r[2] for r in rows], np.float32),
                 red=np.asarray([r[3] for r in rows], np.float32),
                 hist=np.asarray([r[4] for r in rows], np.float32).reshape(-1, 16), done=np.asarray(sorted(done)))
    os.replace(tmp, path)


def workers_for(hw):
    """Parallel decoders: one ffmpeg uses ~1.5 cores for 1080p60 AV1, so several chunks at once fill the CPU;
    NVDEC gets two sessions. CLIPFORGE_SCAN_WORKERS overrides."""
    env = os.getenv('CLIPFORGE_SCAN_WORKERS')
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return 2 if hw else max(1, min(6, (os.cpu_count() or 2)))


def scan_screen(source, duration=None, cancel=None, progress=None, part_path=None, width=64, hw=None):
    """Whole-video screen timeline -> ScreenProfile (or None).

    * Keyframes every <= 8 s (measured first from packet flags, no decoding): one keyframe-only pass; non-key
      packets are dropped before the decoder (-discard nokey).
    * Sparse keyframes (typical for long YouTube AV1/VP9): 1 frame/s on a fixed time grid, the video cut into
      chunks decoded in parallel - on NVDEC when it can decode this file, else on all CPU cores. Frames come from the
      same grid with the same area scaling, so the result equals a single sequential pass.
    * progress(frac, info) gets real progress; finished chunks are saved to `part_path`, so a crash, pause or
      cancel continues where it stopped. A stalled decoder is killed (procutil.Watchdog) with a clear error."""
    from . import accel
    rate = keyframe_rate(source)
    t0 = _start_time(source)
    if duration and (rate is None or rate >= KEY_DENSE):
        rows = _key_pass(source, width, t0, duration, cancel, progress)
        if len(rows) >= max(3.0, duration / 8.0):
            scan_screen.last_info = {'mode': 'key', 'hw': False, 'keyframe_rate': rate}
            return _profile(rows)
        log.info('keyframe scan found %d frames for %.0f s; scanning 1 frame/s', len(rows), duration)
    if not duration:
        rows = _key_pass(source, width, t0, None, cancel, progress)
        scan_screen.last_info = {'mode': 'key', 'hw': False, 'keyframe_rate': rate}
        return _profile(rows)
    if hw is None:
        hw = accel.nvdec_ok(source)
    rows = _dense(source, duration, width, t0, hw, cancel, progress, part_path, rate)
    if part_path:
        try:
            os.remove(part_path)
        except OSError:
            pass
    return _profile(rows)


scan_screen.last_info = None


def _dense(source, duration, width, t0, hw, cancel, progress, part_path, rate):
    import threading
    from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
    from .procutil import JobCancelled, StallError
    size = chunk_seconds(duration, workers_for(hw))
    n = max(1, int(np.ceil(duration / size)))
    # chunk borders sit half-way between grid points, so every grid frame is decided inside exactly one chunk
    bounds = [0.0] + [k * size + 0.5 for k in range(1, n)] + [duration + 1.0]
    key = f'fps1:{width}:{n}:{round(duration, 1)}'
    rows, done = (_load_part(part_path, key) if part_path and os.path.exists(part_path) else (None, set()))
    rows = rows or []
    todo = [i for i in range(n) if i not in done]
    info = {'mode': 'fps1', 'hw': bool(hw), 'keyframe_rate': rate, 'workers': workers_for(hw), 'chunks': n,
            'resumed_chunks': len(done)}
    log.info('screen scan: %s', info)
    lock = threading.Lock()
    secs = {i: 0.0 for i in todo}
    base = sum(min(bounds[i + 1], duration) - bounds[i] for i in done)
    stop = threading.Event()

    def report():
        if progress:
            progress(min(1.0, (base + sum(secs.values())) / duration), info)

    def run(i, use_hw):
        out = []

        def seen(t):
            secs[i] = max(0.0, min(bounds[i + 1], duration) - bounds[i]) if t is None else max(0.0, t - bounds[i])
        for t, row in _chunk(source, bounds[i], bounds[i + 1], width, t0, use_hw,
                             lambda: stop.is_set() or (cancel is not None and cancel()), seen):
            out.append((t,) + row)
        return out

    def run_safe(i):
        try:
            return run(i, hw)
        except RuntimeError as exc:
            if not hw or isinstance(exc, StallError) or stop.is_set():
                raise
            log.warning('NVDEC chunk %d failed (%s); decoding it on the CPU', i, str(exc)[:200])
            info['hw_failed'] = True
            return run(i, False)
    if todo:
        with ThreadPoolExecutor(max_workers=info['workers'], thread_name_prefix='screen') as ex:
            futs = {ex.submit(run_safe, i): i for i in todo}
            try:
                pending = set(futs)
                while pending:
                    fin, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    if cancel and cancel():
                        raise JobCancelled()
                    for f in fin:
                        i = futs[f]
                        part = f.result()
                        with lock:
                            rows += part
                            done.add(i)
                            secs.pop(i, None)
                            base += min(bounds[i + 1], duration) - bounds[i]
                        if part_path:
                            _save_part(part_path, key, rows, done)
                    report()
            except BaseException:
                stop.set()
                if part_path and rows:
                    try:
                        _save_part(part_path, key, rows, done)
                    except Exception:
                        pass
                raise
    scan_screen.last_info = info
    return rows


def _chunk(source, a, b, width, t0, hw, cancel, seen):
    """Decode [a, b) of the 1 frame/s grid -> yields (t, (luma, sat, red, hist16))."""
    import re
    import subprocess
    import threading
    from . import accel
    from .procutil import JobCancelled, Watchdog, _flags, kill_tree
    h = 36
    pre = ['-nostdin', '-hide_banner']
    if hw:
        pre += ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
    if a > 0:
        pre += ['-ss', f'{a:.3f}']
    down = f'hwdownload,format={accel.download_format(source)},' if hw else ''
    vf = f'fps=fps=1:start_time={t0:.6f},{down}scale={width}:{h}:flags=area,showinfo'
    cmd = ['ffmpeg'] + pre + ['-t', f'{b - a:.3f}', '-copyts', '-i', str(source), '-an', '-sn', '-dn', '-vf', vf,
                              '-fps_mode', 'passthrough', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
    size = width * h * 3
    times, tail = [], []
    rx = re.compile(r'pts_time:\s*([-\d.]+)')
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=_flags())
    dog = Watchdog(proc, f'Анализ экрана {a:.0f}-{b:.0f} с' + (' (NVDEC)' if hw else ''))

    def read_err():
        for raw in proc.stderr:
            line = raw.decode('utf-8', 'replace')
            m = rx.search(line)
            if m:
                times.append(float(m.group(1)) - t0)
                dog.touch()
            else:
                tail.append(line)
                del tail[:-20]
    reader = threading.Thread(target=read_err, daemon=True)
    reader.start()
    frames, k = [], 0
    try:
        while True:
            if cancel():
                raise JobCancelled()
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            dog.touch()
            frames.append(np.frombuffer(buf, np.uint8).reshape(h, width, 3))
            while frames and len(times) > k:
                t = times[k]
                k += 1
                fr = frames.pop(0)
                seen(t)
                if a <= t < b:                       # each grid time belongs to exactly one chunk
                    yield t, _frame_row(fr)
        proc.wait(timeout=60)
        reader.join(timeout=10)
        while frames and len(times) > k:
            t = times[k]
            k += 1
            fr = frames.pop(0)
            if a <= t < b:
                yield t, _frame_row(fr)
        seen(None)
    finally:
        dog.stop()
        kill_tree(proc)
    dog.check()
    if proc.returncode and k == 0:
        raise RuntimeError('ffmpeg: ' + ''.join(tail)[-300:])


def _key_pass(source, width, t0, duration, cancel, progress):
    """Keyframes only, one pass (cheap)."""
    rows = []
    for t, row in _chunk_key(source, width, t0, cancel):
        rows.append((t,) + row)
        if progress and duration:
            progress(min(1.0, t / duration), {'mode': 'key', 'hw': False})
    return rows


def _chunk_key(source, width, t0, cancel):
    import re
    import subprocess
    import threading
    from .procutil import JobCancelled, Watchdog, _flags, kill_tree
    h = 36
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-discard', 'nokey', '-skip_frame', 'nokey', '-copyts', '-i',
           str(source), '-an', '-sn', '-dn', '-vf', f'scale={width}:{h}:flags=area,showinfo', '-fps_mode',
           'passthrough', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
    size = width * h * 3
    times = []
    rx = re.compile(r'pts_time:\s*([-\d.]+)')
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=_flags())
    dog = Watchdog(proc, 'Анализ экрана (ключевые кадры)')

    def read_err():
        for raw in proc.stderr:
            m = rx.search(raw.decode('utf-8', 'replace'))
            if m:
                times.append(float(m.group(1)) - t0)
                dog.touch()
    reader = threading.Thread(target=read_err, daemon=True)
    reader.start()
    frames, k = [], 0
    try:
        while True:
            if cancel and cancel():
                raise JobCancelled()
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            dog.touch()
            frames.append(np.frombuffer(buf, np.uint8).reshape(h, width, 3))
            while frames and len(times) > k:
                k += 1
                yield times[k - 1], _frame_row(frames.pop(0))
        proc.wait(timeout=60)
        reader.join(timeout=10)
        while frames and len(times) > k:
            k += 1
            yield times[k - 1], _frame_row(frames.pop(0))
    finally:
        dog.stop()
        kill_tree(proc)
    dog.check()


def scan_keyframes(source, duration=None, cancel=None, timeout=None, width=64):
    """Compatibility wrapper (v5.2 name): the whole-video screen scan."""
    return scan_screen(source, duration, cancel=cancel, width=width)


def screen_to_arrays(sp):
    return {'t': sp.t, 'luma': sp.luma, 'sat': sp.sat, 'red': sp.red, 'hist': sp.hist}


def screen_from_arrays(d):
    if not d or 't' not in d:
        return None
    return ScreenProfile(d['t'], d['luma'], d['sat'], d['red'], d['hist'])
