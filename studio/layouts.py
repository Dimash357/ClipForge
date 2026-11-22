"""ffmpeg filtergraph builders for every vertical layout.

All layouts render directly at the export size (W x H) -> no double scaling, no stretching.
Every builder returns (filter_fragment, caption_margin_v_px).
"""
from . import scenes as scn

BG_SOLID = '0x101018'


def ev(x):
    """Even positive int (yuv420 requirement)."""
    return max(2, int(round(x / 2.0)) * 2)


def _f(x):
    return f'{x:.2f}'.rstrip('0').rstrip('.')


# ------------------------------------------------------------------ helpers
def _blur_bg(W, H, inp, out):
    """Cheap strong blur: scale to cover, crop, shrink, soften, enlarge. Much faster than boxblur at full size."""
    sw, sh = ev(W / 10), ev(H / 10)
    return (f'[{inp}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},scale={sw}:{sh},'
            f'boxblur=2:2,scale={W}:{H}:flags=bilinear,eq=brightness=-0.07:saturation=0.9[{out}]')


def fit_layout(W, H, src_w, src_h, inp, out, bg='blur', zoom=1.0, tag='f'):
    """Whole picture visible (HUD safe) on a blurred/solid background; optional mild zoom ('focus')."""
    if src_w / src_h >= W / H:
        fit_w, fit_h = W, src_h * W / src_w
    else:
        fit_w, fit_h = src_w * H / src_h, H
    vw, vh = fit_w * zoom, fit_h * zoom
    if zoom > 1.0:
        sw, sh = ev(vw), ev(vh)
        cw, ch = min(sw, W), min(sh, H)
        fg = f'scale={sw}:{sh},crop={cw}:{ch}'
        vw, vh = cw, ch
    else:
        fg = f'scale={W}:{H}:force_original_aspect_ratio=decrease'
    if bg == 'solid':
        chain = f'[{inp}]{fg},pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={BG_SOLID},setsar=1[{out}]'
    else:
        chain = (f'[{inp}]split=2[{tag}bg][{tag}fg];' + _blur_bg(W, H, f'{tag}bg', f'{tag}b') + ';' +
                 f'[{tag}fg]{fg}[{tag}f];[{tag}b][{tag}f]overlay=(W-w)/2:(H-h)/2,setsar=1[{out}]')
    video_bottom = (H + min(vh, H)) / 2
    margin = H - (video_bottom + 0.03 * H + 0.06 * H)
    return chain, int(max(0.20 * H, min(0.40 * H, margin)))


def crop_dims(src_w, src_h, W, H):
    cw = ev(min(src_w, src_h * W / H))
    ch = ev(min(src_h, cw * H / W))
    cw = min(cw, src_w // 2 * 2)
    return cw, ch


def _lerp_expr(points, idx):
    """Piecewise-linear expression in `t` through (t, value) points; idx picks x(1) or y(2)."""
    pts = [(p[0], p[idx]) for p in points]
    if len(pts) == 1 or all(abs(v - pts[0][1]) < 0.5 for _, v in pts):
        return _f(pts[0][1])
    expr = _f(pts[-1][1])
    for k in range(len(pts) - 2, -1, -1):
        t0, v0 = pts[k]
        t1, v1 = pts[k + 1]
        if t1 - t0 < 1e-3:
            continue
        expr = f'if(lt(t\\,{_f(t1)})\\,{_f(v0)}+({_f(v1 - v0)})*(t-{_f(t0)})/{_f(t1 - t0)}\\,{expr})'
    return expr


def crop_layout(W, H, src_w, src_h, inp, out, path=None, tol=3.0):
    """9:16 crop window following `path` [(t, x, y)] (None -> centred)."""
    cw, ch = crop_dims(src_w, src_h, W, H)
    if not path:
        x, y = (src_w - cw) // 2, (src_h - ch) // 2
        xe, ye = str(x), str(y)
    else:
        path = scn.simplify_path(path, tol=tol)
        xe, ye = _lerp_expr(path, 1), _lerp_expr(path, 2)
        # keep the window inside the frame whatever interpolation does
        xe = f'min(max(0\\,{xe})\\,{src_w - cw})'
        ye = f'min(max(0\\,{ye})\\,{src_h - ch})'
    return f"[{inp}]crop={cw}:{ch}:x='{xe}':y='{ye}',scale={W}:{H}:flags=bicubic,setsar=1[{out}]", int(0.22 * H)


def cam_layout(W, H, src_w, src_h, inp, out, cam, bg='blur', cam_position='bottom', hide_cam=True, face_rel=(0.5, 0.5)):
    """Gameplay on top (fit, blurred fill, cam area softened to avoid a duplicate face), facecam panel below."""
    cam_px_w, cam_px_h = cam['w'] * src_w, cam['h'] * src_h
    cx, cy = ev(cam['x'] * src_w), ev(cam['y'] * src_h)
    cw, ch = min(ev(cam_px_w), src_w - cx), min(ev(cam_px_h), src_h - cy)
    inset = max(4, int(cw * 0.012)) // 2 * 2     # drop the overlay's own border line
    cx, cy = cx + inset, cy + inset
    cw, ch = max(2, (cw - 2 * inset) // 2 * 2), max(2, (ch - 2 * inset) // 2 * 2)
    nat_h = W * ch / cw
    Hc = ev(max(0.22 * H, min(0.40 * H, nat_h)))
    Hg = H - Hc
    # camera panel: cover W x Hc, crop around the face
    cam_chain = (f'[cam]crop={cw}:{ch}:{cx}:{cy},scale={W}:{Hc}:force_original_aspect_ratio=increase,'
                 f"crop={W}:{Hc}:x='min(max(0\\,{_f(face_rel[0])}*iw-ow/2)\\,iw-ow)':"
                 f"y='min(max(0\\,{_f(face_rel[1])}*ih-oh/2)\\,ih-oh)',setsar=1[campanel]")
    game_src = 'gsrc'
    parts = [f'[{inp}]split=2[gsrc0][cam]']
    if hide_cam:
        m = ev(0.01 * src_w)
        mx, my = max(0, cx - m), max(0, cy - m)
        mw, mh = min(src_w - mx, cw + 2 * m) // 2 * 2, min(src_h - my, ch + 2 * m) // 2 * 2
        parts.append('[gsrc0]split=2[gA][gB]')
        parts.append(f'[gB]crop={mw}:{mh}:{mx}:{my},scale={ev(mw / 60)}:{ev(mh / 60)},scale={mw}:{mh}:flags=bilinear,eq=brightness=-0.10[mask]')
        parts.append(f'[gA][mask]overlay={mx}:{my}[{game_src}]')
    else:
        parts.append(f'[gsrc0]null[{game_src}]')
    fit_chain, _ = fit_layout(W, Hg, src_w, src_h, game_src, 'gpanel', bg=bg, tag='g')
    parts.append(fit_chain)
    parts.append(cam_chain)
    stack = '[gpanel][campanel]vstack=inputs=2' if cam_position == 'bottom' else '[campanel][gpanel]vstack=inputs=2'
    parts.append(f'{stack},setsar=1[{out}]')
    if cam_position == 'bottom':
        margin = Hc + 0.012 * H                      # text sits just above the camera panel, over the blurred band
    else:
        margin = H - (Hc + 0.012 * H + 0.07 * H)     # just below the camera panel
    return ';'.join(parts), int(max(0.20 * H, min(0.45 * H, margin)))


# ------------------------------------------------------------------ whole-clip graph
def seg_layout(seg, ctx, inp, out):
    """Pick builder by scene type. ctx: dict with W,H,src_w,src_h,bg,zoom,cam,cam_position,hide_cam,tracking."""
    W, H, sw, sh = ctx['W'], ctx['H'], ctx['src_w'], ctx['src_h']
    typ = seg['type']
    if typ == 'game_cam' and ctx.get('cam'):
        return cam_layout(W, H, sw, sh, inp, out, ctx['cam'], bg=ctx['bg'], cam_position=ctx.get('cam_position', 'bottom'),
                          hide_cam=ctx.get('hide_cam', True), face_rel=seg.get('face_rel', (0.5, 0.5)))
    if typ in ('talking', 'smart', 'center'):
        return crop_layout(W, H, sw, sh, inp, out, seg.get('path'), tol=ctx.get('path_tol', 3.0))
    zoom = ctx.get('zoom', 1.0) if typ == 'gameplay' else 1.0     # 'fullframe' / 'screen' never zoom
    return fit_layout(W, H, sw, sh, inp, out, bg=ctx['bg'], zoom=zoom, tag='q')


def build_graph(segments, ctx, ass_name='captions.ass', transition='none', tdur=0.25, fps=None):
    """segments: [{start,end,type,...}] relative to clip start. Returns (graph, margin_fn)."""
    n = len(segments)
    margins = []
    parts = []
    use_x = transition == 'fade' and n > 1
    if n == 1:
        frag, mv = seg_layout(segments[0], ctx, 'v0in', 'vc')
        parts.append(f'[0:v]null[v0in]')
        parts.append(frag)
        margins.append(mv)
    else:
        parts.append(f'[0:v]split={n}' + ''.join(f'[s{i}]' for i in range(n)))
        for i, sg in enumerate(segments):
            a, b = sg['start'], sg['end']
            if use_x and i < n - 1:
                b += tdur
            parts.append(f'[s{i}]trim=start={_f(a)}:end={_f(b)},setpts=PTS-STARTPTS[t{i}]')
            frag, mv = seg_layout(sg, ctx, f't{i}', f'l{i}')
            parts.append(frag)
            margins.append(mv)
        if use_x:
            cur = 'l0'
            for i in range(1, n):
                off = segments[i]['start']  # previous scene was extended by tdur, fade starts at the cut
                parts.append(f'[{cur}][l{i}]xfade=transition=fade:duration={_f(tdur)}:offset={_f(off)}[x{i}]')
                cur = f'x{i}'
            parts.append(f'[{cur}]null[vc]')
        else:
            parts.append(''.join(f'[l{i}]' for i in range(n)) + f'concat=n={n}:v=1:a=0[vc]')
    tail = []
    if fps:
        tail.append(f'fps={fps}')
    tail.append(f'ass={ass_name}')
    tail.append('format=yuv420p')
    parts.append('[vc]' + ','.join(tail) + '[vout]')

    bounds = [(sg['start'], sg['end'], m) for sg, m in zip(segments, margins)]

    def margin_for(t):
        for a, b, m in bounds:
            if a <= t < b:
                return m
        return bounds[-1][2]

    return ';'.join(parts), margin_for
