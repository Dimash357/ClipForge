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


# side-crop budgets of the gameplay panel in the cam layout (fraction of the source width that may be cut off)
CAM_GAME_FIT = {'full': 0.0, 'balanced': 0.16, 'fill': 0.30}
CAM_PANEL_MIN, CAM_PANEL_MAX = 0.22, 0.36     # webcam panel height as a share of the output height


def cam_rect_px(cam, src_w, src_h):
    """Webcam overlay rectangle in source pixels (even), minus the overlay's own border line."""
    cx, cy = ev(cam['x'] * src_w), ev(cam['y'] * src_h)
    cx, cy = min(cx, src_w - 4), min(cy, src_h - 4)
    cw, ch = min(ev(cam['w'] * src_w), src_w - cx), min(ev(cam['h'] * src_h), src_h - cy)
    # the detected rectangle can be a few px off the real overlay: trim enough to never show its frame line
    inset = max(8, int(cw * 0.025)) // 2 * 2
    cx, cy = cx + inset, cy + inset
    cw, ch = max(2, (cw - 2 * inset) // 2 * 2), max(2, (ch - 2 * inset) // 2 * 2)
    return cx, cy, cw, ch


def cam_geometry(W, H, src_w, src_h, cam, fit='balanced', panel_h=None, game_anchor=0.3):
    """Pure geometry of the Gameplay + Facecam layout (no ffmpeg strings) -> dict.

    * webcam panel: W x Hc, height follows the overlay's own aspect (clamped) -> least upscaling, sharp image;
    * gameplay panel: W x Hg. The game keeps its aspect ratio; it is scaled up only as far as the side-crop budget
      allows (`fit`), the rest of the panel is background. The crop is pushed toward the webcam's side so the cut
      columns are the ones holding the duplicate webcam;
    * hide: the part of the webcam overlay that is still visible after cropping (source px) or None.
    """
    cx, cy, cw, ch = cam_rect_px(cam, src_w, src_h)
    nat_h = W * ch / cw
    Hc = ev(panel_h) if panel_h else ev(max(CAM_PANEL_MIN * H, min(CAM_PANEL_MAX * H, nat_h)))
    Hg = H - Hc
    budget = CAM_GAME_FIT.get(fit, CAM_GAME_FIT['balanced'])
    # margin around the overlay (its border/shadow must not survive)
    m = ev(0.01 * src_w)
    mx0, my0 = max(0, cx - m), max(0, cy - m)
    mx1, my1 = min(src_w, cx + cw + m), min(src_h, cy + ch + m)
    vis_w, vis_h = src_w, src_h
    if src_w / src_h >= W / Hg:
        # width-limited: zoom so that the panel height is used better, but never crop more than the budget
        max_zoom_fill = (src_w / src_h) / (W / Hg)          # zoom at which the game would exactly fill the panel
        zoom = min(1.0 / (1.0 - budget), max_zoom_fill)
        vis_w = ev(src_w / zoom)
        vis_w = min(vis_w, src_w // 2 * 2)
    cut = src_w - vis_w
    cam_right = (cx + cw / 2) >= src_w / 2
    cam_cols = (src_w - mx0) if cam_right else mx1           # columns from the cam side edge to its inner border
    side = min(cut, cam_cols)                                # spend the crop on the webcam's side first
    rest = cut - side
    if cam_right:
        x0 = rest // 2
    else:
        x0 = side + rest // 2
    x0 = max(0, min(src_w - vis_w, x0 // 2 * 2))
    out_w = W
    out_h = ev(vis_h * W / vis_w)
    if out_h > Hg:                                           # tall source: fit by height instead
        out_h = Hg
        out_w = ev(vis_w * Hg / vis_h)
    # visible remainder of the webcam overlay
    hx0, hx1 = max(mx0, x0), min(mx1, x0 + vis_w)
    hide = None
    if hx1 - hx0 >= 4:
        hide = (mx0, my0, (mx1 - mx0) // 2 * 2, (my1 - my0) // 2 * 2)
    free = Hg - out_h
    return {'Hc': Hc, 'Hg': Hg, 'cam': (cx, cy, cw, ch), 'crop': (x0, 0, vis_w, vis_h), 'game': (out_w, out_h),
            'game_y': int(free * game_anchor) // 2 * 2, 'hide': hide, 'cam_right': cam_right, 'cam_top': (cy + ch / 2) < src_h / 2,
            'crop_frac': round(cut / src_w, 3)}


def _mirror_fill(src_w, src_h, rect, cam_right, cam_top, inp, out, tag='mf'):
    """Rebuild the webcam area from the neighbouring gameplay instead of blurring it.

    Two mirrored copies are taken from just outside the overlay's inner edges (a horizontal mirror continues the
    picture across the inner vertical edge, a vertical mirror across the inner horizontal edge) and blended with a
    distance weight, so both seams are continuous and the patch keeps real game texture. Returns None when the
    neighbourhood is too small (caller falls back).
    """
    x, y, w, h = rect
    # horizontal neighbour (inner vertical edge)
    hx = x - w if cam_right else x + w
    v_y = y + h if cam_top else y - h
    have_h = 0 <= hx and hx + w <= src_w
    have_v = 0 <= v_y and v_y + h <= src_h
    if not (have_h or have_v):
        return None
    parts = [f'[{inp}]split={1 + have_h + have_v}[{tag}base]' + (f'[{tag}hs]' if have_h else '') +
             (f'[{tag}vs]' if have_v else '')]
    if have_h:
        parts.append(f'[{tag}hs]crop={w}:{h}:{hx}:{y},hflip[{tag}h]')
    if have_v:
        parts.append(f'[{tag}vs]crop={w}:{h}:{x}:{v_y},vflip[{tag}v]')
    if have_h and have_v:
        dx = 'X' if cam_right else '(W-1-X)'
        dy = '(H-1-Y)' if cam_top else 'Y'
        wexpr = f'({dy}+0.5)/({dx}+{dy}+1)'
        parts.append(f"[{tag}h][{tag}v]blend=all_expr='A*{wexpr}+B*(1-{wexpr})'[mask]")
    else:
        parts.append(f'[{tag}{"h" if have_h else "v"}]null[mask]')
    parts.append(f'[{tag}base][mask]overlay={x}:{y}[{out}]')
    return ';'.join(parts)


def _blur_fill(rect, inp, out):
    """Legacy v5 behaviour: pixelate + darken the overlay area."""
    mx, my, mw, mh = rect
    return ';'.join([f'[{inp}]split=2[gA][gB]',
                     f'[gB]crop={mw}:{mh}:{mx}:{my},scale={ev(mw / 60)}:{ev(mh / 60)},scale={mw}:{mh}:flags=bilinear,'
                     f'eq=brightness=-0.10[mask]',
                     f'[gA][mask]overlay={mx}:{my}[{out}]'])


def cam_layout(W, H, src_w, src_h, inp, out, cam, bg='blur', cam_position='bottom', hide_cam=True, face_rel=(0.5, 0.5),
               fit='balanced', fill='mirror'):
    """Gameplay panel (as complete as possible, aspect kept, duplicate webcam removed) + a sharp webcam panel."""
    g = cam_geometry(W, H, src_w, src_h, cam, fit)
    Hc, Hg = g['Hc'], g['Hg']
    cx, cy, cw, ch = g['cam']
    # camera panel: crop the overlay, cover W x Hc around the face. Lanczos + light luma sharpening keeps the
    # upscaled webcam crisp (no blur is ever applied to it).
    cam_chain = (f'[cam]crop={cw}:{ch}:{cx}:{cy},scale={W}:{Hc}:force_original_aspect_ratio=increase:flags=lanczos,'
                 f"crop={W}:{Hc}:x='min(max(0\\,{_f(face_rel[0])}*iw-ow/2)\\,iw-ow)':"
                 f"y='min(max(0\\,{_f(face_rel[1])}*ih-oh/2)\\,ih-oh)',unsharp=5:5:0.6:5:5:0.0,setsar=1[campanel]")
    parts = [f'[{inp}]split=2[gsrc0][cam]']
    parts += _game_panel(W, src_w, src_h, g, bg, hide_cam, fill)
    parts.append(cam_chain)
    stack = '[gpanel][campanel]vstack=inputs=2' if cam_position == 'bottom' else '[campanel][gpanel]vstack=inputs=2'
    parts.append(f'{stack},setsar=1[{out}]')
    if cam_position == 'bottom':
        margin = Hc + 0.012 * H                      # text sits just above the camera panel, over the background band
    else:
        margin = H - (Hc + 0.012 * H + 0.07 * H)     # just below the camera panel
    return ';'.join(parts), int(max(0.20 * H, min(0.45 * H, margin)))


def _game_panel(W, src_w, src_h, g, bg, hide_cam, fill):
    """[gsrc0] -> [gpanel]: duplicate webcam removed, game scaled/cropped per geometry `g`, background fill."""
    Hg = g['Hg']
    parts = []
    game_src = 'gsrc'
    if hide_cam and g['hide']:
        frag = None
        if fill != 'blur':
            frag = _mirror_fill(src_w, src_h, g['hide'], g['cam_right'], g['cam_top'], 'gsrc0', game_src)
        parts.append(frag or _blur_fill(g['hide'], 'gsrc0', game_src))
    else:
        parts.append(f'[gsrc0]null[{game_src}]')
    x0, y0, vw, vh = g['crop']
    gw, gh = g['game']
    crop = f'crop={vw}:{vh}:{x0}:{y0},' if (vw, vh) != (src_w, src_h) else ''
    fg = f'{crop}scale={gw}:{gh}:flags=bicubic'
    gx, gy = (W - gw) // 2, g['game_y']
    if gw >= W and gh >= Hg:
        parts.append(f'[{game_src}]{fg},setsar=1[gpanel]')
    elif bg == 'solid':
        parts.append(f'[{game_src}]{fg},pad={W}:{Hg}:{gx}:{gy}:color={BG_SOLID},setsar=1[gpanel]')
    else:
        parts.append(f'[{game_src}]split=2[gbg][gfg]')
        parts.append(_blur_bg(W, Hg, 'gbg', 'gb'))
        parts.append(f'[gfg]{fg}[gf]')
        parts.append(f'[gb][gf]overlay={gx}:{gy},setsar=1[gpanel]')
    return parts


REACTION_CAM_SHARE = 0.40       # webcam panel height in Reaction Priority
REACTION_ZOOM = 1.12            # small extra zoom so the face can be centred / eyes placed on the upper third


def reaction_cam_crop(W, Hc, cw, ch, face_x, face_y, face_h, zoom=REACTION_ZOOM):
    """Scale + crop of the webcam (cw x ch px) into W x Hc: face centred horizontally, eyes on the upper third.
    face_x/face_y: face centre relative to the crop, face_h: face height relative to the crop height."""
    s = max(W / cw, Hc / ch) * zoom
    sw, sh = ev(cw * s), ev(ch * s)
    eye_y = max(0.0, face_y - 0.12 * face_h)           # eyes sit ~12% of the face height above its centre
    x = int(min(max(0.0, face_x * sw - W / 2), sw - W)) // 2 * 2
    y = int(min(max(0.0, eye_y * sh - Hc / 3), sh - Hc)) // 2 * 2
    return sw, sh, x, y


def reaction_layout(W, H, src_w, src_h, inp, out, cam, bg='blur', hide_cam=True, fit='balanced', fill='mirror',
                    face_rel=None):
    """Reaction Priority: webcam on top (~40%), face centred with the eyes on the upper third; game below (~60%)."""
    Hc = ev(REACTION_CAM_SHARE * H)
    g = cam_geometry(W, H, src_w, src_h, cam, fit, panel_h=Hc, game_anchor=0.22)
    cx, cy, cw, ch = g['cam']
    fr = face_rel or cam.get('face_rel') or (0.5, 0.5)
    # face position relative to the (border-trimmed) crop rather than to the detected overlay rectangle
    fx = (cam['x'] * src_w + fr[0] * cam['w'] * src_w - cx) / cw
    fy = (cam['y'] * src_h + fr[1] * cam['h'] * src_h - cy) / ch
    fh = float(cam.get('face_h') or 0.45) * cam['h'] * src_h / ch
    sw, sh, x, y = reaction_cam_crop(W, Hc, cw, ch, min(1.0, max(0.0, fx)), min(1.0, max(0.0, fy)), fh)
    parts = [f'[{inp}]split=2[gsrc0][cam]',
             f'[cam]crop={cw}:{ch}:{cx}:{cy},scale={sw}:{sh}:flags=lanczos,crop={W}:{Hc}:{x}:{y},'
             f'unsharp=5:5:0.6:5:5:0.0,setsar=1[campanel]']
    parts += _game_panel(W, src_w, src_h, g, bg, hide_cam, fill)
    parts.append(f'[campanel][gpanel]vstack=inputs=2,setsar=1[{out}]')
    # captions at the seam: over the lower (shoulder) part of the webcam and the gap above the game, never on the
    # eyes/face and not over the game picture
    seam_gap = g['game_y']
    bottom_of_text = Hc + max(0.03 * H, min(seam_gap - 0.01 * H, 0.05 * H))
    return ';'.join(parts), int(H - bottom_of_text)


def action_crop_margin(H):
    return int(0.24 * H)


# ------------------------------------------------------------------ whole-clip graph
def seg_layout(seg, ctx, inp, out):
    """Pick builder by scene type. ctx: dict with W,H,src_w,src_h,bg,zoom,cam,cam_position,hide_cam,tracking."""
    W, H, sw, sh = ctx['W'], ctx['H'], ctx['src_w'], ctx['src_h']
    typ = seg['type']
    if typ == 'game_cam' and ctx.get('cam'):
        return cam_layout(W, H, sw, sh, inp, out, ctx['cam'], bg=ctx['bg'], cam_position=ctx.get('cam_position', 'bottom'),
                          hide_cam=ctx.get('hide_cam', True), face_rel=seg.get('face_rel', (0.5, 0.5)),
                          fit=ctx.get('cam_fit', 'balanced'), fill=ctx.get('cam_fill', 'mirror'))
    if typ == 'reaction_cam' and ctx.get('cam'):
        return reaction_layout(W, H, sw, sh, inp, out, ctx['cam'], bg=ctx['bg'], hide_cam=ctx.get('hide_cam', True),
                               fit=ctx.get('cam_fit', 'balanced'), fill=ctx.get('cam_fill', 'mirror'),
                               face_rel=seg.get('face_rel'))
    if typ == 'action':
        chain, _ = crop_layout(W, H, sw, sh, inp, out, seg.get('path'), tol=ctx.get('path_tol', 3.0))
        return chain, action_crop_margin(H)
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
