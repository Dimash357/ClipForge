"""AI Auto: pick the best existing edit mode for one clip from the analysis that is already done for it.

Inputs (no extra decoding): the clip's cached scene analysis (faces, webcam overlay, motion, burned-in text),
the whole-video webcam scan, the source aspect ratio and the clip's highlight scores (reaction / audio / action).
Output: (mode, reason, details). Every decision is a readable rule, so the UI can say *why*.
"""
import numpy as np

from . import scenes as scn

LABELS = {'reaction': 'Reaction Priority', 'gameplay_facecam': 'Gameplay Priority', 'smart': 'Smart Reframe',
          'director': 'Auto Scene Director', 'fullframe': 'Без обрезки', 'gameplay': 'Gameplay', 'center': 'Центр'}
TYPE_RU = {'game_cam': 'игра + вебка', 'talking': 'лицо крупно', 'gameplay': 'игра', 'screen': 'экран/интерфейс'}


def scene_shares(samples, cam):
    """Share of samples per scene type (the same classifier the Director uses)."""
    counts = {}
    for s in samples:
        lab, _ = scn.classify_sample(s, cam)
        counts[lab] = counts.get(lab, 0) + 1
    n = max(1, len(samples))
    return {k: v / n for k, v in counts.items()}


def action_concentration(samples, crop_frac):
    """How often the main on-screen action fits inside one 9:16 crop window around its median position (0..1)."""
    acts = [s['act'] for s in samples if s.get('act') and s['act']['share'] < 0.6]
    if len(acts) < max(4, len(samples) // 4):
        return 0.0, len(acts)
    med = float(np.median([a['cx'] for a in acts]))
    # an action point counts only if its centre stays near the median AND the moving pixels are not spread over
    # the whole width (centroid of motion everywhere is also "in the middle"); unknown spread = not concentrated
    inside = sum(1 for a in acts if abs(a['cx'] - med) <= crop_frac * 0.4 and a.get('sx', 1.0) <= crop_frac * 0.45)
    return inside / len(acts), len(acts)


def choose(analysis, src_w, src_h, scores=None, source_cam=None, crop_frac=None):
    """-> (mode, reason, details)."""
    scores = scores or {}
    details = {'aspect': round(src_w / max(1, src_h), 2)}
    if src_w / max(1, src_h) < 0.8:
        return 'smart', 'исходник уже вертикальный — кадр почти не обрезается, Smart Reframe держит лицо в центре', details
    samples = (analysis or {}).get('samples') or []
    if not samples:
        return 'smart', 'анализ кадров недоступен — безопасный режим Smart Reframe', details
    cam = (analysis or {}).get('cam') or source_cam
    details['cam'] = cam.get('corner') if cam else None
    shares = scene_shares(samples, cam)
    details['shares'] = {k: round(v, 2) for k, v in sorted(shares.items(), key=lambda kv: -kv[1])}
    hs = scn.hardsub_summary(samples)
    if hs.get('present'):
        return 'fullframe', 'в исходнике вшитые титры — кадр показан целиком, чтобы текст не обрезался', details
    top, top_share = max(shares.items(), key=lambda kv: kv[1])
    mixed = [t for t, v in shares.items() if v >= 0.2]
    if len(mixed) >= 2 and top_share < 0.7:
        parts = ', '.join(f'{TYPE_RU.get(t, t)} {round(shares[t] * 100)}%' for t in sorted(mixed, key=lambda t: -shares[t]))
        return 'director', f'сцены внутри клипа меняются ({parts}) — раскладка переключается по сценам', details
    if top == 'game_cam' and cam:
        reaction = float(scores.get('reaction') or 0)
        audio = float(scores.get('audio') or 0)
        action = float(scores.get('action') or 0)
        emotional = max(reaction, audio)
        details.update(reaction=reaction, audio=audio, action=action)
        if emotional >= 60 and not (action >= 70 and action > emotional + 5):
            return ('reaction', f'вебка в кадре {round(top_share * 100)}% времени, главное в клипе — реакция стримера '
                                f'(реакция {round(reaction)}, звук {round(audio)}) — лицо сверху крупно, игра снизу', details)
        why = (f'активный геймплей (динамика {round(action)})' if action >= 60 else 'важнее происходящее в игре')
        return 'gameplay_facecam', (f'вебка в кадре {round(top_share * 100)}% времени, {why} — максимум игры сверху, '
                                    'вебка отдельной панелью внизу, без дубля'), details
    if top == 'talking':
        return 'smart', 'в кадре в основном лицо — Smart Reframe плавно ведёт кадр за ним', details
    if top == 'screen':
        return 'fullframe', 'в кадре интерфейс/текст — показан целиком, чтобы всё читалось', details
    # gameplay without a webcam
    conc, n = action_concentration(samples, crop_frac or 0.32)
    details['action_concentration'] = round(conc, 2)
    if conc >= 0.75:
        return 'gameplay_facecam', ('вебки нет, действие держится в одной области кадра — игра на весь вертикальный '
                                    'кадр со слежением за действием'), details
    return 'gameplay', ('вебки нет, действие разбросано по экрану — игра показана целиком на размытом фоне, '
                        'HUD и края не обрезаются'), details
