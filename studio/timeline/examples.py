"""Reference timelines for the other kinds (used by tests and as templates for the coming generators)."""
from . import model as m
from .ids import stable_id
from .validate import ensure_valid


def cartoon(seed='demo'):
    """A 2-scene motion comic: 2 characters, per-scene images with a Ken Burns move, voices, music, captions."""
    tl_id = stable_id('tl', 'cartoon', seed)
    a = lambda *p: stable_id('as', tl_id, *p)          # noqa: E731
    fox_ref, owl_ref = a('ref', 'fox'), a('ref', 'owl')
    chars = [m.Character(id=stable_id('ch', tl_id, 'fox'), name='Лис', description='рыжий лис в синем шарфе',
                         voice='piper:ru_RU-dmitri-medium', refs=[fox_ref], seed=1234),
             m.Character(id=stable_id('ch', tl_id, 'owl'), name='Сова', description='серая сова в очках',
                         voice='piper:ru_RU-irina-medium', refs=[owl_ref], seed=5678)]
    gen = lambda prompt, s: m.Generation(provider='sdxl-local', model='sdxl-base-1.0', prompt=prompt, seed=s)  # noqa
    assets = [
        m.Asset(id=fox_ref, type='image', uri='gen:fox-ref.png', width=1024, height=1024,
                generation=gen('fox character sheet', 1234)),
        m.Asset(id=owl_ref, type='image', uri='gen:owl-ref.png', width=1024, height=1024,
                generation=gen('owl character sheet', 5678)),
        m.Asset(id=a('shot1'), type='image', uri='gen:shot1.png', width=1024, height=1820,
                generation=gen('fox meets owl in a night forest', 42)),
        m.Asset(id=a('shot2'), type='image', uri='gen:shot2.png', width=1024, height=1820,
                generation=gen('owl laughs on a branch', 43)),
        m.Asset(id=a('vo1'), type='audio', uri='gen:vo1.wav', duration=3.2),
        m.Asset(id=a('vo2'), type='audio', uri='gen:vo2.wav', duration=2.6),
        m.Asset(id=a('music'), type='audio', uri='file:library/music/calm_forest.mp3', duration=60.0),
    ]
    s1, s2 = stable_id('sc', tl_id, 1), stable_id('sc', tl_id, 2)
    scenes = [m.Scene(id=s1, start=0.0, duration=4.0, name='Встреча', status='generated',
                      characters=[chars[0].id, chars[1].id], generation=gen('fox meets owl in a night forest', 42)),
              m.Scene(id=s2, start=4.0, duration=4.0, name='Смех', status='draft', characters=[chars[1].id],
                      generation=gen('owl laughs on a branch', 43))]
    it = lambda *p: stable_id('it', tl_id, *p)          # noqa: E731
    ken_burns = [m.Animation(property='zoom', keyframes=[m.Keyframe(t=0.0, value=1.0),
                                                         m.Keyframe(t=4.25, value=1.12, easing='ease_in_out')])]
    v1 = m.ImageItem(id=it('img1'), start=0.0, duration=4.25, scene=s1, asset=a('shot1'), animations=ken_burns)
    v2 = m.ImageItem(id=it('img2'), start=4.0, duration=4.0, scene=s2, asset=a('shot2'))
    style = m.Style(id=stable_id('st', tl_id, 'stream'), font='Montserrat Black', size=104, bold=False,
                    uppercase=True, max_words=1, position='middle')
    tracks = [
        m.Track(id=stable_id('tr', tl_id, 'v'), type='video', items=[
            v1, v2, m.TransitionItem(id=it('fade'), start=4.0, duration=0.25, kind='fade', from_item=v1.id,
                                     to_item=v2.id)]),
        m.Track(id=stable_id('tr', tl_id, 'voice'), type='audio', items=[
            m.AudioItem(id=it('vo1'), start=0.4, duration=3.2, scene=s1, asset=a('vo1'), role='voice'),
            m.AudioItem(id=it('vo2'), start=4.6, duration=2.6, scene=s2, asset=a('vo2'), role='voice')]),
        m.Track(id=stable_id('tr', tl_id, 'music'), type='audio', items=[
            m.AudioItem(id=it('music'), start=0.0, duration=8.0, asset=a('music'), role='music', volume=0.35,
                        fade_in=0.5, fade_out=1.0, duck=True)]),
        m.Track(id=stable_id('tr', tl_id, 'cap'), type='caption', items=[
            m.CaptionItem(id=it('cap'), start=0.0, duration=8.0, style=style.id, words=[
                m.Word(t0=0.4, t1=0.9, text='Привет,', speaker=chars[0].id),
                m.Word(t0=0.9, t1=1.4, text='сова!', emphasis=True, speaker=chars[0].id),
                m.Word(t0=4.6, t1=5.4, text='Ухаха!', emphasis=True, speaker=chars[1].id)])]),
    ]
    return ensure_valid(m.Timeline(id=tl_id, kind='cartoon', duration=8.0, canvas=m.Canvas(fps=30), assets=assets,
                                   scenes=scenes, tracks=tracks, styles=[style], characters=chars,
                                   meta={'title': 'Лис и сова', 'idea': 'лис встречает смешливую сову'}))


def faceless(seed='demo'):
    """A 'facts' short: stock footage per fact, a voice-over, a big text hook, captions, music under the voice."""
    tl_id = stable_id('tl', 'faceless', seed)
    a = lambda *p: stable_id('as', tl_id, *p)          # noqa: E731
    it = lambda *p: stable_id('it', tl_id, *p)          # noqa: E731
    assets = [m.Asset(id=a('stock1'), type='video', uri='file:stock/space.mp4', duration=12.0, width=1080,
                      height=1920, has_audio=False),
              m.Asset(id=a('stock2'), type='video', uri='file:stock/ocean.mp4', duration=9.0, width=1920,
                      height=1080, has_audio=False),
              m.Asset(id=a('vo'), type='audio', uri='gen:voiceover.wav', duration=11.5),
              m.Asset(id=a('music'), type='audio', uri='file:library/music/pulse.mp3', duration=90.0)]
    s1, s2 = stable_id('sc', tl_id, 1), stable_id('sc', tl_id, 2)
    scenes = [m.Scene(id=s1, start=0.0, duration=6.0, name='Факт 1: космос', status='generated'),
              m.Scene(id=s2, start=6.0, duration=6.0, name='Факт 2: океан', status='edited')]
    hook = m.Style(id=stable_id('st', tl_id, 'hook'), font='Montserrat Black', size=120, uppercase=True,
                   position='custom')
    caps = m.Style(id=stable_id('st', tl_id, 'caps'), font='Arial Black', size=96, max_words=2)
    tracks = [
        m.Track(id=stable_id('tr', tl_id, 'v'), type='video', items=[
            m.VideoItem(id=it('v1'), start=0.0, duration=6.0, scene=s1, asset=a('stock1'), volume=0.0,
                        layout=m.Layout(type='fit')),
            m.VideoItem(id=it('v2'), start=6.0, duration=6.0, scene=s2, asset=a('stock2'), volume=0.0,
                        layout=m.Layout(type='crop', params={'path': [[0.0, 420.0, 0.0], [6.0, 900.0, 0.0]]}))]),
        m.Track(id=stable_id('tr', tl_id, 'ov'), type='overlay', items=[
            m.TextItem(id=it('hook'), start=0.0, duration=2.0, scene=s1, text='ТЫ ЭТОГО НЕ ЗНАЛ', style=hook.id,
                       y=0.3, animations=[m.Animation(property='scale', keyframes=[
                           m.Keyframe(t=0.0, value=0.8), m.Keyframe(t=0.2, value=1.0, easing='ease_out')])])]),
        m.Track(id=stable_id('tr', tl_id, 'vo'), type='audio', items=[
            m.AudioItem(id=it('vo'), start=0.3, duration=11.5, asset=a('vo'), role='voice')]),
        m.Track(id=stable_id('tr', tl_id, 'mu'), type='audio', items=[
            m.AudioItem(id=it('mu'), start=0.0, duration=12.0, asset=a('music'), role='music', volume=0.3,
                        duck=True, fade_out=1.5)]),
        m.Track(id=stable_id('tr', tl_id, 'cap'), type='caption', items=[
            m.CaptionItem(id=it('cap'), start=0.0, duration=12.0, style=caps.id, words=[
                m.Word(t0=0.3, t1=0.8, text='Космос'), m.Word(t0=0.8, t1=1.3, text='пахнет', emphasis=True),
                m.Word(t0=6.2, t1=6.8, text='океан')])]),
    ]
    return ensure_valid(m.Timeline(id=tl_id, kind='faceless', duration=12.0, assets=assets, scenes=scenes,
                                   tracks=tracks, styles=[hook, caps], meta={'title': 'Два факта', 'format': 'facts'}))


def placeholder_assets(tl, base_dir, overwrite=False):
    """Small local stand-ins for every file:/gen: asset of a timeline (ffmpeg test patterns and tones), so the
    example timelines render without any generated or downloaded media. -> list of created files."""
    import subprocess
    from pathlib import Path
    from .media import Resolver
    res = Resolver(base_dir)
    made = []
    for i, a in enumerate(tl.assets):
        scheme, rel = a.uri.split(':', 1)
        if scheme not in ('gen', 'file'):
            continue
        dest = (res.root_for(scheme) / rel).resolve()
        if dest.exists() and not overwrite:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dur = f'{a.duration or 5.0:.3f}'
        hz = 220 + 110 * (i % 5)
        if a.type == 'image':
            src = ['-f', 'lavfi', '-i', f'testsrc2=s={a.width or 1024}x{a.height or 1024}:r=1', '-frames:v', '1']
        elif a.type == 'audio':
            src = ['-f', 'lavfi', '-i', f'sine=frequency={hz}:sample_rate=48000:duration={dur}', '-ac', '2']
        elif a.type == 'video':
            src = ['-f', 'lavfi', '-i', f'testsrc2=s={a.width or 1280}x{a.height or 720}:r=30:d={dur}']
            if a.has_audio:
                src += ['-f', 'lavfi', '-i', f'sine=frequency={hz}:sample_rate=48000:duration={dur}', '-shortest',
                        '-c:a', 'aac']
            src += ['-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p']
        else:
            continue
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *src, 'file:' + str(dest)], check=True,
                       timeout=300)
        made.append(dest)
    return made
