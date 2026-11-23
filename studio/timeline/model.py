"""Timeline IR v1: one JSON description of a vertical video, shared by every ClipForge tool.

Generators (Stream Clipper, Cartoon Studio, Video Generator, ...) *describe* a video as a Timeline; editors
(Viral Director, manual edits) change it; one renderer turns it into a file. Standard library only.

Conventions
* time is in seconds (float, millisecond precision when serialised);
  `Item.start` / `Scene.start` / `Word.t0/t1` are on the timeline, `Keyframe.t` is relative to its item,
  `src_in` is the position inside the source asset;
* positions are normalised to the canvas (0..1, origin top-left), sizes in canvas pixels;
* every entity has a stable id `<prefix>_<token>`: derived ids are deterministic (ids.stable_id), so a scene or a
  clip keeps its id when the timeline is rebuilt and can be regenerated on its own;
* `ext` on every node holds namespaced extra data ("clipforge.render": {...}); unknown keys elsewhere are errors.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

SCHEMA_ID = 'clipforge.timeline'
VERSION = '1.0'

Kind = Literal['clip', 'cartoon', 'faceless', 'generic']
AssetType = Literal['video', 'audio', 'image', 'font', 'other']
TrackType = Literal['video', 'overlay', 'audio', 'caption']
Easing = Literal['linear', 'ease_in', 'ease_out', 'ease_in_out', 'hold']
AnimProperty = Literal['x', 'y', 'scale', 'rotation', 'opacity', 'zoom', 'pan_x', 'pan_y', 'volume']
AudioRole = Literal['source', 'voice', 'music', 'sfx']
SceneStatus = Literal['draft', 'generated', 'edited', 'locked']
Fit = Literal['cover', 'contain', 'fill']
TransitionKind = Literal['cut', 'fade', 'dissolve', 'wipe', 'zoom', 'whoosh']
TextAlign = Literal['left', 'center', 'right']
CaptionPosition = Literal['bottom', 'middle', 'top', 'custom']
# the layouts the current renderer (studio/layouts.py) knows, plus generic ones for generated media
LayoutType = Literal['fit', 'fullframe', 'gameplay', 'screen', 'crop', 'smart', 'center', 'talking', 'action',
                     'game_cam', 'reaction_cam']


def _ext():
    return field(default_factory=dict)


@dataclass(kw_only=True)
class Keyframe:
    t: float                       # seconds from the item's start
    value: float
    easing: Easing = 'linear'


@dataclass(kw_only=True)
class Animation:
    property: AnimProperty
    keyframes: List[Keyframe]


@dataclass(kw_only=True)
class Generation:
    """How an asset or a scene was produced: what is needed to regenerate it."""
    provider: str = ''             # 'stream_clipper', 'sdxl-local', 'piper', 'ollama', ...
    model: str = ''
    prompt: str = ''
    negative: str = ''
    seed: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    revision: int = 1              # bumped on every regeneration


@dataclass(kw_only=True)
class Canvas:
    width: int = 1080
    height: int = 1920
    fps: float = 30.0
    background: str = '#000000'


@dataclass(kw_only=True)
class Asset:
    id: str
    type: AssetType
    uri: str                       # 'media:sources/x.mp4', 'cache:<fp>/audio16k.wav', 'file:relative/path', 'gen:<id>'
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    has_audio: Optional[bool] = None
    checksum: str = ''
    generation: Optional[Generation] = None
    ext: Dict[str, Any] = _ext()


@dataclass(kw_only=True)
class Layout:
    """How a source frame is placed on the vertical canvas (see studio/layouts.py)."""
    type: LayoutType
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Word:
    t0: float
    t1: float
    text: str
    emphasis: bool = False         # key word (yellow in the 'stream' caption style)
    speaker: Optional[str] = None


@dataclass(kw_only=True)
class ItemBase:
    id: str
    start: float
    duration: float
    scene: Optional[str] = None
    animations: List[Animation] = field(default_factory=list)
    ext: Dict[str, Any] = _ext()


@dataclass(kw_only=True)
class VideoItem(ItemBase):
    type: Literal['video'] = 'video'
    asset: str
    src_in: float = 0.0
    speed: float = 1.0
    volume: float = 1.0            # the video's own sound (also see AudioItem role 'source')
    layout: Optional[Layout] = None


@dataclass(kw_only=True)
class ImageItem(ItemBase):
    type: Literal['image'] = 'image'
    asset: str
    fit: Fit = 'cover'


@dataclass(kw_only=True)
class TextItem(ItemBase):
    type: Literal['text'] = 'text'
    text: str
    style: Optional[str] = None
    x: float = 0.5
    y: float = 0.5
    align: TextAlign = 'center'


@dataclass(kw_only=True)
class AudioItem(ItemBase):
    type: Literal['audio'] = 'audio'
    asset: str
    role: AudioRole = 'source'
    src_in: float = 0.0
    volume: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    duck: bool = False             # lowered while a 'voice'/'source' item speaks


@dataclass(kw_only=True)
class CaptionItem(ItemBase):
    type: Literal['caption'] = 'caption'
    words: List[Word]
    style: Optional[str] = None


@dataclass(kw_only=True)
class TransitionItem(ItemBase):
    """Joins two items of the same track; `start`/`duration` = the overlap where the transition plays."""
    type: Literal['transition'] = 'transition'
    kind: TransitionKind = 'fade'
    from_item: str
    to_item: str


Item = Union[VideoItem, ImageItem, TextItem, AudioItem, CaptionItem, TransitionItem]
ITEM_TYPES = {cls.__dataclass_fields__['type'].default: cls
              for cls in (VideoItem, ImageItem, TextItem, AudioItem, CaptionItem, TransitionItem)}
# which item types a track may hold
TRACK_ITEMS = {
    'video': {'video', 'image', 'transition'},
    'overlay': {'video', 'image', 'text', 'transition'},
    'audio': {'audio'},
    'caption': {'caption', 'text'},
}


@dataclass(kw_only=True)
class Track:
    id: str
    type: TrackType
    name: str = ''
    items: List[Item] = field(default_factory=list)
    muted: bool = False
    hidden: bool = False
    ext: Dict[str, Any] = _ext()


@dataclass(kw_only=True)
class Style:
    """Text / caption look (mirrors studio/captions.py STYLES)."""
    id: str
    font: str = 'Arial'
    size: int = 72
    bold: bool = True
    uppercase: bool = False
    color: str = '#FFFFFF'
    highlight_color: str = '#FFE14D'
    outline_color: str = '#000000'
    outline: float = 6.0
    max_words: int = 3
    position: CaptionPosition = 'bottom'
    margin_v: Optional[int] = None
    ext: Dict[str, Any] = _ext()


@dataclass(kw_only=True)
class Character:
    """A recurring character (cartoons): keeps the same look and voice in every scene."""
    id: str
    name: str
    description: str = ''
    voice: str = ''
    refs: List[str] = field(default_factory=list)     # reference image asset ids
    seed: Optional[int] = None
    ext: Dict[str, Any] = _ext()


@dataclass(kw_only=True)
class Scene:
    """A time range that is generated / edited / regenerated as one unit."""
    id: str
    start: float
    duration: float
    name: str = ''
    status: SceneStatus = 'draft'
    characters: List[str] = field(default_factory=list)
    generation: Optional[Generation] = None
    notes: str = ''
    ext: Dict[str, Any] = _ext()


@dataclass(kw_only=True)
class Timeline:
    id: str
    kind: Kind = 'generic'
    duration: float
    schema: str = SCHEMA_ID
    version: str = VERSION
    canvas: Canvas = field(default_factory=Canvas)
    assets: List[Asset] = field(default_factory=list)
    scenes: List[Scene] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    styles: List[Style] = field(default_factory=list)
    characters: List[Character] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    ext: Dict[str, Any] = _ext()

    # ---------------------------------------------------------------- lookups
    def asset(self, aid):
        return next((a for a in self.assets if a.id == aid), None)

    def scene(self, sid):
        return next((s for s in self.scenes if s.id == sid), None)

    def items(self, scene=None):
        """All (track, item) pairs, optionally only those of one scene."""
        return [(t, i) for t in self.tracks for i in t.items if scene is None or i.scene == scene]
