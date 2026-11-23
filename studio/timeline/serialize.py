"""Timeline <-> dict / JSON, version migrations, content fingerprints and the JSON Schema.

* from_dict is strict: an unknown key is an error (typos must not silently vanish); free data goes into `ext`.
* to_json(canonical=True) is byte-stable (sorted keys, times rounded to 1 ms) -> fingerprints and diffs.
* migrate(data) upgrades older documents step by step; a newer version than this ClipForge knows is refused.
"""
import dataclasses
import hashlib
import json
import typing
from typing import Any, Dict, List, Literal, Optional, Union, get_args, get_origin

from . import model as m

ROUND = 3        # seconds -> milliseconds


class TimelineError(ValueError):
    """Raised for documents that cannot be read (bad structure / version) or are invalid (see validate.py)."""

    def __init__(self, message, issues=None):
        super().__init__(message)
        self.issues = list(issues or [])


# ------------------------------------------------------------------ migrations
_MIGRATIONS = {}          # 'from version' -> (to version, fn(dict) -> dict)


def register_migration(from_version, to_version):
    def deco(fn):
        _MIGRATIONS[from_version] = (to_version, fn)
        return fn
    return deco


def _ver(v):
    try:
        major, minor = str(v).split('.')
        return int(major), int(minor)
    except (ValueError, AttributeError):
        raise TimelineError(f'invalid timeline version {v!r}')


def migrate(data):
    """Upgrade a timeline dict to the current VERSION (returns a new dict)."""
    if not isinstance(data, dict):
        raise TimelineError('a timeline must be a JSON object')
    if data.get('schema', m.SCHEMA_ID) != m.SCHEMA_ID:
        raise TimelineError(f'not a ClipForge timeline (schema {data.get("schema")!r})')
    data = dict(data)
    seen = set()
    while data.get('version', m.VERSION) != m.VERSION:
        v = data.get('version')
        if _ver(v) > _ver(m.VERSION):
            raise TimelineError(f'timeline version {v} is newer than this ClipForge supports ({m.VERSION}); update ClipForge')
        if v not in _MIGRATIONS or v in seen:
            raise TimelineError(f'no migration from timeline version {v} to {m.VERSION}')
        seen.add(v)
        to, fn = _MIGRATIONS[v]
        data = fn(dict(data))
        data['version'] = to
    data['version'] = m.VERSION
    return data


# ------------------------------------------------------------------ dataclass -> dict
def _round(v):
    return round(v, ROUND) if isinstance(v, float) else v


def _is_float(tp):
    return tp is float or (get_origin(tp) is Union and float in get_args(tp))


def to_dict(obj):
    """Dataclass tree -> plain dict (floats rounded to milliseconds; a float field always serialises as a float,
    so 30 and 30.0 give the same bytes and fingerprints stay stable)."""
    if dataclasses.is_dataclass(obj):
        hints = _hints(type(obj))
        out = {}
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if isinstance(v, int) and not isinstance(v, bool) and _is_float(hints[f.name]):
                v = float(v)
            out[f.name] = to_dict(v)
        return out
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return _round(obj)


def to_json(tl, canonical=True, indent=None):
    return json.dumps(to_dict(tl), ensure_ascii=False, sort_keys=canonical, indent=indent,
                      separators=(',', ':') if indent is None else None)


# ------------------------------------------------------------------ dict -> dataclass (strict)
def _hints(cls):
    return typing.get_type_hints(cls)


def _fail(path, msg):
    raise TimelineError(f'{path}: {msg}', [{'path': path, 'message': msg}])


def _convert(value, tp, path):
    origin = get_origin(tp)
    if tp is Any:
        return value
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if value is None and type(None) in get_args(tp):
            return None
        if all(dataclasses.is_dataclass(a) for a in args) and len(args) > 1:     # Item union -> discriminator
            if not isinstance(value, dict) or value.get('type') not in m.ITEM_TYPES:
                _fail(path, f'unknown item type {value.get("type") if isinstance(value, dict) else value!r}; '
                            f'expected one of {sorted(m.ITEM_TYPES)}')
            return _build(m.ITEM_TYPES[value['type']], value, path)
        return _convert(value, args[0], path)
    if origin is Literal:
        if value not in get_args(tp):
            _fail(path, f'{value!r} is not one of {list(get_args(tp))}')
        return value
    if origin in (list, List):
        if not isinstance(value, list):
            _fail(path, 'expected a list')
        (inner,) = get_args(tp)
        return [_convert(v, inner, f'{path}[{i}]') for i, v in enumerate(value)]
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            _fail(path, 'expected an object')
        return dict(value)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(path, f'expected a number, got {value!r}')
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            _fail(path, f'expected an integer, got {value!r}')
        return value
    if tp is bool:
        if not isinstance(value, bool):
            _fail(path, f'expected true/false, got {value!r}')
        return value
    if tp is str:
        if not isinstance(value, str):
            _fail(path, f'expected a string, got {value!r}')
        return value
    return value


def _build(cls, data, path):
    if not isinstance(data, dict):
        _fail(path, f'expected an object for {cls.__name__}')
    hints = _hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        _fail(path, f'unknown field(s) {unknown} for {cls.__name__} (put extra data into "ext")')
    kwargs = {}
    for name, f in fields.items():
        if name in data:
            kwargs[name] = _convert(data[name], hints[name], f'{path}.{name}' if path else name)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            _fail(f'{path}.{name}' if path else name, 'required field is missing')
    return cls(**kwargs)


def from_dict(data):
    """dict -> Timeline (after migrations). Structural problems raise TimelineError; use validate() for rules."""
    return _build(m.Timeline, migrate(data), '')


def from_json(text):
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise TimelineError(f'not valid JSON: {exc}')
    return from_dict(data)


# ------------------------------------------------------------------ fingerprints
def _hash(obj):
    return hashlib.sha256(json.dumps(to_dict(obj), ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()[:20]


def content_hash(tl):
    """Changes whenever anything in the timeline changes (meta excluded: titles do not change the picture)."""
    d = to_dict(tl)
    d.pop('meta', None)
    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:20]


def scene_fingerprint(tl, scene_id):
    """What one scene renders from: the scene, its items, the assets/styles/characters they use.
    Unchanged fingerprint = the scene's render can be reused; only changed scenes are regenerated."""
    sc = tl.scene(scene_id)
    if sc is None:
        raise TimelineError(f'unknown scene {scene_id!r}')
    items = [(t.id, t.type, it) for t, it in tl.items(scene=scene_id)]
    asset_ids = {getattr(it, 'asset', None) for _, _, it in items} - {None}
    style_ids = {getattr(it, 'style', None) for _, _, it in items} - {None}
    payload = {
        'scene': to_dict(sc),
        'items': [[tid, ttype, to_dict(it)] for tid, ttype, it in items],
        'assets': [to_dict(a) for a in tl.assets if a.id in asset_ids],
        'styles': [to_dict(s) for s in tl.styles if s.id in style_ids],
        'characters': [to_dict(c) for c in tl.characters if c.id in set(sc.characters)],
        'canvas': to_dict(tl.canvas),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:20]


# ------------------------------------------------------------------ JSON Schema (draft 2020-12)
def _schema_for(tp, defs):
    origin = get_origin(tp)
    if tp is Any:
        return {}
    if origin is Union:
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if all(dataclasses.is_dataclass(a) for a in non_none) and len(non_none) > 1:
            return {'oneOf': [_schema_for(a, defs) for a in non_none],
                    'discriminator': {'propertyName': 'type'}}
        inner = _schema_for(non_none[0], defs)
        return {'anyOf': [inner, {'type': 'null'}]} if type(None) in args else inner
    if origin is Literal:
        return {'enum': list(get_args(tp))}
    if origin in (list, List):
        return {'type': 'array', 'items': _schema_for(get_args(tp)[0], defs)}
    if origin in (dict, Dict):
        return {'type': 'object'}
    if dataclasses.is_dataclass(tp):
        name = tp.__name__
        if name not in defs:
            defs[name] = {}                       # placeholder against recursion
            hints = _hints(tp)
            props, required = {}, []
            for f in dataclasses.fields(tp):
                props[f.name] = _schema_for(hints[f.name], defs)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                    required.append(f.name)
                elif f.default is not dataclasses.MISSING and isinstance(f.default, (str, int, float, bool)):
                    props[f.name]['default'] = f.default
            defs[name] = {'type': 'object', 'properties': props, 'required': required, 'additionalProperties': False,
                          'description': (tp.__doc__ or '').strip().splitlines()[0] if tp.__doc__ else ''}
        return {'$ref': f'#/$defs/{name}'}
    return {float: {'type': 'number'}, int: {'type': 'integer'}, bool: {'type': 'boolean'},
            str: {'type': 'string'}}.get(tp, {})


def json_schema():
    """JSON Schema of Timeline IR (for editors, and for LLMs that write timelines as structured output)."""
    defs = {}
    root = _schema_for(m.Timeline, defs)
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema',
            '$id': f'https://clipforge.local/schema/timeline-{m.VERSION}.json',
            'title': f'ClipForge Timeline IR {m.VERSION}', **root, '$defs': defs}
