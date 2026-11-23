"""Ids for timeline entities: `<prefix>_<token>`.

stable_id(...) is deterministic: the same inputs always give the same id, so rebuilding a timeline from the same
project / clip / scene keeps every id (and with it edits, comments and per-scene regeneration).
new_id(...) is random, for entities a user creates by hand.
"""
import hashlib
import re
import secrets

ID_RX = re.compile(r'^[a-z][a-z0-9]{1,7}_[0-9a-z]{6,40}$')
PREFIXES = {
    'tl': 'timeline', 'as': 'asset', 'sc': 'scene', 'tr': 'track', 'it': 'item', 'st': 'style', 'ch': 'character',
}


def stable_id(prefix, *parts):
    raw = '\x1f'.join(str(p) for p in parts).encode('utf-8')
    return f'{prefix}_{hashlib.sha1(raw).hexdigest()[:16]}'


def new_id(prefix):
    return f'{prefix}_{secrets.token_hex(8)}'


def is_valid(value):
    return isinstance(value, str) and bool(ID_RX.match(value))
