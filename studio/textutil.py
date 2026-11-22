"""Small, dependency-free text helpers shared by highlights / captions / metadata (RU + EN)."""
import re

TERMINAL = ('.', '!', '?', '…')
_WORD = re.compile(r'\w+', re.UNICODE)

STOP_EN = set('''a an the and or but if then so to of in on at by for with from as is are was were be been being it its this that
these those i you he she we they me him her us them my your his our their not no yes do does did done have has had will would
can could should just really very also there here what which who whom when where why how about into out up down over than too
get got going gonna like know think mean well okay yeah one two all some any more most other such only own same
'''.split())
STOP_RU = set('''и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о
из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может
они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому
этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой
хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше
чуть том нельзя такой им более всегда конечно всю между это эта вот просто короче типа значит
'''.split())
STOP = STOP_EN | STOP_RU


def detect_lang(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 'en'
    cyr = sum(1 for c in letters if '\u0400' <= c <= '\u04ff')
    return 'ru' if cyr / len(letters) > 0.4 else 'en'


def tokens(text):
    return [w.lower().replace('ё', 'е') for w in _WORD.findall(text)]


def content_tokens(text):
    return [w for w in tokens(text) if len(w) >= 4 and w not in STOP and not w.isdigit()]


def stem(w):
    """Very light suffix trimming so 'клипы'/'клип', 'games'/'game' collapse. Good enough for overlap stats."""
    for suf in ('ами', 'ями', 'ого', 'его', 'ому', 'ему', 'ыми', 'ими', 'ать', 'ять', 'ing', 'ies', 'ed', 'es',
                'ов', 'ев', 'ой', 'ей', 'ах', 'ях', 'ом', 'ем', 'ый', 'ий', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие',
                'ы', 'и', 'а', 'я', 'у', 'ю', 'е', 'о', 's'):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def keyword_set(text):
    return {stem(w) for w in content_tokens(text)}


def keywords(text, n=6):
    freq = {}
    first = {}
    for i, w in enumerate(content_tokens(text)):
        s = stem(w)
        freq[s] = freq.get(s, 0) + 1
        first.setdefault(s, w)
    order = sorted(freq, key=lambda s: (-freq[s], len(first[s]) * -1))
    return [first[s] for s in order[:n]]


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def flat_words(segments):
    """All words with timings. Segments lacking word data get evenly spread synthetic words."""
    out = []
    for seg in segments:
        ws = seg.get('words') or []
        if ws:
            out.extend({'start': w['start'], 'end': w['end'], 'text': w['text']} for w in ws)
            if out:
                out[-1]['seg_end'] = True
        else:
            parts = seg.get('text', '').split()
            if not parts:
                continue
            step = (seg['end'] - seg['start']) / len(parts)
            for k, t in enumerate(parts):
                out.append({'start': seg['start'] + k * step, 'end': seg['start'] + (k + 1) * step, 'text': t,
                            'synthetic': True})
            out[-1]['seg_end'] = True
    return out


def words_reliable(words):
    """Word timings are usable for per-word highlighting if mostly monotonic with positive length."""
    if len(words) < 3 or any(w.get('synthetic') for w in words):
        return False
    good = sum(1 for i, w in enumerate(words)
               if w['end'] > w['start'] and (i == 0 or w['start'] >= words[i - 1]['start'] - 0.02))
    return good / len(words) >= 0.9


def build_sentences(words, pause=0.8):
    """Group words into sentences: terminal punctuation, long pause, or a segment edge followed by a gap."""
    sents, cur = [], []

    def flush():
        if cur:
            sents.append({'start': cur[0]['start'], 'end': cur[-1]['end'], 'words': list(cur),
                          'text': ' '.join(w['text'] for w in cur)})
            cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt['start'] - w['end']) if nxt else 99
        terminal = w['text'].rstrip('"»”)').endswith(TERMINAL)
        soft = w.get('seg_end') and gap >= 0.3
        long_run = len(cur) >= 34 and (w['text'].endswith((',', ';', ':')) or gap >= 0.35)
        if terminal or gap >= pause or soft or long_run or nxt is None:
            flush()
    return sents
