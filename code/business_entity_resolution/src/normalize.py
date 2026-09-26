"""Text normalisation shared by every stage (train and test go through exactly the same code).

No rule here is about a particular country. Three kinds of rules:
  1. Universal, script-level: Unicode NFKC; romanisation of any script (anyascii); word-final nasal
     marks dropped; sound keys and phonetic codes; tokens with digits are house / plot numbers.
  2. Learned from the data of each country, without labels (prep.py): address components so common
     in a country that they cannot identify a business -- states, regions, departments, districts,
     big cities -- are ignored when two addresses are compared (AREAS, per country).
  3. A small multilingual dictionary: legal forms, honorifics and articles (MINOR); unit and
     placeholder words (DROP); English ordinal words. Abbreviations are not listed: a short word
     equals a longer one when it starts with the same letter and its letters appear in it in order
     (St = Street = Saint, Bd = Boulevard, Rd = Road, Intl = International), in any language.
In names, the record's own country label ("India", "France", ...) becomes the token "ctry".
"""
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import jellyfish
from anyascii import anyascii
from rapidfuzz.distance import Levenshtein

DROP = {"unit","apt","apartment","apartments","suite","ste","fl","floor","flr","room","rm","bldg","building",
        "po","box","pmb","no","nos","number","h","hno","house","plot","flat","shop","office","ofc","door",
        "null","na","nan","none","n","a"}
ORD_WORDS = {w: f"{i}o" for i, w in enumerate(
    ["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth","eleventh","twelfth",
     "thirteenth","fourteenth","fifteenth","sixteenth","seventeenth","eighteenth","nineteenth","twentieth"], start=1)}
FLOOR = {"floor", "flr", "fl"}
ORD_RE = re.compile(r"^0*(\d+)(st|nd|rd|th)$")
LEAD_INT = re.compile(r"\d+")
SYMBOL = re.compile("[\u00b0\u00ba\u2116]")           # degree / ordinal / numero signs: "N\u00b0 6" -> "N 6"
# legal forms (many countries), honorifics, articles
MINOR = {"inc","incorporated","llc","llp","lp","ltd","limited","corp","corporation","co","company","plc",
         "pllc","pc","pvt","private","opc","sarl","sas","sasu","eurl","sci","sa","ei","snc","scop","selarl",
         "gmbh","srl","ltda","pty","bhd","sdn","pte",
         "smt","sri","shri","shree","mr","mrs","ms","dr",
         "the","and","of","de","du","des","la","le","les","et","der","und","von","del","los","las"}
INDIC = re.compile(r"[\u0900-\u0dff]+")
TOK = re.compile(r"[a-z0-9]+")
NONALNUM = re.compile(r"[^a-z0-9]+")

DIGRAPHS = (("ph", "f"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
            ("bh", "b"), ("ck", "k"))
SOUND = str.maketrans({"c": "k", "q": "k", "z": "s", "x": "s", "j": "g", "w": "v", "y": "a"})
VOWEL_RUN = re.compile(r"[aeiou]+")
REPEAT = re.compile(r"(.)\1+")
NASAL_M = re.compile(r"m(?=[bcdfgjklnpqrstvwxz])")   # anyascii writes the Indic nasal mark as m
TION = re.compile(r"[ts]ion")                          # "-tion" / "-sion" sound like "shan"
FINAL_NASAL = re.compile("[\u0901\u0902\u0981\u0982\u0a01\u0a02\u0a70\u0a81\u0a82\u0b01\u0b02\u0b82"
                         "\u0c01\u0c02\u0c81\u0c82\u0d01\u0d02](?=[\\s,.()\\-]|$)")
AREAS = {}          # country -> set of address components that are too common to identify anything


def romanize(s):
    s = unicodedata.normalize("NFKC", s or "")
    if s.isascii():
        return s.lower()
    return anyascii(FINAL_NASAL.sub("", s)).lower()


@lru_cache(maxsize=2_000_000)
def skeleton(t):
    """Sound-alike form of a word: "keyr" ~ "care", "lojistiks" ~ "logistics", "praivet" ~ "private"."""
    if t.isdigit():
        return t
    t = TION.sub("sn", t)
    for a, b in DIGRAPHS:
        t = t.replace(a, b)
    t = NASAL_M.sub("n", t).translate(SOUND)
    sk = REPEAT.sub(r"\1", VOWEL_RUN.sub("", t))
    return sk or t[:1]


@lru_cache(maxsize=2_000_000)
def akey(t):
    """Sound key used everywhere (names and addresses, any script): the skeleton, except that a
    one-letter skeleton keeps a marker for a leading vowel group (one more letter):
    "maa" = "ma" = "m" (from मा), "it" = "aiti" = "*t" (from आईटी), "amy" = "*m"."""
    sk = skeleton(t)
    if len(sk) >= 2 or t.isdigit():
        return sk
    for a, b in DIGRAPHS:
        t = t.replace(a, b)
    m = REPEAT.sub(r"\1", VOWEL_RUN.sub("*", NASAL_M.sub("n", t).translate(SOUND)))
    return m.rstrip("*") or m


MINOR_SK = {skeleton(w) for w in MINOR if len(skeleton(w)) >= 3}


@lru_cache(maxsize=2_000_000)
def metaphone(t):
    return jellyfish.metaphone(t) if not t.isdigit() else t


@lru_cache(maxsize=2_000_000)
def soundex(t):
    return jellyfish.soundex(t) if not t.isdigit() else t


@lru_cache(maxsize=2_000_000)
def nysiis(t):
    return jellyfish.nysiis(t) if not t.isdigit() else t


@lru_cache(maxsize=2_000_000)
def sounds_same(a, b):
    """Two words are the same word said differently: same sound key, or (3+ letters) same Metaphone."""
    if akey(a) == akey(b):
        return True
    return len(a) >= 3 and len(b) >= 3 and metaphone(a) == metaphone(b)


@lru_cache(maxsize=2_000_000)
def is_abbrev(a, b):
    """a abbreviates b, in any language: shorter, same first letter, a's letters appear in b in order
    (st/street, st/saint, bd/boulevard, rd/road, pvt/private, intl/international, r/rue)."""
    if a[0] != b[0] or a.isdigit() or b.isdigit():
        return False
    if not ((len(a) <= 4 and len(b) >= len(a) + 2) or (len(a) == 5 and len(b) >= 10)):
        return False                                   # "hotel" is not short for "hostel"
    it = iter(b)
    return all(ch in it for ch in a)


def strip_abbrev(added, dropped):
    """Remove pairs (added word, dropped word) where one abbreviates the other; -> (added, dropped, n)."""
    added, dropped = list(added), list(dropped)
    n = 0
    for a in sorted(added, key=len):
        for i, d in enumerate(dropped):
            if is_abbrev(a, d) or is_abbrev(d, a):
                added.remove(a)
                dropped.pop(i)
                n += 1
                break
    return added, dropped, n


def content(tokens):
    return frozenset(t for t in tokens if t not in MINOR and len(t) > 1)


def cross_content(toks):
    """Content words for sound-based comparison; also drops romanised legal forms ("praivet")."""
    return [t for t in toks if t not in MINOR and len(t) > 1 and skeleton(t) not in MINOR_SK]


def name_tokens(name, country=""):
    """(sorted name words with repeats, has Indic script); the country's own name becomes "ctry"."""
    cw = (country or "").lower()
    toks = ["ctry" if t == cw else t for t in TOK.findall(romanize(name))]
    return tuple(sorted(toks)), bool(INDIC.search(name or ""))


def addr_components(addr):
    """Comma-separated address components, romanised and cleaned (Indic-script words, which in this
    data are state names, are removed; symbols such as N\u00b0 are spaced out)."""
    if not addr or not addr.strip():
        return []
    s = SYMBOL.sub(" ", INDIC.sub(" ", unicodedata.normalize("NFKC", addr)))
    return [c for c in (NONALNUM.sub(" ", comp).strip() for comp in romanize(s).split(",")) if c]


def addr_tokens(addr, country=""):
    """-> (comparison words, retrieval words, house/plot numbers). Comparison words leave out the
    country's area components (learned, see AREAS) and placeholder words. None if nothing is left."""
    comps = addr_components(addr)
    if not comps:
        return None
    area = AREAS.get(country, ())
    words, all_words, nums = set(), set(), []
    for comp in comps:
        is_area = comp in area
        toks = comp.split()
        for i, t in enumerate(toks):
            if (t in ORD_WORDS or ORD_RE.match(t)) and i + 1 < len(toks) and toks[i + 1] in FLOOR:
                continue                                   # "7th floor": a floor, dropped
            if t.isdigit():
                nums.append(t.lstrip("0") or "0")
                continue
            if t in ORD_WORDS:
                t = ORD_WORDS[t]
            elif ORD_RE.match(t):
                t = ORD_RE.match(t).group(1) + "o"
            elif any(c.isdigit() for c in t):
                nums.append(t.lstrip("0") or "0")
                continue
            if t in DROP:
                continue
            all_words.add(t)
            if not is_area:
                words.add(t)
    if not words and not nums and not all_words:
        return None
    return tuple(sorted(words)), tuple(sorted(all_words)), tuple(nums)


def retrieval_tokens(ntoks, af):
    """Typed tokens for TF-IDF retrieval: name words, 4-letter prefixes, sound keys, Soundex; address
    words (areas included), prefixes, sound keys, house numbers, and number x word pairs."""
    c = cross_content(ntoks) or [t for t in ntoks if len(t) > 1] or list(ntoks)
    nt = set()
    for t in c:
        nt.add("n" + t)
        if len(t) > 4:
            nt.add("p" + t[:4])
        nt.add("k" + akey(t))
        if len(t) >= 4 and not t.isdigit():
            nt.add("s" + soundex(t))
    at = set()
    if af is not None:
        _, all_words, nums = af
        for w in all_words:
            at.add("a" + w)
            if len(w) > 4:
                at.add("q" + w[:4])
            if len(w) > 2:
                at.add("b" + akey(w))
        for x in nums:
            at.add("d" + x)
        for x in nums[:3]:
            for w in all_words:
                if len(w) >= 4:
                    at.add("x" + x + "_" + w)
    return " ".join(sorted(nt)), " ".join(sorted(at))


# ------------------------------------------------------------------ pair comparison (from the audit)
def pair_typos(added, dropped):
    """Pair each added word with a dropped word that is a typo (1-2 edits) or sounds the same."""
    added, dropped = set(added), set(dropped)
    pairs = []
    for a in sorted(added, key=len, reverse=True):
        best, bd = None, 99
        for d in dropped:
            if len(a) > 1 and len(d) > 1 and not a.isdigit() and not d.isdigit() and sounds_same(a, d):
                best, bd = d, 0
                break
            if min(len(a), len(d)) < 3:
                continue
            thr = 1 if max(len(a), len(d)) <= 4 else 2
            if abs(len(a) - len(d)) > thr:
                continue
            dist = Levenshtein.distance(a, d, score_cutoff=thr)
            if dist <= thr and dist < bd:
                best, bd = d, dist
        if best is not None:
            pairs.append((a, best))
            dropped.discard(best)
    for a, _ in pairs:
        added.discard(a)
    return pairs, added, dropped


def name_compare_cross(S_toks, R_toks):
    s_c, r_c = cross_content(S_toks), cross_content(R_toks)
    s_map = {akey(t): t for t in s_c}
    r_map = {akey(t): t for t in r_c}
    S, R = Counter(akey(t) for t in s_c), Counter(akey(t) for t in r_c)
    s_minor = {akey(t) for t in S_toks if t not in s_c}
    r_minor = {akey(t) for t in R_toks if t not in r_c}
    cs, cr = set(S), set(R)
    union = cs | cr
    jac = len(cs & cr) / len(union) if union else 1.0
    minor = int(s_minor != r_minor)
    if S == R:
        return (11 if minor else 10), 0, 0, 0, minor, (), (), jac
    pairs, a, d = pair_typos(list((R - S).elements()), list((S - R).elements()))
    if not (cs & cr) and not pairs:
        code = 15
    elif not a and not d:
        code = 11 if minor else 10
    elif a and not d:
        code = 12
    elif d and not a:
        code = 13
    else:
        code = 14
    added = sorted(r_map.get(t, t) for t in a)
    dropped = sorted(s_map.get(t, t) for t in d)
    return code, len(a), len(d), len(pairs), minor, added, dropped, jac


def name_compare(S_toks, s_ind, R_toks, r_ind):
    """-> (difference class 0-15, n added, n dropped, n typo / sound-alike pairs, minor-word change,
    added, dropped, jaccard, legal form only re-spelled: Ltd = Limited, Inc = Incorporated)"""
    if s_ind != r_ind:
        return name_compare_cross(S_toks, R_toks) + (0,)
    S, R = Counter(S_toks), Counter(R_toks)
    cs, cr = content(S), content(R)
    union = cs | cr
    jac = len(cs & cr) / len(union) if union else 1.0
    if S == R:
        return 0, 0, 0, 0, 0, (), (), jac, 0
    added, dropped = list((R - S).elements()), list((S - R).elements())
    am, dm = [t for t in added if t in MINOR], [t for t in dropped if t in MINOR]
    ra, rd, _ = strip_abbrev(am, dm)
    same_form = int(bool(am or dm) and not ra and not rd)
    pairs, a, d = pair_typos(added, dropped)
    ca, cd = content(a), content(d)
    minor = int(bool((a - ca) | (d - cd)))
    typo_shared = any(p[0] in cr and p[1] in cs for p in pairs)
    if cs and cr and not (cs & cr) and not typo_shared:
        code = 9
    elif not ca and not cd:
        code = 3 if (pairs and minor) else 2 if pairs else 1
    elif ca and not cd:
        code = 4 if len(ca) == 1 else 5
    elif cd and not ca:
        code = 6 if len(cd) == 1 else 7
    else:
        code = 8
    return code, len(ca), len(cd), len(pairs), minor, sorted(ca), sorted(cd), jac, same_form


def lead_int(t):
    m = LEAD_INT.match(t)
    return int(m.group(0)[:15]) if m else None


def pair_rel(a, b):
    if a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a):
        return 3
    if Levenshtein.distance(a, b) == 1:
        return 4
    ia, ib = lead_int(a), lead_int(b)
    if ia is not None and ib is not None and abs(ia - ib) <= 50:
        return 5
    return 6


NAN = float("nan")


def num_compare(sn, rn):
    """Numbers as multisets -> (relation 0-6, log gap of closest changed pair, its edit distance,
    n added, n dropped, the changed S1 number is S1's first number)."""
    if sorted(sn) == sorted(rn) or "".join(sn) == "".join(rn):
        return 0, 0.0, 0.0, 0, 0, 0
    cs, cr = Counter(sn), Counter(rn)
    dropped, added = list((cs - cr).elements()), list((cr - cs).elements())
    if not dropped:
        return 1, NAN, NAN, len(added), 0, 0
    if not added:
        return 2, NAN, NAN, 0, len(dropped), int(bool(sn) and sn[0] in dropped)
    best = None
    for a in dropped:
        for b in added:
            key = (pair_rel(a, b), Levenshtein.distance(a, b), a, b)
            if best is None or key < best:
                best = key
    rel, lev, a, b = best
    ia, ib = lead_int(a), lead_int(b)
    gap = math.log1p(abs(ia - ib)) if ia is not None and ib is not None else NAN
    return rel, gap, float(lev), len(added), len(dropped), int(bool(sn) and sn[0] == a)


def addr_compare(sa, ra):
    """-> code, rel, n_added, n_dropped, n_typos, added, dropped, jaccard, gap, numlev, n_num_add,
    n_num_drop, first_changed, n_abbreviation_pairs"""
    if ra is None:
        return 8, 7, 0, 0, 0, (), (), NAN, NAN, NAN, 0, 0, 0, 0
    if sa is None:
        return 9, 7, 0, 0, 0, (), (), NAN, NAN, NAN, 0, 0, 0, 0
    (sw, sn), (rw, rn) = sa, ra
    rel, logdiff, numlev, nadd, ndrop, first = num_compare(sn, rn)
    union = sw | rw
    jac = len(sw & rw) / len(union) if union else 1.0
    if sw == rw:
        return (0 if rel == 0 else 1), rel, 0, 0, 0, (), (), jac, logdiff, numlev, nadd, ndrop, first, 0
    a0, d0, n_abbr = strip_abbrev(rw - sw, sw - rw)
    if not a0 and not d0:
        return (0 if rel == 0 else 1), rel, 0, 0, 0, (), (), jac, logdiff, numlev, nadd, ndrop, first, n_abbr
    pairs, a, d = pair_typos(a0, d0)
    if not a and not d:
        code = 2 if rel == 0 else 3
    elif not (sw & rw) and not pairs and not n_abbr:
        code = 7
    elif a and not d:
        code = 4
    elif d and not a:
        code = 5
    else:
        code = 6
    return code, rel, len(a), len(d), len(pairs), sorted(a), sorted(d), jac, logdiff, numlev, nadd, ndrop, first, n_abbr
