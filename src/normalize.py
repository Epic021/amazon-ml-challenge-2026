"""Record normalization shared by blocking and features.

Produces, per record:
  name_norm  canonical name tokens (ascii, lowercase, abbreviations expanded, leetspeak fixed)
  name_core  name_norm minus legal forms / honorifics / 'the' / 'and'
  addr_norm  canonical address tokens (street types, states, ordinals, NULL tokens removed)
  nums       house/unit numbers found in the address (ordinals excluded, leading zeros stripped)
  nonlatin   1 if the raw name had non-Latin letters (e.g. Devanagari)

  python src/normalize.py        # writes data/norm/{train,test}_s{1,2,3}.parquet
"""
import os
import re
import time
import unicodedata
from multiprocessing import Pool

import ftfy
import pandas as pd
from anyascii import anyascii

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQ = os.path.join(ROOT, "data", "parquet")
NORM = os.path.join(ROOT, "data", "norm")

NAME_ABBR = {
    "pvt": "private", "prvt": "private", "pte": "private", "ltd": "limited", "ltda": "limited", "co": "company",
    "cos": "company", "corp": "corporation", "inc": "incorporated", "intl": "international", "mfg": "manufacturing",
    "svc": "services", "svcs": "services", "assoc": "associates", "assocs": "associates", "bros": "brothers",
    "mgmt": "management", "tech": "technologies", "techs": "technologies", "ent": "enterprises", "cie": "compagnie",
    "ets": "etablissements", "dept": "department", "univ": "university", "hosp": "hospital", "ctr": "center",
    "centre": "center", "st": "saint", "ste": "sainte",
}
LEGAL = set("""private limited llc llp incorporated corporation company lp pllc pc plc gmbh sarl sas sasu eurl
sci sa ei selarl scp snc""".split())
HONOR = set("smt sri shri shree mr mrs ms dr m s the and of dba aka fka formerly".split())
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})

ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "dr": "drive", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "ln": "lane", "ct": "court", "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "hwy": "highway",
    "pkwy": "parkway", "pl": "place", "cir": "circle", "ter": "terrace", "terr": "terrace", "trl": "trail",
    "sq": "square", "cres": "crescent", "crs": "cours", "r": "rue", "imp": "impasse", "chem": "chemin",
    "rte": "route", "fbg": "faubourg", "qu": "quai", "mt": "mount", "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "bldg": "building", "blk": "block",
    "sec": "sector", "opp": "opposite", "nr": "near", "ngr": "nagar", "dist": "district", "mkt": "market",
    "city": "", "no": "", "hno": "", "h": "", "null": "", "nan": "", "na": "", "none": "", "unit": "",
    "door": "", "plot": "", "flat": "", "house": "", "ndeg": "", "bis": "",
}
ORD_WORDS = {w: str(i) for i, w in enumerate(
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
    "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy"}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "jammu and kashmir": "jk", "jammu & kashmir": "jk"}
STATE_MAPS = {"US": US_STATES, "India": IN_STATES}
_STATE_RX = {c: re.compile(r"\b(" + "|".join(sorted(map(re.escape, m), key=len, reverse=True)) + r")\b")
             for c, m in STATE_MAPS.items()}

_NONALNUM = re.compile(r"[^0-9a-z]+")
_ORD = re.compile(r"^(\d+)(st|nd|rd|th)$")
_NUMTOK = re.compile(r"\d+")


def to_ascii(s: str) -> str:
    if not s.isascii():
        s = unicodedata.normalize("NFKC", ftfy.fix_text(s))
        s = re.sub(r"(?i)n\s*[°º]", " no ", s).replace("°", " ").replace("º", " ")
        s = anyascii(s)
    return s.lower()


def has_nonlatin(s: str) -> int:
    return int(any(ord(c) > 0x24F and c.isalpha() for c in s))


def _join_initials(toks):
    """l l c -> llc, u s a -> usa (runs of >=2 single letters)."""
    out, run = [], []
    for t in toks + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.extend(["".join(run)] if len(run) > 1 else run)
            run = []
        if t:
            out.append(t)
    return out


def norm_name(raw: str):
    s = to_ascii(raw).replace("&", " and ").replace("'", "")
    raw_toks, out = _join_initials(_NONALNUM.sub(" ", s).split()), []
    for t in raw_toks:
        if len(t) >= 3 and sum(c.isdigit() for c in t) == 1 and not _ORD.match(t):
            t = t.translate(LEET)                       # c0nsultants -> consultants, we1fare -> welfare
        t = NAME_ABBR.get(t, t)
        if t:
            out.append(t)
    core = [t for t in out if t not in LEGAL and t not in HONOR]
    return " ".join(out), " ".join(core)


def norm_addr(raw: str, country: str):
    s = to_ascii(raw).replace("'", "")
    rx = _STATE_RX.get(country)
    if rx is not None:
        m = STATE_MAPS[country]
        s = rx.sub(lambda g: m[g.group(1)], s)
    toks, nums = [], []
    for t in _NONALNUM.sub(" ", s).split():
        o = _ORD.match(t)
        if o:                                           # 4th -> 4 (street ordinal, not a house number)
            toks.append(o.group(1).lstrip("0") or "0")
            continue
        if t in ORD_WORDS:
            toks.append(ORD_WORDS[t])
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
            nums.append(t)
            toks.append(t)
            continue
        d = _NUMTOK.match(t)
        if d:                                           # 13741d -> 13741 + d
            n = d.group(0).lstrip("0") or "0"
            nums.append(n)
            toks.append(n)
            t = t[d.end():]
        t = ADDR_ABBR.get(t, t)
        if t:
            toks.append(t)
    return " ".join(toks), " ".join(nums)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    nn, nc = zip(*map(norm_name, df.business_name.values)) if len(df) else ((), ())
    an, nu = zip(*map(norm_addr, df.business_address.values, df.country.values)) if len(df) else ((), ())
    return pd.DataFrame({
        "id": df.entity_id.values, "src": df.src.values, "country": df.country.values,
        "name_norm": nn, "name_core": nc, "addr_norm": an, "nums": nu,
        "nonlatin": [has_nonlatin(s) for s in df.business_name.values],
        "addr_empty": (df.business_address.str.strip() == "").astype("int8").values,
    })


def _chunks(df, n=100_000):
    return [df.iloc[i:i + n] for i in range(0, len(df), n)]


def main():
    os.makedirs(NORM, exist_ok=True)
    procs = os.cpu_count() or 4
    for split in ("train", "test"):
        for k in (1, 2, 3):
            t0 = time.time()
            df = pd.read_parquet(f"{PQ}/{split}_s{k}.parquet")
            with Pool(procs) as p:
                out = pd.concat(p.map(normalize_frame, _chunks(df)), ignore_index=True)
            out.to_parquet(f"{NORM}/{split}_s{k}.parquet", index=False)
            print(f"{split}_s{k}: {len(out):,} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
