#!/usr/bin/env python3
"""Dataset Explorer: browse, search and compare the challenge TSV files in a browser.

Usage (from the project root):
    python3 explorer/server.py                      # then open http://127.0.0.1:8765
    python3 explorer/server.py --port 9000 --data path/to/dataset

The first start builds small indexes (row offsets, sorted ID lookups and a
ground-truth reverse map) into explorer/.cache/. That takes about a minute for
the full dataset; later starts load them instantly. The dataset is only read.
Search runs `grep` directly on the files, so it needs no search index.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from array import array
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

try:  # optional: adds a romanized reading of Indic-script names in the entity view
    from anyascii import anyascii
except ImportError:
    anyascii = None

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "dataset" / "student_resource" / "dataset"
CACHE = HERE / ".cache"
ID_RE = re.compile(r"^\s*(S[123])-(\d+)\s*$", re.I)
ERE_SPECIAL = re.compile(r"([\\.\[\](){}*+?|^$])")
INDIC = re.compile(r"[\u0900-\u0DFF]")
BLOCK_ROWS = 1_000_000
GREP_TIMEOUT = 120
MAX_PAGE = 2000
MAX_HITS = 1000

CAT: Catalog  # set in main()


def log(msg):
    print(f"[explorer] {msg}", flush=True)


class BadRequest(Exception):
    pass


class DataFile:
    """One TSV file: header facts plus lazily built, disk-cached indexes."""

    def __init__(self, path: Path):
        self.path = path
        self.split = path.parent.name
        self.name = path.name
        self.fid = f"{self.split}/{self.name}"
        st = path.stat()
        self.size = st.st_size
        with open(path, "rb") as f:
            self.header = f.readline()
            first = f.readline()
        self.columns = self.header.decode("utf-8", "replace").rstrip("\r\n").split("\t")
        self.kind = "ground_truth" if "matched_entity_ids" in self.columns else "source"
        m = re.match(rb"(S[123])-", first)
        self.source = m.group(1).decode() if (m and self.kind == "source") else None
        self.cache_key = CACHE / f"{self.split}-{path.stem}-{self.size}-{st.st_mtime_ns}"
        self.lock = threading.Lock()
        self.starts = None    # byte offset of every line start; line 0 is the header
        self.ids = None       # sorted numeric part of the column-0 IDs
        self.id_rows = None   # data row for each entry of self.ids
        self.rev = None       # ground truth only: {"S2": (ids, gt_rows), "S3": (ids, gt_rows)}
        self.busy = None
        self.error = None

    # ---------- cache helpers ----------
    def _cached(self, *suffixes):
        paths = [Path(f"{self.cache_key}.{s}.npy") for s in suffixes]
        if all(p.exists() for p in paths):
            return [np.load(p, mmap_mode="r") for p in paths]
        return None

    def _save(self, suffix, arr):
        final = Path(f"{self.cache_key}.{suffix}.npy")
        tmp = Path(f"{self.cache_key}.{suffix}.tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, final)
        return np.load(final, mmap_mode="r")

    @property
    def nrows(self):
        return None if self.starts is None else len(self.starts) - 1

    @property
    def ready(self):
        done = self.starts is not None and self.ids is not None
        return done and (self.kind != "ground_truth" or self.rev is not None)

    def info(self):
        if self.error:
            state = "error"
        elif self.busy:
            state = self.busy
        elif self.ready:
            state = "ready"
        elif self.starts is not None:
            state = "browsable"
        else:
            state = "waiting"
        return {"id": self.fid, "name": self.name, "split": self.split, "kind": self.kind,
                "source": self.source, "size": self.size, "columns": self.columns,
                "rows": self.nrows, "state": state, "error": self.error}

    # ---------- row offsets ----------
    def ensure_offsets(self):
        if self.starts is not None:
            return
        with self.lock:
            if self.starts is not None:
                return
            cached = self._cached("starts")
            if cached:
                self.starts = cached[0]
                return
            self.busy = "indexing rows"
            t = time.time()
            parts = [np.zeros(1, dtype=np.int64)]
            base = 0
            with open(self.path, "rb") as f:
                while buf := f.read(1 << 24):
                    nl = np.flatnonzero(np.frombuffer(buf, dtype=np.uint8) == 10)
                    if len(nl):
                        parts.append(nl.astype(np.int64) + (base + 1))
                    base += len(buf)
            starts = np.concatenate(parts)
            if starts[-1] >= self.size:  # trailing newline: no line starts at EOF
                starts = starts[:-1]
            dtype = np.uint32 if self.size < 2**32 else np.int64
            self.starts = self._save("starts", starts.astype(dtype))
            self.busy = None
            log(f"{self.fid}: {self.nrows:,} rows indexed in {time.time() - t:.1f}s")

    def _span(self, r0, r1):
        """Byte range covering data rows [r0, r1)."""
        s = int(self.starts[r0 + 1])
        e = int(self.starts[r1 + 1]) if r1 + 1 < len(self.starts) else self.size
        return s, e

    def read_rows(self, start, limit):
        self.ensure_offsets()
        n = self.nrows
        start = max(0, min(int(start), n))
        end = max(start, min(n, start + int(limit)))
        if end == start:
            return []
        s, e = self._span(start, end)
        with open(self.path, "rb") as f:
            f.seek(s)
            buf = f.read(e - s)
        # split on "\n" only: str.splitlines() would also break on U+2028 and friends
        lines = buf.decode("utf-8", "replace").split("\n")
        return [{"row": start + i, "fields": ln.rstrip("\r").split("\t")}
                for i, ln in enumerate(lines[: end - start])]

    # ---------- ID lookup ----------
    def ensure_ids(self):
        if self.ids is not None:
            return
        self.ensure_offsets()
        with self.lock:
            if self.ids is not None:
                return
            cached = self._cached("ids", "idrows")
            if cached:
                self.ids, self.id_rows = cached
                return
            self.busy = "indexing IDs"
            t = time.time()
            n = self.nrows
            nums = np.full(n, -1, dtype=np.int64)
            with open(self.path, "rb") as f:
                for r0 in range(0, n, BLOCK_ROWS):
                    r1 = min(n, r0 + BLOCK_ROWS)
                    s, e = self._span(r0, r1)
                    f.seek(s)
                    arr = np.frombuffer(f.read(e - s), dtype=np.uint8)
                    nums[r0:r1] = _parse_ids(arr, self.starts[r0 + 1: r1 + 1].astype(np.int64) - s)
            order = np.argsort(nums, kind="stable")
            self.ids = self._save("ids", nums[order])
            self.id_rows = self._save("idrows", order.astype(np.int32))
            self.busy = None
            log(f"{self.fid}: ID index built in {time.time() - t:.1f}s")

    def find(self, num):
        self.ensure_ids()
        i = int(np.searchsorted(self.ids, num))
        if i < len(self.ids) and int(self.ids[i]) == num:
            return int(self.id_rows[i])
        return None

    # ---------- ground truth: matched record -> its S1 row ----------
    def ensure_reverse(self):
        if self.rev is not None or self.kind != "ground_truth":
            return
        self.ensure_offsets()
        with self.lock:
            if self.rev is not None:
                return
            cached = self._cached("rev2ids", "rev2rows", "rev3ids", "rev3rows")
            if cached:
                self.rev = {"S2": (cached[0], cached[1]), "S3": (cached[2], cached[3])}
                return
            self.busy = "indexing matches"
            t = time.time()
            buckets = {"S2": (array("q"), array("i")), "S3": (array("q"), array("i"))}
            col = self.columns.index("matched_entity_ids")
            for r0 in range(0, self.nrows, BLOCK_ROWS):
                for rec in self.read_rows(r0, BLOCK_ROWS):
                    fields = rec["fields"]
                    if len(fields) <= col or not fields[col]:
                        continue
                    for mid in fields[col].split(","):
                        b = buckets.get(mid[:2].upper())
                        if b is not None and mid[3:].isdigit():
                            b[0].append(int(mid[3:]))
                            b[1].append(rec["row"])
            rev = {}
            for src, (ids, rows) in buckets.items():
                ids = np.frombuffer(ids, dtype=np.int64)
                rows = np.frombuffer(rows, dtype=np.int32)
                order = np.argsort(ids, kind="stable")
                tag = src[1]
                rev[src] = (self._save(f"rev{tag}ids", ids[order]),
                            self._save(f"rev{tag}rows", rows[order]))
            self.rev = rev
            self.busy = None
            log(f"{self.fid}: match map built in {time.time() - t:.1f}s")

    def reverse_find(self, src, num):
        self.ensure_reverse()
        pair = self.rev.get(src)
        if pair is None:
            return None
        ids, rows = pair
        i = int(np.searchsorted(ids, num))
        if i < len(ids) and int(ids[i]) == num:
            return int(rows[i])
        return None


def _parse_ids(arr, line_starts):
    """Vectorised parse of the digits in 'Sx-<digits>\\t...' at each line start."""
    ends = np.append(line_starts[1:], len(arr))
    last = arr[np.maximum(ends - 1, 0)]
    content_end = np.where(last == 10, ends - 1, ends)
    tabs = np.flatnonzero(arr == 9)
    if len(tabs):
        k = np.searchsorted(tabs, line_starts)
        tab_after = np.where(k < len(tabs), tabs[np.minimum(k, len(tabs) - 1)], len(arr))
    else:
        tab_after = np.full(len(line_starts), len(arr))
    id_end = np.minimum(tab_after, content_end)
    pos0 = line_starts + 3
    length = np.clip(id_end - pos0, 0, 18)
    val = np.zeros(len(line_starts), dtype=np.int64)
    bad = length == 0
    for j in range(int(length.max(initial=0))):
        m = j < length
        d = arr[np.minimum(pos0 + j, len(arr) - 1)].astype(np.int64) - 48
        bad |= m & ((d < 0) | (d > 9))
        val = np.where(m, val * 10 + d, val)
    val[bad] = -1
    return val


class Catalog:
    def __init__(self, root: Path):
        self.root = root
        paths = [p for p in root.rglob("*.tsv")
                 if "__MACOSX" not in p.parts and not p.name.startswith("._")]
        files = [DataFile(p) for p in paths]
        files.sort(key=lambda f: (f.split != "train", f.split, f.kind == "ground_truth",
                                  f.source or "", f.name))
        self.files = files
        self.by_id = {f.fid: f for f in files}
        self.splits = list(dict.fromkeys(f.split for f in files))

    def source_file(self, split, src):
        return next((f for f in self.files if f.split == split and f.source == src), None)

    def gt_file(self, split):
        return next((f for f in self.files if f.split == split and f.kind == "ground_truth"), None)

    def warm(self):
        t = time.time()
        for step in ("ensure_offsets", "ensure_ids", "ensure_reverse"):
            for f in self.files:
                try:
                    getattr(f, step)()
                except Exception as e:  # keep indexing the other files
                    f.busy, f.error = None, f"{type(e).__name__}: {e}"
                    log(f"{f.fid}: {step} failed: {f.error}")
        log(f"all indexes ready ({time.time() - t:.0f}s)")


# ---------------------------------------------------------------- search
def grep_pattern(q, col_idx, regex, case):
    flags = ["-a"] + ([] if case else ["-i"])
    if col_idx is None:
        return flags + (["-E"] if regex else ["-F"]), q
    body = q if regex else ERE_SPECIAL.sub(r"\\\1", q)
    skip = f"([^\t]*\t){{{col_idx}}}" if col_idx else ""
    return flags + ["-E"], f"^{skip}[^\t]*({body})"


def run_grep(path, flags, pattern, *, limit=None, count=False, data=None):
    cmd = ["grep", *flags] + (["-c"] if count else ["-n", "-m", str(limit)])
    cmd += ["-e", pattern]
    if data is None:
        cmd += ["--", str(path)]
    p = subprocess.run(cmd, input=data, capture_output=True, timeout=GREP_TIMEOUT,
                       env={**os.environ, "LC_ALL": "C"})
    if p.returncode not in (0, 1):
        msg = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise BadRequest(msg[0] if msg else "search failed")
    return p.stdout


def search_file(f, q, col, regex, case, limit):
    col_idx = None
    if col:
        if col not in f.columns:
            return {"file": f.fid, "skipped": f"no column “{col}”"}
        col_idx = f.columns.index(col)
    flags, pattern = grep_pattern(q, col_idx, regex, case)
    t = time.time()
    out = run_grep(f.path, flags, pattern, limit=limit + 2)
    rows = []
    for line in out.split(b"\n"):
        if not line:
            continue
        num, _, content = line.partition(b":")
        n = int(num)
        if n == 1:  # header line
            continue
        rows.append({"row": n - 2,
                     "fields": content.decode("utf-8", "replace").rstrip("\r").split("\t")})
    more = len(rows) > limit
    return {"file": f.fid, "rows": rows[:limit], "more": more,
            "ms": round((time.time() - t) * 1000)}


def count_file(f, q, col, regex, case):
    col_idx = None
    if col:
        if col not in f.columns:
            return 0
        col_idx = f.columns.index(col)
    flags, pattern = grep_pattern(q, col_idx, regex, case)
    total = int(run_grep(f.path, flags, pattern, count=True).strip() or 0)
    header_hit = int(run_grep(None, flags, pattern, count=True, data=f.header).strip() or 0)
    return total - header_hit


def search_params(qs):
    q = qs.get("q", "").replace("\r", " ").replace("\n", " ").strip()
    if not q:
        raise BadRequest("Type something to search for.")
    return q, qs.get("col") or None, qs.get("regex") == "1", qs.get("case") == "1"


# ---------------------------------------------------------------- entities
def record(f, row):
    fields = f.read_rows(row, 1)[0]["fields"]
    rec = {"id": fields[0], "file": f.fid, "split": f.split, "source": f.source,
           "row": row, "columns": f.columns, "fields": fields}
    if anyascii and "business_name" in f.columns:
        name = fields[f.columns.index("business_name")] if len(fields) > f.columns.index("business_name") else ""
        if INDIC.search(name):
            rec["roman"] = anyascii(name).lower()
    return rec


def resolve(split, eid):
    m = ID_RE.match(eid)
    f = CAT.source_file(split, m.group(1).upper()) if m else None
    row = f.find(int(m.group(2))) if f else None
    return record(f, row) if row is not None else {"id": eid, "missing": True, "split": split}


def entity_groups(eid, split=None):
    m = ID_RE.match(eid or "")
    if not m:
        raise BadRequest("Entity IDs look like S1-965667, S2-681193310 or S3-775321672.")
    src, digits = m.group(1).upper(), m.group(2)
    eid, num = f"{src}-{digits}", int(digits)
    splits = [split] if split in CAT.splits else CAT.splits
    groups = []
    for sp in splits:
        f = CAT.source_file(sp, src)
        row = f.find(num) if f else None
        if row is None:
            continue
        rec = record(f, row)
        g = {"split": sp, "focus": eid, "record": rec, "status": "no_gt", "s1": None, "members": []}
        gt = CAT.gt_file(sp)
        if gt is not None:
            gt_row = gt.find(num) if src == "S1" else gt.reverse_find(src, num)
            if gt_row is None:
                g["status"] = "unmatched" if src != "S1" else "missing_gt"
            else:
                fields = gt.read_rows(gt_row, 1)[0]["fields"]
                matched = [x for x in (fields[1] if len(fields) > 1 else "").split(",") if x]
                g["s1"] = rec if src == "S1" else resolve(sp, fields[0])
                g["members"] = [resolve(sp, x) for x in matched]
                g["gt"] = {"file": gt.fid, "row": gt_row}
                g["status"] = "group" if matched else "singleton"
        groups.append(g)
    if not groups:
        raise BadRequest(f"{eid} is not in any {' or '.join(splits)} file.")
    return {"id": eid, "groups": groups}


# ---------------------------------------------------------------- HTTP
def api_files(qs):
    return {"root": str(CAT.root), "files": [f.info() for f in CAT.files]}


def get_file(qs):
    f = CAT.by_id.get(qs.get("file", ""))
    if f is None:
        raise BadRequest("Unknown file.")
    return f


def api_rows(qs):
    f = get_file(qs)
    start = int(qs.get("start") or 0)
    limit = max(1, min(MAX_PAGE, int(qs.get("limit") or 100)))
    rows = f.read_rows(start, limit)
    return {"file": f.fid, "total": f.nrows, "start": rows[0]["row"] if rows else start, "rows": rows}


def api_search(qs):
    q, col, regex, case = search_params(qs)
    limit = max(1, min(MAX_HITS, int(qs.get("limit") or 100)))
    ids = [x for x in qs.get("files", "").split(",") if x]
    files = [CAT.by_id[x] for x in ids if x in CAT.by_id] if ids else CAT.files
    t = time.time()

    def one(f):
        try:
            return search_file(f, q, col, regex, case, limit)
        except BadRequest as e:
            return {"file": f.fid, "error": str(e)}
        except subprocess.TimeoutExpired:
            return {"file": f.fid, "error": "Search took too long; narrow the query."}

    with ThreadPoolExecutor(max_workers=max(1, len(files))) as ex:
        results = list(ex.map(one, files))
    return {"results": results, "ms": round((time.time() - t) * 1000)}


def api_count(qs):
    f = get_file(qs)
    q, col, regex, case = search_params(qs)
    return {"file": f.fid, "count": count_file(f, q, col, regex, case)}


def api_entity(qs):
    return entity_groups(qs.get("id", ""), qs.get("split") or None)


ROUTES = {"/api/files": api_files, "/api/rows": api_rows, "/api/search": api_search,
          "/api/count": api_count, "/api/entity": api_entity}


class Handler(BaseHTTPRequestHandler):
    server_version = "DatasetExplorer/1.0"

    def log_message(self, fmt, *args):  # keep the console for indexing progress
        pass

    def do_GET(self):
        u = urlparse(self.path)
        qs = {k: v[-1] for k, v in parse_qs(u.query, keep_blank_values=True).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            route = ROUTES.get(u.path)
            if route is None:
                return self._json({"error": "Not found."}, 404)
            self._json(route(qs))
        except (BadRequest, ValueError) as e:
            self._json({"error": str(e)}, 400)
        except BrokenPipeError:
            pass
        except Exception as e:
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def _send(self, body, ctype, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    global CAT
    ap = argparse.ArgumentParser(description="Browse, search and compare the challenge TSV files.")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="folder holding train/ and test/")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-warm", action="store_true", help="build indexes on first use, not at startup")
    args = ap.parse_args()
    if not args.data.is_dir():
        sys.exit(f"Data folder not found: {args.data}")
    CACHE.mkdir(exist_ok=True)
    CAT = Catalog(args.data.resolve())
    if not CAT.files:
        sys.exit(f"No .tsv files under {CAT.root}")
    log(f"{len(CAT.files)} files under {CAT.root}")
    if not args.no_warm:
        threading.Thread(target=CAT.warm, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log(f"open http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
