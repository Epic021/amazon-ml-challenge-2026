"""Word evidence features for pair models: smoothed log-odds of the words a record adds to or
drops from its S1 name and address, estimated only from rows whose labels the model may see.

Training rows are encoded out-of-fold (statistics from the other training folds), so no row's
own label leaks into its features; held-out rows use statistics from all training rows.
"""
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

WORDCOLS = ["name_added", "name_dropped", "addr_added", "addr_dropped"]
NAMES = [f"{c}_{s}" for c in WORDCOLS for s in ("lo_max", "lo_min", "lo_mean", "unseen")]


def load_tokens(path, n):
    """Per word column: (row offsets, word ids, vocabulary size, vocabulary, row of each word)."""
    table = pq.read_table(path, columns=WORDCOLS)
    toks = {}
    for col in WORDCOLS:
        lists = pc.split_pattern(table[col].combine_chunks(), pattern=" ")
        flat, parents = pc.list_flatten(lists), pc.list_parent_indices(lists)
        keep = pc.not_equal(flat, "")
        flat, parents = pc.filter(flat, keep), pc.filter(parents, keep).to_numpy().astype(np.int32)
        enc = pc.dictionary_encode(flat)
        ids = enc.indices.to_numpy().astype(np.int32)
        offsets = np.concatenate([[0], np.cumsum(np.bincount(parents, minlength=n))]).astype(np.int64)
        toks[col] = (offsets, ids, len(enc.dictionary), enc.dictionary, parents)
    return toks


def encode(toks, y, stats_mask, n):
    """Features for every row, using label statistics from the rows in stats_mask only."""
    out = []
    for col in WORDCOLS:
        offsets, ids, V, _, parents = toks[col]
        seen_row = stats_mask[parents]
        lab = y[parents]
        pos = np.bincount(ids[seen_row & (lab == 1)], minlength=V).astype(np.float64)
        neg = np.bincount(ids[seen_row & (lab == 0)], minlength=V).astype(np.float64)
        lo = np.log((pos + 1) / (pos.sum() + V)) - np.log((neg + 1) / (neg.sum() + V))
        unseen = (pos + neg) == 0
        lo[unseen] = 0.0
        vals, uns = lo[ids].astype(np.float32), unseen[ids].astype(np.float32)
        lens = np.diff(offsets)
        nz = lens > 0
        starts = offsets[:-1][nz]
        mx, mn, me = (np.full(n, np.nan, np.float32) for _ in range(3))
        nu = np.zeros(n, np.float32)
        if len(vals):
            mx[nz] = np.maximum.reduceat(vals, starts)
            mn[nz] = np.minimum.reduceat(vals, starts)
            me[nz] = np.add.reduceat(vals, starts) / lens[nz]
            nu[nz] = np.add.reduceat(uns, starts)
        out += [mx, mn, me, nu]
    return np.column_stack(out)


def encode_train_and_test(toks, y, fold, train_mask, test_mask, n):
    """Leak-free features: each training fold encoded from the other training folds; test rows
    encoded from all training rows. Rows in neither mask stay NaN."""
    F = np.full((n, len(NAMES)), np.nan, dtype=np.float32)
    for j in np.unique(fold[train_mask]):
        inner = train_mask & (fold == j)
        F[inner] = encode(toks, y, train_mask & (fold != j), n)[inner]
    F[test_mask] = encode(toks, y, train_mask, n)[test_mask]
    return F


def top_words(toks, col, mask, k=15):
    offsets, ids, V, vocab, parents = toks[col]
    cnt = np.bincount(ids[mask[parents]], minlength=V)
    top = np.argsort(-cnt)[:k]
    return {vocab[int(i)].as_py(): int(cnt[i]) for i in top if cnt[i] > 0}
