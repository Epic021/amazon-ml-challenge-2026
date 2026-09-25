#!/usr/bin/env bash
# End-to-end smoke test on ~1% of the data. Run before ANY full-scale launch:
#   bash scripts/smoke.sh
# Uses an isolated data dir, so it never touches data/ or output/.
set -euo pipefail
cd "$(dirname "$0")/.."
export BER_DATA=/tmp/ber_smoke/data BER_OUT=/tmp/ber_smoke/output
rm -rf /tmp/ber_smoke && mkdir -p "$BER_DATA" "$BER_OUT"
T=$(date +%s)
step() { echo; echo "=== $* ($(( $(date +%s) - T ))s)"; }

step make_smoke_data;      python scripts/make_smoke_data.py
step normalize;            python src/normalize.py
step "block train";        python src/block.py --split train
step "block test";         python src/block.py --split test
step "mine train";         python src/mine_equiv.py --split train --min_n 5
step "mine test";          python src/mine_equiv.py --split test --min_n 5
step "features train";     python src/features.py --split train
step "features test";      python src/features.py --split test
step train;                python src/train.py --tag smoke --rounds 60
step decode;               python src/decode.py --tag smoke

step "check outputs"
python - <<'EOF'
import os, pandas as pd, csv
d, o = os.environ["BER_DATA"], os.environ["BER_OUT"]
s1 = set(pd.read_parquet(f"{d}/parquet/test_s1.parquet").entity_id)
rd = lambda f: pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
m, c = rd(f"{o}/matching_results.tsv"), rd(f"{o}/candidate_pairs.tsv")
assert list(m.columns) == ["source1_entity_id", "matched_entity_ids"], m.columns
assert list(c.columns) == ["source1_entity_id", "candidate_entity_ids"], c.columns
assert set(m.source1_entity_id) == s1 and m.source1_entity_id.is_unique, "S1 rows mismatch"
bad = 0
for (_, a), (_, b) in zip(m.set_index("source1_entity_id").matched_entity_ids.items(),
                          c.set_index("source1_entity_id").reindex(m.source1_entity_id).candidate_entity_ids.items()):
    ma, cb = [x for x in a.split(",") if x], set(x for x in b.split(",") if x)
    assert len(ma) == len(set(ma)) and all(x[:3] in ("S2-", "S3-") for x in ma)
    bad += not set(ma) <= cb
assert bad == 0, f"{bad} S1 rows have matches outside candidates"
print(f"OUTPUT CHECKS PASSED: {len(m)} rows, {(m.matched_entity_ids != '').mean():.3f} non-empty")
EOF
step "SMOKE PASSED"
