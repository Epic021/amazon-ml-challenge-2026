# Amazon ML Challenge 2026: Business Entity Resolution

Team repo. Start with **[STRATEGY.md](STRATEGY.md)**: the problem, data findings, validation, pipeline, roles and the 72-hour plan.

## Setup
Place the organizer's `student_resource/` folder (dataset + validator) at the repo root. It is git-ignored and must never be committed.

```
student_resource/dataset/{train,test}/*.tsv
```

## EDA
Open **[eda/report.html](eda/report.html)** in a browser for the visual EDA report: structure, scripts, noise, match structure, hard negatives, test shift and France.

Scripts in `eda/` read from `student_resource/dataset`, or from `$DATA_DIR` if set:
- `eda1.py`: label structure, match counts, basic column stats
- `eda2.py`: source styles, scripts, leakage checks, matched-pair similarity
- `eda3.py`: blocking-recall probe (IDF inverted index), hard-negative examples
- `eda4.py`: test distractor shift, sibling-vs-orphan negatives, number and name-difference patterns, France
- `eda_test.py`: test-set sizes, country mix, train/test overlap

## GPU pod (neural pair scorers)
The pod needs no raw dataset, only 4 files the CPU VM exports (`python scripts/export_pairs_text.py --tag b5 --feat_dir data/feat_b5`):
```
cd /workspace && git clone <repo> && cd amazon-ml-challenge-2026 && git checkout neural-members   # /workspace: kept when the pod stops; volume >= 50 GB
mkdir -p data/xenc   # copy train.parquet, score_train.parquet, score_test.parquet, train_feats.parquet here
bash scripts/gpu_session.sh setup                          # installs, checks files/GPU, caches models; must print SETUP OK
nohup bash scripts/gpu_session.sh auto > /dev/null 2>&1 &  # ~4.5 h cap, stops the pod itself (runpodctl)
tail -f logs/gpu_auto.log
```
Results for the CPU VM: `data/xenc/p_mdeb/`, `data/xenc/p_qwen/` (scores), `data/xenc/m_*/{weights.pt,meta.json}` (models), `logs/`.
Then on the CPU VM: `python src/xenc.py collect --parts data/xenc/p_mdeb --tag xmdeb` (and `p_qwen` → `xqwen`) and STRATEGY.md §7 N6.
