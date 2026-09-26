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

## Fresh CPU VM (everything from scratch)
Downloads the dataset from Google Drive, runs the full CPU pipeline (B5), writes a validated `output/` and the pod bundle. Resumable (re-run skips finished steps). ~7 h on 32 vCPU / 256 GB RAM / 200 GB disk.
```
git clone <repo> && cd amazon-ml-challenge-2026 && git checkout neural-members
mkdir -p logs && nohup bash scripts/setup_cpu_vm.sh > logs/setup_cpu_vm.log 2>&1 &
tail -f logs/setup_cpu_vm.log
```
Output: `output/` (B5 submission) and `data/xenc_for_pod.tar` (copy to the pod's repo root, `tar -xf data/xenc_for_pod.tar`).

## GPU pod (neural pair scorers)
The pod needs no raw dataset, only 4 files the CPU VM exports (`python scripts/export_pairs_text.py --tag b5 --feat_dir data/feat_b5`):
```
cd /workspace && git clone <repo> && cd amazon-ml-challenge-2026 && git checkout neural-members   # /workspace: kept when the pod stops; volume >= 50 GB
tar -xf xenc_for_pod.tar   # the bundle from the CPU VM -> data/xenc/{train,score_train,score_test,train_feats}.parquet
bash scripts/gpu_session.sh setup                          # installs, checks files/GPU, caches models; must print SETUP OK
nohup bash scripts/gpu_session.sh auto > /dev/null 2>&1 &  # ~4.5 h cap, stops the pod itself (runpodctl)
tail -f logs/gpu_auto.log
```
Results for the CPU VM: `data/xenc/p_mdeb/`, `data/xenc/p_qwen/` (scores), `data/xenc/m_*/{weights.pt,meta.json}` (models), `logs/`.
Then on the CPU VM: `python src/xenc.py collect --parts data/xenc/p_mdeb --tag xmdeb` (and `p_qwen` → `xqwen`) and STRATEGY.md §7 N6.
