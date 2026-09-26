# Data analysis, label audit and pipeline: index

Everything here was produced from the challenge's training/test files only. None of the data
itself is committed: the repo `.gitignore` excludes `*.tsv` / `*.parquet`.

| Path | What it is | How to use it |
|---|---|---|
| [`docs/DATASET_STRATEGY.md`](DATASET_STRATEGY.md) | **Start here.** How each kind of record or noise is treated, in plain words, with real examples | read |
| [`docs/METHODOLOGY.md`](METHODOLOGY.md) | The solution methodology: normalization, retrieval, features, two-stage model, decoding, France | read |
| [`eda_output/report.html`](../eda_output/report.html) | EDA report with charts (distributions, noise patterns, anomalies, test vs train) | open in a browser; the scripts `eda.py`, `eda2.py`, `verify.py`, `reverify_*.py` rebuild it |
| [`eda_output/label_audit/REPORT.md`](../eda_output/label_audit/REPORT.md) | Training-label audit: 11 insights with match / no-match examples, the hand-checked cases, and the label-cleaning experiment (5 variants) | read; rerun with `build_pairs.py → analyze.py → curation_experiment.py → examples.py → build_markdown.py` |
| `eda_output/label_audit/report.html` | HTML version of the audit | open in a browser |
| [`explorer/`](../explorer) | Local dataset explorer: browse all train/test files, search across them, open files side by side | `python3 explorer/server.py --port 9000 --data <dataset dir>` then open http://localhost:9000 |
| [`code/business_entity_resolution/`](../code/business_entity_resolution) | The end-to-end pipeline (prep → retrieval → features → two-stage LightGBM → decoder → submission files) | see its `README.md`; `bash run_all.sh A` |

## Key numbers

**EDA and audit**
- 5.6% of S1 are singletons.
- Each S2/S3 record matches at most one S1.
- Identical inputs always get identical labels.
- The main hard case is neighbour decoys: the house number moved and a word such as Holdings or Northside was added.
- Removing "noisy" training labels made a held-out score **worse** (−1.59 / −0.46 / −0.01 points), so the pipeline keeps every label.

**Retrieval on the full train set**
- 98.55% of true pairs are among the candidates, with 52.6 candidates per S1.
- Ceiling macro F0.5: 0.9953.
