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
