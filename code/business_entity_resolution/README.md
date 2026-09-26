# Business Entity Resolution: reproducible pipeline

Produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data,
using only the provided files.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

The code expects the challenge folder layout: `dataset/student_resource/dataset/{train,test}/*.tsv`,
three levels above `src/`. Paths can be overridden:

| Variable | Default | Meaning |
|---|---|---|
| `BER_DATA` | `<repo>/dataset/student_resource/dataset` | folder with `train/` and `test/` |
| `BER_WORK` | `<repo>/work` | intermediate files (about 40 GB at full scale) |
| `BER_OUT` | `<repo>/output` | the two submission files |
| `BER_NPROC` | all cores | worker processes / threads |

## Run

```bash
bash run_all.sh A        # A = keep every training label (the submitted configuration)
```

Stages (each writes to `BER_WORK/<split>/`):

| Script | Does | Output |
|---|---|---|
| `src/prep.py <split>` | normalises every record (romanisation, sound keys, address parsing) | `s1.arrow`, `rec.arrow`, `labels.npz` |
| `src/retrieve.py <split>` | TF-IDF top-K candidates in both directions, per country | `cands.npz` |
| `src/features.py <split>` | pair features | `feats.parquet` |
| `src/model.py train <V>` | word evidence, stage-1 and stage-2 LightGBM (cross-fitted), decoder choice | `work/models/<V>/` + `report.json` |
| `src/model.py test <V>` | test predictions and decoded matches | `test/pred_<V>.npz` |
| `src/submit.py <V>` | writes both TSVs, prints per-country checks, runs the validator | `output/*.tsv` |

Data-handling variants (`<V>`): A keep all labels · B drop pattern-minority pairs · C drop mixed
patterns · D drop pairs contradicted by the out-of-fold model · E downsample easy pairs ·
F drop conflicting empty-address groups · G down-weight contradicted pairs. See
`docs/METHODOLOGY.md`, section 3.

## Licenses

LightGBM (MIT) is the only model. Libraries: numpy, pandas, scipy, scikit-learn, numba (BSD),
pyarrow (Apache-2.0), rapidfuzz, jellyfish (MIT), anyascii (ISC). No external data or services.
