# Best score: tuned strategy C

The code and exact settings behind our best submission file
(`output_C_tuned/matching_results.tsv`).

## Score

Out-of-fold macro F0.5 on the training data. Each half of the S1 businesses is predicted by the model trained on the other half, and scored against untouched labels.

| Model | Macro F0.5 |
|---|---|
| **Tuned C (this folder)** | **0.98331** |
| C, fast mode | 0.98265 |
| Ensemble C + G + E + tuned C | 0.98320 |
| A, keep all labels (first submission) | 0.98217 |

- **Candidate recall:** 98.55% of true pairs retrieved (116.1M train pairs); ceiling macro F0.5 0.9952.
- **Test checks:** the validator PASSes. Predicted singletons are 5.2 / 5.8 / 5.7% and matches per S1 are 3.38 / 3.33 / 3.37 (France / India / US).

## What the model is

| Stage | Setting |
|---|---|
| Normalisation | romanisation of any script, sound keys (Maa = Ma = माँ), Soundex / Metaphone / NYSIIS, area words learned per country without labels, generic abbreviation rule |
| Candidates | TF-IDF over typed tokens, top 10 S1 per record (25 without address) + top 10 records per S1 |
| Stage 1 | LightGBM, trained on **all labels**, 16M rows per half (stratified negatives), learning rate 0.1, early stopping (≈ 360–400 rounds) |
| Self-correction | competition, S1 crowding, twin (other source) and copy features from stage 1 |
| Stage 2 | **Strategy C:** training rows in patterns whose minority label exceeds 5% are dropped (test is never cleaned). LightGBM with `learning_rate 0.05, num_leaves 255, min_data_in_leaf 100, lambda_l2 10`, up to 3000 rounds with early stopping after 100 rounds without improvement (`S2_GRID[5]` in `src/model.py`) |
| Decoder | each record goes to one S1 only; per S1, the set with the best expected F0.5 (λ = 0.4) |

The other stage-2 settings tried scored 0.98316–0.98330, so this one is best by a small margin.

## Reproduce

```bash
pip install -r requirements.txt
BER_DATA=<dataset dir with train/ and test/> BER_WORK=<scratch dir> BER_OUT=<output dir> bash reproduce_best.sh
```

On a 56-vCPU / 224 GB VM this takes about 2.5 hours: roughly 1 hour of preprocessing, 45 minutes of training and 15 minutes for the test predictions.
