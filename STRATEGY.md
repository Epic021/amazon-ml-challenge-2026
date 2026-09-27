# Amazon ML Challenge 2026: Business Entity Resolution. Strategy

**Status (Sep 26, ~20:30 IST)**

| | |
|---|---|
| **Best LB** | **0.983**: STACK-B5 (stage-2 stacker over our B5 model + the teammate's model). Holdout 0.98792 |
| **LB history** | B1 0.952 → B2 0.970 → B3 0.970 → BLEND-B4 0.979 → **STACK-B5 0.983** (teammate alone: 0.975) |
| **Where the LB is lost** | **France ≈ 0.950** (implied). India ≈ 0.987, US ≈ 0.989 |
| **Awaiting LB** | Stacker v2 (holdout 0.98830) + two France probes (`submissions/`) |
| **Deadline** | around early Sep 28 IST (verify on the portal). Freeze ~Sep 27 evening (§7) |

**Where things live:**
- Code: `src/`, `scripts/`.
- The repo is private, so code reaches the VM via `git archive HEAD src scripts requirements.txt | ssh vm tar -x` (n2-highmem-64, `/work/amazon-ml-challenge-2026`).
- Supporting docs: [eda/report.html](eda/report.html) (EDA), [bottle_necks.md](bottle_necks.md) (design options), [research_sota.md](research_sota.md) (literature).

---

## 1. The problem

For each **Source-1 (S1)** business, output the set of **S2/S3** records that are the same business. The set may be empty.

| | Train | Test |
|---|---|---|
| S1 | 2.21M (US 1.32M, India 0.88M) | 1.73M (India 47%, US 38%, **France 15%; no French labels anywhere**) |
| S2 + S3 | 10.3M | 9.97M |

**Metric:** F0.5 per S1, macro-averaged.
- Per entity: `F = 1.25·TP / (|pred| + 0.25·|true|)`.
- An empty prediction for an S1 with no match scores 1.
- **A wrong match costs ~2.7× a missed one**, so precision matters most.

**Rules that shape the solution:**
- No external data or lookups.
- Models must be MIT/Apache and ≤ 8B parameters.
- `candidate_pairs.tsv` must be exactly the scored set; it is audited for recall and reduction.
- The organizers **re-run the code**, so nothing may be tuned on this test file or on the LB.

## 2. Data facts the design rests on

1. **Hard negatives are constructed "neighbour businesses".**
   - They copy an S1 address, change the house number (88% of the time) and add a business-type or place word (Holdings, Midtown; in French Groupe, Développement).
   - In true matches the house number is equal 65% of the time, with typos in 10%.
2. **Each S2/S3 record belongs to at most one S1.**
   - Matches never cross countries.
   - An S1 has 3.46 matches on average; 5.6% have none.
3. **Source noise:**
   - S2 is uppercase registry style, with 9% of names in Indian scripts.
   - S3 has typos, native-script state names and `NULL` tokens.
   - About 3% of addresses are empty.
   - About 3% of matches are renamed ("Ectokor").
4. **France follows the same generator**, with French legal forms (SARL, SAS, EURL), region ↔ department swaps, and ~15 cities, so **streets are very dense**.
   - French names are short and generic: `{token} {generic word} {legal form}`, e.g. "Horizon Amis SARL".
5. **The hardest class is the same address and house number with one name word swapped** ("Horizon Amis SARL" vs "Horizon Club SARL").
   - Train: 43% (India) / 48% (US) of these are true matches, and the model separates them almost perfectly (AUC 0.9998).
   - Test pairs per S1: **France 4.3**, India 2.1, US 0.95.
   - In France **both models score these low (≈ 0.31 vs ≈ 0.70 elsewhere)**, and the LB says France still over-accepts them (§5, STACK-B5).
6. **No leakage** (IDs and row order carry no signal) and **no label shift** (Σp per S1 on test ≈ the holdout's).

## 3. Current pipeline (what produced LB 0.983)

```
raw TSV → normalize → 4 retrievers → union + prune (~70 cand/S1) → 95 features → LightGBM (stage 1, "B5")
        ┐
teammate's model probabilities (OOF on train, full-fit on test) ─┤
        └→ stage 2 stacker (both p + competition features) → isotonic → soft exclusivity → threshold 0.65 → TSVs
```

| Stage | File | What it does |
|---|---|---|
| Parquet | `scripts/tsv_to_parquet.py` | Raw TSV → parquet |
| Normalize | `src/normalize.py` | ftfy, NFKC, anyascii (Indian scripts → Latin), abbreviation and state maps, ordinals, number extraction, `name_core` (legal forms and honorifics removed) |
| Retrieval | `src/block.py`, `src/block_numaddr.py` | TF-IDF + sparse_dot_topn, per country, top-k in both directions:<br>– **word**: name words, consonant skeletons, address words<br>– **namechar**: char 3-grams of `name_core`, S2/S3 → S1<br>– **addr**: address words only<br>– **numaddr**: `number\|token` address keys, top-50 then re-ranked 0.5·key + 0.5·name similarity |
| Union + prune | `src/union.py` | Every retriever's score and ranks kept as features; keep rank_q ≤ 10 or rank_s ≤ 15. Recall 0.9769 of true pairs; **oracle ceiling F0.5 0.9917** |
| Equivalences | `src/mine_equiv.py` | Mined address/name token equivalences |
| Features | `src/features.py` | 95 features (§4.2). Polars windows plus a persistent multiprocessing pool; part files written in a background thread |
| Stage 1 | `src/train.py` | LightGBM binary, lr 0.05, 255 leaves, trained on folds 2–4, early stopping on fold 5 |
| Stage 2 | `src/stage2.py --friend` | LightGBM over folds 5–9 pairs:<br>– inputs: ours p, teammate p, their mean and difference, missing flags;<br>– competition features on each of the three p (record's best p with another S1, margin, share of S1 mass, rank in S1, twins in p-space);<br>– plus 37 stage-1 features.<br>Trained on folds 6–7, early stop on 5 |
| Decode | `src/decode.py` | Isotonic calibration (folds 8–9) → soft exclusivity p′ = o/(1+Σo) over each record's S1s → threshold chosen on fold 5 |
| Checks | organizers' validator `--check-ids`, `scripts/france_diag.py`, `scripts/france_gap.py` | Format and ids; France decoy proxy; France expected-F and disagreement analysis |
| One command | `scripts/run_stack.sh b5 data/feat_b5 friend 6,7 8,9 b5sb` | Stage 2 → decode → validator → France diagnostic |

## 4. Approaches tried (all of them, with outcome)

### 4.1 Retrieval / candidate generation

| Approach | Description | Result | Verdict |
|---|---|---|---|
| Exact key (B0) | Match on name tokens + first house number | Holdout 0.587 | Baseline only |
| **word retriever** | TF-IDF over name words + consonant skeletons + address words; top-k both directions (S1→record, record→S1) | B1 ceiling 0.9768 | ✓ core |
| **namechar retriever** | Char 3-grams of `name_core`, S2/S3 → S1 only; catches typos and glued words | With addr: ceiling 0.9768 → 0.9867 (B2) | ✓ |
| **addr retriever** | Address words only; catches renamed businesses | (with namechar, above) | ✓ |
| Embedding retriever (RET-EMB) | model2vec potion-multilingual on raw names, targeting Indian-script misses | Only 14% recall on its target pairs, 0.16% of true pairs found only by it, India ceiling +0.0021, ~3 h test cost | ✗ dropped (below the +0.003 gate) |
| **numaddr retriever (RET-BLOCK)** | Blocking keys `number\|token` from addresses; TF-IDF top-50, re-ranked by name similarity | Pair recall 0.9655 → 0.9769; **ceiling 0.9867 → 0.9917**; India +0.011 | ✓ (B4) |
| Rank pruning | Keep rank_q ≤ 10 or rank_s ≤ 15 of the union | Bounded candidate set with little recall loss | ✓ |
| uroman transliteration | Romanize scripts before retrieval | No gain; licence unclear | ✗ |
| Wide union + learned reranker (RET-B) | Top-40 from a wide union | Not built: the model is at 99% of its ceiling | ✗ low priority |

### 4.2 Features (stage 1)

| Approach | Description | Result | Verdict |
|---|---|---|---|
| House-number relation | equal / added / dropped / partial / all differ; edit distance, prefix, first-number log diff | Core decoy signal (decoys change the number 88% of the time) | ✓ |
| String similarities | rapidfuzz ratio / token_set / token_sort / partial / Jaro-Winkler on name, core, skeleton, address | Core | ✓ |
| Label word log-odds | For words added or dropped between S1 and candidate: log-odds of match, learned on folds 0–1 labels | +0.001 (B2 vs B2-noLO) | ✓ |
| **Label-free word roles (G3)** | Per country, from unlabeled top-1 pairs: how often a word is "added" vs present in S1 names; learns decoy vocabulary without labels (US: southside, midtown; France: participations, développement, groupe) | France decoy-signature share 2.08% → 0.16%; implied France 0.88 → 0.94 | ✓ key France win |
| Rank context | Gap to S1's best, gap to record's best, margin, mutual best, candidate counts | Core | ✓ |
| Mined equivalences | Token equivalences mined from data | Small | ✓ |
| Name uniqueness (G5) | How many S1 / records share this core name | +0.0005 (B3) | ✓ kept, below the bar |
| Sound-alike key | Phonetic key (digraphs, nasal m→n, c/q→k, vowels dropped): praivet ≈ private | Pilot +0.00085 | ✓ (B5) |
| Address-word log-odds | Label log-odds of added/dropped address words | Pilot +0.00079 | ✓ (B5) |
| Twins | Other-source candidates of the same S1 with the same number set / core name | Pilot +0.0003 | ✓ (B5) |
| Finer number relations | Suffix, concatenation, gap ≤ 10 | Pilot +0.00003 | ✗ dropped |
| Evidence dropout | Randomly blank strong features in training (for robustness in France) | Pilot −0.00044 | ✗ dropped |
| **B5 = the three pilot winners at full scale** | – | **Only +0.0002** (pilot said +0.001) | Kept, but it taught us that pilots overstate: gate raised to +0.002 |
| More training data | Learning curve 10% / 30% / 50% | 0.97472 → 0.97551 → 0.97574 (flat) | ✗ B3-big cancelled |

### 4.3 Model, decoding and ensembling

| Approach | Description | Result | Verdict |
|---|---|---|---|
| LightGBM binary | 255 leaves, lr 0.05, early stopping | Reaches 99.1% of the candidate ceiling | ✓ |
| Isotonic calibration | On folds 6–9 (stage 2: 8–9) | Needed for blending and thresholds | ✓ |
| Hard vs soft exclusivity | Hard: each record to its best S1. Soft: p′ = o/(1+Σo) | Soft +0.0008 | ✓ soft |
| Exact expected-F DP (`efdp.py`) | Per-S1 set choice maximizing expected F0.5; verified against brute force | Does not beat a threshold (candidates are not independent) | ✗ |
| **Fixed blend with teammate** | Isotonic each model, p = 0.5·ours + 0.5·theirs, soft excl, thr 0.45; w and thr tuned on folds 6–9 | Holdout 0.98487 (+0.0029 vs ours, +0.0025 vs theirs); **LB 0.979** | ✓ |
| B5 in the fixed blend | Same, with B5 | Identical to B4 (0.98434 vs 0.98433 at w = 0.3) | ✗ |
| **Stacked blend (stage 2)** | Learn *when* to trust which model: both p, competition features on each, pair features | Holdout **0.98792** (+0.0031); **LB 0.983**; implied France 0.94 → **0.950** | ✓ **current best** |
| Teammate v2 probabilities | Refreshed teammate model: corr 0.99 with v1, 1.5% France flips | Fixed blend with B4: 0.98519 vs 0.98487 (+0.0003) | Used in stacker v2 |
| Stacker v2 | Teammate v2 + 3 training folds + all 131 features, calibrated on fold 9 | Holdout **0.98830** (India 0.9870, US 0.9892), +0.0004 vs v1 | Below the bar; the stacker is saturated. Submitted anyway (unlimited submissions) |

### 4.4 France investigations (no French labels; the LB is the only oracle)

| Investigation | Method | Finding |
|---|---|---|
| Implied France | `LB − 0.468·India − 0.383·US` over 0.15 | B1 0.88 → B2 0.94 → BLEND-B4 0.94 → **STACK-B5 0.950** |
| Decoy proxy (`france_diag.py`) | Share of predicted matches with a changed number + near-identical address + added word | France only 0.2–0.4%, so classic decoys are handled |
| Expected F (`france_gap.py`) | Monte Carlo F0.5 from calibrated p, validated on the holdout (within 0.002 for the blend) | Model believes France ≈ 0.967; the LB says ≈ 0.94 → **confident errors** |
| Borderline pairs | Pairs with 0.2 ≤ p ≤ 0.8 per S1 | France 0.32/S1 vs 0.11–0.14 |
| Ours vs teammate | Disagreement rate and who is right (holdout) | 2× more disagreement in France; each model confidently wrong in opposite directions |
| Word-swap class | Same address + number, one word swapped, sole claimant | France model p ≈ 0.31 vs ≈ 0.70 in India/US (train truth there ≈ 70%) |
| SHAP (`france_shap.py`) | Mean SHAP France vs India/US on that class | Blamed retrieval-rank features (rank_q_word France 53 vs 15/10: crowded French streets) |
| Pilot without rank features (`france_pilot.py`) | V0 all / V1 − per-retriever ranks / V2 − absolute scores | **Failed:** −0.001 holdout, France swap p unchanged (0.335 → 0.331); the model rejects via name similarity instead |
| French pseudo-label word log-odds (3h) | Word statistics from pairs both models agree on | **Not built:** in train the log-odds barely matter even on the swap class (AUC 0.9998 vs 0.9997 without them) |
| Error analysis of STACK-B5 (holdout) | Where the 0.0121 goes | True pairs: 1.0% never retrieved (India 1.5%), 1.9% rejected; 0.31% of predictions wrong. **Two-thirds of the loss is recall.** 63–74% of rejected true pairs have an EMPTY address |
| Empty-address records | Train: 97.7% of empty-address records are matched (vs 73%) | Never decoys; only *which* S1 is uncertain. Name unique among S1 → 0.978 true (model 0.979); shared by 2–3 S1 → 0.407 (model 0.396): **calibrated coin flips, irreducible** |
| Source-quota tie-break | Does the S1 already have a confident match from the same source? | Weak (0.52 vs 0.37); the stacker already has most of it |
| Shared name differences (`diff_support.py`) | Same added/dropped words in an independent other-source record | US swap class 0.10 vs 0.58 true, India no signal; overlaps with twins |
| Consensus (`consensus_test.py`) | Do the S1's confident matches contain the candidate's added words? | 0.995 true when yes, but the models already know (0.995); France almost never fires |
| **Leave-one-country-out (`loco_pilot.py`)** | Train on one country, score the other: an offline France proxy (US→India loses 0.04, like France) | **No feature group hurts transfer**: removing ranks, name counts, unknown-word counts or roles is neutral or worse (roles −0.014). Retrain without features cancelled |
| Unseen-country threshold (LOCO curves) | Best threshold in-country vs cross-country | No consistent shift (US-model: 0.75 → 0.65–0.75; India-model: 0.65 → 0.55): **no principled stricter threshold for unseen countries** |
| LB probes (`country_thr.py`, diagnostic only) | The 0.983 file with only France re-thresholded at 0.75 / 0.85 (control at 0.65 is byte-identical) | ⏳ LB. LB change ÷ 0.15 = France change. Not to be shipped as a tuned knob |
| **Direction test via STACK-B5** | Stacker drops 0.135 France preds/S1 (6× India/US), 29k of 35k are same-number word swaps | LB 0.983 → France +0.009: **France errs by over-accepting** those swaps |

### 4.5 Infrastructure and process

| Approach | Result |
|---|---|
| Polars rewrite of union/context features; persistent pool; async part files | B4 full pipeline 2 h 21 min on 64 vCPU |
| Smoke test (`scripts/smoke.sh`, ~1% data, `/tmp`) | Mandatory before any full run, after bugs found mid-run |
| Evidence gates (§5) | Added after B3-big was launched on an assumption |
| Background chains with logs + summaries | Every run writes `logs/<tag>_summary.txt` (holdout, validator, France proxy) |

## 5. Validation and decision rules

**Folds** (`int(id[3:]) % 10`): 0–1 word statistics · 2–4 stage-1 training · **5 holdout (the score we trust)** · 6–9 calibration, blend tuning and stage-2 training.

**Reading the LB:**
- `LB ≈ 0.468·India + 0.383·US + 0.15·France`, with India and US taken from the holdout. That gives the implied France score.
- The LB shows 3 decimals, so a change needs ≈ +0.001 to be visible.

**Keep a change only if** the holdout improves by **≥ +0.001** (strict; B3 lesson). Prioritize changes worth ≥ +0.003.

**Evidence gates: nothing runs at full scale without a cheap measurement.**

| Change | Cheap evidence | Gate |
|---|---|---|
| Retriever | Realistic sample → train union ceiling | Ceiling ≥ +0.003 |
| Features | 10% pilot, small LightGBM, with vs without | **≥ +0.002** (pilots overstate: B5 went +0.001 → +0.0002) |
| More data | Learning curve on 1 / 2 / 3 folds | Still rising |
| France change | Offline evidence (pilot / diagnostic) first | Then one LB submission; **never tune a France-specific knob on the LB** |

## 6. Plan (in order)

| # | Item | Why | Time | Keep if |
|---|---|---|---|---|
| 1 | **Stacker v2** (running) | More stacker training data + teammate v2 + all features | ~1.5 h | Holdout ≥ 0.98892 → submit |
| 2 | **Domain-shift features in stage 2**: how unusual a pair's features are vs train (e.g. density of same-street records, name length, share of generic words), model disagreement \|p_o − p_t\| | France over-accepts where evidence does not transfer; let the stacker learn caution from shift signals rather than a tuned France knob | 1.5 h | Holdout ≥ −0.0005 **and** France preds/S1 ↓ on swaps; then one LB |
| 3 | G4 house-number roles (locality cardinality, match levels) | France's dense streets | 2 h | ≥ +0.001 or France LB ↑ |
| 4 | G1 normalize v2 (structural rules, no hand-written maps) | Code audit / generality | 2 h | Holdout ≥ current − 0.001 |

**Final 12 h (from ~Sep 27 evening), no new features:**
1. Freeze.
2. **One-command regeneration from raw data**, incl. the **teammate's model** (their code + run command must be in the package, because stage 2 needs their probabilities). Output must match the submitted file.
3. Validator with `--check-ids`.
4. Zip: `output/` + `code/business_entity_resolution/{src,README.md,requirements.txt}` + methodology doc. Resolve the folder clash with the teammate.
5. Upload ≥ 3 h before the deadline.

## 7. How to run (VM, repo root)

Setup:
```
source /work/venv/bin/activate
export PYTHONUNBUFFERED=1
```

**Always `bash scripts/smoke.sh` first** (~6 min). Full runs unset `BER_DATA` / `BER_OUT`.

| # | Step | Command | Time (full) |
|---|---|---|---|
| 0 | TSV → Parquet | `python scripts/tsv_to_parquet.py` | 2 min |
| 1 | Normalize | `python src/normalize.py` | 3 min |
| 2 | Retrievers | `python src/block.py --split S --retriever {word,namechar,addr}`; `python src/block_numaddr.py --split S` | ~25 / 20 / 12 / 15 min |
| 3 | Union (+ ceiling on train) | `python src/union.py --split S` | ~5 min |
| 4 | Equivalences | `python src/mine_equiv.py --split S` | 2 min |
| 5 | Features | `python src/features.py --split S --out data/feat_b5` | ~85 min train |
| 6 | Stage 1 | `python src/train.py --tag b5 --feat_dir data/feat_b5 --drop numx_` | ~80 min |
| 7 | Stage 2 + decode + checks | `bash scripts/run_stack.sh b5 data/feat_b5 friend 6,7 8,9 b5sb` (teammate probs in `data/friend/`) | ~50 min |
| – | Single model decode | `BER_OUT=output_T python src/decode.py --tag T` | ~15 min |
| – | Fixed blend | `python scripts/blend.py --tag T [--friend_dir D] --write --out output_X` | ~25 min |
| – | France tools | `scripts/france_gap.py`, `scripts/france_shap.py`, `scripts/france_pilot.py` | 5–30 min |

## 8. Rules

- **Process:**
  - Smoke test before full runs.
  - Pass the evidence gate before scaling up.
  - One change per run (stacker v2 bundles three, accepted for time).
  - Log every run in §4/§5 and update §6 **before** reordering.
  - Launch VM jobs from `.sh` files with logs, never `pkill -f` in ssh one-liners.
  - Stop the VM when idle.
- **Compliance:**
  - No external lookups.
  - MIT/Apache models only (LightGBM, model2vec). Not allowed: jina-v3 (CC BY-NC), Unidecode (GPL).
  - Per-country tables are recomputed from the input data at run time.
  - Anything learned from unlabeled test inputs (word roles) is disclosed in the methodology doc.
  - No knob tuned on the LB.
- **Team:**
  - Don't run the teammate's pipeline on our VM (we use their probability files).
  - No Claude attribution in commits.

## 9. Team

| Person | Owns |
|---|---|
| P1 | Validation, decoding, experiment log, submission sign-off |
| P2 | VM, retrievers, full runs, final reproduction + zip |
| P3 | Features (G4), stage 2 / domain-shift features |
| P4 | G1, methodology doc; teammate model code + probabilities for the package |
