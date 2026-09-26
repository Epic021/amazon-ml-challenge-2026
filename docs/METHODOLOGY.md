# Business Entity Resolution — Methodology (v2, built on the label audit)

**Status:** The code is written and has been smoke-tested end to end on a 2% slice of the data. The full training run happens on the GCP VM. Numbers marked *[VM]* get filled in from that run.

**Code:** `code/business_entity_resolution/src/`
- `prep.py` → `retrieve.py` → `features.py` → `model.py` → `submit.py`
- shared helpers: `normalize.py`, `decode.py`

**Inputs:** the analysis this plan is built on:
- the EDA (`eda_output/report.html`);
- the training-label audit (`eda_output/label_audit/REPORT.md`).

---

## 1. What the metric and the data force on the design

| Finding (source) | Consequence for the pipeline |
|---|---|
| **Metric: macro F0.5 per S1, singletons included.** A false match costs about 2.7× a miss. An empty prediction scores 1.0 on a singleton (5.6% of S1). | Decide **sets per S1**, not pairs. "No match" is an explicit option. |
| **Each S2/S3 record matches at most one S1.** Matches never cross countries. (EDA) | Search each country separately. **Exclusive assignment:** each record keeps only its best S1. |
| **Identical inputs get identical labels.** 973,425 of 973,426 exact normalized copies are matches. (Audit, insight 1) | The labels are consistent with the text. Deterministic signals such as exact equality can be trusted. |
| **Same name + empty address is assigned to different S1s.** 7,227 conflicting groups. (Insight 2) | Features: empty address; how common the name is among S1s; how many S1s compete for the record. The decoder may answer "no match". |
| **When only the house number differs, the label depends on how the name changed.** Identical name: 99.5% match. One word added: 8.3%. Legal form only: 50%. (Insights 3, 4) | Name-change class × number-change class × **which words** changed. A tree model learns the interactions. |
| **Cross-script names** (Hindi, Tamil, …) against a Latin S1 behave like Latin names once compared by sound. (Insight 6; the Maa Care case) | Romanize, then compare by **sound keys and phonetic codes** in every script. |
| **Pattern classes alone mislabel 3.5% of pairs**; a model that sees how far the number moved and which words were added reaches AUC 0.9992. (Insights 7, 8) | Give the model the fine-grained evidence: log number gap, edit distance, word log-odds. |
| **The exact blocking keys missed 13.2% of matches.** (Insight 10) | Use fuzzy TF-IDF retrieval in both directions, and measure its recall ceiling. |
| **Removing noisy labels hurts the holdout:** −1.59 / −0.46 / −0.01 points. Downsampling easy pairs costs −0.05. (Audit §4) | **Keep every label.** Re-test the cleaning variants inside the real pipeline (§3). |
| **Test ≠ train:** 5.75 vs 4.68 records per S1, expected ~4.1 matches per S1. France (15% of test S1) has no labels. (EDA) | No set-size cap. Features must not depend on country. Robustness to words the model has never seen (§6). |
| **Cross-source agreement:** when the number changed, a twin in the other source with the same number raises the match rate from 4% to 28% (nearby numbers) and from 40% to 90% (far-off numbers). (EDA `twin_signal.json`) | Self-correction features from twins (§5). |

---

## 2. Normalization: language- and area-agnostic

One function normalizes every record from every source and split, `normalize.py`. **It is used only for matching; the outputs always carry the original S1/S2/S3 IDs.**

**Script and spelling**
1. NFKC, then romanize with `anyascii` for any script.
2. **Word-final nasal marks** (anusvara/chandrabindu, in all Indic scripts) are dropped before romanizing, as they are not written in Latin. So माँ → "ma", not "mam".

**Sound key (`akey`)**
- The consonant skeleton:
  - digraphs ph/sh/ch/kh/gh/th/dh/bh/ck are reduced;
  - letters that sound alike are merged (c/q→k, z/x→s, j→g, w→v, y→a);
  - vowels are dropped and doubled letters collapsed.
- **If the key would be a single letter, one more letter is kept** (a marker for a leading vowel), so short words don't collide:
  - "Maa" = "Ma" = मां → `m`
  - "IT" = आईटी → `*t`
  - "Amy" → `*m`, which stays apart from `m`
- Examples: केयर = care → `kr`; गुड = Good → `gd`; लॉजिस्टिक्स = logistics → `lgstks`.

**Phonetic codes:** Soundex, Metaphone and NYSIIS (`jellyfish`, MIT, about 0.3 µs per word).
- Kozhikode / Kozikode → K223
- Lakshmi / Laxmi → L250
- Thiruvananthapuram / Tiruvanantapuram → T615

**Word equivalence, used in every comparison:** two words count as the same word when they are
- identical;
- a typo pair (1–2 edits); or
- sound-alike (same `akey`, or the same Metaphone for words of 3+ letters).

This works in names and addresses, for any language, and for words never seen before. It is cheaper than running edit distance over all word pairs.

**Area / address: learned per country, nothing hard-coded**
- An address component (the text between commas, with no digits) that occurs in at least **0.1% of a country's records** is an *area* part. It is ignored when two addresses are compared, but still used for retrieval.
- The areas are learned by `prep.py` from that split's own records, **without labels** (`WORK/<split>/areas.json`):
  - **US:** 188 areas (state codes and names, big cities, counties);
  - **India:** 220 (states, cities, big localities);
  - **France:** 24 (Hauts-de-France, Nord, Gironde, Loire-Atlantique, Pas-de-Calais, …, St/Saint-Nazaire), learned from the unlabeled test records.
- **Abbreviations need no list:** a short word equals a longer one when it has the same first letter and its letters appear in the longer word in order. So St = Street = Saint, Rd = Road, Bd = Boulevard, R = Rue. Guard: at most 4 letters, and the long word has at least 2 more letters, so "hotel" ≠ "hostel".
  - On the 2% slice, 27% of true pairs use this rule.
  - Legal forms that differ only in spelling (Ltd = Limited) are flagged as the same legal form; PLLC ↔ Corp is flagged as a different one.
- `N°`, `#` and similar symbols are removed.
- English ordinals are unified (5th = fifth) and floors ("7th floor") are dropped.
- House and plot numbers are kept separately and compared as a **multiset**.
- **Replacing the hand-written state/region lists with learned areas left retrieval recall unchanged** on the 20% slice: 99.31% vs 99.29%.

**Country word in names:** the record's own country label ("India", "France", …) becomes the token `ctry`, so what the model learns about "(India)" applies to "(France)".

**What is still hand-written:** a small multilingual dictionary of legal forms, honorifics and articles; unit and placeholder words (unit, apt, suite, floor, null…); and English ordinal words. No list names a country, region or city. There are no lookups and no external data.

---

## 3. Dataset handling: every label is kept; cleaning is an experiment, not an assumption

**Default (variant A):** train on every label, including the flagged ones. In the audit, cleaning made the held-out score worse. The flagged pairs are the real boundary cases the test set also contains.

**Cleaning variants, re-run inside the full pipeline** (`python3 model.py train <V>`):
- Features are computed once; a variant only changes which training rows are used, or their weights.
- Predictions are always made for the **untouched** other half, so all variants are scored on the same labels.
- The score is the final decoded macro F0.5.

| V | What is removed or changed in training | Why try it |
|---|---|---|
| A | nothing (baseline) | — |
| B | pairs labelled against their pattern's majority | audit: −1.59 (idealized) |
| C | whole patterns with >5% minority label | audit: −0.46 |
| D | pairs the first-stage out-of-fold model contradicts strongly (p<0.02 match / p>0.98 no match) | audit: −0.01 |
| E | easy pairs (p<0.002 or >0.998): keep 10% with weight 10 | audit: −0.05, 30% of rows |
| F | empty-address records whose identical name is labelled inconsistently (Insight 2) | the one exact-duplicate conflict |
| G | **down-weight (×0.2)** instead of removing the pairs in D | a soft version of D |

Rules:
- **The best variant on the untouched out-of-fold score is used for test.**
- Evaluation labels are never cleaned.
- Word evidence is recomputed from each variant's kept rows only.

---

## 4. Candidate generation: `candidate_pairs.tsv`

**Retrieval.** Per country, TF-IDF cosine over **typed tokens**:

| Part of the record | Tokens |
|---|---|
| Name | words · 4-letter prefixes (typos at the end) · `akey` sound keys · Soundex (4+ letters) |
| Address | words · prefixes · sound keys · house numbers · **number × address-word pairs** ("305 + ahmedabad") |

- The number × word pairs keep a record findable when its street is cut away and only a number and a common city remain. That was the most common miss: Indic-script names with short addresses.
- Name and address are normalized separately and mixed 50/50, so a long address can't drown the name.
- Tokens that are too common to identify anything are skipped: those in more than 2% of S1s when searching S1s, and more than 0.5% of records when searching records. A query whose tokens are all common still uses its rarest one.
- A numba inverted-index search takes:
  - the **top-10 S1s per record**, widened to the **top-25 for records with no address** (name-only records tie with many S1s);
  - plus the **top-10 records per S1**;
  - unioned.

**Measured recall** (full scale *[VM]*):

| Train slice | True pairs retrieved | True S1 ranked first | Ceiling F0.5 | Candidates per S1 | Time (16 threads) |
|---|---|---|---|---|---|
| 2% of S1 | 99.84% | 98.9% | 0.9995 | 51 | 5 s |
| 20% of S1 | 99.29% | 97.2% | 0.9977 | 52 | 84 s |

- Denser data means more lookalikes, so full-scale recall will be a little lower again.
- At 20%, the number × word tokens and the "-tion" = "-shan" sound rule (फाउंडेशन = foundation) raised recall from 98.98% to 99.29% and cut the time from 206 s to 84 s.

**Pruning.** The first-stage model's probability `p1 ≥ 0.005` prunes that union. The survivors are exactly what the final model scores, and **that set is `candidate_pairs.tsv`**.
- On the 2% slice, pruning keeps about 3.6 pairs per S1 (from 51), and the ceiling moves only from 0.9995 to 0.9993.

---

## 5. Matching model: two stages with self-correction

**Stage 1: pair features**, about 70 in total. Each group maps to an audit insight:

| Group | Features | Insight |
|---|---|---|
| Name change | 16-class name difference (incl. 6 cross-script classes); counts of added / dropped / typo-or-sound-alike words; legal-form change | 3, 5, 6 |
| Name similarity | raw, token-set, partial and sorted ratios; core-name ratio and Jaro-Winkler; word Jaccard; sound-key, Soundex, Metaphone and NYSIIS Jaccard | 5, 6 |
| Numbers | 8-class number change (same / added / dropped / cut short / one character / nearby / far-off); log gap and edit distance of the closest changed pair; numbers added / dropped; whether the changed one is the house number | 3, 4, 8 (the largest gain) |
| Address | 10-class address difference; word counts; token-set and ratio; phonetic Jaccard; empty flags | 2, 5 |
| Word evidence | smoothed log-odds of each added / dropped name and address word, as max / min / mean / unseen, **computed from other folds only** | 3, 8, 9 |
| Commonness | how many S1s share this core name, and this address | 2 |
| Retrieval | cosine score, rank in both directions, gap to the best candidate, number of candidates | 10 |

**Rules for the stage-1 inputs**
- **`country` is not a feature**, so France is not out of range.
- **Evidence dropout:** 15% of training rows see all their words as "unseen", so the model learns what to do when word evidence is missing. That is what French words look like to it.

**Context: self-correction.** From the first-stage probabilities, every pair gets
- **Competition:** the record's best other S1, the margin to it, how many S1s claim the record, and its rank.
- **Crowding:** the S1's other candidates (sum, count of strong ones, rank).
- **Twins:** the best p1 of a record in the *other source*, under the same S1, with the **same house numbers** or the **same name**. When "4" is clearly S1-B's, its twin "3" is pulled to B and away from A.
- **Copies:** the best p1 of an identical (name + address + numbers) record under the same S1.

**Stage 2** re-scores the pruned pairs with the pair features plus the context. On the slice, the margin to the record's best other S1 is already the most used feature.

**Model and training**
- LightGBM (MIT); binary; 255 leaves; learning rate 0.1; early stopping on 2% of S1 groups.
- **Cross-fitted:** the training S1s are split in two halves (hash of the S1 ID). The model trained on one half predicts the other, so every training pair gets an out-of-fold probability. Test gets the average of both halves' models.

**Decoding**
1. **Exclusive assignment:** each record keeps only its most probable S1.
2. Per S1, choose the set size *m* that **maximizes expected F0.5**. The chosen and unchosen candidates are treated as Poisson-binomial counts, plus a small Poisson term λ for matches retrieval missed. *m = 0* means "no match" and scores P(no true match), which protects singletons.
3. λ, or a plain threshold if it scores better, is chosen on the train out-of-fold score.

---

## 6. France (test only, no labels)

The same code path runs for France, and **nothing in the code is specific to it**:
- its areas (regions, departments, cities) are learned from its own unlabeled records;
- abbreviations use the general rule;
- `(France)` → `ctry`, from the country label;
- SARL/SAS/… are in the multilingual legal-form list;
- evidence dropout covers its unseen words;
- no feature depends on country.

**Checked after prediction, per country** (`submit.py`): predicted singleton rate, matches per S1 and share of records assigned. If France is far from US/India, France is broken.

---

## 7. Validation

- **Primary number:** macro F0.5 over **all** training S1s from out-of-fold predictions, after exclusive assignment and decoding.
  - Split by country, and by singleton vs non-singleton.
  - True sets are the full ground truth, so retrieval misses count against us.
- **Also reported:** the retrieval ceiling, the ceiling after pruning, stage AUCs, and the decoder grid (both halves separately, as a stability check).
- **Public leaderboard:** used as a check on direction only.

---

## 8. Compute (GCP) and time plan

The whole pipeline is CPU and RAM bound (LightGBM, rapidfuzz, numba), and **needs no GPU**.

| | |
|---|---|
| **VM** | `n2-standard-32` (32 vCPU, 128 GB), ~$1.6/h; 150 GB pd-balanced, Ubuntu 24.04 |
| **Fallback** | 16 vCPU / 128 GB (`n2-highmem-16`), or 8 vCPU / 64 GB on the free trial (≈3–4× slower) |
| **GPU** | not needed. An optional L4 (`g2-standard-32`) only for a later cross-encoder rerank of uncertain pairs |
| **Estimated run (32 vCPU)** | prep ≈ 3 min / split · retrieval *[VM]* · features ≈ 10 min / split (≈170 µs per pair) · training ≈ 30–40 min · test ≈ 10 min |
| **Total** | first submission ≈ 1.5–2 h after the data is on the VM; each cleaning variant ≈ 20–40 min |
| **Budget** | about $30–60 of the $200 |

**Order of work**
1. Variant A → first leaderboard submission.
2. The cleaning variants B–G, compared on the out-of-fold score.
3. Tune retrieval K, prune threshold and the decoder.
4. Final: best variant → test → validator → submit, then fill the documentation template.

---

## 9. Licenses and fair play

**Model:** LightGBM (MIT).

**Libraries:**

| Library | License |
|---|---|
| numpy, pandas, scipy, scikit-learn | BSD |
| pyarrow | Apache-2.0 |
| numba | BSD |
| rapidfuzz | MIT |
| jellyfish | MIT |
| anyascii | ISC |

**Data:** no external data, APIs, geocoding or gazetteers. The normalization lists are hand-written general knowledge. Test inputs are used only unlabeled, for retrieval statistics and IDF.
