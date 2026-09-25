# Entity Resolution Pipeline: Design Skeleton & Open Decisions

Owner of decisions: team lead ("brain"). Execution: Claude.
Examples are real records from the train/test files or EDA (`eda/`).
Decisions are marked **D1–D20**. Record answers in the Decision Register at the bottom.

---

## Governing principles

1. **Canonicalization only has to be consistent, not correct.** A mapping f is good if
   f(variant A) = f(variant B) for the same thing and different things stay different.
   Whether `st` means Saint or Street doesn't matter, as long as both records map it the same way.
2. **Country is only a partition key.** Code never branches on country values. Anything
   country-specific is learned from the file being processed, or comes from a small language-general list.
3. **Learn from data, recompute at runtime.** Equivalences, optional tokens, IDF and number roles
   are recomputed from the input file on every run, never frozen as saved tables.
4. **Regex splits and extracts; dictionaries map.** Replacements are per-token dict lookups after
   splitting. Never use regex alternation for substitution (it causes partial-match bugs such as `st` in `street` → `streetreet`).
5. **Every rule is tested against a golden edge-case file** (~100 real rows plus expected
   outputs) before any full run.

---

## Stage 0: Data contract and compute

- Map `entity_id` → int32; Parquet; pyarrow strings (object strings on 22M rows waste RAM).
- Bottlenecks:
  - per-record Python loops → multiprocessing on 32 cores;
  - merges on 100M+ pairs → join on int IDs only;
  - rapidfuzz → `process.cpdist(..., workers=-1)`, never a Python loop.
- Trap: TSV must be read with `sep="\t", quoting=QUOTE_NONE, keep_default_na=False, dtype=str`,
  otherwise "NA", "NULL" and "nan" silently become NaN and quotes inside names break rows.

---

## Stage 1: Character canonicalization (text → clean ASCII)

**Required order:** ftfy → NFKC → anyascii → lowercase. No regex runs before this step.

| Edge case | Example | Note |
|---|---|---|
| Mojibake (incl. garbled Indian scripts) | `LÃ©arning`, `à²•à²°à³à²¨à²¾à²Ÿà²•` (broken Kannada for Karnataka) | ftfy before transliteration |
| Native-script names | `राम मार्केटिंग प्राइवेट लिमिटेड`, `माँ कंस्ट्रक्शंस इलेक्ट्रॉनिक्स प्रा. लि.` | Become `praivet`, `pra. li.`: transliterated legal words no English list contains |
| Native-script states | `दिल्ली`, `தமிழ்நாடு`, `उत्तर प्रदेश` | Become `dilli`-like spellings ≠ `delhi` (see Stage 4) |
| Accent noise | `Blúe`, `Fàmilles`, `SÊCTION`, `Bóa`, `Ptit Àmicale` | Folded by anyascii |
| Symbol forms | `N°`, `Nº`, `½` (`19 1/2 STARDUST`), `&` | NFKC turns ½ into `1⁄2` (Unicode fraction slash) |
| Script-specific digits | `०१२` | Python `\d` matches these, but `lstrip("0")` doesn't treat them as zeros |

**Regex hazard:** Python `\w` does NOT match Devanagari vowel signs (Mc/Mn), so it splits words
mid-word. That caused the "456K numeric-only names" artifact. After Stage 1, use only `[a-z0-9]`, never `\w` or `\d`.

- **D1** Transliteration engine: anyascii (ISC, all scripts, crude vowels) vs indic-transliteration (better for Indic scripts). Recommendation: anyascii, with consonant skeletons absorbing the vowel noise.

---

## Stage 2: Tokenization and protected patterns (regex layer)

Protect multi-character patterns before stripping punctuation.

| Pattern | Examples | Risk if stripped first |
|---|---|---|
| Relation markers | `C/o Shiv Kumar`, `S/o Ram Ausan`, `D/o`, `W/o`, French `Chez Lab` | `s/o` → `s o` → `south o`; brings person names into the address |
| Unit and house prefixes | `H.NO`, `No. #31`, `Door No #31`, `##20`, `#6202`, `Gali No-01`, `N° 47`, `Nº 5` | Filler; the useful part is the number after it |
| Slash and compound numbers | `1/239`, `D/94`, `E/1409`, `286/1 & 286/2`, `50-96-2/2`, `S-7`, `KH NO. -570/13` | Keep whole and as parts (D3b) |
| Ranges | `1026-1030 State Route 2835` | Range vs hyphenated ID |
| Alias markers (names) | `aka`, `dba`, `a/k/a`, `t/a`, `fka`, `formerly known as`, `\|` (`... CORPORATION \| www.shivshakti.com`) | Split into name variants (D6) |
| Domains and handles | `heassociates.com`, `prprivate.com`, `@shreefoundation`, `lucknowindiacom` | Remove `www.` and the TLD; joined words stay joined (D7) |
| Brackets | `Lucknow (india) Private`, `Tejjya [Electronics-Private]`, `BRIGHT BOOK (STORE)`, `KC Ecole [S.A.]` | Keep the content, drop the brackets |
| French elision | `de l'Orne` vs `de lOrne`, `D'arcole`, `L'orvasserre` | Token-joining mismatch |
| Null tokens | `NULL`, `N/A`, `nan` (~2.4% of S2/S3) | Match the whole token only (`na` is also a real token) |

- Token joining and splitting: `Rae Bareli` / `Raebareli` / `Raibareilly` (in ONE address),
  `Lege-cap-ferret` / `LÈGE-CAP-FERRET`, `lucknowindiacom` → needs space-insensitive character n-grams (D8).
- Repeated tokens: `VIDYALAYA VIDYALAYA`, `Smt Shriram Shriram Brothers`,
  `Shencottah-627 809 Shencottah-627 809` → dedupe for set features, keep the counts for TF.
- Regex rules:
  - compile every pattern once at module level;
  - no nested quantifiers like `(\w+\s?)+`;
  - word-boundary anchors (`\baka\b`, so "Kaka Foods" isn't split);
  - apply the ASCII flag after Stage 1.
- **D2** Relation markers: drop the marker and the person name that follows, or keep the tokens?

---

## Stage 3a: Name parsing

Output: `name_full`, `name_core` (optional tokens removed), `name_variants[]`, flags.

| Edge case | Example | Decision |
|---|---|---|
| Legal forms and their typos | `L.L.C.`, `Pvt. Ltd`, `Private (Limited)`, `Phiavte Limited`, `Prívate`, `praivet`, `pra. li.` | **D4** exact list vs fuzzy match (edit distance ≤ 2 / skeleton) vs learned optional tokens |
| Honorifics and prefixes | `Smt`, `Sri`, `Shri`, `M/s`, `Mr`, `-- Holloway`, `***` | Strip |
| Number-for-letter typos vs real numbers | `B0a`, `BEAC0N`, `We1fare`, `c0nsultants` vs `3M`, `7-Eleven`, `24 Hour Fitness` | **D5** fix a digit only when it's inside a word (surrounded by letters) |
| Aliases | `Zetaectobelo aka Cozy Massage`, `X dba Y`, `... \| www.site.com` | **D6** compare each variant and take the max |
| Renamed (3.3% of positives) | `Premier Cashew` ↔ `Ectokor` | Rely on the address-only retriever |
| Place or PIN inside the name | `Shencottah-627 809 ... Rubber Public Limited`, `Bordeaux Ecole SARL` | Handled by IDF |
| Very common vocabulary (France) | `sarl`, `club`, `ecole`, `amicale`, `comite` | Low IDF; rely more on the address |

- **D7** Joined-word domains: leave them to character n-grams, or segment them with a word list (needs a dictionary; disclose it)?

---

## Stage 3b: Address parsing (number roles matter most)

Output: `addr_tokens`, `addr_nochar` (spaces removed), typed numbers, locality tokens, flags.

| Number role | Examples | Why it matters |
|---|---|---|
| House or door number | `1795 Westchester`, `#31`, `26 Bis`, `6023-C`, `13741d` | Main signal for rejecting neighbours (differs in 88% of hard negatives; same in 65% of positives) |
| Unit, floor or flat | `Flat 205`, `Fl 0`, `Unit BLDG 16`, `Suite 160`, `1St Floor`, `E/1409` | Partial evidence, often dropped |
| Street ordinal | `4th Street`, `Fifteenth Terrace`, `3Rd Cross`, `22Nd Main 9Th Block` | Part of the street name, not a house number |
| Road or zone ID | `Hwy 65`, `Highway 412`, `State Route 2835`, `Nh-12`, `Sector-1`, `Sector 16B`, `80Feet Rd` | A shared road number looks like a shared house number but isn't |
| Postal code | `Vellore-632 009` (split 3+3), `Raut Ni 111 787`, `44800 Saint-Herblain`, `Madras 32`, `New Delhi-11` | Area-wide: a false match signal if treated as a house number |
| Noise | `364 3573`, `002699`, `1600` vs `160` | Typos |

- **D3** Number typing, country-agnostic:
  - (a) context-word lists;
  - (b) learned from data: a number in many different-name records within a locality is an area code; for each preceding word, measure from the true pairs how stable the number is;
  - (c) positional: the first number is usually the house number.

  Recommendation: (b) + (c), with a small (a) fallback.
- **D3b** Slash and compound numbers: keep `286/1` whole AND as parts `286`, `1`?
- **D9** `Bis`/`Ter` (France): `26 Bis` ≠ `26`. Keep it as a suffix (`26b`) instead of deleting it.
- Directions `n/s/e/w` → north/...: `Block E` → `block east` (consistent, but `S/o` → south is a bug).
- Admin units (TX↔Texas, Nord↔Hauts-de-France, दिल्ली↔Delhi): learned in Stage 4, not hard-coded.
- Component order is shuffled → only order-insensitive comparisons.
- Empty address (3.3% of S2/S3) → `addr_empty` flag, so it reads as missing evidence rather than a mismatch.

---

## Stage 4: Equivalences learned at runtime (the country-agnostic core)

1. Anchor pairs:
   - train: the true pairs;
   - any unlabeled file (test, France, a new country): exact-key pairs (same core name words + same first
     number + a unique S1 key), the B0 method, which was 98% correct on train.
2. Mine from the anchor pairs:
   - **Token equivalences:** differing tokens that co-occur much more than chance (PMI, minimum support):
     `tx↔texas`, `dilli↔delhi`, `nord↔hauts de france`, `r↔rue`, `chm↔chemin`.
   - **Optional tokens:** tokens often present on one side and absent on the other, across many names →
     legal forms and honorifics in any language (`limited`, `llc`, `sarl`, `eurl`, `praivet`, `pra`, `li`, `smt`).
   - **Number roles** (D3b).
3. Apply, then build the blocking and features.

- **D10** Hand-written vs mined. Recommendation: hand-write only the universal core (null tokens,
  relation markers, digit/letter splitting); mine states, regions, legal forms and abbreviations; keep the current maps as a fallback.
- Traps:
  - anchor pairs are biased toward easy records;
  - PMI on rare tokens is noisy (minimum support 20–50);
  - never mine from pairs whose numbers differ (it would teach "Holdings = nothing").

---

## Stage 5: Blocking (candidate generation)

Goal: recall ceiling ≥ 0.97 at ≤ ~40 candidates per S1, per country partition.
The candidate file is audited for recall and reduction ratio.

| Retriever | Catches | Status |
|---|---|---|
| R1 name+address TF-IDF, S2/S3 → S1 top-k | Most pairs; sharp (each record has ≤ 1 S1) | Built (word + skeleton) |
| R2 same, S1 → S2/S3 top-k | S1 with 5–11 matches | Built |
| R3 address-only character n-grams (spaces removed) | Renamed (3.3%), joining variants | D11 |
| R4 name-only within the same locality token | Empty/partial addresses (1.9–3.3%) | D11 |
| R5 multilingual embeddings (e5, MIT) | Transliteration failures | Optional, D12 |

- **D8** Word + skeleton (current) vs character 3-grams on the space-removed string vs both.
  Recommendation: add character 3-grams if the ceiling is below 0.95.
- **D13** k and cap per direction (currently 10 / 40 / 50): tune on the recall-vs-candidates curve.
- Bottlenecks:
  - dense French cities (~15 cities) → huge buckets; tune `max_df`;
  - memory of the sparse product → per country and in chunks;
  - generic name + common street → many near-ties.
- Measure recall by country × source × script (native-script names reached only 77% at top-50) × empty address.

---

## Stage 6: Pair features (country-agnostic)

| Group | Features |
|---|---|
| Number relationship | same / candidate adds / drops / differs; digit edit distance; prefix or substring (`160`/`1600`); absolute difference; slash-part match; first-number equal; candidate has extra unit detail |
| Name differences | added/dropped token log-odds (out-of-fold); optional-token-only difference; max over variants |
| Similarity | rapidfuzz ratio / token_set / partial / Jaro-Winkler on core name, full name, address; TF-IDF cosine per retriever |
| Context | `rank_q`, `rank_s`, mutual best, gap to #2, number of candidates, number of S1 sharing street + number |
| Flags | empty address, native-script name, alias present, source (S2/S3) |

- Never use `country` or raw locality identities.
- **D14** Token log-odds risk failing on France: train-split only / re-estimate on test anchor pairs / optional tokens only.
  Check with the train-on-one-country, score-on-the-other split.
- Leakage trap: log-odds and target statistics must never be computed from the rows being scored.

---

## Stage 7: Model training

- Training pairs = our own blocking output on train (realistic hard negatives), sampled by S1 entity.
- **D15** Training sample size (30% vs 60%).
- Holdout: 10% of S1 entities. The country-shift split is a separate stress test.
- **D16** LightGBM with extra weight on false positives and/or monotone constraints.
- Isotonic calibration on the holdout (the decoder needs it).

---

## Stage 8: Decoding

1. Exclusive assignment: each S2/S3 record goes to its best S1 only.
2. **D17** Global τ vs per-S1 expected-F0.5 top-m with m = 0 allowed (recommended).
3. **D18** S2↔S3 twin consistency.
4. Label-free test checks per country: no-match rate, mean matches per S1 (true is ~4.1 in test), share of records assigned.

Trap: test has more matches per S1 than train. Don't tighten for test.

---

## Stage 9: Output and reproducibility

- One row per `test_source1` ID (France included); candidates ⊇ matches; validator with `--check-ids`.
- One command: `run.py --train-dir --test-dir --out`; Stage 4 and IDF recomputed on every run.
- **D19** "Testland" robustness test (rename a country in a sample).
- **D20** Time and memory budget for a full run on the VM (32 vCPU / ~250 GB).

---

## Known bugs in current `src/normalize.py` / `src/block.py`

1. `S/o`, `D/o`, `W/o`, `C/o` → `s o` → `south o` (`ADDR_ABBR["s"] = "south"`).
2. State-code / street-type collisions: `ct`→court, `fl`→floor, `mt`→mount.
3. `bis` is deleted, so `26 Bis` = `26`.
4. PIN/ZIP parts and road numbers go into `nums` (`Vellore-632 009` → 632, 9; `Hwy 65` → 65).
5. The number-for-letter fix runs on any mixed token: `3m` → `em`, `4g` → `ag`.
6. `ter_` in `ADDR_ABBR` is a dead key; `bis`/`ter` handling is inconsistent.
7. `block.py:52`: leftover dead expression (harmless).
8. Legal-word typos (`phiavte`, `praivet`, `pra`, `li`) aren't removed from `name_core`.

---

## Decision Register

| ID | Question | Recommendation | Decision |
|---|---|---|---|
| D1 | Transliteration engine | anyascii | |
| D2 | Relation markers: drop marker + person name, or keep | Drop the marker, keep the tokens with low weight | |
| D3 | Number typing method | Learned (b) + positional (c), small list (a) as fallback | |
| D3b | Slash/compound numbers: whole and/or parts | Both | |
| D4 | Legal-form detection | Learned optional tokens + fuzzy fallback | |
| D5 | Number-for-letter fix scope | Digits inside words only | |
| D6 | Alias handling | Variants, max similarity | |
| D7 | Joined-word domains | Character n-grams (no dictionary) | |
| D8 | Blocking representation | Word + skeleton now; add character 3-grams if ceiling < 0.95 | |
| D9 | Bis/Ter | Keep as a number suffix | |
| D10 | Hand-written vs mined maps | Mine, with a small hand-written core | |
| D11 | Address-only and name-in-locality retrievers | Yes | |
| D12 | Embedding retriever | Only if recall on native-script names is still low | |
| D13 | k / cap per direction | Tune on the recall curve | |
| D14 | Token log-odds scope | Train split + re-estimate on test anchor pairs; gate with the country-shift split | |
| D15 | Training sample size | 30%, grow if the learning curve says so | |
| D16 | False-positive weighting / monotone constraints | Try both | |
| D17 | Decoder | Expected-F0.5 top-m, m = 0 allowed | |
| D18 | S2↔S3 twin consistency | Stretch goal | |
| D19 | Testland robustness test | Yes, before freezing | |
| D20 | Full-run time and memory budget | ≤ 1 h per full run | |