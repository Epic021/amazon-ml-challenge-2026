# Training-Set Label Audit and Cleaning Experiment

| | |
|---|---|
| Data used | Training files only: `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`. No test file is read anywhere in this work. |
| Code | `eda_output/label_audit/`: `build_pairs.py`, `analyze.py`, `wordfeats.py`, `curation_experiment.py`, `examples.py`, `build_markdown.py`, `build_report.py` |
| Generated from | `audit.json`, `curation_experiment.json`, `examples.json`, `pattern_summary.tsv`. This file is written by `build_markdown.py`, so every number and example below comes from those files. |
| HTML version | [Train Label Audit](https://claude.ai/artifact/LrPyzLZS588ZsyPGe127yE) (private; share it from the page's Share menu) |

Throughout, **match** means the ground truth lists the record under that S1 business, and **no match** means it does not. **Model p** is the out-of-fold probability of a match from the lens-3 model (section 2.4).

---

## Summary

- **Identical inputs get identical labels.** 43,910 groups of records with identical raw text contain **0** label conflicts. Of 973,426 pairs whose normalised name and address equal the S1 record's, **1** is not a match.
- **The one identical input that gets different labels is a name with no address.** 7,227 groups of records with the same name and an empty address are matched to different S1 businesses, or matched and unmatched.
- **Conflicts sit where something differs, mostly where only the numbers differ.** That case covers 2,653,711 pairs with a 46.9% match rate. Whether such a pair is a match depends mostly on how the name changed: 99.46% when the name is identical, much lower when a word is added.
- **Across all 2,121 observed difference patterns,** 1,575 contain both labels, and 299,356 pairs (**3.49%** of those compared) carry the minority label of their pattern.
- **Names in Indian scripts are now compared by sound**, so a word replaced across scripts is visible (the Maa Hospitality vs "मां केयर" case).
- **A model that also sees how the number changed and which words were added separates the labels:** out-of-fold AUC **0.9992**. It contradicts **5,388 labels (0.06%)** with high confidence, and 1.25% of pairs stay uncertain (p between 0.2 and 0.8).
- **Removing conflicting pairs from training makes a held-out score worse:**

  | Change to the training data | Change in holdout macro F0.5 |
  |---|---|
  | Remove pairs labelled against their pattern | -1.59 points |
  | Remove whole mixed patterns | -0.46 points |
  | Remove the pairs a model contradicts | -0.01 points |
  | Keep all hard pairs plus a weighted 10% of easy pairs | -0.05 points, with 30% of the rows |

  **Decision:** keep every label, and build the model to capture the differences described in [section 5](#5-what-the-model-must-capture).

---

## 1. Why this audit exists

Browsing the training data in the explorer turned up cases that looked inconsistent:

- the same street with only the house or plot number changed, sometimes labelled a match and sometimes not;
- the same address with a slightly different name, sometimes matched and sometimes not;
- differences that look like typos, sometimes matched and sometimes not;
- names written in an Indian script next to Latin S1 names.

The request was to flag every such case across the whole training set, not only those kinds. The audit reports what it measured. It does not claim to know which of two conflicting labels is the correct one.

---

## 2. How the audit works

### 2.1 The unit of analysis is a pair

The ground truth labels pairs of an S1 business and an S2/S3 record. Every S2/S3 training record becomes exactly one pair:

| Record | Paired with | Label |
|---|---|---|
| A matched record | Its ground-truth S1 business | match |
| An unmatched record | The single most similar S1 business among those sharing at least one *blocking key* with it | no match |

**The six blocking keys:**
1. the set of address words, with house numbers ignored;
2. the set of core name words, i.e. without legal forms and fillers;
3. key 1 with every word cut to 4 letters;
4. key 2 with every word cut to 4 letters;
5. address words of 4 or more letters only, which ignores stray letters and short tokens such as the `C` in `#C-242`;
6. the sound-alike forms of the core name words (section 2.2), which lets a romanised Hindi name meet its Latin spelling.

A key is used only when its joined text is at least 6 characters (address keys) or 4 characters (name keys). It is skipped when more than 30 S1 businesses share it.

**Choosing the lookalike.** An unmatched record is paired with the candidate that scores highest on `token_set_ratio(name) + token_set_ratio(address)`, using rapidfuzz on romanised, lower-cased text.

**Comparable pairs.** A matched pair is *comparable* when it also shares at least one key with its S1. Unmatched pairs are comparable by construction. Only comparable pairs enter lenses 2 and 3, so both labels are measured the same way.

| Coverage | Count |
|---|---|
| S2/S3 training records (matched / unmatched) | 10,320,219 (7,638,365 / 2,681,854) |
| Unmatched records with no lookalike under any key | 726,041 (27.1% of unmatched) |
| Pairs | 9,594,178 |
| Comparable pairs: matches / no matches | 8,585,331: 6,629,518 / 1,955,813 |
| Matches sharing no key with their own S1 | 1,008,847 (13.2% of matches) |
| S1 businesses / singletons | 2,206,821 / 123,247 |

### 2.2 Normalisation

Every rule is hand-written; the lists are in [section 6](#6-everything-set-by-hand).

**Names**
- Unicode NFKC. Non-ASCII text is romanised with `anyascii` (Indic scripts become Latin letters and accents are removed). Lower-case, and split into `a–z0–9` words. **Repeated words are kept.**
- *Minor* words are 44 legal forms, honorifics and fillers, plus single letters. All other words are *content* words.

**Names in two different scripts** (one side Indic, the other Latin) are compared by a **sound-alike skeleton** of each romanised word:
- replace `ph sh ch kh gh th dh bh ck` with `f s c k g t d b k`;
- read an `m` before a consonant as `n` (anyascii writes the Hindi nasal mark as `m`);
- map `c q → k`, `z x → s`, `j → g`, `w → v`, `y → a`;
- remove vowels and collapse doubled letters.

So `केयर` → `keyr` → `kr` = `care` → `kr`; `लॉजिस्टिक्स` → `lojistiks` → `lgstks` = `logistics`; `प्राइवेट` → `praivet` → `prvt` = `private`.

**Addresses**
- Unicode NFKC. Remove Indic-script words, which in this data are mostly state names. Romanise, lower-case, and turn punctuation into spaces.
- Remove state names and two-letter state codes (US and India).
- Numbers:
  - pure digits and other digit-bearing tokens (`25a`) become house/plot numbers, with leading zeros removed;
  - ordinals (`5th`, and `fifth` → `5th`) are kept as street words.
- Expand 45 abbreviations and drop 34 unit and placeholder words.
- **Numbers are compared as a multiset**, so reordered components count as the same numbers.

### 2.3 Describing a pair: the difference pattern

**How the name differs** (by word counts; the other-script classes use the sound-alike skeletons):

| Class | Definition |
|---|---|
| identical | same words, same counts |
| legal form / filler only | only minor words differ |
| typo only | every differing word has a *typo partner* (1 edit for words up to 4 letters, 2 for longer; both words at least 3 letters) |
| typo + legal form / filler | typos plus minor-word changes |
| one word added / 2+ words added | extra content words, none lost |
| one word dropped / 2+ words dropped | content words lost, none added |
| words replaced | content words both added and lost |
| nothing in common | no content word shared, even allowing typos |
| other script: same words / legal form only / words added / words dropped / words replaced / nothing in common | the same classes, for an Indic name against a Latin name, compared by sound |

**How the address differs:** identical · only numbers differ · typo in words · typo in words + numbers differ · words added · words dropped · words replaced · nothing in common · record address empty · S1 address empty.

**How the numbers differ** (only the numbers that changed are compared; the closest changed pair decides):

| Class | Definition |
|---|---|
| same numbers | equal as a multiset, or once concatenated (`7-15` vs `715`) |
| number added / number dropped | extra or missing numbers, nothing else changed |
| number cut short | one number is the start or end of the other (`516` vs `51`) |
| one character changed | edit distance 1 (`505` vs `507`) |
| nearby | at most 50 apart (`1013` vs `1020`) |
| far-off | anything else |

A **pattern** is the combination (country, source, name change, address change, number change). 2,121 patterns occur.

### 2.4 The four lenses

| Lens | What it checks | What counts as a conflict |
|---|---|---|
| **1. Identical records** | Records grouped by an exact raw-text fingerprint, and by a normalised one (sorted name words + address word set + numbers). No pairing is involved. | Identical copies where some are matched and some not, or copies matched to different S1s |
| **2. Difference patterns** | Comparable pairs grouped by pattern. The majority label is the label of more than half the pattern's pairs; a tie counts as match. | A pair carrying the minority label of its pattern |
| **3. Model** | LightGBM on all 42 pair features, scored out-of-fold | *Strong flag:* a match with p < 0.02, or a no match with p > 0.98. *Confident-learning issue:* the predicted class disagrees with the label, using per-class thresholds equal to each class's mean predicted probability |
| **4. Assignment** | Exact normalised copies between records and S1 businesses | A match that copies a *different* S1, an unmatched record that copies an S1, and matches sharing almost nothing with their S1 |

**The lens-3 model**
- **Settings:** LightGBM, binary objective, 400 rounds, learning rate 0.08, 127 leaves, `min_data_in_leaf` 200, feature fraction 0.9, bagging 0.7, L2 1.0, seed 7.
- **42 features:** 5 categorical (country, source, and the three difference classes), 21 numeric similarities and counts, and 16 word-evidence features.
- **Word evidence:** the smoothed log-odds of each added or dropped word being in a match, `log((pos+1)/(P+V)) − log((neg+1)/(N+V))`, summarised per pair as maximum, minimum, mean, and a count of unseen words. **It is computed from other folds only, so no row sees its own label.**
- **Folds:** 5, grouped by S1 business: `((s1_id × 2654435761) >> 16) mod 5`.
- **Training sample:** each model trains on at most 3,000,000 sampled rows and predicts every pair of its held-out fold.

---

## 3. Insights, each with real examples

Examples are drawn at random with a fixed seed (21) from the pairs that fit each case. Each shows the S1 record, the S2/S3 record and the ground-truth label.

### Insight 1: identical inputs get identical labels

| Check | Result |
|---|---|
| Groups of records with identical raw text | 43,910 groups (89,114 records); **0** with different labels or S1s |
| Pairs whose normalised name and address equal the S1's | 973,426; **973,425 matches, 1 no match** |
| Duplicate S1 businesses | 0 |

An example group of raw-identical records, all matched to the same S1:

| Record | Matched to | Label |
|---|---|---|
| `S3-402327770` Kenet — 307 Green Hill Way, Mount Sterling, Kentucky | `S1-320710183` Kenet — 307 Green Hill Way, Mount Sterling, KY | **match** |
| `S3-520946478` Kenet — 307 Green Hill Way, Mount Sterling, Kentucky | `S1-320710183` Kenet — 307 Green Hill Way, Mount Sterling, KY | **match** |

| Case | S1 record | S2/S3 record | Label | Model p(match) |
|---|---|---|---|---|
| Name and address identical after normalisation | `S1-90842903` Infra Rajalakshmi Gallery Corporation — 19/52B, Delta Arcade, Kozhikode, Kerala | `S2-441131183` Infra Rájalakshmi Corporation Gallery — 019/52B, DELTA ARCADE, KOZHIKODE, Kerala | **match** | 1.000 |
| Name and address identical after normalisation | `S1-665313229` Silver Pilates — 10717 Highway 764, Whitesville, KY | `S3-486304153` Silver  Pilates — 10717 Highway 764, Whitesville, Kentucky | **match** | 1.000 |
| Name and address identical after normalisation | `S1-931199533` Lloyd & Hann Associates — 946 Cherry, Mesa, AZ | `S3-348791065` Lloyd & Associates Hann — 946 Cherry, Mesa, Arizona | **match** | 1.000 |
| The only identical pair that is not a match | `S1-362421543` Family Blue Clinic Inc. — OR, North Bend, 1465 Sherman Avenue | `S3-79931700` Inc Family Blue Clinic — 1465 Sherman Ave, North Bend, Oregon | **no match** | 1.000 |

### Insight 2: the same name with no address gets different labels

After normalisation, 535,100 groups of identical records exist. 7,419 of them carry different labels or S1s, and almost all of those are **names with an empty address**:

| Conflict | Groups, address present | Groups, address empty |
|---|---|---|
| matched to different S1s | 102 | 6,935 |
| matched to different S1s and unmatched | 0 | 79 |
| some copies matched, some unmatched | 11 | 213 |
| **Total** | **113** | **7,227** |

Examples of identical names with no address that were assigned differently:

| Conflict | Record | Assigned to |
|---|---|---|
| matched to different S1s | `S2-414004617` Asset Building — *(empty address)* | `S1-17131159` Asset Building Initiative Inc — 5370 Naiman Parkway, Solon, Unit B, OH |
| matched to different S1s | `S3-688768298` Asset (Building) — *(empty address)* | `S1-690435177` Asset Building Alliance Partners — 16330 Lydia Hill Drive, Unit APT 1109, Chesterfield, MO |
| matched to different S1s | `S2-338317115` Asset Búilding — *(empty address)* | `S1-824684367` Asset Building Project — 2740 Rd 182, Fredericktown, OH |
| matched to different S1s | `S2-349752420` GLOBAL INSTITUTE — *(empty address)* | `S1-300459357` Global Institute — NY, Rosedale, 253-04 147 Avenue |
| matched to different S1s | `S3-476835935` Global  Institute — *(empty address)* | `S1-300459357` Global Institute — NY, Rosedale, 253-04 147 Avenue |
| matched to different S1s | `S3-263633651` Global Institute, — *(empty address)* | `S1-450948185` Global Institute, LLC — 35 Brown Road, Unit 210, Mesa, AZ |
| matched to different S1s | `S2-640634309` Global (Institute) — *(empty address)* | `S1-63656908` Global Institute LLC — 8886 Beatty Road, Cook, MN |

### Insight 3: when only the numbers differ, the label depends on how the name changed

| Case | Pairs | Matches | No matches | Match rate | Against majority |
|---|---|---|---|---|---|
| only the house/plot number differs | 2,653,711 | 1,244,113 | 1,409,598 | 46.88% | 186,094 |
| ...and the name is identical | 437,860 | 435,515 | 2,345 | 99.46% | 2,345 |

**Only numbers differ, by how the name changed:**

| Name change | Pairs | Matches | No matches | Match rate | Against majority |
|---|---|---|---|---|---|
| identical | 437,860 | 435,515 | 2,345 | 99.46% | 2,345 |
| legal form / filler only | 582,408 | 291,370 | 291,038 | 50.03% | 51,395 |
| typo only | 91,207 | 76,034 | 15,173 | 83.36% | 8,780 |
| typo + legal form / filler | 40,299 | 10,818 | 29,481 | 26.84% | 3,698 |
| one word added | 621,720 | 51,552 | 570,168 | 8.29% | 24,835 |
| 2+ words added | 30,310 | 7,531 | 22,779 | 24.85% | 2,631 |
| one word dropped | 64,385 | 61,965 | 2,420 | 96.24% | 2,232 |
| 2+ words dropped | 5,367 | 5,294 | 73 | 98.64% | 69 |
| words replaced | 444,388 | 159,486 | 284,902 | 35.89% | 51,207 |
| nothing in common | 197,425 | 95,412 | 102,013 | 48.33% | 24,087 |
| other script: same words | 33,600 | 33,418 | 182 | 99.46% | 182 |
| other script: legal form / filler only | 9,155 | 1,295 | 7,860 | 14.15% | 1,214 |
| other script: words added | 47,739 | 9,622 | 38,117 | 20.16% | 8,813 |
| other script: words dropped | 403 | 24 | 379 | 5.96% | 23 |
| other script: words replaced | 28,965 | 4,386 | 24,579 | 15.14% | 4,192 |
| other script: nothing in common | 18,480 | 391 | 18,089 | 2.12% | 391 |

| Case | S1 record | S2/S3 record | Label | Model p(match) |
|---|---|---|---|---|
| Name identical, only numbers differ | `S1-233283621` Visasoft Freight, Inc — 3840 Woodland Hills Drive, Tuscaloosa, AL | `S2-164132020` (Inc) Visasoft Freight, — AL, Woodland Hills Dr, TUSCALOOSA | **match** | 1.000 |
| Name identical, only numbers differ | `S1-917390746` Total Energy P.C. — 1780 Patterson Grove Road, Ramseur, NC | `S2-224997357` Total Energy P.C. — 1780-1784 PATTERSON GROVE RD, RAMSEUR, NC | **match** | 1.000 |
| Name identical, only numbers differ | `S1-518952093` SC Allied Enbridge Inc. — 37148 Stoney Run, Selbyville, DE | `S3-756437788` SC Allied Enbridge Inc — 37150 Stoney Run, Selbyville, Delaware | **no match** | 0.517 |
| Name identical, only numbers differ | `S1-218870966` Ellis, Robbins and Dagostino, Inc. — 25642 Gambit Trail, Wittmann, AZ | `S3-298921070` >> Ellis, Robbins and Dagostino, [Inc] — 45646 Gambit Trl, Wittmann, Arizona | **no match** | 0.909 |
| One word added to the name, only numbers differ | `S1-295223005` Kiser Homes — 702 Tailwind Drive, Sacramento, CA | `S2-407850045` Kiser Hmoes Holdings — 705 TAILWIND DRIVE, PO BOX 8268, SACRAMENTO, CA | **no match** | 0.000 |
| One word added to the name, only numbers differ | `S1-110171445` I A & K Ngl Inc. — 1971 Black Jack-grimesland Road, Greenville, NC | `S3-252155016` I Á & K Ngl Summit Inc. Inc. — 1980 Black Jack-Grimesland Rd, Greenville, North Carolina | **no match** | 0.000 |
| One word added to the name, only numbers differ | `S1-849422771` Federal Iron Works LLC — 600 Island Drive, Unit 4, Lincoln County, OR | `S2-852334653` Federal Iron Works  LLC #71864 — LINCOLN COUNTY, 600 ISLAND DR, OR | **match** | 1.000 |
| One word added to the name, only numbers differ | `S1-882609038` Pediatric Dental Care Associates Inc — 2702 Yorkshire Drive, Unit 1026, Phoenix, AZ | `S3-479420598` Pediatric Dental Care Associates Incorporated #81859 — 2704 Yorkshire Drive, # 1026, Phoenix, Arizona | **match** | 0.182 |
| Only a legal form changed, only numbers differ | `S1-960380760` Weekend Investment Limited — C/O Onkar Nath Dwivedi, Dubai Paraspur, Colonelganj, Gonda, Uttar Pradesh | `S3-283885955` Weekend Investment Ltd — Plot 05 C/o Onkar Nath Dwivedi, Dubai Paraspur, Gonda, Colonelganj, UP | **match** | 0.944 |
| Only a legal form changed, only numbers differ | `S1-231960052` Taia Management PLLC — 105 Mobile Street, Fairhope, AL | `S3-924882957` Taia Management Corp. — 118 Mobile St, Fairhope, Alabama | **no match** | 0.002 |

### Insight 4: with the name identical, most number changes are matches; nearby and one-character changes are the mixed ones

| Number change | Pairs | Matches | No matches | Match rate | Against majority |
|---|---|---|---|---|---|
| number added | 114,262 | 114,257 | 5 | 100.00% | 5 |
| number dropped | 177,521 | 177,521 | 0 | 100.00% | 0 |
| number cut short | 89,334 | 89,247 | 87 | 99.90% | 87 |
| number: one character changed | 25,327 | 24,429 | 898 | 96.45% | 898 |
| number nearby (<=50 apart) | 9,943 | 8,707 | 1,236 | 87.57% | 1,236 |
| number far-off | 21,473 | 21,354 | 119 | 99.45% | 119 |

| Case | S1 record | S2/S3 record | Label | Model p(match) |
|---|---|---|---|---|
| Name identical, number added | `S1-11806882` Ascon Foundation — H.No:1-98/90/70/14/P, P.S Residency, Flat 401, 4 Flr, Jaihind Enclave Madhapur, Hyderabad, Telangana | `S3-758849373` Ascon Foundation — No. 990 H.no:1-98/90/70/14/p, P.s Residency, Flat 401, 4 Flr, Jaihind Enclave Madhapur, Hyderabad, TG | **match** | 1.000 |
| Name identical, number added | `S1-249714913` Abs Group of Companies — Bhive, Mahalakshmi Chambers, 7Th Floor, 29, Next To Trinity Metro Station, Opp. Kotak Bank, Yallappa Garden, Yellappa Chetty Layout, Sivachetti Gardens, Bangalore North, Bangalore, Karnataka | `S2-820885817` Abs Group of Companies — H.NO A-94 BHIVE, MAHALAKSHMI CHAMBERS, 7TH FLOOR, 29, NEXT TO TRINITY METRO STATION, OPP. KOTAK BANK, YALLAPPA GARDEN, YELLAPPA CHETTY LAYOUT, SIVACHETTI GARDENS, BANGALORE NORTH, Karnataka | **no match** | 1.000 |
| Name identical, number dropped | `S1-175476223` Thompson Liquor — 2002 Bedford Street, Rome, NY | `S3-339837444` Thompson  Liquor — Bedford Street, Rome, New York | **match** | 1.000 |
| Name identical, number cut short | `S1-16476040` Janenna Boser Capital Great Inc — 1514 Lawrence Road, Wake Forest, NC | `S3-326193753` janenna boser capital great inc — 1514d Lawrence Road, North Carolina, Wake Forest | **match** | 1.000 |
| Name identical, number cut short | `S1-708289642` Celenet Inc. — 156 91, Ulm, MT | `S2-750307000` Inc Celenet — ULM, 15 91, MT | **no match** | 0.994 |
| Name identical, number: one character changed | `S1-19316860` PG Gold Private Limited — Kerala, Tc No.2/21, Trivandrum, Thiruvananthapuram, Blra-2, Bridge Lane, Ulloor, Thiruvananthapuram | `S3-349710292` PG Limited Private (Gold) — Tc No.3/21, Blra-2, Bridge Lane, Ulloor, Thiruvananthapuram, Thiruvananthapuram, Trivandrum, KL | **match** | 0.997 |
| Name identical, number: one character changed | `S1-905994548` Nilinva Inc. — 4235 377, Unit APT 1006, Brownwood, TX | `S3-543152196` Nilinva Inc — Texas, NULL, 4236- 377, Brownwood | **no match** | 0.962 |
| Name identical, number nearby (<=50 apart) | `S1-777015900` Cure Barbershop — 26656 Molidor Road, IL, Ingleside | `S2-994494684` Cure Barbershop — 26632 MOLIDOR ROAD, INGLESIDE, IL | **match** | 0.995 |
| Name identical, number nearby (<=50 apart) | `S1-777034793` Flinn, Battle & Silverman Partnership Inc. — Unit 105, Phoenix, AZ, 6131 16th Street | `S3-442756149` Flinn, Battle & Silverman Partnership [Inc] — 6152 16th Street, Phoenix, # 105, Arizona | **no match** | 0.596 |
| Name identical, number far-off | `S1-583267756` Aurari Jewelers Corp — 427 Church Street, Hagerstown, MD | `S3-653754748` Aurari Jéwelers Corp — 148 Church Street, Hagerstown, Maryland | **match** | 0.998 |
| Name identical, number far-off | `S1-672471413` Armstrong Aurora Inc. — 8368 Pipestone Drive, Fort Worth, TX | `S2-928592534` Inc Armstrong Aurora — FORT WORTH, TX, 837 PIPESTONE DR | **no match** | 0.513 |

### Insight 5: with the address fully identical, most name changes are matches

| Case | Pairs | Matches | No matches | Match rate | Against majority |
|---|---|---|---|---|---|
| only the name differs (address identical or a word typo) | 2,358,535 | 2,324,461 | 34,074 | 98.56% | 33,512 |
| ...and the address is fully identical | 2,243,212 | 2,214,092 | 29,120 | 98.70% | 29,120 |
| only typo-like differences | 430,937 | 418,588 | 12,349 | 97.13% | 11,228 |

"Only typo-like differences" means the name is identical or differs by typos, the address is identical or has word typos, and any number change is a cut-short or one-character change, with at least one typo present.

**Address fully identical, by how the name changed:**

| Name change | Pairs | Matches | No matches | Match rate | Against majority |
|---|---|---|---|---|---|
| identical | 973,426 | 973,425 | 1 | 100.00% | 1 |
| legal form / filler only | 706,396 | 706,176 | 220 | 99.97% | 220 |
| typo only | 176,114 | 171,822 | 4,292 | 97.56% | 4,292 |
| typo + legal form / filler | 29,523 | 27,439 | 2,084 | 92.94% | 2,084 |
| one word added | 176,023 | 175,265 | 758 | 99.57% | 758 |
| 2+ words added | 131,582 | 131,515 | 67 | 99.95% | 67 |
| one word dropped | 149,191 | 149,079 | 112 | 99.92% | 112 |
| 2+ words dropped | 27,733 | 27,727 | 6 | 99.98% | 6 |
| words replaced | 382,229 | 366,061 | 16,168 | 95.77% | 16,168 |
| nothing in common | 258,005 | 253,254 | 4,751 | 98.16% | 4,751 |
| other script: same words | 138,769 | 138,769 | 0 | 100.00% | 0 |
| other script: legal form / filler only | 5,672 | 5,658 | 14 | 99.75% | 14 |
| other script: words added | 41,097 | 41,021 | 76 | 99.82% | 76 |
| other script: words dropped | 106 | 104 | 2 | 98.11% | 2 |
| other script: words replaced | 18,523 | 18,448 | 75 | 99.60% | 75 |
| other script: nothing in common | 2,249 | 1,754 | 495 | 77.99% | 495 |

| Case | S1 record | S2/S3 record | Label | Model p(match) |
|---|---|---|---|---|
| Address identical, name legal form / filler only | `S1-354072557` Elder Fin Limited — Srinivasa, 318, 6Th 'A' Cross, Bhuvanagiri Ombr Layout, Banaswadi, Bangalore, Karnataka | `S2-587428591` ELDER-FIN LTD — SRINIVASA, 318, 6TH 'A' CROSS, BHUVANAGIRI OMBR LAYOUT, BANASWADI, BANGALORE, Karnataka | **match** | 1.000 |
| Address identical, name legal form / filler only | `S1-629384902` Premium Bio Private Limited — Parshwa, Opp. Rajpath Club S.G. Highway, Bodakdev, Ahmedabad, 301/1, Gujarat | `S2-73827510` Premium Bio Limited Limited — 301/1, PARSHWA, OPP. RAJPATH CLUB S.G. HIGHWAY, BODAKDEV, AHMEDABAD, Gujarat | **no match** | 1.000 |
| Address identical, name typo only | `S1-583647511` Superior Secure Earths — 356 Tremont Avenue, Kenmore, NY | `S2-725931519` Superi0r Secure Earths — 356 TREMONT AVENUE, KENMORE, NY | **match** | 0.990 |
| Address identical, name typo only | `S1-131898887` Montora Jersey LLC — 48 Meadow Road, Cortlandt, NY | `S2-332302620` Montori Jersey  LLC — 048 MEADOW ROAD, CORTLANDT, NY | **no match** | 0.950 |
| Address identical, name one word added | `S1-834909267` Prairie Freight LLC — 1185 A Circulo Mercado, Rio Rico, AZ | `S2-4596474` Prairie Prairie Freight — 1185 A CIRCULO MERCADO, RIO RICO, AZ | **match** | 1.000 |
| Address identical, name one word added | `S1-435204568` Pune Restaurants Corporation — Parth Enclave Fln 22 Wing A Sno 20 / 1 A C.T.S.No 293 (P) Karve Nagar Pune, Pune, Maharashtra | `S2-749802037` Pune Restaurants C0rporation Infratech — PARTH ENCLAVE FLN ##22 WING A SNO 20 / 1 A C.T.S.NO 293 (P) KARVE NAGAR PUNE, PUNE, महाराष्ट्र | **no match** | 0.124 |
| Address identical, name words replaced | `S1-145447746` Perfect Global Pvt Ltd — 58 Malharganj, Indore, Madhya Pradesh | `S3-100876375` Perfect Pvt Pvt Ltd Services — Door No 58 Malharganj, Indore, MP | **match** | 1.000 |
| Address identical, name words replaced | `S1-407181351` TC Trg Inc — 2715 Anderson Street, Dallas, TX | `S2-594914475` MONKS TRG INCORPORATED — 2715 ANDERSON ST, DALLAS, TX | **no match** | 0.002 |
| Address identical, name nothing in common | `S1-874731007` Hepworth & Escalante Healthy — 8006 J Street, Omaha, NE | `S3-293565740` Hepworthescalantehealthy.Com — Nebraska, 8006 J Street, Omaha | **match** | 1.000 |
| Address identical, name nothing in common | `S1-60450947` Vellayambalam India Private Limited — House No 2D Sfs Apartments Capital Near Police Head Quarters, Vellayambalam, Trivandrum, Kerala | `S3-884021298` Ventures Hexagon Consultants Private Ltd — House No 2D Sfs Apartments Capital Near Police Head Quarters, Vellayambalam, Trivandrum, KL | **no match** | 0.001 |

### Insight 6: names in another script, compared by sound

A Hindi, Tamil or other Indic name against a Latin S1 name is now split by what changed when the words are compared by sound, instead of being one "other script" class. The tables below give the match rate of each kind of change. Compare them with the Latin-name tables in insights 3 and 5.

**Address fully identical:**

| Name change | Pairs | Matches | No matches | Match rate | Against majority |
|---|---|---|---|---|---|
| other script: same words | 138,769 | 138,769 | 0 | 100.00% | 0 |
| other script: legal form / filler only | 5,672 | 5,658 | 14 | 99.75% | 14 |
| other script: words added | 41,097 | 41,021 | 76 | 99.82% | 76 |
| other script: words dropped | 106 | 104 | 2 | 98.11% | 2 |
| other script: words replaced | 18,523 | 18,448 | 75 | 99.60% | 75 |
| other script: nothing in common | 2,249 | 1,754 | 495 | 77.99% | 495 |

**Only numbers differ:**

| Name change | Pairs | Matches | No matches | Match rate | Against majority |
|---|---|---|---|---|---|
| other script: same words | 33,600 | 33,418 | 182 | 99.46% | 182 |
| other script: legal form / filler only | 9,155 | 1,295 | 7,860 | 14.15% | 1,214 |
| other script: words added | 47,739 | 9,622 | 38,117 | 20.16% | 8,813 |
| other script: words dropped | 403 | 24 | 379 | 5.96% | 23 |
| other script: words replaced | 28,965 | 4,386 | 24,579 | 15.14% | 4,192 |
| other script: nothing in common | 18,480 | 391 | 18,089 | 2.12% | 391 |

| Case | S1 record | S2/S3 record | Label | Model p(match) |
|---|---|---|---|---|
| Name other script: same words | `S1-503846097` Swastik Business Private Limited — E-139, Affordable Housing, Kansuwa, Kota, Rajasthan | `S2-172334389` स्वस्तिक बिजनेस प्राइवेट लिमिटेड — E-139, AFFORDABLE HOUSING, KANSUWA, KOTA, Rajasthan | **match** | 1.000 |
| Name other script: same words | `S1-861604332` Hotel Lakshmi Developers Private Limited — H.No.12-66, 1St Floor, Pr Pally, Sangareddy, Hyderabad, Telangana | `S3-328020812` ಲಕ್ಷ್ಮಿ ಡೆವಲಪರ್ಸ್ ಹೋಟೆಲ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ — #6A, Bangalore, 9Th Main Rmv Extn., ಕರ್ನಾಟಕ | **no match** | 0.001 |
| Name other script: legal form / filler only | `S1-279771671` Ace Properties — Flat No.15, D.No.22-3/A, Albert Apartments Radha Krishna Nagar, Malkajgiri, Secunderabad, Hyderabad, Telangana | `S2-555338229` ఏస్ ప్రాపర్టీస్ — తెలంగాణ, HYDERABAD, FLAT NO.15, SECUNDERABAD | **match** | 0.999 |
| Name other script: legal form / filler only | `S1-265929325` Shiv Global Private Limited — Post Shravan Mines Post Ranuka Tal Chikhli, Chikhli, Valsad, Gujarat | `S2-672189126` શિવ ગ્લોબલ લિમિટેડ — H.NO 035 POST SHRAVAN MINES POST RANUKA TAL CHIKHLI, CHIKHLI, VALSAD, ગુજરાત | **no match** | 0.022 |
| Name other script: words added | `S1-396800482` Best Finance Pvt Ltd — P 3 New C I T Road Bowbazar, South 24 Parganas, Kolkata, West Bengal | `S2-984851927` বেস্ট ফাইন্যান্স প্রাইভেট লিমিটেড — SOUTH 24 PARGANAS, P 3 NEW C I T ROAD BOWBAZAR, West Bengal, KOLKATA | **match** | 1.000 |
| Name other script: words added | `S1-109802400` Red Jay Infrastructure Private Limited — 83/85, Office No.205, Diamond Plaza, Dhanji Street, Zaveri Bazar, Mumbai, Maharashtra | `S3-109006956` रेड जय इंफ्रास्ट्रक्चर गारमेंट्स प्राइवेट लिमिटेड — H.no 858 83/90, Office No.205, Diamond Plaza, Dhanji Street, Zaveri Bazar, Mumbai, महाराष्ट्र | **no match** | 0.000 |
| Name other script: words dropped | `S1-402037994` Golden Estate — No. 85, Thavarazhikath, Kalannjoor, Pathanamthitta, Kerala | `S2-569161091` ഗോൾഡൻ എസ്റ്റേറ്റ് — Kerala, PATHANAMTHITTA, THAVARAZHIKATH, KALANNJOOR, NO. 85, PATHANAMTHITTA | **match** | 0.996 |
| Name other script: words dropped | `S1-56165214` North Energy Private Limited — B/104 Dev Sangam Appt, Near Bhat Motera Road, Koteswar, Koteswar Village, Chandkheda, Ahmedabad, Gujarat | `S2-705169319` નોર્થ આઈટી પ્રાઇવેટ લિમિટેડ — AHMEDABAD, B/117- DEV SANGAM APPT, NEAR BHAT MOTERA ROAD, KOTESWAR, KOTESWAR VILLAGE, ગુજરાત, CHANDKHEDA | **no match** | 0.001 |
| Name other script: words replaced | `S1-911224655` First Infrastructure Private Limited — No. 6-4, Variyar Nagar Ganapathy, Coimbatore, Tamil Nadu | `S3-782968491` ஃபர்ஸ்ட் இன்ஃப்ராஸ்ட்ரக்சர் பிரைவேட் லிமிடெட் — H.no 7-6-4, Variyar Nagar Ganapathy, Coimbatore, TN | **match** | 0.995 |
| Name other script: words replaced | `S1-369299438` Shyam Construction Private Limited — 9-A, Dev Vrund Bh. Uma Bhavan, Anand, Gujarat | `S2-176755689` શ્યામ કન્સલ્ટિંગ પ્રાઇવેટ લિમિટેડ — 3-18-A, DEV VRUND BH. UMA BHAVAN, ANAND, ગુજરાત | **no match** | 0.000 |
| Name other script: nothing in common | `S1-834243708` Sai It Private Limited — B-1211A, 12Th Floor, Ithum Tower-B, Plot No. A-40, Sector-62, Noida, Dadri, Gautam Buddha Nagar, Uttar Pradesh | `S2-626636648` साईं आईटी प्राइवेट लिमिटेड — DADRI, #557 B-1211A, GAUTAM BUDDHA NAGAR, Uttar Pradesh, 12TH FLOOR, ITHUM TOWER-B, PLOT NO. A-40, SECTOR-62, NOIDA | **match** | 0.997 |
| Name other script: nothing in common | `S1-287634481` One Dynamic Engineering Pvt Ltd — Hyderabad, P No. 42/1, Hyderabad, Telangana, 42/2 & 43 Shaikpet | `S2-993447460` మోడర్న్ ఫైనాన్స్ మోటార్స్ ప్రైవేట్ లిమిటెడ్ — P. NO 16, SHAIKPET, HYDERABAD, Telangana | **no match** | 0.000 |

### Insight 7: across all patterns, about 1 pair in 29 carries its pattern's minority label

| | |
|---|---|
| Patterns observed | 2,121 |
| Patterns containing both labels | 1,575 |
| Pairs against their pattern's majority | 299,356 (3.49%) |
| Share of pairs in patterns whose minority exceeds 1% / 5% / 20% | 38.43% / 18.83% / 4.47% |

**The 15 patterns with the most pairs against their majority** (pooled over country and source). The full list is in `pattern_summary.tsv`.

| Name change | Address change | Number change | Pairs | Match rate | Against majority |
|---|---|---|---|---|---|
| words replaced | only numbers differ | number added | 61,930 | 71.34% | 17,749 |
| legal form / filler only | only numbers differ | number: one character changed | 126,783 | 13.46% | 17,063 |
| words replaced | identical | same numbers | 382,229 | 95.77% | 16,168 |
| one word added | only numbers differ | number added | 41,725 | 38.02% | 15,863 |
| words replaced | only numbers differ | number cut short | 45,679 | 70.44% | 13,503 |
| legal form / filler only | only numbers differ | number far-off | 27,328 | 49.29% | 13,470 |
| legal form / filler only | only numbers differ | number cut short | 68,749 | 84.53% | 10,636 |
| one word added | only numbers differ | number cut short | 32,310 | 31.76% | 10,261 |
| words replaced | only numbers differ | number: one character changed | 109,917 | 8.22% | 9,038 |
| nothing in common | only numbers differ | number cut short | 27,805 | 69.67% | 8,433 |
| words replaced | only numbers differ | number far-off | 24,548 | 30.96% | 7,599 |
| legal form / filler only | words dropped | number: one character changed | 20,399 | 30.38% | 6,197 |
| other script: words added | only numbers differ | number added | 12,645 | 51.79% | 6,096 |
| legal form / filler only | only numbers differ | number nearby (<=50 apart) | 157,018 | 3.61% | 5,666 |
| nothing in common | only numbers differ | number: one character changed | 18,511 | 29.64% | 5,486 |

### Insight 8: most pattern-level mixing resolves at a finer level

The pattern classes are coarse. A model that also sees *how far* the number moved and *which* words were added separates the labels far better than the patterns do:

| | |
|---|---|
| Out-of-fold AUC | 0.99920 |
| Log loss / Brier score | 0.02622 / 0.00693 |
| Pairs with p between 0.2 and 0.8 | 1.25% |
| Pairs with p between 0.05 and 0.95 | 4.25% |

**What the model relies on** (the 15 largest shares of split gain):

| Feature | Share |
|---|---|
| `num_logdiff` | 56.2% |
| `num_lev` | 11.6% |
| `name_added_lo_min` | 8.4% |
| `nd` | 6.4% |
| `rel` | 2.9% |
| `name_jac` | 1.9% |
| `n_num_add` | 1.6% |
| `addr_tsr` | 1.6% |
| `addr_added_lo_min` | 1.5% |
| `name_added_lo_mean` | 1.1% |
| `name_tsr` | 0.8% |
| `name_ratio` | 0.8% |
| `name_len_rec` | 0.8% |
| `name_minor` | 0.7% |
| `name_added_lo_max` | 0.4% |

The pattern with the most pairs against its majority is *name: words replaced \| address: only numbers differ \| numbers: number added* (India, S3): 19,287 pairs, 52.95% matches. Within it, the model still separates the two labels:

| Case | S1 record | S2/S3 record | Label | Model p(match) |
|---|---|---|---|---|
| Same pattern, match the model gets right (p > 0.9) | `S1-423833226` Mj & Brothers Pvt. Ltd. — Flat 158Ff, Lumbini App Kaushambi Sec14, Ghaziabad, Uttar Pradesh | `S3-616799956` Mj-and Bitonhtrs Pvt. Ltd. — Ghaziabad, Lumbini App Kaushambi Sec14, UP, No 325 Flat 158Ff | **match** | 0.920 |
| Same pattern, match the model gets right (p > 0.9) | `S1-270074515` Hallmark Clinic Ltd — C-2 Blue Haven Opp Meghalay Flats, Naranpura, Ahmedabad, Gujarat | `S3-492809999` Hallmark Ltd Center — C-2/3 Blue Haven Opp Meghalay Flats, Naranpura, Ahmedabad, ગુજરાત | **match** | 0.999 |
| Same pattern, no match the model gets right (p < 0.1) | `S1-632748659` Media Dav Consultants Limited — Guma Rabindrabazar, Guma, Parganas North, West Bengal | `S3-753092926` Media Dav Producer Limited — Parganas North, WB, #88 Guma Rabindrabazar, Guma | **no match** | 0.000 |
| Same pattern, no match the model gets right (p < 0.1) | `S1-851954767` Expressway Medical Centre Limited — Fenhill House, 2Nd Floor, 254 Perin Nariman Street, Fort, Mumbai, Maharashtra | `S3-751368844` Expressway Nirmaan Ltd — H.no 76 Fenhill House, 2Nd Floor, 254 Perin Nariman Street, Fort, Mumbai, MH | **no match** | 0.001 |

### Insight 9: the model's strongest contradictions have recognisable shapes

| | |
|---|---|
| Matches the model rejects (p < 0.02) | 1,725 |
| No-match pairs the model accepts (p > 0.98) | 5,340 |
| Confident-learning label issues | 5,388 (0.06%). Thresholds: match 0.9905, no match 0.9693 |

| Matches the model rejects: pattern | Pairs |
|---|---|
| legal form / filler only · only numbers differ · number nearby (<=50 apart) | 548 |
| legal form / filler only · only numbers differ · number: one character changed | 294 |
| legal form / filler only · words dropped · number nearby (<=50 apart) | 61 |
| words replaced · only numbers differ · number: one character changed | 51 |
| legal form / filler only · words replaced · number nearby (<=50 apart) | 49 |
| legal form / filler only · typo in words + numbers differ · number nearby (<=50 apart) | 47 |

| No-match pairs the model accepts: pattern | Pairs |
|---|---|
| legal form / filler only · record address empty · n/a | 837 |
| identical · record address empty · n/a | 608 |
| typo only · identical · same numbers | 515 |
| words replaced · identical · same numbers | 312 |
| legal form / filler only · identical · same numbers | 220 |
| nothing in common · identical · same numbers | 199 |

| Case | S1 record | S2/S3 record | Label | Model p(match) |
|---|---|---|---|---|
| Labelled match, model p < 0.02 | `S1-106152514` Anthony, Swann and Rivas Cslm LLC — 1108 Locust Street, Centralia, IL | `S3-334852431` Anthony, 5wann and Rivas Cslm — 1103 Locust St, Centralia, Illinois | **match** | 0.007 |
| Labelled match, model p < 0.02 | `S1-587768670` Pandya Keystone Storage LP — 21 7th Street, Franklin, OH | `S3-269260948` The Pandya Keystone Storage LP — 17 7th Saint, Franklin, Ohio | **match** | 0.012 |
| Labelled match, model p < 0.02 | `S1-115266491` Continental Express Ready — 4497 Unitia Road, Lenoir City, TN | `S3-423211284` Continental Express Ready LP — ##4500 Unitia Road, Lenoir City, Tennessee | **match** | 0.017 |
| Labelled no match, model p > 0.98 | `S1-5997363` SAV Holdings Company — A-121, The Summit, Dlf Phase - 5 Golf Course Road, Gurgaon, Haryana | `S2-683965905` ISAV Holdings Company [Services] (ID: 93990) — HN 5-89 A-121, THE SUMMIT, DLF PHASE - 5 GOLF COURSE ROAD, GURGAON, Haryana | **no match** | 0.991 |
| Labelled no match, model p > 0.98 | `S1-145942212` Pinnacle Cafe LLC — NY, 2 Congers Point Way, Bolton | `S3-156863845` Pinnacle Cafe — *(empty address)* | **no match** | 0.989 |
| Labelled no match, model p > 0.98 | `S1-358282227` Healy, Tebo and Potter Studios — 44 Remsen Avenue, Valley Stream, NY | `S3-938015629` Real, Tebo And Potter Studios Inc — 44 Remsen Avenue, Valley Stream, New York | **no match** | 0.993 |

### Insight 10: the comparison does not reach every record

- 1,008,847 matches share none of the six keys with their own S1, and 726,041 unmatched records have no lookalike at all.
- 20,625 matches share no name word and under 25% of their address words with their S1.

These are outside lenses 2 and 3.

| Case | S1 record | S2/S3 record | Label | Model p(match) |
|---|---|---|---|---|
| Match that shares no blocking key with its S1 | `S1-840673779` Colleen Oconnor Fashion — 3504 Wynnfield Drive, High Point, NC | `S3-266038318` *** Co1leen Oconnor Fashion (LLC) — High Point, North Carolina, 3504 Wynnfield Drdve | **match** | 0.928 |
| Match that shares no blocking key with its S1 | `S1-992616435` IY Midwest Hamilton — 15 Williams Street, Dannemora, NY | `S3-940288982` IY Midwest Hbalomlton — 15 Williams St, Lyon Mountain, New York | **match** | 0.998 |
| Match that shares no blocking key with its S1 | `S1-91162914` New Delhi Cloud Pvt. Ltd. — New Delhi, Central Delhi, Delhi, Reghar Pura Karol Bagh, 3455/3 F/F | `S2-20156774` NEW DELHI CLOUD PVT (LTD) #21703 — 3455/3 F/F, REGHAR PURA KAROL BAGH, NEW DELHI, Delhi | **match** | 0.999 |

Unmatched records with no lookalike:

| Record | Why it has no pair |
|---|---|
| `S2-924074418` Total Robotics (Inc.) — CHARLOTTE CITY, NC, 332 BEARDSLEY DR | no S1 shares a blocking key |
| `S3-870959635` बेस्ट ऑल कंसल्टेंट्स प्रोविजन प्राइवेट लिमिटेड — Mumbai City, Bombay, महाराष्ट्र, Flat No-h/110 | no S1 shares a blocking key |
| `S3-739330080` Shri Sanjana Education Ventures — Door No C, Laxmipuri, Kolhapur, MH | no S1 shares a blocking key |

### Insight 11: the reference itself is clean

- **0 duplicate S1 businesses.**
- **0 matched records are an exact copy of a *different* S1** than their own.
- **1 unmatched record is an exact copy of an S1.**

The single unmatched exact copy is the last example under Insight 1.

### Score impact

| Rule | Macro F0.5 over all S1 | S1 affected | Mean F0.5 on those |
|---|---|---|---|
| Every comparable pair takes its pattern's majority label (in-sample) | 0.9756 | 255,793 | 0.7891 |
| Out-of-fold model, threshold 0.5 | 0.9933 | 71,938 | 0.7955 |

These are ceilings on training data, not leaderboard estimates: pairs outside the comparison are assumed correct.

### The cases checked by hand

Every record from the manual checks, and how the audit sees it. A pair is **flagged** when either:
- its label is the minority label of its exact pattern; or
- the model contradicts it strongly (a match with p < 0.02, or a no match with p > 0.98).

"Not flagged" therefore means the label agrees with how most identical-pattern pairs are labelled, and the model does not strongly disagree. 2 of the 24 records are flagged, and 1 is outside the pattern and model checks.

#### `S1-302869473` International Automation Consultants Inc — 329 Rev Walton Drive, Lockport, IL

Ground truth: singleton (no matches).

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S2-521228093` International Automation Consultants Group — 338-342 REV WALTON DR, LOCKPORT, IL | **no match** | one word added · only numbers differ · number nearby (<=50 apart) | 0.4% of the 107,548 pairs with this exact pattern (US, S2) are matches, so its majority label is no match; this pair agrees. Model p = 0.000: agrees with the label. → not flagged |
| `S3-765861618` International Automation Consultants Corp — 338 Rev Walton Dr, Lockport, Illinois | **no match** | legal form / filler only · only numbers differ · number nearby (<=50 apart) | 3.3% of the 71,487 pairs with this exact pattern (US, S3) are matches, so its majority label is no match; this pair agrees. Model p = 0.003: agrees with the label. → not flagged |

#### `S1-730719211` Renee G. Garcia, O.D. — 516 Main Street, Cleburne, TX

Ground truth: 4 matched records.

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S2-467798366` Renee G. Garcia, — 51 MAIN ST, CLEBURNE, TX | **match** | legal form / filler only · only numbers differ · number cut short | 82.8% of the 29,926 pairs with this exact pattern (US, S2) are matches, so its majority label is match; this pair agrees. Model p = 0.993: agrees with the label. → not flagged |
| `S2-61695240` RENEE G. GARCIA, O.D. — TX, CLEBURNE, 51 MAIN STREET | **match** | identical · only numbers differ · number cut short | 99.9% of the 38,200 pairs with this exact pattern (US, S2) are matches, so its majority label is match; this pair agrees. Model p = 0.999: agrees with the label. → not flagged |
| `S2-241456014` renee g. garcia, o.d. — 51 MAIN STREET, CLEBURNE, TX | **match** | identical · only numbers differ · number cut short | 99.9% of the 38,200 pairs with this exact pattern (US, S2) are matches, so its majority label is match; this pair agrees. Model p = 0.999: agrees with the label. → not flagged |
| `S3-951830396` Renee G Garcia, OD — 516 Main St, Cleburne, Texas | **match** | one word added · identical · same numbers | 99.8% of the 76,886 pairs with this exact pattern (US, S3) are matches, so its majority label is match; this pair agrees. Model p = 1.000: agrees with the label. → not flagged |

#### `S1-262997549` Gabriella's Preferred Security — 1013 Girard Avenue, Indianola, IA

Ground truth: singleton (no matches).

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S2-879297655` Gabriella's Preferred Security Ltd — 001020 GIRARD AVE, INDIANOLA, IA | **no match** | legal form / filler only · only numbers differ · number nearby (<=50 apart) | 3.7% of the 70,793 pairs with this exact pattern (US, S2) are matches, so its majority label is no match; this pair agrees. Model p = 0.011: agrees with the label. → not flagged |
| `S3-910786226` Co Gabriella'S Preferred Security — 1020 Girard Ave, Indianola, Iowa | **no match** | legal form / filler only · only numbers differ · number nearby (<=50 apart) | 3.3% of the 71,487 pairs with this exact pattern (US, S3) are matches, so its majority label is no match; this pair agrees. Model p = 0.009: agrees with the label. → not flagged |

#### `S1-989924267` Garrity Pinnacle Capital LLC — 505 Innovation Way, Walton, KY

Ground truth: 3 matched records.

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S2-941206792` 6arrity Pinnace Capital LLC — 507 INNOVATION WAY, WALTN, KY | **match** | typo only · typo in words + numbers differ · number: one character changed | 33.6% of the 277 pairs with this exact pattern (US, S2) are matches, so its majority label is no match; this pair **goes against it**. Model p = 0.402: leans the other way, but not past the flag thresholds (0.02 / 0.98). → **flagged by pattern** |
| `S2-973482655` Garrity Pinnacle LLC 5ervices — *(empty address)* | **match** | words replaced · record address empty · n/a | This match shares no blocking key with its S1, so it is outside the pattern and model checks. The model still scores it: p = 0.998. |
| `S3-262595741` Garrity Pinnacle Capital — ##505 Innovation Way, Walton, Kentucky | **match** | legal form / filler only · identical · same numbers | 100.0% of the 220,502 pairs with this exact pattern (US, S3) are matches, so its majority label is match; this pair agrees. Model p = 1.000: agrees with the label. → not flagged |

#### `S1-783518091` New Delhi Nath Private Limited — Property No.-K-92 (F.F) K Block New Mahabir Nagar Near Tilak Nagar, New Delhi, Delhi

Ground truth: singleton (no matches).

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S3-188760616` Private New Delhi Nath Overseas Limited — Property No.-k-95 (F.F) K Block New Mahabir Nagar Near Tilak Nagar, New Delhi, DL | **no match** | one word added · only numbers differ · number: one character changed | 1.6% of the 31,509 pairs with this exact pattern (India, S3) are matches, so its majority label is no match; this pair agrees. Model p = 0.000: agrees with the label. → not flagged |
| `S3-44406563` New Delhi Nath L.L.P. — Property No.-k-95 (F.F) K Block New Mahabir Nagar Near Tilak Nagar, New Delhi, दिल्ली | **no match** | legal form / filler only · only numbers differ · number: one character changed | 30.3% of the 9,359 pairs with this exact pattern (India, S3) are matches, so its majority label is no match; this pair agrees. Model p = 0.001: agrees with the label. → not flagged |

#### `S1-14448673` Gail's Medical — 611 5th Street, Hardin, MT

Ground truth: singleton (no matches).

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S3-219141874` The Gail's Medical Holdings — 620 Fifth Street, Hardin, Montana | **no match** | one word added · only numbers differ · number nearby (<=50 apart) | 0.4% of the 110,730 pairs with this exact pattern (US, S3) are matches, so its majority label is no match; this pair agrees. Model p = 0.000: agrees with the label. → not flagged |

#### `S1-115182842` Overy Cardiology — 17958 Quantico Road, Unit UNIT, Apple Valley, CA

Ground truth: 3 matched records.

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S2-464452285` -- Overy Center — 17887 QUANTICO RD, CA, APPLE VALLEY | **match** | words replaced · only numbers differ · number far-off | 39.3% of the 9,793 pairs with this exact pattern (US, S2) are matches, so its majority label is no match; this pair **goes against it**. Model p = 0.996: agrees with the label. → **flagged by pattern** |
| `S2-440934584` Overy Cardiology [Corp] — 17887 QUANTICO ROAD, APPLE VALLEY, CA | **match** | legal form / filler only · only numbers differ · number far-off | 53.4% of the 13,478 pairs with this exact pattern (US, S2) are matches, so its majority label is match; this pair agrees. Model p = 0.968: agrees with the label. → not flagged |
| `S3-176968062` Overy Cardiology — 017958 Quantico Road, PMB 3525, Apple Valley, California | **match** | identical · only numbers differ · number added | 100.0% of the 40,122 pairs with this exact pattern (US, S3) are matches, so its majority label is match; this pair agrees. Model p = 1.000: agrees with the label. → not flagged |
| `S2-716420516` Overy Cardiology Eastgate — 17959. QUANTICO ROAD, APPLE VALLEY, CA | **no match** | one word added · only numbers differ · number: one character changed | 1.3% of the 78,331 pairs with this exact pattern (US, S2) are matches, so its majority label is no match; this pair agrees. Model p = 0.000: agrees with the label. → not flagged |

#### `S1-733655541` Womens Health Center — 2260 B Avenue, Schleswig, IA

Ground truth: singleton (no matches).

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S2-229535269` Womens Health Center  LLC — 2262 B AVENUE, SCHLESWIG, IA | **no match** | legal form / filler only · only numbers differ · number: one character changed | 11.4% of the 55,099 pairs with this exact pattern (US, S2) are matches, so its majority label is no match; this pair agrees. Model p = 0.156: agrees with the label. → not flagged |

#### `S1-570477188` Garrett Chemical — 63 Nahanton Avenue, Milton, MA

Ground truth: singleton (no matches).

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S2-783520744` Garrett Chemical Northside — 67 1/2 NAHANTON AVE, MILTON, MA | **no match** | one word added · only numbers differ · number: one character changed | 1.3% of the 78,331 pairs with this exact pattern (US, S2) are matches, so its majority label is no match; this pair agrees. Model p = 0.000: agrees with the label. → not flagged |

#### `S1-963588162` Maa Hospitality Private Limited — Plot No D-195, Ff Sector 8 Bagdola Dwarka Block D, New Delhi, West Delhi, Delhi

Ground truth: singleton (no matches).

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S3-481869771` मां केयर प्राइवेट लिमिटेड — New Delhi, West Delhi, DL, #C-242 Plot No D-196, Ff Sector 8 Bagdola Dwarka Block D | **no match** | other script: words replaced · words added · number: one character changed | 1.7% of the 240 pairs with this exact pattern (India, S3) are matches, so its majority label is no match; this pair agrees. Model p = 0.000: agrees with the label. → not flagged |

#### `S1-250404567` Integrated Payroll International — Forney, 1220 Falcon Heigh Drive, TX

Ground truth: 1 matched record.

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S2-32283955` Integrated Payroll International LLC — 01220 FALCON HEIGH DRIVE, FORNEY, TX | **match** | legal form / filler only · identical · same numbers | 100.0% of the 214,789 pairs with this exact pattern (US, S2) are matches, so its majority label is match; this pair agrees. Model p = 1.000: agrees with the label. → not flagged |

#### `S1-71142843` Good Logistics Private Limited — Unit-715. B-Wing.Crystal, Plaza 7-Floor New Link Rd, Mumbai, Maharashtra

Ground truth: 2 matched records.

| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |
|---|---|---|---|
| `S2-964683665` गुड लॉजिस्टिक्स प्राइवेट लिमिटेड — UNIT-7-15. B-WING.CRYSTAL, PLAZA 7-FLOOR NEW LINK RD, MUMBAI, Maharashtra | **match** | other script: same words · identical · same numbers | 100.0% of the 86,280 pairs with this exact pattern (India, S2) are matches, so its majority label is match; this pair agrees. Model p = 1.000: agrees with the label. → not flagged |
| `S3-665555605` Good Logistics Private — Unit-715. B-wing.crystal, Plaza 7-Floor New Link Rd, Mumbai, MH | **match** | legal form / filler only · identical · same numbers | 100.0% of the 147,049 pairs with this exact pattern (India, S3) are matches, so its majority label is match; this pair agrees. Model p = 1.000: agrees with the label. → not flagged |

---

## 4. Experiment: does cleaning the training data help?

### 4.1 Setup

- **Holdout:** all pairs of the S1 businesses in fold 0, which is 20% of businesses, never split. That's 441,326 businesses and 1,917,135 pairs. **Holdout labels are never changed**, just like the test set.
- **Training candidates:** the comparable pairs of folds 1–4.
- **Model:** the same LightGBM and 42 features as lens 3. Word evidence is computed without leakage, using only the rows each variant keeps. Every variant trains on a uniform random sample of at most 3,000,000 of its eligible rows (seed 11), so A–D differ in *which* rows are eligible, not in how many are used.
- **Metric:** macro F0.5 over all holdout businesses, singletons included. A business's predicted set is its pairs with p ≥ t, and its true size is its full ground-truth count. F0.5 is reported at t = 0.5, and at the best t on a grid from 0.05 to 0.95. AUC, log loss and calibration are measured on the comparable holdout pairs.

### 4.2 The five variants

| Variant | How its training rows are chosen |
|---|---|
| **A** | everything |
| **B** | remove every pair carrying its pattern's minority label (majorities measured on training rows only) |
| **C** | remove every pair in a pattern whose minority label exceeds 5% |
| **D** | remove pairs a model contradicts: a 2-fold confident-learning pass inside the training data (train on folds 1–2, predict 3–4, and the reverse), with per-class thresholds at each class's mean predicted probability; 4,996 pairs |
| **E** | keep every pair the inner model is unsure about (0.002 < p < 0.998), plus a random 10% of the rest with weight 10, so the class mix is preserved |

### 4.3 Results

| Variant | Training rows | Removed | Holdout F0.5 (t = 0.5) | vs A | Best F0.5 (t) | AUC | Log loss | p in 0.2–0.8: predicted vs actual match rate |
|---|---|---|---|---|---|---|---|---|
| **A** Keep everything | 6,870,136 | 0 | **0.99276** | – | 0.99300 (0.65) | 0.99922 | 0.0255 | 53.3% vs 54.0% |
| **B** Drop pairs against their pattern | 6,630,591 | 239,545 | **0.97683** | -1.59 pts | 0.97688 (0.85) | 0.99056 | 0.2903 | 48.6% vs 46.0% |
| **C** Drop mixed patterns (>5% minority) | 5,569,047 | 1,301,089 | **0.98815** | -0.46 pts | 0.98881 (0.70) | 0.99837 | 0.0393 | 52.6% vs 51.0% |
| **D** Drop pairs a model contradicts | 6,865,140 | 4,996 | **0.99269** | -0.01 pts | 0.99298 (0.65) | 0.99908 | 0.0260 | 53.6% vs 54.3% |
| **E** All hard pairs + 10% of easy pairs, weighted ×10 | 2,040,614 | 4,829,522 | **0.99230** | -0.05 pts | 0.99260 (0.65) | 0.99922 | 0.0258 | 53.4% vs 53.1% |

### 4.4 What the results say

- **B is worse, and not only because of the threshold.** Making every pattern pure in training teaches the model that patterns decide the label outright. On the holdout, where the same patterns are mixed, it is confidently wrong: log loss rises from 0.0255 to 0.2903, AUC falls from 0.99922 to 0.99056, and even its best threshold (0.97688) stays below A at 0.5.
- **C is worse because the model never sees the mixed patterns**, and they still appear in the holdout.
- **D and E are within 0.05 points of A** (D -0.01, E -0.05). With one run each, a gap that small cannot be told apart from sampling noise. E uses 30% of A's eligible training rows, so it is the way to shrink training data when time matters, at that cost.

### 4.5 Caveats

- **The absolute scores are idealised and are not a leaderboard estimate.** Every true match is available as a candidate here, and each unmatched record is compared only with its single closest S1. Only the comparison between variants is meaningful.
- **One run per variant, one seed.** A, D and E cannot be ranked against each other. B and C are worse on every metric at once.

### 4.6 Decision

1. Keep every training label; do not remove the flagged pairs or the mixed patterns.
2. If a smaller training set is needed, subsample the easy pairs with weights (E); on this holdout it cost 0.05 points.
3. Never clean validation or holdout labels, since the test set is labelled the same way.
4. Repeat A against E inside the real pipeline before committing.

---

## 5. What the model must capture

Each insight above points at something the matcher has to see. None of these are rules to hard-code; they are signals for the model to learn from all the labels, which the experiment shows should be kept.

| Signal | Why (insight) | How to give it to the model |
|---|---|---|
| **How the house or plot number changed** | Insights 3, 4 and 8. The number gap is the model's largest source of split gain. | Compare numbers as a multiset (reordered components are equal). Features: the change class (same / added / dropped / cut short / one character / nearby / far-off), the log gap and edit distance of the closest changed pair, and how many numbers were added and dropped. |
| **Which words were added or dropped** | Insights 3, 5 and 9. The same kind of name change goes either way depending on the word. | Word-evidence features (log-odds of each added or dropped word), computed out-of-fold so no row sees its own label. Keep legal forms separate from content words. |
| **Names in another script, compared by sound** | Insight 6, and the Maa Hospitality vs "मां केयर" case (one word replaced across scripts). | Romanise with anyascii, compare sound-alike skeletons, and use the other-script difference classes as features. Add a sound-alike name key to candidate generation. |
| **Addresses that differ by a stray token** | The Maa Care case: `#C-242` made every exact address key differ. | An address key and a comparison that ignore short tokens, alongside the exact ones. |
| **Empty addresses and common names** | Insight 2: identical names without an address are assigned differently. | An empty-address flag and a measure of how common the name is (how many S1 businesses share it). Let the set-level decision return "no match" when several S1s fit equally well. |
| **Combinations, not single rules** | Insights 3–5: each difference alone is weak; the label follows the combination. | Give the difference classes to a tree model as categorical features, and let it learn the interactions. |
| **Pairs outside the easy keys** | Insight 10. | Candidate generation with several complementary keys (exact, prefix, long-word, sound-alike), then measure its recall ceiling. |
| **Calibrated probabilities and set-level decisions** | Section 4: removing conflicting or mixed pairs makes held-out log loss and F0.5 worse. | Train on every label, calibrate on a holdout, and choose each business's set by expected F0.5. |

---

## 6. Everything set by hand

No labels were edited, no record IDs or per-record rules appear in any script, and no test or external data is used. Every hand-set rule and constant is listed here.

**Word lists** (`build_pairs.py`)

| List | Size | Contents |
|---|---|---|
| US states | 51 | 50 states + DC, code → name |
| Indian states and territories | 41 codes → 37 names | e.g. `dl` → `delhi`; some names have two codes (`or`/`od`, `ct`/`cg`, `tg`/`ts`, `uk`/`ut`) |
| Extra state spellings | 8 | orissa, pondicherry, uttaranchal, jammu kashmir, nct of delhi, india, united states, usa |
| Abbreviations | 45 | rd→road, st/str→street, ave/av→avenue, blvd→boulevard, dr→drive, ln→lane, ct→court, pl→place, hwy→highway, pkwy→parkway, cir→circle, n/s/e/w and ne/nw/se/sw→directions, twp→township, hts→heights, sq→square, trl→trail, ter→terrace, expy→expressway, mt→mount, ft→fort, nr→near, opp→opposite, sec→sector, ph→phase, ind/indl→industrial, estt→estate, mkt→market, ngr→nagar, cplx→complex, soc→society, and French r→rue, bd→boulevard, imp→impasse, chem→chemin, all→allee (not in training) |
| Dropped address words | 34 | a, apartment, apartments, apt, bldg, box, building, door, fl, flat, floor, flr, h, hno, house, n, na, nan, no, none, nos, null, number, ofc, office, plot, pmb, po, rm, room, shop, ste, suite, unit |
| Ordinal words | 20 | first…twentieth → 1st…20th |
| Minor name words | 44 | and, co, company, corp, corporation, de, des, dr, du, ei, et, eurl, gmbh, inc, incorporated, la, le, les, limited, llc, llp, lp, ltd, mr, mrs, ms, of, opc, pc, plc, pllc, private, pvt, sa, sarl, sas, sasu, sci, shree, shri, smt, snc, sri, the |
| Sound-alike rules | 9 digraphs + 7 letter maps | digraphs ph sh ch kh gh th dh bh ck → f s c k g t d b k; nasal m before a consonant → n; c q → k, z x → s, j → g, w → v, y → a; vowels removed; doubled letters collapsed |

**Side effects of these lists:**
- State codes are removed before abbreviations are expanded, so `ct`, `ne` and `mt` in an address always count as state codes.
- `n` in the drop list never applies, because it is expanded to `north` first.
- Removing state names makes two identical addresses in different states look the same, on both sides of every pair.
- The sound-alike rules merge some different words (for example `g`/`j`), again on both sides.

**Thresholds and constants**

| Constant | Value |
|---|---|
| Blocking key minimum length | 6 characters (address keys), 4 (name keys) |
| Blocking key skipped if shared by more than | 30 S1 businesses |
| "Long" address words (key 5) | 4 letters or more |
| Lookalike score | `token_set_ratio(name) + token_set_ratio(address)` |
| Typo | edit distance ≤ 1 (words up to 4 letters) or ≤ 2 (longer), both words ≥ 3 letters |
| Nearby number | ≤ 50 apart |
| Pattern majority tie | counts as match |
| Strong model flags | p < 0.02 for a match, p > 0.98 for a no match |
| "Almost nothing in common" match (lens 4) | no shared name word and address Jaccard < 0.25 |
| Variant C cut-off | minority label > 5% |
| Variant E "easy" | inner p ≤ 0.002 or ≥ 0.998; keep 10% with weight 10 |
| Folds | `((s1_id × 2654435761) >> 16) mod 5` |
| Seeds | 7 (audit), 11 (experiment), 21 (examples in this report) |
| LightGBM | binary, 400 rounds, lr 0.08, 127 leaves, `min_data_in_leaf` 200, feature fraction 0.9, bagging 0.7, L2 1.0, 16 threads |
| Training sample cap | 3,000,000 rows |

**Libraries:** pandas, numpy, pyarrow, rapidfuzz (MIT), LightGBM (MIT), scikit-learn (BSD-3), anyascii (ISC). anyascii is not installed in the project's Python; install it to rerun.

---

## 7. Mistakes found and fixed during the work

Each was found by checking results against real records, fixed, and followed by a full rerun. Numbers from before a fix (in chat or in earlier versions of this report) are superseded.

| # | Problem | Effect before the fix | Fix |
|---|---|---|---|
| 1 | The first audit only covered the three reported kinds of case | every other pattern missed | replaced by the general four-lens audit |
| 2 | `MT` was expanded to "mount" before the state check | "Hardin, MT" ≠ "Hardin, Montana" | state codes checked first (caught before any results) |
| 3 | House numbers with letters (`25A`) dropped; ordinals discarded | different addresses looked identical | kept as numbers, and as street words |
| 4 | Only the first number compared, while components come in any order | reordered addresses counted as changed numbers (interim: 387,082 pairs against their pattern) | numbers compared as multisets |
| 5 | Name words compared as sets | "Partners Partners" = "Partners" | names compared by word counts |
| 6 | Model word evidence for training rows included each row's own label | a mild leak (interim AUC 0.99889, 7,269 issues) | word evidence computed out-of-fold |
| 7 | Indic names were one undifferentiated class, and a stray token (`#C-242`) broke every address key | the Maa Hospitality vs "मां केयर" case was never compared, and a word replaced across scripts was invisible | sound-alike name comparison (6 other-script classes), a long-word address key and a sound-alike name key |

---

## 8. Limitations

- **The audit measures inconsistency, not correctness.** Nothing here says which of two conflicting labels is wrong.
- **Matched records are paired with their ground-truth S1.** That works for auditing, but it is not a matching pipeline and cannot be applied to test.
- **Unmatched records are compared with their most similar S1**, which favours lookalikes.
- **Coverage gaps remain** (Insight 10). Decoys whose name *and* address both changed beyond these keys are not found.
- **Normalisation and the sound-alike skeleton are rule-based.** Some variants ("Elm Saint" vs "Elm St", some Hindi spellings) land in broader classes.
- **The experiment's absolute scores are idealised, and each variant was run once.**

---

## 9. Files and how to reproduce

| File | Contents |
|---|---|
| `flagged_pairs.tsv` | every pair flagged by the pattern lens and/or the model lens: label, model p, pattern and its match rate, both IDs, names, addresses, and the added and dropped words |
| `pattern_summary.tsv` | every pattern with pairs, matches, no matches, match rate, majority, pairs against, minority share |
| `duplicate_conflicts.tsv` | records in identical-fingerprint groups with conflicting labels or S1s |
| `assignment_anomalies.tsv`, `s1_duplicates.tsv` | lens-4 anomalies and duplicate S1 businesses |
| `audit.json`, `curation_experiment.json`, `examples.json` | every number and example in this report |
| `report.html` | the HTML version of the audit |
| `work/*.parquet`, `work/*.npy` | intermediate tables, including the model probability and flags for every pair |

Rerun from the project root, after `pip install anyascii`:

```bash
cd eda_output/label_audit && python3 build_pairs.py && python3 analyze.py && python3 curation_experiment.py && python3 examples.py && python3 build_markdown.py && python3 build_report.py
```
