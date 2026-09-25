# Amazon ML Challenge 2026: Business Entity Resolution. Team playbook (72-hour hackathon)

**[DATA]** = measured on the files we were given (scripts in `eda/`; run from repo root with the dataset at `student_resource/dataset` or set `DATA_DIR`). **[RULES]** = README / Unstop listing. **[HYP]** = still to be tested.

---

## ▶ ROADMAP: read this first (living section, updated as work lands)

When anyone asks "what next?", the answer comes from this section. Update the status table and the experiment log after every run.
Design reasoning and decision options: [bottle_necks.md](bottle_necks.md). Research backing each item, with sources: [research_sota.md](research_sota.md).

### R0. Rules of engagement
1. **Smoke test before every full-scale run:** `bash scripts/smoke.sh`. It runs the whole pipeline on ~1% of the data in about 3 minutes, in an isolated folder. No full run on the VM starts until it prints `SMOKE PASSED`.
2. **The decision metric is holdout macro F0.5 after decoding** (fold 5; see §3). Change one thing at a time. Keep a change only if the holdout improves by at least **+0.3 points** and train-on-US / test-on-India (and the reverse) doesn't get worse.
3. **No hand-written country knowledge in new code.**
   - Allowed: structural rules (Unicode, digits, joining initials) and documented closed lists (legal forms).
   - Everything country-specific is learned: from train labels, or from unlabeled test data *inside the pipeline at run time*.
   - Nothing computed on this particular test file may be hard-coded.
4. **Submit to the LB** whenever the holdout improves and `validate_submission.py --check-ids` prints PASS. Log every submission below.
5. Commits carry **no Claude attribution**. Stop the VM when nobody is using it.
6. **Every VM launch uses `PYTHONUNBUFFERED=1`** and logs progress (LightGBM every 50 rounds), so we can always report exact progress.
7. **Licences:** only MIT or Apache-2.0 models may touch the pipeline. That includes retrieval, features, pseudo-labels and distillation. Examples:
   - **not allowed:** jina-embeddings-v3 (CC BY-NC);
   - **allowed:** multilingual-e5, BGE-M3, model2vec, Multilingual-MiniLM, mdeberta, xlm-roberta.
   
   Full list in research_sota.md §3.

### R1. Status

**Leaderboard context (Sep 26, ~00:00 IST):**
- #1 is at 0.987 and **#50 at 0.979**.
- The timer showed about 2 days 3 h left, so the **deadline is around early Sep 28 IST** (verify on the portal).
- **Our blocking v1 caps us at 0.977**: that is the macro F0.5 a perfect matcher would get on our pruned candidates (India 0.965, US 0.985).
- **So recall is the bottleneck for reaching the top 50.** Precision work alone can't get us there.

| Stage | Script | State |
|---|---|---|
| EDA + report | `eda/` | ✅ Done |
| VM (32 vCPU / 251 GB) + data as Parquet | `scripts/setup_vm.sh`, `scripts/tsv_to_parquet.py` | ✅ Done |
| Scorer, B0 exact-key baseline | `src/metric.py`, `src/b0_exact.py` | ✅ Train 0.587, validator PASS. Upload `output/matching_results.tsv` if not done yet |
| Smoke test | `scripts/smoke.sh` | ✅ Passes end to end |
| Normalize v1 (contains hand-written maps; replaced in R3) | `src/normalize.py` | ✅ |
| Blocking v1: word + skeleton TF-IDF, both directions | `src/block.py` | ✅ Train pair recall **0.941**. After pruning (rank_q≤10 or rank_s≤15): **0.937** at 58.7M pairs |
| Equivalence miner v1 (used as features) | `src/mine_equiv.py` | ✅ 266 rules from train, 194 from test (it learned France's region ↔ department pairs without labels) |
| Features, train/test | `src/features.py` | ✅ 58.7M train / 50.3M test pairs, 60 features (31 min / 25 min) |
| LightGBM + calibration + decoding → **B1 submission** | `src/train.py`, `src/decode.py` | ✅ **Holdout 0.9662** (ceiling 0.9768). Validator PASS. File ready to upload |
| **B1: namechar + addr retrievers, union** | `src/block.py --retriever`, `src/union.py` | ✅ train. **Pair recall 0.941 → 0.968 (pruned 0.937 → 0.966). Ceiling 0.9768 → 0.9867** (India 0.9749, US 0.9946). Found only by: word 3.1%, namechar 0.8%, addr 1.1%. Pruned pairs 59M → 110M. Test retrievers running |
| **G3: per-country label-free word roles** | `src/features.py` (`build_role_tables`) | ✅ Full run. What each country's table learned: |

**G3, what the label-free word tables learned (full run):**
- **US:** exactly the decoy vocabulary (southside, eastgate, midtown, riverside, holdings, uptown).
- **India:** mostly transliterated Indian-script words (jvelrs, medikls), a confound.
- **France:** the French decoy words score as decoy-like (participations −1.99, holding −1.99, international −2.01, développement −1.35, groupe −1.31, france −1.13).
- **Risk:** France's scale is shifted (median −0.69), which could make the model over-reject French matches.
- **Watch:** France predicted matches per S1 vs B1's 3.18. If it drops, standardize the scores within each country (G3b).
| **B2 = union candidates + G3 → retrain → decode** | chained on the VM | ⏳ Queued automatically (features → train → decode). ETA ~3 h |
| **D1/D2: exact expected-F DP + soft exclusivity** | `src/efdp.py`, `src/decode.py` | ✅ Written and **smoke-tested**. DP = brute force in 300/300 cases; ~2 min for all of test. Re-decodes B1 as soon as its predictions exist |

### R2. Next steps, in order (owner = suggested; est = wall-clock)

**P0a: recall. Moved up on Sep 26; the ceiling of 0.977 is below rank 50.**

Why blocking misses true pairs (449k missed = 5.9% of true pairs):

| Cause | Share |
|---|---|
| Name in an Indian script | 25% |
| Domain names, handles, typos in numbers | 24% |
| Both similar, but ranked out | 21% |
| Candidate address empty | 18% |
| Business renamed | 12% |

Char 3-gram retrievers (name with spaces removed, name only, address only) target about 75–80% of these.

| ID | Work | Est | Keep if |
|---|---|---|---|
| B1 | **Char 3-gram TF-IDF retrievers**, per country, each direction, added to the union: (a) name+address, (b) name only (for empty addresses), (c) address only (for renamed businesses). Drop n-grams with very high document frequency so the matrix product stays affordable | 2–3 h | Ceiling ≥ 0.985 on train |
| B2 | Wide union → cheap LightGBM reranker → top 40 per S1 | 3 h | Recall@40 ≥ 0.965 |

**P0c: France. Moved up on Sep 26 after the B1 leaderboard score.**
- **The LB/holdout gap implies France ≈ 0.88.** Test is 46.8% India, 38.3% US, 15% France, and 0.468·0.9546 + 0.383·0.9739 + 0.15·F = 0.952.
- **Label-free check:** France's false matches are the organizers' decoys with French words: "Mangeurs Union **Développement** SARL, **3** R. Albert Einstein" vs "Mangeurs Union SARL, **1** Rue Albert Einstein", and likewise "**Participations**", "**France**". The added-word log-odds only know English words ("holdings", "group"), so these are scored as neutral.
- **Worth about +1.2 LB points** if France reaches the US/India level.

| ID | Work | Est | Keep if |
|---|---|---|---|
| G3 | **Per-country word roles** replace word-identity log-odds for added/dropped words. For each token, compute from its own country's records: how often it appears in S1 names, its IDF, and how often it's the "extra" token among same-address pairs. Language-independent | 2 h | Holdout not worse; train-on-US/test-on-India gap shrinks; France predicted "added generic word + number differs" matches drop |
| F1 | **Synthetic French decoys** for validation: from confident French pairs, change the number by ±1–2 and/or add a frequent French S1 word. Measure the rejection rate before and after G3 | 1.5 h | Rejection rate ≥ the train decoy rejection rate |

> **Sep 26, ~03:00 IST:** the union ceiling is 0.9867. At B1's efficiency (98.9%), that means a holdout of ~0.976. India (0.975) is now the recall bottleneck.
> The next recall step is model2vec for Indian-script names, or the wide-union reranker.

**P0b: decoding. Cheap, directly scored; needs only cached predictions (`decode.py`).**

| ID | Work | Est | Keep if |
|---|---|---|---|
| D1 | **Exact expected-F0.5 DP.** Poisson-binomial over prefix and suffix plus a Poisson term λ for blocking misses (research §4). Replaces the ratio approximation | 1–2 h | ≥ +0.3 on holdout, bootstrap CI > 0 |
| D2 | **Soft exclusivity** p′ = o/(1+Σo) vs the current hard argmax | 0.5 h | Better of the two |
| D3 | **Label-shift diagnosis:** mean Σp per S1 on test vs train, per country. Set λ from it | 0.5 h | Diagnostic |

**P2: generalization / remove the hand-written maps (the France fix).**

| ID | Work | Est | Keep if |
|---|---|---|---|
| G1 | **normalize v2 = structural rules only.** Delete the US/India state and street maps. Use uroman (MIT, needs attribution) for non-ASCII tokens, falling back to anyascii | 2 h | Holdout within −0.1 of v1 (a neutral result is a win: the code becomes generic) |
| G2 | **Miner v2:** align differing tokens (DP on character similarity), score with G² + NPMI, keep one-to-one two-way links, plausibility-based support thresholds, drop rules that fire on hard negatives. Train labels + test anchors, recomputed at run time | 3 h | Explained-difference features gain importance; holdout +0.2 or more |
| G3 | **Per-country frequency-based word-difference features** (a token's own-country IDF/frequency role) replace the word-identity log-odds | 2 h | The train-on-one-country gap shrinks, holdout not worse |
| G4 | **House-number roles** from how many different S1 names share the number in a locality; match levels (exact / suffix / typo / missing / different) | 2 h | Fewer false positives in the ≥92 similarity band |

**P3: France.**

| ID | Work | Est | Keep if |
|---|---|---|---|
| F1 | **Synthetic French hard negatives** built like the organizers' generator: confident French anchor pairs with the number changed or a business/direction word added. Use them for calibration and stress tests | 2 h | France's predicted singleton rate and matches per S1 move toward US/India levels |
| F2 | Pseudo-labels on French pairs (p > 0.98, mutual best, same number), disclosed | 1 h | LB improves |

**P4: stretch goals, only after P0–P2 have been submitted.**

| ID | Work | Est |
|---|---|---|
| S1 | Cross-encoder (Multilingual-MiniLM, MIT): train on the laptop GPU on S1 disjoint from the LightGBM sample, export ONNX int8, score the top 3–5 per S1 on the VM, use as a feature | 4–6 h |
| S2 | S2↔S3 twins as weight-2 units in the decoder | 2–3 h |
| S3 | Train on 60% instead of 30% of S1 | 1 h (run) |

**Final phase (last ~12 h): no new features.**
1. Freeze.
2. Regenerate from raw data with one command and check the output matches the submitted file.
3. Run the validator (`--check-ids`).
4. Zip: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, methodology doc.
5. Final upload at least 3 h before the deadline.

### R3. Suggested allocation
- **P1 (validation + decoding):** D1–D3, bootstrap CIs, experiment log, submission sign-off.
- **P2 (pipeline + VM):** B1–B2, full runs, submissions, final reproduction and zip.
- **P3 (features):** G3, G4, then S3.
- **P4 (normalization + France):** G1, G2, F1, F2, methodology doc. S1 on the laptop GPU if time allows.

### R4. Experiment and submission log

| ID | Change | Holdout macro F0.5 | Pair recall | LB | Notes |
|---|---|---|---|---|---|
| B0 | Exact key (country + name tokens + first number) | 0.587 (all train) | 0.371 | – | Precision 0.982 |
| B1 | Blocking v1 (word) + 60 features + LightGBM (1,642 rounds) + isotonic + **soft exclusivity + threshold 0.55** | **0.9662** (India 0.9546, US 0.9739; singletons 0.9818, non-singletons 0.9652) | 0.937 | **0.952** | Ceiling 0.9768, so the model reaches 98.9% of it. Details below |

B1 details:
- **Soft vs hard exclusivity:** soft beats hard by +0.0008.
- **Exact DP vs plain threshold:** the DP (0.9657) did **not** beat the threshold (0.9662). The likely cause is dependence between candidates, since the DP assumes independent probabilities.
- **Label-shift check:** the mean Σp per S1 is similar on test (France 3.34, India 3.14, US 3.42) and on the holdout (3.25). So **no strong shift**; the earlier estimate of ~4.1 matches per S1 is not supported.
- **France:** predicted 3.18 matches per S1 and 5.6% empty, in line with US and India. France looks healthy.
- **Top features by gain:** gap_c, rank_q, is_c_best, add_lo_sum, add_lo_min, num_rel.

---

## 0. Key facts (all confirmed)

- **Format [RULES]:**
  - A **72-hour hackathon, 25–27 Sep 2026.** Teams of 3–4.
  - The public LB is live and uses a subset of test; the private LB uses the rest.
  - **Ties go to the earlier submission.**
  - **The top 500 at the 48-hour mark get +$100 AWS credit.**
  - Deliverables: the scored `matching_results.tsv`, a zip (code, `candidate_pairs.tsv`, README, requirements) and the methodology doc. Unstop says 1–2 pages; the README says there is no page limit, so aim for about 2 dense pages.
- **Metric [RULES]:** F0.5 per S1 entity, macro-averaged over all S1, singletons included:
  - The score for one entity is `1.25·TP / (|pred| + 0.25·|true|)`.
  - **Singleton:** an empty prediction scores 1, anything else scores 0.
  - **Non-singleton predicted empty:** TP = 0, so the score is 0.
  - With 4 true matches: predicting 3 correct scores 0.94, while 4 correct + 1 wrong scores 0.83. **A false positive costs about 2.7× a miss.**
- **Rules [RULES]:**
  - The final model must be MIT or Apache-2.0 licensed and ≤8B parameters. The rule is about the *model*, so utility libraries under other permissive licenses are fine.
  - **No external lookups:** no geocoding, registries, internet data or gazetteers.
  - Hand-written generic maps (street abbreviations, state codes, legal forms) are domain knowledge and fine. Document them.
  - Blocking and IDF over the unlabeled test inputs are unavoidable and fine. Pseudo-labels are a grey area: use them only if they clearly help, and disclose them.
- **`candidate_pairs.tsv`** must be the exact set the final model scores. It gets audited for recall ceiling and reduction ratio, so keep it around 20–40 candidates per S1.

---

## 1. What we are building (send this to teammates)

For each of the 1.73M test S1 businesses, we retrieve about 30 plausible records from about 10M S2/S3 records. Retrieval is done per country, over normalized and transliterated text, in both directions.

A LightGBM model scores each pair. Its most important inputs are **the relationship between the two house numbers** and **which name words were added or dropped**. The final step picks each S1's output set to maximize expected F0.5. That step enforces that each S2/S3 record belongs to at most one S1, and it answers "no match" when that is the better bet.

The competition is decided by blocking recall, rejecting the constructed "neighbour" negatives, and set-level decoding. Model size does not decide it.

---

## 2. The data: what matters

| | Train | Test |
|---|---|---|
| S1 | 2,206,821 (US 1.32M / India 0.88M) | 1,732,544 (India 810k / US 663k / **France 259k**) |
| S2 / S3 | 5.03M / 5.29M | 4.89M / 5.08M |

**Label structure [DATA]:**
- 5.58% of S1 are singletons, the same in both countries and across ID ranges. S1 averages 3.46 matches: 0–5 from S2 and 0–6 from S3.
- **Each S2/S3 record matches at most one S1.**
- Matches never cross countries.
- About 26% of train S2/S3 records are unmatched.

**Test shift, now resolved [DATA]:**
- Test has 5.75 S2/S3 records per S1, against 4.68 in train.
- The similarity distribution of each record's best S1 candidate is nearly the same in train and test. Reweighting by train match rates estimates that **71–74% of test S2/S3 records are true matches (train: 74%).**
- So the extra records in test mean **more matches per S1 (about 4.1–4.2), not more distractors.** The decoder should expect *larger* true sets, and we must not over-tighten it.

**Sources differ in style [DATA]:**
- **S1:** clean, all Latin script, never an empty address.
- **S2:** registry style. 67% of addresses are UPPERCASE, with prefixes like `H.NO` and `##`. 9% of names are in Indic scripts (5% Devanagari). 3.4% of addresses are empty.
- **S3:** directory style. Full state names, Indic-script state names, `NULL`/`N/A`, `CITY` suffixes, typos. 3.3% of addresses are empty.
- Mojibake is rare; ftfy fixes it.

**How positives differ from S1 [DATA]:**
- 82% of positives have name and address similarity both ≥80. Only **3.3% are renamed** (name similarity <50: "Ectokor", "lucknowindiacom", "X aka Y"), and 1.9% have an empty address.
- Name tokens **added** in positives: dba/aka/formerly/fka/com/www, center, services, lp, smt/sri/shri/mr, plus typos and transliterations ("praivet").
- Name tokens **dropped** in positives: legal forms (limited, private, llc, incorporated, llp).
- House number relationship in positives: **equal 65%**, candidate drops the number 12%, **differs 10%** (typos such as 31→30, 1600→160, 13741d), candidate adds a number 10%.

**The constructed negatives (the main challenge) [DATA]:**
- 44% of unmatched records have a near-identical address (similarity ≥90) to some S1, and 36% also have a similar name. They are **"neighbour businesses"**. Only 16% are unlike any S1.
- For unmatched records whose address is near-identical to their nearest S1 (similarity ≥85), the house number relationship is **different in 88.2%, equal in only 2.5%**, and the candidate adds a number in 8%.
- These negatives usually **add a business-type or geographic word:** holdings, group, enterprises, industries, exports, public, overseas, ventures, infratech, and north/south/east/west, uptown/downtown/midtown, metro, central, valley, harbor, summit, eastgate/westgate. Examples: "UD International **Holdings** Ltd" at "NO D/94 …" vs "UD International Ltd"; "Comité des Nou **Développement**, 16 Rue Victor Martel" vs "Comité des Nou, 15 Rue Victor Martel".
- **Even at the highest similarity level (≥92), 14% of records whose best S1 looks nearly identical are negatives.** That band decides precision.

**France [DATA]:**
- It follows the same generator:
  - S2 is uppercase;
  - abbreviations (R/R./AV./BD/IMP, N°, Bis);
  - the region is swapped for the department and back: Nord / Pas-de-Calais ↔ Hauts-de-France, Gironde ↔ Nouvelle-Aquitaine, Loire-Atlantique ↔ Pays de la Loire;
  - accent noise ("Fàmilles", "SÊCTION");
  - legal forms swapped (SARL/SAS/SASU/EURL/SCI/SA/EI);
  - **the same neighbour-negative pattern:** a word such as "Développement" or "Participations" is added and the house number changes.
- France covers only about 15 cities in 3 regions (Lille, Roubaix, Tourcoing, Dunkerque, Calais, Bordeaux, Pessac, Mérignac, La Teste-de-Buch, Lège-Cap-Ferret, Nantes, Saint-Nazaire, Saint-Herblain, Pornic, La Baule). That means **very dense streets**, so house-number features matter even more there.
- Name vocabulary repeats heavily (sarl, sas, club, ecole, amicale, comite), which makes names weaker evidence than in US/India.

**Leakage [DATA]: none.** IDs, row order and file positions carry no signal, and train and test share no entities. 34% of test S1 names also appear in train, but that is the generator reusing vocabulary, not overlap.

**Blocking probe [DATA]:** crude word-level IDF retrieval (S2/S3 → S1) gets R@1 76–89% and R@50 88–95%, and only 77% for Indic-script names. Char n-grams, transliteration and extra retrievers have to close that gap.

---

## 3. Validation (P1 owns)

For 72 hours we use a **single holdout, not 5-fold.** **As implemented,** train S1 entities are split by `int(id[3:]) % 10` (IDs are random, so this is effectively a stratified random split):

| Folds | Share | Use |
|---|---|---|
| 0–1 | 20% | **Stats:** word log-odds and mined equivalences. Never used to train or evaluate the model |
| 2–4 | 30% | **LightGBM training** |
| 5 | 10% (~220k S1) | **Holdout:** early stopping, decoder tuning, **the score we trust** |
| 6–9 | 40% | **Calibration** (isotonic) and competitors for exclusive assignment |

- **Blocking runs on the full training universe** (all S1 + all S2/S3), so density is realistic.
- **The score we trust:** macro F0.5 on the holdout **after** exclusive assignment + decoding. Pair-level AUC and F1 are only diagnostics.

**Always report:**
- the **blocking ceiling** (the F0.5 a perfect matcher would get on our candidates);
- a split by country × source × singleton;
- the **leave-one-country-out** gap (train on US → score India, and the reverse), which stands in for France.

**Label-free checks on the test predictions, per country:**
- predicted singleton rate: expect a few percent;
- mean predicted matches per S1: expect about 3.5–4 (true is about 4.1, and we lose some to precision);
- share of test S2/S3 records assigned: expect about 60–70% (true is about 73%).

If France is clearly off from US/India, France is broken.

**Public LB:** use it as a check on France and on distribution shift. If the holdout and the LB disagree on *direction*, stop and investigate before chasing the LB.

---

## 4. Pipeline and models

**Normalize (shared library):**
- ftfy → NFKC → anyascii (romanizes the Indic scripts; ISC license) → lowercase.
- Strip `null|n/a|nan|#|ndeg|no.|h.no|door no`.
- Map abbreviations: US street types, `R/AV/BD/IMP/CHEM/ALL`.
- Map state/region equivalences: TX↔Texas, KA↔Karnataka, दिल्ली→Delhi via anyascii plus a small map, Nord↔Hauts-de-France, and so on. **Mine these pairs from matched train pairs and test co-occurrence where possible.**
- Convert ordinal words to digits (Fourth→4th).
- Strip leading zeros.
- Produce three derived fields:
  - `name_core`: the name with legal forms and honorifics removed;
  - `nums`: the ordered house, unit and slash numbers;
  - `locality`: the city and area tokens.

**Blocking: a union of retrievers, per country, capped at about 30–40 per S1:**
1. Char 3-gram TF-IDF over `name_core + address`, top-k in **both directions**. S2/S3 → S1 is sharp, because each record has at most one true S1. S1 → S2/S3 covers S1s with many matches. Use `sparse_dot_topn`.
2. Address-only TF-IDF, for the 3.3% of renamed businesses.
3. `name_core` within the same locality token, for the 1.9% with empty addresses.
4. (If there is time) multilingual-e5-small (MIT) + faiss, for cross-script names.

Target: **ceiling ≥ 0.97.**

**Matcher:** a LightGBM on about 60–100 features. First a cheap LightGBM prunes each S1's candidates to about 20. Those pruned candidates are what goes into `candidate_pairs.tsv`.
- **Number relationship (most important):**
  - equal / added / dropped / differs;
  - for differing numbers: digit edit distance, prefix or substring (160 vs 1600), absolute difference;
  - equality of the first number, equality of the unit/slash parts;
  - whether the candidate has extra unit detail.
- **Name differences:**
  - the added and dropped token sets, scored with **log-odds learned on the training split** (holdings, group or north suggest a negative; dba, aka or center suggest a positive);
  - legal-form change;
  - `name_core` similarities: rapidfuzz ratio / token_set / partial / Jaro-Winkler, char TF-IDF cosine.
- **Address:** address similarities, locality equal, state equal after mapping, empty flags, source flag, Indic-script flag.
- **Context:** rank and score gap for the candidate within its S1, rank of this S1 within the candidate's options (reverse rank), mutual best, how many S1 share this street and number.
- **Do not use `country` as a feature.**

**Decoding (P1):**
1. Isotonic calibration.
2. **Exclusive assignment:** each S2/S3 goes to its highest-probability S1.
3. **Expected-F0.5 top-m selection per S1:** calculate `E[1.25·X/(m + 0.25(X+Y))]` exactly, with X and Y as Poisson-binomials over the selected and unselected probabilities. Include m = 0, whose expected score is `P(no true matches)`.
4. Correct the test prior: the test entities have about 20% more matches, so check that the decoded set sizes are consistent with that.

**Stretch goals, only once the core has been submitted:**
- **A cross-encoder** (xlm-roberta-base or mdeberta-v3-base, MIT), trained on the laptop 4060 on hard pairs from the LightGBM, and run only on the uncertain band (p between 0.15 and 0.85).
- **S2↔S3 twin linking,** so that records with empty addresses inherit evidence from their twins.

**Not doing:** LLMs, metric learning, libpostal (its model is trained on OpenStreetMap, which counts as external data), 5-fold CV, and big hyperparameter searches.

---

## 5. What separates a top solution (in priority order)

1. **Number-relationship features plus a model of the added words.** They separate neighbour negatives from typo-number positives. This is the single biggest precision lever: 14% of the pairs that look nearly identical are negatives. It transfers to France, because France was built the same way.
2. **Set-level decoding + exclusive assignment,** instead of one global threshold. Include "no match" as an option (m = 0).
3. **A blocking ceiling ≥0.97:** both directions, plus address-only and locality-name retrievers, plus transliteration.
4. **France handled on purpose:**
   - French abbreviation, legal-form and region maps;
   - log-odds of added words **re-estimated on France test pairs without labels**: use pairs where the address and number are equal as stand-in positives, and pairs where the number differs as stand-in negatives;
   - check the per-country match-count statistics.
5. **Getting on the board early.** Ties go to the earlier submission, and the 48-hour cutoff pays out.

---

## 6. Four people, integrated through shared files

**Shared file formats (Parquet), fixed in hour 1:**
- `records_norm`: id, src, country, name_norm, name_core, addr_norm, nums, locality, flags
- `candidates`: s1_id, cand_id, retriever flags, scores, ranks in both directions
- `features`
- `preds`: s1_id, cand_id, p

| | P1: validation + decoding | P2: blocking + pipeline | P3: features + LightGBM | P4: normalization + France + stretch |
|---|---|---|---|---|
| Owns | Scorer, holdout, ceiling, leave-one-country-out, calibration, exclusive assignment, expected-F0.5 decoder, per-country checks on test, submission sign-off | Cloud VM, conversion to Parquet, TF-IDF retrievers in both directions, union, full test runs, output TSVs + validator, reproducible zip | Number and name-difference features, context features, pruning + main LightGBM | Normalization maps (US/IN/FR), transliteration, France log-odds without labels, then the cross-encoder or twin linking. Writes the methodology doc |
| First 3 hours | Scorer + holdout; test the decoder on synthetic probabilities | VM up, data to Parquet, first TF-IDF blocker | Feature functions on sample pairs from the probe | Normalizer v1, used by everyone |

- **Integration:** the full pipeline (P2) calls the modules owned by P3 and P1. P2 generates submissions and P1 signs them off.
- **Sync:** a 10-minute check-in every 6 hours. **Shifts:** two people are always awake. P2 and P1 cover opposite halves of the night, so every overnight job has an owner.

---

## 7. Experiments (in priority order; keep a change only if the holdout macro F0.5 improves after decoding)

> E1–E8 and E10 are done or built into B1. **The live priority list is ROADMAP R2 at the top.** This table is kept for reference.

| ID | Question | Keep if / act on |
|---|---|---|
| E1 | Scorer sanity: all-empty should score 0.056, perfect ground truth 1.0 | Exact match with those values |
| E2 | Ceiling of char 3-gram TF-IDF, k = 10/20/40, each direction and the union | Choose k with ceiling ≥0.96 |
| E3 | Normalization ablation: anyascii, state maps, ordinals, NULL strip | ≥ +0.3 point on the ceiling for each step |
| E4 | Baseline LightGBM (20 features) + a global threshold → **first LB submission** | This becomes the reference |
| E5 | Expected-F0.5 decoder + exclusive assignment vs the global threshold | ≥ +0.5 point |
| E6 | Number-relationship features | A big drop in false positives at the highest-similarity level |
| E7 | Added/dropped token log-odds + legal-form change | Better singleton accuracy |
| E8 | Context features (reverse rank, mutual best, gap) | ≥ +0.3 point |
| E9 | Leave-one-country-out with and without token-identity features | Drop the features that widen the gap |
| E10 | Address-only + locality-name retrievers | Ceiling +0.5 point at ≤ +10 candidates |
| E11 | France maps + log-odds without labels; check France's predicted singleton rate and matches per S1 | Statistics move toward US/India levels and the LB rises |
| E12 | Size of the training sample (30% → 60%) | ≥ +0.2 point |
| E13 | (Stretch) Cross-encoder on the uncertain band, blended | ≥ +0.5 point on the entities it touches |
| E14 | (Stretch) S2↔S3 twin linking | Recall up with no precision loss |

---

## 8. Compute

- **Cloud, from hour 1:** a high-memory CPU VM (64–128 GB RAM, 16–32 vCPUs) for blocking and features at full scale. RAM is the bottleneck, not the GPU. Process one country at a time, and cache every stage as Parquet, named by a config hash, so a crashed run resumes.
- **The 4060 laptop:** e5 embeddings (fp16), cross-encoder fine-tuning (sequence length ≤128, fp16, batch size 16–32). First confirm that `torch.cuda.is_available()` returns True.
- **No A100 or H100 is needed.** Every participant gets $200 of AWS credits, and the top 500 at 48 hours get +$100. Together with the GCP credits, that is plenty for CPU VMs.

---

## 9. The 72 hours (T0 = when we start; check the exact deadline on the portal)

> Original plan. **Actual progress and the current order of work live in ROADMAP R1–R2.** Keep the milestones below: a strong submission before T48, freeze in the last ~12 h.

| Window | Goal | Exit criteria |
|---|---|---|
| T0–T3 | Repo, file formats, VM, Parquet, normalizer v1, scorer, holdout | Everyone has read access to Parquet data and the normalizer |
| T3–T12 | TF-IDF blocking (both directions) → baseline LightGBM → holdout score → **run on full test → first LB submission** | A valid submission on the LB, with holdout vs LB compared |
| T12–T30 | E5–E8, E10: decoder, number and name-difference features, context features, extra retrievers | ≥2 more submissions, each better on the holdout |
| T30–T46 | E9, E11, E12: France work, pruning, a larger training set; stretch work starts on the laptop | **A strong submission before T48** (the top-500 credit cutoff) |
| T46–T62 | Stretch blend if it earns its place; final calibration; **regenerate everything from raw data on a clean VM** | The reproduced output matches the submitted file |
| T62–T72 | Freeze. Validator (`--check-ids`), per-country checks, zip, methodology doc, final upload **at least 3 hours before the deadline** | Package done |

**Last 24 hours:** no new features. Only re-decode and recalibrate. One owner (P2) regenerates everything and one reviewer (P1) signs off.

---

## 10. Failure modes and how to prevent them

- **TSV parsing:** always read with `sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False, dtype=str`.
- **Pair-level metrics:** they hide singleton false positives. Only trust the decoded macro F0.5.
- **Treating "same address" as a match:** that is exactly how the constructed negatives were built. Always condition on the number relationship and on the added words.
- **Treating "number differs" as a hard reject:** 10% of positives have number typos. Let the model learn it.
- **Over-tightening for test:** test has *more* matches per S1, not more distractors.
- **Country-specific token features:** they collapse on France. Gate them with leave-one-country-out.
- **The same S2/S3 under several S1:** exclusive assignment prevents it.
- **Output errors:** a missing France row, S1 IDs inside match lists, CSV instead of TSV, or a candidate file that doesn't cover every match. Write rows by iterating over the `test_source1` IDs, and always run the validator.
- **Wasting time:** running the cross-encoder on every pair, 5-fold CV, hyperparameter searches, or building before the ceiling is measured.

## 11. Methodology doc outline (P4 drafts from T46)

1. EDA insights: the generator's noise operations and the neighbour-negative pattern.
2. Blocking: retrievers, k, ceiling, reduction ratio.
3. Features and model.
4. Decoding: exclusive assignment + expected-F0.5.
5. France handling.
6. Holdout vs LB; error analysis.
7. License list: LightGBM (MIT); any transformer used (MIT); anyascii (ISC), used as a library.
