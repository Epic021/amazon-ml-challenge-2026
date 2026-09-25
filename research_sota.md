# State-of-the-art research against `bottle_necks.md`

This is a companion to `bottle_necks.md`. It holds research findings on each component, suggested changes to the Decision Register, rule and licence flags, and a reading list for each component.

**Tags:**
- **[VERIFIED]** was checked locally, by running code or doing the maths.
- **[EST]** is an estimate, not a published number.
- **[CHECK]** is a source we could not confirm and should verify before quoting it in the methodology doc.

---

## 0. Top 10 changes, ranked by expected gain per hour

| # | Change | Stage | Why | Cost |
|---|---|---|---|---|
| 1 | **Retrieve wide, then rerank down to 40.** Pool ~200–300 candidates per S1 from all retrievers, prune them with a light LightGBM ranker, and write the top 40 to `candidate_pairs.tsv`. | 5 | At 40 out of ~10M, the reduction ratio is ≈1 whatever we do. Recall@40 after the rerank is the only audited number that moves. This is what the Foursquare 2022 winners did. | 3–4 h |
| 2 | **Compute expected F0.5 exactly** (a Poisson-binomial DP plus a Poisson term for missed matches) in place of the ratio-of-expectations approximation in `src/decode.py` | 8 | The top-k theorem makes the search n+1 sets. The DP was verified against brute force. | 2 h |
| 3 | **Soft exclusivity** instead of hard argmax. `matching_results.tsv` is one row per S1 with an ID list, so hard exclusivity is not required by the format. | 8 | Per-S1 decoding on renormalised p′ is Bayes-optimal for a metric that is separable per S1. Hard argmax throws away information. | 1 h |
| 4 | **Character 3-gram BM25/TF-IDF retriever** on name+address and on address alone (spaces removed) | 5 | Sparkly (VLDB 2023): 3-grams were the best tokeniser, and top-k BM25 beat 8 blockers, deep ones included. | 2 h |
| 5 | **Type each number by locality cardinality**: how many distinct S1 names share this number within the same locality or postcode | 3b | This was the strongest signal for postal, road or zone numbers vs house numbers, and it needs no language lists. | 2 h |
| 6 | **Uroman for non-ASCII tokens**, plus **cross-script equivalences mined from train matches** | 1, 4 | uroman handles schwa deletion (`bareli`, `chennai`); anyascii does not (`breli`, `cenni`). Only mining fixes exonyms like `dilli`→`delhi`. | 2–3 h |
| 7 | **Synthetic French hard negatives** that copy the organisers' generator: take confident French anchor pairs, change the house number, or add a business-type or direction word | 6–7 | Calibrates and stress-tests the model on France without labels. | 2 h |
| 8 | **Country-transferable name-difference features**: IDF and frequency of each added/dropped token computed per country from that country's own records, instead of raw token-identity log-odds | 6 | Token identities (`holdings`) don't carry over to French (`groupe`). Frequency and role do. | 2 h |
| 9 | **Small multilingual cross-encoder** run on the top 3–5 candidates per S1, used as a LightGBM feature (out-of-fold) | 6–7 | This is the Foursquare pattern. It covers French and native scripts. Expected +1–3 points in the ≥92 similarity band [EST]. | 4–6 h, GPU |
| 10 | **Fine-tuned multilingual-e5-small bi-encoder** as an extra retriever | 5 | SC-Block: supervised contrastive training beats BM25 at small k. Off-the-shelf embeddings probably won't fix the 77% recall on native scripts. | 4–6 h, GPU |

---

## 0b. Compute plan: VM with 32 vCPU / 256 GB RAM / 200 GB disk, no GPU

**Bottom line: no GPU is needed for the core pipeline.** Top-10 items 1–8 all run on CPU. Only #9 (cross-encoder) and #10 (fine-tuned e5 bi-encoder) want a GPU. Both are optional extras, worth +1–3 points [EST].

| Item | Runs on | Notes |
|---|---|---|
| ftfy / uroman / anyascii | CPU | Dedupe unique strings or tokens first, then `Pool(32)` |
| TF-IDF / BM25 3-gram retrieval | CPU | `sparse_dot_topn` with `n_threads=32`, per country, chunked |
| model2vec static embeddings (the dense vote in the union) | CPU | Embedding lookups only; ~24M strings take minutes. **Use this in place of e5.** |
| Wide union (≤300 per S1) + reranker | CPU | See the budget below |
| Mining, features, LightGBM, isotonic, DP decoder | CPU | Numba DP: seconds to a minute [EST] |
| #9 Cross-encoder (Multilingual-MiniLM) | **GPU for training** | Train on the **laptop RTX 4060** (~1M pairs in ~10–20 min [EST]). Export to ONNX int8 and run inference on the VM CPU. |
| #10 Fine-tuned e5-small retriever | **GPU for training *and* encoding** | Encoding ~12M test strings on CPU is ~1–2 h [EST], and train needs as much again. **Skip unless we get a GPU instance.** model2vec covers the cheap version. |

**Cross-encoder without a VM GPU (recommended route for #9):**
1. **Train on the laptop GPU.** Use S1 entities *disjoint* from the LightGBM training sample, e.g. another 30% of train S1. Then its scores on LightGBM's rows are already out-of-sample, with no K-fold retraining.
2. **Export and move.** Export ONNX int8 (a file of about 100 MB [EST]) and copy it to the VM.
3. **Score on the VM CPU.** Score the top 3 per S1:
   - test: 1.73M × 3 ≈ 5.2M pairs;
   - train sample plus holdout: about 2–3M pairs.

   At ~1–2k pairs/s [EST], that is **1–2 h per full run**. It breaks the ≤1 h budget in D20, so run it once and cache the scores.

**Optional GPU instance.** A single mid-range GPU for 4–6 h (L4 or A10G class, 24 GB) would make both #9 and #10 comfortable. Get it only if the core pipeline is done and the cross-encoder shows a gain on the holdout.

**rapidfuzz throughput** [VERIFIED on the laptop, 1 core, ~40-character strings]:

| Scorer | Pairs/s per core |
|---|---|
| `ratio` | 2.5M |
| `token_set_ratio` | 0.32M |
| `partial_ratio` | 0.15M |

On 32 vCPUs (about 16–24 effective cores), each `token_set_ratio` feature over 500M pairs takes about 1–2 min. So a **300-wide union is CPU-feasible** for cheap features. The real risk is Python-level number and token logic: vectorise it, or run it only on the top 40.

**Memory and disk budget:**
- **Wide-union reranker:** 1.73M × 300 ≈ 520M test pairs × (2 int32 IDs + ~20 float32 features) ≈ 46 GB. Compute it in RAM per country and **never write it to disk**. Only the top-40 IDs are written.
- **Full feature matrices at top 40:**

  | Matrix | Pairs | Size (~100 float32 features, before compression) |
  |---|---|---|
  | Test | 69M | ≈ 28 GB |
  | Train sample (30%) | 26M | ≈ 10 GB |

- **Disk rules** (200 GB is enough if we're disciplined):
  - Parquet with zstd;
  - float16 for similarity features;
  - keep at most 2 feature versions;
  - delete intermediate candidate sets.
- **Largest fixed costs:**
  - raw data plus Parquet: < 10 GB;
  - the Python environment with CPU torch: ~3 GB;
  - models: < 2 GB.

---

## 1. Decision Register: research-backed updates

| ID | Current recommendation | Research says | Suggested decision |
|---|---|---|---|
| D1 | anyascii | anyascii drops every inherent "a" (`breli`, `tmilnadu`, `cenni`). **uroman** (MIT plus an acknowledgement clause) gives `bareli`, `cennai`, `tamilnadu`, `lakhnau`, `kumar` once doubled letters are collapsed. It is 2.8 ms per string, so run it only on unique non-ASCII tokens. IndicXlit (MIT, 11M params) is the best neural option, but installing fairseq is painful. [VERIFIED on sample] | **uroman on unique non-ASCII tokens, anyascii as fallback, then cross-script equivalences mined from train matches** |
| D2 | Drop the marker, keep tokens with low weight | No contrary evidence | Keep |
| D3 | Learned (b) + positional (c) + list (a) | Add **locality cardinality**, the strongest signal. Production matchers (UK MoJ `uk_address_matcher`) key each number by role and penalise a *same-role contradiction* much more than a *missing number*. | (b) + (c) + cardinality. Build a per-number feature vector rather than a tagger. A CRF is a stretch goal. |
| D3b | Whole number and its parts | Agreed. Also encode match levels: exact / same integer but different suffix / suffix missing / range overlap / digit typo (edit distance 1, transposition, prefix) / missing / different. | Keep, and add the match levels |
| D4 | Learned optional tokens + fuzzy fallback | Seed the list from **cleanco**'s legal-term lists (MIT). Fuzzy-match only the last 1–3 tokens (similarity ≥ 0.8). **Keep legal form as a feature** (equal / conflict / missing) rather than silently deleting it. | Keep, plus the seed list and legal-form flags |
| D5 | Digits inside words only | No contrary evidence | Keep |
| D6 | Variants, max similarity | Foursquare and Linacre agree | Keep |
| D7 | Character n-grams, no dictionary | Also add a **corpus joiner**: if `w1+w2` exists as a corpus token, emit the joined form. A splitter (SymSpell/wordninja) is fine only with a **corpus-derived** frequency file. wordninja's bundled Wikipedia model counts as external data. | Character n-grams + corpus joiner |
| D8 | Word + skeleton; add 3-grams if the ceiling is below 0.95 | Sparkly found 3-grams best (2- and 4-grams worst), IDF essential and TF irrelevant on short fields, and top-k better than thresholds. | **Add 3-grams now**, not conditionally |
| D11 | Yes | Agreed. For name-in-locality, drop tokens with document frequency above ~1% of the country and cap the block size. | Keep |
| D12 | Only if native-script recall stays low | Off-the-shelf embeddings won't fix crude transliteration. Supervised contrastive fine-tuning (SC-Block) does. model2vec `potion-multilingual-128M` (MIT) is a cheap CPU vote. | **model2vec on CPU** (the VM has no GPU). Fine-tuned e5-small only if we get a GPU instance. |
| D13 | Tune on the recall curve | Allocate candidates by each retriever's *marginal* recall over the union. Use an adaptive k per S1 from ranker scores, with more for empty-address and native-script queries. Sparkly found both directions gave little recall gain; check this on our own curve. | Wide union → rerank → 40 |
| D14 | Train split + test anchors, gated by country shift | Replace token-identity log-odds with per-country IDF/frequency buckets (optional / discriminative / unknown). Use Monroe "Fightin' Words" log-odds with a Dirichlet prior where log-odds remain. Validate leave-one-country-out (India→US and US→India). | Frequency-based features + leave-one-country-out gate |
| D15 | 30% | No evidence either way | Keep |
| D16 | Try both | Monotone constraints only on clearly directional features (`monotone_constraints_method='advanced'`). With calibrated p and proper decoding, **FP weighting ≈ moving the threshold**, so pick one and calibrate after. | Monotone on ~6 features, **no FP weighting** |
| D17 | Expected-F0.5 top-m, m = 0 allowed | Confirmed by Ye et al. 2012 (top-k theorem). Use the **exact DP** (Algorithm §4). Sanity check: at large n the optimal threshold is p ≥ **0.8·F\***, i.e. F\*/(1+β²). [VERIFIED by simulation] | Exact DP |
| D18 | Stretch goal | Treat a twin pair as one unit, a weight-2 Bernoulli in the DP. Don't take the transitive closure for the final output, because chains over-merge. Gain estimated at 0.5–1.5 points [EST]. | Stretch goal. Do it after D17 and soft exclusivity. |
| D19, D20 | Yes / ≤ 1 h | No new info | Keep |

**New decisions to add:**
- **D21** Wide union plus a reranker to cut to 40 (recommended: yes).
- **D22** Soft vs hard exclusivity (recommended: soft p′, compared against hard on the holdout).
- **D23** Synthetic French hard negatives (recommended: yes).
- **D24** Cross-encoder as a feature (recommended: yes if there is a GPU, top 3–5 per S1).
- **D25** Label-shift correction: diagnose first (recommended: see §4.4).

---

## 2. Per-stage findings

### Stages 1–2: canonicalization and tokenization
- **Use `regex` (Apache-2.0), not `re`.** [VERIFIED] `re`'s `\w+` breaks Devanagari into pieces (`['प','र','इव','ट',…]`); `regex` keeps whole words, and `\X` returns grapheme clusters. This is the root cause of the numeric-only-name artifact.
- **Fractions:** NFKC turns `½` into `1⁄2` with **U+2044**, not `/`. Map U+2044 and U+2215 to `/`. NFKC also turns `№` and `º` into "No". anyascii converts native digits (`०१` → `01`); do it before extracting numbers.
- **Throughput:** ftfy costs about 113 µs per string.
  - Dedupe unique strings first, and skip ftfy when `s.isascii() and '&' not in s`.
  - Run a `multiprocessing.Pool(32)` with `imap(chunksize=10_000)`.
  - Avoid polars `map_elements`, which is single-threaded.
- **Phonetic key:** use a custom consonant skeleton. Map `ph→f, v→w, z→j, q/c/ck→k, sh→s`, drop `h` after a consonant, keep the first letter, drop later vowels, and collapse repeated letters.
  - It gives `delhi=dilli=dl`, `Raebareli=Raibareilly=rbrl`, `praivet=private=prwt`.
  - Two fixes are needed: treat final `w` as a vowel (`lucknow`), and map `m→n` before a consonant.
  - For French: drop a final silent `e/s/t/x`, map `qu→k` and `eau/au→o`.
  - Use it only for blocking and as a feature; its precision is low.
  - Also check `indic-soundex` (MIT, Jan 2026, untested).
- **Joined and split words:** compare with spaces removed, plus char 3–4-grams on the space-free string, plus the corpus joiner (D7).
- **Nothing rule-based turns `dilli` into `delhi`.** It is an exonym, and only equivalences mined from matches fix it.

### Stage 3: parsing
- **No pretrained address parser:**
  - libpostal's parser is trained on OSM and OpenAddresses, which counts as external data.
  - deepparse is LGPL and trained on libpostal data.
  - Shiprocket's Indian NER was trained on external data.
- **The libpostal *dictionaries*** (`resources/dictionaries/{en,fr,hi}`, hand-curated MIT text files) are a safe source to copy abbreviation maps from, with a note in the doc.
- **France:** the BAN/AFNOR repetition indices `bis/ter/qua/qui` are part of the house number (`12bis`). AFNOR street types (av, bd, ch, fg, pl, imp, rte) go into the documented hand map.
- **Never strip business-type words** (holdings, group, north, metro). They are the signal for rejecting hard negatives.

### Stage 4: mining equivalences
The recipe, which upgrades `src/mine_equiv.py`:
1. Per anchor pair, remove the shared tokens.
2. Align the leftover tokens with a **monotone DP scored by character similarity** (Arasu et al., VLDB 2009). The fallback is IBM Model 1 EM in about 40 lines of numpy.
3. Score aligned pairs with **Dunning G² plus NPMI**. PMI alone overrates rare pairs (Moore 2004).
4. Keep only **one-to-one best links in both directions** (Melamed's competitive linking), which generalises our two-way agreement.
5. Set the support bar by how plausible the pair looks:
   - **abbreviation**: same first letter and the short form is a subsequence of the long one (`chm⊂chemin`, `tx⊂texas`), Tao et al. VLDB 2018: ≥ 5 pairs across ≥ 3 localities;
   - **transliteration**: skeleton edit distance ≤ 1: same bar as abbreviations;
   - **pure co-occurrence** (`nord↔hauts-de-france`): ≥ 50 pairs across ≥ 10 localities.
6. **Precision gate:** measure how often each rule fires on hard-negative candidate pairs, and drop it if that is more than ~0.3× its rate on positives.
7. **Cross-script:** align native-script S1 tokens with the Latin tokens of their matched S2/S3 records (`दिल्ली→delhi`, `प्रा.→pvt`, `लि.→ltd`).
8. Never map a token to nothing here. Optional tokens are handled by the log-odds/IDF features.

### Stage 5: blocking
- **Sparkly (VLDB 2023) takeaways:**
  - BM25 over 3-grams of the concatenated attributes, with top-k probing, beat 8 blockers, including deep ones.
  - At 98% recall its candidate set was 2.5% of pairs, against 10% for the deep blockers.
  - On MusicBrainz 10M, recall@50 was 94–98% against 40% for an autoencoder.
  - IDF is essential and TF barely matters.
  - Its automatic variant sums per-attribute BM25 scores and won on 10 of 15 datasets.
- **Implementation:**
  - Use `sparse_dot_topn` v1 `sp_matmul_topn(..., n_threads=32)` with `zip_sp_matmul_topn` for column chunks. Partition by country, 50k-row query chunks.
  - BM25 = saturated TF × IDF on the index side and binary or TF-IDF weights on the query side, using the same matmul.
  - Skip PyNNDescent, SVD+FAISS, MinHash, canopy clustering and unsupervised deep blockers.
- **Dense index, if we build one:** flat inner-product search, since 10M × 384 fp16 = 7.7 GB. HNSW with M=32 needs ~18 GB and a 30–60 min build. IVF-PQ is not needed.
- **Hubs:** use the candidate's in-degree as a reranker feature. Consider CSLS-style normalisation (score minus the candidate's mean top-k score).
- **Dense cities:** use IDF/BM25, drop tokens with document frequency above 0.5–1%, and within big locality blocks run a name-3-gram top-k instead of enumerating the block.

### Stages 6–7: features and model
- **GBDT is the right core.**
  - On short structured records, feature-based GBDT is near parity with transformers: DBLP-ACM 96.7 vs 97.9 F1 (a 2026 comparison [CHECK]).
  - Transformers win on long, messy product text, which is not our data.
  - LLM matchers are 39–59× slower for ≤ 2 F1, and are out of scope at 50M pairs.
- **Cross-encoder as a feature:**
  - Model: `microsoft/Multilingual-MiniLM-L12-H384` (MIT) first, `mdeberta-v3-base` (MIT) if there is time.
  - Input: `name_a [SEP] addr_a [SEP] name_b [SEP] addr_b`, max length 64–96.
  - Train on the ranker's top-k (the hard pairs), one epoch.
  - Estimated speed on the RTX 4060 in bf16 [EST]: MiniLM trains at ~2k pairs/s and infers at 6–10k pairs/s. On CPU, ONNX int8 gives ~1–2k pairs/s.
  - **Must be out-of-fold for the training rows.**
- **Number features:** use the match levels from D3b, **separately for each role** (house / unit / road / postal).
- **"Distinguishability"** (Linacre): the score gap between the best and second-best candidate. It is already in the plan as `gap to #2`; keep it high in the feature list.
- **Splink:** borrow only its **term-frequency adjustment** (u = the token's frequency within the country, floored at 0.001). Skip the package.
- **France shift, ranked by payoff:**
  1. Leave-one-country-out validation, keeping features with a small gap.
  2. Frequency- or role-based name-difference features instead of token identities.
  3. Synthetic French hard negatives.
  4. Pseudo-labels on French pairs with p > 0.98, mutual best and the same house number (disclose this).
  5. Adversarial validation to find features that shift.
  - Skip DADER and ZeroER (≈67 F1).
- **Calibration:** isotonic on the holdout, then check it survives the leave-one-country-out split. Stay with a binary objective, because the decoder needs absolute probabilities. A `rank_xendcg` model can be added as a feature.

### Stage 8: decoding
See §4 for the algorithm. Everything below is verified against the papers and by brute force.

---

## 3. Rule and licence flags

**Avoid:**

| Item | Problem |
|---|---|
| Unidecode | GPL |
| aksharamukha | AGPL |
| abydos | GPL |
| deepparse | LGPL, trained on external data |
| Jellyfish-7B/8B | CC BY-NC |
| jina-embeddings-v3 | CC BY-NC |
| EmbeddingGemma | Gemma terms, not MIT/Apache |

The licence rule is formally about the *model*. GPL-family utilities are still a hygiene risk in the code zip, and permissive alternatives exist, so skip them.

**External data, treat as a violation:**
- the libpostal parser (OSM-trained);
- Shiprocket's Indian NER;
- gazetteer-based Indian parsers;
- wordninja's bundled Wikipedia model;
- GLEIF / ISO 20275 as a lookup table.

**OK, but disclose in the methodology doc:**
- uroman (MIT; its LICENSE file asks for acknowledgement);
- cleanco term lists;
- libpostal dictionary text files;
- AFNOR abbreviations;
- statistics computed on unlabelled test records (IDF, mined equivalences, anchor pairs);
- pseudo-labels.

**Allowed models:**

| Model | Licence |
|---|---|
| multilingual-e5 | MIT |
| BGE-M3 | MIT |
| gte-multilingual | Apache-2.0 |
| Qwen3-Embedding-0.6B | Apache-2.0 |
| LaBSE | Apache-2.0 |
| model2vec potion | MIT |
| Multilingual-MiniLM | MIT |
| mdeberta-v3 | MIT |
| xlm-roberta | MIT |
| IndicXlit | MIT |

---

## 4. Decoder specification (verified)

Metric per S1: F = 1.25·TP / (k + 0.25·T). The empty prediction with T = 0 scores 1; k ≥ 1 with T = 0 scores 0.

**Top-k theorem** (Lewis 1995; Ye et al. 2012, Thm 9): under independence, some top-k by p is optimal. A brute-force check confirmed this still holds with a Poisson term for missed matches. So only n+1 sets need scoring.

**Missed positives:** T = S_cand + M, with M ~ Poisson(λ_i). Two ways to set λ_i:
- simple: λ_i = (1−R)/R · Σ_j p_ij, where R is the holdout recall ceiling;
- better: λ_i = max(0, N̂_i − Σ_j p_ij), where N̂_i comes from a per-S1 match-count regressor.

```
sort p descending; drop p < 1e-3
E[0] = prod_j (1 - p_j) * exp(-lambda)
for k in 1..n:
    A_k = PoissonBinomial(p_1..p_k)                     # prefix, updated incrementally
    B_k = PoissonBinomial(p_{k+1}..p_n) (*) Poisson(lambda)   # suffix, precomputed, truncated
    E[k] = sum_a sum_b A_k[a] * B_k[b] * 1.25*a / (k + 0.25*(a+b))
k* = argmax_k E[k]
```
- The Poisson-binomial step is `d'[s] = (1-p)·d[s] + p·d[s-1]`.
- Cost is O(n³) per S1 with n ≤ 30. In numba across 32 cores, 1.73M S1 should take seconds to a minute [EST].
- Ye et al. Algorithm 1 gives O(n²) for rational β².

**Soft exclusivity:** with odds o_ij = p_ij/(1−p_ij), use **p′_ij = o_ij / (1 + Σ_k o_kj)**. This is exact when the labels are independent Bernoullis conditioned on "each record has at most one S1". Then decode each S1 separately on p′.
- A better option is a stage-2 stacker with competition features: max over other S1s of p_kj, i's rank among the S1s for j, Σ_k p_kj and the twin partner's p. Recalibrate after it.

**Twins:** a unit with weight 2 changes the DP step to `d'[s] = (1-p)d[s] + p·d[s-2]`. Top-k is then heuristic, not provably optimal; use GFM (Waegeman 2014) if it matters.

**Label shift (4.1 vs 3.5 matches per S1): diagnose first.** Compare the mean Σ_j p_ij on test with train.
- **It rises:** the shift is covariate (denser entities). Only λ changes; don't reweight.
- **It stays flat:** apply Saerens EM (2002) after bias-corrected temperature scaling. Shrink the correction by 50% for France.

**Validation:**
- Use a holdout grouped by S1 and stratified by country.
- Compare three decoders: global threshold vs top-m cap plus threshold vs the DP (with and without λ and p′).
- Bootstrap 1000× over S1s and report the paired difference with a 95% CI.
- Re-run on a holdout subsampled so the mean true count per S1 is ~4.1.
- Tune at most 2–3 knobs. Never select on the public LB alone.

A brute-force verification script for the DP is `efm.py` in the session scratchpad. Copy it into `scripts/` if we keep it.

---

## 5. Things to study, per component

### Stages 1–2: canonicalization and tokenization
1. `regex` module: `\w`, `\X` and Unicode properties. https://pypi.org/project/regex/
2. uroman: Python API, `lcode`, caching. https://github.com/isi-nlp/uroman
3. IndicXlit and the Aksharantar paper (why romanisation varies). https://github.com/AI4Bharat/IndicXlit
4. indic-soundex rules, to borrow into the skeleton key. https://github.com/maverickMehul/indic-soundex
5. SymSpell word segmentation with a corpus-built dictionary. https://symspellpy.readthedocs.io/en/latest/examples/word_segmentation.html
6. Polars and multiprocessing pitfalls. https://docs.pola.rs/user-guide/misc/multiprocessing/

### Stage 3: name and address parsing
1. UK MoJ `uk_address_matcher`: numeric-role keys and contradiction penalties (PR #504). https://github.com/moj-analytical-services/uk_address_matcher
2. Robin Linacre on address matching: block-relative TF and distinguishability. https://www.robinlinacre.com/address_matching/
3. cleanco legal-term lists as the documented seed. https://github.com/psolin/cleanco
4. libpostal dictionaries (copy only the abbreviation text files). https://github.com/openvenues/libpostal/tree/master/resources/dictionaries
5. French address norms: BAN repetition indices and AFNOR XP Z10-011 street types. http://ressources.sitilr.fr/document/types-de-voie-norme-afnor-xp-z-10-011/
6. Stretch: parserator / sklearn-crfsuite for a weakly supervised number-role CRF. https://github.com/datamade/parserator

### Stage 4: mining equivalences
1. Arasu, Chaudhuri, Kaushik, "Learning String Transformations from Examples", VLDB 2009. http://www.vldb.org/pvldb/vol2/vldb09-226.pdf
2. Tao et al., "Approximate String Joins with Abbreviations", VLDB 2018. http://www.vldb.org/pvldb/vol11/p53-tao.pdf
3. Moore 2004 on G² vs PMI for rare events, for setting support thresholds. https://www.microsoft.com/en-us/research/wp-content/uploads/2004/07/rare-events-final-rev.pdf
4. Melamed, "Models of Translational Equivalence" (competitive linking); IBM Model 1 EM tutorial.
5. Monroe, Colaresi, Quinn 2008, "Fightin' Words": log-odds with a Dirichlet prior.
6. Singh & Gulwani, VLDB 2012 (semantic string transformations), skim only. http://vldb.org/pvldb/vol5/p740_rishabhsingh_vldb2012.pdf

### Stage 5: blocking
1. **Sparkly**, VLDB 2023, sections 3.4, 4.4 and 5. https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf (code: https://github.com/anhaidgroup/sparkly)
2. sparse_dot_topn v1 README (`sp_matmul_topn`, `zip_sp_matmul_topn`). https://github.com/ing-bank/sparse_dot_topn
3. **SC-Block**, supervised contrastive blocking. https://arxiv.org/abs/2303.03132
4. UniBlocker, dense vs Sparkly and ensembles. https://arxiv.org/abs/2404.14831
5. Papadakis et al. blocking survey (block purging, meta-blocking). https://arxiv.org/abs/1905.06167
6. Sentence-Transformers: MultipleNegativesRankingLoss, hard-negative mining, ONNX/int8 efficiency. https://sbert.net/docs/sentence_transformer/usage/efficiency.html
7. model2vec (static embeddings, distilling from our own fine-tuned e5). https://github.com/MinishLab/model2vec

### Stages 6–7: features and model
1. **Foursquare Location Matching 2022**, 1st place (re:waiwai). https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution
   - 7th place: https://future-architect.github.io/articles/20220720a/
   - Code: https://github.com/TheoViel/kaggle_foursquare
2. Papadakis et al. 2023, "benchmark linearity" (why GBDT works on structured ER). https://arxiv.org/abs/2307.01231
3. AnyMatch: selecting hard pairs and balancing labels for small matchers. https://arxiv.org/abs/2409.04073
4. Sudowoodo: synthetic negatives and pseudo-labels. https://github.com/megagonlabs/sudowoodo
5. Steiner, Peeters, Bizer: fine-tuned LLM matchers transfer poorly across domains. https://arxiv.org/abs/2409.08185
6. Two 2026 studies, both [CHECK]:
   - GBDT vs Ditto vs Qwen3 with runtimes: https://arxiv.org/html/2606.28823
   - cross-encoders and shortcut learning: https://arxiv.org/abs/2607.24688
7. LightGBM monotone constraints (`advanced` method). https://lightgbm.readthedocs.io/en/latest/Parameters.html and https://arxiv.org/abs/2011.00986
8. Splink term-frequency adjustments. https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html

### Stage 8: decoding and calibration
1. **Ye, Chai, Lee, Chieu, ICML 2012**, "Optimizing F-measures: A Tale of Two Approaches", Algorithm 1. https://icml.cc/2012/papers/175.pdf
2. Faron's Instacart kernel (the same DP with a "None" option). https://kaggle.com/cpmpml/f1-score-expectation-maximization-in-o-n
3. Waegeman et al. 2014, GFM for dependent labels (twins). https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf
4. Lipton, Elkan, Naryanaswamy 2014, optimal threshold F\*/2; for F0.5 it becomes 0.8·F\*. https://arxiv.org/abs/1402.1892
5. Alexandari et al. 2020, EM plus bias-corrected temperature scaling for label shift. https://arxiv.org/abs/1901.06852
   - Also Saerens 2002 and BBSE: https://arxiv.org/abs/1802.03916
6. Papadakis et al. 2022, bipartite matching for clean-clean ER (why we *don't* need Hungarian). https://arxiv.org/abs/2112.14030
