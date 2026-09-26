#!/usr/bin/env python3
"""Render REPORT.md from audit.json, curation_experiment.json, examples.json and
pattern_summary.tsv. Every number and example in the report comes from those files."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "audit.json").read_text())
E = json.loads((HERE / "curation_experiment.json").read_text())
X = json.loads((HERE / "examples.json").read_text())
PS = pd.read_csv(HERE / "pattern_summary.tsv", sep="\t")
O, L1, L2, L3, L4, IMP = (A["overview"], A["lens1_exact_duplicates"], A["lens2_patterns"],
                          A["lens3_model"], A["lens4_assignment"], A["score_impact"])
INS = X["insights"]


# ---------------------------------------------------------------- helpers
def n(x):
    return f"{int(x):,}"


def pc(x, d=2):
    return f"{x:.{d}f}%"


def esc(s):
    return str(s if s is not None else "").replace("|", "\\|").replace("\n", " ").strip()


def rec(i, name, addr):
    return f"`{esc(i)}` {esc(name)} — {esc(addr) or '*(empty address)*'}"


def lab(s):
    return "**match**" if s == "match" else "**no match**"


def ptxt(p):
    return f"{p:.3f}" if isinstance(p, (int, float)) else "–"


def examples(groups, note=True):
    rows = []
    for g in groups:
        for r in g["rows"]:
            rows.append(f"| {esc(g['caption'].rsplit(':', 1)[0])} | {rec(r['s1_id'], r['s1_name'], r['s1_addr'])} | "
                        f"{rec(r['rec_id'], r['rec_name'], r['rec_addr'])} | {lab(r['label'])} | {ptxt(r['model_p'])} |")
    if not rows:
        return "*No examples exist for this case.*\n"
    head = "| Case | S1 record | S2/S3 record | Label | Model p(match) |\n|---|---|---|---|---|\n"
    return head + "\n".join(rows) + "\n"


def mix_rows(rows, key):
    out = ["| " + key.capitalize() + " | Pairs | Matches | No matches | Match rate | Against majority |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {esc(r[key])} | {n(r['pairs'])} | {n(r['matches'])} | {n(r['non_matches'])} | "
                   f"{pc(r['match_rate_pct'])} | {n(r['against_pattern_majority'])} |")
    return "\n".join(out) + "\n"


def bd(name):
    return L2["breakdowns"][name]


rc = L2["reported_cases"]


def case_row(k):
    v = rc[k]
    t = v["matches"] + v["non_matches"]
    return (f"| {k} | {n(t)} | {n(v['matches'])} | {n(v['non_matches'])} | {pc(100 * v['matches'] / t)} | "
            f"{n(v['against_pattern_majority'])} |")


dup_raw, dup_norm = L1["identical raw text"], L1["identical after normalisation"]
by_addr = dup_norm.get("conflict_groups_by_address", {})
conf_empty = sum(v["address empty"] for v in by_addr.values())
conf_present = sum(v["address present"] for v in by_addr.values())
ident = rc["name and address identical"]
num_only, num_only_same = rc["only the house/plot number differs"], rc["...and the name is identical"]
name_only, name_only_same = (rc["only the name differs (address identical or a word typo)"],
                             rc["...and the address is fully identical"])
typo_case = rc["only typo-like differences"]
exp = {r["variant"][0]: r for r in E["results"]}
base_f = exp["A"]["macro_f05_at_0.5"]


def delta(v):
    return f"{(exp[v]['macro_f05_at_0.5'] - base_f) * 100:+.2f}"


# other-script classes, from the breakdowns
def pick_rows(rows, prefix):
    return [r for r in rows if str(r["name change"]).startswith(prefix)]


os_same_addr = pick_rows(bd("address identical, by name change"), "other script")
os_num_only = pick_rows(bd("only numbers differ, by name change"), "other script")
fin = INS["finer_level"]["pattern"]

# ---------------------------------------------------------------- hand-checked cases
def reason(r):
    if r.get("paired") is False:
        return ("Not compared: no S1 business shares any of the six blocking keys with this record, so it sits "
                "outside the pattern and model checks.")
    parts = []
    if r["s1_id"] != r["expected_s1"]:
        parts.append(f"The audit compared it with `{r['s1_id']}` (its closest S1), not `{r['expected_s1']}`.")
    p = r["model_p"]
    model_side = "match" if p >= 0.5 else "no match"
    if not r["comparable"]:
        parts.append(f"This match shares no blocking key with its S1, so it is outside the pattern and model checks. "
                     f"The model still scores it: p = {p:.3f}.")
        return " ".join(parts)
    maj = "match" if r["pattern_majority"] == "match" else "no match"
    agree = maj == r["label"]
    parts.append(f"{pc(r['pattern_match_rate'], 1)} of the {n(r['pattern_pairs'])} pairs with this exact pattern "
                 f"({r['country']}, {r['source']}) are matches, so its majority label is {maj}; this pair "
                 f"{'agrees' if agree else '**goes against it**'}.")
    strong = r["flag_model"]
    if strong:
        parts.append(f"Model p = {p:.3f}: **strongly contradicts** the label.")
    elif model_side == r["label"]:
        parts.append(f"Model p = {p:.3f}: agrees with the label.")
    else:
        parts.append(f"Model p = {p:.3f}: leans the other way, but not past the flag thresholds (0.02 / 0.98).")
    flagged = [x for x, hit in (("pattern", r["flag_pattern"]), ("model", r["flag_model"])) if hit]
    parts.append(f"→ **flagged by {' and '.join(flagged)}**" if flagged else "→ not flagged")
    return " ".join(parts)


hand_md = []
for h in X["hand_cases"]:
    s1 = h["s1"]
    size = s1["true_size"]
    hand_md.append(f"#### `{s1['s1_id']}` {esc(s1['name'])} — {esc(s1['addr'])}\n\n"
                   f"Ground truth: {'singleton (no matches)' if size == 0 else f'{size} matched record' + ('s' if size != 1 else '')}.\n\n"
                   "| Record | Label | Pattern (name · address · numbers) | Why it is or isn't flagged |\n|---|---|---|---|")
    for r in h["records"]:
        pattern = r.get("pattern", "–")
        pattern = " · ".join(x.split(": ", 1)[1] for x in pattern.split(" | ")) if " | " in pattern else pattern
        hand_md.append(f"| {rec(r['rec_id'], r['rec_name'], r['rec_addr'])} | {lab(r['label'])} | {esc(pattern)} | {esc(reason(r))} |")
    hand_md.append("")
hand_md = "\n".join(hand_md)

hand_flagged = sum(1 for h in X["hand_cases"] for r in h["records"] if r.get("flag_pattern") or r.get("flag_model"))
hand_total = sum(len(h["records"]) for h in X["hand_cases"])
hand_uncovered = sum(1 for h in X["hand_cases"] for r in h["records"] if r.get("paired") is False or r.get("comparable") is False)

# ---------------------------------------------------------------- raw duplicate example
rd = INS["identical"]["raw_duplicate_group"]
rd_rows = "\n".join(f"| {rec(r['rec_id'], r['name'], r['addr'])} | {rec(a['s1_id'], a['name'], a['addr']) if a else '–'} | **match** |"
                    for r, a in zip(rd["records"], rd["assigned"]))
eg_rows = []
for g in INS["empty_address"]["conflict_groups"]:
    for r in g["records"]:
        s = r["s1"]
        eg_rows.append(f"| {esc(g['conflict'])} | `{esc(r['rec_id'])}` {esc(r['name'])} — *(empty address)* | "
                       f"{rec(s['s1_id'], s['name'], s['addr']) if s else 'unmatched'} |")
eg_rows = "\n".join(eg_rows)
unpaired_rows = "\n".join(f"| {rec(r['rec_id'], r['name'], r['addr'])} | no S1 shares a blocking key |"
                          for r in INS["coverage"]["unpaired_unmatched"])

fi = "\n".join(f"| `{k}` | {v:.1f}% |" for k, v in L3["feature_importance_gain_pct"].items())
top15 = []
for r in L2["top_patterns_pooled"][:15]:
    a, b, c = [x.split(": ", 1)[1] for x in r["pattern"].split(" | ")]
    top15.append(f"| {a} | {b} | {c} | {n(r['pairs'])} | {pc(r['match_rate_pct'])} | {n(r['against_majority'])} |")
top15 = "\n".join(top15)
rej = "\n".join(f"| {esc(' · '.join(x.split(': ', 1)[1] for x in k.split(' | ')))} | {n(v)} |"
                for k, v in list(L3["top_patterns_matches_rejected"].items())[:6])
acc = "\n".join(f"| {esc(' · '.join(x.split(': ', 1)[1] for x in k.split(' | ')))} | {n(v)} |"
                for k, v in list(L3["top_patterns_non_matches_accepted"].items())[:6])

exp_rows = []
labels = {"A": "Keep everything", "B": "Drop pairs against their pattern", "C": "Drop mixed patterns (>5% minority)",
          "D": "Drop pairs a model contradicts", "E": "All hard pairs + 10% of easy pairs, weighted ×10"}
for v in "ABCDE":
    r = exp[v]
    band = r["uncertain_band_predicted_vs_actual_match_rate"]
    exp_rows.append(f"| **{v}** {labels[v]} | {n(r['training_pairs'])} | {n(r['removed_from_training'])} | "
                    f"**{r['macro_f05_at_0.5']:.5f}** | {'–' if v == 'A' else delta(v) + ' pts'} | "
                    f"{r['macro_f05_best_threshold']:.5f} ({r['best_threshold']:.2f}) | {r['holdout_auc']:.5f} | "
                    f"{r['holdout_log_loss']:.4f} | {band[0]*100:.1f}% vs {band[1]*100:.1f}% |")
exp_rows = "\n".join(exp_rows)

# ---------------------------------------------------------------- the report
md = f"""# Training-Set Label Audit and Cleaning Experiment

| | |
|---|---|
| Data used | Training files only: `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`. No test file is read anywhere in this work. |
| Code | `eda_output/label_audit/`: `build_pairs.py`, `analyze.py`, `wordfeats.py`, `curation_experiment.py`, `examples.py`, `build_markdown.py`, `build_report.py` |
| Generated from | `audit.json`, `curation_experiment.json`, `examples.json`, `pattern_summary.tsv`. This file is written by `build_markdown.py`, so every number and example below comes from those files. |
| HTML version | [Train Label Audit](https://claude.ai/artifact/LrPyzLZS588ZsyPGe127yE) (private; share it from the page's Share menu) |

Throughout, **match** means the ground truth lists the record under that S1 business, and **no match** means it does not. **Model p** is the out-of-fold probability of a match from the lens-3 model (section 2.4).

---

## Summary

- **Identical inputs get identical labels.** {n(dup_raw['duplicate_groups'])} groups of records with identical raw text contain **{n(dup_raw['groups_some_matched_some_unmatched'] + dup_raw['groups_matched_to_different_s1'])}** label conflicts. Of {n(ident['pairs'])} pairs whose normalised name and address equal the S1 record's, **{n(ident['non_matches'])}** is not a match.
- **The one identical input that gets different labels is a name with no address.** {n(conf_empty)} groups of records with the same name and an empty address are matched to different S1 businesses, or matched and unmatched.
- **Conflicts sit where something differs, mostly where only the numbers differ.** That case covers {n(num_only['pairs'])} pairs with a {pc(100 * num_only['matches'] / num_only['pairs'], 1)} match rate. Whether such a pair is a match depends mostly on how the name changed: {pc(num_only_same['matches'] / num_only_same['pairs'] * 100)} when the name is identical, much lower when a word is added.
- **Across all {n(L2['patterns'])} observed difference patterns,** {n(L2['patterns_with_both_labels'])} contain both labels, and {n(L2['pairs_against_their_pattern_majority'])} pairs (**{pc(L2['pct_of_comparable_pairs'])}** of those compared) carry the minority label of their pattern.
- **Names in Indian scripts are now compared by sound**, so a word replaced across scripts is visible (the Maa Hospitality vs "मां केयर" case).
- **A model that also sees how the number changed and which words were added separates the labels:** out-of-fold AUC **{L3['auc']:.4f}**. It contradicts **{n(L3['confident_learning_label_issues'])} labels ({pc(L3['confident_learning_pct'])})** with high confidence, and {pc(L3['pct_pairs_uncertain_0.2_to_0.8'])} of pairs stay uncertain (p between 0.2 and 0.8).
- **Removing conflicting pairs from training makes a held-out score worse:**

  | Change to the training data | Change in holdout macro F0.5 |
  |---|---|
  | Remove pairs labelled against their pattern | {delta('B')} points |
  | Remove whole mixed patterns | {delta('C')} points |
  | Remove the pairs a model contradicts | {delta('D')} points |
  | Keep all hard pairs plus a weighted 10% of easy pairs | {delta('E')} points, with {exp['E']['training_pairs'] / exp['A']['training_pairs']:.0%} of the rows |

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
| S2/S3 training records (matched / unmatched) | {n(O['s2_s3_records'])} ({n(O['matched_records'])} / {n(O['unmatched_records'])}) |
| Unmatched records with no lookalike under any key | {n(O['unmatched_without_lookalike'])} ({pc(100 * O['unmatched_without_lookalike'] / O['unmatched_records'], 1)} of unmatched) |
| Pairs | {n(O['pairs'])} |
| Comparable pairs: matches / no matches | {n(O['comparable_pairs'])}: {n(O['comparable_matches'])} / {n(O['comparable_non_matches'])} |
| Matches sharing no key with their own S1 | {n(O['matches_sharing_no_key_with_their_s1'])} ({pc(100 * O['matches_sharing_no_key_with_their_s1'] / O['matched_records'], 1)} of matches) |
| S1 businesses / singletons | {n(O['s1_records'])} / {n(O['s1_singletons'])} |

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

A **pattern** is the combination (country, source, name change, address change, number change). {n(L2['patterns'])} patterns occur.

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

Examples are drawn at random with a fixed seed ({X['seed']}) from the pairs that fit each case. Each shows the S1 record, the S2/S3 record and the ground-truth label.

### Insight 1: identical inputs get identical labels

| Check | Result |
|---|---|
| Groups of records with identical raw text | {n(dup_raw['duplicate_groups'])} groups ({n(dup_raw['records_in_duplicate_groups'])} records); **{n(dup_raw['groups_some_matched_some_unmatched'] + dup_raw['groups_matched_to_different_s1'])}** with different labels or S1s |
| Pairs whose normalised name and address equal the S1's | {n(ident['pairs'])}; **{n(ident['matches'])} matches, {n(ident['non_matches'])} no match** |
| Duplicate S1 businesses | {n(L1['duplicate_s1_records']['groups_identical_after_normalisation'])} |

An example group of raw-identical records, all matched to the same S1:

| Record | Matched to | Label |
|---|---|---|
{rd_rows}

{examples(INS['identical']['groups'])}
### Insight 2: the same name with no address gets different labels

After normalisation, {n(dup_norm['duplicate_groups'])} groups of identical records exist. {n(dup_norm['groups_some_matched_some_unmatched'] + dup_norm['groups_matched_to_different_s1'])} of them carry different labels or S1s, and almost all of those are **names with an empty address**:

| Conflict | Groups, address present | Groups, address empty |
|---|---|---|
""" + "\n".join(f"| {k} | {n(v['address present'])} | {n(v['address empty'])} |" for k, v in by_addr.items()) + f"""
| **Total** | **{n(conf_present)}** | **{n(conf_empty)}** |

Examples of identical names with no address that were assigned differently:

| Conflict | Record | Assigned to |
|---|---|---|
{eg_rows}

### Insight 3: when only the numbers differ, the label depends on how the name changed

| Case | Pairs | Matches | No matches | Match rate | Against majority |
|---|---|---|---|---|---|
{case_row('only the house/plot number differs')}
{case_row('...and the name is identical')}

**Only numbers differ, by how the name changed:**

{mix_rows(bd('only numbers differ, by name change'), 'name change')}
{examples(INS['numbers_only']['groups'])}
### Insight 4: with the name identical, most number changes are matches; nearby and one-character changes are the mixed ones

{mix_rows(bd('name identical, only numbers differ, by number change'), 'number change')}
{examples(INS['number_change']['groups'])}
### Insight 5: with the address fully identical, most name changes are matches

| Case | Pairs | Matches | No matches | Match rate | Against majority |
|---|---|---|---|---|---|
{case_row('only the name differs (address identical or a word typo)')}
{case_row('...and the address is fully identical')}
{case_row('only typo-like differences')}

"Only typo-like differences" means the name is identical or differs by typos, the address is identical or has word typos, and any number change is a cut-short or one-character change, with at least one typo present.

**Address fully identical, by how the name changed:**

{mix_rows(bd('address identical, by name change'), 'name change')}
{examples(INS['address_identical']['groups'])}
### Insight 6: names in another script, compared by sound

A Hindi, Tamil or other Indic name against a Latin S1 name is now split by what changed when the words are compared by sound, instead of being one "other script" class. The tables below give the match rate of each kind of change. Compare them with the Latin-name tables in insights 3 and 5.

**Address fully identical:**

{mix_rows(os_same_addr, 'name change') if os_same_addr else '*None.*'}
**Only numbers differ:**

{mix_rows(os_num_only, 'name change') if os_num_only else '*None.*'}
{examples(INS['other_script']['groups'])}
### Insight 7: across all patterns, about 1 pair in {round(100 / L2['pct_of_comparable_pairs'])} carries its pattern's minority label

| | |
|---|---|
| Patterns observed | {n(L2['patterns'])} |
| Patterns containing both labels | {n(L2['patterns_with_both_labels'])} |
| Pairs against their pattern's majority | {n(L2['pairs_against_their_pattern_majority'])} ({pc(L2['pct_of_comparable_pairs'])}) |
| Share of pairs in patterns whose minority exceeds 1% / 5% / 20% | {pc(L2['pct_pairs_in_patterns_with_minority_over_1pct'])} / {pc(L2['pct_pairs_in_patterns_with_minority_over_5pct'])} / {pc(L2['pct_pairs_in_patterns_with_minority_over_20pct'])} |

**The 15 patterns with the most pairs against their majority** (pooled over country and source). The full list is in `pattern_summary.tsv`.

| Name change | Address change | Number change | Pairs | Match rate | Against majority |
|---|---|---|---|---|---|
{top15}

### Insight 8: most pattern-level mixing resolves at a finer level

The pattern classes are coarse. A model that also sees *how far* the number moved and *which* words were added separates the labels far better than the patterns do:

| | |
|---|---|
| Out-of-fold AUC | {L3['auc']:.5f} |
| Log loss / Brier score | {L3['log_loss']:.5f} / {L3['brier']:.5f} |
| Pairs with p between 0.2 and 0.8 | {pc(L3['pct_pairs_uncertain_0.2_to_0.8'])} |
| Pairs with p between 0.05 and 0.95 | {pc(L3['pct_pairs_uncertain_0.05_to_0.95'])} |

**What the model relies on** (the 15 largest shares of split gain):

| Feature | Share |
|---|---|
{fi}

The pattern with the most pairs against its majority is *{esc(fin['pattern'])}* ({fin['country']}, {fin['source']}): {n(fin['pairs'])} pairs, {pc(fin['match_rate'])} matches. Within it, the model still separates the two labels:

{examples(INS['finer_level']['groups'])}
### Insight 9: the model's strongest contradictions have recognisable shapes

| | |
|---|---|
| Matches the model rejects (p < 0.02) | {n(L3['matches_model_rejects_p_below_0.02'])} |
| No-match pairs the model accepts (p > 0.98) | {n(L3['non_matches_model_accepts_p_above_0.98'])} |
| Confident-learning label issues | {n(L3['confident_learning_label_issues'])} ({pc(L3['confident_learning_pct'])}). Thresholds: match {L3['confident_learning_thresholds']['match']}, no match {L3['confident_learning_thresholds']['non_match']} |

| Matches the model rejects: pattern | Pairs |
|---|---|
{rej}

| No-match pairs the model accepts: pattern | Pairs |
|---|---|
{acc}

{examples(INS['model_flags']['groups'])}
### Insight 10: the comparison does not reach every record

- {n(O['matches_sharing_no_key_with_their_s1'])} matches share none of the six keys with their own S1, and {n(O['unmatched_without_lookalike'])} unmatched records have no lookalike at all.
- {n(L4['matches_sharing_no_name_word_and_little_address'])} matches share no name word and under 25% of their address words with their S1.

These are outside lenses 2 and 3.

{examples(INS['coverage']['groups'])}
Unmatched records with no lookalike:

| Record | Why it has no pair |
|---|---|
{unpaired_rows}

### Insight 11: the reference itself is clean

- **{n(L1['duplicate_s1_records']['groups_identical_after_normalisation'])} duplicate S1 businesses.**
- **{n(L4['matched_record_is_exact_copy_of_a_different_s1'])} matched records are an exact copy of a *different* S1** than their own.
- **{n(L4['unmatched_record_is_exact_copy_of_an_s1'])} unmatched record is an exact copy of an S1.**

The single unmatched exact copy is the last example under Insight 1.

### Score impact

| Rule | Macro F0.5 over all S1 | S1 affected | Mean F0.5 on those |
|---|---|---|---|
| Every comparable pair takes its pattern's majority label (in-sample) | {IMP['pattern_majority_rule']['macro_f05_all_s1']:.4f} | {n(IMP['pattern_majority_rule']['s1_touched'])} | {IMP['pattern_majority_rule']['mean_f05_on_touched_s1']:.4f} |
| Out-of-fold model, threshold 0.5 | {IMP['model_threshold_0.5']['macro_f05_all_s1']:.4f} | {n(IMP['model_threshold_0.5']['s1_touched'])} | {IMP['model_threshold_0.5']['mean_f05_on_touched_s1']:.4f} |

These are ceilings on training data, not leaderboard estimates: pairs outside the comparison are assumed correct.

### The cases checked by hand

Every record from the manual checks, and how the audit sees it. A pair is **flagged** when either:
- its label is the minority label of its exact pattern; or
- the model contradicts it strongly (a match with p < 0.02, or a no match with p > 0.98).

"Not flagged" therefore means the label agrees with how most identical-pattern pairs are labelled, and the model does not strongly disagree. {hand_flagged} of the {hand_total} records {'is' if hand_flagged == 1 else 'are'} flagged, and {hand_uncovered} {'is' if hand_uncovered == 1 else 'are'} outside the pattern and model checks.

{hand_md}
---

## 4. Experiment: does cleaning the training data help?

### 4.1 Setup

- **Holdout:** all pairs of the S1 businesses in fold 0, which is 20% of businesses, never split. That's {n(E['holdout_s1_businesses'])} businesses and {n(E['holdout_pairs'])} pairs. **Holdout labels are never changed**, just like the test set.
- **Training candidates:** the comparable pairs of folds 1–4.
- **Model:** the same LightGBM and 42 features as lens 3. Word evidence is computed without leakage, using only the rows each variant keeps. Every variant trains on a uniform random sample of at most 3,000,000 of its eligible rows (seed 11), so A–D differ in *which* rows are eligible, not in how many are used.
- **Metric:** macro F0.5 over all holdout businesses, singletons included. A business's predicted set is its pairs with p ≥ t, and its true size is its full ground-truth count. F0.5 is reported at t = 0.5, and at the best t on a grid from 0.05 to 0.95. AUC, log loss and calibration are measured on the comparable holdout pairs.

### 4.2 The five variants

| Variant | How its training rows are chosen |
|---|---|
| **A** | everything |
| **B** | remove every pair carrying its pattern's minority label (majorities measured on training rows only) |
| **C** | remove every pair in a pattern whose minority label exceeds 5% |
| **D** | remove pairs a model contradicts: a 2-fold confident-learning pass inside the training data (train on folds 1–2, predict 3–4, and the reverse), with per-class thresholds at each class's mean predicted probability; {n(E['confident_learning_issues_in_training'])} pairs |
| **E** | keep every pair the inner model is unsure about (0.002 < p < 0.998), plus a random 10% of the rest with weight 10, so the class mix is preserved |

### 4.3 Results

| Variant | Training rows | Removed | Holdout F0.5 (t = 0.5) | vs A | Best F0.5 (t) | AUC | Log loss | p in 0.2–0.8: predicted vs actual match rate |
|---|---|---|---|---|---|---|---|---|
{exp_rows}

### 4.4 What the results say

- **B is worse, and not only because of the threshold.** Making every pattern pure in training teaches the model that patterns decide the label outright. On the holdout, where the same patterns are mixed, it is confidently wrong: log loss rises from {exp['A']['holdout_log_loss']:.4f} to {exp['B']['holdout_log_loss']:.4f}, AUC falls from {exp['A']['holdout_auc']:.5f} to {exp['B']['holdout_auc']:.5f}, and even its best threshold ({exp['B']['macro_f05_best_threshold']:.5f}) stays below A at 0.5.
- **C is worse because the model never sees the mixed patterns**, and they still appear in the holdout.
- **D and E are within {max(abs(float(delta('D'))), abs(float(delta('E')))):.2f} points of A** (D {delta('D')}, E {delta('E')}). With one run each, a gap that small cannot be told apart from sampling noise. E uses {exp['E']['training_pairs'] / exp['A']['training_pairs']:.0%} of A's eligible training rows, so it is the way to shrink training data when time matters, at that cost.

### 4.5 Caveats

- **The absolute scores are idealised and are not a leaderboard estimate.** Every true match is available as a candidate here, and each unmatched record is compared only with its single closest S1. Only the comparison between variants is meaningful.
- **One run per variant, one seed.** A, D and E cannot be ranked against each other. B and C are worse on every metric at once.

### 4.6 Decision

1. Keep every training label; do not remove the flagged pairs or the mixed patterns.
2. If a smaller training set is needed, subsample the easy pairs with weights (E); on this holdout it cost {abs(float(delta('E'))):.2f} points.
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
"""
(HERE / "REPORT.md").write_text(md)
print(f"REPORT.md written ({len(md) / 1024:.0f} KB)")
