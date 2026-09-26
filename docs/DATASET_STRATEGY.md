# How we treat the data: dataset strategy in plain words

Every example is a real pair from the training data (S1 = reference record, S2/S3 = the other sources). "Match" / "no match" is the ground-truth label.

---

## 1. Labels: we keep all of them

**The situation:** some pairs look inconsistent. For example, two near-identical pairs can get opposite labels.

**What we do:** train on every label, and don't delete the "suspicious" ones.

**Why:** we tested removing them, and every kind of removal made the held-out score worse:

| We removed… | Held-out score |
|---|---|
| pairs whose label disagrees with similar pairs | −1.59 points |
| whole mixed groups | −0.46 points |
| pairs a model disagrees with | −0.01 points |

The "odd" pairs are the real hard cases, and the test set has them too. If the model never sees them, it gets them wrong at test time.

**What we still try on the VM:** the same removals inside the real pipeline, each as one command (variants B–G). We keep one only if it beats "keep everything" on labels nobody touched.

---

## 2. Same business, written differently: normalize, then compare

| S1 | S2/S3 | What we do |
|---|---|---|
| Elder Fin Limited — 318, 6Th 'A' Cross, Banaswadi, Bangalore | ELDER-FIN LTD — 318, 6TH 'A' CROSS, BANASWADI, BANGALORE | lower-case, drop punctuation. **Ltd = Limited** is noted as "same legal form, different spelling" |
| Gail's Medical — 611 **5th** Street, Hardin, **MT** | … 620 **Fifth** Street, Hardin, **Montana** | Fifth = 5th. MT and Montana are both "area" parts (below), so they are ignored |
| … 17 **R.** Vaucanson | … 17 **Rue** Vaucanson | **abbreviation rule:** a short word equals a long one if it starts with the same letter and its letters appear in it in order (R = Rue, St = Street, Bd = Boulevard, Rd = Road). This works in any language, with no list |

---

## 3. Other scripts and spelling by sound (Maa = Ma = माँ)

| S1 | S2/S3 | What we do |
|---|---|---|
| Swastik Business Private Limited | स्वस्तिक बिजनेस प्राइवेट लिमिटेड | write it in Latin letters (`svastik bijanes praivet limited`), then compare **by sound**: `svstk` = `svstk`. Same name |
| Maa Hospitality | मां केयर | माँ/मां → "ma" (the silent nasal sign is dropped). **Maa = Ma = मां** by sound key. केयर = "care" by sound. So one word is the same and one is replaced |
| Good Logistics | गुड लॉजिस्टिक्स | गुड = good, लॉजिस्टिक्स = logistics (sound keys `gd`, `lgstks`) |
| Kiser **Homes** | Kiser **Hmoes** Holdings | Hmoes = Homes (typo / same sound). The real difference is the added word "Holdings" |
| … Kozhikode | … Kozikode | Soundex: both K223, the same place |

**One-letter keys get one more letter:** "Maa" and "Ma" both become `m`. "Amy" becomes `*m` (it starts with a vowel), so it doesn't collide with them.

---

## 4. Area words: learned from the data, not written by hand

**The situation:** the same place gets written as a state, a region, a department, or not at all:
- "Lille, Nord" vs "Lille, Hauts-de-France"
- "Tyler, TX" vs "Tyler, Texas"

**What we do:** any address part that shows up in more than 0.1% of a country's records can't tell two businesses apart, so we ignore it when comparing addresses. The pipeline learned these by itself, with no labels:

| Country | Learned areas |
|---|---|
| US | all state codes and names, big cities, counties (188) |
| India | states, cities, big localities like "Andheri East" (220) |
| France (test only, never trained on) | Hauts-de-France, Nord, Gironde, Loire-Atlantique, Pas-de-Calais, Lille, Bordeaux, Nantes… (24) |

**Result:** no country is hard-coded. A new country gets its own areas the same way.

---

## 5. Only the house number changed: the model looks at how it changed and what else changed

| S1 | S2/S3 | Label | How we read it |
|---|---|---|---|
| Renee G. Garcia — **516** Main St | … **51** Main St | match | number **cut short**: usually a typo (99.9% match when the name is the same) |
| Garrity Pinnacle Capital — **505** Innovation Way | … **507** Innovation Way | match | one digit changed, name typos only: a mixed case, so the model decides from the details |
| Gabriella's Preferred Security — **1013** Girard Ave | … Security **Ltd** — **1020** Girard Ave | no match | number **nearby** + legal form added: 3.7% match. A neighbour business |

**What the model gets as features:**
- the type of number change (same / added / dropped / cut short / one digit / nearby / far);
- how far the number moved;
- how many digits changed;
- how the name changed.

---

## 6. A word was added to the name: the word itself is evidence

| S1 | S2/S3 | Label | Why |
|---|---|---|---|
| Garrett Chemical — 63 Nahanton Ave | Garrett Chemical **Northside** — 67 1/2 Nahanton Ave | no match | "Northside", "Holdings", "Group", "Overseas" are typical **neighbour-business** words |
| Overy Cardiology | Overy Cardiology **[Corp]** | match | a legal form, not a new business |

**What we do:** for every added or dropped word, the model gets how often it appeared in matches vs non-matches in training. This is computed only from *other* parts of the data, so no pair sees its own label.

---

## 7. Empty address: same name, many possible businesses

| S2/S3 (no address) | Possible S1s |
|---|---|
| Global Institute | Global Institute (NY); Global Institute, LLC (AZ); Global Institute LLC (MN) |

**What we do:** give the model how many S1s have this name and how many S1s compete for the record. If it can't tell, the final step answers **"no match"**. For F0.5, a wrong guess costs about 2.7× more than a miss.

---

## 8. Self-correction: look at the whole picture, then re-score

**Example:** record 3 (S2) looks like both S1-A and S1-B. Its twin record 4 (S3, same house number) is clearly S1-B's.
- The first model scores every pair on its own.
- The second model then sees, for each pair:
  - how strongly the record's best *other* S1 claims it;
  - how full this S1 already is;
  - whether the record's **twin in the other source** (same number or same name) belongs to this S1.

  So record 3 gets pulled to S1-B.
- **Final rule:** each record goes to only one S1.

**Why:** in training, when a number differs but the other source has the same number, the match rate jumps: 4% → 28% for nearby numbers, and 40% → 90% for far-off ones.

---

## 9. Words the model has never seen (France)

**The situation:** French words like "Développement" or "Groupement" never occur in training.

**What we do:** in 15% of training rows, we hide the word evidence ("pretend every word is new"). The model learns to decide from the other signals when words are unknown. The country is **not** a model input.

---

## 10. Singletons (S1 with no match)

**What we do:** for every S1, the final step compares "predict nothing" with "predict the top 1, 2, 3…" and keeps whichever has the higher **expected F0.5**. A singleton scores 1.0 only if we predict nothing, so weak candidates are dropped.
