#!/usr/bin/env python3
"""Assemble the self-contained HTML EDA report with embedded charts."""
import base64, json
from pathlib import Path

OUT = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/eda_output")
CH = OUT / "charts"

def img(name):
    b = (CH / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()

I = {n: img(f"{n}") for n in [
    "01_records_per_source.png","02_country_distribution.png","03_name_length_dist.png",
    "04_address_len_and_missing.png","05_match_count_dist.png","06_match_composition.png",
    "07_name_script_mix.png","08_noise_patterns.png"]}

HTML = f"""<meta charset="utf-8">
<title>Business Entity Resolution — EDA Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  --paper:#F6F7F9; --card:#FFFFFF; --ink:#161A20; --muted:#5A6472;
  --line:#E2E6EC; --accent:#2C7A88; --accent-soft:#DCEBEE;
  --warn:#B65431; --warn-soft:#F6E6DE; --good:#3E7A55;
  --shadow:0 1px 2px rgba(20,24,32,.04),0 8px 24px rgba(20,24,32,.05);
  --mono:ui-monospace,"SF Mono","JetBrains Mono","Cascadia Code",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --paper:#0F1318; --card:#161B22; --ink:#E7ECF2; --muted:#9AA6B4;
    --line:#262D37; --accent:#5FB6C4; --accent-soft:#123037; --warn:#E08150;
    --warn-soft:#2E1D14; --good:#6FB98A;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35); }}
}}
:root[data-theme="dark"] {{ --paper:#0F1318; --card:#161B22; --ink:#E7ECF2; --muted:#9AA6B4;
  --line:#262D37; --accent:#5FB6C4; --accent-soft:#123037; --warn:#E08150;
  --warn-soft:#2E1D14; --good:#6FB98A;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35); }}
:root[data-theme="light"] {{ --paper:#F6F7F9; --card:#FFFFFF; --ink:#161A20; --muted:#5A6472;
  --line:#E2E6EC; --accent:#2C7A88; --accent-soft:#DCEBEE; --warn:#B65431;
  --warn-soft:#F6E6DE; --good:#3E7A55;
  --shadow:0 1px 2px rgba(20,24,32,.04),0 8px 24px rgba(20,24,32,.05); }}

*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1080px;margin:0 auto;padding:56px 24px 96px}}
.eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);font-weight:600;margin:0 0 14px}}
h1{{font-size:clamp(30px,5vw,46px);line-height:1.08;margin:0 0 16px;text-wrap:balance;
  letter-spacing:-.02em;font-weight:750}}
.lede{{font-size:18px;color:var(--muted);max-width:64ch;margin:0 0 36px}}
h2{{font-size:24px;margin:0 0 4px;letter-spacing:-.01em;text-wrap:balance}}
.sec-no{{font-family:var(--mono);font-size:12px;color:var(--accent);letter-spacing:.1em}}
section{{margin-top:56px;padding-top:32px;border-top:1px solid var(--line)}}
p{{margin:12px 0;max-width:70ch}}
.muted{{color:var(--muted)}}
strong{{font-weight:650}}
.mono{{font-family:var(--mono)}}

.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:8px 0 4px}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 18px 16px;
  box-shadow:var(--shadow)}}
.stat .num{{font-family:var(--mono);font-size:27px;font-weight:600;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;color:var(--ink)}}
.stat .lbl{{font-size:12.5px;color:var(--muted);margin-top:4px;line-height:1.35}}
.stat .sub{{font-family:var(--mono);font-size:11px;color:var(--accent);margin-top:8px}}

figure{{margin:26px 0 8px;background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:16px;box-shadow:var(--shadow)}}
figure img{{width:100%;height:auto;display:block;border-radius:6px}}
figcaption{{font-size:13px;color:var(--muted);margin-top:12px;padding:0 4px;font-family:var(--mono)}}

.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;
  box-shadow:var(--shadow)}}
.callout{{border-left:3px solid var(--accent);background:var(--accent-soft);border-radius:0 10px 10px 0;
  padding:14px 18px;margin:20px 0}}
.callout.warn{{border-left-color:var(--warn);background:var(--warn-soft)}}
.callout .t{{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  font-weight:600;margin-bottom:4px}}
.callout.warn .t{{color:var(--warn)}} .callout:not(.warn) .t{{color:var(--accent)}}

table{{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line)}}
th{{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);font-weight:600}}
td.n,th.n{{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}}
tbody tr:hover{{background:var(--accent-soft)}}
caption{{caption-side:bottom;font-size:12px;color:var(--muted);margin-top:8px;text-align:left;
  font-family:var(--mono)}}

.tag{{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:20px;
  border:1px solid var(--line);color:var(--muted);margin:2px 4px 2px 0}}
ul.tight{{margin:10px 0;padding-left:20px}} ul.tight li{{margin:7px 0;max-width:68ch}}
.ex{{font-family:var(--mono);font-size:12.5px;background:var(--accent-soft);border-radius:6px;
  padding:2px 6px;color:var(--ink)}}
.foot{{margin-top:64px;padding-top:20px;border-top:1px solid var(--line);font-size:12.5px;
  color:var(--muted);font-family:var(--mono)}}
@media(max-width:720px){{.stats{{grid-template-columns:repeat(2,1fr)}}.grid2{{grid-template-columns:1fr}}}}
</style>

<div class="wrap">
  <p class="eyebrow">ML Challenge 2026 · Exploratory Data Analysis</p>
  <h1>Business Entity Resolution: what the data actually looks like</h1>
  <p class="lede">A structural and statistical audit of ~24M business records drawn from three
  independent, noisy sources — the geography, the missingness, the multilingual mess, the noise
  patterns, and the shape of the matching problem you're being scored on.</p>

  <div class="stats">
    <div class="stat"><div class="num">24.23 M</div><div class="lbl">Source records total</div><div class="sub">12.53M train · 11.70M test</div></div>
    <div class="stat"><div class="num">2.21 M</div><div class="lbl">Source-1 entities to resolve (train)</div><div class="sub">1.73M in test</div></div>
    <div class="stat"><div class="num">3</div><div class="lbl">Countries</div><div class="sub">US · India · France*</div></div>
    <div class="stat"><div class="num">5.6 %</div><div class="lbl">Singletons (no match)</div><div class="sub">score 1.0 if predicted empty</div></div>
  </div>
  <p class="muted" style="font-size:13px;margin-top:14px">* France appears <strong>only in the test set</strong> — a zero-shot country you must still resolve. F<sub>0.5</sub> is macro-averaged per Source-1 entity, precision-weighted 2×.</p>

  <!-- 1 -->
  <section>
    <p class="sec-no">01 / STRUCTURE</p>
    <h2>Three sources, one reference, a 1 : 2.3 : 2.4 size ratio</h2>
    <p>Source 1 is the deduplicated reference (2.21M train rows); Sources 2 and 3 are roughly
    2.3× larger and hold the noisy fragments you match back to S1. Every <span class="mono">entity_id</span>
    prefix is internally consistent (all S1- in source1, etc.), there are <strong>zero duplicate IDs</strong>
    anywhere, and every S1 entity appears in ground truth exactly once. The four columns
    (<span class="mono">entity_id, business_name, business_address, country</span>) are present in all files.</p>
    <figure><img alt="Record counts per source" src="{I['01_records_per_source.png']}">
      <figcaption>Fig 1 — Train vs test volume per source. Test is ~0.93× the train scale and structurally identical.</figcaption></figure>
  </section>

  <!-- 2 -->
  <section>
    <p class="sec-no">02 / GEOGRAPHY</p>
    <h2>~60% US / 40% India in train — and a whole new country at test time</h2>
    <p>Training covers only <strong>US</strong> and <strong>India</strong> (≈60/40 in S1). The test set adds
    <strong>France</strong> — <span class="mono">259,452</span> of the 1.73M test S1 entities (<strong>15.0%</strong>) —
    that never appears in training. Whatever you build for name/address normalization has to generalize
    to French conventions (<span class="ex">Rue</span>, <span class="ex">Avenue</span>, regions like
    <span class="ex">Nouvelle-Aquitaine</span>, suffixes <span class="ex">SAS / SARL / SA</span>) with no labeled examples.</p>
    <figure><img alt="Country composition per source, train vs test" src="{I['02_country_distribution.png']}">
      <figcaption>Fig 2 — Country mix by source. France (red) is present across all three test sources but absent from train.</figcaption></figure>
    <div class="callout warn"><div class="t">Caveat — open-set country</div>
      The README explicitly warns: do not hard-code, filter, or one-hot to {{US, India}}. France is ~15% of
      test entities and every one must appear in your submission. A US/India-only pipeline silently loses that slice.</div>
  </section>

  <!-- 3 -->
  <section>
    <p class="sec-no">03 / MISSINGNESS &amp; DUPLICATES</p>
    <h2>Names are always present; addresses drop out ~3%; names repeat a lot</h2>
    <div class="grid2">
      <div class="card">
        <p style="margin-top:0"><strong>Missing values</strong></p>
        <ul class="tight">
          <li><span class="mono">business_name</span> is never empty in any file.</li>
          <li><span class="mono">business_address</span> is empty in <strong>3.3–3.4%</strong> of S2/S3 rows
          (train), ~2.6–2.7% in test. S1 addresses are never empty.</li>
          <li>~169K (S2) and ~176K (S3) records must be matched on <em>name + country alone</em>.</li>
        </ul>
      </div>
      <div class="card">
        <p style="margin-top:0"><strong>Duplicate business names</strong></p>
        <ul class="tight">
          <li>S1: <strong>30.3%</strong> of names are non-unique; S2/S3: ~12%.</li>
          <li>Generic names (<span class="ex">Baba Foods Private Limited</span>) collide across distinct
          entities — name equality alone is <em>not</em> identity.</li>
          <li>Address and country are needed to disambiguate same-name businesses.</li>
        </ul>
      </div>
    </div>
    <figure><img alt="Address length distribution and missing rates" src="{I['04_address_len_and_missing.png']}">
      <figcaption>Fig 3 — Left: address length density (note the spike at 0 = empty). Right: empty-name vs empty-address rate per source.</figcaption></figure>
  </section>

  <!-- 4 -->
  <section>
    <p class="sec-no">04 / FIELD SHAPE</p>
    <h2>Names ~24 chars, addresses long-tailed to 250+</h2>
    <p>Business names are tight and consistent across sources (median 24–25 chars, p99 ≈ 42–50).
    Addresses are far more variable — median ~40 chars but a long right tail (p99 ≈ 115–126, max 256+),
    driven by verbose Indian addresses that stack building, area, landmark, city, district and state
    into one comma-salad field with inconsistent component order.</p>
    <figure><img alt="Business name length distribution" src="{I['03_name_length_dist.png']}">
      <figcaption>Fig 4 — Name-length density is near-identical across the three sources — a stable, well-behaved field.</figcaption></figure>
    <table>
      <caption>Table 1 — Field length percentiles (train). Values in characters.</caption>
      <thead><tr><th>Field · source</th><th class="n">median</th><th class="n">p95</th><th class="n">p99</th><th class="n">max</th></tr></thead>
      <tbody>
        <tr><td>Name · S1</td><td class="n">24</td><td class="n">37</td><td class="n">42</td><td class="n">105</td></tr>
        <tr><td>Name · S2</td><td class="n">25</td><td class="n">40</td><td class="n">48</td><td class="n">104</td></tr>
        <tr><td>Address · S1</td><td class="n">41</td><td class="n">103</td><td class="n">124</td><td class="n">256</td></tr>
        <tr><td>Address · S2</td><td class="n">37</td><td class="n">96</td><td class="n">118</td><td class="n">249</td></tr>
      </tbody>
    </table>
  </section>

  <!-- 5 -->
  <section>
    <p class="sec-no">05 / LANGUAGE &amp; SCRIPT</p>
    <h2>S1 is 100% Latin; S2/S3 are genuinely multilingual</h2>
    <p>The reference source (S1) is entirely Latin script. But <strong>~15% of S2/S3 names are non-Latin</strong>:
    ~5.3% Devanagari plus a spread of Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Gurmukhi and Oriya
    (each ~0.1–0.8%), and ~5–6% carry Latin accents (French, transliteration diacritics). Because S1 is Latin-only,
    matching a Devanagari S2 name (<span class="ex">राम मार्केटिंग प्राइवेट लिमिटेड</span>) to its Latin S1 counterpart
    is a <strong>cross-script transliteration problem</strong>, not a string-similarity one.</p>
    <figure><img alt="Name script mix per source" src="{I['07_name_script_mix.png']}">
      <figcaption>Fig 5 — Script composition of names. S1 is pure Latin; S2/S3 mix Devanagari + regional scripts + accented Latin.</figcaption></figure>
    <p><span class="tag">Devanagari 5.3%</span><span class="tag">Latin-accented ~5.8%</span><span class="tag">Telugu 0.8%</span>
    <span class="tag">Kannada 0.7%</span><span class="tag">Tamil 0.7%</span><span class="tag">Bengali 0.6%</span>
    <span class="tag">Gujarati 0.6%</span><span class="tag">Malayalam 0.4%</span><span class="tag">Oriya 0.15%</span><span class="tag">Gurmukhi 0.13%</span></p>
  </section>

  <!-- 6 -->
  <section>
    <p class="sec-no">06 / NOISE PATTERNS</p>
    <h2>The corruption is deliberate, structured, and source-specific</h2>
    <p>S2 and S3 records are perturbed in consistent, learnable ways; S1 stays clean. The prevalences below are
    measured over the full training files.</p>
    <figure><img alt="Noise pattern prevalence per source" src="{I['08_noise_patterns.png']}">
      <figcaption>Fig 6 — Noise-pattern prevalence. Domains, leading junk and landmark addresses appear in S2/S3 but essentially never in S1.</figcaption></figure>
    <div class="grid2">
      <div class="card">
        <p style="margin-top:0"><strong>Name noise</strong> (S2/S3 only)</p>
        <ul class="tight">
          <li>Legal-suffix drift: ~51–54% carry <span class="mono">Ltd/Pvt/LLC/Corp…</span> vs 65% in S1 — inconsistent presence.</li>
          <li><strong>~4%</strong> names replaced by a web domain (<span class="ex">womenshealthgroup.com</span>).</li>
          <li><strong>~0.9%</strong> leading junk: <span class="ex">-- Varanasi Property Ltd</span>, <span class="ex">*** Continental Green</span>.</li>
          <li>Trade-name markers: <span class="ex">a/k/a</span>, <span class="ex">t/a</span>, <span class="ex">formerly known as</span>, bracket suffixes <span class="ex">(ID: 99194)</span>.</li>
          <li>Char-level typos &amp; leetspeak: <span class="ex">C0nsultants</span>, <span class="ex">Dévelopment</span>.</li>
        </ul>
      </div>
      <div class="card">
        <p style="margin-top:0"><strong>Address noise</strong></p>
        <ul class="tight">
          <li>~90% of S2/S3 addresses contain digits; component <em>order is shuffled</em> (state/city/street permuted).</li>
          <li>Landmark references in ~3–5%: <span class="ex">Near SBI ATM</span>, <span class="ex">Opp. Cross Words</span>.</li>
          <li>Abbreviation variance: <span class="ex">Rd/Road</span>, <span class="ex">St/Street</span>, <span class="ex">HN/H.No</span>.</li>
          <li>Mixed-script addresses: state written as <span class="ex">तेलंगाना</span> or <span class="ex">தமிழ்நாடு</span>.</li>
        </ul>
      </div>
    </div>
  </section>

  <!-- 7 -->
  <section>
    <p class="sec-no">07 / THE MATCHING PROBLEM</p>
    <h2>Most entities have a small cluster; a few have many</h2>
    <p>Across the 2.21M labeled S1 entities: <strong>94.4% have at least one match</strong> and 5.6% are singletons.
    The match-count distribution is tight — <strong>mean 3.46, median 3</strong>, p95 = 6, p99 = 8, capped at 11.
    Matches split almost evenly between S2 (48.4%) and S3 (51.6%), and most matched entities link to
    <em>both</em> sources.</p>
    <div class="grid2">
      <figure style="margin:0"><img alt="Match count distribution" src="{I['05_match_count_dist.png']}">
        <figcaption>Fig 7 — Matches per S1 entity. A clean unimodal cluster peaking at 3–4.</figcaption></figure>
      <figure style="margin:0"><img alt="Match composition" src="{I['06_match_composition.png']}">
        <figcaption>Fig 8 — Singleton share (left) and which sources matched entities link to (right).</figcaption></figure>
    </div>
    <table>
      <caption>Table 2 — Ground-truth matching summary (train, 2,206,821 S1 entities).</caption>
      <thead><tr><th>Metric</th><th class="n">Value</th><th>Metric</th><th class="n">Value</th></tr></thead>
      <tbody>
        <tr><td>Singletons</td><td class="n">123,247 (5.6%)</td><td>Mean matches / entity</td><td class="n">3.46</td></tr>
        <tr><td>Entities w/ ≥1 match</td><td class="n">2,083,574</td><td>Max matches</td><td class="n">11</td></tr>
        <tr><td>Link to S2 &amp; S3 both</td><td class="n">1,776,047</td><td>S2 : S3 matched-ID split</td><td class="n">48% : 52%</td></tr>
        <tr><td>S2-only / S3-only</td><td class="n">143K / 164K</td><td>Total matched IDs</td><td class="n">7.64 M</td></tr>
      </tbody>
    </table>
  </section>

  <!-- 8 -->
  <section>
    <p class="sec-no">08 / ANOMALIES &amp; CAVEATS</p>
    <h2>Read before you model</h2>
    <div class="callout"><div class="t">Blocking signal — exact names miss ~80%</div>
      Only <strong>19.7%</strong> of S2 and <strong>20.8%</strong> of S3 names exactly match a normalized S1 name
      (lowercased, punctuation-stripped). Exact/hash blocking alone caps recall near 20%. You need fuzzy blocking —
      TF-IDF / n-gram / phonetic — <em>plus</em> a cross-script transliteration path for the ~15% non-Latin S2/S3 names.</div>
    <div class="callout warn"><div class="t">Correction — the "455K numeric-only names" is an artifact</div>
      A naïve <span class="mono">[\\d\\W]+</span> regex flagged ~456K S2 names as "numeric/symbol-only". On inspection
      these are <strong>non-Latin script names</strong> (Devanagari, Tamil, etc.) that the ASCII-oriented
      <span class="mono">\\w</span> class treats as non-word. <strong>Pure-digit names = 0.</strong> A reminder to
      make every regex Unicode-aware on this dataset.</div>
    <ul class="tight">
      <li><strong>Same name ≠ same business.</strong> 30% of S1 names repeat; disambiguation needs address + country.</li>
      <li><strong>Empty-address records</strong> (~3%) can only be matched on name+country — expect lower confidence there.</li>
      <li><strong>Country is a label, not a filter.</strong> Treat it as an open set of strings; France is unseen at train.</li>
      <li><strong>Matched IDs are S2-/S3- only</strong> — no self-matches to S1, confirmed on a 200K-row sample. Mirror this in output.</li>
      <li><strong>Encoding:</strong> data is UTF-8 with mixed scripts in one field; always read TSV with <span class="mono">sep="\\t"</span> (commas live inside address &amp; ID-list fields).</li>
    </ul>
  </section>

  <!-- 9 -->
  <section>
    <p class="sec-no">09 / IMPLICATIONS FOR THE MODEL</p>
    <h2>What the EDA tells you to build</h2>
    <ul class="tight">
      <li><strong>Blocking must be fuzzy + multilingual.</strong> ~80% of true partners won't share an exact
      normalized name; ~15% aren't even in Latin script. Candidate generation is your recall ceiling.</li>
      <li><strong>Normalize legal suffixes &amp; abbreviations</strong> (Pvt↔Private, Ltd↔Limited, Rd↔Road) and
      strip leading junk / domain wrappers before comparison.</li>
      <li><strong>Address matching needs order-invariant features</strong> — components are permuted; token-set /
      bag-of-tokens similarity beats sequential edit distance.</li>
      <li><strong>Lean precision.</strong> F<sub>0.5</sub> punishes false merges 2×; with 30% duplicate names,
      requiring address/country agreement before merging protects precision.</li>
      <li><strong>Don't neglect singletons.</strong> 5.6% of entities score a full 1.0 for a correctly-predicted
      empty list — and 0.0 if you force a merge. A calibrated "no-match" threshold is free points.</li>
      <li><strong>Validate France zero-shot.</strong> Hold out a country-shifted split to approximate the unseen
      French slice rather than a random split.</li>
    </ul>
  </section>

  <p class="foot">Generated from full train + test scan · 24.23M source records + 2.21M ground-truth rows ·
  matplotlib charts embedded · figures computed on complete files (scripts sampled at 200–500K for speed).
  Underlying stats in <span class="mono">summary.json</span> / <span class="mono">supplementary.json</span>.</p>
</div>
"""
(OUT / "report.html").write_text(HTML)
print("Wrote report.html", len(HTML), "bytes")
