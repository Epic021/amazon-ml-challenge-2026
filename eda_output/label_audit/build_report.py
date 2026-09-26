#!/usr/bin/env python3
"""Render eda_output/label_audit/audit.json as a self-contained HTML report (report.html)."""
import json
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "audit.json").read_text())
O, L1, L2, L3, L4, IMP = (A["overview"], A["lens1_exact_duplicates"], A["lens2_patterns"],
                          A["lens3_model"], A["lens4_assignment"], A["score_impact"])
EX = A["examples"]


def n(x):
    return f"{int(x):,}"


def p(x, d=1):
    return f"{x:.{d}f}%"


def split_bar(matches, non_matches):
    tot = matches + non_matches
    m = 100 * matches / tot if tot else 0
    return (f'<div class="bar" role="img" aria-label="{m:.1f}% match, {100-m:.1f}% non-match">'
            f'<span class="bm" style="width:{m:.2f}%"></span><span class="bn" style="width:{100-m:.2f}%"></span></div>')


def contested(matches, non_matches):
    tot = matches + non_matches
    minority = min(matches, non_matches) / tot if tot else 0
    if minority >= 0.20:
        return '<span class="chip hot">heavily mixed</span>'
    if minority >= 0.05:
        return '<span class="chip warm">mixed</span>'
    if minority >= 0.01:
        return '<span class="chip mild">some conflict</span>'
    return '<span class="chip calm">consistent</span>'


def mix_table(rows, first_col, first_key):
    body = []
    for r in rows:
        m, nm = r["matches"], r["non_matches"]
        body.append(
            f"<tr><td>{escape(str(r[first_key]))}</td><td class='num'>{n(m + nm)}</td>"
            f"<td class='num'>{n(m)}</td><td class='num'>{n(nm)}</td>"
            f"<td class='barcell'>{split_bar(m, nm)}</td><td class='num'>{p(100*m/(m+nm) if m+nm else 0)}</td>"
            f"<td>{contested(m, nm)}</td></tr>")
    return (f"<div class='tw'><table><thead><tr><th>{first_col}</th><th class='num'>pairs</th><th class='num'>match</th>"
            f"<th class='num'>non-match</th><th>label mix</th><th class='num'>match rate</th><th></th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def pair_card(r):
    lab = r.get("label", "")
    cls = "lm" if lab == "match" else "ln"
    against = '<span class="chip hot">against its pattern</span>' if r.get("against_majority") else ""
    mp = r.get("model_p")
    model = f'<span class="mp">model p(match) {mp:.2f}</span>' if isinstance(mp, (int, float)) else ""
    words = []
    for k, lbl in (("name_added", "name +"), ("name_dropped", "name −"), ("addr_added", "address +"), ("addr_dropped", "address −")):
        if r.get(k):
            words.append(f"<span class='w'>{lbl} {escape(r[k])}</span>")
    return f"""
<div class="pair">
  <div class="ph"><span class="lab {cls}">{escape(lab)}</span>{against}{model}</div>
  <div class="pg">
    <div><div class="pid">{escape(r.get('s1_id',''))}</div><div class="pn">{escape(r.get('s1_name',''))}</div>
         <div class="pa">{escape(r.get('s1_addr','')) or '<i>empty</i>'}</div></div>
    <div><div class="pid">{escape(r.get('rec_id',''))}</div><div class="pn">{escape(r.get('rec_name',''))}</div>
         <div class="pa">{escape(r.get('rec_addr','')) or '<i>empty</i>'}</div></div>
  </div>
  {'<div class="pw">' + ''.join(words) + '</div>' if words else ''}
</div>"""


# ---------------------------------------------------------------- sections
comp = O["comparable_pairs"]
dup_raw, dup_norm = L1["identical raw text"], L1["identical after normalisation"]
s1d = L1["duplicate_s1_records"]
imp_p, imp_m = IMP["pattern_majority_rule"], IMP["model_threshold_0.5"]

by_addr = dup_norm.get("conflict_groups_by_address", {})
conf_empty = sum(v["address empty"] for v in by_addr.values())
conf_present = sum(v["address present"] for v in by_addr.values())
tiles = [
    (n(comp), "pairs compared", "S2/S3 records each set against their S1 or closest S1 lookalike"),
    (p(L2["pct_of_comparable_pairs"], 2), "labelled against their pattern", f"{n(L2['pairs_against_their_pattern_majority'])} pairs"),
    (n(L2["patterns_with_both_labels"]), "patterns carry both labels", f"of {n(L2['patterns'])} observed patterns"),
    (n(conf_empty + conf_present), "identical-record conflicts",
     f"groups after normalisation; {n(conf_empty)} of them have no address"),
    (p(L3["confident_learning_pct"], 2), "labels the model contradicts", f"{n(L3['confident_learning_label_issues'])} pairs, out-of-fold"),
    (f"{imp_p['macro_f05_all_s1']:.4f}", "F0.5 ceiling", "if every pattern takes its majority label"),
]
rc_ = L2["reported_cases"]


def share(v):
    t = v["matches"] + v["non_matches"]
    return f"{100 * v['matches'] / t:.2f}%" if t else "n/a"


findings = [
    f"<b>Exact duplicates never conflict.</b> {n(dup_raw['duplicate_groups'])} groups of records with identical raw text: "
    f"{n(dup_raw['groups_some_matched_some_unmatched'] + dup_raw['groups_matched_to_different_s1'])} carry different labels or S1s.",
    f"<b>Identical name and address.</b> {n(rc_['name and address identical']['pairs'])} pairs, "
    f"{share(rc_['name and address identical'])} matched; {n(rc_['name and address identical']['against_pattern_majority'])} against.",
    f"<b>Only the house/plot number differs.</b> {n(rc_['only the house/plot number differs']['pairs'])} pairs, "
    f"{share(rc_['only the house/plot number differs'])} matched. With the name also identical: "
    f"{n(rc_['...and the name is identical']['pairs'])} pairs, {share(rc_['...and the name is identical'])} matched, "
    f"{n(rc_['...and the name is identical']['against_pattern_majority'])} against.",
    f"<b>Only the name differs, address fully identical.</b> {n(rc_['...and the address is fully identical']['pairs'])} pairs, "
    f"{share(rc_['...and the address is fully identical'])} matched, "
    f"{n(rc_['...and the address is fully identical']['against_pattern_majority'])} against.",
    f"<b>Only typo-like differences.</b> {n(rc_['only typo-like differences']['pairs'])} pairs, "
    f"{share(rc_['only typo-like differences'])} matched, {n(rc_['only typo-like differences']['against_pattern_majority'])} against.",
    f"<b>Identical names with no address.</b> {n(conf_empty)} groups of records whose normalised name is identical and whose address "
    f"is empty carry different labels or different S1s.",
    f"<b>Across all patterns.</b> {n(L2['patterns_with_both_labels'])} of {n(L2['patterns'])} patterns carry both labels; "
    f"{n(L2['pairs_against_their_pattern_majority'])} pairs ({p(L2['pct_of_comparable_pairs'], 2)}) go against their pattern's majority; "
    f"{p(L2['pct_pairs_in_patterns_with_minority_over_5pct'])} of pairs sit in patterns whose minority label exceeds 5%.",
    f"<b>Model check.</b> Out-of-fold AUC {L3['auc']:.4f}. {n(L3['confident_learning_label_issues'])} labels "
    f"({p(L3['confident_learning_pct'], 2)}) contradict the model at class-calibrated confidence; "
    f"{p(L3['pct_pairs_uncertain_0.2_to_0.8'])} of pairs stay between p = 0.2 and 0.8.",
    f"<b>Matches with nothing in common.</b> {n(L4['matches_sharing_no_name_word_and_little_address'])} matched pairs share no "
    f"name word and under a quarter of their address words.",
]
findings_html = "<ul class='findings'>" + "".join(f"<li>{f}</li>" for f in findings) + "</ul>"
tiles_html = "".join(f"<div class='tile'><div class='tv'>{v}</div><div class='tl'>{l}</div><div class='ts'>{s}</div></div>"
                     for v, l, s in tiles)

rc = L2["reported_cases"]
reported_rows = [{"case": k, **v} for k, v in rc.items()]
reported_html = mix_table(reported_rows, "reported case", "case")

bd = L2["breakdowns"]
bd_num = mix_table(bd["name identical, only numbers differ, by number change"], "number change", "number change")
bd_numname = mix_table(bd["only numbers differ, by name change"], "name change", "name change")
bd_addrsame = mix_table(bd["address identical, by name change"], "name change", "name change")

top = L2["top_patterns_pooled"][:25]
top_html = mix_table([{"pattern": r["pattern"].replace(" | ", " · "), "matches": r["matches"], "non_matches": r["non_matches"]}
                      for r in top], "pattern (name · address · numbers)", "pattern")


def kv_table(d, k1, k2):
    return ("<div class='tw'><table><thead><tr><th>" + k1 + "</th><th class='num'>" + k2 + "</th></tr></thead><tbody>"
            + "".join(f"<tr><td>{escape(str(k))}</td><td class='num'>{n(v)}</td></tr>" for k, v in d.items())
            + "</tbody></table></div>")


def l1_table(d, title):
    flat = {k.replace("_", " "): v for k, v in d.items() if not isinstance(v, dict)}
    return kv_table(flat, title, "count")


def l1_split(d):
    by = d.get("conflict_groups_by_address", {})
    if not by:
        return "<p class='muted'>No conflicting groups.</p>"
    rows = "".join(f"<tr><td>{escape(conf)}</td><td class='num'>{n(v['address present'])}</td>"
                   f"<td class='num'>{n(v['address empty'])}</td></tr>" for conf, v in by.items())
    return ("<div class='tw'><table><thead><tr><th>conflict</th><th class='num'>groups, address present</th>"
            "<th class='num'>groups, address empty</th></tr></thead><tbody>" + rows + "</tbody></table></div>")


def words_html(d):
    return "".join(f"<span class='w'>{escape(k)} <b>{n(v)}</b></span>" for k, v in d.items()) or "<span class='muted'>none</span>"


ex2 = []
for g in EX.get("lens2", [])[:10]:
    cards = "".join(pair_card(r) for r in g["rows"])
    ex2.append(f"""<div class="exg"><div class="exh"><b>{escape(g['pattern'].replace(' | ', ' · '))}</b>
      <span class="muted">{escape(g['country'])} · {escape(g['source'])} · {n(g['pairs'])} pairs · {g['match_rate_pct']:.1f}% match</span></div>
      <div class="pairs">{cards}</div></div>""")

ex3 = "".join(pair_card({**r, "against_majority": False}) for r in EX.get("lens3", [])[:12])

# holdout experiment: does cleaning the training labels help?
exp_path = HERE / "curation_experiment.json"
exp_html = ""
if exp_path.exists():
    E = json.loads(exp_path.read_text())
    base = E["results"][0]["macro_f05_at_0.5"]
    rows = "".join(
        f"<tr><td>{escape(r['variant'][3:])}</td><td class='num'>{n(r['training_pairs'])}</td>"
        f"<td class='num'>{n(r['removed_from_training'])}</td><td class='num'>{r['macro_f05_at_0.5']:.4f}</td>"
        f"<td class='num'>{(r['macro_f05_at_0.5'] - base) * 100:+.2f}</td><td class='num'>{r['holdout_log_loss']:.3f}</td></tr>"
        for r in E["results"])
    exp_html = f"""
  <section>
    <p class="lens">Holdout experiment</p>
    <h2>Does removing the contradictions from training help?</h2>
    <p>Holdout: every pair of 20% of the S1 businesses ({n(E['holdout_s1_businesses'])} businesses, {n(E['holdout_pairs'])} pairs),
    labels untouched, as in test. Each variant trains the same model on the remaining businesses with different pairs removed.</p>
    <div class="tw"><table><thead><tr><th>training data</th><th class="num">training pairs</th><th class="num">removed</th>
    <th class="num">holdout macro F0.5</th><th class="num">vs keep all (points)</th><th class="num">log loss</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  </section>"""

dup_rows = EX.get("lens1", [])[:24]
dup_html = ("<div class='tw'><table><thead><tr><th>fingerprint</th><th>conflict</th><th>group</th><th>record</th>"
            "<th>assigned to</th><th>name</th><th>address</th></tr></thead><tbody>"
            + "".join(f"<tr><td>{escape(str(r['fingerprint']))}</td><td>{escape(str(r['conflict']))}</td><td class='num'>{r['gid']}</td>"
                      f"<td class='mono'>{escape(str(r['rec_id']))}</td><td class='mono'>{escape(str(r['assigned_s1']))}</td>"
                      f"<td>{escape(str(r['name']))}</td><td>{escape(str(r['addr']))}</td></tr>" for r in dup_rows)
            + "</tbody></table></div>") if dup_rows else "<p class='muted'>No conflicting identical records.</p>"

an_rows = EX.get("lens4", [])[:16]
an_html = ("<div class='tw'><table><thead><tr><th>anomaly</th><th>record</th><th>record text</th>"
           "<th>assigned S1</th><th>exact lookalike S1</th></tr></thead><tbody>"
           + "".join(f"<tr><td>{escape(str(r.get('anomaly','')))}</td><td class='mono'>{escape(str(r.get('rec_id','')))}</td>"
                     f"<td>{escape(str(r.get('name','')))}<br><span class='muted'>{escape(str(r.get('addr','')))}</span></td>"
                     f"<td><span class='mono'>{escape(str(r.get('assigned_s1_s1_id') or 'unmatched'))}</span><br>"
                     f"{escape(str(r.get('assigned_s1_name') or ''))}<br><span class='muted'>{escape(str(r.get('assigned_s1_addr') or ''))}</span></td>"
                     f"<td><span class='mono'>{escape(str(r.get('lookalike_s1_s1_id') or ''))}</span><br>"
                     f"{escape(str(r.get('lookalike_s1_name') or ''))}<br><span class='muted'>{escape(str(r.get('lookalike_s1_addr') or ''))}</span></td></tr>"
                     for r in an_rows)
           + "</tbody></table></div>") if an_rows else "<p class='muted'>None found.</p>"

fi = L3["feature_importance_gain_pct"]
fi_html = "".join(f"<div class='fi'><span>{escape(k)}</span><span class='fib'><i style='width:{min(100, v*2):.1f}%'></i></span>"
                  f"<span class='num'>{v:.1f}%</span></div>" for k, v in fi.items())

html = f"""<title>Train Label Audit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root {{
  --ground:#F2F4F6; --surface:#FFFFFF; --raised:#F7F8FA; --ink:#12171D; --muted:#5A6471; --faint:#8993A0;
  --line:#DCE1E7; --accent:#0F6E7C; --accent-soft:#DDEEF1;
  --match:#2D7A4F; --match-soft:#DFF0E6; --non:#B14A2A; --non-soft:#F7E3DA;
  --hot:#A4481E; --hot-soft:#F8E2D6; --warm:#8A6400; --warm-soft:#F6ECD2; --mild:#566170; --mild-soft:#E9EDF1;
  --sans:"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono:"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0E1216; --surface:#151A20; --raised:#11161B; --ink:#E3E8EE; --muted:#9AA5B2; --faint:#6D7885;
    --line:#252D36; --accent:#5DB6C4; --accent-soft:#12303A;
    --match:#6CC08D; --match-soft:#16301F; --non:#E5895F; --non-soft:#361F16;
    --hot:#EE9A70; --hot-soft:#3A2016; --warm:#E3B656; --warm-soft:#34290F; --mild:#A3AEBB; --mild-soft:#1E252D;
    color-scheme:dark;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0E1216; --surface:#151A20; --raised:#11161B; --ink:#E3E8EE; --muted:#9AA5B2; --faint:#6D7885;
  --line:#252D36; --accent:#5DB6C4; --accent-soft:#12303A;
  --match:#6CC08D; --match-soft:#16301F; --non:#E5895F; --non-soft:#361F16;
  --hot:#EE9A70; --hot-soft:#3A2016; --warm:#E3B656; --warm-soft:#34290F; --mild:#A3AEBB; --mild-soft:#1E252D;
  color-scheme:dark;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font:15px/1.6 var(--sans); }}
.wrap {{ max-width:1120px; margin:0 auto; padding-inline:20px; padding-block:48px 96px; }}
h1 {{ font-size:clamp(28px,4.2vw,40px); line-height:1.12; letter-spacing:-.015em; margin:0 0 12px; text-wrap:balance; font-weight:700; }}
h2 {{ font-size:22px; margin:0 0 6px; letter-spacing:-.01em; text-wrap:balance; }}
h3 {{ font-size:15px; margin:26px 0 8px; }}
p {{ margin:10px 0; max-width:72ch; }}
.eyebrow {{ font:600 11.5px var(--mono); letter-spacing:.12em; text-transform:uppercase; color:var(--accent); margin:0 0 12px; }}
.lede {{ color:var(--muted); font-size:17px; max-width:70ch; }}
.muted {{ color:var(--muted); }}
.mono {{ font-family:var(--mono); font-size:13px; }}
section {{ margin-top:48px; padding-top:28px; border-top:1px solid var(--line); }}
.lens {{ font:600 11.5px var(--mono); letter-spacing:.1em; text-transform:uppercase; color:var(--faint); margin:0 0 4px; }}
.tiles {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:12px; margin:26px 0 8px; }}
.tile {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }}
.tv {{ font:600 28px/1.1 var(--mono); letter-spacing:-.02em; font-variant-numeric:tabular-nums; }}
.tl {{ font-weight:600; margin-top:6px; }}
.ts {{ color:var(--muted); font-size:13px; line-height:1.4; margin-top:2px; }}
.note {{ background:var(--raised); border:1px solid var(--line); border-radius:10px; padding:12px 16px; color:var(--muted); font-size:14px; }}
.tw {{ overflow-x:auto; margin:12px 0; border:1px solid var(--line); border-radius:10px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th {{ text-align:left; font:600 11px var(--mono); letter-spacing:.06em; text-transform:uppercase; color:var(--muted);
      padding:9px 12px; border-bottom:1px solid var(--line); background:var(--raised); white-space:nowrap; }}
td {{ padding:8px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
tbody tr:last-child td {{ border-bottom:0; }}
.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap; }}
.barcell {{ min-width:150px; width:22%; }}
.bar {{ display:flex; height:10px; border-radius:5px; overflow:hidden; background:var(--line); margin-top:5px; }}
.bm {{ background:var(--match); }} .bn {{ background:var(--non); }}
.chip {{ display:inline-block; font:600 11px var(--mono); padding:2px 8px; border-radius:999px; white-space:nowrap; }}
.hot {{ color:var(--hot); background:var(--hot-soft); }} .warm {{ color:var(--warm); background:var(--warm-soft); }}
.mild {{ color:var(--mild); background:var(--mild-soft); }} .calm {{ color:var(--match); background:var(--match-soft); }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--muted); margin:6px 0 0; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
.exg {{ margin:22px 0; }}
.exh {{ display:flex; flex-wrap:wrap; gap:4px 12px; align-items:baseline; margin-bottom:8px; }}
.pairs {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(330px, 1fr)); gap:10px; }}
.pair {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:10px 12px; font-size:13.5px; }}
.ph {{ display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-bottom:6px; }}
.lab {{ font:700 11px var(--mono); text-transform:uppercase; letter-spacing:.06em; padding:2px 8px; border-radius:4px; }}
.lm {{ color:var(--match); background:var(--match-soft); }} .ln {{ color:var(--non); background:var(--non-soft); }}
.mp {{ margin-left:auto; font:12px var(--mono); color:var(--muted); }}
.pg {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.pid {{ font:12px var(--mono); color:var(--faint); }}
.pn {{ font-weight:600; line-height:1.35; overflow-wrap:anywhere; }}
.pa {{ font:12.5px/1.45 var(--mono); color:var(--muted); overflow-wrap:anywhere; }}
.pw {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:8px; }}
.w {{ font:12px var(--mono); background:var(--raised); border:1px solid var(--line); border-radius:6px; padding:1px 6px; }}
.words {{ display:flex; flex-wrap:wrap; gap:5px; margin:6px 0 14px; }}
.fi {{ display:grid; grid-template-columns:minmax(120px, 220px) 1fr auto; gap:10px; align-items:center; font:13px var(--mono); padding:3px 0; }}
.fib {{ height:8px; background:var(--line); border-radius:4px; overflow:hidden; }}
.fib i {{ display:block; height:100%; background:var(--accent); }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
ul.files {{ padding-left:18px; }} ul.files li {{ margin:4px 0; }}
ul.findings {{ padding-left:20px; margin:10px 0 0; max-width:80ch; }} ul.findings li {{ margin:8px 0; }}
code {{ font:13px var(--mono); background:var(--raised); border:1px solid var(--line); border-radius:4px; padding:0 4px; }}
@media (max-width: 760px) {{ .tiles, .grid2 {{ grid-template-columns:1fr; }} .pg {{ grid-template-columns:1fr; }} }}
</style>
<div class="wrap">
  <p class="eyebrow">ML Challenge 2026 · training-set label audit</p>
  <h1>Where the training labels contradict themselves</h1>
  <p class="lede">Every S2/S3 record in the training set was paired with its ground-truth S1 business, or, if unmatched,
  with the closest S1 lookalike. Each pair was then described only by what can be observed: how the name, the address
  and the numbers differ. This report counts, across the whole training set, how often pairs that look the same way
  carry opposite labels.</p>
  <div class="tiles">{tiles_html}</div>
  <h2 style="margin-top:34px">Findings</h2>
  {findings_html}
  <p class="note">This audit measures inconsistency: pairs described the same way, or identical records, that received
  different labels. It cannot tell which of two conflicting labels is the wrong one. All counts are from the training
  files; test has no labels.</p>

  <section>
    <p class="lens">Lens 1 · identical records</p>
    <h2>Identical records with different labels</h2>
    <p>Records grouped by fingerprint, with no similarity measure involved. A conflict is a group of identical records where
    some copies are matched and others are not, or where copies are matched to different S1 businesses.</p>
    <div class="grid2">
      {l1_table(dup_raw, "identical raw text")}
      {l1_table(dup_norm, "identical after normalisation")}
    </div>
    <h3>Conflicting groups, by whether the copies have an address</h3>
    <p>Copies with an empty address are identical names with nothing else to tell them apart.</p>
    {l1_split(dup_norm)}
    <h3>Duplicate S1 records (the reference is described as deduplicated)</h3>
    {l1_table(s1d, "S1 duplicates")}
    <h3>Examples</h3>
    {dup_html}
  </section>

  <section>
    <p class="lens">Lens 2 · difference patterns</p>
    <h2>The reported cases, measured on every pair</h2>
    <p>Each row counts all comparable pairs of that kind. The bar shows the label mix: <b>green</b> matched, <b>clay</b> not matched.</p>
    <div class="legend"><span><i style="background:var(--match)"></i>match</span><span><i style="background:var(--non)"></i>non-match</span></div>
    {reported_html}
    <h3>Name identical, only the numbers differ, by kind of number change</h3>
    {bd_num}
    <h3>Only the numbers differ, by kind of name change</h3>
    {bd_numname}
    <h3>Address fully identical, by kind of name change</h3>
    {bd_addrsame}
  </section>

  <section>
    <p class="lens">Lens 2 · every pattern</p>
    <h2>Beyond the reported cases: the most contradicted patterns</h2>
    <p>{n(L2['patterns'])} distinct patterns (country × source × name change × address change × number change);
    {n(L2['patterns_with_both_labels'])} of them carry both labels. {p(L2['pct_pairs_in_patterns_with_minority_over_5pct'])} of
    comparable pairs sit in patterns where the minority label exceeds 5%, and {p(L2['pct_pairs_in_patterns_with_minority_over_20pct'])}
    where it exceeds 20%. Ranked by the number of pairs labelled against their pattern's majority, pooled over country and source:</p>
    {top_html}
  </section>

  <section>
    <p class="lens">Lens 3 · model</p>
    <h2>Labels a model finds implausible</h2>
    <p>A gradient-boosted model was trained on every pair feature: the difference pattern, similarity scores, number gaps,
    and fold-wise evidence from which words were added or dropped. It was scored out-of-fold, grouped by S1. It separates the labels well
    (AUC {L3['auc']:.4f}), yet {p(L3['pct_pairs_uncertain_0.2_to_0.8'])} of pairs stay between 0.2 and 0.8.</p>
    <div class="tw"><table><tbody>
      <tr><td>matches the model rejects (p &lt; 0.02)</td><td class="num">{n(L3['matches_model_rejects_p_below_0.02'])}</td></tr>
      <tr><td>non-matches the model accepts (p &gt; 0.98)</td><td class="num">{n(L3['non_matches_model_accepts_p_above_0.98'])}</td></tr>
      <tr><td>confident-learning label issues (class-calibrated thresholds)</td><td class="num">{n(L3['confident_learning_label_issues'])}</td></tr>
      <tr><td>pairs with p between 0.05 and 0.95</td><td class="num">{p(L3['pct_pairs_uncertain_0.05_to_0.95'])}</td></tr>
    </tbody></table></div>
    <div class="grid2">
      <div><h3>Where the rejected matches sit</h3>{kv_table(L3['top_patterns_matches_rejected'], 'pattern', 'pairs')}</div>
      <div><h3>Where the accepted non-matches sit</h3>{kv_table(L3['top_patterns_non_matches_accepted'], 'pattern', 'pairs')}</div>
    </div>
    <h3>Words added to the name in flagged pairs</h3>
    <p class="muted">Matches the model rejects</p><div class="words">{words_html(L3['top_added_name_words']['matches the model rejects'])}</div>
    <p class="muted">Non-matches the model accepts</p><div class="words">{words_html(L3['top_added_name_words']['non-matches the model accepts'])}</div>
    <h3>What the model relies on (share of split gain)</h3>
    {fi_html}
    <h3>Flagged pairs</h3>
    <div class="pairs">{ex3}</div>
  </section>

  <section>
    <p class="lens">Lens 4 · assignment</p>
    <h2>Records that look exactly like another business</h2>
    {kv_table(L4, 'anomaly', 'records')}
    {an_html}
  </section>

  <section>
    <p class="lens">Effect on the metric</p>
    <h2>What the contradictions cost</h2>
    <p>Macro F0.5 on the training S1 entities if a matcher labelled every pair perfectly except these. Pairs outside the
    comparison (unmatched records with no lookalike, matches sharing no key with their S1) are assumed correct, so these
    are optimistic ceilings.</p>
    <div class="tw"><table><thead><tr><th>rule</th><th class="num">macro F0.5, all S1</th><th class="num">S1 touched</th><th class="num">mean F0.5 on touched</th></tr></thead><tbody>
      <tr><td>every pattern takes its majority label</td><td class="num">{imp_p['macro_f05_all_s1']:.4f}</td><td class="num">{n(imp_p['s1_touched'])}</td><td class="num">{imp_p['mean_f05_on_touched_s1']:.4f}</td></tr>
      <tr><td>out-of-fold model, threshold 0.5</td><td class="num">{imp_m['macro_f05_all_s1']:.4f}</td><td class="num">{n(imp_m['s1_touched'])}</td><td class="num">{imp_m['mean_f05_on_touched_s1']:.4f}</td></tr>
    </tbody></table></div>
  </section>

  {exp_html}

  <section>
    <p class="lens">Examples</p>
    <h2>Same pattern, opposite labels</h2>
    <p>For the most contradicted patterns: pairs that follow the pattern's majority label next to pairs that don't. Left: S1. Right: the S2/S3 record.</p>
    {''.join(ex2)}
  </section>

  <section>
    <p class="lens">Method and limits</p>
    <h2>How the pairs were built</h2>
    <p><b>Pairing.</b> Matched records are set against their ground-truth S1. Unmatched records are set against the most similar
    S1 that shares one of six blocking keys: identical address words (house numbers ignored), identical core name, 4-letter-prefix
    versions of both, address words of 4+ letters only, and sound-alike forms of the name words. A matched pair counts as comparable
    only if it shares one of those keys too, so both labels are measured the same way. {n(O['unmatched_without_lookalike'])} unmatched
    records have no lookalike and {n(O['matches_sharing_no_key_with_their_s1'])} matches share no key with their S1; both sit
    outside the pattern and model lenses.</p>
    <p><b>Normalisation.</b> Text is romanised (Indic scripts), lower-cased and split into words. In addresses, state names and codes,
    unit words and Indic-script words are ignored, ordinals become street words, and street abbreviations are expanded; numbers are
    compared separately, as a multiset. In names, legal forms, honorifics and fillers are counted as minor words, and repeated words
    are kept. A name in an Indic script is compared with a Latin one through sound-alike forms of each word. A typo is a word 1–2
    character edits from a word on the other side.</p>
    <p><b>What this cannot tell.</b> Pairing an unmatched record with its most similar S1 favours lookalikes, which makes
    contradictions easier to find, not harder. Blocking misses decoys whose name and address both changed. None of these counts
    say which of two conflicting labels is correct.</p>
    <h3>Files (eda_output/label_audit/)</h3>
    <ul class="files">
      <li><code>flagged_pairs.tsv</code>: {n(A['flagged_pairs_total'])} pairs labelled against their pattern and/or rejected by the model, with both texts</li>
      <li><code>pattern_summary.tsv</code>: every pattern with match / non-match counts</li>
      <li><code>duplicate_conflicts.tsv</code>: identical records with conflicting labels</li>
      <li><code>s1_duplicates.tsv</code>: duplicate S1 records</li>
      <li><code>assignment_anomalies.tsv</code>: records that are exact copies of a different S1</li>
      <li><code>build_pairs.py</code>, <code>analyze.py</code>, <code>build_report.py</code>: the pipeline that produced all of the above</li>
    </ul>
  </section>
</div>
"""
(HERE / "report.html").write_text(html)
print(f"report.html written ({len(html)/1024:.0f} KB)")
