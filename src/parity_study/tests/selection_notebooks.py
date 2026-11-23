"""Concise notebook cells for the frozen tetrahedron selection experiment."""
from parity_study.tests.notebooks import md, code


def selection_cells(protocol):
    seeds = protocol['confirmation_seeds']
    n = len(seeds)
    primary_cut = next(c for c in protocol['cuts'] if c['id'] == protocol['primary_selection'])
    selection_definition = (
        '\n\n**Selection used in the catalogue plots (`shape_2`):**\n'
        '$$\n' + rf"B:\quad {primary_cut['low']}\leq r=\frac{{\ell_1}}{{\ell_3}}<{primary_cut['high']}"
        + '\n$$\n'
        'This keeps about **33% of tetrahedra**, including both signs of $\\widehat\\chi$. '
        'The same bounds apply to every helicity and IC; reflection leaves membership unchanged.')
    if 'extension' in protocol:
        extension = protocol['extension']
        introduction = (f"**{n} displacement realizations (seeds {seeds[0]}–{seeds[-1]}):** "
                        f"{len(extension['previous_seeds'])} retained from the original confirmation, plus "
                        f"{len(extension['added_seeds'])} newly generated. The selection cut is unchanged. "
                        "The Gaussian density seed, ZA field and lattice remain fixed.")
        inference = ("The extension was requested after viewing the ten-seed result. "
                     "The combined p-values are **nominal summaries**; the original ten-seed test remains saved. "
                     "The added realizations are also summarized separately below.")
    else:
        introduction = (f"The cut was frozen before generating displacement seeds {seeds[0]}–{seeds[-1]}. "
                        "The Gaussian density seed, ZA field and lattice remain fixed.")
        inference = ("The starred cut is the one primary test, with a two-sided p<0.05 threshold "
                     "and the same sign as discovery.")
    return [md(r'''## 6. Can a tetrahedron selection strengthen the signal?

Use the **same reflection-invariant selection** \(B\) for every helicity. A chirality band keeps both signs, for example \(0.50\leq|\widehat\chi|<0.55\). The selected catalogue signal and matched difference are
$$
S_{s,B}(\epsilon)=\frac{1}{N_{s,B}(\epsilon)}\sum_{t\in B}\widehat\chi_{s,t}(\epsilon),\qquad
D_{s,B}=S_{s,B}(+1)-S_{s,B}(-1).
$$
\(N_{s,B}\) is the number selected; \(\overline D_B\) averages the matched differences over seeds.

We explore **three chirality bands, three size groups and three shape groups** at $A=2$. The **reference vertex** has the smallest sum of squared distances to the other three vertices. Sort the lengths of its **three incident edges** as $\ell_1\leq\ell_2\leq\ell_3$. Define size $g=(\ell_1\ell_2\ell_3)^{1/3}/(L/32)$ and shape $r=\ell_1/\ell_3$. Size and shape boundaries divide the original ZA-only catalogue into thirds and are then held fixed.''' + selection_definition),
        code(r'''from parity_study.storage import digest
SELECTION = selection_output(OUT)
protocol_path = SELECTION/"protocol.json"
protocol = json.loads(protocol_path.read_text())
assert Path(protocol["discovery_run"]).resolve() == RUN.resolve()
assert protocol["A"] == comparison["A"]
mirror_rows = []
for low, high in [(0.45,0.50),(0.50,0.55)]:
    pos = np.flatnonzero(np.isclose(difference["bin_centers"], (low+high)/2)).item()
    neg = np.flatnonzero(np.isclose(difference["bin_centers"], -(low+high)/2)).item()
    mirror_rows.append(dict(band=f"{low:.2f}–{high:.2f}",
        positive=f"{difference['mean'][pos]:+.5f} ± {difference['sem'][pos]:.5f}",
        negative=f"{difference['mean'][neg]:+.5f} ± {difference['sem'][neg]:.5f}"))
display(Markdown("**Check the mirrored histogram bins (discovery seeds 0–9):**"))
show_table(mirror_rows, [("band","|χ̂| range"),("positive","Δp at positive χ̂ ± SEM"),
                         ("negative","Δp at negative χ̂ ± SEM")])
display(Markdown("The **0.50–0.55 negative-chirality counterpart is consistent with zero**; "
                 "the positive difference is in the neighbouring **−0.50 to −0.45** bin."))
winner = next(c for c in protocol["cuts"] if c["id"] == protocol["primary_selection"])
discovery_primary = next(r for r in protocol["discovery_summaries"] if r["selection"] == winner["id"])
display(Markdown(f"**Frozen primary cut:** {winner['label']}, "
    f"**{winner['low']:.5f} ≤ r < {winner['high']:.5f}**. "
    f"It retains **{100*discovery_primary['retained_fraction']:.1f}%** of tetrahedra. "
    "It was chosen by the largest absolute paired mean divided by its SEM among the nine cuts. "
    "These discovery results guided the choice; confirmation uses fresh seeds."))
selection_checks = json.loads((SELECTION/"transformation_checks.json").read_text())
assert selection_checks["passed"]
display(Markdown("**Selection checks passed:** reflections reverse every selected signal; rotations and translations preserve it."))'''),
        md(f'## 7. Does the selected signal persist with {n} realizations?\n\n' + introduction +
           '\n\n**Selected catalogue:** average $\\widehat\\chi$ over the tetrahedra passing the fixed $r=\\ell_1/\\ell_3$ interval in Section 6 (about 33% retained). ' +
           '\n\n' + r'''Each point below is \(\overline D_B\); bars are **95% Student-t intervals across seeds** (wider than one SEM). Purple shows discovery; green shows the confirmation ensemble. A bar crossing zero does not distinguish the two helicities at this interval's confidence level.''' +
           '\n\n' + inference + ' Other cuts are secondary; their p-values are corrected for nine comparisons.'),
        code(r'''from copy import deepcopy
from parity_study.selections import selection_comparison_figure, selected_catalogue_figure
confirmation_path = SELECTION/"confirmation.json"
if not confirmation_path.exists():
    raise FileNotFoundError("Independent selection confirmation is not finished yet.")
confirmation = json.loads(confirmation_path.read_text())
assert digest(protocol_path) == confirmation["protocol_sha256"]
display_protocol = deepcopy(protocol)
for cut in display_protocol["cuts"]:
    if cut["variable"] in ("size", "shape"):
        symbol = "g" if cut["variable"] == "size" else "r"
        lo, hi = cut["low"], cut["high"]
        cut["label"] = (f"{symbol} < {hi:.4f}" if lo is None else
                        f"{symbol} ≥ {lo:.4f}" if hi is None else f"{lo:.4f} ≤ {symbol} < {hi:.4f}")
fig = selection_comparison_figure(display_protocol, confirmation)
save_figure(fig,"selection_discovery_confirmation")
plt.close(fig)
primary = confirmation["primary"]
display(Markdown(f"**{confirmation['conclusion']}**  \n"
    f"**{primary['n']} realizations:** D̄_B = **{primary['mean']:+.5f} ± {primary['sem']:.5f} (SEM)**; "
    f"95% interval **[{primary['ci95_low']:+.5f}, {primary['ci95_high']:+.5f}]**; "
    f"two-sided {'nominal ' if 'extension' in protocol else ''}**p = {primary['p_two_sided']:.4g}**."))
if "added_seeds_primary" in confirmation:
    added = confirmation["added_seeds_primary"]
    display(Markdown(f"**The {added['n']} added realizations alone:** "
        f"D̄_B = **{added['mean']:+.5f} ± {added['sem']:.5f} (SEM)**; "
        f"two-sided **p = {added['p_two_sided']:.4g}**."))
band_results = [r for r in confirmation["summaries"] if r["selection"].startswith("abs_")]
show_table(band_results,[("label","Chirality selection"),("mean",f"{primary['n']}-seed D̄_B"),
    ("sem","SEM"),("retained_fraction","Fraction retained"),
    ("p_secondary_bonferroni","p (9-cut correction)")])
with np.load(SELECTION/"confirmation.npz") as z:
    confirmed_arrays = {k:z[k] for k in z.files}
display(Markdown(f"**Selected catalogue signals: {len(confirmed_arrays['seeds'])} realizations per helicity.** "
                 "One coloured dot per catalogue; "
                 "black diamonds and bars show the mean ± one SEM for ε = −1, 0, +1."))
fig = selected_catalogue_figure(protocol, confirmed_arrays)
save_figure(fig,"selected_catalogue_chi_hat_signal")
plt.close(fig)
display(Markdown("The uncertainty covers **helical-displacement realizations at one fixed density field**. "
                 "Repeating the frozen cut across independent Gaussian density fields is needed to test whether it generalizes."))
analysis = json.loads((OUT/"analysis.json").read_text())
analysis["selection_test"] = dict(protocol_sha256=confirmation["protocol_sha256"],
    discovery_run=protocol["discovery_run"], confirmation_run=confirmation["confirmation_run"],
    primary_selection=confirmation["primary_selection"], primary=primary,
    primary_confirmed=confirmation["primary_confirmed"],
    same_sign_as_discovery=confirmation["same_sign_as_discovery"])
atomic_json(OUT/"analysis.json", analysis)
selection_link = "saved/field_PVdisp/" + SELECTION.relative_to(OUT).as_posix()
display(Markdown(f"[Selection protocol]({selection_link}/protocol.json) · "
                 f"[Catalogue measurements]({selection_link}/confirmation_catalogues.csv) · "
                 f"[Complete selection results]({selection_link}/confirmation.json)"))''')]
