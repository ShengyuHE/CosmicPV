"""Concise notebook section for the frozen-cut test across independent ICs."""
from parity_study.tests.notebooks import code
from parity_study.tests.study_notebooks import md


def independent_ic_cells(protocol):
    seeds = protocol['seeds']
    return [md(f'''## 8. Does the signal persist across independent Gaussian initial conditions?

**{len(seeds)} independent initial conditions (seeds {seeds[0]}–{seeds[-1]}), at displacement strength A={protocol['A']:g}.** Each realization draws a new Gaussian density/ZA field and an independent helical pair. Within a realization, ε=−1, 0, +1 share those fields and the same particle lattice.

**Selected catalogue:** average $\\widehat\\chi$ over the tetrahedra passing the same fixed $r=\\ell_1/\\ell_3$ interval in Section 6 (about 33% retained). The cut and sample size were fixed before generation. Each IC supplies one matched difference, $D_{{s,B}}=S_{{s,B}}(+1)-S_{{s,B}}(-1)$. **SEM and 95% intervals now include variation in both density and displacement**, with one observation per IC. The fixed-IC and independent-IC estimates are shown separately.
'''), code(r'''
from parity_study.paths import independent_ic_output
from parity_study.independent_ics import comparison_figure
IC_OUT = independent_ic_output(OUT)
if IC_OUT is None:
    raise FileNotFoundError("No completed independent-IC report is active")
ic_protocol_path = IC_OUT/"protocol.json"
ic_protocol = json.loads(ic_protocol_path.read_text())
ic_report = json.loads((IC_OUT/"confirmation.json").read_text())
assert digest(ic_protocol_path) == ic_report["protocol_sha256"]
ic_primary = ic_report["primary"]
ic_audit = ic_report["independent_ic_audit"]
assert ic_audit["passed"]
display(Markdown(f"**Audit passed:** {ic_audit['independent_initial_conditions']} distinct density/ZA fields "
                 f"and helical pairs; all {ic_audit['checked_catalogues']} catalogue maps match their saved fields."))
display(Markdown(f"**{ic_report['conclusion']}**  \n"
    f"Selected mean difference **D̄_B = {ic_primary['mean']:+.5f} ± {ic_primary['sem']:.5f} (SEM)**; "
    f"95% interval **[{ic_primary['ci95_low']:+.5f}, {ic_primary['ci95_high']:+.5f}]**; "
    f"two-sided **p = {ic_primary['p_two_sided']:.4g}**. "
    f"The unchanged cut retains **{100*ic_primary['retained_fraction']:.1f}%** of tetrahedra."))
show_table(ic_report["summaries"], [("label","Selection"),("n","Independent ICs"),
    ("mean","Mean D̄_B"),("sem","SEM"),("ci95_low","95% lower"),("ci95_high","95% upper")])
fig = comparison_figure(ic_protocol, ic_report)
save_figure(fig, "independent_ic_selection_comparison")
plt.close(fig)
display(Markdown("The primary test uses the one frozen shape cut: two-sided paired Student-t, "
                 "p < 0.05 and the same sign as discovery. The all-tetrahedron baseline is descriptive. "
                 "The earlier fixed-IC 25-seed interval remains a nominal summary of its extended sample."))
'''), code(r'''
with np.load(IC_OUT/"confirmation.npz") as z:
    ic_arrays = {k:z[k] for k in z.files}
fig = selected_catalogue_figure(ic_protocol, ic_arrays)
fig.axes[0].set_title(f"Frozen shape selection · {len(ic_arrays['seeds'])} independent ICs · A={ic_protocol['A']:g}\n"
                     "One catalogue per dot; mean ± one SEM")
save_figure(fig, "independent_ic_selected_catalogue_signal")
plt.close(fig)
ic_control = next(row for row in ic_report["controls"] if row["selection"] == ic_protocol["primary_selection"])
control_rows = [dict(control=name, **ic_control[key]) for name,key in
                [("ε = 0, A = 2", "epsilon0"), ("ZA only, A = 0", "za_only")]]
show_table(control_rows, [("control","Selected control"),("mean","Mean S_B"),("sem","SEM"),
                         ("ci95_low","95% lower"),("ci95_high","95% upper")])
display(Markdown("Controls are descriptive checks of the selected signal around zero. "
                 "This experiment uses initial-condition catalogues with the same toy spectrum, lattice "
                 "and resolution; gravitational evolution is a later test."))
ic_link = "saved/field_PVdisp/" + IC_OUT.relative_to(OUT).as_posix()
display(Markdown(f"[Frozen protocol]({ic_link}/protocol.json) · "
                 f"[Catalogue measurements]({ic_link}/confirmation_catalogues.csv) · "
                 f"[Independent-IC results]({ic_link}/confirmation.json)"))
''')]
