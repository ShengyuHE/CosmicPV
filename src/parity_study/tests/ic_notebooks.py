"""Compact results notebook for Gaussian mesh initial conditions."""
from parity_study.tests.notebooks import notebook, code, TABLE_HELPER, report_filename
from parity_study.tests.study_notebooks import md
from parity_study.paths import project_for_run
from parity_study.storage import digest
from pathlib import Path


def mesh_notebook(run):
    project = project_for_run(run)
    filename = report_filename(run.name)
    topic = Path(filename or f"{run.name}.ipynb").stem
    title = "# Fiducial PV initial conditions" if filename == "field_PVdisp.ipynb" else "# Gaussian-field initial conditions with PV"
    cells = [md(title + r'''

**Question:** does changing the displacement's handedness produce a detectable change in tetrahedron chirality?

**PV** means parity violation; **helicity** describes the handedness of the displacement field's twisting pattern. Our estimator \(\widehat\chi\) is normalized tetrahedron chirality, a dimensionless number in \([-1,1]\) whose sign reverses under reflection ([geometric definition](basic.ipynb)).

**Chirality reminder.** Order each tetrahedron's vertices by increasing \(s_i=\sum_{j\ne i}|\mathbf r_i-\mathbf r_j|^2\), then label them \(\mathbf r_0,\ldots,\mathbf r_3\). With \(\mathbf v_a=\mathbf r_a-\mathbf r_0\) for \(a=1,2,3\),
\[
\chi=\mathbf v_1\cdot(\mathbf v_2\times\mathbf v_3),\qquad
\widehat\chi=\frac{\chi}{|\mathbf v_1||\mathbf v_2||\mathbf v_3|}\in[-1,1].
\]
\(\chi\) is six times the signed tetrahedron volume; \(\widehat\chi\) removes the overall size scale. The ordering makes the sign independent of input vertex labels; tied scores give zero chirality. Reflection reverses both signs. The **catalogue signal** \(S_s(\epsilon)\) averages \(\widehat\chi\) over tetrahedra (Section 4).

Construct a Gaussian density mesh, derive its Zel'dovich approximation (ZA) displacement, add an independent helical displacement, then place lattice particles using their sum. This follows the initial-condition construction in [Shim et al., Eqs. 1–8](https://arxiv.org/html/2406.06080v2), using our controlled toy spectrum. Gravitational evolution is a later step.'''),
        code(f'''
from pathlib import Path
import contextlib, csv, html, io, json, sys
import numpy as np
from IPython.display import display, Markdown, HTML
PROJECT = Path({str(project)!r})
RUN = Path({str(run)!r})
sys.path.insert(0, str(PROJECT/"src"))
from parity_study.paths import OUTPUT_ROOT, expose_output, selection_output, independent_ic_output
from parity_study.storage import atomic_json, atomic_npz
from parity_study.summarize import summarize
from parity_study.initial_conditions import cic_density, displace_mesh
OUT = OUTPUT_ROOT/{topic!r}
(OUT/"figures").mkdir(parents=True, exist_ok=True)
expose_output(PROJECT, OUT)
with contextlib.redirect_stdout(io.StringIO()):
    summary = summarize(RUN)
config = json.loads((RUN/"config.json").read_text())
field = config["field"]
validation = json.loads((RUN/"validation/report.json").read_text())
pilot = config.get("purpose") == "technical_pilot"
SELECTION = selection_output(OUT)
has_selection = (SELECTION/"protocol.json").exists()
display(Markdown(f"**{{'Discovery ensemble' if has_selection else 'Technical pilot' if pilot else 'Signal scan'}}:** "
                 f"{{summary['completed_records']}}/{{summary['expected_records']}} saved catalogues; "
                 f"validation **{{'passed' if validation['passed'] else 'failed'}}**."))
if pilot and not has_selection:
    display(Markdown("**This small run checks the construction and measurement pipeline; it makes no detection claim.**"))
if has_selection:
    selection_protocol = json.loads((SELECTION/"protocol.json").read_text())
    selection_seeds = selection_protocol["confirmation_seeds"]
    display(Markdown("**Sections 1–5:** original displacement seeds 0–9. "
                     f"**Sections 6–7:** tetrahedron selections with **{{len(selection_seeds)}} realizations** "
                     f"(displacement seeds {{selection_seeds[0]}}–{{selection_seeds[-1]}}) at A=2."))
if independent_ic_output(OUT) is not None:
    independent_protocol = json.loads((independent_ic_output(OUT)/"protocol.json").read_text())
    display(Markdown(f"**Section 8:** **{{len(independent_protocol['seeds'])}} independent Gaussian initial conditions**, "
                     "with the same frozen shape cut and A=2; both density and helical fields vary."))
if not summary["complete"]:
    display(Markdown("**Partial ensemble: results are provisional.**"))
fixed_density = field.get("density_seed") is not None
if fixed_density:
    display(Markdown(f"**Fixed initial condition:** density seed **{{field['density_seed']}}** "
                     f"sets the Gaussian density and ZA displacement for all **{{config['scan']['seeds']}}** "
                     "helical-displacement seeds. The lattice is also fixed. "
                     "**Error bars describe displacement variation at this one initial condition.**"))
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams.update({{"figure.dpi":110,"font.size":10}})
def save_figure(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(OUT/"figures"/f"{{name}}.{{suffix}}", dpi=160, bbox_inches="tight")
    plt.show()
def save_table(rows, name):
    if rows:
        with (OUT/f"{{name}}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
{TABLE_HELPER}
'''),
        md(r'''## 1. Field first, particles second

\(\delta\) is density contrast (density divided by its mean, minus one); \(\delta_{\mathbf k}\) is its Fourier mode at wavevector \(\mathbf k\), with \(k=|\mathbf k|\). GRF means Gaussian random field, \(P_\delta\) its power spectrum, and \(\boldsymbol\Psi_{\rm ZA}\) its ZA displacement:
$$
\delta_{\bf k}\sim\mathrm{GRF}[P_\delta(k)],\qquad
\boldsymbol\Psi_{{\rm ZA},{\bf k}}=\frac{i\mathbf k}{k^2}\delta_{\bf k},\qquad
-\nabla\cdot\boldsymbol\Psi_{\rm ZA}=\delta.
$$
Our toy shape is \(P_\delta\propto k^n e^{-k^2R_\delta^2}\): \(n\) is the spectral slope, \(R_\delta\) the density smoothing length, and \(\sigma_\delta\) the ensemble root mean square (RMS) density contrast.
For the helical displacement, \(R_h\) is its smoothing length, \(W_{{\rm hel},{\bf k}}\) its normalized mode weight, \(P_{\rm hel}\) its power, and \(A\) the **displacement strength**:
$$
W_{{\rm hel},{\bf k}}\propto P_\delta(k)e^{-k^2R_h^2}/k^4,\quad
\sum_{\bf k}W_{{\rm hel},{\bf k}}=1,\quad P_{\rm hel}=A^2W_{\rm hel}.
$$
This normalization makes \(A\) the ensemble RMS helical displacement, in the same length units as the box; individual realizations fluctuate around this scale.

\(\epsilon\) controls the helicity mixture: \(+1\) selects the positive-helicity field \(\mathbf u_+\), \(-1\) selects the negative-helicity field \(\mathbf u_-\), and \(0\) mixes equal expected power. The independent fields \(\mathbf u_\pm\) each have unit ensemble RMS. For initial lattice positions \(\mathbf q\), final positions \(\mathbf x\), and periodic box side \(L\):
$$
\mathbf x=\left[\mathbf q+\boldsymbol\Psi_{\rm ZA}(\mathbf q)
+A\left(\sqrt{\frac{1+\epsilon}{2}}\mathbf u_+(\mathbf q)
+\sqrt{\frac{1-\epsilon}{2}}\mathbf u_-(\mathbf q)\right)\right]\bmod L.
$$
All components are evaluated at the same \(\mathbf q\); the scalar and helical fields use independent draws.'''),
        code('''show_table([dict(L=field["L"], N=field["N"], particle_grid=field["particle_grid"],
    Ngal=field["Ngal"], sigma=field["sigma_delta"], Rh=field["helical_smoothing"],
    spectrum=str(field["spectrum"]), A=str(config["scan"]["amplitudes"]))],
    [("L","Box L"),("N","Mesh N"),("particle_grid","Particle grid"),("Ngal","Particles"),
     ("sigma","Density RMS σδ"),("Rh","Helical smoothing Rh"),("spectrum","Density spectrum"),("A","Displacement strength A")])
example = next(iter(sorted((RUN/"fields").glob("main_*.npz"))), None)
if example is None:
    raise FileNotFoundError("No saved mesh realization yet; complete a seed block first.")
with np.load(example) as z:
    arrays = {k:z[k] for k in ("q", "za", "plus", "minus", "delta_mesh")}
    meta = json.loads(str(z["metadata_json"]))
display(Markdown(f"**Illustration seed:** {meta['seed']}. The paired catalogues share this density field and particle lattice."))
if fixed_density:
    display(Markdown(f"Density seed **{field['density_seed']}** is held fixed across all displacement seeds."))
kmax = field.get("kmax") or np.pi*min(field["N"],field["particle_grid"])/field["L"]
spacing = field["L"]/field["particle_grid"]
display(Markdown(f"Particle spacing **{spacing:g}**; Fourier cutoff **k_max={kmax:.5g}** "
                 "(limited by both mesh and particle sampling)."))'''),
        md(r'''## 2. What does the particle density look like?

The first panel shows the input Gaussian field; the others show densities measured by depositing particles onto the mesh with cloud-in-cell (CIC) interpolation. All panels average the same central slab on a display grid no finer than the particle lattice. Finite displacements can change the final density beyond the linear relation above.'''),
        code(r'''L, N = field["L"], field["N"]
A = config["scan"]["amplitudes"][-1]
plot_N = min(N, field["particle_grid"])
if N % plot_N:
    plot_N = max(n for n in range(1,plot_N+1) if N % n == 0)
step = N // plot_N
density = [arrays["delta_mesh"][::step,::step,::step]]
labels = ["Input Gaussian δ"]
for amp,eps,label in [(0.,0.,"ZA only (A=0)"),(A,1.,r"ZA + helical ($\epsilon=+1$)"),(A,-1.,r"ZA + helical ($\epsilon=-1$)")]:
    cat, _ = displace_mesh(arrays, A=amp, eps=eps, L=L)
    density.append(cic_density(cat, L=L, N=plot_N))
    labels.append(label)
slab = slice(max(0,plot_N//2-1), min(plot_N,plot_N//2+2))
panels = [d[:,:,slab].mean(axis=2) for d in density]
lim = max(abs(p).max() for p in panels)
fig, axes = plt.subplots(1,4,figsize=(13,3.3),constrained_layout=True)
for ax,p,label in zip(axes,panels,labels):
    im=ax.imshow(p.T, origin="lower",extent=[0,L,0,L],cmap="RdBu_r",vmin=-lim,vmax=lim)
    ax.set(title=label,xlabel="x",ylabel="y")
fig.colorbar(im,ax=axes,label="δ",shrink=.75)
save_figure(fig,"initial_density_comparison")
display(Markdown(f"Display grid: **{plot_N}³**, slab thickness **{(slab.stop-slab.start)*L/plot_N:g}**; "
                 f"field mesh: **{N}³**. Displacement strength **A={A:g}**."))
atomic_npz(OUT/"example_density_meshes.npz", density=np.stack(density), labels=np.array(labels),
           seed=np.array(meta["seed"]), helical_A=np.array(A), display_N=np.array(plot_N),
           field_N=np.array(N))'''),
        md(r'''## 3. Does the implementation pass its checks?

Check the ZA density relation, the two helicities, particle-mass conservation, and the parity transformation of \(\widehat\chi\).'''),
        code('''v = validation["field"]
checks = [dict(test="−div ΨZA = δ",value=v["za_density_max_error"],passed=v["za_density_max_error"]<1e-12),
          dict(test="curl ΨZA = 0",value=v["za_curl_max"],passed=v["za_curl_max"]<1e-12),
          dict(test="Positive helicity",value=v["plus"]["normalized_helicity"],passed=abs(v["plus"]["normalized_helicity"]-1)<1e-12),
          dict(test="Negative helicity",value=v["minus"]["normalized_helicity"],passed=abs(v["minus"]["normalized_helicity"]+1)<1e-12),
          dict(test="CIC mean density contrast",value=validation["cic_mean_delta"],passed=abs(validation["cic_mean_delta"])<1e-12),
          dict(test="χ̂ transformation error",value=max(r["absolute_error"] for r in validation["T1"].values()),
               passed=all(r["passed"] for r in validation["T1"].values()))]
audit_path = RUN/"validation/fixed_ic_ensemble.json"
if fixed_density and audit_path.exists():
    audit = json.loads(audit_path.read_text())
    checks.append(dict(test=f"Fixed density/lattice and ZA across {audit['displacement_seeds']} seeds",
                       value=audit["max_za_difference"], passed=audit["passed"]))
show_table(checks,[("test","Check"),("value","Measured"),("passed","Pass")])
save_table(checks,"construction_checks")
atomic_json(OUT/"construction_checks.json",dict(passed=all(c["passed"] for c in checks),checks=checks,
    source_run=RUN.name,model=field["model"]))'''),
        md(r'''## 4. Catalogue signal

A seed \(s\) labels one random realization. The **catalogue signal** \(S_s(\epsilon)\) is the mean normalized chirality over its \(N_{{\rm tet},s}\) tetrahedra, indexed by \(t\):
$$
S_s(\epsilon)=\frac{1}{N_{{\rm tet},s}}\sum_t\widehat\chi_{s,t}(\epsilon).
$$
**Each coloured dot is one catalogue.** The x-axis is its signal \(S_s\), and each row is a helicity setting \(\epsilon\). Matched seeds share the Gaussian density field and particle lattice across the three settings.

Black diamonds show the mean across seeds; horizontal bars show one **standard error of the mean (SEM)**, the sample standard deviation of the catalogue signals divided by the square root of the number of seeds. The \(\epsilon=0\) control has the same displacement strength \(A\), with equal expected power in the two helicities.'''),
        code(r'''scan = summary["primary_scan"]
paired = scan["rows"]
nulls = [r for r in summary["ensemble_statistics"] if r["case"].startswith("null_") and r["statistic"]=="chi_hat"]
from parity_study.signal_plots import epsilon_comparison, catalogue_signal_figure
comparison = epsilon_comparison(RUN)
if comparison is None:
    display(Markdown("No complete matched ε=−1,0,+1 catalogues saved at the control amplitude yet."))
else:
    display(Markdown(f"**Displacement strength A={comparison['A']:g}; "
                     f"{len(comparison['seeds'])} matched seeds per ε.** "
                     f"A={comparison['A']:g} means an ensemble RMS helical displacement of "
                     f"{comparison['A']:g} in the same length units as the box (L={field['L']:g})."))
    fig = catalogue_signal_figure(comparison)
    save_figure(fig, "catalogue_chi_hat_signal")
    plt.close(fig)
    save_table(comparison["catalogue_rows"],"epsilon_chi_hat_catalogues")
    save_table(comparison["summary_rows"],"epsilon_chi_hat_summary")
    atomic_npz(OUT/"epsilon_chi_hat.npz", **{
        key: comparison[key] for key in ("A","epsilons","seeds","signals","tetrahedron_counts",
                                         "histogram_counts","histogram_edges","histogram_density")})
save_table(paired,"paired_chi_hat")
save_table(nulls,"null_chi_hat")'''),
        md(r'''## 5. Histogram difference: \(\epsilon=+1\) minus \(\epsilon=-1\)

At the same \(A\), subtract the negative-helicity histogram from the positive-helicity histogram **in each bin**. Here \(s\) labels a matched seed, \(N_s\) is the number of matched seeds, and \(p_{s,\epsilon}\) is one catalogue's probability density:
$$
\Delta p_s(\widehat\chi)=p_{s,+1}(\widehat\chi)-p_{s,-1}(\widehat\chi),\qquad
\Delta p(\widehat\chi)=\frac{1}{N_s}\sum_s\Delta p_s(\widehat\chi).
$$
The plotted \(\Delta p\) is the average difference. **Above zero:** more probability density for \(\epsilon=+1\); **below zero:** more for \(\epsilon=-1\); **zero:** the two agree in that bin.

**SEM = standard error of the mean**, the estimated uncertainty in this average difference:
$$
\mathrm{SEM}(\widehat\chi)=
\frac{\text{sample standard deviation of }\Delta p_s(\widehat\chi)}{\sqrt{N_s}}.
$$
Each point has error bars \(\Delta p\pm1\,\mathrm{SEM}\). It measures uncertainty across the displacement realizations used here; a 1-SEM bar is not a 95% confidence interval. Bins are correlated, so individual excursions alone do not establish a PV detection.'''),
        code(r'''from parity_study.signal_plots import histogram_difference, histogram_difference_figure
difference = None
if comparison is not None:
    difference = histogram_difference(comparison)
    fig = histogram_difference_figure(difference)
    save_figure(fig, "chi_hat_probability_difference")
    plt.close(fig)
    save_table(difference["rows"], "chi_hat_probability_difference")
    atomic_npz(OUT/"chi_hat_probability_difference.npz", **{
        key: difference[key] for key in ("A","seeds","histogram_edges","bin_centers",
                                         "per_seed_difference","mean","sem","mean_covariance")
        if difference[key] is not None})
atomic_json(OUT/"analysis.json",dict(source_run=RUN.name,complete=summary["complete"],
    purpose=config.get("purpose"),primary_statistic="chi_hat",primary_scan=scan,
    density_seed=field.get("density_seed"), ensemble=summary["ensemble"],
    direct_epsilon_comparison=(dict(A=comparison["A"], seeds=comparison["seeds"].tolist(),
        rows=comparison["summary_rows"], histogram_weighting="equal catalogue weights")
        if comparison is not None else None),
    histogram_difference=(dict(definition="p(epsilon=+1) - p(epsilon=-1)",
        A=difference["A"], seeds=difference["seeds"].tolist(),
        uncertainty="Pointwise SEM of matched-seed differences; bins are correlated")
        if difference is not None else None)))'''),
        md(f'[Figures](saved/{topic}/figures/) · [Tables and saved meshes](saved/{topic}/) · [Run configuration](saved/{run.name}/config.json)')]
    if filename == 'field_PVdisp.ipynb':
        from parity_study.paths import OUTPUT_ROOT, selection_output
        import json
        protocol_path = selection_output(OUTPUT_ROOT/topic)/'protocol.json'
        if protocol_path.exists():
            from parity_study.tests.selection_notebooks import selection_cells
            cells[-1:-1] = selection_cells(json.loads(protocol_path.read_text()))
        from parity_study.paths import independent_ic_output
        ic_out = independent_ic_output(OUTPUT_ROOT/topic)
        if ic_out is not None:
            from parity_study.tests.independent_ic_notebooks import independent_ic_cells
            cells[-1:-1] = independent_ic_cells(json.loads((ic_out/'protocol.json').read_text()))
    out = notebook(cells, kind='run', run_name=run.name)
    out.metadata['parity_results'].update(topic=topic, format_version=9, primary_statistic='chi_hat',
                                         model='mesh_za_helical', report_generator_sha256=digest(Path(__file__)),
                                         signal_plots_sha256=digest(Path(__file__).with_name("signal_plots.py")))
    for cell in out.cells:
        if cell.cell_type == 'code':
            cell.metadata['jupyter'] = {'source_hidden': True}
    return out
