"""Question-led notebooks for the PV foundations and displacement model."""
from parity_study.tests.notebooks import md as _markdown, code, notebook, TABLE_HELPER


def md(source):
    # Dollar delimiters survive Jupyter/nbconvert Markdown parsing; backslash
    # delimiters can otherwise be consumed as ordinary Markdown escapes.
    return _markdown(source.replace(r"\[", "$$").replace(r"\]", "$$")
                     .replace(r"\(", "$").replace(r"\)", "$"))


def setup(results, topic, runs):
    return code(f'''
from pathlib import Path
import contextlib
import csv
import html
import io
import itertools
import json
import sys
import numpy as np
from IPython.display import display, Markdown, HTML
PROJECT = next((p for p in (Path.cwd(), *Path.cwd().parents)
                if (p/"src/parity_study").is_dir() and (p/"results").is_dir()),
               Path({str(results.parent)!r}))
sys.path.insert(0, str(PROJECT/"src"))
from parity_study.paths import OUTPUT_ROOT
RESULTS = OUTPUT_ROOT
OUT = RESULTS/{topic!r}
(OUT/"figures").mkdir(parents=True, exist_ok=True)
from parity_study.summarize import summarize
from parity_study.storage import atomic_json
from parity_study.catalogue import TEMPLATE, chi_hat_ref
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams.update({{"figure.dpi":110, "axes.grid":True, "grid.alpha":0.2, "font.size":11}})
summaries = {{}}
for name in {runs!r}:
    with contextlib.redirect_stdout(io.StringIO()):
        summaries[name] = summarize(RESULTS/name)
    s = summaries[name]
    display(Markdown(f"Saved ensemble **{{name}}**: {{s['completed_records']:,}} / "
                     f"{{s['expected_records']:,}} measurements; "
                     f"**{{'complete' if s['complete'] else 'partial — conclusions provisional'}}**."))

def save_figure(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(OUT/"figures"/f"{{name}}.{{suffix}}", dpi=170, bbox_inches="tight")
    plt.show()

def save_table(rows, name):
    if rows:
        with (OUT/f"{{name}}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

def family_p(rows):
    p = [r["p_two_sided"] for r in rows if r.get("p_two_sided") is not None]
    return min(1., len(p)*min(p)) if p else None

{TABLE_HELPER}
''')


def basic_notebook(results):
    cells = [
        md(r'''# Basic PV tests

Does our chirality statistic transform correctly, detect known chirality, and give zero for symmetric ensembles?'''),
        setup(results, 'basic', ['20261121_baseline']),
        code('''
summary = summaries["20261121_baseline"]
RUN = RESULTS/"20261121_baseline"
validation = json.loads((RUN/"validation/report.json").read_text())
positive = [r for r in summary["ensemble_statistics"]
            if r["case"].startswith("positive_") and r["statistic"]=="chi_hat"]
null = [r for r in summary["ensemble_statistics"] if r["case"].startswith("null_")]
geometry_ok = all(v["passed"] for v in validation["T1"].values())
right = next(r for r in positive if r["case"]=="positive_f1")
left = next(r for r in positive if r["case"]=="positive_f0")
controls_ok = right["ci95_low"]>0 and left["ci95_high"]<0
p_null = family_p(null)
display(Markdown(
    f"**Geometry:** {'passes' if geometry_ok else 'check failures'}. "
    f"**Known chirality:** {'recovered' if controls_ok else 'check controls'}. "
    f"**Nulls:** {'no detected bias' if p_null>=.05 else 'inspect deviations'} "
    f"(adjusted p={p_null:.3f})."))
'''),
        md(r'''## 1. What is parity violation?

A parity-odd statistic changes sign under reflection:
$$
S(\mathcal P C)=-S(C).
$$
A parity-symmetric ensemble has \(\mathbb E[S]=0\). A nonzero value in one catalogue can arise from sampling noise.'''),
        md(r'''## 2. How do we measure chirality?

Order the four vertices by \(s_i=\sum_{j\ne i}|\mathbf r_i-\mathbf r_j|^2\). From the first ordered vertex, form the three edge vectors \(\mathbf v_1,\mathbf v_2,\mathbf v_3\):
$$
\chi=\det(\mathbf v_1,\mathbf v_2,\mathbf v_3),\qquad
\widehat\chi=\frac{\chi}{|\mathbf v_1||\mathbf v_2||\mathbf v_3|}\in[-1,1].
$$
Average over the catalogue's tetrahedra. This ordering makes the sign independent of input vertex labels; tied scores receive zero weight.'''),
        code(r'''
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
shape = TEMPLATE-TEMPLATE.mean(axis=0)
rot = Rotation.from_euler("xyz",[25,35,50],degrees=True).as_matrix()
shapes = [shape, -shape, shape@rot.T]
labels = ["Original", "Parity inversion", "Proper rotation"]
fig = plt.figure(figsize=(12,4))
values = []
for index, (points,label) in enumerate(zip(shapes,labels),1):
    score = ((points[:,None]-points[None])**2).sum(axis=-1).sum(axis=1)
    ordered = points[np.argsort(score)]
    value = float(chi_hat_ref(points))
    values.append(dict(transformation=label, chi_hat=value))
    ax = fig.add_subplot(1,3,index,projection="3d")
    faces = [ordered[list(face)] for face in itertools.combinations(range(4),3)]
    ax.add_collection3d(Poly3DCollection(faces,alpha=.16,facecolor="tab:blue",edgecolor="k"))
    ax.scatter(*ordered.T,s=35,color="tab:blue")
    for j,point in enumerate(ordered,1):
        ax.text(*point,str(j),fontsize=11)
    ax.set(xlim=(-2,2),ylim=(-2,2),zlim=(-2,2),xlabel="x",ylabel="y",zlabel="z",
           title=label+rf": $\widehat\chi={value:+.2f}$")
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=22,azim=35)
fig.suptitle("The same unequal-edge tetrahedron; labels follow the distance ordering")
save_figure(fig,"tetrahedron_handedness")
permuted = np.array([chi_hat_ref(shape[list(p)]) for p in itertools.permutations(range(4))])
relabel_error = float(np.max(abs(permuted-chi_hat_ref(shape))))
show_table(values,[("transformation","Transformation"),("chi_hat","Normalized chirality")])
display(Markdown(f"**Answer:** reflection flips the sign; rotation preserves it. "
                 f"All 24 input vertex permutations agree, with maximum difference {relabel_error:.3g}."))
save_table(values,"tetrahedron_example")
'''),
        md(r'''## 3. Does the full pipeline transform correctly?

Reference: 10,000 points, \(L=1000\).
**Expected:** reflections flip the sign; rotations and translations preserve it. Cell counts and total volume stay unchanged.'''),
        code(r'''signs = {"reference":1,"inversion":-1,"swap":-1,"rotation90":1,"translation":1}
reference = validation["T1"]["reference"]["sum_chi_hat"]
geometry = [dict(transformation=name, expected_sum=signs[name]*reference, **v)
            for name,v in validation["T1"].items()]
show_table(geometry,[("transformation","Transformation"),("expected_sum","Expected sum χ̂"),
                    ("sum_chi_hat","Measured sum χ̂"),("absolute_error","Absolute error"),
                    ("n_tetrahedra","Tetrahedra"),("volume_fraction","Volume / box"),
                    ("passed","Passed")])
geometry_pass = all(r["passed"] for r in geometry)
max_error = max(r["absolute_error"] for r in geometry)
display(Markdown(f"**Answer:** {'all transformations pass' if geometry_pass else 'inspect failed checks'}. "
                 f"The largest chirality-sum error is {max_error:.3g}. "
                 ""))
save_table(geometry,"geometry_checks")'''),
        md(r'''## 4. Can we recover known chirality?

2,500 randomly oriented molecules per catalogue; 20 seeds per right-handed fraction \(f_R\).
**Expected:** molecule mean \(\widehat\chi\simeq2f_R-1\); the full-catalogue signal is diluted. Error bars: 1 SEM across seeds.'''),
        code(r'''fractions = [float(r["case"].split("_f")[1]) for r in positive]
molecules = summary["molecular_controls"]
def molecule_rows(quantity):
    return sorted([r for r in molecules if r["quantity"]==quantity],
                  key=lambda r:float(r["case"].split("_f")[1]))
fig,axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
identified = molecule_rows("measured_mean")
axes[0].plot([0,1],[-1,1],"k--",label="Expected: 2 f_R − 1")
axes[0].errorbar(fractions,[r["mean"] for r in identified],
                 yerr=[r["sem"] for r in identified],fmt="o",capsize=3,label="Identified molecules")
axes[1].errorbar(fractions,[r["mean"] for r in positive],yerr=[r["sem"] for r in positive],
                 fmt="o-",capsize=3,label="All Delaunay cells")
contribution = molecule_rows("molecule_contribution_to_all")
axes[1].plot(fractions,[r["mean"] for r in contribution],"s--",label="Molecule contribution / all cells")
for ax in axes:
    ax.axhline(0,color="k",lw=.7)
    ax.set(xlabel="Right-handed fraction f_R",ylabel="Mean normalized chirality")
    ax.legend(fontsize=9)
axes[0].set_title("Recovery of known chiral objects")
axes[1].set_title("Dilution in the full catalogue")
save_figure(fig,"positive_controls")
show_table(positive,[("case","Case"),("n","Catalogues"),("mean","Mean χ̂"),("sem","SEM"),
                     ("ci95_low","95% interval low"),("ci95_high","95% interval high")])
left = next(r for r in positive if r["case"]=="positive_f0")
right = next(r for r in positive if r["case"]=="positive_f1")
balanced = next(r for r in positive if r["case"]=="positive_f0.5")
recovered = left["ci95_high"]<0 and right["ci95_low"]>0
display(Markdown(f"**Answer:** {'both injected signs are recovered' if recovered else 'the controls need inspection'}. "
                 f"The balanced mixture is {balanced['t']:+.2f} SEM from zero. "
                 ""))
save_table(positive,"positive_controls")
save_table(molecules,"molecule_diagnostics")'''),
        md(r'''## 5. Are symmetric ensembles consistent with zero?

200 seeds per null, using the original amplitude convention:

| Null | Parameters |
| --- | --- |
| Poisson | Undisplaced, L=1000 |
| Equal helicity | ε=0, A=0.002, L=1000 |
| Strong displacement | ε=0, A=1, L=1 |

Plot: mean \(\widehat\chi\) and 95% intervals across seeds. The family p-value accounts for all nine null comparisons.'''),
        code(r'''selected = [r for r in null if r["statistic"]=="chi_hat"]
case_labels = {"null_poisson":"Poisson","null_symmetric":"Equal helicity",
               "null_notebook_L1_A1":"Original strong null"}
fig,ax = plt.subplots(figsize=(8,3.5),constrained_layout=True)
means = np.array([r["mean"] for r in selected])
low = np.array([r["ci95_low"] for r in selected])
high = np.array([r["ci95_high"] for r in selected])
ax.errorbar(means,np.arange(len(selected)),xerr=[means-low,high-means],fmt="o",capsize=4)
ax.axvline(0,color="k",lw=1)
ax.set(yticks=np.arange(len(selected)),yticklabels=[case_labels[r["case"]] for r in selected],
       xlabel="Ensemble mean normalized chirality (95% intervals)",
       title="Do the uncentered null means differ from zero?")
save_figure(fig,"null_means")
show_table(null,[("case","Null"),("statistic","Statistic"),("n","Catalogues"),
                 ("mean","Mean"),("sem","SEM"),("t","Mean / SEM"),("p_two_sided","Uncorrected p")])
p_null = family_p(null)
display(Markdown(f"**Answer:** {'no null bias is detected' if p_null>=.05 else 'inspect the null deviations'}. "
                 f"The conservative family p-value is {p_null:.3g}. "
                 ""))
save_table(null,"null_controls")'''),
        
        code(r'''atomic_json(OUT/"conclusions.json",dict(geometry_pass=geometry_pass,
    geometry_max_error=max_error,vertex_relabelling_max_error=relabel_error,
    positive_controls_recovered=recovered,null_family_p=p_null,
    source_run=RUN.name,source_complete=summary["complete"]))'''),
        md(r'''[Figures](saved/basic/figures/) · [Tables](saved/basic/) · [Displacement model](intro_PVdisp.ipynb)''')]
    out = notebook(cells,kind='run',run_name='20261121_baseline')
    out.metadata['parity_results'].update(topic='basic',format_version=4)
    for cell in out.cells:
        if cell.cell_type == 'code':
            cell.metadata['jupyter'] = {'source_hidden': True}
    return out


def displacement_notebook(results):
    from parity_study.tests.intro_fiducial import active_example
    active = active_example()
    if active is not None:
        import importlib.util
        import json
        from parity_study.paths import OUTPUT_ROOT
        from pathlib import Path
        record = json.loads((OUTPUT_ROOT/'intro_disp_PV/active_example.json').read_text())
        spec = importlib.util.spec_from_file_location('active_fiducial_intro',
            Path(record.get('report_source',active/'code/src/parity_study/tests/intro_fiducial.py')))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.notebook_template(active)
    cells = [
        md(r'''# Helical displacement field

**A controls the ensemble RMS displacement; ε controls helicity.** The examples use normalized power.'''),
        setup(results, 'intro_disp_PV', ['20261121_baseline','20261121_notebook_current']),
        code(r'''scans = [s["primary_scan"] for s in summaries.values()]
complete = all(s["complete"] for s in summaries.values())
no_detection = all(s["bonferroni_p"] is not None and s["bonferroni_p"] >= .05 for s in scans)
display(Markdown("**Current catalogue result" + ("" if complete else " (provisional)") + ":** " +
    (r"no detectable helicity contrast in mean $\widehat\chi$ at the scan-wide 5% threshold (section 7). "
     "A weaker catalogue signal remains possible."
     if no_detection else "inspect the helicity contrasts in section 7.")))'''),
        code(r'''example_path = OUT/"examples.npz"
if not example_path.exists():
    raise FileNotFoundError("Prepare cached examples with python -m parity_study.tutorial_examples.")
examples = json.loads((OUT/"examples.json").read_text())
with np.load(example_path) as z:
    q, plus, minus = z["q"], z["plus"], z["minus"]
    plane, slices, epsilons = z["plane"], z["slices"], z["epsilons"]
params = examples["parameters"]
L = params["L"]

display(Markdown(f"**Example:** N={params['N']}, L={L:g}, {len(q):,} points; "
                 f"{examples['field']['n_modes']} modes, wavelengths L/4 to L."))'''),
        md(r'''## 1. Displacing the points

Starting from uniform random positions:
$$
\mathbf x=\left[\mathbf q+R\boldsymbol\psi_\epsilon(\mathbf q)\right]\bmod L,\qquad
\boldsymbol\psi_\epsilon
=A\left[\sqrt{\frac{1+\epsilon}{2}}\mathbf u_+
+\sqrt{\frac{1-\epsilon}{2}}\mathbf u_-\right].
$$
\(\mathbf u_\pm\) are independent unit-power fields of opposite helicity. ε=+1/−1 selects one sign; ε=0 mixes equal power. Here \(R=1\).'''),
        md(r'''## 2. Building the field

Choose a transverse orthonormal frame \(\mathbf e_1,\mathbf e_2\), with \(\mathbf e_2=\widehat{\mathbf k}\times\mathbf e_1\):
$$
\mathbf e_\pm=\frac{\mathbf e_1\pm i\mathbf e_2}{\sqrt2},
\qquad i\mathbf k\times\mathbf e_\pm=\pm|\mathbf k|\mathbf e_\pm.
$$
Then
$$
\mathbf u_\pm(\mathbf q)=\mathrm{Re}\sum_{\mathbf k}
z_{\pm,\mathbf k}\sqrt{W_{\mathbf k}}\,
\mathbf e_\pm(\mathbf k)e^{i\mathbf k\cdot\mathbf q},
\qquad \sum_{\mathbf k}W_{\mathbf k}=1.
$$
The complex Gaussian draws have unit variance in each real/imaginary component. Our top hat uses \(W=1/N_{\rm modes}\) on active modes, excluding zero and unpaired Nyquist modes.'''),
        md(r'''### Opposite helicities

For a wave along z:
$$
\boldsymbol\psi_\pm=(\cos kz,\mp\sin kz,0),\qquad
\nabla\times\boldsymbol\psi_\pm=\pm k\boldsymbol\psi_\pm.
$$
The curves connect field-vector tips; the arrows rotate in opposite senses.'''),
        code(r'''
zline = np.linspace(0,1,25)
fig = plt.figure(figsize=(10,4))
for index,hand in enumerate((1,-1),1):
    vx = np.cos(2*np.pi*zline)
    vy = -hand*np.sin(2*np.pi*zline)
    ax = fig.add_subplot(1,2,index,projection="3d")
    ax.quiver(np.zeros_like(zline),np.zeros_like(zline),zline,
              vx,vy,np.zeros_like(zline),length=1,color="tab:blue",alpha=.6,
              arrow_length_ratio=.12)
    ax.plot(vx,vy,zline,color="tab:orange",lw=2)
    ax.set(xlabel="ψx / A",ylabel="ψy / A",
           title=f"Helicity {hand:+d}: curl ψ = {hand:+d} k ψ",
           xlim=(-1.2,1.2),ylim=(-1.2,1.2),zlim=(0,1))
    ax.set_box_aspect((1,1,1.3))
    ax.view_init(elev=22,azim=30)
    ax.text2D(-.16,.55,"z / L",transform=ax.transAxes,rotation=90,va="center",clip_on=False)
save_figure(fig,"single_mode_helicity")
'''),
        md(r'''## 3. Random-field examples

Same underlying fields, \(z=L/2\), A=4. Arrows: \((\psi_x,\psi_y)\); color: \(\psi_z\). All panels share the same scales.'''),
        code(r'''
A_slice = 4.
vectors = A_slice*slices
limit = float(abs(vectors[:,:,2]).max())
fig,axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for ax,eps,vec in zip(axes,epsilons,vectors):
    grid_axis = np.unique(plane[:,0])
    artist = ax.pcolormesh(grid_axis,grid_axis,vec[:,2].reshape(len(grid_axis),len(grid_axis)),
                          shading="nearest",cmap="coolwarm",vmin=-limit,vmax=limit,alpha=.85)
    ax.quiver(plane[:,0],plane[:,1],vec[:,0],vec[:,1],
              angles="xy",scale_units="xy",scale=1,color=".15",width=.003)
    ax.set(xlim=(0,L),ylim=(0,L),aspect="equal",xlabel="x",ylabel="y",title=f"ε = {eps:+g}, A = {A_slice:g}")
fig.colorbar(artist,ax=axes,label="Out-of-plane displacement ψz",shrink=.85)
save_figure(fig,"random_field_slices")
'''),
        md(r'''## 4. A is RMS displacement

For \(R=1\):
$$
P_{\psi,\mathbf k}=A^2W_{\mathbf k},\qquad
\sqrt{\mathbb E[\langle|\boldsymbol\psi|^2\rangle]}=A.
$$
A is an RMS length, not the mean magnitude. The vector mean is zero; each realization's measured RMS fluctuates.

**Left:** vary A in one realization. **Right:** 24 seeds per grid size; expected squared RMS/A²=1. Error bars: 1 SEM.'''),
        code(r'''amplitudes = np.array([0.,1.,2.,4.,8.])
fig,axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
amplitude_rows = []
for eps in epsilons:
    u = np.sqrt((1+eps)/2)*plus + np.sqrt((1-eps)/2)*minus
    rms = np.sqrt(np.mean(np.sum(u*u,axis=1)))
    axes[0].plot(amplitudes,amplitudes*rms,"o-",label=f"ε={eps:+g}")
    amplitude_rows += [dict(A=float(A),eps=float(eps),expected_rms=float(A),measured_rms=float(A*rms)) for A in amplitudes]
axes[0].plot(amplitudes,amplitudes,"k--",label="Expected RMS = A")
axes[0].set(xlabel="A (length units)",ylabel="Measured displacement RMS",title="One finite realization")
normalization = examples["normalization"]
normal_rows = []
for eps,offset in zip(epsilons,(-.015,0,.015)):
    counts,means,sems = [],[],[]
    for grid in (8,16,24):
        group = [r for r in normalization if r["eps"]==eps and r["N"]==grid]
        samples = np.array([r["mean_square"] for r in group])
        n_modes = group[0]["n_modes"]
        mean,sem = float(samples.mean()),float(samples.std(ddof=1)/np.sqrt(len(samples)))
        counts.append(n_modes)
        means.append(mean)
        sems.append(sem)
        normal_rows.append(dict(N=grid,n_modes=n_modes,eps=float(eps),seeds=len(samples),mean_square_over_A2=mean,sem=sem))
    axes[1].errorbar(np.array(counts)*np.exp(offset),means,yerr=sems,fmt="o",capsize=3,label=f"ε={eps:+g}")
axes[1].axhline(1,color="k",ls="--",label="Expected = 1")
axes[1].set(xscale="log",xlabel="Active Fourier modes (points slightly offset)",
            ylabel="Ensemble mean squared displacement / A²",title="Normalization across mode counts")
for ax in axes:
    ax.legend(fontsize=9)
save_figure(fig,"amplitude_normalization")
save_table(amplitude_rows,"amplitude_scaling")
save_table(normal_rows,"normalization_check")

ratios = [r["mean_square_over_A2"] for r in normal_rows]
display(Markdown(f"**Result:** mean squared RMS/A² spans {min(ratios):.3f}–{max(ratios):.3f} "
                 "across the tested mode counts and helicities (expected: 1)."))'''),
        md(r'''## 5. Moving the points

Track the same initial slab, \(0.45L<q_z<0.55L\), at ε=+1. Gray: initial positions; blue: wrapped final positions. Arrows omit projected boundary crossings.

Displacement strength relative to spacing: \(A/\ell\), where \(\ell=L/N_{\rm gal}^{1/3}\).'''),
        code(r'''
slab = abs(q[:,2]-L/2)<.05*L
initial = q[slab]
unit = plus[slab]
ell = L/len(q)**(1/3)
fig,axes = plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
motion_rows = []
for ax,A in zip(axes,(0.,2.,6.)):
    unwrapped = initial+A*unit
    final = unwrapped % L
    inside = ((unwrapped[:,:2]>=0)&(unwrapped[:,:2]<L)).all(axis=1)
    ax.scatter(initial[:,0],initial[:,1],s=12,color=".7",label="Initial")
    if A:
        ax.quiver(initial[inside,0],initial[inside,1],A*unit[inside,0],A*unit[inside,1],
                  angles="xy",scale_units="xy",scale=1,color="tab:orange",alpha=.6,width=.003)
    ax.scatter(final[:,0],final[:,1],s=12,color="tab:blue",label="Displaced")
    ax.set(xlim=(0,L),ylim=(0,L),aspect="equal",xlabel="x",ylabel="y",
           title=f"A={A:g}; expected RMS / spacing={A/ell:.2f}")
    motion_rows.append(dict(A=A,expected_rms_over_spacing=A/ell,tracked_points=len(initial),
                            projected_boundary_crossings=int((~inside).sum())))
axes[0].legend(fontsize=9)
save_figure(fig,"point_displacements")
save_table(motion_rows,"particle_examples")
'''),
        md(r'''## 6. Numerical checks

**Expected:** direct Fourier and NUFFT evaluations agree; extreme helicities give h=±1; longitudinal power vanishes. The mixed realization can fluctuate around h=0.'''),
        code(r'''
diagnostics = examples["diagnostics"]
show_table(diagnostics,[("eps","ε"),("spectral_rms","Unit-field spectral RMS"),
                        ("normalized_helicity","Measured h"),("longitudinal_fraction","Longitudinal fraction")])
error = max(examples["nufft_direct_max_errors"])
extremes_ok = all(abs(r["normalized_helicity"]-r["eps"])<1e-12
                  for r in diagnostics if abs(r["eps"])==1)
transverse_ok = all(r["longitudinal_fraction"]<1e-28 for r in diagnostics)
display(Markdown(f"**Measured maximum NUFFT/direct difference:** {error:.3g} (required <10⁻⁷).  \n"
                 f"**Extreme helicities correct:** {extremes_ok}. **Transverse:** {transverse_ok}."))
save_table(diagnostics,"field_diagnostics")
atomic_json(OUT/"checks.json",dict(nufft_direct_max_error=error,extreme_helicities_pass=extremes_ok,
                                  transverse_pass=transverse_ok,amplitude_convention="rms"))
'''),
        
        md(r'''## 7. Does mean \(\widehat\chi\) detect the injected helicity?

Use the normalized tetrahedron chirality from [basic.ipynb](basic.ipynb):
$$
S_s=\frac{1}{N_{{\rm tet},s}}\sum_t\widehat\chi_{s,t},\qquad
D_s=S_s(+1)-S_s(-1).
$$
Saved scans: **100 matched seed pairs per A** (100 catalogues per helicity), \(L=1000\), 10,000 points. Errors are 1 SEM across pairs. The p-value correction covers only \(\widehat\chi\), across all amplitudes within each scan. This focus is adopted for follow-up tests; these existing scans are exploratory.

**These scans use the original amplitude convention:** \(A_{\rm rms}=\sqrt{A_{\rm old}N_{\rm modes}}\). The panels have different grids and Nyquist treatment. The normalized examples above are field illustrations, not new catalogue PV measurements.
'''),
        code(r'''fig,axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
scan_summary = []
scan_measurements = []
for ax,(name,s) in zip(axes,summaries.items()):
    run = RESULTS/name
    config = json.loads((run/"config.json").read_text())
    with (run/"summary/realizations.csv").open() as stream:
        realizations = list(csv.DictReader(stream))
    primary = s["primary_scan"]
    selected = primary["rows"]
    strengths = []
    for r in selected:
        cases = {f"scan_A{r['A']:g}_plus",f"scan_A{r['A']:g}_minus"}
        strengths.append(float(np.mean([float(v["rms_over_spacing"]) for v in realizations if v["case"] in cases])))
    ax.errorbar(strengths,[r["mean"] for r in selected],yerr=[r["sem"] for r in selected],fmt="o-",capsize=3)
    ax.axhline(0,color="k",lw=.7)
    ax.set(xscale="log",xlabel="Measured unwrapped RMS / mean spacing",
           ylabel=r"$\overline{D} = \langle\widehat\chi\rangle_+ - \langle\widehat\chi\rangle_-$",
           title=f"N={config['field']['N']}; original amplitude convention")
    ticks = np.geomspace(min(strengths),max(strengths),4)
    ax.set_xticks(ticks,labels=[f"{v:.2g}" for v in ticks])
    ax.minorticks_off()
    peak = primary["peak"]
    scan_summary.append(dict(run=name,N=config["field"]["N"],convention="original per-mode power",
        complete=s["complete"],statistic=primary["statistic"],pairs_min=min(r["n"] for r in selected),
        pairs_max=max(r["n"] for r in selected),comparisons=primary["comparisons"],
        max_abs_t=abs(peak["t"]),peak_A=peak["A"],peak_p_local=peak["p_two_sided"],
        bonferroni_p=primary["bonferroni_p"]))
    scan_measurements.extend(dict(run=name,N=config["field"]["N"],rms_over_spacing=strength,**r)
                             for strength,r in zip(strengths,selected))
save_figure(fig,"saved_helicity_scans")

show_table(scan_summary,[("N","Grid N"),("pairs_min","Pairs / A"),("comparisons","Amplitudes"),
                         ("peak_A","Peak A (old)"),("max_abs_t","Largest |Δ/SEM|"),
                         ("peak_p_local","Local p"),("bonferroni_p","Scan-adjusted p")])
sems = [r["sem"] for r in scan_measurements]
display(Markdown(f"**Sensitivity:** SEM on the mean χ̂ contrast ranges from {min(sems):.2g} to {max(sems):.2g}. " +
                 ("The largest local fluctuation does not pass the amplitude-scan correction."
                  if no_detection else "Inspect the candidate contrast with independent realizations.")))
save_table(scan_summary,"saved_scan_conclusions")
save_table(scan_measurements,"saved_chi_hat_scan")
atomic_json(OUT/"saved_scan_conclusions.json",dict(primary_statistic="chi_hat",
    complete=complete,no_detection_at_scan_wide_5_percent=no_detection,
    correction="Bonferroni over amplitudes within each scan",scans=scan_summary))'''),
        
        md(r'''[Figures](saved/intro_disp_PV/figures/) · [Data and tables](saved/intro_disp_PV/) · [Basic PV tests](basic.ipynb)''')]
    out = notebook(cells,kind='run',run_name='20261121_notebook_current')
    out.metadata['parity_results'].update(topic='intro_disp_PV',format_version=5,primary_statistic='chi_hat',
                                         source_runs=['20261121_baseline','20261121_notebook_current'])
    for cell in out.cells:
        if cell.cell_type == 'code':
            cell.metadata['jupyter'] = {'source_hidden': True}
    return out
