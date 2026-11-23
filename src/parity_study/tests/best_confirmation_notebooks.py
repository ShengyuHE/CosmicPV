"""Concise catalogue-signal distribution report for fresh confirmation ICs."""
import json
from pathlib import Path

from parity_study.tests.notebooks import TABLE_HELPER, code, md, notebook


def confirmation_notebook(run):
    run = Path(run).resolve()
    config = json.loads((run/'config.json').read_text())
    out = Path(config['confirmation_protocol']['path']).parent
    p = json.loads((out/'protocol.json').read_text())
    n = len(p['seeds'])
    cells = [md(fr'''# Does the best tested configuration reproduce the PV signal?

**{n} fresh independent Gaussian initial conditions**, seeds {p['seeds'][0]}–{p['seeds'][-1]}, with displacement strength **$A={p['A']:g}\ h^{{-1}}\mathrm{{Mpc}}$**. The box is $L=1000\ h^{{-1}}\mathrm{{Mpc}}$, the field mesh is $256^3$, and the particle lattice is $32^3$.

The amplitude, toy spectrum and tetrahedron selection are fixed from the earlier studies. Each realization has three matched catalogues: $\epsilon=-1$ (negative helicity), $0$ (equal expected helical powers), and $+1$ (positive helicity). Both the Gaussian density and helical fields vary between realizations; the three cases share those draws within a realization. The earlier amplitude-scan ICs are excluded.
'''), code(f'''
from pathlib import Path
import html,json,sys
import numpy as np
from IPython.display import display,Markdown,HTML
RUN=Path({str(run)!r})
OUT=Path({str(out)!r})
sys.path.insert(0,str(RUN/"code/src"))
from parity_study.storage import digest
from parity_study.best_confirmation import distribution_figure
protocol=json.loads((OUT/"protocol.json").read_text())
report=json.loads((OUT/"analysis.json").read_text())
assert report["complete"] and report["protocol_sha256"]==digest(OUT/"protocol.json")
assert report["independent_ic_audit"]["passed"]
with np.load(OUT/"confirmation.npz") as z:
    signals=z["signals"]
    paired=z["paired"]
    seeds=z["seeds"]
assert seeds.tolist()==protocol["seeds"]
primary_index=next(i for i,c in enumerate(protocol["cuts"]) if c["id"]==protocol["primary_selection"])
display(Markdown(f"**Complete:** {{len(seeds)}} ICs × 3 helicities = **{{3*len(seeds)}} catalogue signals**, "
                 f"plus {{len(seeds)}} ZA-only controls. Distinct fields and exact matched particle maps verified."))
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams.update({{"figure.dpi":110,"font.size":11}})
{TABLE_HELPER}
'''), md(r'''## 1. What is the catalogue signal?

Order the tetrahedron vertices by their sums of squared distances to the other vertices. From the first vertex, take the three edge vectors in that order:
$$\widehat\chi_t=\frac{\mathbf v_1\cdot(\mathbf v_2\times\mathbf v_3)}{|\mathbf v_1||\mathbf v_2||\mathbf v_3|}.$$
Keep both chirality signs with the fixed shape selection
$$B:\quad 0.6905111410000623\leq\frac{\ell_{\min}}{\ell_{\max}}<0.7567675934849439,$$
where the lengths are those of the **three edges from the reference vertex**.

For realization $s$, the **catalogue signal** is $S_{s,B}(\epsilon)=\mathrm{mean}_{t\in B}\widehat\chi_t$. Each histogram below contains 50 catalogue signals, one per IC; its horizontal axis is $S_{s,B}$.
'''), md(r'''## 2. Catalogue-signal distributions: three helicities

Shaded histograms show measured probability densities, each integrating to one. The curves are Gaussian approximations $\mathcal N(\mu,\sigma_S^2)$ fitted using the **sample mean** $\mu$ and **sample standard deviation** $\sigma_S$ across catalogues. Gaussianity is not assumed to be exact.

The curve width $\sigma_S$ describes catalogue-to-catalogue scatter. The **standard error of the mean**, $\mathrm{SEM}=\sigma_S/\sqrt{50}$, describes uncertainty in the ensemble mean and is smaller than the curve width.
'''), code(r'''
fig=distribution_figure(report,signals,primary_index)
(OUT/"figures").mkdir(parents=True,exist_ok=True)
for ext in ("png","pdf"):
    fig.savefig(OUT/"figures"/f"catalogue_signal_distributions.{ext}",dpi=180,bbox_inches="tight")
plt.show()
plt.close(fig)
rows=[r for r in report["distributions"] if r["selection"]==report["primary_selection"]]
show_table(rows,[("epsilon","ε"),("n","Catalogues"),("mean","Mean μ"),
                 ("sd","Scatter σS"),("sem","SEM")])
'''), md(r'''## 3. Is the helicity difference reproduced?

For each matched realization, define $D_{s,B}=S_{s,B}(+1)-S_{s,B}(-1)$. Its mean $\overline D_B$ measures the helicity separation. Negative values mean that positive-helicity catalogues have the lower chirality signal.

The primary test is a two-sided paired Student t test at $p<0.05$, requiring the negative sign found previously. Its SEM uses the scatter of the **50 matched differences**, including their correlation. A, the cut, and the sample size were frozen before generation.
'''), code(r'''
p=report["primary"]
show_table([p],[("n","Matched ICs"),("mean","Mean D̄B"),("sem","SEM"),
                ("ci95_low","95% lower"),("ci95_high","95% upper"),("p_two_sided","Two-sided p")])
if report["primary_confirmed"]:
    display(Markdown(f"**The catalogue PV response is reproduced on 50 fresh ICs:** "
        f"$\\overline D_B={p['mean']:.6f}\\pm{p['sem']:.6f}$ (one SEM)."))
else:
    display(Markdown("**This sample does not establish the previously observed signed response.**"))
control=next(r for r in report["controls"] if r["selection"]==report["primary_selection"])["epsilon0"]
display(Markdown(f"**Equal-power control (ε=0):** mean signal {control['mean']:+.6f} ± {control['sem']:.6f} SEM; "
                 f"95% interval [{control['ci95_low']:+.6f}, {control['ci95_high']:+.6f}]."))
baseline=next(r for r in report["summaries"] if r["selection"]=="all")
display(Markdown(f"**All-tetrahedron comparison:** mean matched difference {baseline['mean']:+.6f} "
                 f"± {baseline['sem']:.6f} SEM (descriptive)."))
'''), md(r'''This verifies the catalogue response at the chosen configuration. Establishing a physical amplitude or a resolution-independent optimum requires separate tests.

[Protocol](saved/field_PVdisp_B/protocol.json) · [Catalogue signals](saved/field_PVdisp_B/confirmation_catalogues.csv) · [Gaussian parameters](saved/field_PVdisp_B/distribution_summary.csv) · [Saved arrays](saved/field_PVdisp_B/confirmation.npz) · [Figures](saved/field_PVdisp_B/figures/)
''')]
    nb = notebook(cells,kind='run',run_name=run.name)
    nb.metadata['parity_results'].update(topic='field_PVdisp_B',primary_statistic='chi_hat')
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.metadata['jupyter']={'source_hidden':True}
    return nb
