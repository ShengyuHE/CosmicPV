"""Create and execute results notebooks from saved catalogue measurements."""
from datetime import datetime
from pathlib import Path
import os
import sys
from textwrap import dedent
from inspect import cleandoc
import shutil
from zoneinfo import ZoneInfo
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from parity_study.storage import digest
from parity_study.paths import OUTPUT_ROOT, expose_output, project_for_run


def md(source):
    return nbformat.v4.new_markdown_cell(cleandoc(source).strip())


def code(source):
    return nbformat.v4.new_code_cell(dedent(source).strip())


def notebook(cells, *, kind, run_name=None):
    return nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name':'python3', 'display_name':'Python 3', 'language':'python'},
        'language_info': {'name':'python', 'version':sys.version.split()[0]},
        'parity_results': {'kind':kind, 'run':run_name, 'format_version':1},
    })


TABLE_HELPER = '''
def show_table(rows, columns):
    """Display a compact table; columns are (dictionary key, displayed heading)."""
    if not rows:
        display(Markdown("No measurements available for this section."))
        return
    def fmt(value):
        if value is None:
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.5g}"
        return str(value)
    heading = "".join(f"<th>{html.escape(title)}</th>" for _, title in columns)
    body = "".join("<tr>" + "".join(
        f"<td>{html.escape(fmt(row.get(key)))}</td>" for key, _ in columns) + "</tr>"
        for row in rows)
    display(HTML("<div style='overflow-x:auto'><table><thead><tr>" + heading +
                 "</tr></thead><tbody>" + body + "</tbody></table></div>"))
'''


def _detail_notebook(run):
    root = project_for_run(run)
    setup = f'''
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import contextlib
import csv
import html
import io
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
from IPython.display import display, Markdown, HTML, Image

RUN_NAME = {run.name!r}
PROJECT = next((p for p in (Path.cwd(), *Path.cwd().parents)
                if (p / "results" / "saved" / RUN_NAME / "config.json").is_file()), Path({str(root)!r}))
RUN = Path({str(run)!r})
sys.path.insert(0, str(PROJECT / "src"))
from parity_study.summarize import summarize

# Refresh statistics from completed NPZ records. This never generates a catalogue.
with contextlib.redirect_stdout(io.StringIO()):
    summary = summarize(RUN)
config = json.loads((RUN / "config.json").read_text())
manifest = json.loads((RUN / "manifest.json").read_text())
convention = config["field"].get("amplitude_convention", "notebook")
status = "COMPLETE" if summary["complete"] else "PARTIAL — jobs or measurements remain"
updated = datetime.now(ZoneInfo("Europe/Zurich")).isoformat(timespec="seconds")
display(Markdown(f"**{{status}}** · {{summary['completed_records']:,}} / "
                 f"{{summary['expected_records']:,}} catalogue measurements.  \\nUpdated: {{updated}}"))
display(Markdown(config.get("description", "")))
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams.update({{"figure.dpi":110, "axes.grid":True, "grid.alpha":0.2}})
{TABLE_HELPER}
'''
    return notebook([
        md(f'''# Parity violation results — {run.name}

        This notebook is the results view for the saved run. **Run All** refreshes its tables and plots from completed records; catalogue generation and Delaunay triangulation stay in the batch jobs. Save the notebook after rerunning to retain the displayed outputs.

        [All runs](../overview.ipynb) · [Run configuration](config.json) · [Provenance](manifest.json)
        '''),
        code(setup),
        md('''## Parameters and amplitude convention

        The convention is taken from the run's saved configuration, not from today's generator defaults. Runs created before normalization use the original per-mode power amplitude. Their numerical A values must not be interpreted as RMS lengths.
        '''),
        code(r'''
        field = config["field"]
        show_table([{"parameter": k, "value": v} for k, v in field.items()],
                   [("parameter", "Field parameter"), ("value", "Value")])
        if convention == "rms":
            display(Markdown(r"**RMS convention:** $P_{\psi,k}=A^2P_k/\sum_{\mathbf{k}}P_k$. "
                              r"The expected displacement RMS is $A$ for response $R=1$."))
        else:
            display(Markdown(r"**Original notebook convention:** $P_{\psi,k}=A\,W_k$, "
                              r"with $W_k$ the mode-band mask. The expected RMS is "
                              r"$\sqrt{A N_{\rm modes}}$ for $R=1$. "
                              "These data predate the normalized RMS-amplitude implementation."))
        display(Markdown(f"Original notebook commit: `{manifest.get('notebook_commit', 'unrecorded')}`. "
                         "Parameters, simulation source, and executable are retained in this run directory."))
        completeness = [{"case": name, **values} for name, values in summary["completeness"].items()]
        show_table(completeness, [("case","Case"),("completed","Saved seeds"),("expected","Planned seeds")])
        '''),
        md('''## Pipeline validation

        These tests check the reference catalogue and measurement pipeline. The validation catalogue uses its separately recorded parameters; it is not a measurement at every production amplitude. Proper transformations preserve chirality, reflections reverse its sign, and the cells should fill the periodic volume exactly.
        '''),
        code(r'''
        validation_file = RUN / "validation/report.json"
        if validation_file.exists():
            validation = json.loads(validation_file.read_text())
            display(Markdown(f"**Reference validation passed: {validation.get('passed', False)}**"))
            rows = [{"transformation": name, **values} for name, values in validation.get("T1", {}).items()]
            show_table(rows, [("transformation","Transformation"),("n_tetrahedra","Tetrahedra"),
                             ("sum_chi_hat","Sum of normalized chirality"),("absolute_error","Symmetry error"),
                             ("volume_fraction","Volume / box volume"),("passed","Passed")])
            display(Markdown(f"NUFFT/direct maximum displacement difference: "
                             f"`{validation.get('nufft_direct_max_error', float('nan')):.3g}`."))
            helicity = validation.get("helicity", {})
            show_table([{"field":k, **helicity[k]} for k in ("plus","minus") if k in helicity],
                       [("field","Field"),("normalized_helicity","Normalized helicity"),
                        ("longitudinal_fraction","Longitudinal power fraction")])
        else:
            display(Markdown("Validation report is not available yet."))
        xml_file = RUN / "validation/pytest.xml"
        if xml_file.exists():
            suites = ET.parse(xml_file).getroot()
            show_table([dict(s.attrib) for s in suites.iter("testsuite")],
                       [("tests","Regression tests"),("failures","Failures"),("errors","Errors"),("time","Seconds")])
        '''),
        md('''## Positive control: chiral molecules

        The table and plot use the mean normalized chirality across all tetrahedra. Errors are **one standard error across independent seeds**. Identified molecules are shown separately from the cells connecting molecules; the latter are measured, not assumed to contribute exactly zero.
        '''),
        code(r'''
        positive = [r for r in summary["ensemble_statistics"]
                    if r["case"].startswith("positive_") and r["statistic"] == "chi_hat"]
        show_table(positive, [("case","Case"),("n","Seeds"),("mean","Mean normalized chirality"),
                              ("sem","SEM"),("ci95_low","95% CI low"),("ci95_high","95% CI high")])
        if positive:
            fig, ax = plt.subplots(figsize=(7,4), constrained_layout=True)
            valid = [r for r in positive if r["sem"] is not None]
            ax.errorbar([float(r["case"].split("_f")[1]) for r in valid],
                        [r["mean"] for r in valid], yerr=[r["sem"] for r in valid],
                        fmt="o-", capsize=3, label="All tetrahedra")
            molecule = [r for r in summary["molecular_controls"]
                        if r["quantity"] == "molecule_contribution_to_all"]
            ax.plot([float(r["case"].split("_f")[1]) for r in molecule],
                    [r["mean"] for r in molecule], "s--", label="Identified molecule contribution")
            ax.axhline(0, color="black", lw=.7)
            ax.set(xlabel="Right-handed molecule fraction", ylabel="Mean normalized chirality",
                   title=f"{status} | {min(r['n'] for r in positive)}–{max(r['n'] for r in positive)} seeds per fraction")
            ax.legend()
            plt.show()
            show_table([r for r in summary["molecular_controls"]
                        if r["quantity"] in ("measured_mean","other_mean","n_identified")],
                       [("case","Case"),("quantity","Quantity"),("n","Seeds"),("mean","Mean"),("sem","SEM")])
        '''),
        md('''## Null cases

        These catalogue means should have ensemble expectation zero. A value around two standard errors from zero can occur by chance, especially across several correlated statistics. Sample-standardized histograms provide a diagnostic; exact Gaussian histograms are not a requirement.
        '''),
        code(r'''
        null_rows = [r for r in summary["ensemble_statistics"]
                     if r["case"].startswith("null_") and r["statistic"] == "chi_hat"]
        show_table(null_rows, [("case","Null case"),("statistic","Statistic"),("n","Seeds"),
                              ("mean","Mean"),("sem","SEM"),("t","Mean / SEM"),
                              ("p_two_sided","Uncorrected p")])
        null_plot = RUN / "summary/null_distributions.png"
        if null_rows and null_plot.exists():
            display(Markdown(f"**{status}** — seed counts are listed above."))
            display(Image(filename=str(null_plot)))
        '''),
        md(r'''## Helical displacement scan

        For each amplitude, compare matching seeds using
        $D_s=S_s(\epsilon=+1)-S_s(\epsilon=-1)$.
        The uncertainty is the scatter of $D_s$ divided by $\sqrt{N_s}$.
        Tetrahedra share vertices and are not treated as independent observations.
        Use $S_s=\langle\widehat\chi\rangle_s$, the catalogue mean of normalized chirality.
        Scan-adjusted p-values use Bonferroni correction across the planned amplitudes.
        '''),
        code(r'''
        paired = summary["primary_scan"]["rows"]
        show_table(paired, [("A","A"),("n","Seed pairs"),
                           ("mean","Mean difference"),("sem","SEM"),("t","Difference / SEM"),
                           ("p_two_sided","Local p"),("p_bonferroni","Scan-adjusted p")])
        if paired:
            fig, ax = plt.subplots(figsize=(7,4), constrained_layout=True)
            xlabel = r"Displacement strength $A$" if convention == "rms" else "A (original per-mode power)"
            rows = [r for r in paired if r["sem"] is not None]
            if rows:
                ax.errorbar([r["A"] for r in rows], [r["mean"] for r in rows],
                            yerr=[r["sem"] for r in rows], fmt="o-", capsize=3)
            ax.axhline(0, color="black", lw=.7)
            ax.set(xscale="log", xlabel=xlabel,
                   ylabel=r"$\overline{D}=\langle D_s\rangle_s$")
            fig.suptitle(f"{status} | {min(r['n'] for r in paired)}–{max(r['n'] for r in paired)} seed pairs; errors = 1 SEM")
            plt.show()
        '''),
        md('''## Actual displacement strength

        The dimensionless ratio is the measured **unwrapped** displacement RMS divided by the mean point spacing $\ell=L/N_{\rm gal}^{1/3}$. It helps compare runs with different mode counts and amplitude conventions. Wrapping positions into the box does not reduce the recorded displacement RMS.
        '''),
        code(r'''
        with (RUN / "summary/realizations.csv").open() as stream:
            realizations = list(csv.DictReader(stream))
        strength = []
        for A in config["scan"]["amplitudes"]:
            for hand in ("plus", "minus"):
                case = f"scan_A{A:g}_{hand}"
                ratios = [float(r["rms_over_spacing"]) for r in realizations
                          if r["case"] == case and r.get("rms_over_spacing")]
                if ratios:
                    strength.append({"A":A, "helicity":hand, "n":len(ratios),
                                     "mean_ratio":float(np.mean(ratios)),
                                     "min_ratio":float(np.min(ratios)), "max_ratio":float(np.max(ratios))})
        show_table(strength, [("A","A"),("helicity","Helicity"),("n","Seeds"),
                              ("mean_ratio","Mean RMS / spacing"),("min_ratio","Minimum"),("max_ratio","Maximum")])
        '''),
        md('''## Interpretation and notes

        Pipeline validation establishes correct transformation behaviour. The molecule control tests sensitivity to deliberately chiral point arrangements. Detecting injected field helicity through the final scalar catalogue is a separate statistical question. For a transverse displacement field, the first-order density response vanishes, so sensitivity depends on nonlinear structure and on the chosen statistic.
        '''),
        code(r'''
        display(Markdown(f"**Analysis status: {status}.** "
                         f"{summary['completed_records']:,} of {summary['expected_records']:,} planned catalogue measurements are included."))
        usable = [r for r in summary["primary_scan"]["rows"] if r["t"] is not None]
        if usable:
            peak = max(usable, key=lambda r: abs(r["t"]))
            display(Markdown(f"The largest absolute paired mean/SEM currently shown is **{abs(peak['t']):.2f}**, "
                             f"for `{peak['statistic']}` at A={peak['A']:g}, using {peak['n']} seed pairs. "
                             "This is an uncorrected scan maximum, not a discovery significance."))
        display(Markdown("Use the full ensemble before choosing a scientific interpretation. "
                         "Shape or size selections developed from these measurements need independent validation seeds."))
        '''),
        md('''### Research notes

        Add your interpretation and follow-up questions here. Automatic refresh executes the existing notebook and preserves its cell sources, including notes you add.
        '''),
        md('''## Access the saved measurements

        Each NPZ file retains its catalogue, statistics, metadata, and (when enabled) tetrahedra. This example inspects one saved record; it performs no simulation.
        '''),
        code(r'''
        example = next(iter(sorted((RUN / "records").glob("*/seed_*.npz"))), None)
        if example is not None:
            with np.load(example) as saved:
                metadata = json.loads(str(saved["metadata_json"]))
                print("Record:", example.relative_to(RUN))
                print("Arrays:", saved.files)
                print("Catalogue shape:", saved["catalogue"].shape)
                print("Statistics [chi, chi_hat, sign]:", saved["statistics"])
                print("Parameters:", metadata["parameters"])
                print("Diagnostics:", metadata["diagnostics"])
                # tetrahedra = saved["tetrahedra"]
                # columns: chi, chi_hat, l_min, l_mid, l_max
        '''),
    ], kind='run', run_name=run.name)


def run_notebook(run):
    """Present each run as explicit tests, with the scientific conclusion first."""
    import json
    config = json.loads((run/'config.json').read_text())
    base = _detail_notebook(run)
    b = base.cells
    has_positive = config['positive']['seeds'] > 0
    has_null = config['null']['seeds'] > 0
    field = config['field']
    amplitudes = ', '.join(f'{a:g}' for a in config['scan']['amplitudes'])
    title = 'Baseline: pipeline validation and helicity sensitivity' if has_null else 'Large-amplitude helicity scan'
    coverage = 'T0–T4 below' if has_null else 'T0, T1, and T4 below; the molecule and null ensembles (T2/T3) belong to the baseline notebook'
    cells = [md(f'''# {title}

This notebook demonstrates **{coverage}**.
The physical question is whether reversing the injected field helicity changes the mean normalized chirality χ̂ of the resulting Delaunay tetrahedra.

**Run:** `{run.name}`. Data and plots are in [{run.name}/](saved/{run.name}/).
Use **Run All** to refresh the analysis from saved measurements, then save the notebook. No catalogues are rebuilt.
'''), b[1],
        md('''## Read this first: what do the tests establish?

**T0–T3 validate ingredients of the analysis. T4 measures the response to the helical displacement model.** Passing a geometry test or detecting an artificial chiral molecule does not by itself establish a helicity signal in displaced catalogues.

The table below states the result of each test included in this run. A partial run is labeled provisional. Statistical errors are calculated across independent seeds.
'''),
        code(r'''
validation_file = RUN / "validation/report.json"
validation = json.loads(validation_file.read_text()) if validation_file.exists() else {}
primary = summary["primary_scan"]
paired = primary["rows"]
null_rows = [r for r in summary["ensemble_statistics"]
             if r["case"].startswith("null_") and r["statistic"] == "chi_hat"]
positive = [r for r in summary["ensemble_statistics"]
            if r["case"].startswith("positive_") and r["statistic"] == "chi_hat"]

def family_check(rows):
    # Conservative correction for the explicitly listed family of comparisons.
    usable = [r for r in rows if r.get("p_two_sided") is not None]
    if not usable:
        return None, 0
    return min(1., len(usable)*min(r["p_two_sided"] for r in usable)), len(usable)

verdicts = []
recovered = False
p_null = None
err = validation.get("nufft_direct_max_error")
helicity = validation.get("helicity", {})
field_ok = (err is not None and err < 1e-5 and
            all(name in helicity and abs(helicity[name]["normalized_helicity"]-sign)<1e-12
                and helicity[name]["longitudinal_fraction"]<1e-28
                for name,sign in (("plus",1),("minus",-1))))
verdicts.append({"test":"T0 — field generation", "question":"Are the numerical field and its helicity correct?",
                 "result":"PASS" if field_ok else "Check validation report",
                 "evidence":f"NUFFT/direct error = {err:.3g}; helicities +1 / −1" if err is not None else "No report yet"})
transforms = validation.get("T1", {})
geometry_ok = bool(transforms) and all(r["passed"] for r in transforms.values())
verdicts.append({"test":"T1 — geometry", "question":"Does the measured chirality transform correctly?",
                 "result":"PASS" if geometry_ok else "Check validation report",
                 "evidence":f"{len(transforms)} reference/transformed catalogues; equal cell counts and correct volume"})
if positive:
    right = next((r for r in positive if r["case"]=="positive_f1"), None)
    left = next((r for r in positive if r["case"]=="positive_f0"), None)
    recovered = (right is not None and left is not None and right["ci95_low"] is not None
                 and left["ci95_high"] is not None and right["ci95_low"]>0 and left["ci95_high"]<0)
    evidence = (f"Mean χ̂: right {right['mean']:+.4f}, left {left['mean']:+.4f}" if right and left else "Awaiting both handedness controls")
    verdicts.append({"test":"T2 — known chiral molecules", "question":"Can the estimator detect deliberately chiral points?",
                     "result":"Injected chirality recovered" if recovered else "Awaiting sufficient control evidence",
                     "evidence":evidence})
if null_rows:
    p_null, n_null = family_check(null_rows)
    verdicts.append({"test":"T3 — null catalogues", "question":"Is there a detectable spurious parity signal?",
                     "result":"No detected null bias" if p_null is not None and p_null>=.05 else "Inspect null deviations",
                     "evidence":f"{n_null} checks; smallest Bonferroni-adjusted p = {p_null:.3g}" if p_null is not None else "Insufficient seeds"})
usable = [r for r in paired if r.get("t") is not None]
p_scan, n_scan = primary["bonferroni_p"], primary["comparisons"]
if usable:
    peak = primary["peak"]
    verdicts.append({"test":"T4 — helicity reversal", "question":"Does ε=+1 differ from ε=−1?",
                     "result":"No detection at the scan-wide 5% threshold" if p_scan>=.05 else "Candidate difference; investigate independently",
                     "evidence":f"Largest |Δ/SEM| = {abs(peak['t']):.2f}; adjusted p = {p_scan:.3g}; {n_scan} comparisons"})
show_table(verdicts, [("test","Test"),("question","Question"),("result","Result"),("evidence","Evidence")])
if not summary["complete"]:
    display(Markdown("**These conclusions are provisional because the ensemble is incomplete.**"))
if p_scan is not None and p_scan >= .05:
    display(Markdown("**Scientific conclusion:** mean χ̂ shows no detectable helicity contrast at the scan-wide 5% threshold. A weaker catalogue signal remains possible."))
display(Markdown("Scan p-values use a Bonferroni correction across planned amplitudes for χ̂ only; null checks correct across null cases. Uncertainties come from independent seed pairs."))
'''),
        md('''## Experiment being tested

The primary statistic is **χ̂**, normalized tetrahedron chirality in [−1,1], averaged over each catalogue. Uncertainties are calculated across independent seeds. The saved arrays also retain χ and sign(χ) as diagnostics.
'''),
        code(r'''
show_table([{"L":config["field"]["L"], "N":config["field"]["N"], "Ngal":config["field"]["Ngal"],
             "amplitudes":str(config["scan"]["amplitudes"]), "pairs":config["scan"]["seeds"],
             "convention":convention}],
           [("L","Box size"),("N","Fourier grid N"),("Ngal","Points per catalogue"),
            ("amplitudes","Scan amplitudes A"),("pairs","Planned seed pairs per A"),("convention","Amplitude convention")])
if convention == "notebook":
    display(Markdown(r"**These simulations use the original amplitude convention:** $P_{\psi,k}=A W_k$, "
                      r"giving expected displacement RMS $\sqrt{A N_{\rm modes}}$ for $R=1$. "
                      "**They do not test the newly normalized RMS-amplitude generator.** "
                      "That code update needs a separately labeled run. The measured RMS/spacing is shown under T4."))
else:
    display(Markdown(r"**Normalized RMS amplitude:** $P_{\psi,k}=A^2P_k/\sum_{\mathbf k}P_k$, "
                      r"so the expected displacement RMS is $A$ for response $R=1$."))
'''),
        md('''## T0 — Does the catalogue generator produce the intended field?

**Test:** compare NUFFT evaluation with direct Fourier summation on the same realization (N=16, L=100, 1,000 points, seed 42). Separately measure helicity and longitudinal power on an N=16, L=1000 field with Nyquist planes removed (seed 5).

**Expected:** numerical agreement (maximum error below 10⁻⁵ for the saved reference test), normalized helicity +1/−1 for the two extreme settings, and negligible longitudinal power. These establish that the field was generated correctly; they do not predict the catalogue statistic's signal-to-noise ratio.
'''),
        code(r'''
show_table([{"error":err, "threshold":1e-5, "passed":field_ok}],
           [("error","Maximum NUFFT/direct difference"),("threshold","Required upper bound"),("passed","T0 passed")])
show_table([{"field":name, **helicity[name]} for name in ("plus","minus") if name in helicity],
           [("field","Field"),("normalized_helicity","Measured normalized helicity"),
            ("longitudinal_fraction","Longitudinal power fraction")])
'''),
        md('''## T1 — Does the chirality measurement obey parity and periodic geometry?

**Test:** use the reference catalogue and apply inversion, an axis swap, a 90° rotation, and a periodic translation.

**Expected:** inversion and axis swap reverse the sign of the chirality sum. Rotation and translation preserve it. Every case has the same tetrahedron count, and the sum of tetrahedron volumes equals the box volume. The validation uses the saved reference setup (N=64, L=1000, A=0.002 in the original convention).
'''),
        code(r'''
expected_signs = {"reference":1,"inversion":-1,"swap":-1,"rotation90":1,"translation":1}
show_table([{"transformation":name,"expected_sign":expected_signs[name],**values} for name,values in transforms.items()],
           [("transformation","Transformation"),("expected_sign","Expected sign / reference"),
            ("n_tetrahedra","Tetrahedra"),("sum_chi_hat","Sum χ̂"),("absolute_error","Symmetry error"),
            ("volume_fraction","Volume / box volume"),("passed","Passed")])
display(Markdown("**T1 conclusion:** " + ("the geometric implementation passes all reference transformations." if geometry_ok else "the validation needs inspection.")))
''')]
    if has_positive:
        cells += [md(f'''## T2 — Can the pipeline recover known chiral point structures?

**Test:** inject {config['positive']['Ncl']:,} small chiral tetrahedral molecules, randomly positioned and rotated. Vary the right-handed fraction over {config['positive']['fractions']}, with {config['positive']['seeds']} independent realizations per fraction.

**Expected:** identified molecules have mean χ̂ close to `2 f_R − 1`. The all-cell mean has the appropriate sign but is diluted by other Delaunay cells. The balanced mixture should be consistent with zero within sampling uncertainty. Connecting cells are measured separately.

**What this demonstrates:** sensitivity to a deliberately chiral point arrangement. It does not establish sensitivity to the displacement model in T4. Plot error bars are one SEM across seeds.
'''), b[7], code(r'''
balanced = next((r for r in positive if r["case"] == "positive_f0.5"), None)
display(Markdown("**T2 conclusion:** " + ("the estimator recovers both signs of injected chirality." if recovered
                 else "the pure-handedness controls do not yet establish recovery of both signs.")))
if balanced is not None and balanced["sem"] is not None:
    display(Markdown(f"The balanced mixture gives mean χ̂ = {balanced['mean']:+.4g} ± {balanced['sem']:.3g} "
                     f"(1 SEM), or {balanced['t']:+.2f} standard errors from zero."))
''')]
    if has_null:
        cells += [md(f'''## T3 — Do parity-symmetric catalogues generate a false signal?

**Test:** {config['null']['seeds']} independent seeds for each null case:

- `null_poisson`: undisplaced Poisson points.
- `null_symmetric`: equal helicity power, ε=0, with A={config['null']['A']:g} and L={field['L']:g}.
- `null_notebook_L1_A1`: reproduction of the original strong-displacement null, A=1 and L=1.

**Expected:** the ensemble mean of χ̂ is zero. The opening summary corrects across the null cases.

The histograms below subtract each sample mean and divide by its sample standard deviation. They show distribution shape; the uncentered means in the table test bias.
'''), b[9], code(r'''
display(Markdown("**T3 conclusion:** " + ("no null mean passes the family-wide 5% threshold; "
                 "these ensembles show no detected spurious parity bias." if p_null is not None and p_null >= .05
                 else "inspect the null deviations before interpreting the scan.")))
''')]
    cells += [md(f'''## T4 — Is the injected helicity detectable in Delaunay chirality?

**Test:** {config['scan']['seeds']} matched seed pairs per amplitude, with N={field['N']}, L={field['L']:g}, {field['Ngal']:,} points, and A={amplitudes}. For each seed, compare maximum positive and negative field helicity.

**Measurement:** `S_s = mean(chi_hat)` over a catalogue's tetrahedra; `D_s = S_s(ε=+1) − S_s(ε=−1)`. Report the mean D and its SEM across matched seed pairs. The paired catalogues share initial positions and random draws.

**Expected under no detectable response:** a difference consistent with zero. The table, plot, and scan-wide test all use χ̂, with correction over the planned amplitudes.
''')]
    scan_cell = b[11]
    cells += [scan_cell,
        code(r'''
if usable:
    display(Markdown(f"**T4 result:** the largest absolute difference/SEM is {abs(peak['t']):.2f} "
                     f"for `{peak['statistic']}` at A={peak['A']:g}, using {peak['n']} seed pairs. "
                     f"The conservative scan-wide p-value is {p_scan:.3g} across {n_scan} comparisons."))
    display(Markdown("**T4 conclusion:** " + ("no difference passes the stated scan-wide 5% threshold. "
                     "A weaker response remains possible; a nondetection does not invalidate the geometry checks." if p_scan>=.05
                     else "a candidate difference passes the stated threshold; inspect its physical scaling and validate it with independent realizations.")))
'''),
        md('''### How strong were the actual displacements?

The following ratio uses the **unwrapped displacement RMS / mean point spacing**. This is the useful comparison between different mode counts and amplitude conventions. Very large values can substantially rearrange the point catalogue; a larger input amplitude need not give a larger measured parity signal.
'''), b[13],
        md('''## Scope of the conclusion and next study

The completed tests distinguish numerical correctness, sensitivity to artificial chiral structures, and sensitivity to the chosen displacement model. The transverse field has no first-order density response, so a signal in the final scalar catalogue depends on nonlinear structure and the statistic being measured.

A follow-up with the normalized RMS-amplitude convention should specify A in catalogue length units and its ratio to the mean spacing. Tetrahedron size/shape analysis can use the saved arrays; any weighting selected from these results should be validated with independent seeds.
'''),
        md('''## Research notes

Add your interpretation and follow-up questions here. Normal refreshes preserve notebook source cells and notes.
'''),
        md('''## Reproducibility

The run folder retains all raw measurements and diagnostic statistics. The primary χ̂ table includes local and amplitude-corrected p-values.
'''),
        code(r'''
completeness = [{"case":name, **values} for name,values in summary["completeness"].items()]
show_table(completeness, [("case","Case"),("completed","Saved seeds"),("expected","Planned seeds")])
xml_file = RUN / "validation/pytest.xml"
if xml_file.exists():
    suites = ET.parse(xml_file).getroot()
    show_table([dict(s.attrib) for s in suites.iter("testsuite")],
               [("tests","Regression tests"),("failures","Failures"),("errors","Errors")])
display(Markdown(f"[Configuration](saved/{RUN_NAME}/config.json) · [Provenance](saved/{RUN_NAME}/manifest.json) · "
                 f"[Saved plots and tables](saved/{RUN_NAME}/summary/)"))
'''), b[17], b[18]]
    out = notebook(cells, kind='run', run_name=run.name)
    out.metadata['parity_results'].update(format_version=3, primary_statistic='chi_hat')
    for cell in out.cells:
        if cell.cell_type == 'code':
            cell.metadata['jupyter'] = {'source_hidden': True}
    return out


def overview_notebook(results):
    root = results.parent
    return notebook([
        md('''# Parity violation studies — results overview

        Open a run notebook below for its validation, controls, parity measurements, and interpretation. Run notebooks refresh statistics from saved records and retain the displayed results when saved. Catalogue generation remains in batch jobs.

        This overview reads the latest saved summaries and counts the records currently on disk. If the live count exceeds the summarized count, rerun the corresponding results notebook.
        '''),
        code(f'''
        from pathlib import Path
        import html
        import json
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import numpy as np
        from IPython.display import display, Markdown, HTML
        PROJECT = next((p for p in (Path.cwd(), *Path.cwd().parents)
                        if (p / "results").is_dir() and (p / "src/parity_study").is_dir()), Path({str(root)!r}))
        import sys
        sys.path.insert(0, str(PROJECT / "src"))
        from parity_study.paths import OUTPUT_ROOT
        RESULTS = OUTPUT_ROOT
        print("Updated:", datetime.now(ZoneInfo("Europe/Zurich")).isoformat(timespec="seconds"))
        '''),
        code(TABLE_HELPER),
        code(r'''
        rows = []
        for run in sorted(RESULTS.iterdir()):
            if not run.is_dir() or not (run / "config.json").is_file():
                continue
            config = json.loads((run / "config.json").read_text())
            path = run / "summary/summary.json"
            summary = json.loads(path.read_text()) if path.exists() else {}
            convention = config["field"].get("amplitude_convention", "notebook")
            rows.append({"run":run.name, "grid":config["field"]["N"], "box":config["field"]["L"],
                         "convention":"RMS length" if convention == "rms" else "Original per-mode power",
                         "status":"Complete" if summary.get("complete") else "Partial / not summarized",
                         "summarized":summary.get("completed_records",0),
                         "live":len(list((run / "records").glob("*/seed_*.npz"))),
                         "planned":summary.get("expected_records"),
                         "summary_time":summary.get("generated_at", "See run notebook")})
        show_table(rows, [("run","Run"),("grid","Grid N"),("box","Box L"),("convention","Meaning of A"),
                          ("status","Status"),("summarized","Summarized"),("live","Saved now"),
                          ("planned","Planned"),("summary_time","Summary updated")])
        for row in rows:
            name = row["run"]
            from parity_study.notebooks import report_filename
            filename = report_filename(name)
            if filename is None:
                display(Markdown(f"### {name} — archived; [saved data](saved/{name}/)"))
            else:
                display(Markdown(f"### [{name}]({filename})"))
            config = json.loads((RESULTS / name / "config.json").read_text())
            display(Markdown(config.get("description", "")))
        '''),
        md(r'''## Comparing amplitude conventions

        **Original notebook:** $P_{\psi,k}=A W_k$, so the expected displacement RMS is $\sqrt{A N_{\rm modes}}$ for response $R=1$.

        **Normalized RMS:** $P_{\psi,k}=A^2P_k/\sum_{\mathbf k}P_k$, so the expected displacement RMS is $A$ for $R=1$.

        Old runs retain their original definition. Compare their measured displacement RMS/spacing, not the raw numerical A values, against future normalized runs. Means and uncertainties use independent catalogue seeds; tetrahedra are not independent samples.
        '''),
    ], kind='overview')


def execute_and_save(path, fresh, *, rebuild=False):
    """Refresh existing notebooks without replacing their source cells or notes."""
    path = Path(path)
    before = digest(path) if path.exists() else None
    old = nbformat.read(path, as_version=4) if before and path.stat().st_size else None
    if old is not None and old.metadata.get('parity_results', {}).get('kind') != fresh.metadata['parity_results']['kind']:
        raise ValueError(f'Refusing to replace an unrelated notebook: {path}')
    if old is not None and rebuild:
        output = OUTPUT_ROOT/fresh.metadata['parity_results'].get('topic', path.stem)
        output.mkdir(parents=True, exist_ok=True)
        expose_output(path.parent.parent, output)
        history = output/'notebook_history'
        history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo('Europe/Zurich')).strftime('%Y%m%d_%H%M%S_%f')
        shutil.copy2(path, history/f'{stamp}.ipynb')
    nb = old if old is not None and not rebuild else fresh
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-Xfrozen_modules=off', '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(nb, km=km, timeout=600, allow_errors=False,
                            resources={'metadata':{'path':str(path.parent)}})
    # NotebookClient does not own a supplied kernel manager; close it explicitly.
    try:
        client.execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
        km.cleanup_resources()
    nb.metadata['parity_results']['executed_at'] = datetime.now(ZoneInfo('Europe/Zurich')).isoformat(timespec='seconds')
    nbformat.validate(nb)
    if (digest(path) if path.exists() else None) != before:
        raise RuntimeError(f'Notebook edited during refresh; its edits were preserved: {path}')
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    nbformat.write(nb, tmp)
    tmp.replace(path)
    print(f'Saved executed notebook: {path}', flush=True)
    return path


def write_run_notebook(run, *, rebuild=False):
    run = Path(run).resolve()
    if not (run/'config.json').is_file():
        raise FileNotFoundError(f'No saved run configuration in {run}')
    project = project_for_run(run)
    reports = project/'results'
    expose_output(project, run)
    filename = report_filename(run.name)
    if filename is None:
        print(f"Archived field run: {run.name}; the fiducial report uses the larger-box run.", flush=True)
        return None
    path = reports/filename
    import json
    if json.loads((run/'config.json').read_text()).get('purpose') == 'best_configuration_confirmation':
        from parity_study.tests.best_confirmation_notebooks import confirmation_notebook
        fresh = confirmation_notebook(run)
    elif json.loads((run/'config.json').read_text()).get('purpose') == 'amplitude_scan':
        from parity_study.tests.amplitude_notebooks import amplitude_notebook
        fresh = amplitude_notebook(run)
    elif json.loads((run/'config.json').read_text())['field'].get('model') == 'mesh_za_helical':
        from parity_study.tests.ic_notebooks import mesh_notebook
        fresh = mesh_notebook(run)
    elif run.name == '20261121_baseline':
        from parity_study.tests.study_notebooks import basic_notebook
        fresh = basic_notebook(reports)
    elif run.name == '20261121_notebook_current':
        from parity_study.tests.study_notebooks import displacement_notebook
        from parity_study.tests.intro_fiducial import active_example
        if active_example() is None:
            from parity_study.tests.tutorial_examples import build_examples
            build_examples(project)
        fresh = displacement_notebook(reports)
    else:
        fresh = run_notebook(run)
    nested = run/'results.ipynb'
    if nested.exists() and not path.exists():
        nested.rename(path)
    return execute_and_save(path, fresh, rebuild=rebuild)


def report_filename(run_name):
    """Route active reports; historical field tests have no visible notebook."""
    active_amplitude = OUTPUT_ROOT/'field_PVdisp_A/active_scan.json'
    if active_amplitude.exists():
        import json
        if run_name in json.loads(active_amplitude.read_text()).get('superseded_runs', []):
            return None
    if run_name in {'20261121_mesh_ic_smoke', '20261121_mesh_ic_L1000_N256',
                    '20261121_mesh_ic_L1000_N256_v2',
                    '20261121_mesh_ic_L1000_N256_selection_confirmation',
                    '20261121_mesh_ic_L1000_N256_selection_25',
                    '20261121_mesh_ic_L1000_N256_independentIC_25'}:
        return None
    return {'20261121_baseline':'basic.ipynb',
            '20261122_mesh_ic_L1000_N256_A16_confirmation_50':'field_PVdisp_B.ipynb',
            '20261121_mesh_ic_L1000_N256_A_extended':'field_PVdisp_A.ipynb',
            '20261121_mesh_ic_L1000_N256_A_scan':'field_PVdisp_A.ipynb',
            '20261121_mesh_ic_L1000_N256_fixedIC_10':'field_PVdisp.ipynb',
            '20261121_notebook_current':'intro_PVdisp.ipynb'}.get(run_name, f'{run_name}.ipynb')


def write_overview_notebook(results):
    results = Path(results).resolve()
    return execute_and_save(results/'overview.ipynb', overview_notebook(results))
