# TPIVa parity violation study

Reproducible Python and Slurm version of Josue Aubert's catalogue-injection notebook. The original checkout remains in `tp4a-pv-delaunay/`; new experiments live at this project root.

## Git checkout

```bash
git clone --recurse-submodules https://github.com/ShengyuHE/CosmicPV.git
cd CosmicPV
```

For an existing clone, run `git submodule update --init --recursive`. The original student repository at `tp4a-pv-delaunay/` and upstream DIVE at `logs/vendor/DIVE/` are submodules pinned to the revisions used here. The DIVE patch and source manifest are tracked in `logs/patches/` and `logs/vendor/sources.json`; experiment configurations and the notebook review are tracked in `logs/configs/` and `logs/reports/`.

Git includes the executed `results/*.ipynb` reports and `results/fiducial_configuration.json`. Generated catalogues, arrays, exported figures, job logs, the local environment, and compiled builds remain outside version control. Scratch outputs live in `/home/users/h/hes/scratch/projects/cosmicPV`; the local `results/saved/` links are excluded. Embedded notebook plots can be viewed directly after cloning; rerunning the reports requires their saved scratch inputs and the cluster environment described below.

## Results notebooks

Open the executed notebooks directly in `results/`:

- [basic.ipynb](results/basic.ipynb): what parity violation means for a point catalogue, how to define tetrahedron chirality, reflection/rotation checks, and positive/null controls.
- [intro_PVdisp.ipynb](results/intro_PVdisp.ipynb): the current L=1000, A=16 displacement model, its normalized smoothed k⁻⁴ power, helicity, and field/particle examples from the saved 256³-mesh, 32³-particle fiducial realization.
- [field_PVdisp.ipynb](results/field_PVdisp.ipynb): the fiducial L=1000, 256³-mesh, 32³-particle study. Sections 1–5 introduce the field and original ten-seed results; Sections 6–7 examine the frozen shape selection with 25 displacement realizations at fixed density. Section 8 shows the completed 25-independent-Gaussian-IC test. Section 5 retains the +1 minus −1 histogram difference.
- [field_PVdisp_A.ipynb](results/field_PVdisp_A.ipynb): the completed matched displacement-strength scan, with catalogue signals for ε=−1,0,+1 and selected/unselected helicity differences versus A.
- [field_PVdisp_B.ipynb](results/field_PVdisp_B.ipynb): fresh confirmation at A=16 with 50 independent ICs, catalogue-signal histograms and Gaussian approximations for ε=−1,0,+1, and the paired helicity difference.
- [field_PVdisp_selection.ipynb](results/field_PVdisp_selection.ipynb): exploratory transfer of fixed geometric cuts across matched random thinning of ten existing ICs. Compares the original three-edge ratio with five fixed bins of six-edge mean-ratio shape Q, keeping the fields identical.

The notebooks follow scientific questions rather than simulation run order. Tables and plots are embedded so they are visible without execution.

The fiducial field notebook plots the catalogue signal `S_s(epsilon)=mean(chi_hat)` on the horizontal axis, with one point per catalogue and the mean ± 1 SEM across seeds. Its next cell retains `Delta p(chi_hat)=p(+1)-p(-1)`, subtracting probability-density histograms within each matched seed before averaging; error bars there are pointwise 1 SEM, and bins are correlated. The epsilon=0 case uses equal helical power at A=2; the three-way comparison is available only at this saved amplitude. Figures, catalogue signals, per-seed histogram differences, their mean covariance, and CSV tables are saved in scratch `cosmicPV/field_PVdisp/`. Earlier small-box notebook outputs are archived in scratch `cosmicPV/archive/`; automatic refresh skips the historical small-box run and the failed large-box preflight.

Sections 6–7 test reflection-invariant tetrahedron selections at A=2. Discovery uses saved seeds 0–9: three symmetric |χ̂| bands and three groups each in size and shape, with geometry boundaries derived from the ZA-only catalogue. The largest absolute paired t statistic chooses one primary cut, frozen in scratch `field_PVdisp/selections/protocol.json` before generating independent displacement seeds 10–19. The confirmation run is `20261121_mesh_ic_L1000_N256_selection_confirmation`; its results appear in the same fiducial notebook. The primary test uses a two-sided Student-t p-value and matching effect sign; secondary cuts use a nine-comparison correction. Per-catalogue measurements, selected counts, source hashes, transformation checks and the protocol are retained in the `selections/` folder. These tests condition on the same density seed 0 and do not establish behavior across independent density fields.

The selection ensemble is extended to **25 realizations, seeds 10–34**, in `20261121_mesh_ic_L1000_N256_selection_25`. Seeds 10–19 are reused after checking their field parameters, construction code, DIVE binary, particle positions and file hashes; only seeds 20–34 require new catalogues. The cut, A=2, density seed 0, L=1000 and N=256 remain fixed. The extension protocol and outputs live in scratch `field_PVdisp/selections/extended_25/`; the original ten-seed protocol and results remain unchanged. The expanded sample was requested after seeing the ten-seed result, so its combined p-values are nominal summaries, with the fifteen added seeds summarized separately. `field_PVdisp/active_selection.json` selects the report version used by the notebook. Report rendering archives the previous figures and notebook before activating the completed extension.

The independent-IC experiment uses **25 independent Gaussian ICs, seeds 100–124**, with one independent helical pair per IC and matched ε=−1,0,+1 at A=2. The shape bounds remain exactly `0.6905111410000623 <= r < 0.7567675934849439`; thresholds are not re-estimated from the new catalogues. `logs/configs/ic-L1000-N256-independentIC-25.json` and scratch `field_PVdisp/independent_ic_25/protocol.json` fix the sample size, single primary test, and original-protocol hash before generation. The completed run `20261121_mesh_ic_L1000_N256_independentIC_25` saves 100 catalogues, including 25 ZA-only controls. Its audit verifies distinct density/ZA/helical fields and exact matched particle maps. The selected mean difference is **−0.004055 ± 0.000564 SEM**, with 95% interval [−0.005218, −0.002891] and two-sided p=1.97e−7; the unselected difference is −0.000946 ± 0.000254 SEM. Uncertainties include both density and displacement scatter. The earlier fixed-density estimate is shown separately, without pooling the two ensembles. `scripts/render_field_report.py --independent-ic-dir <output-directory>` activates the completed result and adds Section 8 to the same notebook.

The amplitude study `20261121_mesh_ic_L1000_N256_A_scan` interprets L=1000 as **1000 comoving h⁻¹ Mpc** and scans **A=0,0.25,0.5,1,2,4,8,16 h⁻¹ Mpc**. It matches the same 25 independent ICs and frozen shape cut across A, reusing 100 audited A=2/ZA catalogues and generating 450 new catalogues. Every nonzero A has ε=−1,0,+1; at A=0 they share one ZA-only catalogue. The primary family contains seven nonzero amplitudes, with a Bonferroni correction; six paired contrasts against A=2 form a separate family. Cross-A covariance is saved. This characterizes the response on previously used ICs, so choosing an optimal A would need fresh confirmation data. Configuration: `logs/configs/ic-L1000-N256-A-scan.json`; protocol, measurements, plots, and report provenance: scratch `field_PVdisp_A/`, linked at `results/saved/field_PVdisp_A/`. Validation, ensemble, analysis, and notebook rendering run as dependent Slurm jobs.

A is the **ensemble RMS helical displacement**, a physical length independent of mesh spacing. For this toy spectrum, Δx=3.90625 h⁻¹ Mpc, particle spacing=31.25 h⁻¹ Mpc, expected ZA RMS≈3.547 h⁻¹ Mpc, and helical RMS wavenumber≈0.01970 h Mpc⁻¹. The report shows A relative to these scales and the dimensionless gradient diagnostic A·k_rms. Giving the box physical units does not calibrate the toy spectrum to a cosmology or redshift. Resolution comparisons should keep A and the physical spectral band fixed.

All **550/550 catalogues** completed and passed the independent-IC/matched-map audit; all 46 regression tests passed. With the frozen selection, A=1 h⁻¹ Mpc is the smallest tested amplitude detected after the seven-amplitude correction: mean D=−0.001813 ± 0.000366 SEM, corrected p=0.000334. A=0.25 and 0.5 are not detected after correction. The response magnitude rises through A=16, where mean D=−0.103836 ± 0.000963 SEM; A=2 reproduces the previous arrays exactly. Larger A strengthens this toy-model response, but the scan does not establish a physically preferred amplitude or numerical convergence. The executed notebook includes the full response and a low-A zoom; figures are saved as PNG and PDF. A report-only import fix is recorded separately in the run's `report_code/manifest.json`, leaving the scientific snapshot intact.

The completed larger-amplitude extension adds **A=24,32,48,64,96,128,256 h⁻¹ Mpc**, keeping the same 25 ICs, shape cut, and three helicities. It reuses the initial 550 catalogues and generates 525 new ones in `20261121_mesh_ic_L1000_N256_A_extended`. Protocol and analysis outputs are stored separately in `field_PVdisp_A/extended/`; the extension updates the same `field_PVdisp_A.ipynb`, preserving its previous version. Two rankings distinguish maximum signal magnitude from maximum |mean D|/SEM. Joint bootstrap resampling of ICs measures ranking stability, while all 91 nonzero-amplitude pairs use corrected paired comparisons. The 14-amplitude detection correction covers the combined grid. This extension is exploratory because its range was chosen after inspecting the first scan; an optimum on this grid requires fresh-IC confirmation and numerical convergence checks before adopting it as a fiducial model.

The **best tested A is 16 h⁻¹ Mpc for both selected signal magnitude and detection score**. Mean D is −0.103836 ± 0.000963 SEM at A=16, −0.094364 ± 0.001067 at A=24, −0.057119 ± 0.001044 at A=32, and −0.000035 ± 0.001157 at A=256. The A=16 versus A=24 signed-response contrast has corrected p=1.30e−8. All 10,000 paired bootstrap resamples select A=16 for magnitude; 89.98% select it for detection score, with 9.46% selecting A=24. These frequencies describe ranking stability, not probabilities of a true optimum. All **1,075 catalogues** passed the audit and all **48 tests** passed. A finer grid is needed to locate a continuous maximum near the measured peak. `field_PVdisp_A/active_scan.json` prevents ordinary refreshes from replacing the extended report with the initial scan; the scientific snapshot is unchanged by the separately recorded plot-axis correction.

The fresh confirmation run `20261122_mesh_ic_L1000_N256_A16_confirmation_50` fixes **A=16**, the exact original shape cut, and **50 independent ICs, seeds 1000–1049**, before generation. Each IC supplies three matched helicities plus a ZA-only control, for 200 catalogues in total. The one primary test is the two-sided paired Student t test of D=S(+1)−S(−1), requiring the original negative sign; the amplitude-scan ICs are excluded. Scratch `field_PVdisp_B/` stores the frozen protocol, per-catalogue measurements, descriptive Gaussian parameters, analysis, figures and provenance. The histogram contains one selected catalogue-mean χ̂ per IC and case. Gaussian curves use the sample mean and sample standard deviation across ICs; their width is not SEM, and exact Gaussianity is not assumed. Configuration: `logs/configs/ic-L1000-N256-A16-confirmation-50.json`. Validation, generation, analysis and report rendering run as dependent Slurm jobs; the report can also be refreshed with `scripts/update_notebooks.py --run results/saved/20261122_mesh_ic_L1000_N256_A16_confirmation_50`.

All **200/200 catalogues** passed the independent-IC/matched-map audit and all **51 tests** passed. The fresh selected response is **−0.104211 ± 0.000590 SEM**, with 95% interval [−0.105396, −0.103025] and paired two-sided p=2.17e−70. The three selected catalogue distributions have means **+0.052790, −0.000244, −0.051421** and sample standard deviations **0.003348, 0.002780, 0.003298** for ε=−1,0,+1, respectively. The equal-power and ZA-only controls are consistent with zero. This independently confirms the signed catalogue response at A=16; it does not establish a physical or continuous optimum. Generation used eight concurrent array tasks after confirming idle cluster capacity; the operational override is recorded in the run's `scheduler_override.json` separately from its frozen scientific inputs.

The displacement study and follow-up reports use **catalogue-mean χ̂ (normalized tetrahedron chirality)** as the primary statistic. Compare ε=+1 and ε=−1 with matching seeds, and correct the χ̂ p-values across the planned amplitudes in each scan. The historical top-hat, random-point scans have 100 seed pairs per amplitude and show no scan-wide detection (adjusted p=0.111 for N=64 and 0.387 for N=256). Those earlier results are archived; the current Gaussian-IC, ZA-plus-helical confirmation is in `field_PVdisp_B.ipynb`. The saved χ and sign(χ) columns remain available as diagnostics.

Generated outputs are stored in `/home/users/h/hes/scratch/projects/cosmicPV/`: the dated run folders hold catalogues, fields, tetrahedra, frozen source, and summaries; `basic/`, `intro_disp_PV/`, `field_PVdisp/`, and `field_PVdisp_A/` hold exported tables, figures, notebook history, and cached examples. The notebooks remain here in `results/`. All output-folder links are grouped in `results/saved/` and point to the scratch folders; notebook links use this layout. New submissions also write directly to scratch; the location is defined in `src/parity_study/paths.py`.

Choose **Run All** to refresh from saved measurements and cached examples, then save the notebook to retain its new outputs. This performs no catalogue generation or Delaunay triangulation. Partial ensembles display their completed/planned counts. The small illustrations use the normalized RMS convention; the dated catalogue ensembles retain their original per-mode amplitudes.

To refresh notebooks from the command line:

```bash
.venv/bin/python scripts/update_notebooks.py --all
.venv/bin/python scripts/update_notebooks.py --run results/saved/20261121_baseline
```

The refresh script preserves `basic.ipynb`, `intro_PVdisp.ipynb`, and `field_PVdisp.ipynb` for the active studies. The fiducial field report uses `20261121_mesh_ic_L1000_N256_fixedIC_10`; historical field tests, including the four-seed varying-density pilot, do not generate separate notebooks. Other new runs receive `results/<run>.ipynb`. The usual `python -m parity_study.summarize --run results/saved/<run>` command also refreshes the corresponding notebook. Use `--no-notebook` for just JSON/CSV/PNG files, or add `--overview` to `update_notebooks.py` for an optional index.

Automatic refresh reads and executes existing notebooks without replacing source cells or research notes. Concurrent edits stop the refresh without being overwritten. Use `--rebuild` explicitly to adopt a new template; the old notebook is archived in its matching output folder's `notebook_history/`.

The current introduction uses `parity_study.tests.intro_fiducial` and `slurm/intro_fiducial.sh`. It reuses production seed 1000 at L=1000 and A=16, verifies all three example particle maps against the saved catalogues, and plots the current `H(k)=P_delta(k)*exp[-(k R_h)^2]/k^4` spectrum with discrete-mode normalization. Figures, arrays, per-mode weights, RMS tables, frozen code and the previous notebook are saved in scratch `20261123_intro_PVdisp_L1000_A16/`, linked under `results/saved/`. The pointer `intro_disp_PV/active_example.json` makes ordinary refreshes and template rebuilds use this fiducial introduction. Notebook **Run All** reads cached arrays and regenerates plots without rebuilding fields or triangulations. The older small top-hat examples and their generator `tutorial_examples` remain available for historical reproduction.

## Fiducial Gaussian mesh initial conditions

The new fiducial model is `field.model="mesh_za_helical"`, following the initial-condition construction in [Shim et al., arXiv:2406.06080v2, Eqs. 1–8](https://arxiv.org/html/2406.06080v2). It builds a real Gaussian density mesh, derives `Psi_ZA(k)=i*k*delta(k)/k**2`, adds an independent transverse helical displacement, and evaluates their sum at a regular cell-centred particle lattice: `x=(q+Psi_ZA(q)+Psi_hel(q)) % L`. Both terms use the same Lagrangian coordinates. This generates initial conditions only; it performs no N-body evolution. The paper's detection uses spin observables, so it does not establish detectability with our tetrahedron χ̂ statistic.

The selected toy density spectrum is `k**index * exp(-(k*smoothing)**2)`, normalized over active modes to ensemble density RMS `sigma_delta`. The helical displacement-power shape is `P_delta(k)*exp(-(k*helical_smoothing)**2)/k**4`, normalized to unit total power and multiplied by `A**2`. We call A the **displacement strength**; its normalization is the ensemble RMS of the helical component. Total displacement includes ZA as well. Neither field is rescaled to fix an individual realization's RMS. Zero and unpaired Nyquist modes are excluded. Particles evaluate the mesh Fourier series with NUFFT to avoid interpolation-induced smoothing.

```python
from parity_study.initial_conditions import build_ic_catalogue
cat = build_ic_catalogue(L=100., N=16, particle_grid=16, Ngal=4096,
                         sigma_delta=.2, helical_smoothing=6.25,
                         spectrum={"kind":"power_law", "index":0., "smoothing":6.25},
                         A=2., eps=1., density_seed=0, seed=0)
```

`Ngal` must equal `particle_grid**3`. Matching seeds preserves the density field, particle lattice, and helical draws across amplitudes and helicities. Nulls are `null_za` (A=0) and `null_symmetric` (equal helical power). New production caches retain the Gaussian density mesh, lattice coordinates, and evaluated ZA/positive/negative displacement vectors. Full Fourier coefficients and vector meshes remain available through `mesh_fields(full_output=True)` for diagnostics, but are not duplicated in large production caches. Earlier full caches remain readable. The ZA displacement also supplies the direction of an eventual ZA velocity; this project does not yet assign cosmology-dependent physical velocities.

Set `field.density_seed=0` to hold the Gaussian density and ZA displacement fixed while `seed` varies only the helical fields. Omitting `density_seed` preserves the older convention in which density and helical fields both vary with seed. The ten-seed fixed-IC run uses seeds 0–9 and amplitudes 0.5 and 2; its ZA-only reference is measured once. SEMs describe displacement scatter at that fixed IC, excluding density cosmic variance; the fixed-IC control means are not assumed to have zero expectation. `scripts/check_fixed_ic_ensemble.py` checks the saved density and lattice for exact equality, ZA agreement to 1e-12, distinct helical fields, and every saved particle map.

For mesh ensembles, top-level `seed_start` selects a new contiguous range of actual RNG seeds (default 0). The selection confirmation sets `seed_start=10`, not merely new output labels: field generation, filenames, metadata, completeness checks and resume all use seeds 10–19. Submissions with `selection_protocol` run the appropriate fixed-IC or independent-IC audit and frozen selection analysis in their summary stage without creating another notebook.

`scripts/reuse_fixed_ic_inputs.py --source <old-run> --run <new-prepared-run>` can reuse compatible density and helical components after checking their configurations and manifests. It records source hashes and reuses a catalogue only when its density and helicity seeds match and its particle positions reproduce exactly. The old four-seed run remains intact; its first four helical draws can be reused, but only its seed-zero catalogues share the new fixed density.

```bash
# Small construction/measurement check (also the default configuration):
.venv/bin/python scripts/submit.py --config logs/configs/ic-smoke.json --name 20261121_mesh_ic_smoke
# Prepared larger experiment; submit separately when ready:
.venv/bin/python scripts/submit.py --config logs/configs/ic-fiducial.json
# Larger-box technical pilot:
.venv/bin/python scripts/submit.py --config logs/configs/ic-L1000-N256.json --name 20261121_mesh_ic_L1000_N256_v2
# Ten helical draws at fixed density seed 0:
.venv/bin/python scripts/submit.py --config logs/configs/ic-L1000-N256-fixedIC-10.json --name 20261121_mesh_ic_L1000_N256_fixedIC_10
```

The completed larger-box run writes the single fiducial report `results/field_PVdisp.ipynb`, with figures and tables in scratch `cosmicPV/field_PVdisp/`. The original small-box report is archived. `ic-fiducial.json` is an additional prepared scan, not automatically submitted. Vary `A`, `helical_smoothing`, `sigma_delta`, the density spectrum, and mesh/particle resolution using distinct frozen configurations. The amplitude correction within one run does not account for selecting across multiple configurations; candidates from a parameter search require independent confirmation seeds.

The fiducial L=1000/N=256 test uses 32³ particles, limiting the active band to `kmax=pi*32/1000`. A, sigma_delta, and the absolute smoothing lengths match the earlier small pilot, so their ratios to box size and particle spacing change. Validation and reporting resources can be overridden independently with `scheduler.validate` and `scheduler.summary`; this mesh test requests 48 GB for field-generation jobs.

For a tabulated shape, set `spectrum={"kind":"table", "path":"Pk.txt"}` (two columns k and P, path relative to the configuration). Submission embeds the table and its source hash in the frozen configuration. Positive P values must cover all active k; interpolation is logarithmic and extrapolation is rejected. Its overall amplitude is replaced by `sigma_delta`. Inline `k` and `p` arrays are also accepted.

Earlier saved ensembles remain the `poisson_helical` construction without a Gaussian density/ZA component; their results do not test these new ICs. The existing `build_catalogue` API preserves that construction for reproducibility.

## Displacement amplitude

New Python catalogue calls default to `amplitude_convention="rms"`. The top-hat spectrum is normalized over the active three-dimensional Fourier modes (after the zero/Nyquist exclusions):

```text
P_normalized(k) = P_shape(k) / sum_over_active_modes(P_shape)
P_displacement(k) = A**2 * P_normalized(k)
E[|psi(x)|**2] = A**2
```

Thus A is the ensemble RMS displacement magnitude, in the same length units as L. The response parameter multiplies this by `abs(response)`. This describes displacements before periodic wrapping. The vector mean is zero; the mean magnitude and each realization's measured RMS are different quantities. Individual realizations are not rescaled to force their measured RMS to A.

```python
from parity_study.catalogue import build_catalogue
cat = build_catalogue(A=20., L=1000., N=64, Ngal=10000, eps=1., seed=42)
# Expected displacement RMS is 20 catalogue length units.
```

For the current flat spectrum, the mode weights are `1/N_active_modes`. The cached unit fields have expected RMS 1; multiplying them by A implements the A-squared power without recalculating the fields. At fixed A, changing resolution or the occupied band changes the spatial correlations but keeps the expected total displacement variance fixed.

The explicit `amplitude_convention="notebook"` option reproduces the old definition, `P_displacement(k)=A*band_mask(k)`. For identical mode masks, its conversion is `A_rms=sqrt(A_notebook*N_active_modes)`. Existing submitted runs retain that old convention in their frozen source. Newly submitted configurations must set `field.amplitude_convention` explicitly to `"rms"` or `"notebook"`; old field caches are rejected if used with the new convention. Summaries label the amplitude convention.

## Layout

| Path | Purpose |
| --- | --- |
| `tp4a-pv-delaunay/` | Original student repository and saved notebook |
| `src/parity_study/` | Reusable field and catalogue construction, Delaunay measurement, run/storage utilities and summaries |
| `src/parity_study/tests/` | Specific scientific studies, selection/confirmation analyses, figures, tutorial examples and notebook generators |
| `configs/` | Explicit experiment parameters and resource requests |
| `vendor/DIVE/` | Student's unmodified DIVE repository, pinned by `vendor/sources.json` |
| `patches/dive-parity.patch` | Five-column output, unique periodic cells, precision, tie handling |
| `scripts/` | Environment/build setup, submission, resume |
| `slurm/job.sh` | Batch entry point for validation, seed blocks, and summaries |
| `tests/` | Numerical and geometry regression checks |
| `results/basic.ipynb`, `results/intro_PVdisp.ipynb` | Question-led foundations and displacement introduction |
| `results/saved/basic/`, `results/saved/intro_disp_PV/`, `results/saved/field_PVdisp/` | Links to exported figures, tables, conclusions, and cached examples in scratch |
| `results/<run>.ipynb` | Executed report: test questions, expectations, measurements, and conclusions |
| `results/saved/<run>/` | Link to the run's scratch folder: frozen source, parameters, logs, and measurements |
| `reports/notebook-review.md` | Scientific and implementation audit |

The main physical implementation is in `initial_conditions.py` (Gaussian density, ZA and helical fields), `catalogue.py` (Fourier fields and particle displacements), and `delaunay.py` (tetrahedra and chirality). Shared execution and I/O helpers also remain at the package root.

`geometry.py` defines parity-even six-edge shape and physical size. `parity_stat(..., geometry=True)` appends geometric volume and six sorted edge lengths to the existing five columns, using DIVE's optional `-g` flag. Volume is retained even when exact vertex-ordering ties give zero chirality. The six-edge quality is `Q = 12*(3*V)**(2/3)/sum(l_ij**2)`, with Q=1 for a regular tetrahedron. `tests/selection_geometry.py` compares fixed Q bins and the original ratio cut at 32,768, 16,384 and 8,192 particles, using nested identical particle indices across helicities. Output, frozen code, input hashes and plots are in scratch `selection_geometry_20261122/`, linked under `results/saved/`. These reused ICs support an exploratory comparison, not fresh confirmation or a lattice-resolution convergence claim.

The completed ten-IC thinning pilot reconstructed 120 catalogues; all 62 regression tests passed. The original cut's signal-to-noise gain relative to all tetrahedra falls from 1.54 to 0.55 to 0.19 as particle count decreases. No tested fixed Q bin improves sensitivity at all three densities. The predefined bin `0.4 <= Q < 0.6` is a promising post-hoc compromise: its mean matched differences are −0.05478 ± 0.00635, −0.02954 ± 0.00257, and −0.00854 ± 0.00165 SEM. It retains 0.23%, 9.53%, and 22.33% of tetrahedra, respectively. Highlighting it by the largest minimum absolute mean-to-SEM ratio was an additional exploratory interpretation, recorded in `interpretation.json`; it requires fresh confirmation before replacing the fiducial cut. The scientific snapshot is preserved, with report-only plot fixes and interpretation recorded under `report_code/`.

`tests/selection_bins.py` postprocesses saved tetrahedra into fixed-width (0.02) bins of both the three-reference-edge ratio `r` and six-edge quality `Q`. The 50-realization extension uses IC seeds 1000–1049 from the existing A16 confirmation run: ten geometry realizations are verified and reused, and forty are measured from saved positions. The report `results/field_PVdisp_selection.ipynb` displays histogram-style bars of the equal-realization mean `D_B = S_B(+1) - S_B(-1)` over the complete matched pairs available in each bin. SEM is the sample standard deviation of those paired differences divided by `sqrt(n_pairs)`. Small numbers label bins with fewer than 50 pairs; only bins with no complete pair are hatched. A single pair has a mean but no estimated SEM. The original r cut, highlighted Q interval, thinning fractions, and bin widths remain fixed. This combined response map includes the original exploratory sample.

The expanded run, job logs and frozen code are in scratch `selection_geometry_20261122_50/`, linked under `results/saved/`. The current `fine_bins_available_pairs/` report contains counts, per-realization signals, PNG/PDF plots, and the executed notebook. The earlier report requiring all 50 pairs remains in `fine_bins/`, and the ten-realization results remain in `selection_geometry_20261122/fine_bins/`. For another ensemble size, use `parity_study.tests.selection_geometry prepare --n-seeds N`, then `reuse` for compatible saved geometry, `measure` for missing seeds, and `report-bins` to build the fine-bin report. Notebook counts, SEM definitions, labels and output links are derived from the run configuration.

`tests/selection_comparison.py` compares the wider `0.4 <= Q < 0.8` interval with `0.4 <= Q < 0.6`, the frozen r cut, and all tetrahedra on the same 50 ICs. It combines disjoint-bin chirality sums and counts within each catalogue before forming matched differences, and checks the union directly against 18 saved geometry catalogues. At 32,768, 16,384 and 8,192 particles, widening Q retains 157.26, 7.07 and 3.25 times as many tetrahedra, while the absolute mean-to-SEM ratio changes by factors of 6.45, 1.11 and 0.78. Extra counts therefore do not guarantee greater sensitivity. Section 4 of `field_PVdisp_selection.ipynb` shows the comparison; its frozen code, counts, per-catalogue signals and plots are saved in `selection_geometry_20261122_50/Q_wide_comparison/`. This interval was chosen after inspecting the ensemble and remains exploratory; the r cut remains fiducial.

Section 4 of `field_PVdisp_B.ipynb` adds catalogue-signal histograms and Gaussian curves for the adopted fiducial Q interval `0.4 <= Q < 0.8`, with 50 signals per helicity at 32,768 particles. `tests/confirmation_q.py` reuses the saved selection measurements, verifies the 150 catalogue identities against the original confirmation arrays, and checks three epsilon=0 catalogues directly against saved geometry. Its frozen inputs, code, signals and PNG/PDF plots are in scratch `field_PVdisp_B/Q_0p4_0p8/`. The original frozen-r confirmation remains in Sections 1–3; the Q section describes a selection made using this same ensemble.

`tests/helicity_half.py` runs the intermediate helicities `epsilon=-0.5,+0.5` on 50 additional independent IC seeds 1050–1099. The two cases share density, ZA, lattice and helical draws within each seed, with A=16 and the Q interval frozen before production. Each job saves its fields, positions and 12-column Delaunay geometry. Slurm validation → measurement array → summary verifies reflection symmetry, periodic volumes, independent fields and matched particle maps. The separate `tests/helicity_half_notebook.py` report combines these distributions with the earlier epsilon=-1,0,+1 reference curves, preserving the distinction between the two seed groups. Outputs and job IDs are in `results/saved/20261123_mesh_ic_L1000_N256_A16_epshalf_50/`; the report updates Section 4 of `field_PVdisp_B.ipynb` only after all 50 new pairs pass the ensemble audit.

The half-helicity run is complete: 100 new catalogues, all 17 targeted tests and selected-parity checks passed, and the final audit verified 50 distinct density/ZA/helical draws and all matched particle maps. The Q-selected mean difference is −0.024863 ± 0.000471 SEM, about 49.5% of the earlier full-helicity difference. Section 4 now displays all five helicities, with 50 catalogue signals per curve; CSV/NPZ measurements, Gaussian parameters, paired summaries and exported figures are in the run's `report/` folder.

`results/field_PVdisp_Pk.ipynb` compares the ordinary density power of five saved catalogues per helicity, 25 total: seeds 1000–1004 for epsilon=-1,0,+1 and 1050–1054 for epsilon=-0.5,+0.5. Reusable `power_spectrum.py` computes nonzero Fourier sums with FINUFFT and the normalization `P=V*mean_modes(|sum(exp(-ik.x))/N_p|^2)`. The study driver `tests/catalogue_power.py` and batch job `slurm/catalogue_power.sh` save shell spectra, matched differences, plots and the executed notebook in scratch `20261123_catalogue_power_A16_five_each/`, linked under `results/saved/`. This measures all particles, without a tetrahedron cut. There is no mass-assignment window or Poisson subtraction; V/N_p is only a reference because the particle load is a displaced lattice. Both conjugate modes are counted, while SEM uses the five independent catalogues/pairs. The report separates the input Fourier band from the higher-k discreteness diagnostic and explains why parity-even density power cannot identify the sign of helicity. Checks cover direct Fourier sums, reflection/translation invariance, volume units, a known density mode, and Poisson normalization.

The current power notebook focuses on the **original density spectrum versus the density spectrum after adding helical displacement**. `tests/catalogue_power_baseline.py` reuses those 25 final spectra, measures the ten actual saved Gaussian meshes, and constructs their matched ZA-only coordinates from cached `q` and `za` without regenerating fields or triangulations. It shows `P_ZA+hel/P_input-1` and `P_ZA+hel/P_ZA-1`, forming each ratio within an IC before averaging five realizations. The baseline with no helical displacement has A=0; epsilon=0 at A=16 still has a transverse displacement. Mesh Fourier coefficients use the same normalization as the point spectra, checked by Parseval and a known linear density mode. Plotted ratios use complete shells within the input support; the broad-band summary includes all active modes. Configuration, baseline coordinates and spectra, comparisons, figures, validation and the previous notebook are saved in scratch `20261123_catalogue_power_matched_baseline_five_each/`, linked under `results/saved/`. The earlier helicity-sign comparison remains archived in its original run.

Study modules now use `parity_study.tests`: `amplitude_scan`, `selections`, `independent_ics`, and `best_confirmation`. Their plot and notebook modules live alongside them. For example:

```bash
.venv/bin/python -m parity_study.tests.amplitude_scan --help
.venv/bin/python -m parity_study.tests.best_confirmation --help
.venv/bin/python -m pytest tests
```

The top-level `tests/` directory contains automated regression tests. A package search-path fallback keeps the established `parity_study.<study_module>` imports in saved notebooks working. Frozen run snapshots retain their original layout; their data and executed notebooks are preserved.

## Environment and build

The current environment is `.venv` (Python 3.11). FINUFFT is installed locally; NumPy/SciPy/Matplotlib are inherited from the cluster Anaconda installation. Exact direct dependency versions are recorded in `requirements.txt` and in each run's manifest. The batch scripts limit BLAS threads and pass the allocated CPU count to FINUFFT.

On this cluster, rebuild with:

```bash
cd /home/users/h/hes/projects/2026_DT_Parity
bash scripts/setup.sh
```

`build_dive.sh` loads GCC 12.3.0, CGAL 5.6, CMake 3.26.3, GMP 6.2.1, MPFR 4.2.0, and Boost 1.82.0. It applies the patch to a build copy; `vendor/DIVE` remains pristine. The DIVE executable retains the upstream GPLv3 license. Its source, patch, and license are copied into every run.

## Submit experiments

```bash
.venv/bin/python scripts/submit.py --config configs/baseline.json
.venv/bin/python scripts/submit.py --config configs/notebook-current.json
```

Each submission creates a unique run directory in scratch and a link under project `results/saved/`, then freezes the configuration, Python source, tests, reference notebook, patched executable, and upstream C++ source. SHA-256 checks prevent a job from silently using altered run inputs. The student checkout and source revision are recorded. The Python environment is recorded but is not copied; keep `.venv` unchanged while jobs run.

The job dependency chain is validation → ensemble array → summary. Validation uses `debug-cpu`; ensembles use `public-cpu`. A validation failure cancels the dependent ensemble. Summary runs after the array ends, including when tasks fail, and reports missing realizations. Each array task handles five seeds; baseline allows eight concurrent tasks, and the larger-grid scan allows four.

The baseline has `L=1000`, `N=64`, and 10,000 points. It runs 200 seeds for each of three null cases (Poisson, weak symmetric displacement, and the notebook's original `L=1, A=1` null), 100 seeds for each of six amplitudes and both helicities, and 20 seeds for each of four chiral-molecule fractions: 1,880 saved catalogue measurements. It excludes unpaired even-grid Nyquist modes.

The current-notebook run uses `N=256`, 10,000 points, `A=5,10,50,100`, and 100 seeds for each helicity: 800 measurements. It preserves the notebook's Nyquist convention and fixes the box size. These amplitudes produce very large displacements and serve as a reproduction/stress test. The notebook's saved small-amplitude output was generated with different, incompletely recorded settings; the baseline is an explicit new experiment, not an exact recreation of those stale outputs.

## Monitor and resume

```bash
squeue -u "$USER"
cat results/saved/<run>/jobs.json
cat results/saved/<run>/validation/report.json
.venv/bin/python -m parity_study.summarize --run results/saved/<run>
.venv/bin/python scripts/resume.py --run results/saved/<run> --dry-run
.venv/bin/python scripts/resume.py --run results/saved/<run>
```

Replace `<run>` with a run directory name. Resume checks for live jobs, resubmits only incomplete seed blocks, and uses the frozen code. Existing result records and field caches are reused. A completed record is written atomically, so interrupted writes do not look complete. To change scientific parameters, create a new run.

## Saved data

`fields/main_<seed>.npz` stores the original positions and unit-amplitude maximum-positive/negative-helicity displacements. All amplitudes reuse these fields. General epsilon is formed as `sqrt((1+eps)/2)*plus + sqrt((1-eps)/2)*minus`, then multiplied by A in the RMS convention, or `sqrt(A)` in the notebook convention. The original-null reproduction uses a separate cache with the notebook convention.

`records/<case>/seed_<seed>.npz` contains:

- `catalogue`: positions in the explicitly recorded box;
- `tetrahedra`: columns `chi, chi_hat, l_min, l_mid, l_max`;
- `statistics`: catalogue means of `chi`, `chi_hat`, and `sign(chi)`;
- `metadata_json`: parameters, displacement RMS/spacing, tetrahedron count, timing, box size, and volume check;
- for positive controls, injected handedness and the identified-molecule mask.

```python
import json
import numpy as np
with np.load('results/saved/<run>/records/null_poisson/seed_0000.npz') as result:
    catalogue = result['catalogue']
    tetrahedra = result['tetrahedra']
    metadata = json.loads(str(result['metadata_json']))
```

`summary/` contains JSON completeness and statistical results; CSV files for individual realizations, ensemble statistics, paired helicity differences, and molecule controls; and PNG plots. `paired_chi_hat.csv` and `summary.json` → `primary_scan` provide the primary analysis, with local Student-t p-values and Bonferroni correction over all planned amplitudes (including unfinished amplitudes in partial runs). Main scan and null plots show χ̂. Uncertainties use independent seeds, never tetrahedron counts as independent samples. Paired differences use matching seeds. Cross-amplitude correlations remain in the saved realizations for any later joint test.

Raw tetrahedra and catalogues are retained for later shape/size analysis. Expect several GB for both configured studies together. Set `save_tetrahedra` to false only in a new configuration if raw tetrahedra are unnecessary.
