"""Direct chi_hat signals and histograms from saved matched catalogues."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

EPSILONS = (-1, 0, 1)
COLORS = ("#2166ac", "#666666", "#b2182b")
LINESTYLES = ("-", "--", ":")


def epsilon_comparison(run, *, A=None, bins=40):
    """Compare fixed-A epsilon=-1,0,+1 using their common saved seeds.

    Every catalogue has equal weight in the tetrahedron-density histogram.
    No independence assumption is made for tetrahedra within a catalogue.
    """
    run = Path(run)
    config = json.loads((run/"config.json").read_text())
    A = float(config["null"]["A"] if A is None else A)
    if A != float(config["null"]["A"]):
        raise ValueError("The saved epsilon=0 control is available only at null.A")
    cases = (f"scan_A{A:g}_minus", "null_symmetric", f"scan_A{A:g}_plus")
    groups = [{int(p.stem.split("_")[-1]): p
               for p in (run/"records"/case).glob("seed_*.npz")} for case in cases]
    seeds = sorted(set(groups[0]).intersection(*groups[1:]))
    if not seeds:
        return None
    signals = np.empty((3, len(seeds)))
    counts = np.empty((3, len(seeds), bins), dtype=np.int64)
    ntet = np.empty((3, len(seeds)), dtype=np.int64)
    edges = np.linspace(-1., 1., bins+1)
    catalogue_rows = []
    for i, (eps, case, group) in enumerate(zip(EPSILONS, cases, groups)):
        for j, seed in enumerate(seeds):
            path = group[seed]
            with np.load(path) as z:
                meta = json.loads(str(z["metadata_json"]))
                if (meta["seed"] != seed or meta["parameters"]["A"] != A
                        or meta["parameters"]["eps"] != eps):
                    raise ValueError(f"Record does not match the fixed-A epsilon comparison: {path}")
                chi_hat = z["tetrahedra"][:, 1]
                value = float(z["statistics"][list(z["statistic_names"]).index("chi_hat")])
            if (not np.isfinite(chi_hat).all() or np.any(abs(chi_hat) > 1+1e-10)
                    or not np.isclose(chi_hat.mean(), value, rtol=0, atol=1e-12)):
                raise ValueError(f"Inconsistent saved chi_hat values: {path}")
            signals[i, j] = value
            ntet[i, j] = len(chi_hat)
            counts[i, j] = np.histogram(np.clip(chi_hat, -1, 1), bins=edges)[0]
            catalogue_rows.append(dict(A=A, epsilon=eps, seed=seed, S=value,
                                       n_tetrahedra=len(chi_hat), case=case))
    density = (counts/ntet[..., None]/np.diff(edges)).mean(axis=1)
    means = signals.mean(axis=1)
    sems = signals.std(axis=1, ddof=1)/np.sqrt(len(seeds)) if len(seeds)>1 else None
    rows = [dict(A=A, epsilon=eps, n=len(seeds), mean=float(means[i]),
                 sem=float(sems[i]) if sems is not None else None)
            for i, eps in enumerate(EPSILONS)]
    return dict(A=A, epsilons=np.array(EPSILONS), seeds=np.array(seeds),
                signals=signals, tetrahedron_counts=ntet, histogram_counts=counts,
                histogram_edges=edges, histogram_density=density,
                catalogue_rows=catalogue_rows, summary_rows=rows)


def direct_signal_figures(data):
    """Return a direct epsilon plot and two unstandardized histograms."""
    eps = data["epsilons"]
    signals = data["signals"]
    n = len(data["seeds"])
    means = signals.mean(axis=1)
    sems = signals.std(axis=1, ddof=1)/np.sqrt(n) if n>1 else None
    labels = [rf"$\epsilon={e:+d}$" if e else r"$\epsilon=0$" for e in eps]
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    for j in range(n):
        ax.plot(eps, signals[:, j], color="0.8", lw=.7, zorder=1)
    for i, color in enumerate(COLORS):
        ax.scatter(np.full(n, eps[i]), signals[i], color=color, alpha=.65, s=25,
                   label="Individual seeds" if i==0 else None, zorder=2)
    ax.errorbar(eps, means, yerr=sems, fmt="kD", capsize=5, ms=6,
                label="Mean ± 1 SEM" if n>1 else "Mean", zorder=3)
    ax.axhline(0, color="k", lw=.7, alpha=.5)
    ax.set(xticks=eps, xticklabels=["−1", "0", "+1"], xlabel=r"$\epsilon$",
           ylabel=r"$S_s(\epsilon)=\langle\widehat{\chi}\rangle_{\mathrm{tet}}$",
           title=f"Direct signal · A={data['A']:g} · {n} matched seeds")
    ax.legend(fontsize=9)

    hist, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    seed_edges = np.histogram_bin_edges(signals.ravel(), bins="auto")
    for i, (label, color, style) in enumerate(zip(labels, COLORS, LINESTYLES)):
        axes[0].stairs(data["histogram_density"][i], data["histogram_edges"],
                       color=color, label=label, lw=1.9, linestyle=style)
        axes[1].hist(signals[i], bins=seed_edges, histtype="step", color=color,
                     label=label, lw=1.9, linestyle=style)
        # Rug marks show the individual values even in a four-seed pilot.
        axes[1].plot(signals[i], np.full(n, .025+.035*i), "|", color=color,
                     transform=axes[1].get_xaxis_transform(), markersize=7)
    axes[0].set(xlabel=r"$\widehat{\chi}$", ylabel="Probability density",
                title="Tetrahedron chirality")
    axes[1].set(xlabel=r"$S_s(\epsilon)=\langle\widehat{\chi}\rangle_{\mathrm{tet}}$",
                ylabel="Number of catalogues", title=f"Catalogue means · {n} seeds per ε")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes:
        ax.axvline(0, color="k", lw=.7, alpha=.4)
        ax.legend(fontsize=9)
    hist.suptitle(f"Displacement strength A={data['A']:g}")
    return {"direct_chi_hat_by_epsilon": fig, "chi_hat_epsilon_histograms": hist}


def catalogue_signal_figure(data):
    """Show one catalogue mean per seed, with signal on the horizontal axis."""
    signals = data["signals"]
    n = len(data["seeds"])
    means = signals.mean(axis=1)
    sems = signals.std(axis=1, ddof=1)/np.sqrt(n) if n>1 else None
    rows = np.arange(len(data["epsilons"]))
    offsets = np.linspace(-.15, .15, n) if n>1 else np.zeros(1)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for i, color in enumerate(COLORS):
        ax.scatter(signals[i], rows[i]+offsets, color=color, s=35, alpha=.8,
                   label="One catalogue (one seed)" if i==0 else None, zorder=3)
    ax.errorbar(means, rows, xerr=sems, fmt="kD", capsize=5, ms=6,
                label="Mean ± 1 standard error" if sems is not None else "Mean",
                zorder=4)
    ax.axvline(0, color="k", lw=.8, alpha=.6)
    labels = [rf"$\epsilon={eps:+d}$" if eps else r"$\epsilon=0$"
              for eps in data["epsilons"]]
    ax.set(xlabel=r"Catalogue signal $S_s(\epsilon)=\langle\widehat{\chi}\rangle_{\mathrm{tet}}$",
           ylabel="Helicity setting", yticks=rows, yticklabels=labels,
           ylim=(len(rows)-.1, -.6),
           title=f"Catalogue signals · displacement strength A={data['A']:g}\n"
                 f"{n} catalogues per ε ({n} matched seeds)")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower center", fontsize=9)
    return fig


def probability_histogram_figure(data):
    """Show the three raw chi_hat probability densities on shared bins."""
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    for eps, density, color, style in zip(
            data["epsilons"], data["histogram_density"], COLORS, LINESTYLES):
        label = rf"$\epsilon={eps:+d}$" if eps else r"$\epsilon=0$"
        label += {-1: " (negative helicity)", 0: " (equal power)", 1: " (positive helicity)"}[int(eps)]
        ax.stairs(density, data["histogram_edges"], color=color,
                  linestyle=style, lw=1.9, label=label)
    ax.axvline(0, color="k", lw=.7, alpha=.4)
    ax.set(xlabel=r"Tetrahedron chirality $\widehat{\chi}$", ylabel=r"Probability density $p_\epsilon(\widehat{\chi})$",
           xlim=(-1, 1), title=f"Displacement strength A={data['A']:g} · {len(data['seeds'])} matched seeds")
    ax.legend()
    return fig


def histogram_difference(data):
    """Positive-minus-negative density contrast with matched-seed errors."""
    edges = data["histogram_edges"]
    per_seed = data["histogram_counts"]/data["tetrahedron_counts"][..., None]/np.diff(edges)
    plus = np.flatnonzero(data["epsilons"] == 1).item()
    minus = np.flatnonzero(data["epsilons"] == -1).item()
    differences = per_seed[plus] - per_seed[minus]
    n = len(data["seeds"])
    mean = differences.mean(axis=0)
    sem = differences.std(axis=0, ddof=1)/np.sqrt(n) if n>1 else None
    covariance = np.cov(differences, rowvar=False, ddof=1)/n if n>1 else None
    centers = (edges[:-1]+edges[1:])/2
    rows = [dict(chi_hat_left=float(edges[i]), chi_hat_right=float(edges[i+1]),
                 chi_hat_center=float(centers[i]), delta_p=float(mean[i]),
                 sem=float(sem[i]) if sem is not None else None, n=n)
            for i in range(len(centers))]
    return dict(A=data["A"], seeds=data["seeds"], histogram_edges=edges,
                bin_centers=centers, per_seed_difference=differences,
                mean=mean, sem=sem, mean_covariance=covariance, rows=rows)


def histogram_difference_figure(data):
    """Plot the paired density contrast with pointwise 1-SEM error bars."""
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    ax.stairs(data["mean"], data["histogram_edges"], color="#762a83", lw=1)
    ax.errorbar(data["bin_centers"], data["mean"], yerr=data["sem"], fmt="o",
                color="#762a83", ms=3, capsize=2,
                label="Mean difference ± 1 standard error" if data["sem"] is not None else "Mean difference")
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(0, color="k", lw=.6, alpha=.3)
    ax.set(xlabel=r"Tetrahedron chirality $\widehat{\chi}$",
           ylabel=r"Probability-density difference $\Delta p(\widehat{\chi})$",
           xlim=(-1, 1), title=r"Histogram difference: $\epsilon=+1$ minus $\epsilon=-1$"
           + f"\nDisplacement strength A={data['A']:g} · {len(data['seeds'])} matched seeds")
    ax.legend(fontsize=9)
    return fig
