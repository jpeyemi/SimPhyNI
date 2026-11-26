"""
SimPhyNI result plots, built from the long results table (any number of traits).

    summary_plots(long, traits, outdir, q_col="pval_bh", alpha=0.05, top_n=40)

writes
    simphyni_heatmap.png      signed shrunken effect for the top_n most-connected traits,
                              clustered; cells with q >= alpha are blank, * marks q < alpha
                              and ** q < alpha/10
    simphyni_volcano.png      every pair: shrunken effect vs -log10 q (density for the bulk,
                              points for significant pairs; co-confined pairs outlined)
    simphyni_pvalues.png      histogram of raw p-values (should be flat apart from a peak
                              near 0 if there is signal) — a quick calibration check
    simphyni_top_traits.png   traits with the most significant partners (positive / negative)
    volcano_<trait>.png       per query trait (one-vs-rest runs, query_traits=[...])
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

POS, NEG, GREY = "#d1495b", "#2e86ab", "#bdbdbd"


def _effect(long):
    if "effect_shrunk" in long and long["effect_shrunk"].notna().any():
        return long["effect_shrunk"].to_numpy(), "shrunken effect (log-odds units)"
    return long["effect_z"].to_numpy(), "effect (z)"


def heatmap(long, out, q_col="pval_bh", alpha=0.05, top_n=40):
    sig = long[long[q_col] < alpha]
    if sig.empty:
        return None
    deg = pd.concat([sig["T1"], sig["T2"]]).value_counts()
    keep = list(deg.index[:top_n])
    sub = long[long["T1"].isin(keep) & long["T2"].isin(keep)]
    eff, elab = _effect(sub)
    M = pd.DataFrame(0.0, index=keep, columns=keep)
    Q = pd.DataFrame(1.0, index=keep, columns=keep)
    for a, b, e, q in zip(sub["T1"], sub["T2"], eff, sub[q_col]):
        M.loc[a, b] = M.loc[b, a] = e if q < alpha else 0.0
        Q.loc[a, b] = Q.loc[b, a] = q
    order = keep
    if len(keep) > 2:
        try:
            import warnings
            from scipy.cluster.hierarchy import leaves_list, linkage
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                order = [keep[k] for k in leaves_list(linkage(M.to_numpy(), "average"))]
        except Exception:  # noqa: BLE001
            pass
    M, Q = M.loc[order, order], Q.loc[order, order]
    vmax = max(np.nanmax(np.abs(M.to_numpy())), 1e-9)
    n = len(order)
    size = min(max(6, 0.28 * n + 2), 22)
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    data = np.ma.masked_where(M.to_numpy() == 0, M.to_numpy())
    cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("#f5f5f5")
    im = ax.imshow(data, cmap=cm, norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="nearest")
    fs = max(4, min(9, 220 // max(n, 1)))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=90, fontsize=fs); ax.set_yticklabels(order, fontsize=fs)
    if n <= 60:
        for r in range(n):
            for c in range(n):
                q = Q.iat[r, c]
                if r != c and q < alpha:
                    ax.text(c, r, "**" if q < alpha / 10 else "*", ha="center", va="center", fontsize=fs, color="black")
    cb = fig.colorbar(im, ax=ax, shrink=0.6); cb.set_label(elab + "\nred = co-occur, blue = avoid")
    ax.set_title(f"Significant associations, {n} most-connected traits\n"
                 f"blank: {q_col} \u2265 {alpha};  *: < {alpha};  **: < {alpha / 10:g}", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def volcano(long, out, q_col="pval_bh", alpha=0.05, title=None):
    eff, elab = _effect(long)
    y = -np.log10(np.clip(long[q_col].to_numpy(), 1e-300, 1))
    sig = long[q_col].to_numpy() < alpha
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ns = ~sig
    if ns.sum() > 20000:
        ax.hexbin(eff[ns], y[ns], gridsize=80, bins="log", cmap="Greys", mincnt=1, linewidths=0)
    else:
        ax.scatter(eff[ns], y[ns], s=6, c=GREY, alpha=0.5, edgecolors="none", rasterized=True)
    d = long["direction"].to_numpy()
    for dd, colr, lab in ((1, POS, "positive"), (-1, NEG, "negative")):
        m = sig & (d == dd)
        ax.scatter(eff[m], y[m], s=10, c=colr, alpha=0.8, edgecolors="none", label=f"{lab} ({m.sum():,})", rasterized=True)
    if "co_confined" in long:
        m = sig & long["co_confined"].to_numpy()
        if m.any():
            ax.scatter(eff[m], y[m], s=22, facecolors="none", edgecolors="black", linewidths=0.5,
                       label=f"co-confined ({m.sum():,})", rasterized=True)
    ax.axhline(-np.log10(alpha), color="grey", ls="--", lw=0.8)
    ax.set_xlabel(elab); ax.set_ylabel(f"-log10 {q_col}")
    ax.set_title(title or f"{len(long):,} pairs")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def pvalue_histogram(long, out):
    p = long["pval_naive"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p, bins=50, range=(0, 1), color="#6c757d", edgecolor="white")
    ax.axhline(len(p) / 50, color=POS, ls="--", lw=0.8, label="uniform (no signal)")
    ax.set_xlabel("raw p-value (pval_naive)"); ax.set_ylabel("pairs")
    ax.set_title("p-value distribution (flat with a peak near 0 is expected)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def top_traits(traits, out, n=30):
    t = traits.assign(tot=traits["n_significant_pos"] + traits["n_significant_neg"])
    t = t[t["tot"] > 0].sort_values("tot", ascending=False).head(n)
    if t.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, max(3, 0.25 * len(t) + 1)))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["n_significant_pos"], color=POS, label="positive")
    ax.barh(y, -t["n_significant_neg"], color=NEG, label="negative")
    ax.set_yticks(y); ax.set_yticklabels(t["trait"], fontsize=7)
    ax.axvline(0, color="black", lw=0.6)
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(int(v))}"))
    ax.set_xlabel("significant partners (left: negative, right: positive)")
    ax.set_title(f"Traits with the most significant associations (top {len(t)})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def summary_plots(long, traits, outdir, q_col="pval_bh", alpha=0.05, top_n=40, query_traits=None):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    made = [volcano(long, outdir / "simphyni_volcano.png", q_col, alpha),
            pvalue_histogram(long, outdir / "simphyni_pvalues.png"),
            top_traits(traits, outdir / "simphyni_top_traits.png")]
    if not query_traits:
        made.append(heatmap(long, outdir / "simphyni_heatmap.png", q_col, alpha, top_n))
    for tr in (query_traits or []):
        sub = long[(long["T1"] == tr) | (long["T2"] == tr)]
        if len(sub):
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(tr))
            made.append(volcano(sub, outdir / f"volcano_{safe}.png", q_col, alpha, title=f"{tr}: {len(sub):,} partners"))
    return [m for m in made if m is not None]
