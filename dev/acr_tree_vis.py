"""
acr_tree_vis.py — Interactive tree viewer comparing ancestral state reconstructions
from PastML JOINT, PastML MPPA, fast ACR (empirical), and fast ACR (ML).

All nodes (leaves + internal) are colored by inferred P(state=1).
Gene switching is fast: topology is drawn once; only scatter colors are updated.

Usage
-----
    python dev/acr_tree_vis.py \\
        --tree   tests/panx/ecoli_accessory.nwk \\
        --traits tests/panx/ecoli_accessory.csv \\
        [--gene  yahB]

Controls
--------
    ← / →            navigate genes
    ◀ / ▶  buttons   navigate genes
    hover a node     tooltip with P(1) from every method
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
from ete3 import Tree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "dev"))
sys.path.insert(0, str(REPO / "simphyni" / "scripts"))

from fast_binary_acr import build_tree_arrays, build_obs_matrix, fast_acr, marginal_probs_df
from run_ancestral_reconstruction import label_internal_nodes

try:
    from pastml.acr import acr as pastml_acr
    from pastml.ml import JOINT, MPPA
    PASTML_AVAILABLE = True
except ImportError:
    PASTML_AVAILABLE = False
    print("[WARN] pastml not importable — PastML panels will be empty")


# ---------------------------------------------------------------------------
# Colours / style
# ---------------------------------------------------------------------------

CMAP      = "RdBu_r"           # 0 → blue, 1 → red
BG        = "white"
TEXT      = "#222222"
TOPO_COL  = "#cccccc"
GRID_COL  = "#eeeeee"

PANELS = [
    ("pastml_joint",   "PastML JOINT"),
    ("pastml_mppa",    "PastML MPPA"),
    ("fast_empirical", "Fast ACR  empirical π₁"),
    ("fast_ml",        "Fast ACR  ML  π₁ + sf"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "axes.edgecolor": "#dddddd",
})


# ---------------------------------------------------------------------------
# Tree layout
# ---------------------------------------------------------------------------

def compute_tree_coords(ete_tree):
    node_x, node_y = {}, {}
    for node in ete_tree.traverse("preorder"):
        node_x[node.name] = 0.0 if node.is_root() else node_x[node.up.name] + max(node.dist, 0.0)

    def _y(node):
        if node.is_leaf():
            node_y[node.name] = float(_y.counter)
            _y.counter += 1
        else:
            for c in node.children:
                _y(c)
            ys = [node_y[c.name] for c in node.children]
            node_y[node.name] = (min(ys) + max(ys)) / 2.0
    _y.counter = 0
    _y(ete_tree)
    return node_x, node_y


def build_topology_segments(ete_tree, node_x, node_y):
    """Return branch segments as a list for LineCollection (drawn once)."""
    segs = []
    for node in ete_tree.traverse():
        if node.is_root():
            continue
        px, py = node_x[node.up.name], node_y[node.up.name]
        cx, cy = node_x[node.name],    node_y[node.name]
        segs.append([(px, cy), (cx, cy)])   # horizontal
    for node in ete_tree.traverse():
        if node.is_leaf():
            continue
        ys = [node_y[c.name] for c in node.children]
        segs.append([(node_x[node.name], min(ys)), (node_x[node.name], max(ys))])  # vertical
    return segs


# ---------------------------------------------------------------------------
# ACR runner
# ---------------------------------------------------------------------------

def run_all_acr(gene, trait_df, ete_tree, ta):
    out = {k: {} for k in ("pastml_joint", "pastml_mppa", "fast_empirical", "fast_ml")}
    out["params_emp"] = out["params_ml"] = (np.nan, np.nan, np.nan)

    # fast ACR
    if gene in trait_df.columns:
        obs = build_obs_matrix(ta, trait_df[[gene]])
        re  = fast_acr(ta, obs, [gene], mode="empirical")
        rm  = fast_acr(ta, obs, [gene], mode="ml")
        out["fast_empirical"] = marginal_probs_df(re, 0)["1"].astype(float).to_dict()
        out["fast_ml"]        = marginal_probs_df(rm, 0)["1"].astype(float).to_dict()
        out["params_emp"] = (float(re.sf[0]), float(re.pi1[0]), float(re.log_lh[0]))
        out["params_ml"]  = (float(rm.sf[0]), float(rm.pi1[0]), float(rm.log_lh[0]))

    # PastML
    if PASTML_AVAILABLE and gene in trait_df.columns:
        col = trait_df[gene].dropna()
        t   = ete_tree.copy()
        ann = col.astype(str).to_frame(name=gene)
        missing = {l.name for l in t.get_leaves()} - set(ann.index)
        if missing:
            t.prune([l for l in t.get_leaves() if l.name in ann.index],
                    preserve_branch_length=True)
        for node in t.traverse():
            if not node.is_root() and node.dist <= 0:
                node.dist = 1e-8
        if ann[gene].nunique() >= 2:
            try:
                pastml_acr(t, df=ann, prediction_method=JOINT, model="F81")
                d = {}
                for node in t.traverse():
                    v = getattr(node, gene, None)
                    if v is not None:
                        try:
                            s = next(iter(v)) if isinstance(v, (set, frozenset)) else str(v).split("|")[0]
                            d[node.name] = float(int(s))
                        except (ValueError, StopIteration):
                            pass
                out["pastml_joint"] = d
            except Exception as e:
                print(f"[WARN] JOINT: {e}")
            try:
                mres = pastml_acr(t, df=ann, prediction_method=MPPA, model="F81")
                mp   = mres[0]["marginal_probabilities"]
                if "1" in mp.columns:
                    out["pastml_mppa"] = mp["1"].astype(float).to_dict()
            except Exception as e:
                print(f"[WARN] MPPA: {e}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree",   default="tests/panx/ecoli_accessory.nwk")
    ap.add_argument("--traits", default="tests/panx/ecoli_accessory.csv")
    ap.add_argument("--gene",   default=None)
    args = ap.parse_args()

    tree_path   = REPO / args.tree
    traits_path = REPO / args.traits

    print(f"Loading tree:   {tree_path}")
    ete_tree = Tree(str(tree_path), format=1)
    label_internal_nodes(ete_tree)

    print(f"Loading traits: {traits_path}")
    trait_df    = pd.read_csv(traits_path, index_col=0)
    trait_names = list(trait_df.columns)
    n_leaves    = len(ete_tree.get_leaves())
    n_nodes     = len(list(ete_tree.traverse()))

    print(f"  {n_leaves} leaves, {n_nodes} nodes, {len(trait_names)} traits")
    print("Building tree arrays & warming up Numba ...")
    ta  = build_tree_arrays(ete_tree)
    obs_w = build_obs_matrix(ta, trait_df.iloc[:, :1])
    fast_acr(ta, obs_w, [trait_names[0]], mode="empirical")
    fast_acr(ta, obs_w, [trait_names[0]], mode="ml")
    print("  done\n")

    node_x, node_y = compute_tree_coords(ete_tree)
    topo_segs      = build_topology_segments(ete_tree, node_x, node_y)

    # Node arrays (fixed order, used for all scatter plots)
    all_names = [n.name for n in ete_tree.traverse()]
    all_xs    = np.array([node_x[nm] for nm in all_names])
    all_ys    = np.array([node_y[nm] for nm in all_names])
    leaf_mask = np.array([ete_tree.search_nodes(name=nm)[0].is_leaf()
                          for nm in all_names])

    x_max     = all_xs.max() * 1.08
    init_arr  = np.full(len(all_names), 0.5)   # mid-colormap placeholder until first gene loads

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig_h = max(9, n_leaves * 0.22 + 3)
    fig   = plt.figure(figsize=(24, fig_h), facecolor=BG)

    outer   = gridspec.GridSpec(2, 1, figure=fig,
                                height_ratios=[1, 0.055], hspace=0.03)
    top_gs  = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0],
                                               width_ratios=[1, 1, 1, 1, 0.04],
                                               wspace=0.04)
    ctrl_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1],
                                               wspace=0.15)

    axes    = [fig.add_subplot(top_gs[i]) for i in range(4)]
    ax_cbar = fig.add_subplot(top_gs[4])
    ax_prev = fig.add_subplot(ctrl_gs[0])
    ax_info = fig.add_subplot(ctrl_gs[1:4])
    ax_next = fig.add_subplot(ctrl_gs[4])

    for ax in [ax_prev, ax_info, ax_next, ax_cbar]:
        ax.set_facecolor(BG)

    # ------------------------------------------------------------------
    # Draw topology + create scatter once per panel (axes never cleared)
    # ------------------------------------------------------------------
    cmap_obj = plt.get_cmap(CMAP)
    cmap_obj.set_bad(color="#e8e8e8", alpha=0.5)   # NaN → light grey

    scatters = []
    for i, (method_key, title) in enumerate(PANELS):
        ax = axes[i]
        ax.set_facecolor(BG)
        ax.set_xlim(-0.01 * x_max, x_max)
        ax.set_ylim(n_leaves - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#dddddd")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Topology as single LineCollection
        lc = LineCollection(topo_segs, colors=TOPO_COL, lw=0.55, zorder=1)
        ax.add_collection(lc)

        # Scatter — init with 0.5 (visible grey-ish placeholder); updated in place on gene load
        sc = ax.scatter(all_xs, all_ys, c=init_arr.copy(),
                        cmap=cmap_obj, vmin=0.0, vmax=1.0,
                        s=20, zorder=3, edgecolors="none",
                        linewidths=0)
        scatters.append(sc)

        ax.set_title(title, color=TEXT, fontsize=9, pad=5, fontweight="bold")

    # ------------------------------------------------------------------
    # Single shared colorbar
    # ------------------------------------------------------------------
    cbar = fig.colorbar(scatters[-1], cax=ax_cbar)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])
    cbar.ax.tick_params(labelsize=7, labelcolor=TEXT, length=3)
    cbar.ax.set_ylabel("P(state = 1)", fontsize=7, color=TEXT, labelpad=6)
    cbar.outline.set_edgecolor("#cccccc")

    # ------------------------------------------------------------------
    # Control bar
    # ------------------------------------------------------------------
    for ax in [ax_prev, ax_info, ax_next]:
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    info_txt = ax_info.text(0.5, 0.5, "", transform=ax_info.transAxes,
                            color=TEXT, fontsize=11, ha="center", va="center",
                            fontweight="bold")

    btn_prev = Button(ax_prev, "◀  Prev",  color="#f0f0f0", hovercolor="#dde5ff")
    btn_next = Button(ax_next, "Next  ▶",  color="#f0f0f0", hovercolor="#dde5ff")
    for btn in (btn_prev, btn_next):
        btn.label.set_color(TEXT)
        btn.label.set_fontsize(10)

    # ------------------------------------------------------------------
    # Hover annotation (floating, figure-space)
    # ------------------------------------------------------------------
    hover_box = dict(boxstyle="round,pad=0.5", fc="white", ec="#aaaaaa",
                     alpha=0.95, lw=0.8)
    hover_annot = fig.text(0.5, 0.5, "", visible=False,
                            bbox=hover_box, fontsize=8,
                            fontfamily="monospace", color=TEXT,
                            transform=fig.transFigure, zorder=200,
                            va="bottom", ha="left")

    def on_hover(event):
        if event.inaxes not in axes:
            if hover_annot.get_visible():
                hover_annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        # Use the hovered axis's own transform so pixel coords are correct
        xy_disp = event.inaxes.transData.transform(np.column_stack([all_xs, all_ys]))
        dists   = np.hypot(xy_disp[:, 0] - event.x, xy_disp[:, 1] - event.y)
        idx     = int(np.argmin(dists))

        if dists[idx] > 22:
            if hover_annot.get_visible():
                hover_annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        node_name = all_names[idx]
        gene      = trait_names[state["idx"]]
        acr_data  = cache.get(gene, {})

        lines = [f" {node_name} "]
        lines.append("─" * (max(len(node_name), 28) + 2))
        for mk, title in PANELS:
            val  = acr_data.get(mk, {}).get(node_name, np.nan)
            bar  = ("█" * int(round(val * 8))).ljust(8) if np.isfinite(val) else "·" * 8
            vstr = f"{val:.3f}" if np.isfinite(val) else "  n/a"
            lines.append(f" {title[:24]:<24s} {vstr}  {bar}")

        # Position in figure fraction — nudge right/up from cursor, clamp at edges
        cw, ch = fig.canvas.get_width_height()
        fx = min(event.x / cw + 0.012, 0.83)
        fy = min(event.y / ch + 0.015, 0.92)
        hover_annot.set_position((fx, fy))
        hover_annot.set_text("\n".join(lines))
        hover_annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    # ------------------------------------------------------------------
    # Gene update (fast: only set_array, no redraw of topology)
    # ------------------------------------------------------------------
    cache = {}
    state = {"idx": 0}

    def _param_line(method_key, acr_data):
        if method_key == "fast_empirical":
            sf, pi1, lh = acr_data.get("params_emp", (np.nan,)*3)
        elif method_key == "fast_ml":
            sf, pi1, lh = acr_data.get("params_ml", (np.nan,)*3)
        else:
            return ""
        if np.isfinite(lh):
            return f"sf={sf:.1f}  π₁={pi1:.3f}  lh={lh:.1f}"
        return ""

    def show_gene(idx):
        gene = trait_names[idx]
        info_txt.set_text(f"[ {idx + 1} / {len(trait_names)} ]   {gene}")

        if gene not in cache:
            print(f"Computing ACR for '{gene}' ...", end="  ", flush=True)
            cache[gene] = run_all_acr(gene, trait_df, ete_tree, ta)
            print("done")

        acr_data = cache[gene]

        for i, (method_key, title) in enumerate(PANELS):
            p1_map = acr_data.get(method_key, {})
            vals   = np.array([p1_map.get(nm, np.nan) for nm in all_names])
            scatters[i].set_array(vals)

            sub = _param_line(method_key, acr_data)
            label = f"{title}\n{sub}" if sub else title
            axes[i].set_title(label, color=TEXT, fontsize=9, pad=5,
                              fontweight="bold")

        fig.suptitle(gene, color=TEXT, fontsize=13, y=0.998, fontweight="bold")
        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    start = args.gene if args.gene in trait_names else trait_names[0]
    state["idx"] = trait_names.index(start)

    def on_prev(_):
        state["idx"] = max(0, state["idx"] - 1)
        show_gene(state["idx"])

    def on_next(_):
        state["idx"] = min(len(trait_names) - 1, state["idx"] + 1)
        show_gene(state["idx"])

    def on_key(event):
        if event.key == "right":
            on_next(None)
        elif event.key == "left":
            on_prev(None)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    fig.canvas.mpl_connect("key_press_event", on_key)

    show_gene(state["idx"])
    print("Controls:  ← / →  arrow keys  or  ◀ / ▶ buttons  to navigate genes")
    print("           Hover any node for P(1) tooltip from all methods")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    main()
