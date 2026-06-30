"""
tree_vis.py — Interactive phylogenetic tree + trait heatmap viewer.

Usage:
    python tree_vis.py --tree tree.nwk --traits traits.csv [options]

    --tree      Path to Newick tree file
    --traits    Path to CSV (rows = taxa, cols = traits; first col = taxon names)
    --cmap      Matplotlib colormap (default: Blues)
    --window    Number of traits to display at once (default: 20)
    --vmin      Colorscale minimum (default: data min)
    --vmax      Colorscale maximum (default: data max)

    /Users/jpeyemi/Downloads/bench_0/synthetic_data.csv

    /Users/jpeyemi/Downloads/bench_0/tree.nwk
"""

import argparse
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, TextBox
import numpy as np
import pandas as pd
from Bio import Phylo


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Interactive phylo tree + trait heatmap")
    p.add_argument("--tree",   required=True,  help="Newick tree file")
    p.add_argument("--traits", required=True,  help="CSV trait file (taxa × traits)")
    p.add_argument("--cmap",   default="Blues", help="Matplotlib colormap name")
    p.add_argument("--window", type=int, default=20, help="Traits visible at once")
    p.add_argument("--vmin",   type=float, default=None, help="Colorscale min")
    p.add_argument("--vmax",   type=float, default=None, help="Colorscale max")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading & alignment
# ---------------------------------------------------------------------------

def load_data(tree_path: str, traits_path: str):
    tree = Phylo.read(tree_path, "newick")
    df   = pd.read_csv(traits_path, index_col=0)
    df.index = df.index.astype(str)

    leaves = [leaf.name for leaf in tree.get_terminals()]
    missing = set(leaves) - set(df.index)
    if missing:
        print(f"[warn] {len(missing)} tree leaves not in trait file — they will be NaN rows.")
        df = df.reindex(leaves)
    else:
        df = df.loc[leaves]

    return tree, df, leaves


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

COLORMAPS = ["Blues", "viridis", "plasma", "RdBu_r", "coolwarm", "YlOrRd", "Greens"]


def main():
    args = parse_args()

    print(f"Loading tree:   {args.tree}")
    print(f"Loading traits: {args.traits}")
    tree, df, leaves = load_data(args.tree, args.traits)

    n_leaves  = len(leaves)
    n_traits  = df.shape[1]
    win       = min(args.window, n_traits)
    cmap_name = args.cmap
    vmin      = args.vmin if args.vmin is not None else np.nanmin(df.values)
    vmax      = args.vmax if args.vmax is not None else np.nanmax(df.values)

    print(f"  {n_leaves} taxa | {n_traits} traits | window = {win}")

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, max(8, n_leaves * 0.25 + 3)))
    fig.patch.set_facecolor("#1a1a2e")

    # Row heights: controls row at bottom, main panels above
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1, 0.12],
        hspace=0.05,
    )
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=outer[0],
        width_ratios=[1.4, 3, 0.08],
        wspace=0.03,
    )
    ctrl_gs = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=outer[1],
        wspace=0.4,
    )

    ax_tree  = fig.add_subplot(top_gs[0])
    ax_heat  = fig.add_subplot(top_gs[1])
    ax_cbar  = fig.add_subplot(top_gs[2])
    ax_slide = fig.add_subplot(ctrl_gs[0:2])   # slider spans cols 0–1
    ax_cmapL = fig.add_subplot(ctrl_gs[2])     # prev cmap button
    ax_cmapR = fig.add_subplot(ctrl_gs[3])     # next cmap button

    for ax in [ax_tree, ax_heat, ax_cbar, ax_slide, ax_cmapL, ax_cmapR]:
        ax.set_facecolor("#1a1a2e")

    # -----------------------------------------------------------------------
    # Draw the tree
    # -----------------------------------------------------------------------
    Phylo.draw(tree, axes=ax_tree, do_show=False, label_func=lambda x: "")
    ax_tree.set_facecolor("#1a1a2e")
    ax_tree.tick_params(colors="gray")
    ax_tree.set_title("Phylogenetic Tree", color="white", fontsize=12, pad=6)
    ax_tree.set_xlabel("")
    ax_tree.set_ylabel("")
    # Phylo.draw puts y-ticks at 1..N — remove them; keep x for branch lengths
    ax_tree.set_yticks([])
    for sp in ax_tree.spines.values():
        sp.set_visible(False)
    ax_tree.xaxis.label.set_color("gray")
    ax_tree.tick_params(axis="x", colors="gray", labelsize=7)

    # Leaf labels on the right side of the tree panel (tight, small)
    ax_tree.set_ylim(n_leaves + 0.5, 0.5)   # top-to-bottom, matching heatmap
    y_ticks = np.arange(1, n_leaves + 1)
    ax_tree.set_yticks(y_ticks)
    ax_tree.set_yticklabels(leaves, fontsize=max(4, min(8, 120 // n_leaves)),
                            color="lightgray")
    ax_tree.yaxis.set_tick_params(length=0)
    ax_tree.yaxis.set_label_position("right")
    ax_tree.yaxis.tick_right()

    # -----------------------------------------------------------------------
    # Initial heatmap
    # -----------------------------------------------------------------------
    # extent: [xmin, xmax, ymax, ymin] — note y is flipped so row 0 is top
    def make_extent(start):
        end = start + win
        return [-0.5, end - start - 0.5, n_leaves + 0.5, 0.5]

    im = ax_heat.imshow(
        df.iloc[:, :win].values,
        aspect="auto",
        cmap=cmap_name,
        interpolation="nearest",
        extent=make_extent(0),
        vmin=vmin,
        vmax=vmax,
    )
    ax_heat.set_facecolor("#1a1a2e")
    ax_heat.set_ylim(n_leaves + 0.5, 0.5)
    ax_heat.set_yticks([])

    def set_xtick_labels(start):
        cols = df.columns[start: start + win]
        ticks = np.arange(len(cols))
        ax_heat.set_xticks(ticks)
        ax_heat.set_xticklabels(cols, rotation=60, ha="right",
                                 fontsize=max(5, min(9, 180 // win)),
                                 color="lightgray")

    set_xtick_labels(0)
    ax_heat.tick_params(axis="x", colors="lightgray")
    for sp in ax_heat.spines.values():
        sp.set_color("#444466")

    def update_title(start):
        end = min(start + win, n_traits)
        ax_heat.set_title(
            f"Traits  {start + 1} – {end}  /  {n_traits}",
            color="white", fontsize=11, pad=6,
        )

    update_title(0)

    # -----------------------------------------------------------------------
    # Colorbar
    # -----------------------------------------------------------------------
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.ax.yaxis.set_tick_params(color="lightgray")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="lightgray", fontsize=8)
    cbar.outline.set_edgecolor("#444466")

    # -----------------------------------------------------------------------
    # Slider
    # -----------------------------------------------------------------------
    ax_slide.set_facecolor("#2a2a4a")
    slider = Slider(
        ax=ax_slide,
        label="Trait window start",
        valmin=0,
        valmax=max(0, n_traits - win),
        valinit=0,
        valstep=1,
        color="#5566bb",
    )
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    # -----------------------------------------------------------------------
    # Colormap cycle buttons
    # -----------------------------------------------------------------------
    cmap_idx = [COLORMAPS.index(cmap_name) if cmap_name in COLORMAPS else 0]

    btn_prev = Button(ax_cmapL, "◀ cmap", color="#2a2a4a", hovercolor="#3a3a6a")
    btn_next = Button(ax_cmapR, "cmap ▶", color="#2a2a4a", hovercolor="#3a3a6a")
    for btn in (btn_prev, btn_next):
        btn.label.set_color("white")
        btn.label.set_fontsize(9)

    # -----------------------------------------------------------------------
    # Update callbacks
    # -----------------------------------------------------------------------
    def refresh(start, cmap_override=None):
        start = int(start)
        end   = start + win
        data  = df.iloc[:, start:end].values
        im.set_data(data)
        im.set_extent(make_extent(start))
        if cmap_override:
            im.set_cmap(cmap_override)
        im.set_clim(vmin=np.nanmin(data) if args.vmin is None else vmin,
                    vmax=np.nanmax(data) if args.vmax is None else vmax)
        set_xtick_labels(start)
        update_title(start)
        fig.canvas.draw_idle()

    def on_slide(val):
        refresh(slider.val)

    def on_prev(event):
        cmap_idx[0] = (cmap_idx[0] - 1) % len(COLORMAPS)
        refresh(slider.val, COLORMAPS[cmap_idx[0]])

    def on_next(event):
        cmap_idx[0] = (cmap_idx[0] + 1) % len(COLORMAPS)
        refresh(slider.val, COLORMAPS[cmap_idx[0]])

    slider.on_changed(on_slide)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    # Keyboard: left/right arrows to shift window
    def on_key(event):
        cur = int(slider.val)
        if event.key == "right":
            slider.set_val(min(cur + 1, n_traits - win))
        elif event.key == "left":
            slider.set_val(max(cur - 1, 0))
        elif event.key == "shift+right":
            slider.set_val(min(cur + win, n_traits - win))
        elif event.key == "shift+left":
            slider.set_val(max(cur - win, 0))

    fig.canvas.mpl_connect("key_press_event", on_key)

    fig.suptitle(
        f"{args.tree}  ×  {args.traits}",
        color="gray", fontsize=9, y=0.995,
    )

    print("Controls: drag slider | ◀/▶ buttons to cycle colormap | ← → arrow keys to shift window | Shift+← → to jump one window")
    plt.show()


if __name__ == "__main__":
    main()
