#!/usr/bin/env python3
"""
tree_heatmap.py
───────────────
Phylogenetic tree + binary trait presence/absence heatmap.
Works with any newick tree and any CSV trait file where the first
column contains tip labels.

Dependencies:
    pip install biopython matplotlib pandas numpy

────────────────────────────────────────────────────────────────────
USAGE EXAMPLES
────────────────────────────────────────────────────────────────────

# Minimal — one trait file, all columns, one color:
  python tree_heatmap.py --tree my.nwk --traits traits.csv

# Multiple trait files, each gets its own color block:
  python tree_heatmap.py \
      --tree my.nwk \
      --traits phenotype.csv gene_piv.csv \
      --labels Phenotype Genes

# Select specific columns from a file (comma-separated, no spaces):
  python tree_heatmap.py \
      --tree my.nwk \
      --traits gene_piv.csv \
      --columns "geneA,geneB,geneC"

# Two files, select columns from each, add a row-sum column per block:
  python tree_heatmap.py \
      --tree my.nwk \
      --traits phenotype.csv gene_piv.csv \
      --columns "* | geneA,geneB,geneC" \
      --sum-cols " | ClusterSum" \
      --labels Phenotype GeneCluster \
      --output figure.pdf

  (* means "all columns"; leading | space = no selection for first file)

# Inspect what columns are available in a traits file, then exit:
  python tree_heatmap.py --list-columns gene_piv.csv

────────────────────────────────────────────────────────────────────
COLUMN / SUM-COL SYNTAX (--columns / --sum-cols)
────────────────────────────────────────────────────────────────────
Both flags accept a pipe-delimited ( | ) string with one slot per
--traits file. Commas separate columns within each slot.

  --columns "geneA,geneB | geneC,geneD"
               ^ file 1       ^ file 2

  --sum-cols "ClusterA | ClusterB"

Use * or leave a slot blank for "all columns" / "no sum column".
"""

import sys
import argparse
import textwrap

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Bio import Phylo


# ══════════════════════════════════════════════════════════════════════════════
#  Color palette
# ══════════════════════════════════════════════════════════════════════════════

ABSENT_COLOR = "#f0f0f0"

NAMED_COLORS = {
    "blue":   "#1E88E5",
    "red":    "#D81B60",
    "green":  "#43A047",
    "orange": "#FB8C00",
    "purple": "#8E24AA",
    "teal":   "#00897B",
    "brown":  "#6D4C41",
    "indigo": "#3949AB",
}

_AUTO_PALETTE = list(NAMED_COLORS.values())


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        prog="tree_heatmap.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Draw a phylogenetic tree with a binary trait presence/absence heatmap.
            See the module docstring for full usage examples.
        """),
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--tree", metavar="FILE",
        help="Newick tree file.",
    )
    p.add_argument(
        "--traits", metavar="FILE", nargs="+",
        help="One or more trait CSV files. First column must contain tip IDs.",
    )

    # ── Column selection ──────────────────────────────────────────────────────
    p.add_argument(
        "--columns", metavar="SPEC", default=None,
        help=(
            "Pipe-separated ( | ) per-file column lists. "
            "Comma-separate columns within each slot. "
            "Use * or leave blank for all columns. "
            "Example:  --columns \"geneA,geneB | *\""
        ),
    )
    p.add_argument(
        "--sum-cols", metavar="SPEC", default=None, dest="sum_cols",
        help=(
            "Pipe-separated per-file names for a binary row-sum column "
            "(1 if any gene in the block is present). "
            "Leave a slot blank to skip. "
            "Example:  --sum-cols \" | ClusterSum\""
        ),
    )

    # ── Aesthetics ────────────────────────────────────────────────────────────
    p.add_argument(
        "--colors", metavar="COLOR", nargs="+", default=None,
        help=(
            f"One color per --traits file. "
            f"Named options: {', '.join(NAMED_COLORS)}. "
            "Hex values (#RRGGBB) also accepted. "
            "Auto-assigned from palette if omitted."
        ),
    )
    p.add_argument(
        "--labels", metavar="LABEL", nargs="+", default=None,
        help="Legend label for each traits file. Defaults to the filename stem.",
    )
    p.add_argument(
        "--no-separators", action="store_true", default=False,
        help="Suppress vertical dividers between trait-file blocks.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--output", "-o", metavar="FILE", default="tree_heatmap.pdf",
        help="Output file. Extension sets format: .pdf .png .svg  (default: tree_heatmap.pdf)",
    )

    # ── Figure layout ─────────────────────────────────────────────────────────
    p.add_argument("--fig-width",    type=float, default=None,
                   help="Figure width in inches (auto-scaled if omitted).")
    p.add_argument("--fig-height",   type=float, default=None,
                   help="Figure height in inches (auto-scaled if omitted).")
    p.add_argument("--tree-frac",    type=float, default=0.30,
                   help="Fraction of figure width for the tree panel (default: 0.30).")
    p.add_argument("--tip-fontsize", type=float, default=5,
                   help="Tip label font size (default: 5).")
    p.add_argument("--col-fontsize", type=float, default=7,
                   help="Column label font size (default: 7).")
    p.add_argument("--col-angle",    type=float, default=45,
                   help="Column label rotation in degrees (default: 45).")
    p.add_argument("--dpi",          type=int,   default=200,
                   help="Output resolution in DPI (default: 200).")

    # ── Utility ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--list-columns", metavar="FILE",
        help="Print all column names in a traits CSV and exit.",
    )

    return p


# ── Argument parsing helpers ──────────────────────────────────────────────────

def _split_pipe(spec, n_files):
    """Split a pipe-delimited spec into exactly n_files slots."""
    if spec is None:
        return [""] * n_files
    parts = [s.strip() for s in spec.split("|")]
    parts += [""] * n_files   # pad
    return parts[:n_files]


def _resolve_color(raw, idx):
    """Return a hex color string from a name, hex value, or auto index."""
    if raw is None:
        return _AUTO_PALETTE[idx % len(_AUTO_PALETTE)]
    raw = raw.strip()
    if raw in NAMED_COLORS:
        return NAMED_COLORS[raw]
    if raw.startswith("#"):
        return raw
    try:
        import matplotlib.colors as mc
        return mc.to_hex(raw)
    except (ValueError, KeyError):
        print(f"  WARNING: unrecognised color '{raw}', using auto palette.",
              file=sys.stderr)
        return _AUTO_PALETTE[idx % len(_AUTO_PALETTE)]


def _stem(fpath):
    import os
    return os.path.splitext(os.path.basename(fpath))[0]


def build_trait_blocks(trait_files, columns_spec, sum_cols_spec,
                       colors_raw, labels_raw, add_separators):
    """Assemble the list of trait-block dicts from parsed CLI arguments."""
    n         = len(trait_files)
    col_slots = _split_pipe(columns_spec, n)
    sum_slots = _split_pipe(sum_cols_spec, n)

    blocks = []
    for i, fpath in enumerate(trait_files):
        # Column selection
        slot = col_slots[i]
        columns = None if (slot == "" or slot == "*") \
                  else [c.strip() for c in slot.split(",") if c.strip()]

        # Summary column
        sum_col = sum_slots[i].strip() or None

        # Color
        raw_color = (colors_raw[i] if colors_raw and i < len(colors_raw) else None)
        color_hex = _resolve_color(raw_color, i)

        # Label
        label = (labels_raw[i] if labels_raw and i < len(labels_raw) else None) \
                or _stem(fpath)

        blocks.append(dict(
            file      = fpath,
            columns   = columns,
            sum_col   = sum_col,
            color_hex = color_hex,
            label     = label,
            # separators between blocks (not after the last one)
            separator = add_separators and (i < n - 1),
        ))

    return blocks


# ══════════════════════════════════════════════════════════════════════════════
#  Tree utilities
# ══════════════════════════════════════════════════════════════════════════════

def _tip_order(clade):
    """Return tip names in DFS draw order (top → bottom)."""
    if clade.is_terminal():
        return [clade.name]
    tips = []
    for child in clade.clades:
        tips.extend(_tip_order(child))
    return tips


def _compute_xy(clade, x=0.0, tip_counter=None):
    """
    Assign (x, y) to every clade.
      x = cumulative branch length from root
      y = tip index for leaves; mean-of-children for internal nodes
    Returns {id(clade): (x, y)}.
    """
    if tip_counter is None:
        tip_counter = [0]

    pos = {}
    cx  = x + (clade.branch_length or 0.0)

    if clade.is_terminal():
        pos[id(clade)] = (cx, float(tip_counter[0]))
        tip_counter[0] += 1
    else:
        child_pos = {}
        for child in clade.clades:
            child_pos.update(_compute_xy(child, cx, tip_counter))
        pos.update(child_pos)
        cy = np.mean([child_pos[id(ch)][1] for ch in clade.clades])
        pos[id(clade)] = (cx, cy)

    return pos


def draw_tree(ax, tree, tip_order, fontsize=5):
    """Render a rectangular phylogram on *ax*."""
    pos   = _compute_xy(tree.root, x=0.0)
    max_x = max(v[0] for v in pos.values())

    def _draw(clade):
        x, y = pos[id(clade)]
        if not clade.is_terminal():
            cys = [pos[id(ch)][1] for ch in clade.clades]
            ax.plot([x, x], [min(cys), max(cys)],
                    color="black", lw=0.7, solid_capstyle="butt")
            for child in clade.clades:
                cx, cy = pos[id(child)]
                ax.plot([x, cx], [cy, cy],
                        color="black", lw=0.7, solid_capstyle="butt")
                _draw(child)
        else:
            ax.plot([x, max_x], [y, y], color="#cccccc", lw=0.4, ls="--")
            ax.text(max_x * 1.01, y, clade.name,
                    va="center", ha="left", fontsize=fontsize, clip_on=False)

    _draw(tree.root)
    ax.set_ylim(-0.5, len(tip_order) - 0.5)
    ax.invert_yaxis()
    ax.axis("off")


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_block(block: dict) -> pd.DataFrame:
    """Load one traits file, subset columns, binarise, optionally add sum col."""
    df = pd.read_csv(block["file"], index_col=0, dtype=str)

    if block.get("columns"):
        missing = [c for c in block["columns"] if c not in df.columns]
        if missing:
            print(f"  WARNING: columns not found in {block['file']}: {missing}",
                  file=sys.stderr)
        df = df[[c for c in block["columns"] if c in df.columns]]

    # Binarise: non-zero numeric or non-empty string → 1
    df = df.fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)
    df = (df > 0).astype(int)

    if block.get("sum_col"):
        name = block["sum_col"]
        df[name] = (df.sum(axis=1) > 0).astype(int)

    return df


def build_merged(tip_order: list, trait_blocks: list) -> tuple:
    """
    Load and outer-join all trait blocks aligned to tree tip order.
    Returns (merged_df, col_meta) where
      col_meta = [(col_name, color_hex, separator_after), ...]
    """
    frames   = []
    col_meta = []

    for block in trait_blocks:
        df = load_block(block)
        for col in df.columns:
            col_meta.append((col, block["color_hex"], False))
        if block.get("separator") and col_meta:
            # Mark the last column of this block for a post-separator
            col_meta[-1] = (col_meta[-1][0], col_meta[-1][1], True)
        frames.append(df)

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.join(df, how="outer")

    merged = merged.reindex(tip_order).fillna(0).astype(int)
    return merged, col_meta


# ══════════════════════════════════════════════════════════════════════════════
#  Heatmap drawing
# ══════════════════════════════════════════════════════════════════════════════

def draw_heatmap(ax, df: pd.DataFrame, col_meta: list,
                 col_fontsize=7, col_angle=45,
                 sep_color="#555555", sep_lw=1.2):
    """Render presence/absence tiles on *ax* aligned to tree row order."""
    n_rows, n_cols = df.shape
    tip_order = list(df.index)

    for ci, (col, color_hex, sep_after) in enumerate(col_meta):
        vals = df[col].values
        for ri, v in enumerate(vals):
            rect = plt.Rectangle(
                [ci, n_rows - ri - 1], 1, 1,
                facecolor=color_hex if v else ABSENT_COLOR,
                edgecolor="white", lw=0.3,
            )
            ax.add_patch(rect)
        if sep_after:
            ax.axvline(ci + 1, color=sep_color, lw=sep_lw, zorder=5)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(
        [c for c, _, _ in col_meta],
        fontsize=col_fontsize, rotation=col_angle,
        ha="left", rotation_mode="anchor",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_legend(trait_blocks: list) -> list:
    """One legend entry per trait block."""
    handles = [mpatches.Patch(color=ABSENT_COLOR, label="Absent")]
    for b in trait_blocks:
        handles.append(
            mpatches.Patch(color=b["color_hex"],
                           label=f"Present — {b['label']}")
        )
    return handles


# ══════════════════════════════════════════════════════════════════════════════
#  Auto figure sizing
# ══════════════════════════════════════════════════════════════════════════════

def auto_figsize(n_tips, n_cols, tree_frac):
    """Scale figure so tiles are approximately square and labels have room."""
    height         = max(8.0,  n_tips * 0.12)
    heat_panel_w   = max(6.0,  n_cols * 0.18)
    width          = max(14.0, heat_panel_w / max(1.0 - tree_frac, 0.1))
    return width, height


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = parse_args()
    args   = parser.parse_args()

    # ── Utility: list columns and exit ─────────────────────────────────────────
    if args.list_columns:
        df = pd.read_csv(args.list_columns, index_col=0, nrows=0)
        print(f"\nColumns in  {args.list_columns}  ({len(df.columns)} total):\n")
        for i, c in enumerate(df.columns, 1):
            print(f"  {i:>5}.  {c}")
        print()
        sys.exit(0)

    # ── Validate required arguments ────────────────────────────────────────────
    if not args.tree:
        parser.error("--tree is required.")
    if not args.traits:
        parser.error("--traits is required (provide at least one CSV file).")

    # ── Assemble trait blocks ──────────────────────────────────────────────────
    trait_blocks = build_trait_blocks(
        trait_files    = args.traits,
        columns_spec   = args.columns,
        sum_cols_spec  = args.sum_cols,
        colors_raw     = args.colors,
        labels_raw     = args.labels,
        add_separators = not args.no_separators,
    )

    # ── Load tree ──────────────────────────────────────────────────────────────
    print(f"Loading tree:  {args.tree}")
    tree      = Phylo.read(args.tree, "newick")
    tip_order = _tip_order(tree.root)
    n_tips    = len(tip_order)
    print(f"  {n_tips} tips")

    # ── Load & merge traits ────────────────────────────────────────────────────
    print("Loading traits:")
    for b in trait_blocks:
        cols_desc = f"{len(b['columns'])} columns" if b["columns"] else "all columns"
        sum_desc  = f"  +sum→{b['sum_col']}" if b["sum_col"] else ""
        print(f"  {b['file']}  [{cols_desc}]{sum_desc}")

    df_merged, col_meta = build_merged(tip_order, trait_blocks)
    n_cols = len(col_meta)
    print(f"  Merged: {n_cols} trait columns × {n_tips} genomes")

    # ── Figure size ────────────────────────────────────────────────────────────
    fw, fh = auto_figsize(n_tips, n_cols, args.tree_frac)
    if args.fig_width:
        fw = args.fig_width
    if args.fig_height:
        fh = args.fig_height

    fig     = plt.figure(figsize=(fw, fh))
    gap     = 0.01
    tf      = args.tree_frac
    ax_tree = fig.add_axes([0.02,     0.02, tf - gap,      0.88])
    ax_heat = fig.add_axes([tf + gap, 0.02, 1 - tf - 0.08, 0.88])

    # ── Render ─────────────────────────────────────────────────────────────────
    print("Drawing tree…")
    draw_tree(ax_tree, tree, tip_order, fontsize=args.tip_fontsize)

    print("Drawing heatmap…")
    draw_heatmap(
        ax_heat, df_merged, col_meta,
        col_fontsize = args.col_fontsize,
        col_angle    = args.col_angle,
    )

    handles = build_legend(trait_blocks)
    fig.legend(
        handles=handles, loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        frameon=True, fontsize=9,
        title="Trait Presence", title_fontsize=9,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    print(f"Saving → {args.output}")
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()