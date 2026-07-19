"""Paper plotting functions, extracted verbatim from the analysis notebooks.

Each function draws one paper figure (or one panel of it)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter
from astropy.table import Table

from .common import HZ_INNER, HZ_OUTER, CATEGORY_COLORS, CATEGORY_ORDER


def _lighten(color, factor=0.45):
    r, g, b, _ = mcolors.to_rgba(color)
    return (r + (1-r)*factor, g + (1-g)*factor, b + (1-b)*factor, 1.0)


_CMAP = LinearSegmentedColormap.from_list(
    "custom_inferno",
    [
        (0.0, "#000000"), (0.1, "#000000"), (0.15, "#420a68"),
        (0.2, "#6a176e"), (0.3, "#932667"), (0.45, "#bc3754"),
        (0.6, "#dd513a"), (0.7, "#f37819"), (0.8, "#fca50a"),
        (0.9, "#f6d746"), (1.0, "#fcffa4"),
    ],
)


def _log_hist2d(x_raw, y_raw, log_x_edges, log_y_edges):
    """2D histogram binned in log-space; returns (x_edges, y_edges, log10(counts+1))."""
    mask = (x_raw > 0) & (y_raw > 0)
    x, y = x_raw[mask], y_raw[mask]
    counts, _, _ = np.histogram2d(np.log10(x), np.log10(y), bins=[log_x_edges, log_y_edges])
    x_edges = 10 ** log_x_edges
    y_edges = 10 ** log_y_edges
    z_log = np.log10(counts.T + 1)
    return x_edges, y_edges, z_log


def double_hist_heatmap(catalog1, catalog2, catalog1_name, catalog2_name, figsize=(12, 5), title=None):
    n_bins_x = 60
    n_bins_y = 45
    log_x_edges = np.linspace(np.log10(0.3), np.log10(100), n_bins_x + 1)
    log_y_edges = np.linspace(np.log10(0.01), np.log10(10000), n_bins_y + 1)

    cat1_mask = catalog1['r_earth'].mask | catalog1['pl_insol'].mask
    cat2_mask = catalog2['r_earth'].mask | catalog2['pl_insol'].mask

    # compute both histograms up front so the shared color scale can be
    # derived from the true combined max, matching Plotly's auto-ranged coloraxis
    x1_edges, y1_edges, z1 = _log_hist2d(catalog1['r_earth'][~cat1_mask], catalog1['pl_insol'][~cat1_mask], log_x_edges, log_y_edges)
    x2_edges, y2_edges, z2 = _log_hist2d(catalog2['r_earth'][~cat2_mask], catalog2['pl_insol'][~cat2_mask], log_x_edges, log_y_edges)
    vmax = max(z1.max(), z2.max())

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    mesh = None
    for ax, x_edges, y_edges, z_log, name in zip(
        axes, [x1_edges, x2_edges], [y1_edges, y2_edges], [z1, z2], [catalog1_name, catalog2_name]
    ):
        mesh = ax.pcolormesh(x_edges, y_edges, z_log, cmap=_CMAP, vmin=0, vmax=vmax, shading='flat', edgecolors='face', rasterized=True)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(name)

        ax.add_patch(Rectangle(
            (0.5, min(HZ_OUTER, HZ_INNER)), 1.0, abs(HZ_INNER - HZ_OUTER),
            edgecolor=(1, 0, 0, 1.0),      # opaque red border
            facecolor=(1, 0, 0, 0.12),     # translucent red fill
            linewidth=1.5
        ))
        xticks = [0.1, 0.3, 0.5, 1, 1.5, 3, 5, 10, 25, 100]
        yticks = [0.01, 0.1, HZ_INNER, 1, HZ_OUTER, 10, 100, 1000, 10000]
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in xticks]))
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_major_formatter(FixedFormatter([str(v) for v in yticks]))
        ax.set_xlabel(r"Radius $R_p \; [R_\oplus]$")

    axes[0].set_ylabel(r"Insolation $S_p \; [S_\oplus]$")

    fig.subplots_adjust(right=0.88, wspace=0.15)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Count")

    # colorbar ticks at fixed count values, but only up to the actual vmax
    tick_counts = [c for c in [0, 1, 10, 100, 1000] if np.log10(c + 1) < vmax - 1e-9]
    top_count = round(10 ** vmax - 1)          # actual max count, back-converted from vmax
    tick_counts.append(top_count)

    cbar.set_ticks([np.log10(c + 1) for c in tick_counts])
    cbar.set_ticklabels([str(c) for c in tick_counts])

    if title:
        fig.suptitle(title)


def spectral_grid_heatmap(catalogs, names, categories, figsize=None, title=None):
    n_bins_x, n_bins_y = 60, 45
    log_x_edges = np.linspace(np.log10(0.3), np.log10(100), n_bins_x + 1)
    log_y_edges = np.linspace(np.log10(0.01), np.log10(10000), n_bins_y + 1)

    n_rows, n_cols = len(categories), len(catalogs)
    if figsize is None:
        figsize = (5 * n_cols + 2, 3.1 * n_rows + 1)

    # per (category-row, catalog-col) histogram; shared vmax across the whole grid
    grid_data = []
    vmax = 0
    for cat_label in categories:
        row_data = []
        for cat in catalogs:
            nan_mask = cat['r_earth'].mask | cat['pl_insol'].mask
            keep = (cat['spectral_category'] == cat_label) & (~nan_mask)
            x_edges, y_edges, z = _log_hist2d(cat['r_earth'][keep], cat['pl_insol'][keep], log_x_edges, log_y_edges)
            row_data.append((x_edges, y_edges, z))
            vmax = max(vmax, z.max())
        grid_data.append(row_data)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    mesh = None
    xticks = [0.3, 1, 3, 10, 25, 100]
    yticks = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    for i, cat_label in enumerate(categories):
        for j, cat_name in enumerate(names):
            ax = axes[i, j]
            x_edges, y_edges, z = grid_data[i][j]
            mesh = ax.pcolormesh(x_edges, y_edges, z, cmap=_CMAP, vmin=0, vmax=vmax,
                                 shading='flat', edgecolors='face', rasterized=True)
            ax.set_xscale('log')
            ax.set_yscale('log')

            if i == 0:
                ax.set_title(cat_name)
            if j == 0:
                ax.set_ylabel(f"{cat_label}\n" + r"$S_p \; [S_\oplus]$")
            if i == n_rows - 1:
                ax.set_xlabel(r"$R_p \; [R_\oplus]$")

            ax.add_patch(Rectangle(
                (0.5, min(HZ_OUTER, HZ_INNER)), 1.0, abs(HZ_INNER - HZ_OUTER),
                edgecolor=(1, 0, 0, 1.0), facecolor=(1, 0, 0, 0.12), linewidth=1.2
            ))
            ax.xaxis.set_major_locator(FixedLocator(xticks))
            ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in xticks]))
            ax.yaxis.set_major_locator(FixedLocator(yticks))
            ax.yaxis.set_major_formatter(FixedFormatter([str(v) for v in yticks]))

    fig.subplots_adjust(right=0.91, wspace=0.1, hspace=0.1)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Count")
    tick_counts = [c for c in [0, 1, 10, 100, 1000] if np.log10(c + 1) < vmax - 1e-9]
    top_count = round(10 ** vmax - 1)          # actual max count, back-converted from vmax
    tick_counts.append(top_count)
    cbar.set_ticks([np.log10(c + 1) for c in tick_counts])
    cbar.set_ticklabels([str(c) for c in tick_counts])
    if title:
        fig.suptitle(title)


def plot_distr(catalog, ax, color="#1f77b4"):
    confirmed = catalog["status"] == "CONFIRMED"
    ms = 0.5
    plt.scatter(catalog['r_earth'][~confirmed], catalog['pl_insol'][~confirmed], s=ms, color=_lighten(color), label="candidate planets")
    plt.scatter(catalog['r_earth'][confirmed], catalog['pl_insol'][confirmed],  s=ms, color=color, label="confirmed planets")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r'Planet radius $R_p$ $[R_\oplus]$')
    plt.ylabel(r'Insolation S [$S_\oplus$]')
    rect = Rectangle((0.5, HZ_OUTER), 1, HZ_INNER - HZ_OUTER,
                linewidth=.5, edgecolor='r', facecolor='#FF000020')
    rect.set_label('Rocky + Temperate')
    ax.add_patch(rect)
    ax.legend()
    print(np.sum((catalog["r_earth"] > 0) & (catalog["pl_insol"] > 0)), "valid entries")


def plot_rocky_temp_bars(ax: plt.Axes, catalog: Table, title: str, maximum: int, width=0.55):
    x = np.arange(len(CATEGORY_ORDER))
    inside_thresh = max(0.1*maximum, 1)
    tick_offset = 0.125

    for i, cat in enumerate(CATEGORY_ORDER):
        color = CATEGORY_COLORS[cat]
        n_confirmed = sum((catalog["rocky_temp_status"] == "Confirmed") & (catalog["spectral_category"] == cat))
        n_uncertain = sum((catalog["rocky_temp_status"] == "Uncertain") & (catalog["spectral_category"] == cat))
        ax.bar(i, n_confirmed, width, color=color, alpha=0.35)
        if n_confirmed:
            ax.bar(
                x=i,
                height=n_confirmed,
                width=width,
                bottom=0,
                color=color,
                edgecolor=color
            )
        if n_uncertain:
            ax.bar(
                x=i,
                height=n_uncertain,
                width=width,
                bottom=n_confirmed,
                alpha=0.3,
                color=color,
            )

        bar_total = n_confirmed + n_uncertain

        small_segs = []
        for seg_count, seg_bottom, transparent in [
            (n_confirmed, 0, False),
            (n_uncertain,  n_confirmed, True),
        ]:
            if seg_count == 0:
                continue
            seg_center_y = seg_bottom + seg_count / 2
            seg_high_y = seg_bottom + seg_count * 0.8
            if seg_count >= inside_thresh:
                ax.text(i, seg_center_y, str(seg_count),
                        ha='center', va='center', fontsize=9, color='black')
            else:
                small_segs.append((seg_high_y, seg_count, transparent))

        for j, (seg_center_y, seg_count, transparent) in enumerate(small_segs):
            ann_color = color if not transparent else _lighten(color, 0.7)
            if j % 2 == 0:
                ax.annotate(str(seg_count),
                            xy=(i, seg_center_y),
                            xytext=(i + width/2 + tick_offset, seg_center_y),
                            fontsize=9, va='center', ha='left', color=ann_color, alpha=1 if transparent else 1,
                            arrowprops=dict(arrowstyle='-', color=ann_color, lw=1.0,  alpha=(1 if transparent else 1)))
            else:
                ax.annotate(str(seg_count),
                            xy=(i, seg_center_y),
                            xytext=(i - width/2 - tick_offset, seg_center_y),
                            fontsize=9, va='center', ha='right', color=ann_color, alpha=1 if transparent else 1,
                            arrowprops=dict(arrowstyle='-', color=ann_color, lw=1.0, alpha=(1 if transparent else 1)))

        ax.text(i, bar_total + maximum * 0.025, f'Total: {bar_total}',
                ha='center', va='bottom', fontsize=11)

    legend_handles = []
    legend_handles.append(Patch(color='#555555', alpha=1.0,
                            label='confirmed rocky, temperate planets'))
    legend_handles.append(Patch(color='#555555', alpha=0.3,
                                    label='uncertain rocky, temperate planets'))

    ax.set_xlim(-width, len(CATEGORY_ORDER) - 1 + width)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_ORDER)
    ax.set_ylim(0, maximum)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(handles=legend_handles)
