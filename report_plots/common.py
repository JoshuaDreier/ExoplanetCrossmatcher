"""Shared constants and figure-export helpers for the paper plot notebooks.

Every notebook in this directory produces exactly one paper output (named after
its file) and exports it into ``report/figures/`` at the repository root via
:func:`save_figure` (or directly, for the ``.tex`` tables).
"""

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

# repo root (the directory containing the crossmatching package)
ROOT = pathlib.Path(__file__).resolve().parents[1]

# export target for all paper figures and tables
FIGURES_DIR = ROOT / "report" / "figures"

mpl.rcParams["figure.dpi"] = 300

# Habitable zone flux limits in S/S_earth — Kopparapu et al. (2014), optimistic bounds
HZ_INNER = 1.7   # inner (hot) edge — Recent Venus
HZ_OUTER = 0.35  # outer (cold) edge — Early Mars

CATEGORY_COLORS = {
    'Sun-like':            '#F5A623',
    'Low-luminosity':      '#E05C00',
    'Very-low-luminosity': '#C0392B',
    'Other':               '#95A5A6',
}
CATEGORY_ORDER = ['Very-low-luminosity', 'Low-luminosity', 'Sun-like', 'Other']


def save_figure(name, fig=None, **savefig_kwargs):
    """Save the current (or given) matplotlib figure as FIGURES_DIR/<name>.pdf."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = dict(bbox_inches="tight", dpi=300)
    kwargs.update(savefig_kwargs)
    target = FIGURES_DIR / f"{name}.pdf"
    (fig if fig is not None else plt).savefig(target, **kwargs)
    print(f"saved {target}")
    return target
