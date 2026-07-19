"""Rocky/temperate candidate LaTeX table builder (tabularray + siunitx).

Produces the two appendix tables of the paper; both are longtblr environments
with an embedded caption/label, ready for a bare \\input{} inside a
landscape block.
"""

import pathlib
import re

import numpy as np
from astropy import units as u


def _is_masked(v):
    return v is np.ma.masked or (isinstance(v, float) and not np.isfinite(v))

def _is_computed_src(src):
    # Measured-value provenance strings are bare source names ("input", "nea",
    # "toi", ...); enrichment-derived values always carry a "fn(...)" call.
    return bool(src) and "(" in str(src)

def _fmt_num(value, err_hi=None, err_lo=None):
    """Format value (+ optional uncertainty) as a siunitx \\num{} token.

    Uses siunitx's compact digit-uncertainty notation -- "1.23(4)" for a
    symmetric uncertainty, "1.23(4:5)" for an asymmetric one (added in
    siunitx 3.3) -- rather than a free-decimal "1.23+0.04-0.05" form, which
    siunitx's \\num parser does not recognise and silently drops.
    """
    if _is_masked(value):
        return None
    value = float(value)
    err_hi = None if _is_masked(err_hi) or err_hi is None or err_hi <= 0 else float(err_hi)
    err_lo = None if _is_masked(err_lo) or err_lo is None or err_lo <= 0 else float(err_lo)

    # Drop uncertainties that are negligible relative to the value (e.g. sub-1e-6 au
    # errors on a 0.02 au semi-major axis) instead of rounding to a misleading "(0)".
    errs = [e for e in (err_hi, err_lo) if e is not None]
    if errs and value != 0 and min(errs) / abs(value) < 1e-4:
        err_hi = err_lo = None
        errs = []

    if errs:
        decimals = max(0, -int(np.floor(np.log10(min(errs)))) + 1)
    elif value != 0:
        decimals = max(0, 2 - int(np.floor(np.log10(abs(value)))))
    else:
        decimals = 2
    decimals = min(decimals, 6)

    v_str = f"{value:.{decimals}f}"
    if err_hi is not None and err_lo is not None:
        hi_digits = int(round(err_hi * 10 ** decimals))
        lo_digits = int(round(err_lo * 10 ** decimals))
        if hi_digits == lo_digits:
            return f"\\num{{{v_str}({hi_digits})}}"
        return f"\\num{{{v_str}({hi_digits}:{lo_digits})}}"
    return f"\\num{{{v_str}}}"

def _escape_latex(text):
    return re.sub(r"([_%#&$])", r"\\\1", str(text))

def _clean_planet_name(name):
    name = re.sub(r"\s+", " ", str(name)).strip()
    name = re.sub(r"^NAME\s+", "", name)
    return _escape_latex(name)

COMPUTED_MARK = r"\textsuperscript{a}"      # value derived by the enrichment pipeline
MSINI_MARK = r"\textsuperscript{b}"         # minimum mass (msini), not true mass
RADIUS_BOUND_MARK = r"\textsuperscript{c}"  # radius estimated from mass/msini, not measured
NOT_CONFIRMED_MARK = r"\textsuperscript{d}" # Exo-MerCat status is CANDIDATE or CONTROVERSIAL
MJUP_TO_MEARTH = u.M_jupiter.to(u.M_earth)

def _name_cell(row):
    name = _clean_planet_name(row["exo-mercat_name"])
    if str(row["status"]) in ("CANDIDATE", "CONTROVERSIAL"):
        name += NOT_CONFIRMED_MARK
    return name

def _radius_cell(row):
    r = row["r_earth"]
    if not _is_masked(r):
        return _fmt_num(r, row["r_earth_max"], row["r_earth_min"])
    lo, hi = row["r_earth_lower_bound"], row["r_earth_upper_bound"]
    if _is_masked(lo) or _is_masked(hi):
        return "\\textemdash"
    mid = 0.5 * (float(lo) + float(hi))
    return _fmt_num(mid, float(hi) - mid, mid - float(lo)) + RADIUS_BOUND_MARK

def _mass_cell(row):
    bm = row["bestmass"]
    if _is_masked(bm):
        return "\\textemdash"
    hi, lo = row["bestmass_max"], row["bestmass_min"]
    cell = _fmt_num(
        float(bm) * MJUP_TO_MEARTH,
        None if _is_masked(hi) else float(hi) * MJUP_TO_MEARTH,
        None if _is_masked(lo) else float(lo) * MJUP_TO_MEARTH,
    )
    if str(row["bestmass_provenance"]) == "Msini":
        cell += MSINI_MARK
    return cell

def _simple_cell(row, value_key, hi_key=None, lo_key=None, src_key=None):
    value = row[value_key]
    if _is_masked(value):
        return "\\textemdash"
    hi = row[hi_key] if hi_key else None
    lo = row[lo_key] if lo_key else None
    cell = _fmt_num(value, hi, lo)
    if src_key is not None and _is_computed_src(row[src_key]):
        cell += COMPUTED_MARK
    return cell

def _spectype_cell(row):
    st = row["st_spectype"]
    if _is_masked(st) or str(st).strip() in ("", "?"):
        return "\\textemdash"
    cell = _escape_latex(st)
    if _is_computed_src(row["st_spectype_src"]):
        cell += COMPUTED_MARK
    return cell

NAME_COL_WIDTH = "3.5cm"  # tabularray Q[] wrapping width for the Planet Name column

LEGEND = (
    r"{\footnotesize\textsuperscript{a}~value derived by the enrichment pipeline rather than directly "
    r"measured; \textsuperscript{b}~minimum mass ($M \sin i$) rather than true mass; "
    r"\textsuperscript{c}~radius estimated from mass or $M \sin i$ (not directly measured); "
    r"\textsuperscript{d}~Only a candidate planet.\par}"
)

def build_rocky_temperate_table_tex(catalog, out_path, caption, label, long=False,
                                    name_col_width=NAME_COL_WIDTH, sma_col_width=None,
                                    include_legend=True):
    """Write a tabularray+siunitx table of rocky+temperate planets (Confirmed section,
    then Uncertain section) to out_path, for \\input{} into a LaTeX landscape block.

    The environment is always a longtblr with an embedded caption/label block;
    long=True additionally enables pagination with a repeating header row
    (rowhead=1) for tables too large to fit on one page. name_col_width /
    sma_col_width set fixed wrapping widths (tabularray Q[] cells) for the
    Planet Name and Semi-major Axis columns (sma_col_width=None keeps a plain
    auto-sized column). include_legend appends the a/b/c/d marker legend
    paragraph after the table.
    """
    sub = catalog[catalog["rocky_temp_status"] != ""]
    sma_col = rf"Q[l,wd={sma_col_width}]" if sma_col_width else "l"
    opts = (rf"colspec = {{Q[l,wd={name_col_width}] l l l l {sma_col} c}}, font=\footnotesize, rowsep=1pt, "
            r"row{1} = {font=\bfseries\footnotesize}")
    if long:
        opts += ", rowhead=1"

    lines = [
        r"\begin{longtblr}[",
        rf"  caption = {{{caption}}},",
        rf"  label = {{{label}}},",
        rf"]{{{opts}}}",
        r"\hline\hline",
        r"Planet & Insolation Flux [\si{\Searth}] & Radius [\si{\Rearth}] & Mass [\si{\Mearth}] "
        r"& Distance [\si{\parsec}] & Semi-major Axis [\si{\astronomicalunit}] & Spectral Type \\",
        r"\hline",
    ]

    for status in ("Confirmed", "Uncertain"):
        block = sub[sub["rocky_temp_status"] == status]
        block = block[np.argsort(np.ma.filled(np.ma.asarray(block["sy_dist"], dtype=float), np.inf), kind="stable")]
        lines.append(rf"\SetCell[c=7]{{c}}{{\textbf{{{status}}}}} \\")
        lines.append(r"\hline")
        for row in block:
            name = _name_cell(row)
            insol = _simple_cell(row, "pl_insol", "pl_insol_max", "pl_insol_min", "pl_insol_src")
            radius = _radius_cell(row)
            mass = _mass_cell(row)
            dist = _simple_cell(row, "sy_dist", "sy_dist_max", "sy_dist_min", "sy_dist_src")
            smaxis = _simple_cell(row, "a", "a_max", "a_min", "a_src")
            spectype = _spectype_cell(row)
            lines.append(f"{name} & {insol} & {radius} & {mass} & {dist} & {smaxis} & {spectype} \\\\")
        lines.append(r"\hline")

    lines[-1] = r"\hline\hline"
    lines.append(r"\end{longtblr}")
    if include_legend:
        lines.append(LEGEND)
    else:
        lines.append("")

    tex = "\n".join(lines) + "\n"
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    return tex
