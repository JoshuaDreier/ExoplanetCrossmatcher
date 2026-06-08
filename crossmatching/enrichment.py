from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.table import MaskedColumn, Table

if TYPE_CHECKING:
    from crossmatching.param_sources.base import StellarParamSource

T_SUN = 5778.0  # K

_TEFF_SPECTYPE = [
    (30000, 'O5'), (25000, 'O7'), (20000, 'O9'),
    (10000, 'B0'), ( 9000, 'B5'),
    ( 7500, 'A0'), ( 7000, 'A5'),
    ( 6500, 'F0'), ( 6200, 'F5'), ( 6000, 'F8'),
    ( 5900, 'G0'), ( 5700, 'G2'), ( 5600, 'G5'), ( 5400, 'G8'),
    ( 5200, 'K0'), ( 4900, 'K2'), ( 4700, 'K4'), ( 4600, 'K5'), ( 4200, 'K7'),
    ( 3900, 'M0'), ( 3700, 'M1'), ( 3500, 'M2'), ( 3300, 'M3'),
    ( 3200, 'M4'), ( 3000, 'M5'), ( 2800, 'M6'), ( 2600, 'M7'), ( 2400, 'M8'),
]


def classify_spectral_type(sptype: str) -> str:
    """Map a spectral type string to a broad stellar category.

    Returns one of: 'Sun-like', 'Low-luminosity', 'Very-low-luminosity', 'Other'.
    """
    s = str(sptype).strip()
    if s == 'null':
        return 'Other'
    if re.match(r'^D[A-Z0-9]', s):
        return 'Other'
    if re.search(r'IV|III|II', s):
        return 'Other'
    m = re.match(r'^d?([OBAFGKM])(\d+(?:\.\d+)?)', s, re.I)
    if not m:
        return 'Other'
    letter = m.group(1).upper()
    subtype = float(m.group(2))
    if letter in ('O', 'B', 'A'):
        return 'Other'
    if letter in ('F', 'G'):
        return 'Sun-like'
    if letter == 'K':
        return 'Sun-like' if subtype <= 5 else 'Low-luminosity'
    if letter == 'M':
        return 'Low-luminosity' if subtype < 3 else 'Very-low-luminosity'
    return 'Other'


def teff_to_spectype(teff: float) -> str:
    """Return an approximate spectral type string (e.g. '~G2') from effective temperature."""
    if not (teff > 0):
        return ''
    closest = min(_TEFF_SPECTYPE, key=lambda x: abs(x[0] - teff))
    return f'~{closest[1]}'


def spectype_display(spec: str, teff: float) -> str:
    """Spectral type for display: use the actual string if available, else derive from teff."""
    s = str(spec).strip()
    if s and s != 'null':
        return s
    return teff_to_spectype(teff)


def temperate_mask(
    flux_rel,
    flux_rel_err1,
    flux_rel_err2,
    lower: float,
    upper: float,
    use_interval: bool = False,
) -> np.ndarray:
    """Boolean mask selecting planets within [lower, upper] insolation (S_earth).

    Parameters
    ----------
    flux_rel : array-like or MaskedColumn
        Central insolation values.
    flux_rel_err1 : array-like, MaskedColumn, or None
        Upper 1-sigma uncertainty (positive magnitude). Masked/NaN → treated as 0.
    flux_rel_err2 : array-like, MaskedColumn, or None
        Lower 1-sigma uncertainty (positive magnitude). Masked/NaN → treated as 0.
    lower, upper : float
        Insolation bounds (inclusive).
    use_interval : bool
        False (default): include only planets whose central flux_rel is in range.
        True: include planets where [flux_rel − err2, flux_rel + err1] overlaps
        [lower, upper] — i.e., the planet *could* be temperate within uncertainty.
    """
    flux = np.ma.asarray(flux_rel, dtype=float)
    valid = ~np.ma.getmaskarray(flux)

    if not use_interval:
        return valid & (flux.data >= lower) & (flux.data <= upper)

    def _to_err(arr):
        if arr is None:
            return np.zeros(len(flux))
        a = np.ma.asarray(arr, dtype=float)
        bad = np.ma.getmaskarray(a) | ~np.isfinite(a.data)
        return np.where(bad, 0.0, np.abs(a.data))

    err1 = _to_err(flux_rel_err1)
    err2 = _to_err(flux_rel_err2)
    return valid & (flux.data - err2 <= upper) & (flux.data + err1 >= lower)


def ms_radius_from_teff(teff: float) -> float:
    """Rough ZAMS Teff → radius (R/R_sun) for main-sequence fallback."""
    if not (teff > 0):
        return 0.0
    return max((teff / T_SUN) ** 1.8, 0.05)


def _col_float(table: Table, col: str):
    """Return (values, mask) float arrays for a table column."""
    c = table[col]
    if isinstance(c, MaskedColumn):
        mask = np.ma.getmaskarray(c)
        vals = np.ma.filled(c.astype(float), fill_value=0.0)
    else:
        vals = np.asarray(c, dtype=float)
        mask = ~np.isfinite(vals)
        vals = np.where(mask, 0.0, vals)
    return vals, mask


class StellarParamMerger:
    """Enriches a catalog table with stellar and derived planetary parameters.

    Applies a priority-ordered chain of StellarParamSource objects: for each
    row the first source that provides a value for a given parameter wins.
    Computes derived physical columns (r_earth, a, flux_rel, spectral_category)
    from the merged parameters. Dataset analysis (HZ thresholds, population
    statistics) is left to the caller.
    """

    def __init__(self, sources: list[StellarParamSource]):
        self.sources = sources

    def enrich(self, table: Table) -> Table:
        """Return a copy of table with enriched columns added/replaced.

        Value columns (MaskedColumn):
          st_teff, st_rad, st_mass, sy_vmag, sy_dist, st_logg, st_met,
          st_lum, pl_eqt, r_earth, flux_rel
        Two-sided uncertainty columns (MaskedColumn, masked when unavailable):
          *_err1 = upper 1σ (positive magnitude), *_err2 = lower 1σ (positive magnitude)
          st_teff_err1/2, st_rad_err1/2, st_mass_err1/2, sy_vmag_err1/2, sy_dist_err1/2,
          st_logg_err1/2, st_met_err1/2, st_lum_err1/2, pl_eqt_err1/2, flux_rel_err1/2
          Propagation follows standard astrophysical asymmetric quadrature: each direction
          (upper/lower) is tracked separately through the chain of derived quantities.
        String columns:
          st_spectype, spectral_category
          st_teff_src, st_rad_src, st_mass_src, sy_vmag_src, sy_dist_src,
          st_logg_src, st_met_src, st_lum_src, pl_eqt_src,
          r_earth_src, a_src, flux_rel_src
        """
        n = len(table)

        # ── value arrays ──────────────────────────────────────────────────────
        teff        = np.zeros(n)
        teff_mask   = np.ones(n, bool)
        rad         = np.zeros(n)
        rad_mask    = np.ones(n, bool)
        mass        = np.zeros(n)
        mass_mask   = np.ones(n, bool)
        insol       = np.zeros(n)
        insol_mask  = np.ones(n, bool)
        logg        = np.zeros(n)
        logg_mask   = np.ones(n, bool)
        pl_eqt      = np.zeros(n)
        pl_eqt_mask = np.ones(n, bool)
        vmag        = np.full(n, np.nan)
        vmag_mask   = np.ones(n, bool)
        dist        = np.full(n, np.nan)
        dist_mask   = np.ones(n, bool)
        met         = np.full(n, np.nan)
        met_mask    = np.ones(n, bool)
        spec        = [''] * n

        # ── two-sided uncertainty arrays (NaN = absent) ───────────────────────
        # err1 = upper 1σ (positive magnitude), err2 = lower 1σ (positive magnitude)
        teff_err1   = np.full(n, np.nan); teff_err2   = np.full(n, np.nan)
        rad_err1    = np.full(n, np.nan); rad_err2    = np.full(n, np.nan)
        mass_err1   = np.full(n, np.nan); mass_err2   = np.full(n, np.nan)
        insol_err1  = np.full(n, np.nan); insol_err2  = np.full(n, np.nan)
        logg_err1   = np.full(n, np.nan); logg_err2   = np.full(n, np.nan)
        pl_eqt_err1 = np.full(n, np.nan); pl_eqt_err2 = np.full(n, np.nan)
        vmag_err1   = np.full(n, np.nan); vmag_err2   = np.full(n, np.nan)
        dist_err1   = np.full(n, np.nan); dist_err2   = np.full(n, np.nan)
        met_err1    = np.full(n, np.nan); met_err2    = np.full(n, np.nan)

        # ── provenance lists (empty string = masked / no data) ────────────────
        teff_src    = [''] * n
        rad_src     = [''] * n
        mass_src    = [''] * n
        insol_src   = [''] * n
        logg_src    = [''] * n
        pl_eqt_src  = [''] * n
        vmag_src    = [''] * n
        dist_src    = [''] * n
        met_src     = [''] * n

        for i, row in enumerate(table):
            merged:      dict = {}
            merged_src:  dict = {}
            merged_err1: dict = {}
            merged_err2: dict = {}
            for source in self.sources:
                sn = source.source_name
                d = source.get(row)
                for k, v in d.items():
                    if k.endswith('_err1') or k.endswith('_err2'):
                        continue  # bound below alongside their value key
                    if k not in merged:
                        merged[k]      = v
                        merged_src[k]  = sn
                        # Bind both error sides from the same winning source
                        merged_err1[k] = d.get(f'{k}_err1')
                        merged_err2[k] = d.get(f'{k}_err2')

            def _bind(param, arr_val, arr_mask, arr_src, arr_e1, arr_e2):
                if param in merged:
                    arr_val[i]  = merged[param]
                    arr_mask[i] = False
                    arr_src[i]  = merged_src[param]
                    if merged_err1.get(param) is not None:
                        arr_e1[i] = merged_err1[param]
                    if merged_err2.get(param) is not None:
                        arr_e2[i] = merged_err2[param]
                    return True
                return False

            _bind('teff',   teff,   teff_mask,   teff_src,   teff_err1,   teff_err2)
            if not _bind('rad', rad, rad_mask, rad_src, rad_err1, rad_err2):
                if not teff_mask[i]:
                    r = ms_radius_from_teff(teff[i])
                    if r > 0:
                        rad[i] = r
                        rad_mask[i] = False
                        rad_src[i] = f"ms(teff:{teff_src[i]})"
                        # ∂r/∂T = 0 in the clamped regime (r == 0.05), so error is 0 there
                        if r > 0.05 and teff[i] > 0:
                            coeff = 1.8 * r / teff[i]
                            if np.isfinite(teff_err1[i]):
                                rad_err1[i] = coeff * teff_err1[i]
                            if np.isfinite(teff_err2[i]):
                                rad_err2[i] = coeff * teff_err2[i]
                        else:
                            rad_err1[i] = 0.0
                            rad_err2[i] = 0.0
            _bind('mass',   mass,   mass_mask,   mass_src,   mass_err1,   mass_err2)
            _bind('insol',  insol,  insol_mask,  insol_src,  insol_err1,  insol_err2)
            _bind('logg',   logg,   logg_mask,   logg_src,   logg_err1,   logg_err2)
            _bind('met',    met,    met_mask,    met_src,    met_err1,    met_err2)
            _bind('pl_eqt', pl_eqt, pl_eqt_mask, pl_eqt_src, pl_eqt_err1, pl_eqt_err2)
            _bind('vmag',   vmag,   vmag_mask,   vmag_src,   vmag_err1,   vmag_err2)
            _bind('dist',   dist,   dist_mask,   dist_src,   dist_err1,   dist_err2)
            if 'spec' in merged:
                spec[i] = merged['spec']

        st_spectype = [
            spectype_display(spec[i], teff[i] if not teff_mask[i] else 0.0)
            for i in range(n)
        ]

        # ── r_earth ───────────────────────────────────────────────────────────
        r_vals, r_orig_mask = _col_float(table, 'r')
        r_valid = ~r_orig_mask & (r_vals > 0)
        r_earth = MaskedColumn(r_vals * u.R_jup.to(u.R_earth), mask=~r_valid,
                               name='r_earth', description='Planet radius [R_earth]')
        r_earth_src = np.where(r_valid, 'emc', '')

        # ── st_lum ────────────────────────────────────────────────────────────
        lum_mask = rad_mask | teff_mask
        lum = np.where(~lum_mask, rad ** 2 * (teff / T_SUN) ** 4, np.nan)
        st_lum_src = [
            f"r:{rad_src[i]} teff:{teff_src[i]}" if not lum_mask[i] else ''
            for i in range(n)
        ]

        # ── st_lum_err1/2: asymmetric propagation of L = R²(T/T_sun)⁴ ────────
        # Both R and T have positive partial derivatives w.r.t. L, so:
        #   σ_L+/L = sqrt((2 σ_R+/R)² + (4 σ_T+/T)²)  [upper: all inputs go up]
        #   σ_L-/L = sqrt((2 σ_R-/R)² + (4 σ_T-/T)²)  [lower: all inputs go down]
        def _rel(arr_e, arr_v, arr_m):
            return np.where(arr_m | ~(arr_v > 0) | ~np.isfinite(arr_e), 0.0, arr_e / arr_v)

        rad_rel1  = _rel(rad_err1,  rad,  rad_mask)
        rad_rel2  = _rel(rad_err2,  rad,  rad_mask)
        teff_rel1 = _rel(teff_err1, teff, teff_mask)
        teff_rel2 = _rel(teff_err2, teff, teff_mask)

        lum_rel1_sq = (2.0 * rad_rel1) ** 2 + (4.0 * teff_rel1) ** 2
        lum_rel2_sq = (2.0 * rad_rel2) ** 2 + (4.0 * teff_rel2) ** 2
        has_lum_err1 = ~lum_mask & (lum_rel1_sq > 0)
        has_lum_err2 = ~lum_mask & (lum_rel2_sq > 0)
        with np.errstate(invalid='ignore'):
            lum_err1 = np.where(has_lum_err1, np.abs(lum) * np.sqrt(lum_rel1_sq), np.nan)
            lum_err2 = np.where(has_lum_err2, np.abs(lum) * np.sqrt(lum_rel2_sq), np.nan)

        # ── semi-major axis with Kepler fallback ──────────────────────────────
        a_vals, a_orig_mask = _col_float(table, 'a')
        p_vals, p_orig_mask = _col_float(table, 'p')
        mass_for_kepler = np.where(~mass_mask, mass, 1.0)
        kepler_cond = ~p_orig_mask & (p_vals > 0) & ~mass_mask
        a_kepler = np.where(
            kepler_cond,
            (mass_for_kepler * (p_vals / 365.25) ** 2) ** (1 / 3),
            0.0,
        )
        emc_a_valid = (~a_orig_mask) & (a_vals > 0)
        a_merged = np.where(emc_a_valid, a_vals, a_kepler)
        a_valid  = a_merged > 0

        a_src_list = [
            'emc' if emc_a_valid[i]
            else (f"kepler(mass:{mass_src[i]} p:emc)" if kepler_cond[i] else '')
            for i in range(n)
        ]

        # ── a_err1/2: Kepler only — σ_a = (1/3) a σ_M/M (EMC a has no error) ─
        kepler_base = kepler_cond & (mass_for_kepler > 0)
        a_err1 = np.where(kepler_base & np.isfinite(mass_err1),
                          (1.0/3.0) * a_kepler * mass_err1 / mass_for_kepler, np.nan)
        a_err2 = np.where(kepler_base & np.isfinite(mass_err2),
                          (1.0/3.0) * a_kepler * mass_err2 / mass_for_kepler, np.nan)

        # ── insolation flux ───────────────────────────────────────────────────
        with np.errstate(divide='ignore', invalid='ignore'):
            flux_calc = np.where(a_valid, lum / a_merged ** 2, np.nan)
        flux_final = np.where(~insol_mask, insol, flux_calc)
        flux_mask  = ~np.isfinite(flux_final)

        flux_rel_src = [
            f"insol:{insol_src[i]}" if not insol_mask[i]
            else (f"r:{rad_src[i]} teff:{teff_src[i]} a:{a_src_list[i]}"
                  if np.isfinite(flux_calc[i]) else '')
            for i in range(n)
        ]

        # ── flux_rel_err1/2 ───────────────────────────────────────────────────
        # Asymmetric propagation — F = L/a²:
        #   ∂F/∂L > 0, ∂F/∂a < 0, so F rises when L↑ or a↓:
        #   σ_F+ uses (lum_err1, a_err2)   [both push F up]
        #   σ_F- uses (lum_err2, a_err1)   [both push F down]
        # Note: adding asymmetric errors in quadrature is the standard astrophysical
        # convention (e.g. NEA, Kopparapu+); the underlying distributions are not
        # strictly Gaussian, so treat these as representative 1σ bounds.
        flux_rel_err1 = np.full(n, np.nan)
        flux_rel_err2 = np.full(n, np.nan)
        # Case 1: direct insol
        flux_rel_err1 = np.where(~insol_mask & np.isfinite(insol_err1), insol_err1, flux_rel_err1)
        flux_rel_err2 = np.where(~insol_mask & np.isfinite(insol_err2), insol_err2, flux_rel_err2)
        # Case 2: computed flux
        computed_case = insol_mask & np.isfinite(flux_calc) & ~lum_mask & (lum != 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Upper: lum_err1 (L↑) + a_err2 (a↓ → F↑)
            lum_up_sq  = np.where(has_lum_err1 & computed_case, (lum_err1 / lum) ** 2, 0.0)
            a_down_sq  = np.where(computed_case & np.isfinite(a_err2) & (a_merged > 0),
                                  (2.0 * a_err2 / a_merged) ** 2, 0.0)
            # Lower: lum_err2 (L↓) + a_err1 (a↑ → F↓)
            lum_dn_sq  = np.where(has_lum_err2 & computed_case, (lum_err2 / lum) ** 2, 0.0)
            a_up_sq    = np.where(computed_case & np.isfinite(a_err1) & (a_merged > 0),
                                  (2.0 * a_err1 / a_merged) ** 2, 0.0)
        err1_sq = lum_up_sq + a_down_sq
        err2_sq = lum_dn_sq + a_up_sq
        flux_rel_err1 = np.where(computed_case & (err1_sq > 0),
                                 np.abs(flux_calc) * np.sqrt(err1_sq), flux_rel_err1)
        flux_rel_err2 = np.where(computed_case & (err2_sq > 0),
                                 np.abs(flux_calc) * np.sqrt(err2_sq), flux_rel_err2)

        spectral_category = [classify_spectral_type(s) for s in st_spectype]

        def _mc(arr, name, desc=''):
            return MaskedColumn(arr, mask=~np.isfinite(arr), name=name, description=desc)

        result = table.copy()
        result['st_teff']            = MaskedColumn(teff,   mask=teff_mask,    name='st_teff')
        result['st_teff_err1']       = _mc(teff_err1,  'st_teff_err1', 'Teff upper 1σ [K]')
        result['st_teff_err2']       = _mc(teff_err2,  'st_teff_err2', 'Teff lower 1σ [K]')
        result['st_teff_src']        = teff_src
        result['st_rad']             = MaskedColumn(rad,    mask=rad_mask,     name='st_rad')
        result['st_rad_err1']        = _mc(rad_err1,   'st_rad_err1',  'Radius upper 1σ [R_sun]')
        result['st_rad_err2']        = _mc(rad_err2,   'st_rad_err2',  'Radius lower 1σ [R_sun]')
        result['st_rad_src']         = rad_src
        result['st_mass']            = MaskedColumn(mass,   mask=mass_mask,    name='st_mass')
        result['st_mass_err1']       = _mc(mass_err1,  'st_mass_err1', 'Mass upper 1σ [M_sun]')
        result['st_mass_err2']       = _mc(mass_err2,  'st_mass_err2', 'Mass lower 1σ [M_sun]')
        result['st_mass_src']        = mass_src
        result['st_logg']            = MaskedColumn(logg,   mask=logg_mask,    name='st_logg',
                                                    description='Stellar log surface gravity [cm/s^2]')
        result['st_logg_err1']       = _mc(logg_err1,  'st_logg_err1', 'log g upper 1σ [cm/s^2]')
        result['st_logg_err2']       = _mc(logg_err2,  'st_logg_err2', 'log g lower 1σ [cm/s^2]')
        result['st_logg_src']        = logg_src
        result['st_met']             = MaskedColumn(met,    mask=met_mask,     name='st_met',
                                                    description='Stellar metallicity [Fe/H]')
        result['st_met_err1']        = _mc(met_err1,   'st_met_err1',  'Metallicity upper 1σ [dex]')
        result['st_met_err2']        = _mc(met_err2,   'st_met_err2',  'Metallicity lower 1σ [dex]')
        result['st_met_src']         = met_src
        result['sy_vmag']            = MaskedColumn(vmag,   mask=vmag_mask,    name='sy_vmag')
        result['sy_vmag_err1']       = _mc(vmag_err1,  'sy_vmag_err1', 'Vmag upper 1σ [mag]')
        result['sy_vmag_err2']       = _mc(vmag_err2,  'sy_vmag_err2', 'Vmag lower 1σ [mag]')
        result['sy_vmag_src']        = vmag_src
        result['sy_dist']            = MaskedColumn(dist,   mask=dist_mask,    name='sy_dist')
        result['sy_dist_err1']       = _mc(dist_err1,  'sy_dist_err1', 'Distance upper 1σ [pc]')
        result['sy_dist_err2']       = _mc(dist_err2,  'sy_dist_err2', 'Distance lower 1σ [pc]')
        result['sy_dist_src']        = dist_src
        result['st_spectype']        = st_spectype
        result['st_lum']             = MaskedColumn(lum,    mask=lum_mask,     name='st_lum',
                                                    description='Stellar luminosity [L_sun]')
        result['st_lum_err1']        = _mc(lum_err1,   'st_lum_err1',  'Luminosity upper 1σ [L_sun]')
        result['st_lum_err2']        = _mc(lum_err2,   'st_lum_err2',  'Luminosity lower 1σ [L_sun]')
        result['st_lum_src']         = st_lum_src
        result['r_earth']            = r_earth
        result['r_earth_src']        = r_earth_src
        result['pl_eqt']             = MaskedColumn(pl_eqt, mask=pl_eqt_mask,  name='pl_eqt',
                                                    description='Planet equilibrium temperature [K]')
        result['pl_eqt_err1']        = _mc(pl_eqt_err1, 'pl_eqt_err1', 'Teq upper 1σ [K]')
        result['pl_eqt_err2']        = _mc(pl_eqt_err2, 'pl_eqt_err2', 'Teq lower 1σ [K]')
        result['pl_eqt_src']         = pl_eqt_src
        result['a_src']              = a_src_list
        result['flux_rel']           = MaskedColumn(flux_final, mask=flux_mask, name='flux_rel',
                                                    description='Insolation flux [S_earth]')
        result['flux_rel_err1']      = _mc(flux_rel_err1, 'flux_rel_err1', 'Insolation upper 1σ [S_earth]')
        result['flux_rel_err2']      = _mc(flux_rel_err2, 'flux_rel_err2', 'Insolation lower 1σ [S_earth]')
        result['flux_rel_src']       = flux_rel_src
        result['spectral_category']  = spectral_category
        return result
