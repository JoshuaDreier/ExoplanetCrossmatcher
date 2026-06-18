# Enrichment
> [!warning] This feature is a bit crude and should only be used to very broadly categorize planets, not for actual per planet parameters

`ParamMerger.enrich()` adds stellar and derived planetary parameters to a crossmatch result table.  See [Architecture — Enrichment pipeline](architecture.md#enrichment-pipeline) for the full data-flow diagram.

## Concept: priority-ordered source chain
Parameters are merged from multiple sources using a **first-source-wins** rule: for each row and each parameter, the first source in the list that provides a non-null value wins, and its value plus both asymmetric error bars are recorded.
A provenance (`*_src`) column tracks which source won for each parameter.

```python
from crossmatching.enrichment import StellarParamMerger
from crossmatching import (
    HpicStellarParamSource,
    NeaStellarParamSource,
    SimbadStellarParamSource,
    EpicStellarParamSource,
    ToiStellarParamSource,
)

# Recommended priority order for an HPIC × EMC crossmatch
merger = StellarParamMerger([
    hpic_src,    # highest priority — from HPIC crossmatch output
    nea_src,     # second — NASA Exoplanet Archive pscomppars
    epic_src,    # third — K2 EPIC catalog
    toi_src,     # fourth — TESS TOI catalog
    simbad_src,  # fallback — SIMBAD spectroscopic measurements
])

enriched = merger.enrich(emc_result)
```

## Loading the parameter sources
Each source follows the same `load(from_file=...)` caching pattern as catalogs.
```python
import glob

nea_src = NeaStellarParamSource()
nea_src.load(from_file="pscomppars.txt", format="ascii")

epic_src = EpicStellarParamSource()
epic_src.load(from_file=sorted(glob.glob("InputSources/epic_init*.csv"))[-1], format="ascii.csv")

toi_src = ToiStellarParamSource()
toi_src.load(from_file=sorted(glob.glob("InputSources/toi_init*.csv"))[-1], format="ascii.csv")

simbad_src = SimbadStellarParamSource()
simbad_src.load(from_file="simbad_params.txt")

# HpicStellarParamSource is constructed from the in-memory crossmatch result
hpic_src = HpicStellarParamSource(emc_result)
hpic_src.load()
```

## What each source provides

| Source | Parameters |
|--------|-----------|
| `HpicStellarParamSource` | `teff`, `rad`, `vmag`, `spec` — from HPIC LC4 via the crossmatch |
| `NeaStellarParamSource` | `teff`, `rad`, `mass`, `logg`, `insol`, `dist`, `met` |
| `EpicStellarParamSource` | `teff`, `rad`, `mass`, `logg` |
| `ToiStellarParamSource` | `teff`, `rad`, `mass`, `logg` |
| `SimbadStellarParamSource` | `teff`, `rad`, `logg` — from SIMBAD `mesFe_h` measurements |

## Derived columns

`enrich()` computes several columns that are not directly available in any source:

| Column                | Formula                                                     | Notes                                                                |
| --------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------- |
| `st_lum`              | R²(T/T☉)⁴                                                   | Masked when `rad` or `teff` is unavailable                           |
| `st_rad` (fallback)   | (T/T☉)^1.8                                                  | ZAMS approximation used only when no source provides a radius        |
| `a` (semi-major axis) | Direct from catalog, or Kepler's 3rd law: (M(P/Pyr)²)^(1/3) | Kepler fallback used when catalog `a` is missing                     |
| `pl_insol`            | L/a²                                                        | Insolation in S⊕; direct `insol` from NEA takes priority             |
| `r_earth`             | r_Jup × (R_Jup/R_Earth)                                     | Direct planetary radius in R⊕                                        |
| `r_earth_min/max`     | Chen & Kipping from msini                                   | Only populated when `r_earth` is masked (RV-only detections)         |
| `spectral_category`   | Derived from `st_spectype`                                  | `'Sun-like'`, `'Low-luminosity'`, `'Very-low-luminosity'`, `'Other'` |

All value columns have corresponding `*_err1` (upper 1σ) and `*_err2` (lower 1σ)
columns, propagated asymmetrically through the derivation chain.

## Classification masks

Two boolean-array helpers are provided for population filtering:

```python
from crossmatching.enrichment import rocky_mask, temperate_mask

# Confirmed rocky planets (0.5–1.5 R⊕)
is_rocky = rocky_mask(enriched["r_earth"], enriched["r_earth_min"], enriched["r_earth_max"])

# Allow uncertain cases (radius interval overlaps rocky range)
is_possibly_rocky = rocky_mask(
    enriched["r_earth"], enriched["r_earth_min"], enriched["r_earth_max"],
    use_interval=True,
)

# Temperate planets (0.35–1.77 S⊕, Kopparapu+ 2013 conservative HZ)
is_temperate = temperate_mask(
    enriched["pl_insol"], enriched["pl_insol_err1"], enriched["pl_insol_err2"],
    lower=0.35, upper=1.77,
)

candidates = enriched[is_rocky & is_temperate]
```

`use_interval=True` includes planets whose error-bar interval *overlaps* the
target range rather than only planets whose central value falls within it.

## msini radius bounds

For RV-only planets where no transit radius is known, `enrich()` estimates a (very broad) radius range from the minimum projected mass (msini):
- `r_min`: radius from msini directly (face-on)
- `r_max`: radius from msini / sin_min, where `sin_min = 0.3` by defau-lt, giving a 95.45% (1σ confidence) upper bound, under the assumption of an isotropic inclination prior $i$ distribution. This can be computed by sol

[TBD]

For a more detailed discussion of posterior distributions, refer to https://arxiv.org/pdf/1007.0245, we will (crudely) use ${ \sin (i)=  0.3 }$, derived from a naive posterior distribution at 1σ confidence by default, but the parameter can be changed as a class parameter

## Part B — Enrichment output

`StellarParamMerger.enrich()` returns a copy of the input crossmatch table with
the columns below added or replaced.  All `MaskedColumn` entries are masked
(not NaN) when the value is unavailable.

### Stellar parameter columns

| Column | Unit | Type | Description |
|--------|------|------|-------------|
| `st_teff` | K | MaskedColumn | Effective temperature |
| `st_teff_err1` | K | MaskedColumn | Upper 1σ uncertainty (positive magnitude) |
| `st_teff_err2` | K | MaskedColumn | Lower 1σ uncertainty (positive magnitude) |
| `st_teff_src` | — | str | Source that provided `st_teff` |
| `st_rad` | R☉ | MaskedColumn | Stellar radius |
| `st_rad_err1` | R☉ | MaskedColumn | Upper 1σ |
| `st_rad_err2` | R☉ | MaskedColumn | Lower 1σ |
| `st_rad_src` | — | str | Source (e.g. `'nea'`, `'ms(teff:hpic)'` for ZAMS fallback) |
| `st_mass` | M☉ | MaskedColumn | Stellar mass |
| `st_mass_err1` | M☉ | MaskedColumn | Upper 1σ |
| `st_mass_err2` | M☉ | MaskedColumn | Lower 1σ |
| `st_mass_src` | — | str | Source |
| `st_logg` | cm/s² | MaskedColumn | Stellar log surface gravity |
| `st_logg_err1` | cm/s² | MaskedColumn | Upper 1σ |
| `st_logg_err2` | cm/s² | MaskedColumn | Lower 1σ |
| `st_logg_src` | — | str | Source |
| `st_met` | dex | MaskedColumn | Metallicity [Fe/H] |
| `st_met_err1` | dex | MaskedColumn | Upper 1σ |
| `st_met_err2` | dex | MaskedColumn | Lower 1σ |
| `st_met_src` | — | str | Source |
| `sy_vmag` | mag | MaskedColumn | Visual magnitude |
| `sy_vmag_err1` | mag | MaskedColumn | Upper 1σ |
| `sy_vmag_err2` | mag | MaskedColumn | Lower 1σ |
| `sy_vmag_src` | — | str | Source |
| `sy_dist` | pc | MaskedColumn | Distance |
| `sy_dist_err1` | pc | MaskedColumn | Upper 1σ |
| `sy_dist_err2` | pc | MaskedColumn | Lower 1σ |
| `sy_dist_src` | — | str | Source |
| `st_spectype` | — | str | Spectral type string (from source, or derived from `st_teff` as `'~G2'`) |
| `st_lum` | L☉ | MaskedColumn | Stellar luminosity derived as R²(T/T☉)⁴ |
| `st_lum_err1` | L☉ | MaskedColumn | Upper 1σ (asymmetric propagation) |
| `st_lum_err2` | L☉ | MaskedColumn | Lower 1σ |
| `st_lum_src` | — | str | Provenance string (e.g. `'r:nea teff:hpic'`) |

### Planetary parameter columns

| Column | Unit | Type | Description |
|--------|------|------|-------------|
| `pl_eqt` | K | MaskedColumn | Planet equilibrium temperature |
| `pl_eqt_err1` | K | MaskedColumn | Upper 1σ |
| `pl_eqt_err2` | K | MaskedColumn | Lower 1σ |
| `pl_eqt_src` | — | str | Source |
| `r_earth` | R⊕ | MaskedColumn | Planet radius (from catalog radius in R_Jup, converted) |
| `r_earth_src` | — | str | `'emc'` when from catalog; `''` when masked |
| `r_earth_min` | R⊕ | MaskedColumn | Lower bound on radius from msini (Chen & Kipping); masked when `r_earth` is known |
| `r_earth_max` | R⊕ | MaskedColumn | Upper bound on radius from msini; masked when `r_earth` is known |
| `pl_insol` | S⊕ | MaskedColumn | Insolation flux relative to Earth; direct `insol` from NEA if available, else L/a² |
| `pl_insol_err1` | S⊕ | MaskedColumn | Upper 1σ (asymmetric propagation through L and a) |
| `pl_insol_err2` | S⊕ | MaskedColumn | Lower 1σ |
| `pl_insol_src` | — | str | Provenance (e.g. `'nea'` for direct source values or `'derived(r:hpic teff:hpic a:provided)'`) |
| `a_src` | — | str | Semi-major axis provenance (`'emc'`, or `'kepler(mass:nea p:emc)'`) |
| `spectral_category` | — | str | Broad classification: `'Sun-like'`, `'Low-luminosity'`, `'Very-low-luminosity'`, `'Other'` |

### Masking semantic

- `MaskedColumn` entries are **masked** (not filled with 0 or NaN) when absent.
  Accessing `.data` on a masked column fills with 0 for numeric types — always
  use `np.ma.getmaskarray()` or `np.ma.filled()` explicitly.
- Error columns (`*_err1`, `*_err2`) are masked via `~np.isfinite` on the
  backing array (NaN = no error available).
- `*_src` string columns use `''` (empty string) to indicate no value.

### Error bar convention

Following NEA and standard astrophysical practice:
- `err1` = upper 1σ, always a **positive magnitude** (add to get upper bound)
- `err2` = lower 1σ, always a **positive magnitude** (subtract to get lower bound)

Asymmetric errors are propagated through derived quantities (luminosity, flux)
using standard quadrature in each direction separately.

## Sourcing & Computation Pathways for Astronomical Quantities

### Sourced-Only Quantities
Sourced sequentially via priority tier chain: $\text{HPIC} \to \text{NEA} \to \text{EPIC} \to \text{TOI} \to \text{SIMBAD}$.
$$q \in \{T_{\text{eff}}, M_{\text{star}}, \log g, \text{[Fe/H]}, V_{\text{mag}}, K_{\text{mag}}, d, T_{\text{eq}}\}$$
If all source values are masked, the final value is masked.

### Derived and Fallback Quantities

#### Stellar Radius ($R_{\text{star}}$)
1. **Primary**: Priority tier source.
2. **Fallback**: Derived from $T_{\text{eff}}$ using (in priority order):
   * **Mann 2015 ($M_{K_s}$)**: If $T_{\text{eff}} < 4200\text{ K}$, $K_{\text{mag}}$, $d$ exist and absolute magnitude $4.0 \le M_{K_s} \le 10.5$:
     $$R_{\text{star}} = f(M_{K_s}, \text{[Fe/H]})$$
   * **Torres 2010**: If $3900\text{ K} < T_{\text{eff}} < 8500\text{ K}$ and $\log g$ exists:
     $$R_{\text{star}} = f(T_{\text{eff}}, \log g, \text{[Fe/H]})$$
   * **Mann 2015 ($T_{\text{eff}}$)**: If $T_{\text{eff}} < 4000\text{ K}$:
     $$R_{\text{star}} = f(T_{\text{eff}}, \text{[Fe/H]})$$
   * **ZAMS Power-law**: Final fallback based on $T_{\text{eff}}$ and Spectral Type.

#### Stellar Luminosity ($L_{\text{star}}$)
1. **Primary**: Priority tier source.
2. **Fallback**: Derived via:
   $$L_{\text{star}} = R_{\text{star}}^2 \cdot \left(\frac{T_{\text{eff}}}{5778.0}\right)^4$$
   * **Error Propagation**:
     $$\sigma_L^{\pm} = L_{\text{star}} \cdot \sqrt{\left(2 \cdot \frac{\sigma_R^{\pm}}{R_{\text{star}}}\right)^2 + \left(4 \cdot \frac{\sigma_T^{\pm}}{T_{\text{eff}}}\right)^2}$$

#### Planet Semi-Major Axis ($a$)
1. **Primary**: Catalog table input column.
2. **Fallback**: Derived via Kepler's Third Law:
   $$a = \left(M_{\text{star}} \cdot \left(\frac{P}{365.25}\right)^2\right)^{1/3}$$
   * *Fallback Defaults*: If $M_{\text{star}}$ is masked, $M_{\text{star}} = 1.0\ M_{\odot}$.
   * **Error Propagation (Kepler Case)**:
     $$\sigma_a^{\pm} = \frac{1}{3} \cdot a \cdot \frac{\sigma_{M}^{\pm}}{M_{\text{star}}}$$

#### Insolation Flux ($S_{\text{eff}}$)
1. **Primary**: Priority tier source.
2. **Fallback**: Derived via:
   $$S_{\text{eff}} = \frac{L_{\text{star}}}{a^2}$$
   * **Error Propagation**:
     $$\sigma_{S_{\text{eff}}}^{+} = S_{\text{eff}} \cdot \sqrt{\left(\frac{\sigma_L^{+}}{L_{\text{star}}}\right)^2 + \left(2 \cdot \frac{\sigma_a^{-}}{a}\right)^2}$$
     $$\sigma_{S_{\text{eff}}}^{-} = S_{\text{eff}} \cdot \sqrt{\left(\frac{\sigma_L^{-}}{L_{\text{star}}}\right)^2 + \left(2 \cdot \frac{\sigma_a^{+}}{a}\right)^2}$$

#### Planet Radius Limits ($R_{\text{lower}}, R_{\text{upper}}$)
*Calculated only if direct planet radius is masked or $\le 0$.*
* **Inputs**: $M \sin i$ converted to $M_{\oplus}$.
* **Lower Bound**:
  $$R_{\text{lower}} = f_{\text{Chen-Kipping}}(M \sin i)$$
* **Upper Bound**:
  $$R_{\text{upper}} = f_{\text{Chen-Kipping}}\left(\frac{M \sin i}{\sin i_{\text{min}}}\right) \quad \text{where } \sin i_{\text{min}} = 0.3$$







  ----
  # Reference: Derived Parameter Source (`src`) Strings and Formulations

Below is the list of all possible `src` strings for the derived quantities in the stellar/planetary parameter enrichment pipeline.

---

## 1. Stellar Effective Temperature (`st_teff`)

| Source String (`src`)                     | Derivation / Formula                                                                     | Input Quantities                                                                 | Meaning / Notes                                                                                                                                      |
| :---------------------------------------- | :--------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SB_derived(rad:<rad_src> lum:<lum_src>)` | Stefan-Boltzmann Law:<br>$$T_{\text{eff}} = T_{\odot} \left(\frac{L}{R^2}\right)^{1/4}$$ | $R$ (Stellar Radius)<br>$L$ (Stellar Luminosity)<br>$T_{\odot} \approx 5778.0$ K | Physical derivation when both radius and luminosity are successfully matched but temperature is missing.                                             |
| `spectype_derived(spec:<spec_src>)`       | Inverted Spectral Class-to-Teff mapping via `spectype_to_teff()`.                        | $SpT$ (Stellar Spectral Type)                                                    | Interpolates spectral classifications based on standard astronomical temperature relations for main-sequence classes (O, B, A, F, G, K, M, L, T, Y). |

---

## 2. Stellar Radius (`st_rad`)

| Source String (`src`) | Derivation / Formula | Input Quantities | Meaning / Notes |
| :--- | :--- | :--- | :--- |
| `logg_derived(mass:<mass_src> logg:<logg_src>)` | Gravity-Mass relation:<br>$$R = 10^{0.5 \left(4.43797 + \log_{10} M - \log g\right)}$$ | $M$ (Stellar Mass)<br>$\log g$ (Surface Gravity) | Physical derivation using solar surface gravity $\log g_{\odot} \approx 4.43797$ dex. |
| `SB_derived(lum:<lum_src> teff:<teff_src>)` | Stefan-Boltzmann Law:<br>$$R = \sqrt{L} \left(\frac{T_{\odot}}{T_{\text{eff}}}\right)^2$$ | $L$ (Stellar Luminosity)<br>$T_{\text{eff}}$ (Effective Temp) | Physical derivation. |
| `mann_mks(kmag:<kmag_src>)` | Mann et al. 2015 polynomial:<br>$$R = f(M_{K_s}, [\text{Fe/H}])$$ | $K_s$ (2MASS Magnitude)<br>$d$ (Distance)<br>$[\text{Fe/H}]$ (Metallicity) | Empirical absolute K-band magnitude relation calibrated specifically for K7 to M7 dwarfs (scatter ~3%). *Ref: Mann et al. (2015) ApJ, 804, 64.* |
| `torres(teff:<teff_src> logg:<logg_src>)` | Torres et al. 2010 polynomial:<br>$$\log R = f(\log T_{\text{eff}}, \log g, [\text{Fe/H}])$$ | $T_{\text{eff}}$<br>$\log g$<br>$[\text{Fe/H}]$ | Polynomial calibration for FGK main-sequence stars (scatter ~3%). *Ref: Torres et al. (2010) A&ARv, 18, 67.* |
| `mann_teff(teff:<teff_src>)` | Mann et al. 2015 Teff-only polynomial:<br>$$R = f(T_{\text{eff}}, [\text{Fe/H}])$$ | $T_{\text{eff}}$<br>$[\text{Fe/H}]$ | Temperature-based empirical radius model for late-type M dwarfs. *Ref: Mann et al. (2015) ApJ, 804, 64.* |
| `ms(teff:<teff_src>)` | Zero-Age Main Sequence (ZAMS) power law:<br>$$R \propto T_{\text{eff}}^{\alpha}$$ | $T_{\text{eff}}$<br>$SpT$ (Spectral Type) | General power-law fallback representing a standard dwarf sequence. |

---

## 3. Stellar Mass (`st_mass`)

| Source String (`src`) | Derivation / Formula | Input Quantities | Meaning / Notes |
| :--- | :--- | :--- | :--- |
| `logg_derived(rad:<rad_src> logg:<logg_src>)` | Gravity-Radius relation:<br>$$M = 10^{\log g - 4.43797 + 2\log_{10} R}$$ | $R$ (Stellar Radius)<br>$\log g$ (Surface Gravity) | Physical derivation of mass using the solar surface gravity scaling relations. |

---

## 4. Stellar Luminosity (`st_lum`)

| Source String (`src`) | Derivation / Formula | Input Quantities | Meaning / Notes |
| :--- | :--- | :--- | :--- |
| `r:<rad_src> teff:<teff_src>` | Stefan-Boltzmann Equation:<br>$$L = R^2 \left(\frac{T_{\text{eff}}}{T_{\odot}}\right)^4$$ | $R$ (Stellar Radius)<br>$T_{\text{eff}}$ (Effective Temp) | Classical blackbody stellar luminosity calculation normalized to Solar units. |

---

## 5. Insolation Flux (`pl_insol`)

| Source String (`src`) | Derivation / Formula | Input Quantities | Meaning / Notes |
| :--- | :--- | :--- | :--- |
| `<source_name>` | Direct source value | Source insolation column | Direct catalog value, e.g. `nea`, `epic`, `toi`, or `input`. |
| `derived(r:<rad_src> teff:<teff_src> a:<a_src>)` | Insolation scaling:<br>$$S_{\text{eff}} = \frac{L}{a^2} = \frac{R^2 (T_{\text{eff}}/T_{\odot})^4}{a^2}$$ | $R$ (Stellar Radius)<br>$T_{\text{eff}}$ (Effective Temp)<br>$a$ (Semi-major axis in au) | Computes incoming flux relative to Earth's solar insolation when luminosity was derived from radius and temperature. |
| `derived(lum:<lum_src> a:<a_src>)` | Insolation scaling:<br>$$S_{\text{eff}} = \frac{L}{a^2}$$ | $L$ (Stellar Luminosity)<br>$a$ (Semi-major axis in au) | Computes incoming flux relative to Earth's solar insolation when luminosity is direct or independently derived. |
| `derived(eqt:<eqt_src>)` | Insolation-to-temperature inverse relation:<br>$$S_{\text{eff}} = \left(\frac{T_{\text{eq}}}{254.793}\right)^4$$ | $T_{\text{eq}}$ (Equilibrium Temp) | Reconstructed insolation value derived from equilibrium temperature under a Bond Albedo assumption $A_B = 0.3$. |

---

## 6. Equilibrium Temperature (`pl_eqt`)

| Source String (`src`) | Derivation / Formula | Input Quantities | Meaning / Notes |
| :--- | :--- | :--- | :--- |
| `derived(insol:<insol_src>)` | Equilibrium Temperature equation:<br>$$T_{\text{eq}} \approx 254.793 \cdot S_{\text{eff}}^{0.25}$$ | $S_{\text{eff}}$ (Insolation Flux) | Calculated fallback assuming a standard Bond Albedo $A_B = 0.3$. |
