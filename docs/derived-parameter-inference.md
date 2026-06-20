# Derived Parameter Inference

This page documents the formulas used when `ParamFiller.enrich(..., disable_calculations=False)` fills missing values through the pure `infer_*` functions in `crossmatching/enrichment/inference.py`.

The values here are fallback estimates. They preserve the best available sourced value whenever one exists, then use the derivations below only for missing values.

## Conventions

- Stellar radii, masses, and luminosities are in solar units.
- Planetary semi-major axis is in au.
- Orbital period is in days.
- Planet insolation `pl_insol` is in Earth flux units.
- Planet equilibrium temperature `pl_eqt` is in K.
- `msini` is interpreted as Jupiter masses before conversion to Earth masses.
- `err1` and `err2` are positive upper and lower 1-sigma magnitudes.
- The code uses `T_SUN = 5778.0 K`.

Provenance strings in `*_src` columns intentionally include the source labels of the input quantities used for a derivation. Some strings preserve historical spelling from the implementation, for example `StephanBoltzmann_derived(...)`.

## Stellar Effective Temperature

`infer_star_teff()` first keeps an existing or sourced `st_teff`. If it is missing, it tries these fallbacks.

| Source string | Formula | Inputs | Notes |
| --- | --- | --- | --- |
| `StephanBoltzmann_derived(rad:<rad_src> lum:<lum_src>)` | $$T_{\mathrm{eff}} = T_\odot \left(\frac{L}{R^2}\right)^{1/4}$$ | `st_rad`, `st_lum` | Direct rearrangement of the Stefan-Boltzmann luminosity relation. |
| `spectype_derived(spec:<spec_src>)` | Interpolate $T_{\mathrm{eff}}$ from the normalized spectral type. | `st_spectype` | Uses O-star anchors from Martins et al. 2005 and B-M dwarf anchors from Pecaut & Mamajek 2013. |

For spectral-type-derived temperatures, the uncertainty is approximated from the neighboring spectral-class temperature range returned by `get_spectral_class_range()`.

## Stellar Radius

`infer_star_radius()` keeps an existing or sourced `st_rad`. If radius is missing, the code tries the following paths in order.

| Source string                                             | Formula                                                                                                                                | Inputs                                  | Notes                                                                                                                                                                                                                                              |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `logg_derived(mass:<mass_src> logg:<logg_src>)`           | $$R = 10^{\frac{1}{2}\left(\log g_\odot + \log_{10} M - \log g\right)}$$                                                               | `st_mass`, `st_logg`                    | From $g/g_\odot = M/R^2$, with $\log g_\odot = 4.438$.                                                                                                                                                                                             |
| `StephanBoltzmann_derived(lum:<lum_src> teff:<teff_src>)` | $$R = \sqrt{L}\left(\frac{T_\odot}{T_{\mathrm{eff}}}\right)^2$$                                                                        | `st_lum`, `st_teff`                     | Rearranged Stefan-Boltzmann relation.                                                                                                                                                                                                              |
| `mann_mks(kmag:<kmag_src>)`                               | $$M_{K_s} = K_s - 5\log_{10}\left(\frac{d}{10\ \mathrm{pc}}\right)$$ $$R = a + bM_{K_s} + cM_{K_s}^2$$ then metallicity correction     | `sy_kmag`, `sy_dist`, optional `st_met` | Mann et al. 2015 K7-M7 dwarf relation. Code applies it for `st_teff < 4200 K` and $4.0 \le M_{K_s} \le 10.5$. Scatter is represented as 2.9%.                                                                                                      |
| `torres(teff:<teff_src> logg:<logg_src>)`                 | $$X = \log_{10}T_{\mathrm{eff}} - 4.1$$ $$\log R = b_0 + b_1X + b_2X^2 + b_3X^3 + b_4(\log g)^2 + b_5(\log g)^3 + b_6[\mathrm{Fe/H}]$$ | `st_teff`, `st_logg`, optional `st_met` | Torres et al. 2010 spectroscopic calibration. Code applies it for `3900 K < st_teff < 8500 K` when logg exists. Scatter is represented as 3%.                                                                                                      |
| `mann_teff(teff:<teff_src>)`                              | $$x = \frac{T_{\mathrm{eff}}}{3500}$$ $$R = a + bx + cx^2 + dx^3$$ then metallicity correction                                         | `st_teff`, optional `st_met`            | Mann et al. 2015 temperature relation for late-type dwarfs. Code applies it for `st_teff < 4000 K`; scatter is represented as 9.3% with metallicity or 13.4% without it.                                                                           |
| `ms(teff:<teff_src>)`                                     | $$R = \max\left[\left(\frac{T_{\mathrm{eff}}}{T_\odot}\right)^\alpha,\ 0.05\right]$$                                                   | `st_teff`, optional `st_spectype`       | Last-resort main-sequence approximation inspired by empirical main-sequence temperature-radius behavior. The exponent is chosen by spectral class or temperature: M/very cool stars use 2.5, K/cool stars 2.1, A/B/O/hot stars 1.0, otherwise 1.8. |

The Mann coefficients used by the code are from the Mann et al. 2015 erratum table, as noted in `radius_estimation.py`.

## Stellar Mass

`infer_star_mass()` keeps an existing or sourced `st_mass`. If mass is missing and both radius and surface gravity are available, it uses the inverse of the stellar surface-gravity scaling relation:

| Source string | Formula | Inputs | Notes |
| --- | --- | --- | --- |
| `logg_derived(rad:<rad_src> logg:<logg_src>)` | $$M = 10^{\log g - \log g_\odot + 2\log_{10}R}$$ | `st_rad`, `st_logg` | From $g/g_\odot = M/R^2$, with $\log g_\odot = 4.438$. |

## Stellar Luminosity

`infer_stellar_luminosity()` keeps an existing or sourced `st_lum`. If luminosity is missing and radius plus effective temperature are available, it uses:

| Source string | Formula | Inputs | Notes |
| --- | --- | --- | --- |
| `derived(rad:<rad_src> teff:<teff_src>)` | $$L = R^2\left(\frac{T_{\mathrm{eff}}}{T_\odot}\right)^4$$ | `st_rad`, `st_teff` | Stefan-Boltzmann relation in solar-normalized units. |

## Semi-Major Axis

`infer_semi_major_axis()` keeps an existing or sourced semi-major axis. If it is missing, period and stellar mass are required:

| Source string                                 | Formula                                                                       | Inputs            | Notes                                                                                                                                       |
| --------------------------------------------- | ----------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `kepler(mass:<mass_src> period:<period_src>)` | $$a = \left[M_\star\left(\frac{P}{365.25\text{ days}}\right)^2\right]^{1/3}$$ | period, `st_mass` | Kepler's third law in au, solar masses, and years. The code does not assume a default stellar mass; if mass is missing, `a` remains masked. |

## Planet Insolation

`infer_planet_insolation()` keeps an existing or sourced `pl_insol`. If it is missing, the code tries luminosity plus semi-major axis first, then equilibrium temperature.

| Source string                      | Formula                                                                        | Inputs                    | Notes                                                                                                                                                                     |
| ---------------------------------- | ------------------------------------------------------------------------------ | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `derived(lum:<lum_src> a:<a_src>)` | $$S_{\mathrm{eff}} = \frac{L}{a^2}$$                                           | `st_lum`, semi-major axis | Solar-normalized inverse-square flux. If luminosity itself was derived from radius and Teff, the string is rewritten as `derived(r:<rad_src> teff:<teff_src> a:<a_src>)`. |
| `derived(eqt:<eqt_src>)`           | $$S_{\mathrm{eff}} = \left(\frac{T_{\mathrm{eq}}}{254.793\text{ K}}\right)^4$$ | `pl_eqt`                  | Inverse of the equilibrium-temperature fallback below.                                                                                                                    |

The constant `254.793 K` corresponds to the blackbody equilibrium temperature at one Earth flux for a Bond albedo near Earth's usual $A_B = 0.3$:

$$T_{\mathrm{eq}} = \left(\frac{S(1-A_B)}{4\sigma}\right)^{1/4}.$$

This is useful for consistent internal fallback behavior, not as a detailed climate model.

## Planet Equilibrium Temperature

`infer_planet_equilibrium_temperature()` keeps an existing or sourced `pl_eqt`. If it is missing and insolation is available, it uses:

| Source string | Formula | Inputs | Notes |
| --- | --- | --- | --- |
| `derived(insol:<insol_src>)` | $$T_{\mathrm{eq}} = 254.793\,S_{\mathrm{eff}}^{1/4}$$ | `pl_insol` | Same blackbody equilibrium-temperature convention as above. |

## Radius Bounds From `msini`

`infer_msini_radius_bounds()` only produces bounds when direct planet radius is missing or invalid and `pl_mass` or  `msini` is available. The output column names follow the resolved planet-radius key, for example `r_lower_bound` and `r_upper_bound` for Exo-MerCat.

If `pl_mass` available (with errors `pl_masserr1` ${ \sigma^{+}_{M} }$,  `pl_masserr2` ${ \sigma^{-}_{M}}$), then 

| Output                            | Formula                                                                    | Inputs                                                        | Notes                                                                                                                                               |
| --------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<planet_radius_key>_lower_bound` | $$R_{\mathrm{low}} = R_{\mathrm{CK17}}(M-2 \sigma_{M}^{-})$$               | `pl_mass`,  `pl_masserr2`<br>(errors set to 0 if unavailable) | We "simulate" the Chen & Kipping relations large scatter by going two standard deviations away, giving us a rough ${ 2\sigma }$-confidence interval |
| `<planet_radius_key>_upper_bound` | $$R_{\mathrm{high}} = R_{\mathrm{CK17}}\left(M + 2\sigma_{M} ^{+}\right)$$ | `pl_mass`, `pl_masserr1`<br>(errors set to 0 if unavailable)` |                                                                                                                                                     |

If only `msini` available:

| Output                            | Formula                                                                             | Inputs                   | Notes                                                                                               |
| --------------------------------- | ----------------------------------------------------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------- |
| `<planet_radius_key>_lower_bound` | $$R_{\mathrm{low}} = R_{\mathrm{CK17}}\left(M\sin i\right)$$                        | `msini`                  | Converts Jupiter masses to Earth masses, then applies the Chen & Kipping 2017 mass-radius relation. |
| `<planet_radius_key>_upper_bound` | $$R_{\mathrm{high}} = R_{\mathrm{CK17}}\left(\frac{M\sin i}{\sin i_{\min}}\right)$$ | `msini`, `msini_sin_min` | Uses `msini_sin_min=0.3` by default.                                                                |

The implemented Chen-Kipping relation is:

$$
R_{\mathrm{CK17}}(M) =
\begin{cases}
1.008M^{0.279}, & M < 2.04\\
0.808M^{0.589}, & 2.04 \le M < 132\\
17.74M^{-0.044}, & M \ge 132
\end{cases}
$$

where $M$ and $R$ are in Earth units. The bounds are deliberately broad and are mainly intended to keep radial-velocity planets in candidate-filtering workflows.

## Spectral-Type Display And Category

`infer_spectral_type_display()` keeps a real source spectral type for display. If the source value is missing or `null`, it uses the nearest main-sequence temperature anchor:

$$SpT_{\mathrm{display}} \approx SpT(T_{\mathrm{eff}}).$$

The O-star anchors come from Martins et al. 2005. The B-M main-sequence anchors come from Pecaut & Mamajek 2013. The broad `spectral_category` column is then produced by `classify_spectral_type()`:

| Category | Rule |
| --- | --- |
| `Sun-like` | F, G, or K0-K5 dwarfs |
| `Low-luminosity` | K6-K9 or M0-M2.9 dwarfs |
| `Very-low-luminosity` | M3 and later dwarfs |
| `Other` | Hot O/B/A stars, evolved luminosity classes, white dwarfs, and unparseable strings |

## Error Propagation

Where possible, the code propagates asymmetric errors separately for the upper and lower directions. The general pattern is standard quadrature in fractional errors. For example:

$$
\sigma_L^\pm =
L\sqrt{\left(2\frac{\sigma_R^\pm}{R}\right)^2 +
\left(4\frac{\sigma_T^\pm}{T_{\mathrm{eff}}}\right)^2}.
$$

For inversions where increasing one input lowers the output, the code swaps the appropriate upper/lower uncertainty contribution. Missing uncertainty terms are ignored in quadrature; if no useful terms remain, the output uncertainty is masked.

## References

- Chen, J. & Kipping, D. 2017, "Probabilistic Forecasting of the Masses and Radii of Other Worlds", *ApJ*, 834, 17. [arXiv:1603.08614](https://arxiv.org/abs/1603.08614)
- Boyajian, T. S. et al. 2012, "Stellar Diameters and Temperatures II. Main Sequence K & M Stars", *ApJ*, 757, 112. [arXiv:1208.2431](https://arxiv.org/abs/1208.2431)
- Mann, A. W., Feiden, G. A., Gaidos, E., Boyajian, T. & von Braun, K. 2015, "How to Constrain Your M Dwarf", *ApJ*, 804, 64. [arXiv:1501.01635](https://arxiv.org/abs/1501.01635)
- Martins, F., Schaerer, D. & Hillier, D. J. 2005, "A new calibration of stellar parameters of Galactic O stars", *A&A*, 436, 1049. [arXiv:astro-ph/0503346](https://arxiv.org/abs/astro-ph/0503346)
- Pecaut, M. J. & Mamajek, E. E. 2013, "Intrinsic Colors, Temperatures, and Bolometric Corrections of Pre-Main Sequence Stars", *ApJS*, 208, 9. [arXiv:1307.2657](https://arxiv.org/abs/1307.2657)
- Prsa, A. et al. 2016, "Nominal values for selected solar and planetary quantities: IAU 2015 Resolution B3". [arXiv:1605.09788](https://arxiv.org/abs/1605.09788)
- Torres, G., Andersen, J. & Gimenez, A. 2010, "Accurate masses and radii of normal stars: modern results and applications", *A&ARv*, 18, 67. [arXiv:0908.2624](https://arxiv.org/abs/0908.2624)
