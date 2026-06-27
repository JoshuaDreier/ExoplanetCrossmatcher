from __future__ import annotations
import re
import numpy as np


def standardize_spectral_type(spectype: str) -> str:
    """Normalize a raw spectral type string to a simple ``<Letter><Subtype>`` form.

    Handles the wide variety of real-world spectral type notation found in
    astronomical catalogues, including:

    * Metallic-line prefixes: strings of the form ``kXhYmZ``
      (Ca II K-line type X, hydrogen-line type Y, metal-line type Z).  The
      hydrogen-line class ``Y`` is the closest proxy for effective temperature
      and is returned.  Strings starting with ``h`` (hydrogen-line class first)
      are handled analogously.  Strings of the plain form ``A0mA1Va`` (base
      type followed by an ``m``-suffix metals type) return the base type.
    * (sub)dwarf prefixes: ``sd``, ``esd``, ``usd``, ``d/sd``,
      ``d`` are stripped before extraction.
    * composite systems: only the primary component (before the
      first ``+`` or ``/``) is kept.
    * parenthesised qualifiers: text in parentheses (e.g. ``(n)``, ``(e)``)
      is removed.   
    * White-dwarf types: strings beginning with ``D`` followed by a
      letter or digit (e.g. ``DA2``, ``DQ``) are not stars on the main
      sequence; they return ``''``.
    * strip ``~`` from beginning, added by this package in infer_spectral_type if inferred from teff

    Parameters
    ----------
    spectype : str
        Raw spectral type string from a catalogue (e.g. ``'kA4hA5mA5Va'``,
        ``'sdM3.5'``, ``'G9III+A7.5'``).

    Returns
    -------
    str
        Simplified spectral type string (e.g. ``'A5'``, ``'M3.5'``, ``'G9'``),
        or ``''`` if the input cannot be reduced to a recognisable class.
    """
    s = str(spectype).strip()
    if not s or s in ('null', '--', 'nan'):
        return ''
    # 0. Strip leading '~' marker potentially added by infer_spectral_type.
    s = s.lstrip('~')

    # 1. White-dwarf types (DA, DB, DQ, …): not main-sequence, return empty.
    #    Note: case-sensitive — lowercase 'd' is a dwarf prefix, not a WD designation.
    if re.match(r'^D[A-Z0-9]', s):
        return ''

    # 2. Metallic-line (Am) notation: kXhYmZ  →  use hydrogen class Y.
    #    The luminosity-class suffix between k-type and 'h' is skipped.
    #    Example: kF3VhF5mF5(II-III)  →  F5
    am_khm = re.match(r'k[OBAFGKM]\d*\.?\d*[IVab]*h([OBAFGKM]\d*\.?\d*)', s, re.I)
    if am_khm:
        s = am_khm.group(1)
    else:
        # Strings starting with 'h' give hydrogen class directly.
        # Example: hF5gF5mF3  →  F5
        am_h = re.match(r'h([OBAFGKM]\d*\.?\d*)', s, re.I)
        if am_h:
            s = am_h.group(1)
        else:
            # kXmZ (no 'h' part): use metals class Z.
            # Example: kA8mF0Vp  →  F0
            am_km = re.match(r'k[OBAFGKM]\d*\.?\d*[IVab]*m([OBAFGKM]\d*\.?\d*)', s, re.I)
            if am_km:
                s = am_km.group(1)
            else:
                # Plain base+metals: A0mA1Va  →  A0  (keep the base type)
                am_plain = re.match(r'([OBAFGKM]\d*\.?\d*)m[OBAFGKM]', s, re.I)
                if am_plain:
                    s = am_plain.group(1)

    # 3. Normalise slash-prefix variants before the binary split so that
    #    strings like 's/sdM5' are not mistaken for a binary system.
    #    e.g. s/sd → sd,  d/sd → sd
    s = re.sub(r'^(?:s|d)/sd', 'sd', s, flags=re.I)

    # 4. For binary / composite systems take only the primary component.
    #    Example: G9III+A7.5  →  G9III,   M2.5V/M3V  →  M2.5V
    s = re.split(r'[+/]', s)[0].strip()

    # 5. Strip subdwarf / dwarf prefixes.
    #    Example: sdM3.5  →  M3.5,   esdM0  →  M0,   dM4.5e  →  M4.5e
    s = re.sub(r'^(?:esd|usd|d/sd|s/sd|sd|d)[\s:]*', '', s, flags=re.I)

    # 5. Remove parenthesised qualifiers: (n), (e), (k), …
    s = re.sub(r'\(.*?\)', '', s)

    s = s.strip()

    # 6. Extract the primary spectral class letter + numeric subtype.
    m = re.match(r'([OBAFGKMLTY])(\d+\.?\d*)?', s.upper())
    if not m:
        return ''

    letter = m.group(1)
    subtype = m.group(2) or ''
    return letter + subtype


def classify_spectral_type(sptype: str) -> str:
    """Map a spectral type string to a broad stellar category.

    Returns one of: 'Sun-like', 'Low-luminosity', 'Very-low-luminosity', 'Other'.
    """
    s = str(sptype).strip()
    if s == 'null':
        return 'Other'

    # White-dwarf types
    if re.match(r'^D[A-Z0-9]', s):
        return 'Other'

    # Evolved (giant / subgiant) luminosity classes
    if re.search(r'IV|III|II', s):
        return 'Other'

    std = standardize_spectral_type(s)
    if not std:
        return 'Other'

    m = re.match(r'([OBAFGKM])(\d+\.?\d*)?', std.upper())
    if not m:
        return 'Other'

    letter = m.group(1).upper()
    subtype = float(m.group(2)) if m.group(2) else 5.0

    if letter in ('O', 'B', 'A'):
        return 'Other'
    if letter in ('F', 'G'):
        return 'Sun-like'
    if letter == 'K':
        return 'Sun-like' if subtype <= 5 else 'Low-luminosity'
    if letter == 'M':
        return 'Low-luminosity' if subtype < 3 else 'Very-low-luminosity'
    return 'Other'


def spectype_display(spec: str, teff: float) -> str:
    """Spectral type for display: use the actual string if available, else derive from teff."""
    s = standardize_spectral_type(spec)
    if s:
        return s
    return teff_to_spectype(teff)

 
_LETTER_ORDER = "OBAFGKMLTY"
 
# (spectral_type, Teff_K) — strictly hot → cool
_TEFF_SPECTYPE: list[tuple[str, int]] = [
    # O  —  Martins+2005
    ("O3",  44200), ("O4",  42900), ("O5",  40900), ("O5.5", 39700),
    ("O6",  39500), ("O6.5",38100), ("O7",  37100), ("O7.5", 36100),
    ("O8",  35100), ("O8.5",34100), ("O9",  33100), ("O9.5", 32000),
    # B–M  —  Pecaut & Mamajek 2013
    ("B0",  29700), ("B1",  25400), ("B2",  22000), ("B3",  17600),
    ("B5",  15400), ("B6",  14500), ("B7",  13400), ("B8",  11400), ("B9", 10500),
    ("A0",   9600), ("A1",   9330), ("A2",   9040), ("A3",   8750),
    ("A4",   8480), ("A5",   8180), ("A7",   7920),
    ("F0",   7220), ("F2",   7030), ("F3",   6910), ("F5",   6640),
    ("F6",   6510), ("F7",   6340), ("F8",   6160),
    ("G0",   5930), ("G1",   5860), ("G2",   5780), ("G5",   5660), ("G8", 5440),
    ("K0",   5280), ("K1",   5170), ("K2",   5040), ("K3",   4830),
    ("K4",   4600), ("K5",   4410), ("K6",   4230), ("K7",   4070),
    ("K8",   3990), ("K9",   3930),
    ("M0",   3850), ("M1",   3660), ("M2",   3560), ("M3",   3410),
    ("M4",   3200), ("M5",   3030), ("M6",   2850), ("M7",   2650),
    ("M8",   2500), ("M9",   2400),
]
 
def _parse_spt(spt: str) -> float:
    """Spectral type string → float index (class_rank * 10 + subtype).

    Calls :func:`standardize_spectral_type` first so that complex real-world
    strings (subdwarf prefixes, Am notation, binary companions, …) are reduced
    to a simple ``<Letter><Subtype>`` form before the index is computed.
    Luminosity-class suffixes and peculiarity flags are ignored.
    """
    std = standardize_spectral_type(spt)
    if not std:
        raise ValueError(f"Cannot parse spectral type: {spt!r}")
    m = re.match(r"([OBAFGKMLTY])(\d+\.?\d*)?", std.upper())
    if m is None:
        raise ValueError(f"Cannot parse spectral type: {spt!r}")
    letter, num = m.group(1), m.group(2)
    return _LETTER_ORDER.index(letter) * 10 + (float(num) if num else 5.0)

 
_nums  = np.array([_parse_spt(s) for s, _ in _TEFF_SPECTYPE])  # strictly increasing
_teffs = np.array([t for _, t in _TEFF_SPECTYPE], dtype=float)  # strictly decreasing
 
 
def spectype_to_teff(spectype: str) -> float:
    """Interpolated Teff (K) for a main-sequence spectral type.
    Accepts fractional subtypes ('K2.5'), lum-class suffixes ('G2V'), and
    peculiarity flags. Raises ValueError on unparseable input.
    
    Sources
    O3–O9.5 : Martins, Schaerer & Hillier 2005, A&A 436, 1049 (Table 1, dwarf scale)
    B0–M9   : Pecaut & Mamajek 2013, ApJS 208, 9
            https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    
    """
    if not standardize_spectral_type(spectype):
        return np.nan
    return float(np.interp(_parse_spt(spectype), _nums, _teffs))
 
 
def teff_to_spectype(teff: float, *, prefix: str = "~") -> str:
    """Nearest-anchor main-sequence spectral type for a given Teff.
    Returns '' for non-positive Teff. Output is always a discrete anchor
    entry; use spectype_to_teff() for the interpolated inverse.
    
    Sources
    O3–O9.5 : Martins, Schaerer & Hillier 2005, A&A 436, 1049 (Table 1, dwarf scale)
    B0–M9   : Pecaut & Mamajek 2013, ApJS 208, 9
            https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    
    
    """
    if not (teff > 0):
        return ""
    nearest = int(np.argmin(np.abs(_teffs - teff)))
    return f"{prefix}{_TEFF_SPECTYPE[nearest][0]}"


def get_spectral_class_range(spectype: str) -> tuple[float, float]:
    """Get the min and max Teff (K) for the closest characterization of spectype.

    Uses :func:`standardize_spectral_type` and :func:`_parse_spt` to convert
    the spectral type to a float index and determines the range based on
    whether it resolves to a specific subclass or only a broad letter class.

    Parameters
    ----------
    spectype : str
        The full spectral type string (e.g., 'G2V', 'M5', 'kA4hA5mA5Va').

    Returns
    -------
    tuple[float, float]
        A tuple of (min_teff, max_teff). Returns (0.0, 0.0) if the spectral
        class is invalid or not found.
    """
    std = standardize_spectral_type(spectype)
    if not std:
        return 0.0, 0.0

    m = re.match(r"([OBAFGKMLTY])(\d+\.?\d*)?", std.upper())
    if not m:
        return 0.0, 0.0

    letter = m.group(1)
    subtype = m.group(2)

    try:
        target_idx = _parse_spt(spectype)
    except ValueError:
        return 0.0, 0.0

    if subtype is not None:
        matches = np.where(np.isclose(_nums, target_idx))[0]
        if len(matches) > 0:
            idx = matches[0]
            num_anchors = len(_TEFF_SPECTYPE)

            if idx > 0:
                t_max = _teffs[idx - 1]
            else:
                t_max = _teffs[idx] + (_teffs[idx] - _teffs[idx + 1])

            if idx < num_anchors - 1:
                t_min = _teffs[idx + 1]
            else:
                t_min = _teffs[idx] - (_teffs[idx - 1] - _teffs[idx])

            return float(t_min), float(t_max)

    class_rank = _LETTER_ORDER.index(letter)
    min_idx = class_rank * 10
    max_idx = (class_rank + 1) * 10

    class_indices = np.where((_nums >= min_idx) & (_nums < max_idx))[0]
    if len(class_indices) > 0:
        class_teffs = _teffs[class_indices]
        return float(np.min(class_teffs)), float(np.max(class_teffs))

    return 0.0, 0.0


